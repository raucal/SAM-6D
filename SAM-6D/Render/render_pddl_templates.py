import blenderproc as bproc

import argparse
import hashlib
import json
import os
from pathlib import Path

# Depth EXRs are generated locally by Blender and read by imageio/OpenCV.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2
import numpy as np
import trimesh

DESCRIPTION = "Render the 42 official views, with XYZ explicitly in CAD millimetres for PEM."


def xyz_from_depth_mm(depth, K, camera_pose, scale):
    """Invert Blender's camera projection into the unscaled CAD frame."""
    depth = np.asarray(depth, dtype=np.float64)
    v, u = np.indices(depth.shape)
    camera_points = np.stack([(u - K[0, 2]) * depth / K[0, 0],
                              -(v - K[1, 2]) * depth / K[1, 1], -depth], axis=-1)
    world = camera_points @ camera_pose[:3, :3].T + camera_pose[:3, 3]
    return world / scale


def main():
    import bpy

    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--cad_path", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    cad = trimesh.load_mesh(args.cad_path)
    # All vertices, rather than a random surface sample, determine normalization.
    radius = np.linalg.norm(cad.vertices, axis=1).max()
    scale = 1.0 / (2.0 * radius)
    poses_file = Path(__file__).resolve().parents[1] / "Instance_Segmentation_Model/utils/poses/predefined_poses/cam_poses_level0.npy"
    poses = np.load(poses_file)
    folder = args.output_dir / "templates"
    folder.mkdir(parents=True, exist_ok=True)
    scratch = folder / ".render_tmp"
    scratch.mkdir(exist_ok=True)
    # Invalidate completion before rendering, so an interrupted rerender is rejected.
    (folder / "render_metadata.json").unlink(missing_ok=True)
    bproc.init()
    bproc.camera.set_resolution(args.image_size, args.image_size)
    bproc.renderer.set_cpu_threads(args.threads)
    # Use one scene and animate the camera/light over the 42 reference views.
    # Depth output is configured once; BlenderProc retains its global flag.
    blender_mesh = bpy.data.meshes.new("cad_mm")
    blender_mesh.from_pydata(cad.vertices.tolist(), [], cad.faces.tolist())
    blender_mesh.update()
    blender_object = bpy.data.objects.new("cad", blender_mesh)
    bpy.context.collection.objects.link(blender_object)
    obj = bproc.types.MeshObject(blender_object)
    obj.set_scale([scale] * 3)
    material = bproc.material.create("cad")
    material.set_principled_shader_value("Base Color", [0.4, 0.4, 0.4, 1.0])
    obj.replace_materials(material)
    light = bproc.types.Light()
    light.set_type("POINT")
    light.set_energy(1000)
    for index, source_pose in enumerate(poses):
        camera_pose = source_pose.copy()
        camera_pose[:3, 1:3] *= -1
        camera_pose[:3, 3] *= 0.002
        bproc.camera.add_camera_pose(camera_pose, frame=index)
        light.set_location(2.5 * camera_pose[:3, 3], frame=index)
    bproc.renderer.enable_depth_output(activate_antialiasing=False, output_dir=str(scratch))
    bproc.renderer.set_max_amount_of_samples(args.samples)
    # Newer imageio can pick an EXR reader that silently returns uint8.
    # Read the locally rendered EXR explicitly, preserving float depth.
    bproc.renderer.render(output_dir=str(scratch), return_data=False)
    K = bproc.camera.get_intrinsics_as_K_matrix()
    for index in range(len(poses)):
        camera_pose = bproc.camera.get_camera_pose(frame=index)
        depth = cv2.imread(str(scratch / f"depth_{index:04d}.exr"), cv2.IMREAD_UNCHANGED)
        color = cv2.imread(str(scratch / f"rgb_{index:04d}.png"), cv2.IMREAD_UNCHANGED)
        if depth is None or color is None or depth.dtype.kind != "f":
            raise ValueError("No se ha leído la profundidad EXR en coma flotante.")
        if depth.ndim == 3:
            depth = depth[..., 0]
        mask = np.isfinite(depth) & (depth > 0) & (depth < 10)
        if not np.any(mask):
            raise ValueError(f"Vista sin objeto: {index}")
        # Depth is along camera Z. Blender uses +X/right, +Y/up, -Z/forward.
        xyz_mm = np.zeros((*depth.shape, 3), dtype=np.float32)
        # Invalid background values must not participate in inverse projection.
        xyz_mm[mask] = xyz_from_depth_mm(np.where(mask, depth, 0), K, camera_pose, scale)[mask]
        # Raster precision can move edge pixels slightly, but not change units.
        tolerance = max(0.5, radius * 0.01)
        if np.any(xyz_mm[mask] < cad.bounds[0] - tolerance) or np.any(xyz_mm[mask] > cad.bounds[1] + tolerance):
            np.savez_compressed(folder / "projection_debug.npz", depth=depth, K=K,
                                camera_pose=camera_pose, actual_camera_pose=bproc.camera.get_camera_pose(),
                                scale=scale, bounds=cad.bounds, object_bounds=obj.get_bound_box(),
                                xyz_bounds=np.stack([xyz_mm[mask].min(0), xyz_mm[mask].max(0)]))
            raise ValueError(f"XYZ fuera del CAD: {xyz_mm[mask].min(0)} .. {xyz_mm[mask].max(0)}, esperado {cad.bounds}")
        cv2.imwrite(str(folder / f"rgb_{index}.png"), color[..., :3])
        cv2.imwrite(str(folder / f"mask_{index}.png"), mask.astype(np.uint8) * 255)
        np.save(folder / f"xyz_{index}.npy", xyz_mm)
        print(f"CAD {args.cad_path.stem}: vista {index + 1}/{len(poses)}", flush=True)
    for index in range(len(poses)):
        for name in (f"depth_{index:04d}.exr", f"rgb_{index:04d}.png"):
            (scratch / name).unlink(missing_ok=True)
    if not any(scratch.iterdir()):
        scratch.rmdir()
    (folder / "projection_debug.npz").unlink(missing_ok=True)
    (folder / "render_metadata.json").write_text(json.dumps({
        "xyz_unit": "mm", "views": len(poses), "render_scale_per_mm": scale,
        "image_size": args.image_size, "samples": args.samples, "threads": args.threads,
        "cad_sha256": hashlib.sha256(args.cad_path.read_bytes()).hexdigest()
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
