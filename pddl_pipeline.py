"""Run SAM-6D on the prepared PDDL CAD catalog and RealSense captures."""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_cad_root = next(p for p in _Path(__file__).resolve().parents if (p/'config/settings.json').is_file())
_sys.path.insert(0, str(_cad_root/'pipeline'))

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from settings import SAM6D as ROOT, CORE, PREPARED, NATIVE, PIPELINE, path as configured_path
DEFAULT_MANIFEST = PREPARED / "manifest.json"


def read_dataset(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError(f"Primero preparar los datos: falta {path}")
    data = json.loads(path.read_text())
    if data.get("schema_version") != 1:
        raise ValueError("Versión de manifest no compatible.")
    return path, data


def select(items, requested, field):
    if not requested:
        return items
    unknown = set(requested) - {item[field] for item in items}
    if unknown:
        raise ValueError(f"{field} desconocido: {sorted(unknown)}")
    return [item for item in items if item[field] in requested]


def template_problems(base, parts):
    missing = []
    for part in parts:
        folder = base / part["sam_template_dir"]
        for index in range(42):
            if any(not (folder / f"{prefix}_{index}.{ext}").is_file()
                   for prefix, ext in (("rgb", "png"), ("mask", "png"), ("xyz", "npy"))):
                missing.append(part["key"])
                break
        metadata = folder / "render_metadata.json"
        if not metadata.is_file():
            if part["key"] not in missing:
                missing.append(part["key"])
        else:
            render_info = json.loads(metadata.read_text())
            cad_hash = hashlib.sha256((base / part["sam_obj"]).read_bytes()).hexdigest()
            if render_info.get("xyz_unit") != "mm" or render_info.get("cad_sha256") != cad_hash or render_info.get("views") != 42:
                missing.append(part["key"])
    return missing


def doctor(base, parts, segmentor="sam", backbone="vit_h"):
    modules = ["torch", "torchvision", "hydra", "omegaconf", "pytorch_lightning",
               "segment_anything", "cv2", "trimesh", "pycocotools", "distinctipy",
               "imageio", "skimage", "blenderproc", "gorilla", "timm", "einops"]
    available = {name: importlib.util.find_spec(name) is not None for name in modules}
    cuda = False
    cuda_error = None
    if available["torch"]:
        try:
            import torch
            cuda = torch.cuda.is_available()
        except Exception as error:
            cuda_error = str(error)
    ism = CORE / "Instance_Segmentation_Model"
    pem = CORE / "Pose_Estimation_Model"
    weights = [ism / "checkpoints/segment-anything/sam_vit_h_4b8939.pth",
               ism / "checkpoints/dinov2/dinov2_vitl14_pretrain.pth",
               pem / "checkpoints/sam-6d-pem-base.pth"]
    fastsam_weight = ism / "checkpoints/FastSAM/FastSAM-x.pt"
    pem_weights = [p for p in weights[2:] if p.is_file() and p.stat().st_size > 0]
    report = {"python": sys.executable, "modules": available,
              "cuda_available": cuda, "cuda_error": cuda_error,
              "weights": {str(p.relative_to(CORE)): p.is_file() and p.stat().st_size > 0 for p in weights[:2]},
              "pem_checkpoints": [str(p) for p in pem_weights],
              "pointnet2_extensions": [str(p) for p in (NATIVE / "pointnet2").glob("*.so")],
              "fastsam_ready": fastsam_weight.is_file() and importlib.util.find_spec("ultralytics") is not None,
              "blenderproc_executable": shutil.which("blenderproc"),
              "parts_missing_metric_templates": template_problems(base, parts)}
    # Existing rendered templates are sufficient for inference; Blender is needed only to render new ones.
    report["segmentation_ready"] = all(available[m] for m in modules[:12]) and all(report["weights"].values()) and not report["parts_missing_metric_templates"]
    if segmentor == "fastsam":
        report["segmentation_ready"] = report["fastsam_ready"] and weights[1].is_file() and not report["parts_missing_metric_templates"]
    report["rendering_module_available"] = (configured_path("blender")).is_file()
    report["render_backend"] = "Blender 3.3.1 standalone; 42 views at 256px, 16 samples"
    report["pose_ready"] = report["segmentation_ready"] and cuda and bool(pem_weights) and bool(report["pointnet2_extensions"]) and all(available[m] for m in ("gorilla", "timm", "einops"))
    return report


def run(command, cwd, dry_run):
    print(f"[{cwd}] {shlex.join([str(c) for c in command])}", flush=True)
    if not dry_run:
        subprocess.run([str(c) for c in command], cwd=cwd, check=True)


def pose(args, manifest_path, dataset, parts, frames, output):
    """Feed each detected class to the upstream single-CAD PEM, then restore frames."""
    import numpy as np
    base = manifest_path.parent
    for frame in frames:
        result_dir = output / frame["name"]
        source = result_dir / "detection_ism.json"
        if not source.is_file():
            raise ValueError(f"Primero ejecutar segment: {source}")
        detections = json.loads(source.read_text())
        combined = []
        for part in parts:
            selected = [d for d in detections if d["category_id"] == part["category_id"] and d["score"] > args.min_score]
            if not selected:
                continue
            part_dir = result_dir / "pose" / part["key"]
            command = [sys.executable, PIPELINE / "workers/pose_worker.py",
                       "--output", part_dir / "pose.json", "--cad", base / part["sam_ply"],
                       "--rgb", base / frame["rgb"], "--depth", base / frame["depth"],
                       "--camera", base / frame["camera"], "--detections", part_dir / "selected.json",
                       "--templates", base / part["sam_template_dir"], "--min-score", args.min_score]
            if not args.dry_run:
                part_dir.mkdir(parents=True, exist_ok=True)
                templates = part_dir / "templates"
                if not templates.exists():
                    templates.symlink_to((base / part["sam_template_dir"]).resolve(), target_is_directory=True)
                (part_dir / "selected.json").write_text(json.dumps(selected))
            run(command, CORE / "Pose_Estimation_Model", args.dry_run)
            if args.dry_run:
                continue
            results = json.loads((part_dir / "pose.json").read_text())["poses"]
            for detection in results:
                T = np.eye(4)
                T[:3, :3] = detection["R"]
                detection["t"] = detection["t_mm"]
                T[:3, 3] = detection["t"]
                detection.update(part_key=part["key"], pose_translation_unit="mm",
                    T_camera_from_sam_mm=T.tolist(),
                    T_camera_from_local_mm=(T @ np.linalg.inv(part["T_local_from_sam_mm"])).tolist())
                combined.append(detection)
        if not args.dry_run:
            (result_dir / "detection_pem.json").write_text(json.dumps(combined, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["doctor", "render", "segment", "pose"])
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--part", action="append", help="CAD key; repeat to select several. Default: all CADs.")
    parser.add_argument("--frame", action="append", help="Capture name; repeat to select several. Default: all RGB-D frames.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-score", type=float, default=0.3)
    parser.add_argument("--stability-score", type=float, default=0.97)
    parser.add_argument("--segmentor", choices=("sam", "fastsam"), default="sam")
    parser.add_argument("--backbone", choices=("vit_h", "vit_l", "vit_b"), default="vit_h")
    parser.add_argument("--blenderproc", default="blenderproc")
    parser.add_argument("--render-size", type=int, default=512)
    parser.add_argument("--render-samples", type=int, default=50)
    parser.add_argument("--render-threads", type=int, default=8)
    parser.add_argument("--blender-install-path", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        if not 0 <= args.min_score <= 1 or not 0 <= args.stability_score <= 1:
            raise ValueError("Los umbrales deben estar entre 0 y 1.")
        manifest_path, data = read_dataset(args.manifest)
        base = manifest_path.parent
        parts = select(data["parts"], args.part, "key")
        frames = select(data["frames"], args.frame, "name")
        output = (args.output or base / "sam6d/results").resolve()
        if args.command == "doctor":
            print(json.dumps(doctor(base, parts, args.segmentor, args.backbone), indent=2))
            return 0
        if args.command == "render":
            for part in parts:
                command = [sys.executable, PIPELINE / "sam6d_ui/render_reference.py",
                     "--mesh", base / part["sam_obj"], "--output", base / part["sam_template_dir"],
                     "--poses", CORE / "Instance_Segmentation_Model/utils/poses/predefined_poses/cam_poses_level0.npy"]
                run(command, CORE / "Render", args.dry_run)
        else:
            if not args.dry_run:
                report = doctor(base, parts, args.segmentor, args.backbone)
                key = "segmentation_ready" if args.command == "segment" else "pose_ready"
                if not report[key]:
                    print(json.dumps(report, indent=2))
                    raise ValueError(f"Entorno o plantillas incompletos. Consulta {PIPELINE.parent / 'installation/README.md'} y ejecuta run.sh sam doctor.")
            if args.command == "segment":
                command = [sys.executable, CORE / "Instance_Segmentation_Model/run_inference_pddl.py",
                           "--manifest", manifest_path, "--output", output,
                           "--min-score", args.min_score, "--stability-score", args.stability_score,
                           "--segmentor", args.segmentor, "--backbone", args.backbone]
                for part in parts:
                    command.extend(["--part", part["key"]])
                for frame in frames:
                    command.extend(["--frame", frame["name"]])
                run(command, CORE / "Instance_Segmentation_Model", args.dry_run)
            else:
                pose(args, manifest_path, data, parts, frames, output)
        return 0
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
