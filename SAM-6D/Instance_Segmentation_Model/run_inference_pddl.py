"""Joint CAD classification of SAM proposals using the official SAM-6D ISM."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
import torch
import trimesh
from hydra import compose, initialize
from hydra.utils import instantiate
from segmentor_runtime import configure, move_segmentor

from model.utils import Detections, mask_to_rle
from utils.bbox_utils import CropResizePad
from utils.poses.pose_utils import get_obj_poses_from_template_level, load_index_level_in_level2


def visualize(rgb, detections, masks, path):
    overlay = np.array(rgb).copy()
    for detection, mask in zip(detections, masks):
        cid = detection["category_id"]
        color = np.array([(71 * cid) % 200 + 55, (137 * cid) % 200 + 55, (43 * cid) % 200 + 55])
        overlay[mask] = (0.6 * overlay[mask] + 0.4 * color).astype(np.uint8)
    image = Image.fromarray(overlay)
    draw = ImageDraw.Draw(image)
    for detection in detections:
        x, y, w, h = detection["bbox"]
        draw.rectangle([x, y, x + w, y + h], outline="white", width=2)
        draw.text((x, max(0, y - 12)), f"{detection['part_name']} {detection['score']:.2f}", fill="white", stroke_width=1, stroke_fill="black")
    image.save(path)


def load_references(model, parts, base, device):
    descriptors, appearance, pointcloud = [], [], []
    processor = CropResizePad(224)
    for part in parts:
        print(f"Referencias DINOv2: {part['key']}", flush=True)
        folder = base / part["sam_template_dir"]
        templates, masks, boxes = [], [], []
        for index in range(42):
            with Image.open(folder / f"mask_{index}.png") as source:
                mask = source.convert("L")
                box = mask.getbbox()
                if box is None:
                    raise ValueError(f"Plantilla vacía: {folder}, vista {index}")
                mask = torch.from_numpy(np.array(mask) / 255).float()
            with Image.open(folder / f"rgb_{index}.png") as source:
                rgb = torch.from_numpy(np.array(source.convert("RGB")) / 255).float()
            # Match the normalization used by process_rgb_proposals for queries.
            rgb = (rgb - torch.tensor([0.485, 0.456, 0.406])) / torch.tensor([0.229, 0.224, 0.225])
            templates.append(rgb * mask[:, :, None])
            masks.append(mask)
            boxes.append(box)
        boxes = torch.tensor(np.array(boxes))
        templates = processor(images=torch.stack(templates).permute(0, 3, 1, 2), boxes=boxes).to(device)
        masks = processor(images=torch.stack(masks)[:, None], boxes=boxes).to(device)
        descriptors.append(model.descriptor_model.compute_features(templates, token_name="x_norm_clstoken"))
        appearance.append(model.descriptor_model.compute_masked_patch_feature(templates, masks[:, 0]))
        mesh = trimesh.load_mesh(base / part["sam_ply"])
        pointcloud.append(torch.from_numpy(mesh.sample(2048).astype(np.float32) / 1000.0))
    poses = get_obj_poses_from_template_level(level=2, pose_distribution="all")
    poses[:, :3, 3] *= 0.4
    model.ref_data = {"descriptors": torch.stack(descriptors),
                      "appe_descriptors": torch.stack(appearance),
                      "pointcloud": torch.stack(pointcloud).to(device),
                      "poses": torch.tensor(poses[load_index_level_in_level2(0, "all")], dtype=torch.float32, device=device)}


def segment_frame(model, frame, parts, base, device, min_score):
    with Image.open(base / frame["rgb"]) as source:
        rgb = source.convert("RGB")
    with Image.open(base / frame["depth"]) as source:
        depth = np.array(source).astype(np.int32)
    camera = json.loads((base / frame["camera"]).read_text())
    batch = {"depth": torch.from_numpy(depth)[None].to(device),
             "cam_intrinsic": torch.tensor(camera["cam_K"], dtype=torch.float32, device=device).reshape(1, 3, 3),
             "depth_scale": torch.tensor([camera["depth_scale"]], device=device)}
    start = time.perf_counter()
    proposals = model.segmentor_model.generate_masks(np.array(rgb))
    print(f"{frame['name']}: propuestas SAM generadas", flush=True)
    detections = Detections(proposals)
    if len(detections) == 0:
        return rgb, [], []
    detections.remove_very_small_detections(model.post_processing_config.mask_post_processing)
    if len(detections) == 0:
        return rgb, [], []
    # A visible mask without valid depth cannot contribute a geometric score or PEM pose.
    valid_depth = batch["depth"][0] > 0
    keep = ((detections.masks > 0.5) & valid_depth).sum(dim=(1, 2)) > 32
    detections.filter(keep)
    if len(detections) == 0:
        return rgb, [], []
    query, patches = model.descriptor_model(np.array(rgb), detections)
    selected, objects, semantic, best = model.compute_semantic_score(query)
    if len(selected) == 0:
        return rgb, [], []
    detections.filter(selected)
    patches = patches[selected]
    appearance, reference = model.compute_appearance_score(best, objects, patches)
    uv = model.project_template_to_image(best, objects, batch, detections.masks)
    geometry, visible = model.compute_geometric_score(uv, detections, patches, reference, model.visible_thred)
    score = (semantic + appearance + geometry * visible) / (2 + visible)
    detections.add_attribute("scores", score)
    detections.add_attribute("object_ids", objects)
    detections.filter(torch.isfinite(score) & (score >= min_score))
    if len(detections) == 0:
        return rgb, [], []
    detections.apply_nms_per_object_id(model.post_processing_config.nms_thresh)
    detections.to_numpy()
    elapsed = time.perf_counter() - start
    results, masks = [], []
    for index in np.argsort(-detections.scores):
        part = parts[int(detections.object_ids[index])]
        x1, y1, x2, y2 = detections.boxes[index].tolist()
        mask = detections.masks[index] > 0.5
        results.append({"scene_id": 0, "image_id": frame["frame_id"],
                        "category_id": part["category_id"], "part_key": part["key"],
                        "part_name": part["name"], "score": float(detections.scores[index]),
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "segmentation": mask_to_rle(mask), "time": elapsed})
        masks.append(mask)
    return rgb, results, masks


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--part", action="append")
    parser.add_argument("--frame", action="append")
    parser.add_argument("--min-score", type=float, default=0.3)
    parser.add_argument("--stability-score", type=float, default=0.97)
    parser.add_argument("--segmentor", choices=("sam", "fastsam"), default="sam")
    parser.add_argument("--backbone", choices=("vit_h", "vit_l", "vit_b"), default="vit_h")
    args = parser.parse_args()
    base = args.manifest.parent
    dataset = json.loads(args.manifest.read_text())
    parts = [p for p in dataset["parts"] if not args.part or p["key"] in args.part]
    frames = [f for f in dataset["frames"] if not args.frame or f["name"] in args.frame]
    if not parts or not frames:
        raise ValueError("Seleccionar al menos una pieza y una captura.")
    torch.manual_seed(0)
    np.random.seed(0)
    with initialize(version_base=None, config_path="configs"):
        cfg = compose(config_name="run_inference.yaml", overrides=["model=ISM_fastsam"] if args.segmentor == "fastsam" else [])
    configure(cfg, args.segmentor, args.backbone, args.stability_score)
    model = instantiate(cfg.model)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.descriptor_model.model.to(device)
    model.descriptor_model.model.device = device
    move_segmentor(model, device, args.segmentor)
    load_references(model, parts, base, device)
    summary = {"status": "inferred", "device": str(device), "min_score": args.min_score,
               "stability_score": args.stability_score, "segmentor": args.segmentor, "backbone": args.backbone,
               "catalog": [p["key"] for p in parts], "frames": []}
    for frame in frames:
        print(f"Analizando RGB-D: {frame['name']}", flush=True)
        folder = args.output / frame["name"]
        folder.mkdir(parents=True, exist_ok=True)
        rgb, detections, masks = segment_frame(model, frame, parts, base, device, args.min_score)
        (folder / "detection_ism.json").write_text(json.dumps(detections, indent=2) + "\n")
        visualize(rgb, detections, masks, folder / "overlay.png")
        for index, mask in enumerate(masks):
            Image.fromarray(mask.astype(np.uint8) * 255).save(folder / f"mask_{index:03d}.png")
        summary["frames"].append({"name": frame["name"], "detections": len(detections)})
        print(f"{frame['name']}: {len(detections)} detecciones", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
