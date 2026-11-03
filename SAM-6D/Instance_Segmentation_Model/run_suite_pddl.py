"""Benchmark the original SAM-6D ISM assignment against every prepared CAD."""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_cad_root = next(p for p in _Path(__file__).resolve().parents if (p/'config/settings.json').is_file())
_sys.path.insert(0, str(_cad_root/'pipeline'))

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import shutil
import time

from hydra import compose, initialize
from hydra.utils import instantiate
from segmentor_runtime import configure, move_segmentor
import numpy as np
from PIL import Image
import torch
import trimesh

from model.utils import Detections, mask_to_rle
from run_inference_pddl import visualize
from utils.bbox_utils import CropResizePad
from utils.poses.pose_utils import get_obj_poses_from_template_level, load_index_level_in_level2


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@contextmanager
def measured(timings, name, device):
    synchronize(device)
    start = time.perf_counter()
    try:
        yield
    finally:
        synchronize(device)
        timings[name] = time.perf_counter() - start


def choose_device(requested):
    available = torch.cuda.is_available()
    if requested == "cuda" and not available:
        raise RuntimeError("Se solicitó CUDA, pero no está disponible; no se ejecutará en CPU silenciosamente.")
    return torch.device("cuda" if requested == "cuda" or (requested == "auto" and available) else "cpu")


def aggregate(scores, method):
    if method == "mean":
        return scores.mean(dim=-1)
    if method == "median":
        return scores.median(dim=-1).values
    if method == "max":
        return scores.max(dim=-1).values
    if method == "avg_5":
        return scores.topk(k=5, dim=-1).values.mean(dim=-1)
    raise ValueError(f"Agregación desconocida: {method}")


def compare_all(model, query, device):
    """Same metric/aggregation as compute_semantic_score, with exclusive CAD timers."""
    values, views, times = [], [], []
    for index in range(len(model.ref_data["descriptors"])):
        clock = {}
        with measured(clock, "semantic_s", device):
            raw = model.matching_config.metric(query, model.ref_data["descriptors"][index:index + 1])[:, 0]
            values.append(aggregate(raw, model.matching_config.aggregation_function))
            views.append(raw.argmax(dim=-1))
        times.append(clock["semantic_s"])
    return torch.stack(values, dim=1), torch.stack(views, dim=1), times


def assign(semantic, views, threshold):
    best, objects = semantic.max(dim=1)
    selected = torch.arange(len(best), device=semantic.device)[best > threshold]
    objects = objects[selected]
    return selected, objects, best[selected], views[selected, objects]


def hash_files(paths):
    digest = hashlib.sha256()
    for path in paths:
        with Path(path).open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def load_references(model, parts, base, device, cache):
    cache.mkdir(parents=True, exist_ok=True)
    checkpoint = Path("checkpoints/dinov2/dinov2_vitl14_pretrain.pth")
    # All inputs affecting cached float32 descriptors, including extractor code.
    extractor = hash_files([checkpoint, Path("model/dinov2.py"), Path("utils/bbox_utils.py")])
    descriptors, appearance, points, records = [], [], [], []
    processor = CropResizePad(224)
    for part in parts:
        start = time.perf_counter()
        folder = base / part["sam_template_dir"]
        inputs = [folder / f"{prefix}_{i}.png" for i in range(42) for prefix in ("rgb", "mask")]
        inputs.append(base / part["sam_ply"])
        signature = {"version": 2, "extractor_sha256": extractor, "inputs_sha256": hash_files(inputs),
                     "torch": str(torch.__version__), "dtype": "float32", "size": 224,
                     "patch_threshold": float(model.descriptor_model.validpatch_thresh),
                     "sampling_seed": part["category_id"], "views": 42}
        key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        target = cache / f"{part['key']}_{key}.pt"
        hit = target.is_file()
        if hit:
            data = torch.load(target, map_location="cpu", weights_only=True)
            if data["signature"] != signature:
                raise ValueError(f"Caché de referencias inconsistente: {target}")
        else:
            print(f"Referencias 42 vistas: {part['key']}", flush=True)
            rgbs, masks, boxes = [], [], []
            for index in range(42):
                with Image.open(folder / f"mask_{index}.png") as source:
                    image = source.convert("L")
                    box = image.getbbox()
                    if box is None:
                        raise ValueError(f"Plantilla vacía: {part['key']}, {index}")
                    mask = torch.from_numpy(np.array(image) / 255).float()
                with Image.open(folder / f"rgb_{index}.png") as source:
                    rgb = torch.from_numpy(np.array(source.convert("RGB")) / 255).float()
                rgb = (rgb - torch.tensor([0.485, 0.456, 0.406])) / torch.tensor([0.229, 0.224, 0.225])
                rgbs.append(rgb * mask[:, :, None])
                masks.append(mask)
                boxes.append(box)
            boxes = torch.tensor(np.array(boxes))
            rgbs = processor(images=torch.stack(rgbs).permute(0, 3, 1, 2), boxes=boxes).to(device)
            masks = processor(images=torch.stack(masks)[:, None], boxes=boxes).to(device)
            cls, patches = [], []
            for offset in range(0, 42, model.descriptor_model.chunk_size):
                end = offset + model.descriptor_model.chunk_size
                c, p = model.descriptor_model.compute_cls_and_patch_features(rgbs[offset:end], masks[offset:end, 0])
                cls.append(c.cpu())
                patches.append(p.cpu())
            # Per-CAD seed makes cache hits and misses independent of catalog order.
            np.random.seed(part["category_id"])
            cloud = trimesh.load_mesh(base / part["sam_ply"]).sample(2048)
            data = {"signature": signature, "cls": torch.cat(cls), "patch": torch.cat(patches),
                    "points": torch.from_numpy(cloud.astype(np.float32) / 1000)}
            temporary = target.with_suffix(".tmp")
            torch.save(data, temporary)
            temporary.replace(target)
        descriptors.append(data["cls"].to(device))
        appearance.append(data["patch"].to(device))
        points.append(data["points"].to(device))
        synchronize(device)
        records.append({"part_key": part["key"], "cache_hit": hit, "cache_key": key,
                        "preparation_s": time.perf_counter() - start})
    poses = get_obj_poses_from_template_level(level=2, pose_distribution="all")
    poses[:, :3, 3] *= 0.4
    model.ref_data = {"descriptors": torch.stack(descriptors), "appe_descriptors": torch.stack(appearance),
                      "pointcloud": torch.stack(points),
                      "poses": torch.tensor(poses[load_index_level_in_level2(0, "all")], dtype=torch.float32, device=device)}
    return {"parts": records, "extractor_sha256": extractor}


STAGES = ("input_s", "sam_s", "proposal_filter_s", "dino_s", "semantic_s", "assignment_s",
          "appearance_s", "projection_s", "geometry_s", "postprocess_s")


def process_frame(model, frame, parts, base, device, min_score, confidence):
    synchronize(device)
    start = time.perf_counter()
    timing = dict.fromkeys(STAGES, 0.0)
    with measured(timing, "input_s", device):
        with Image.open(base / frame["rgb"]) as source:
            rgb = source.convert("RGB")
        with Image.open(base / frame["depth"]) as source:
            depth = np.array(source).astype(np.int32)
        camera = json.loads((base / frame["camera"]).read_text())
        batch = {"depth": torch.from_numpy(depth)[None].to(device),
                 "cam_intrinsic": torch.tensor(camera["cam_K"], dtype=torch.float32, device=device).reshape(1, 3, 3),
                 "depth_scale": torch.tensor([camera["depth_scale"]], device=device)}
    with measured(timing, "sam_s", device):
        proposals = model.segmentor_model.generate_masks(np.array(rgb))
    detections = Detections(proposals)
    counts = {"sam": len(detections)}
    with measured(timing, "proposal_filter_s", device):
        if len(detections):
            detections.remove_very_small_detections(model.post_processing_config.mask_post_processing)
        if len(detections):
            keep = ((detections.masks > 0.5) & (batch["depth"][0] > 0)).sum(dim=(1, 2)) > 32
            detections.filter(keep)
    counts["valid_depth_and_size"] = len(detections)
    semantic = views = None
    cad_times = [0.0] * len(parts)
    results, masks = [], []
    if len(detections):
        detections.add_attribute("proposal_ids", torch.arange(len(detections), device=device))
        with measured(timing, "dino_s", device):
            query, patches = model.descriptor_model(np.array(rgb), detections)
        with measured(timing, "semantic_s", device):
            semantic, views, cad_times = compare_all(model, query, device)
        with measured(timing, "assignment_s", device):
            selected, objects, values, best = assign(semantic, views, model.matching_config.confidence_thresh)
            detections.filter(selected)
            patches = patches[selected]
        counts["semantic_selected"] = len(detections)
        if len(detections):
            with measured(timing, "appearance_s", device):
                appearance, reference = model.compute_appearance_score(best, objects, patches)
            with measured(timing, "projection_s", device):
                uv = model.project_template_to_image(best, objects, batch, detections.masks)
            with measured(timing, "geometry_s", device):
                geometry, visible = model.compute_geometric_score(uv, detections, patches, reference, model.visible_thred)
            with measured(timing, "postprocess_s", device):
                score = (values + appearance + geometry * visible) / (2 + visible)
                detections.add_attribute("scores", score)
                detections.add_attribute("object_ids", objects)
                detections.add_attribute("semantic_scores", values)
                detections.add_attribute("appearance_scores", appearance)
                detections.add_attribute("geometry_scores", geometry)
                detections.add_attribute("visible_ratios", visible)
                detections.filter(torch.isfinite(score) & (score >= min_score))
                if len(detections):
                    detections.apply_nms_per_object_id(model.post_processing_config.nms_thresh)
                    detections.to_numpy()
                    for i in np.argsort(-detections.scores):
                        part = parts[int(detections.object_ids[i])]
                        x1, y1, x2, y2 = detections.boxes[i].tolist()
                        mask = detections.masks[i] > 0.5
                        results.append({"scene_id": 0, "image_id": frame["frame_id"],
                                        "category_id": part["category_id"], "part_key": part["key"],
                                        "part_name": part["name"], "proposal_id": int(detections.proposal_ids[i]),
                                        "score": float(detections.scores[i]),
                                        "semantic_score": float(detections.semantic_scores[i]),
                                        "appearance_score": float(detections.appearance_scores[i]),
                                        "geometry_score": float(detections.geometry_scores[i]),
                                        "visible_ratio": float(detections.visible_ratios[i]),
                                        "bbox": [x1, y1, x2 - x1, y2 - y1], "segmentation": mask_to_rle(mask)})
                        masks.append(mask)
    with measured(timing, "comparison_export_s", device):
        semantic_cpu = semantic.cpu().tolist() if semantic is not None else []
        views_cpu = views.cpu().tolist() if views is not None else []
        region = results[0]["proposal_id"] if results else (
            int(semantic.max(dim=1).values.argmax()) if semantic is not None else None)
        comparison = [{"part_key": p["key"], "part_name": p["name"], "category_id": p["category_id"],
                       "semantic_score": semantic_cpu[region][j] if region is not None else None,
                       "best_template": views_cpu[region][j] if region is not None else None,
                       "semantic_comparison_s": cad_times[j],
                       "compared_proposals": counts["valid_depth_and_size"]}
                      for j, p in enumerate(parts)]
    synchronize(device)
    timing["processing_s"] = time.perf_counter() - start
    for detection in results:
        detection["time"] = timing["processing_s"]
    candidate = {k: v for k, v in results[0].items() if k != "segmentation"} if results else None
    accepted = [i for i, d in enumerate(results) if d["score"] >= confidence]
    case = {"name": frame["name"], "frame_id": frame["frame_id"], "status": "ok", "device": str(device),
            "recognized": bool(accepted), "candidate": candidate,
            "visible_components": sorted({results[i]["part_key"] for i in accepted}),
            "proposal_counts": counts, "detections": len(results), "timing": timing,
            "comparison_region_proposal_id": region,
            "comparison_scope": "primary_detection" if results else "best_semantic_proposal_without_retained_detection",
            "comparisons": comparison, "proposal_semantic_scores": semantic_cpu,
            "proposal_best_templates": views_cpu}
    return case, rgb, results, masks, accepted


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame", action="append")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--min-score", type=float, default=0.3)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--stability-score", type=float, default=0.97)
    parser.add_argument("--segmentor", choices=("sam", "fastsam"), default="sam")
    parser.add_argument("--backbone", choices=("vit_h", "vit_l", "vit_b"), default="vit_h")
    parser.add_argument("--pose",action="store_true",help="Estimate PEM poses of accepted detections")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--reference-cache", type=Path, required=True)
    args = parser.parse_args()
    if not (0 <= args.min_score <= args.confidence <= 1) or not 0<=args.stability_score<=1 or args.threads < 1:
        parser.error("Se requiere 0 <= min-score <= confidence <= 1 y threads >= 1.")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("La salida debe estar vacía: se preservan las ejecuciones anteriores.")
    dataset = json.loads(args.manifest.read_text())
    parts = dataset["parts"]  # Always the full CAD catalog; no class restriction by image name.
    frames = [f for f in dataset["frames"] if not args.frame or f["name"] in args.frame]
    unknown = set(args.frame or ()) - {f["name"] for f in frames}
    if unknown or not frames or not parts:
        parser.error(f"Selección vacía o capturas desconocidas: {sorted(unknown)}")
    device = choose_device(args.device)
    if args.pose and device.type!="cuda":parser.error("PEM requiere CUDA")
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    np.random.seed(0)
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    try:
        gpu = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
        gpu_diagnostic = {"returncode": gpu.returncode, "output": (gpu.stdout + gpu.stderr).strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        gpu_diagnostic = {"error": str(error)}
    run = {"schema_version": 1, "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
           "manifest": str(args.manifest.resolve()), "manifest_sha256": hash_files([args.manifest]),
           "catalog": [{k: p[k] for k in ("key", "name", "category_id")} for p in parts],
           "selected_frames": [f["name"] for f in frames], "frames": [],
           "device": {"requested": args.device, "used": str(device), "torch": str(torch.__version__),
                      "torch_cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
                      "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                      "nvidia_smi": gpu_diagnostic, "threads": torch.get_num_threads(),
                      "python": platform.python_version(), "cpu": platform.processor()},
           "settings": {"min_score": args.min_score, "confidence": args.confidence,
                        "stability_score": args.stability_score, "pose":args.pose, "pose_backend":"pem" if args.pose else None, "segmentor": args.segmentor, "backbone": args.backbone, "warmup_cases": 0, "repeats": 1,
                        "filename_used_for_classification": False, "pointcloud_seed_per_cad": True},
           "timing_definitions": {
               "processing_s": "RGB-D read + SAM + DINO + all-CAD comparison + original ISM refinement + RLE; excludes image/file writes and model/reference setup",
               "output_s": "RGB, depth, mask, overlay and detection files written for this case",
               "case_s": "processing_s + output_s + optional pose_pipeline_s; excludes suite report generation and ISM model/reference setup",
               "semantic_comparison_s": "Exclusive DINO semantic metric and aggregation for one CAD over all valid proposals; SAM and DINO extraction shared by all CADs",
               "comparison_scores": "All-CAD DINO similarities on the SAME primary proposal; only the assigned CAD receives appearance and geometry refinement as in SAM-6D",
               "zero_comparison_time": "No valid proposal; CAD comparison was not performed",
               "synchronization": "CUDA synchronized before and after each timer; perf_counter wall time",
               "first_case": "No discarded warmup; first case can include cold initialization costs"}, "setup_timing": {}}
    write_json(args.output / "run.json", run)
    try:
        with measured(run["setup_timing"], "model_load_s", device):
            with initialize(version_base=None, config_path="configs"):
                cfg = compose(config_name="run_inference.yaml", overrides=["model=ISM_fastsam"] if args.segmentor == "fastsam" else [])
            run['settings']['stability_score_yaml']=float(cfg.model.segmentor_model.stability_score_thresh) if args.segmentor=='sam' else None
            configure(cfg, args.segmentor, args.backbone, args.stability_score)
            model = instantiate(cfg.model)
            model.eval()
            model.descriptor_model.model.to(device)
            model.descriptor_model.model.device = device
            move_segmentor(model, device, args.segmentor)
        run['settings']['stability_score_effective']=float(model.segmentor_model.stability_score_thresh) if args.segmentor=='sam' else None
        run['settings']['stability_score_applies']=args.segmentor=='sam'
        if args.segmentor=='sam' and not np.isclose(run['settings']['stability_score_effective'],args.stability_score):raise ValueError('SAM no aplicó el umbral solicitado')
        with measured(run["setup_timing"], "references_s", device):
            run["references"] = load_references(model, parts, args.manifest.parent, device, args.reference_cache)
        run["algorithm"] = {"semantic_aggregation": model.matching_config.aggregation_function,
                            "semantic_threshold": float(model.matching_config.confidence_thresh),
                            "nms_iou": float(model.post_processing_config.nms_thresh),
                            "sam": args.backbone if args.segmentor=="sam" else "fastsam", "dino": "dinov2_vitl14", "views_per_cad": 42,
                            "final_score": "(semantic + appearance + geometry * visible_ratio) / (2 + visible_ratio)",
                            "worker_sha256": hash_files([Path(__file__)])}
        write_json(args.output / "run.json", run)
        for number, frame in enumerate(frames, 1):
            print(f"[{number}/{len(frames)}] {frame['name']} | {device} | {len(parts)} CAD", flush=True)
            case, rgb, detections, masks, accepted = process_frame(
                model, frame, parts, args.manifest.parent, device, args.min_score, args.confidence)
            folder = args.output / "cases" / frame["name"]
            folder.mkdir(parents=True)
            with measured(case["timing"], "output_s", device):
                rgb.save(folder / "rgb.png")
                with Image.open(args.manifest.parent / frame["depth"]) as depth:
                    depth.save(folder / "depth_raw.png")
                visualize(rgb, [detections[i] for i in accepted], [masks[i] for i in accepted], folder / "overlay.png")
                for i, mask in enumerate(masks):
                    Image.fromarray(mask.astype(np.uint8) * 255).save(folder / f"mask_{i:03d}.png")
                write_json(folder / "detection_ism.json", detections)
            shutil.copy2(args.manifest.parent/frame['camera'],folder/'camera.json')
            if args.pose:
                sys.path.append(str(_cad_root/'pipeline'))
                from suites.sam6d_suite_pose import run_case_pose
                case['pose']=run_case_pose(folder,frame,parts,args.manifest.parent,args.confidence)
                case['timing']['pose_pipeline_s']=case['pose']['elapsed_s']
            case["timing"]["case_s"] = case["timing"]["processing_s"] + case["timing"]["output_s"] + case["timing"].get("pose_pipeline_s",0)
            write_json(folder / "case.json", case)
            run["frames"].append(case)
            write_json(args.output / "run.json", run)
            candidate = case["candidate"]
            label = candidate["part_key"] if case["recognized"] else "sin identificación fiable"
            print(f"  {label} | proceso {case['timing']['processing_s']:.3f} s | export {case['timing']['output_s']:.3f} s", flush=True)
        run["status"] = "complete"
    except Exception as error:
        run["status"] = "error"
        run["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        synchronize(device)
        run["wall_s"] = time.perf_counter() - start
        run["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(args.output / "run.json", run)


if __name__ == "__main__":
    main()
