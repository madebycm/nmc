"""
Server-side validation: run full pipeline on val images, compute blended mAP.
Usage: python validate_server.py
"""
import json
import numpy as np
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import subprocess
import tempfile

DATA_ROOT = Path("/mnt/SFS-qZE4t9Aw/data")
COCO_ROOT = DATA_ROOT / "coco_dataset" / "train"
YOLO_ROOT = DATA_ROOT / "yolo"
SUBMISSION_DIR = Path("/clade/ng/submission")


def get_val_image_ids():
    """Get val image IDs from YOLO val directory."""
    with open(COCO_ROOT / "annotations.json") as f:
        coco = json.load(f)

    val_stems = set(p.stem for p in (YOLO_ROOT / "val" / "images").iterdir())
    val_ids = set()
    for im in coco["images"]:
        if Path(im["file_name"]).stem in val_stems:
            val_ids.add(im["id"])
    return val_ids


def run_inference(val_img_dir, output_path):
    """Run the submission pipeline on val images."""
    cmd = [
        "python", str(SUBMISSION_DIR / "run.py"),
        "--input", str(val_img_dir),
        "--output", str(output_path),
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"STDERR: {result.stderr}")
        raise RuntimeError(f"Inference failed: {result.returncode}")
    print(f"Inference complete. Output: {output_path}")


def evaluate(pred_path, maxDets=100):
    """Compute detection + classification mAP."""
    with open(COCO_ROOT / "annotations.json") as f:
        coco_data = json.load(f)

    val_ids = get_val_image_ids()
    val_images = [im for im in coco_data["images"] if im["id"] in val_ids]
    val_anns = [a for a in coco_data["annotations"] if a["image_id"] in val_ids]
    print(f"Val: {len(val_images)} images, {len(val_anns)} annotations")

    # Count images with >100 GT
    from collections import Counter
    img_counts = Counter(a["image_id"] for a in val_anns)
    over100 = sum(1 for c in img_counts.values() if c > maxDets)
    print(f"Images with >{maxDets} GT boxes: {over100}")

    with open(pred_path) as f:
        preds = json.load(f)
    print(f"Predictions: {len(preds)}")

    # Only keep predictions for val images
    preds = [p for p in preds if p["image_id"] in val_ids]
    print(f"Val predictions: {len(preds)}")

    if not preds:
        print("No predictions for val set!")
        return 0.0

    # --- Detection mAP (single-class) ---
    det_gt = {
        "images": val_images,
        "categories": [{"id": 1, "name": "product"}],
        "annotations": [{**a, "id": i, "category_id": 1}
                        for i, a in enumerate(val_anns, 1)],
    }
    det_preds = [{
        "image_id": p["image_id"],
        "category_id": 1,
        "bbox": p["bbox"],
        "score": p["score"],
    } for p in preds]

    gt_path = Path("/tmp/val_gt_det.json")
    with open(gt_path, "w") as f:
        json.dump(det_gt, f)

    coco_gt = COCO(str(gt_path))
    coco_dt = coco_gt.loadRes(det_preds)
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.params.maxDets = [1, 10, maxDets]
    ev.evaluate()
    ev.accumulate()
    print(f"\n=== DETECTION mAP (maxDets={maxDets}) ===")
    # Extract mAP@0.5 manually to avoid numpy 2.x crash
    precision = ev.eval["precision"]  # (T, R, K, A, M)
    # IoU=0.5 is index 0, all areas index 0, maxDets[-1] index 2
    p50 = precision[0, :, :, 0, 2]
    p50 = p50[p50 > -1]
    det_map50 = float(np.mean(p50)) if len(p50) > 0 else 0.0
    print(f"Detection mAP@0.5: {det_map50:.4f}")

    # --- Classification mAP (multi-class) ---
    cls_gt = {
        "images": val_images,
        "categories": coco_data["categories"],
        "annotations": [{**a, "id": i}
                        for i, a in enumerate(val_anns, 1)],
    }
    cls_path = Path("/tmp/val_gt_cls.json")
    with open(cls_path, "w") as f:
        json.dump(cls_gt, f)

    coco_cls_gt = COCO(str(cls_path))
    coco_cls_dt = coco_cls_gt.loadRes(preds)
    ev2 = COCOeval(coco_cls_gt, coco_cls_dt, "bbox")
    ev2.params.maxDets = [1, 10, maxDets]
    ev2.evaluate()
    ev2.accumulate()
    print(f"\n=== CLASSIFICATION mAP (maxDets={maxDets}) ===")
    p50c = ev2.eval["precision"]
    p50c = p50c[0, :, :, 0, 2]
    p50c = p50c[p50c > -1]
    cls_map50 = float(np.mean(p50c)) if len(p50c) > 0 else 0.0
    print(f"Classification mAP@0.5: {cls_map50:.4f}")

    # Blended score
    score = 0.7 * det_map50 + 0.3 * cls_map50
    print(f"\n{'=' * 50}")
    print(f"Detection mAP@0.5:       {det_map50:.4f} (× 0.7 = {0.7*det_map50:.4f})")
    print(f"Classification mAP@0.5:  {cls_map50:.4f} (× 0.3 = {0.3*cls_map50:.4f})")
    print(f"BLENDED SCORE:           {score:.4f}")
    print(f"{'=' * 50}")

    return score


def main():
    print("=== Full Pipeline Validation ===\n")

    # Create symlinks to val images in a temp dir
    val_ids = get_val_image_ids()
    with open(COCO_ROOT / "annotations.json") as f:
        coco_data = json.load(f)
    id_to_file = {im["id"]: im["file_name"] for im in coco_data["images"]}

    val_dir = Path("/tmp/val_images")
    val_dir.mkdir(exist_ok=True)
    for vid in val_ids:
        src = COCO_ROOT / "images" / id_to_file[vid]
        dst = val_dir / id_to_file[vid]
        if not dst.exists():
            dst.symlink_to(src)

    pred_path = Path("/tmp/val_predictions.json")

    # Run inference
    run_inference(val_dir, pred_path)

    # Evaluate
    evaluate(pred_path)


if __name__ == "__main__":
    main()
