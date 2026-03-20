"""
Validate our submission pipeline against the held-out val set.
Computes detection mAP and classification mAP using pycocotools.
"""
import json
import numpy as np
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

SCRIPT_DIR = Path(__file__).parent
DATA_ROOT = SCRIPT_DIR / "data" / "coco_dataset" / "train"
YOLO_ROOT = SCRIPT_DIR / "data" / "yolo"


def load_val_ground_truth():
    """Load COCO annotations for val images only."""
    with open(DATA_ROOT / "annotations.json") as f:
        coco = json.load(f)
    with open(YOLO_ROOT / "split_info.json") as f:
        split = json.load(f)

    val_ids = set(split["val_ids"])

    # Filter to val images only
    val_images = [im for im in coco["images"] if im["id"] in val_ids]
    val_anns = [a for a in coco["annotations"] if a["image_id"] in val_ids]

    gt = {
        "images": val_images,
        "categories": coco["categories"],
        "annotations": val_anns,
    }
    return gt


def evaluate_predictions(gt_data, pred_file):
    """Evaluate predictions using COCO mAP."""
    # Save GT to temp file for pycocotools
    gt_path = SCRIPT_DIR / "data" / "val_gt_temp.json"
    with open(gt_path, "w") as f:
        json.dump(gt_data, f)

    coco_gt = COCO(str(gt_path))

    # Load predictions
    with open(pred_file) as f:
        preds = json.load(f)

    if not preds:
        print("No predictions!")
        return

    # 1. Detection mAP (category-agnostic)
    # Remap all predictions and GT to single category for detection eval
    det_preds = []
    for p in preds:
        det_preds.append({
            "image_id": p["image_id"],
            "category_id": 1,  # single class
            "bbox": p["bbox"],
            "score": p["score"],
        })

    det_gt_data = {
        "images": gt_data["images"],
        "categories": [{"id": 1, "name": "product"}],
        "annotations": [
            {**a, "category_id": 1} for a in gt_data["annotations"]
        ],
    }
    det_gt_path = SCRIPT_DIR / "data" / "val_gt_det_temp.json"
    with open(det_gt_path, "w") as f:
        json.dump(det_gt_data, f)

    coco_det_gt = COCO(str(det_gt_path))
    coco_det_dt = coco_det_gt.loadRes(det_preds)
    eval_det = COCOeval(coco_det_gt, coco_det_dt, "bbox")
    eval_det.evaluate()
    eval_det.accumulate()
    print("\n=== DETECTION mAP (category-agnostic) ===")
    eval_det.summarize()
    det_map50 = eval_det.stats[1]  # mAP@0.5

    # 2. Classification mAP (with correct categories)
    coco_cls_dt = coco_gt.loadRes(preds)
    eval_cls = COCOeval(coco_gt, coco_cls_dt, "bbox")
    eval_cls.evaluate()
    eval_cls.accumulate()
    print("\n=== CLASSIFICATION mAP (with categories) ===")
    eval_cls.summarize()
    cls_map50 = eval_cls.stats[1]  # mAP@0.5

    # Final score
    score = 0.7 * det_map50 + 0.3 * cls_map50
    print(f"\n{'='*50}")
    print(f"Detection mAP@0.5:       {det_map50:.4f}")
    print(f"Classification mAP@0.5:  {cls_map50:.4f}")
    print(f"FINAL SCORE:             {score:.4f}")
    print(f"  (0.7 × {det_map50:.4f} + 0.3 × {cls_map50:.4f})")
    print(f"{'='*50}")

    # Cleanup temp files
    gt_path.unlink(missing_ok=True)
    det_gt_path.unlink(missing_ok=True)

    return score


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, help="Path to predictions.json")
    args = parser.parse_args()

    gt = load_val_ground_truth()
    print(f"Val images: {len(gt['images'])}")
    print(f"Val annotations: {len(gt['annotations'])}")

    evaluate_predictions(gt, args.predictions)


if __name__ == "__main__":
    main()
