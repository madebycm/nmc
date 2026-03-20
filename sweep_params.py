"""Sweep score fusion and top-K params on val predictions."""
import json
import numpy as np
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from collections import defaultdict

COCO_ROOT = Path("/mnt/SFS-qZE4t9Aw/data/coco_dataset/train")
YOLO_ROOT = Path("/mnt/SFS-qZE4t9Aw/data/yolo")

with open(COCO_ROOT / "annotations.json") as f:
    coco_data = json.load(f)
val_stems = set(p.stem for p in (YOLO_ROOT / "val" / "images").iterdir())
val_ids = set()
for im in coco_data["images"]:
    if Path(im["file_name"]).stem in val_stems:
        val_ids.add(im["id"])
val_images = [im for im in coco_data["images"] if im["id"] in val_ids]
val_anns = [a for a in coco_data["annotations"] if a["image_id"] in val_ids]

det_gt = {
    "images": val_images,
    "categories": [{"id": 1, "name": "product"}],
    "annotations": [{**a, "id": i, "category_id": 1} for i, a in enumerate(val_anns, 1)],
}
cls_gt = {
    "images": val_images,
    "categories": coco_data["categories"],
    "annotations": [{**a, "id": i} for i, a in enumerate(val_anns, 1)],
}
with open("/tmp/dg.json", "w") as f:
    json.dump(det_gt, f)
with open("/tmp/cg.json", "w") as f:
    json.dump(cls_gt, f)

with open("/tmp/val_flip_tta.json") as f:
    preds = json.load(f)
preds = [p for p in preds if p["image_id"] in val_ids]


def eval_preds(preds_list):
    det_preds = [{"image_id": p["image_id"], "category_id": 1, "bbox": p["bbox"], "score": p["score"]} for p in preds_list]
    cg = COCO("/tmp/dg.json")
    cd = cg.loadRes(det_preds)
    ev = COCOeval(cg, cd, "bbox")
    ev.params.maxDets = [1, 10, 100]
    ev.evaluate()
    ev.accumulate()
    p50 = ev.eval["precision"][0, :, :, 0, 2]
    p50 = p50[p50 > -1]
    det = float(np.mean(p50))

    cg2 = COCO("/tmp/cg.json")
    cd2 = cg2.loadRes(preds_list)
    ev2 = COCOeval(cg2, cd2, "bbox")
    ev2.params.maxDets = [1, 10, 100]
    ev2.evaluate()
    ev2.accumulate()
    p50c = ev2.eval["precision"][0, :, :, 0, 2]
    p50c = p50c[p50c > -1]
    cls = float(np.mean(p50c))
    return det, cls, 0.7 * det + 0.3 * cls


# Baseline
d, c, b = eval_preds(preds)
print(f"Baseline (all {len(preds)} preds): det={d:.4f} cls={c:.4f} blend={b:.4f}")

# Top-K per image
by_img = defaultdict(list)
for p in preds:
    by_img[p["image_id"]].append(p)

for k in [80, 90, 100, 110, 120]:
    topk = []
    for img_id in by_img:
        sorted_preds = sorted(by_img[img_id], key=lambda x: x["score"], reverse=True)
        topk.extend(sorted_preds[:k])
    d, c, b = eval_preds(topk)
    print(f"Top-{k:3d}/img: det={d:.4f} cls={c:.4f} blend={b:.4f}")

print()
print("--- Score fusion sweep (need raw det/cls scores) ---")
print("Note: current preds only have fused score = det*cls")
print("Need to re-run inference to test det*(a+(1-a)*cls)")
