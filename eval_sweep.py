"""Evaluate sweep results with different score fusions."""
import json
import numpy as np
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

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


def eval_preds(preds_list):
    if not preds_list:
        return 0, 0, 0
    det_preds = [{"image_id": p["image_id"], "category_id": 1, "bbox": p["bbox"], "score": p["score"]}
                 for p in preds_list]
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


import sys
for pred_file in sys.argv[1:]:
    print(f"\n=== {Path(pred_file).name} ===")
    with open(pred_file) as f:
        raw = json.load(f)
    raw = [p for p in raw if p["image_id"] in val_ids]
    print(f"  {len(raw)} predictions")

    # Test different fusion: score = det * (a + (1-a)*cls)
    for a in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        preds = []
        for p in raw:
            ds = p["det_score"]
            cs = p["cls_score"]
            fused = ds * (a + (1 - a) * cs)
            preds.append({
                "image_id": p["image_id"],
                "category_id": p["category_id"],
                "bbox": p["bbox"],
                "score": round(fused, 4),
            })
        d, c, b = eval_preds(preds)
        label = "det*cls" if a == 0.0 else f"det*({a:.1f}+{1-a:.1f}*cls)" if a < 1.0 else "det_only"
        print(f"  a={a:.1f} [{label:25s}]: det={d:.4f} cls={c:.4f} blend={b:.4f}")
