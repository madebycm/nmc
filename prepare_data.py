"""Convert COCO annotations to YOLO format and create train/val split."""
import json
import random
from pathlib import Path
from collections import Counter

random.seed(42)

DATA_ROOT = Path(__file__).parent / "data" / "coco_dataset" / "train"
YOLO_ROOT = Path(__file__).parent / "data" / "yolo"

def main():
    with open(DATA_ROOT / "annotations.json") as f:
        coco = json.load(f)

    images = {im["id"]: im for im in coco["images"]}
    categories = {c["id"]: c["name"] for c in coco["categories"]}

    # Group annotations by image
    anns_by_img = {}
    for ann in coco["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    # 80/20 split by image (stratified isn't needed - all images are dense)
    img_ids = sorted(images.keys())
    random.shuffle(img_ids)
    split_idx = int(len(img_ids) * 0.8)
    train_ids = set(img_ids[:split_idx])
    val_ids = set(img_ids[split_idx:])

    print(f"Train images: {len(train_ids)}, Val images: {len(val_ids)}")

    # Create YOLO directory structure
    for split in ["train", "val"]:
        (YOLO_ROOT / split / "images").mkdir(parents=True, exist_ok=True)
        (YOLO_ROOT / split / "labels").mkdir(parents=True, exist_ok=True)

    # Convert and symlink
    stats = {"train": Counter(), "val": Counter()}
    for img_id, img_info in images.items():
        split = "train" if img_id in train_ids else "val"
        w, h = img_info["width"], img_info["height"]
        fname = img_info["file_name"]
        stem = Path(fname).stem

        # Symlink image
        src = DATA_ROOT / "images" / fname
        dst = YOLO_ROOT / split / "images" / fname
        if not dst.exists():
            dst.symlink_to(src.resolve())

        # Write YOLO label file
        # Single-class: all category_id -> 0
        # Multi-class: use original category_id
        lines_single = []
        lines_multi = []
        for ann in anns_by_img.get(img_id, []):
            bx, by, bw, bh = ann["bbox"]
            # COCO [x,y,w,h] -> YOLO [cx, cy, w, h] normalized
            cx = (bx + bw / 2) / w
            cy = (by + bh / 2) / h
            nw = bw / w
            nh = bh / h
            # Clamp to [0, 1]
            cx = max(0, min(1, cx))
            cy = max(0, min(1, cy))
            nw = max(0, min(1, nw))
            nh = max(0, min(1, nh))
            lines_single.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            lines_multi.append(f"{ann['category_id']} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            stats[split][ann["category_id"]] += 1

        # Write single-class labels
        label_dir_s = YOLO_ROOT / split / "labels"
        with open(label_dir_s / f"{stem}.txt", "w") as f:
            f.write("\n".join(lines_single))

        # Write multi-class labels (separate dir)
        label_dir_m = YOLO_ROOT / f"{split}_multi" / "labels"
        label_dir_m.mkdir(parents=True, exist_ok=True)
        with open(label_dir_m / f"{stem}.txt", "w") as f:
            f.write("\n".join(lines_multi))

    # Create data.yaml for single-class detection
    single_yaml = YOLO_ROOT / "data_single.yaml"
    single_yaml.write_text(f"""path: {YOLO_ROOT.resolve()}
train: train/images
val: val/images

nc: 1
names: ['product']
""")

    # Create data.yaml for multi-class detection
    multi_yaml = YOLO_ROOT / "data_multi.yaml"
    names_list = [categories.get(i, f"class_{i}") for i in range(356)]
    names_str = json.dumps(names_list, ensure_ascii=False)
    multi_yaml.write_text(f"""path: {YOLO_ROOT.resolve()}
train: train/images
val: val/images

nc: 356
names: {names_str}
""")

    # But multi-class needs its own label dirs - symlink images, swap labels
    for split in ["train", "val"]:
        multi_img_dir = YOLO_ROOT / f"{split}_multi" / "images"
        multi_img_dir.mkdir(parents=True, exist_ok=True)
        for img in (YOLO_ROOT / split / "images").iterdir():
            dst = multi_img_dir / img.name
            if not dst.exists():
                dst.symlink_to(img.resolve())

    multi_yaml.write_text(f"""path: {YOLO_ROOT.resolve()}
train: train_multi/images
val: val_multi/images

nc: 356
names: {names_str}
""")

    print(f"Train annotations: {sum(stats['train'].values())}")
    print(f"Val annotations: {sum(stats['val'].values())}")
    print(f"Single-class yaml: {single_yaml}")
    print(f"Multi-class yaml: {multi_yaml}")
    print(f"Categories in val: {len(stats['val'])}")

    # Save split info for later use
    split_info = {
        "train_ids": sorted(train_ids),
        "val_ids": sorted(val_ids),
        "categories": categories,
    }
    with open(YOLO_ROOT / "split_info.json", "w") as f:
        json.dump(split_info, f)
    print("Done.")


if __name__ == "__main__":
    main()
