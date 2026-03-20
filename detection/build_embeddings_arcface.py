"""
Extract reference embeddings from ArcFace-trained classifier.
Uses CLS token features (same as kNN branch in run.py).

Usage:
  python build_embeddings_arcface.py --data /root/ng/data --weights /root/ng/output/classifier_arcface.safetensors --output /root/ng/output
"""
import argparse
import json
import numpy as np
from pathlib import Path

import torch
import timm
from safetensors.torch import load_file
from PIL import Image

NUM_CLASSES = 356
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def prepare_image(img_path):
    """Resize(256, bicubic) -> CenterCrop(224) -> normalize."""
    img = Image.open(str(img_path)).convert('RGB')
    w, h = img.size
    scale = 256 / min(w, h)
    nw, nh = round(w * scale), round(h * scale)
    img = img.resize((nw, nh), Image.BICUBIC)
    left = (nw - 224) // 2
    top = (nh - 224) // 2
    img = img.crop((left, top, left + 224, top + 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    mean = torch.tensor(CLIP_MEAN).view(3, 1, 1)
    std = torch.tensor(CLIP_STD).view(3, 1, 1)
    return (tensor - mean) / std


def prepare_crop(img, bbox):
    """Crop from image, then standard preprocess."""
    x, y, w, h = bbox
    crop = img.crop((int(x), int(y), int(x + w), int(y + h)))
    cw, ch = crop.size
    scale = 256 / min(cw, ch)
    nw, nh = round(cw * scale), round(ch * scale)
    crop = crop.resize((nw, nh), Image.BICUBIC)
    left = (nw - 224) // 2
    top = (nh - 224) // 2
    crop = crop.crop((left, top, left + 224, top + 224))
    arr = np.array(crop, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    mean = torch.tensor(CLIP_MEAN).view(3, 1, 1)
    std = torch.tensor(CLIP_STD).view(3, 1, 1)
    return (tensor - mean) / std


@torch.no_grad()
def extract_embeddings(model, tensors, device, batch_size=64):
    """Extract CLS token embeddings."""
    all_emb = []
    for i in range(0, len(tensors), batch_size):
        batch = torch.stack(tensors[i:i+batch_size]).half().to(device)
        features = model.forward_features(batch)
        cls_feat = features[:, 0].float()
        cls_feat = cls_feat / (cls_feat.norm(dim=1, keepdim=True) + 1e-8)
        all_emb.append(cls_feat.cpu())
    return torch.cat(all_emb, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = Path(args.data)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = timm.create_model('eva02_base_patch16_clip_224', pretrained=False, num_classes=NUM_CLASSES)
    sd = load_file(args.weights)
    sd = {k: v.float() for k, v in sd.items()}
    model.load_state_dict(sd)
    model.train(False)
    model = model.half().to(device)
    print(f"Loaded model from {args.weights}")

    # 1. Training crop embeddings
    ann_path = data_dir / 'train' / 'annotations.json'
    img_dir = data_dir / 'train' / 'images'
    with open(ann_path) as f:
        coco = json.load(f)

    id_to_file = {img['id']: img['file_name'] for img in coco['images']}
    crop_tensors = []
    crop_labels = []
    print("Extracting training crop embeddings...")
    for ann in coco['annotations']:
        bbox = ann['bbox']
        if bbox[2] < 2 or bbox[3] < 2:
            continue
        img = Image.open(str(img_dir / id_to_file[ann['image_id']])).convert('RGB')
        t = prepare_crop(img, bbox)
        crop_tensors.append(t)
        crop_labels.append(ann['category_id'])

    crop_emb = extract_embeddings(model, crop_tensors, device)
    print(f"  Training crops: {crop_emb.shape[0]}")

    # 2. Product image embeddings
    ean_map_path = data_dir / 'ean_to_catid.json'
    prod_tensors = []
    prod_labels = []
    if ean_map_path.exists():
        with open(ean_map_path) as f:
            ean_to_catid = json.load(f)
        print("Extracting product image embeddings...")
        for ean_dir in sorted(data_dir.iterdir()):
            if not ean_dir.is_dir():
                continue
            cat_id = ean_to_catid.get(ean_dir.name)
            if cat_id is None:
                continue
            for img_path in ean_dir.iterdir():
                if img_path.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                    t = prepare_image(img_path)
                    prod_tensors.append(t)
                    prod_labels.append(cat_id)

    if prod_tensors:
        prod_emb = extract_embeddings(model, prod_tensors, device)
        print(f"  Product images: {prod_emb.shape[0]}")
        all_emb = torch.cat([crop_emb, prod_emb], dim=0)
        all_labels = crop_labels + prod_labels
    else:
        all_emb = crop_emb
        all_labels = crop_labels

    # Save
    emb_np = all_emb.half().numpy()
    np.save(str(output_dir / 'ref_embeddings_finetuned.npy'), emb_np)
    with open(str(output_dir / 'ref_labels.json'), 'w') as f:
        json.dump(all_labels, f)

    print(f"\nSaved: {emb_np.shape} embeddings ({emb_np.nbytes/1e6:.1f} MB)")
    print(f"Labels: {len(all_labels)} entries, {len(set(all_labels))} unique classes")


if __name__ == '__main__':
    main()
