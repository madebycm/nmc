"""Train YOLOv8m single-class detector with compatibility patches."""
import numpy as np
import torch

# Patch np.trapz for numpy 2.0+ (ultralytics 8.1.0 uses deprecated np.trapz)
if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid

# Patch torch.load for ultralytics compatibility with torch 2.10+
_orig_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _orig_load(*args, **kwargs)
torch.load = _patched_load

from ultralytics import YOLO

def main():
    model = YOLO("yolov8m.pt")
    results = model.train(
        data="data/yolo/data_single.yaml",
        epochs=40,
        imgsz=640,
        batch=16,
        device="cpu",
        patience=10,
        augment=True,
        mosaic=1.0,
        mixup=0.1,
        project="runs",
        name="detect_single",
        exist_ok=True,
        verbose=True,
    )
    print(f"\nTraining complete. Best mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")

    # Export to ONNX
    best_path = "runs/detect_single/weights/best.pt"
    best_model = YOLO(best_path)
    # Export at 1280 for sandbox inference (L4 GPU can handle it)
    best_model.export(format="onnx", imgsz=1280, opset=17, half=False, simplify=True)
    print("Exported to ONNX")

if __name__ == "__main__":
    main()
