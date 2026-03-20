"""Quick YOLOv8m training for fast first submission."""
import torch
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
        epochs=50,
        imgsz=640,
        batch=16,
        device="cpu",
        patience=10,
        augment=True,
        mosaic=1.0,
        project="runs",
        name="detect_fast",
        exist_ok=True,
        verbose=True,
        workers=4,
    )
    print(f"\nBest mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")

    best_path = "runs/detect_fast/weights/best.pt"
    best_model = YOLO(best_path)
    best_model.export(format="onnx", imgsz=1280, opset=17, half=False, simplify=True)
    print("Exported to ONNX")

if __name__ == "__main__":
    main()
