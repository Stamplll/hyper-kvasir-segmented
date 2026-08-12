import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO11 segmentation inference for polyp detection."
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to an image, video, folder, or stream source.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/polyp_yolo11n_seg_best.pt",
        help="Path to trained model weights (.pt).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.79,
        help="Confidence threshold for predictions.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save prediction visualizations to runs/predict.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display prediction window during inference.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    source_path = Path(args.source)
    if not source_path.exists() and not args.source.startswith(("rtsp://", "http://", "https://")):
        raise FileNotFoundError(f"Source not found: {source_path}")

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    model = YOLO(str(weights_path))

    results = model.predict(
        source=str(source_path) if source_path.exists() else args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        save=args.save,
        show=args.show,
        project="runs/predict",
        name="polyp_inference",
        exist_ok=True,
        verbose=True,
    )

    print(f"Done. Total predictions: {len(results)}")


if __name__ == "__main__":
    main()
