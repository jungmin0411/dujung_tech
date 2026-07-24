#!/usr/bin/env python3
"""Run the exported YOLOv10x good/bad weld detector on images or video."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = PACKAGE_ROOT / "model/weld_yolov10x_box010_e12.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="Image, directory, video, webcam index, or stream URL")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output", type=Path, default=PACKAGE_ROOT / "runs")
    parser.add_argument("--name", default="prediction")
    parser.add_argument("--conf", type=float, default=0.60)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None, help="Example: 0, cpu; default is automatic")
    parser.add_argument("--save-txt", action="store_true")
    parser.add_argument("--save-conf", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = args.weights.expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(weights)
    device = args.device or ("0" if torch.cuda.is_available() else "cpu")
    model = YOLO(str(weights))
    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=device,
        save=True,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        project=str(args.output.expanduser().resolve()),
        name=args.name,
        exist_ok=True,
        verbose=True,
    )
    save_dir = Path(results[0].save_dir).resolve() if results else args.output.resolve()
    print(f"Saved predictions: {save_dir}")


if __name__ == "__main__":
    main()
