#!/usr/bin/env python3
"""Benchmark YOLO PyTorch forward vs Ultralytics predict on the runtime device."""

import argparse
import os
import statistics
import time

import numpy as np
import torch
from ultralytics import YOLO


def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def _percentile(values, pct):
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return ordered[idx]


def _summary(name, values):
    print(
        f"{name}: n={len(values)} avg={statistics.mean(values):.1f}ms "
        f"p50={statistics.median(values):.1f}ms p90={_percentile(values, 90):.1f}ms "
        f"min={min(values):.1f}ms max={max(values):.1f}ms",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/workspace/asd_runtime_slim/models/yolov11n-face.pt")
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iters", type=int, default=30)
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    print(f"torch={torch.__version__} cuda={torch.version.cuda} cudnn={torch.backends.cudnn.version()}")
    if args.device.startswith("cuda"):
        print(
            f"cuda devices={torch.cuda.device_count()} current={torch.cuda.current_device()} "
            f"name={torch.cuda.get_device_name(torch.cuda.current_device())}"
        )
    print(f"model={args.model} imgsz={args.imgsz} source={args.width}x{args.height} half={args.half}")

    model = YOLO(args.model)
    model.to(args.device)
    if hasattr(model, "fuse"):
        model.fuse()

    dtype = torch.float16 if args.half else torch.float32
    module = model.model.to(args.device).eval()
    if args.half:
        module.half()

    tensor = torch.zeros((1, 3, args.imgsz, args.imgsz), device=args.device, dtype=dtype)
    with torch.inference_mode():
        for _ in range(args.warmup):
            _ = module(tensor)
        _sync(args.device)

        forward_ms = []
        for _ in range(args.iters):
            t0 = time.time()
            _ = module(tensor)
            _sync(args.device)
            forward_ms.append((time.time() - t0) * 1000.0)

    _summary("raw_module_forward", forward_ms)

    image = np.zeros((args.height, args.width, 3), dtype=np.uint8)
    predict_ms = []
    ultra_pre = []
    ultra_inf = []
    ultra_post = []
    with torch.inference_mode():
        for _ in range(args.warmup):
            _ = model.predict(
                source=image,
                imgsz=args.imgsz,
                device=args.device,
                half=args.half,
                verbose=False,
                stream=False,
                augment=False,
            )
        _sync(args.device)

        for _ in range(args.iters):
            t0 = time.time()
            results = model.predict(
                source=image,
                imgsz=args.imgsz,
                device=args.device,
                half=args.half,
                verbose=False,
                stream=False,
                augment=False,
            )
            _sync(args.device)
            predict_ms.append((time.time() - t0) * 1000.0)
            speed = getattr(results[0], "speed", {}) or {}
            ultra_pre.append(float(speed.get("preprocess", 0.0)))
            ultra_inf.append(float(speed.get("inference", 0.0)))
            ultra_post.append(float(speed.get("postprocess", 0.0)))

    _summary("ultralytics_predict_wall", predict_ms)
    _summary("ultralytics_preprocess", ultra_pre)
    _summary("ultralytics_inference", ultra_inf)
    _summary("ultralytics_postprocess", ultra_post)


if __name__ == "__main__":
    main()
