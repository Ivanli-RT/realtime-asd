# asd/yolo_wrapper.py
import os
import math
import time
from typing import List

import numpy as np
import torch
import types
from config import asd_config as C
from .tracking import FaceDet

torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# ========== Jetson Orin 兼容修补 ==========
if not hasattr(torch, "distributed"):
    torch.distributed = types.SimpleNamespace()
if not hasattr(torch.distributed, "is_initialized"):
    torch.distributed.is_initialized = lambda: False


def _assert_torchvision_nms_available():
    try:
        import torchvision
        from torchvision.ops import nms

        test_boxes = torch.empty((0, 4), dtype=torch.float32)
        test_scores = torch.empty((0,), dtype=torch.float32)
        nms(test_boxes, test_scores, 0.5)
        print(f"[YOLOWrapper] torchvision={torchvision.__version__}, nms=ok")
    except Exception as e:
        raise RuntimeError(
            "torchvision.ops.nms is unavailable. Install the Jetson-matched "
            "torchvision wheel built from pytorch/vision v0.20.0 for the current "
            "NVIDIA torch build, then retry."
        ) from e


_assert_torchvision_nms_available()

# ========== 导入ultralytics ==========
try:
    from ultralytics import YOLO
except ImportError as e:
    raise ImportError(f"ultralytics not installed. Please run: pip install ultralytics") from e


def _require_inference_device(requested_device: str) -> str:
    requested_device = (requested_device or "cuda").strip().lower()
    allow_cpu = bool(getattr(C, "ALLOW_CPU_INFERENCE", False))

    if requested_device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "[YOLOFaceDetector] CUDA inference is required, but torch.cuda.is_available() is False. "
                "Fix the CUDA/PyTorch runtime instead of falling back to CPU."
            )
        return requested_device

    if requested_device == "cpu" and allow_cpu:
        return "cpu"

    raise RuntimeError(
        f"[YOLOFaceDetector] Refusing to run inference on '{requested_device}'. "
        "Set ASD_MODEL_DEVICE=cuda, or explicitly set ASD_ALLOW_CPU_INFERENCE=1 for debugging only."
    )


def _torch_module_device(model) -> str:
    module = getattr(model, "model", None)
    if module is None:
        return "<unknown>"
    try:
        return str(next(module.parameters()).device)
    except StopIteration:
        return "<no-parameters>"
    except Exception as exc:
        return f"<unknown: {exc}>"


class YOLOFaceDetector:
    """
    YOLOv10s-face人脸检测器包装类
    针对Jetson Orin优化
    """
    def __init__(
        self,
        model_path: str = None,
        use_trt: bool = None,
        engine_path: str = None,
        conf_th: float = None,
        imgsz: int = None,
        iou_thresh: float = None
    ):
        # 参数优先级: 构造函数参数 > 配置文件
        requested_model_path = model_path or C.YOLO_MODEL_PATH
        self.trt_engine_path = engine_path or getattr(C, "YOLO_TRT_ENGINE_PATH", getattr(C, "YOLO_TRT_ENGINE", ""))
        if use_trt is None:
            self.use_trt = bool(getattr(C, "YOLO_USE_TRT", False)) or str(requested_model_path).endswith(".engine")
        else:
            self.use_trt = bool(use_trt)
        if self.use_trt:
            if str(requested_model_path).endswith(".engine") and engine_path is None:
                self.trt_engine_path = requested_model_path
            if not self.trt_engine_path:
                raise ValueError("[YOLOFaceDetector] YOLO_TRT_ENGINE_PATH is empty but YOLO_USE_TRT=True")
            self.model_path = self.trt_engine_path
        else:
            self.model_path = requested_model_path
        self.conf_th = conf_th if conf_th is not None else C.FACE_DETECTION_CONF_THRESH
        self.imgsz = imgsz or C.YOLO_IMGSZ
        self.iou_thresh = iou_thresh if iou_thresh is not None else C.YOLO_IOU_THRESH
        
        # 设备选择：默认强制 CUDA，禁止静默退回 CPU。
        self.device = _require_inference_device(getattr(C, "MODEL_DEVICE", "cuda"))
        if self.use_trt and not self.device.startswith("cuda"):
            raise RuntimeError("[YOLOFaceDetector] TensorRT requires CUDA device")
        self.backend = "tensorrt" if self.use_trt else "torch"
        self.trt_dynamic = bool(getattr(C, "YOLO_TRT_DYNAMIC", False)) if self.use_trt else False
        self.trt_min_size = self._normalize_hw(getattr(C, "YOLO_TRT_MIN_SIZE", (480, 640)), (480, 640))
        self.trt_max_size = self._normalize_hw(getattr(C, "YOLO_TRT_MAX_SIZE", (720, 1280)), (720, 1280))
        self.trt_stride = int(getattr(C, "YOLO_TRT_STRIDE", 32))
        self.profile_enable = bool(getattr(C, "YOLO_PROFILE_ENABLE", False))
        self.profile_interval_sec = max(0.5, float(getattr(C, "YOLO_PROFILE_INTERVAL_SEC", 3.0)))
        self._last_profile_log_ts = 0.0
        
        # 检查模型文件
        if not os.path.isfile(self.model_path):
            kind = "TensorRT engine" if self.use_trt else "YOLO model"
            if self.use_trt:
                hint = "Set YOLO_TRT_ENGINE_PATH to a valid .engine file"
            else:
                hint = f"Please download yolov11n-face.pt to {C.MODEL_DIR}"
            raise FileNotFoundError(f"{kind} not found: {self.model_path}\n{hint}")
        
        print(
            f"[YOLOFaceDetector] Loading model ({self.backend}): "
            f"{self.model_path} (device={self.device})"
        )
        if self.device.startswith("cuda"):
            print(
                f"[YOLOFaceDetector] CUDA available: device_count={torch.cuda.device_count()} "
                f"current={torch.cuda.current_device()} name={torch.cuda.get_device_name(torch.cuda.current_device())}",
                flush=True,
            )
        
        # 加载YOLO模型
        self.model = YOLO(self.model_path)
        if not self.use_trt and hasattr(self.model, "to"):
            self.model.to(self.device)
            actual_device = _torch_module_device(self.model)
            if not actual_device.startswith(self.device):
                raise RuntimeError(
                    f"[YOLOFaceDetector] PyTorch YOLO model is on {actual_device}, expected {self.device}"
                )
            print(f"[YOLOFaceDetector] PyTorch model verified on {actual_device}", flush=True)
        
        # Jetson optimization: disable unnecessary features
        if not self.use_trt and hasattr(self.model, "fuse"):
            try:
                self.model.fuse()
            except Exception as e:
                print(f"[YOLOFaceDetector] Warning: model.fuse() failed: {e}")
        
        # 预热模型(触发warmup,确保NMS patch生效)
        print("[YOLOFaceDetector] Warming up model...")
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        warmup_imgsz = self._get_infer_size(dummy_img)
        try:
            _ = self.model.predict(
                source=dummy_img,
                conf=0.25,
                iou=0.45,
                imgsz=warmup_imgsz,
                device=self.device,
                verbose=False,
                half=(
                    self.device.startswith("cuda")
                    and not self.use_trt
                    and not bool(getattr(C, "YOLO_DISABLE_FP16", False))
                ),
                augment=False,
            )
            print("[YOLOFaceDetector] Warmup completed successfully")
        except Exception as e:
            print(f"[YOLOFaceDetector] Warning: Warmup failed: {e}")
        
        print(
            f"[YOLOFaceDetector] Init OK - conf={self.conf_th:.2f}, "
            f"imgsz={self.imgsz}, iou={self.iou_thresh:.2f}"
        )

    @staticmethod
    def _normalize_hw(value, fallback):
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return (int(value[0]), int(value[1]))
        if isinstance(value, (int, float)):
            v = int(value)
            return (v, v)
        return fallback

    def _get_infer_size(self, frame_bgr):
        if not (self.use_trt and self.trt_dynamic):
            return self.imgsz
        h, w = frame_bgr.shape[:2]
        min_h, min_w = self.trt_min_size
        max_h, max_w = self.trt_max_size
        tgt_h = min(max(int(h), int(min_h)), int(max_h))
        tgt_w = min(max(int(w), int(min_w)), int(max_w))
        if self.trt_stride and self.trt_stride > 1:
            stride = int(self.trt_stride)
            down_h = (tgt_h // stride) * stride
            down_w = (tgt_w // stride) * stride
            if down_h < int(min_h):
                up_h = int(math.ceil(tgt_h / stride) * stride)
                tgt_h = up_h if up_h <= int(max_h) else int(max_h)
            else:
                tgt_h = down_h
            if down_w < int(min_w):
                up_w = int(math.ceil(tgt_w / stride) * stride)
                tgt_w = up_w if up_w <= int(max_w) else int(max_w)
            else:
                tgt_w = down_w
        return (tgt_h, tgt_w)

    def detect(self, frame_bgr: np.ndarray) -> List[FaceDet]:
        """
        检测人脸
        
        Args:
            frame_bgr: BGR格式图像 (H, W, 3)
            
        Returns:
            List[FaceDet]: 检测到的人脸列表
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return []
        if bool(getattr(C, "YOLO_FORCE_CONTIGUOUS_INPUT", True)):
            frame_bgr = np.ascontiguousarray(frame_bgr)
        
        # YOLO推理
        try:
            imgsz = self._get_infer_size(frame_bgr)
            use_half = (
                self.device.startswith("cuda")
                and not self.use_trt
                and not bool(getattr(C, "YOLO_DISABLE_FP16", False))
            )
            t0 = time.time()
            with torch.inference_mode():
                results_list = self.model.predict(
                    source=frame_bgr,
                    conf=self.conf_th,
                    iou=self.iou_thresh,
                    imgsz=imgsz,
                    device=self.device,
                    verbose=False,
                    stream=False,
                    # Jetson优化参数
                    half=use_half,
                    augment=not bool(getattr(C, "YOLO_DISABLE_TTA", True)),
                )
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
            predict_ms = (time.time() - t0) * 1000.0
        except Exception as e:
            print(f"[YOLOFaceDetector] Inference error: {e}")
            return []
        
        if not results_list or results_list[0].boxes is None:
            return []
        
        results = results_list[0]
        if self.profile_enable:
            now = time.time()
            if now - self._last_profile_log_ts >= self.profile_interval_sec:
                self._last_profile_log_ts = now
                speed = getattr(results, "speed", {}) or {}
                print(
                    "[YOLOFaceDetector][PROFILE] "
                    f"backend={self.backend} model={os.path.basename(self.model_path)} "
                    f"frame={frame_bgr.shape[1]}x{frame_bgr.shape[0]} imgsz={imgsz} "
                    f"device={self.device} half={use_half} predict_ms={predict_ms:.1f} "
                    f"ultra_pre={float(speed.get('preprocess', 0.0)):.1f} "
                    f"ultra_inf={float(speed.get('inference', 0.0)):.1f} "
                    f"ultra_post={float(speed.get('postprocess', 0.0)):.1f} "
                    f"boxes={len(results.boxes) if results.boxes is not None else 0}",
                    flush=True,
                )
        
        # 提取检测框和置信度
        if len(results.boxes) > 0:
            boxes_xyxy = results.boxes.xyxy.cpu().numpy()  # (N, 4)
            scores = results.boxes.conf.cpu().numpy()      # (N,)
        else:
            return []
        
        # 转换为FaceDet格式,过滤小框
        dets: List[FaceDet] = []
        for bbox, score in zip(boxes_xyxy, scores):
            x1, y1, x2, y2 = bbox
            w_box = x2 - x1
            h_box = y2 - y1
            
            # 过滤过小的人脸
            if w_box < C.MIN_FACE_SIZE or h_box < C.MIN_FACE_SIZE:
                continue
            
            dets.append(
                FaceDet(
                    x1=int(x1),
                    y1=int(y1),
                    x2=int(x2),
                    y2=int(y2),
                    score=float(score),
                )
            )
        
        return dets
