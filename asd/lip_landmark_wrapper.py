# asd/lip_landmark_wrapper.py
# -*- coding: utf-8 -*-
"""
唇部关键点检测器封装

复用 lip_kp_det 项目的 ResNet50 关键点检测模型，
适配 fast-asd-main 项目结构。

特点:
- 支持批量推理 (多个人脸 ROI 一次前向)
- 只提取唇部相关关键点 (节省后处理开销)
- 兼容 TensorRT 加速 (未来扩展)
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

try:
    from config import asd_config as C
except Exception:
    C = None


def _cfg(name: str, default):
    return getattr(C, name, default) if C is not None else default


def _require_inference_device(requested_device: str, component: str) -> str:
    requested_device = (requested_device or "cuda").strip().lower()
    allow_cpu = bool(_cfg("ALLOW_CPU_INFERENCE", False))
    if requested_device == "auto":
        requested_device = str(_cfg("MODEL_DEVICE", "cuda")).strip().lower() or "cuda"

    if requested_device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"[{component}] CUDA inference is required, but torch.cuda.is_available() is False. "
                "Fix the CUDA/PyTorch runtime instead of falling back to CPU."
            )
        return requested_device

    if requested_device == "cpu" and allow_cpu:
        return "cpu"

    raise RuntimeError(
        f"[{component}] Refusing to run inference on '{requested_device}'. "
        "Set ASD_MODEL_DEVICE=cuda, or explicitly set ASD_ALLOW_CPU_INFERENCE=1 for debugging only."
    )


# ============== 模型定义 (复用 lip_kp_det) ==============

def _conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class _Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out


class ResNet50Landmark(nn.Module):
    """ResNet50 关键点回归网络"""

    def __init__(self, num_classes: int = 196, img_size: int = 256, dropout_factor: float = 0.5):
        super().__init__()
        self.inplanes = 64
        block = _Bottleneck
        layers = [3, 4, 6, 3]

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        pool_kernel = int(img_size / 32)
        self.avgpool = nn.AvgPool2d(pool_kernel, stride=1, ceil_mode=True)
        self.dropout = nn.Dropout(p=float(dropout_factor))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride=1, downsample=None))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


# ============== Letterbox 变换 ==============

@dataclass(frozen=True)
class LetterboxTransform:
    """Letterbox 变换参数"""
    in_w: int
    in_h: int
    out_w: int
    out_h: int
    ratio: float
    pad_x: int
    pad_y: int


def letterbox(
    img: np.ndarray,
    new_shape: Tuple[int, int] = (256, 256),
    color: Tuple[int, int, int] = (128, 128, 128),
) -> Tuple[np.ndarray, LetterboxTransform]:
    """等比例缩放并填充"""
    out_w, out_h = int(new_shape[0]), int(new_shape[1])
    in_h, in_w = img.shape[:2]

    ratio = min(out_w / float(in_w), out_h / float(in_h))
    new_unpad_w = int(round(in_w * ratio))
    new_unpad_h = int(round(in_h * ratio))

    resized = cv2.resize(img, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)

    pad_x = (out_w - new_unpad_w) // 2
    pad_y = (out_h - new_unpad_h) // 2

    canvas = np.full((out_h, out_w, 3), color, dtype=img.dtype)
    canvas[pad_y: pad_y + new_unpad_h, pad_x: pad_x + new_unpad_w] = resized

    tfm = LetterboxTransform(
        in_w=in_w, in_h=in_h, out_w=out_w, out_h=out_h,
        ratio=float(ratio), pad_x=int(pad_x), pad_y=int(pad_y),
    )
    return canvas, tfm


def preprocess_landmark_input(img_256_bgr: np.ndarray) -> torch.Tensor:
    """预处理: (img - 128) / 256, HWC → CHW"""
    x = img_256_bgr.astype(np.float32)
    x = (x - 128.0) / 256.0
    x = np.transpose(x, (2, 0, 1))
    return torch.from_numpy(x)


def project_points_back(
    pts_norm: np.ndarray,
    tfm: LetterboxTransform,
    bbox_xyxy: Tuple[int, int, int, int],
) -> np.ndarray:
    """将关键点从 256×256 画布投影回原图坐标"""
    x1, y1, _, _ = bbox_xyxy
    pts = pts_norm.astype(np.float32).copy()

    # 判断输出是归一化 [0,1] 还是像素坐标
    finite = np.isfinite(pts).all(axis=1)
    if finite.any():
        pts_f = pts[finite]
        mx = float(np.max(pts_f))
        mn = float(np.min(pts_f))
        looks_normalized = (mn >= -1.0) and (mx <= 2.0)
        if looks_normalized:
            pts[:, 0] *= float(tfm.out_w)
            pts[:, 1] *= float(tfm.out_h)

    # 撤销 padding 和缩放
    pts[:, 0] = (pts[:, 0] - float(tfm.pad_x)) / float(tfm.ratio)
    pts[:, 1] = (pts[:, 1] - float(tfm.pad_y)) / float(tfm.ratio)

    # 加上 bbox 偏移
    pts[:, 0] += float(x1)
    pts[:, 1] += float(y1)
    return pts


# ============== 唇部关键点检测器 ==============

@dataclass
class LipLandmarkResult:
    """单个人脸的关键点检测结果"""
    track_id: int
    bbox_xyxy: Tuple[int, int, int, int]
    landmarks_98: np.ndarray  # (98, 2) 全部关键点
    lip_landmarks: np.ndarray  # (20, 2) 唇部关键点 (77-96)
    is_valid: bool


class LipLandmarkDetector:
    """
    唇部关键点检测器
    
    基于 ResNet50 的 98 点人脸关键点检测，
    重点提取唇部 20 个关键点用于运动分析。
    
    支持 TensorRT 加速推理。
    """

    # 唇部关键点索引 (0-based, 对应 77-96)
    LIP_INDICES = list(range(76, 96))

    def __init__(
        self,
        weights_path: str,
        device: Optional[str] = None,
        input_size: int = 256,
        enable_debug: bool = False,
        use_trt: bool = False,
        trt_engine_path: Optional[str] = None,
    ):
        """
        Args:
            weights_path: ResNet50 关键点模型权重路径
            device: 推理设备 ("cuda" / "cpu" / None=auto)
            input_size: 输入尺寸 (默认 256)
            enable_debug: 是否打印调试信息
            use_trt: 是否使用 TensorRT 加速
            trt_engine_path: TensorRT 引擎路径 (如果 use_trt=True)
        """
        self.input_size = input_size
        self.enable_debug = enable_debug
        self.use_trt = use_trt
        self.trt_engine_path = trt_engine_path
        self.last_timing = {}

        # 选择设备：默认强制 CUDA，禁止静默退回 CPU。
        if device is None:
            device = str(_cfg("MODEL_DEVICE", "cuda"))
        self.device = _require_inference_device(device, "LipLandmark")
        
        # TensorRT 需要 CUDA
        if self.use_trt and not self.device.startswith("cuda"):
            raise RuntimeError("[LipLandmark] TensorRT requires CUDA device")
        
        if self.use_trt and trt_engine_path and os.path.isfile(trt_engine_path):
            # 使用 TensorRT 引擎
            self._init_trt_engine(trt_engine_path)
        else:
            if self.use_trt:
                raise FileNotFoundError(f"[LipLandmark] TRT engine not found: {trt_engine_path}")
            self.use_trt = False
            # 加载 PyTorch 模型
            self.model = ResNet50Landmark(num_classes=196, img_size=input_size)
            self._load_weights(weights_path)
            self.model.to(self.device).eval()
            actual_device = str(next(self.model.parameters()).device)
            if not actual_device.startswith(self.device):
                raise RuntimeError(
                    f"[LipLandmark] PyTorch model is on {actual_device}, expected {self.device}"
                )

        if self.enable_debug:
            backend = "TensorRT" if self.use_trt else "PyTorch"
            print(f"[LipLandmark] Model loaded on {self.device} ({backend})", flush=True)

        if bool(_cfg("LIP_WARMUP_ENABLE", True)):
            self._warmup()

    def _warmup(self) -> None:
        """Run one tiny inference during startup to avoid first-face UI stalls."""
        try:
            dummy = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
            _ = self.detect(dummy, [(0, 0, self.input_size, self.input_size)], track_ids=[-1])
            if self.device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
            print("[LipLandmark] Warmup completed", flush=True)
        except Exception as exc:
            print(f"[LipLandmark] Warmup skipped: {exc}", flush=True)
    
    def _init_trt_engine(self, engine_path: str) -> None:
        """初始化 TensorRT 引擎"""
        try:
            import tensorrt as trt
            
            self.trt_logger = trt.Logger(trt.Logger.WARNING)
            
            with open(engine_path, "rb") as f:
                self.trt_runtime = trt.Runtime(self.trt_logger)
                self.trt_engine = self.trt_runtime.deserialize_cuda_engine(f.read())
            
            self.trt_context = self.trt_engine.create_execution_context()
            
            # 获取输入输出绑定信息
            self.trt_input_name = self.trt_engine.get_tensor_name(0)
            self.trt_output_name = self.trt_engine.get_tensor_name(1)
            
            # 分配 GPU 缓冲区
            import pycuda.driver as cuda
            import pycuda.autoinit  # noqa
            
            # 输入缓冲区 (batch_size, 3, 256, 256) - 支持动态 batch
            self.trt_max_batch = 8  # 最大 batch size
            input_shape = (self.trt_max_batch, 3, self.input_size, self.input_size)
            output_shape = (self.trt_max_batch, 196)
            
            self.trt_input_host = np.zeros(input_shape, dtype=np.float32)
            self.trt_output_host = np.zeros(output_shape, dtype=np.float32)
            
            self.trt_input_device = cuda.mem_alloc(self.trt_input_host.nbytes)
            self.trt_output_device = cuda.mem_alloc(self.trt_output_host.nbytes)
            
            self.cuda = cuda
            self.trt = trt
            
            print(f"[LipLandmark] TensorRT engine loaded: {engine_path}", flush=True)
            
        except ImportError as e:
            raise RuntimeError(f"[LipLandmark] TensorRT import failed: {e}") from e
        except Exception as e:
            raise RuntimeError(f"[LipLandmark] TensorRT init failed: {e}") from e
    
    def _infer_trt(self, batch_tensor: np.ndarray) -> np.ndarray:
        """TensorRT 推理"""
        batch_size = batch_tensor.shape[0]
        
        # 复制输入到 host buffer
        self.trt_input_host[:batch_size] = batch_tensor
        
        # 设置动态 batch size
        self.trt_context.set_input_shape(self.trt_input_name, (batch_size, 3, self.input_size, self.input_size))
        
        # Host -> Device
        self.cuda.memcpy_htod(self.trt_input_device, self.trt_input_host[:batch_size])
        
        # 推理
        self.trt_context.set_tensor_address(self.trt_input_name, int(self.trt_input_device))
        self.trt_context.set_tensor_address(self.trt_output_name, int(self.trt_output_device))
        self.trt_context.execute_async_v3(0)
        
        # Device -> Host
        self.cuda.memcpy_dtoh(self.trt_output_host[:batch_size], self.trt_output_device)
        
        return self.trt_output_host[:batch_size].copy()

    def _load_weights(self, path: str) -> None:
        """加载权重 (兼容 DataParallel 前缀)"""
        if not os.path.isfile(path):
            raise FileNotFoundError(f"[LipLandmark] Weights not found: {path}")

        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        # 清理 module. 前缀
        if isinstance(state, dict):
            cleaned = {}
            for k, v in state.items():
                nk = k[len("module."):] if k.startswith("module.") else k
                cleaned[nk] = v
            state = cleaned

        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if self.enable_debug and (missing or unexpected):
            print(f"[LipLandmark] Missing keys: {missing}", flush=True)
            print(f"[LipLandmark] Unexpected keys: {unexpected}", flush=True)

    @torch.no_grad()
    def detect(
        self,
        image_bgr: np.ndarray,
        face_bboxes: List[Tuple[int, int, int, int]],
        track_ids: Optional[List[int]] = None,
    ) -> List[LipLandmarkResult]:
        """
        批量检测多个人脸的关键点（真正的 batch 推理）
        
        同一帧的多个人脸一次性进行 batch 推理，充分利用 GPU 并行能力。
        
        Args:
            image_bgr: 原图 (BGR)
            face_bboxes: 人脸框列表 [(x1, y1, x2, y2), ...]
            track_ids: 对应的 track ID 列表 (可选)
            
        Returns:
            LipLandmarkResult 列表
        """
        total_start = time.perf_counter()
        crop_ms = 0.0
        letterbox_ms = 0.0
        preprocess_ms = 0.0
        infer_ms = 0.0
        project_ms = 0.0
        if image_bgr is None or image_bgr.size == 0:
            self.last_timing = {
                "crop_ms": 0.0,
                "letterbox_ms": 0.0,
                "preprocess_ms": 0.0,
                "infer_ms": 0.0,
                "project_ms": 0.0,
                "batch_total_ms": (time.perf_counter() - total_start) * 1000.0,
                "num_faces": 0.0,
                "valid_faces": 0.0,
            }
            return []
        
        if len(face_bboxes) == 0:
            self.last_timing = {
                "crop_ms": 0.0,
                "letterbox_ms": 0.0,
                "preprocess_ms": 0.0,
                "infer_ms": 0.0,
                "project_ms": 0.0,
                "batch_total_ms": (time.perf_counter() - total_start) * 1000.0,
                "num_faces": 0.0,
                "valid_faces": 0.0,
            }
            return []

        h, w = image_bgr.shape[:2]
        
        if track_ids is None:
            track_ids = list(range(len(face_bboxes)))
        
        # ==========================
        # 预处理：裁剪并构建 batch
        # ==========================
        batch_tensors = []
        batch_tfms = []  # 保存每个人脸的变换参数
        batch_bboxes = []  # 保存规范化后的 bbox
        valid_indices = []  # 有效人脸的索引
        
        for idx, bbox in enumerate(face_bboxes):
            x1, y1, x2, y2 = bbox
            x1 = max(0, min(int(x1), w - 1))
            y1 = max(0, min(int(y1), h - 1))
            x2 = max(0, min(int(x2), w))
            y2 = max(0, min(int(y2), h))
            
            # 检查 bbox 有效性
            if x2 <= x1 or y2 <= y1:
                batch_bboxes.append((x1, y1, x2, y2))
                batch_tfms.append(None)
                continue
            
            # 裁剪人脸区域
            crop_start = time.perf_counter()
            crop = image_bgr[y1:y2, x1:x2]
            crop_ms += (time.perf_counter() - crop_start) * 1000.0
            if crop.size == 0:
                batch_bboxes.append((x1, y1, x2, y2))
                batch_tfms.append(None)
                continue
            
            # Letterbox 到 256×256
            letterbox_start = time.perf_counter()
            crop_256, tfm = letterbox(crop, new_shape=(self.input_size, self.input_size))
            letterbox_ms += (time.perf_counter() - letterbox_start) * 1000.0
            
            # 预处理
            preprocess_start = time.perf_counter()
            tensor = preprocess_landmark_input(crop_256)
            preprocess_ms += (time.perf_counter() - preprocess_start) * 1000.0
            batch_tensors.append(tensor)
            batch_tfms.append(tfm)
            batch_bboxes.append((x1, y1, x2, y2))
            valid_indices.append(idx)
        
        # ==========================
        # Batch 推理
        # ==========================
        results = []
        
        if len(batch_tensors) > 0:
            # 组成 batch tensor
            batch_input = torch.stack(batch_tensors, dim=0)  # (N, 3, 256, 256)
            
            infer_start = time.perf_counter()
            if self.use_trt:
                # TensorRT 推理
                batch_np = batch_input.numpy().astype(np.float32)
                batch_output = self._infer_trt(batch_np)  # (N, 196)
            else:
                # PyTorch 推理
                batch_input = batch_input.to(self.device)
                batch_output = self.model(batch_input)  # (N, 196)
                batch_output = batch_output.detach().float().cpu().numpy()  # (N, 196)
            infer_ms += (time.perf_counter() - infer_start) * 1000.0
        
        # ==========================
        # 解析结果
        # ==========================
        valid_idx_ptr = 0
        
        for idx, (bbox, tfm, tid) in enumerate(zip(batch_bboxes, batch_tfms, track_ids)):
            if tfm is None:
                # 无效人脸
                results.append(LipLandmarkResult(
                    track_id=tid,
                    bbox_xyxy=bbox,
                    landmarks_98=np.zeros((98, 2), dtype=np.float32),
                    lip_landmarks=np.zeros((20, 2), dtype=np.float32),
                    is_valid=False,
                ))
            else:
                # 有效人脸：从 batch 输出中提取对应结果
                y = batch_output[valid_idx_ptr].reshape(-1, 2)  # (98, 2)
                valid_idx_ptr += 1
                
                # 投影回原图坐标
                project_start = time.perf_counter()
                pts = project_points_back(y, tfm=tfm, bbox_xyxy=bbox)
                project_ms += (time.perf_counter() - project_start) * 1000.0
                
                # 提取唇部关键点
                lip_pts = pts[self.LIP_INDICES].astype(np.float32)
                
                results.append(LipLandmarkResult(
                    track_id=tid,
                    bbox_xyxy=bbox,
                    landmarks_98=pts.astype(np.float32),
                    lip_landmarks=lip_pts,
                    is_valid=True,
                ))
            self.last_timing = {
                "crop_ms": crop_ms,
                "letterbox_ms": letterbox_ms,
                "preprocess_ms": preprocess_ms,
                "infer_ms": infer_ms,
                "project_ms": project_ms,
                "batch_total_ms": (time.perf_counter() - total_start) * 1000.0,
                "num_faces": float(len(face_bboxes)),
                "valid_faces": float(len(batch_tensors)),
            }
        
        return results

    def _detect_single(
        self,
        image_bgr: np.ndarray,
        bbox: Tuple[int, int, int, int],
        track_id: int,
        img_w: int,
        img_h: int,
    ) -> LipLandmarkResult:
        """检测单个人脸的关键点"""
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(int(x1), img_w - 1))
        y1 = max(0, min(int(y1), img_h - 1))
        x2 = max(0, min(int(x2), img_w))
        y2 = max(0, min(int(y2), img_h))

        if x2 <= x1 or y2 <= y1:
            return LipLandmarkResult(
                track_id=track_id,
                bbox_xyxy=(x1, y1, x2, y2),
                landmarks_98=np.zeros((98, 2), dtype=np.float32),
                lip_landmarks=np.zeros((20, 2), dtype=np.float32),
                is_valid=False,
            )

        # 裁剪人脸区域
        crop = image_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return LipLandmarkResult(
                track_id=track_id,
                bbox_xyxy=(x1, y1, x2, y2),
                landmarks_98=np.zeros((98, 2), dtype=np.float32),
                lip_landmarks=np.zeros((20, 2), dtype=np.float32),
                is_valid=False,
            )

        # Letterbox 到 256×256
        crop_256, tfm = letterbox(crop, new_shape=(self.input_size, self.input_size))

        # 预处理
        x = preprocess_landmark_input(crop_256).unsqueeze(0).to(self.device)

        # 推理
        y = self.model(x)  # (1, 196)
        y = y.detach().float().cpu().numpy().reshape(-1, 2)

        # 投影回原图坐标
        pts = project_points_back(y, tfm=tfm, bbox_xyxy=(x1, y1, x2, y2))

        # 提取唇部关键点
        lip_pts = pts[self.LIP_INDICES].astype(np.float32)

        return LipLandmarkResult(
            track_id=track_id,
            bbox_xyxy=(x1, y1, x2, y2),
            landmarks_98=pts.astype(np.float32),
            lip_landmarks=lip_pts,
            is_valid=True,
        )

    @torch.no_grad()
    def detect_batch(
        self,
        face_crops_112: List[np.ndarray],
        track_ids: List[int],
    ) -> List[Optional[np.ndarray]]:
        """
        批量检测 (输入为已裁剪的 112×112 人脸)
        
        用于与现有 ASD 流程集成，复用已裁剪的人脸 ROI。
        
        Args:
            face_crops_112: 人脸 ROI 列表，每个 (112, 112, 3) BGR
            track_ids: 对应的 track ID 列表
            
        Returns:
            关键点列表，每个 (98, 2) 或 None
        """
        total_start = time.perf_counter()
        resize_ms = 0.0
        preprocess_ms = 0.0
        infer_ms = 0.0
        if not face_crops_112:
            self.last_timing = {
                "resize_ms": 0.0,
                "preprocess_ms": 0.0,
                "infer_ms": 0.0,
                "batch_total_ms": (time.perf_counter() - total_start) * 1000.0,
                "num_faces": 0.0,
                "valid_faces": 0.0,
            }
            return []

        results = []
        for crop, tid in zip(face_crops_112, track_ids):
            if crop is None or crop.size == 0:
                results.append(None)
                continue

            try:
                # 将 112×112 缩放到 256×256
                resize_start = time.perf_counter()
                crop_256 = cv2.resize(crop, (self.input_size, self.input_size))
                resize_ms += (time.perf_counter() - resize_start) * 1000.0

                # 预处理
                preprocess_start = time.perf_counter()
                x = preprocess_landmark_input(crop_256).unsqueeze(0).to(self.device)
                preprocess_ms += (time.perf_counter() - preprocess_start) * 1000.0

                # 推理
                infer_start = time.perf_counter()
                y = self.model(x)
                infer_ms += (time.perf_counter() - infer_start) * 1000.0
                pts = y.detach().float().cpu().numpy().reshape(-1, 2)

                # 注意: 这里返回的坐标是相对于 256×256 画布的
                # 需要在外部根据实际 bbox 进行投影
                results.append(pts.astype(np.float32))

            except Exception as e:
                if self.enable_debug:
                    print(f"[LipLandmark] Error processing track {tid}: {e}", flush=True)
                results.append(None)

        self.last_timing = {
            "resize_ms": resize_ms,
            "preprocess_ms": preprocess_ms,
            "infer_ms": infer_ms,
            "batch_total_ms": (time.perf_counter() - total_start) * 1000.0,
            "num_faces": float(len(face_crops_112)),
            "valid_faces": float(len(results)),
        }

        return results


# ============== 测试代码 ==============
if __name__ == "__main__":
    import sys

    # 测试模型加载
    print("Testing LipLandmarkDetector...")

    # 假设权重路径
    weights_path = "../model/resnet_50-epoch-724.pth"
    if not os.path.exists(weights_path):
        weights_path = "model/resnet_50-epoch-724.pth"

    if os.path.exists(weights_path):
        detector = LipLandmarkDetector(weights_path, enable_debug=True)
        print("Model loaded successfully!")

        # 测试推理
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        dummy_bbox = [(100, 100, 200, 200)]
        results = detector.detect(dummy_image, dummy_bbox, track_ids=[0])
        print(f"Results: {len(results)}")
        for r in results:
            print(f"  track_id={r.track_id}, valid={r.is_valid}, landmarks shape={r.landmarks_98.shape}")
    else:
        print(f"Weights file not found: {weights_path}")
