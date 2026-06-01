# asd/model_wrapper.py
# -*- coding: utf-8 -*-
"""
TalkNet ASD model wrapper (Realtime)

- Feature extraction:
  1) audio -> MFCC (Ta x 13)
  2) face clips -> gray 112x112 (Tv frames)

- Inference:
  - TensorRT engine (preferred if enabled)
  - PyTorch (.model) fallback (optional)

- IMPORTANT:
  - infer_batch signature accepts fps=... for compatibility with existing worker code
  - 输出为 raw ASD score/logit（不做 sigmoid/softmax）
  - infer_batch 返回 List[np.ndarray]，每张脸对应一条长度为 Tv 的 raw 序列（便于上层做滑窗聚合）
"""

from __future__ import annotations

import os
import sys
import threading
import importlib.util
from concurrent.futures import ThreadPoolExecutor, as_completed
from types import SimpleNamespace
from typing import List, Optional

import cv2
import numpy as np

import torch

try:
    import torchaudio
    from torchaudio.compliance import kaldi as torchaudio_kaldi
except Exception:
    torchaudio = None
    torchaudio_kaldi = None


# =========================
# Config loader (robust)
# =========================
def _build_namespace_from_module(mod) -> SimpleNamespace:
    d = {}
    for k in dir(mod):
        if k.isupper():
            d[k] = getattr(mod, k)
    return SimpleNamespace(**d)


def _load_config() -> SimpleNamespace:
    """
    兼容以下两种写法：
    1) from config.asd_config import C   （C 是 SimpleNamespace）
    2) import config.asd_config as asd_config （模块里是一堆大写常量）
    """
    try:
        from config.asd_config import C as _C  # type: ignore
        # _C 可能就是 SimpleNamespace
        return _C if isinstance(_C, SimpleNamespace) else SimpleNamespace(**dict(_C.__dict__))
    except Exception:
        pass

    # fallback: import module
    from config import asd_config as _mod  # type: ignore

    # 模块里如果自己也构造了 C（你文档里做了兼容别名）
    if hasattr(_mod, "C"):
        _C = getattr(_mod, "C")
        if isinstance(_C, SimpleNamespace):
            return _C
        try:
            return SimpleNamespace(**dict(_C.__dict__))
        except Exception:
            pass

    return _build_namespace_from_module(_mod)


C = _load_config()


# =========================
# Defaults (if missing)
# =========================
def _get(name: str, default):
    return getattr(C, name, default)


FAST_ASD_ROOT = _get("FAST_ASD_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
MODEL_DEVICE = _get("MODEL_DEVICE", "cuda")
ALLOW_CPU_INFERENCE = bool(_get("ALLOW_CPU_INFERENCE", False))
FACE_CROP_SIZE = int(_get("FACE_CROP_SIZE", 112))

AUDIO_SAMPLE_RATE = int(_get("AUDIO_SAMPLE_RATE", 16000))
MFCC_NUM_CEPS = int(_get("MFCC_NUM_CEPS", 13))
MFCC_WINLEN = float(_get("MFCC_WINLEN", 0.025))
MFCC_WINSTEP = float(_get("MFCC_WINSTEP", 0.010))
MFCC_NUM_MELS = int(_get("MFCC_NUM_MELS", 26))
MFCC_NFFT = int(_get("MFCC_NFFT", 512))

TALKNET_WEIGHTS_PATH = _get("TALKNET_WEIGHTS_PATH", os.path.join(FAST_ASD_ROOT, "models", "pretrain_TalkSet.model"))

# TRT config names (兼容你已有写法)
TALKNET_USE_TRT = bool(_get("TALKNET_USE_TRT", False))

# 你可能用 TALKNET_TRT_ENGINE / TALKNET_TRT_ENGINE_PATH
TALKNET_TRT_ENGINE_PATH = str(_get("TALKNET_TRT_ENGINE_PATH", _get("TALKNET_TRT_ENGINE", ""))).strip()

TALKNET_TRT_TV = int(_get("TALKNET_TRT_TV", _get("TALKNET_TV", _get("TARGET_VIDEO_FRAMES", 8))))
TALKNET_TRT_TA = int(_get("TALKNET_TRT_TA", _get("TALKNET_TA", 32)))
TALKNET_TRT_VERBOSE = bool(_get("TALKNET_TRT_VERBOSE", False))

# 是否在 TRT 开启时仍加载 PyTorch 兜底（默认 False，更省显存）
TALKNET_TRT_KEEP_TORCH_FALLBACK = bool(_get("TALKNET_TRT_KEEP_TORCH_FALLBACK", False))
TALKNET_TRT_FALLBACK_TO_TORCH = bool(_get("TALKNET_TRT_FALLBACK_TO_TORCH", True))
TALKNET_TRT_WARMUP_ENABLE = bool(_get("TALKNET_TRT_WARMUP_ENABLE", True))
TALKNET_RAISE_ON_INFER_ERROR = bool(_get("TALKNET_RAISE_ON_INFER_ERROR", True))

# 并行推理配置
# 由于 cross-attention 限制必须 batch=1，可以用多线程并行处理多个人脸
# 每个线程使用独立的 TRT runner，需要额外显存

# 设为 1 则禁用并行（串行处理）
TALKNET_PARALLEL_WORKERS = int(_get("TALKNET_PARALLEL_WORKERS", 2))


def _require_inference_device(requested_device: str, component: str) -> str:
    requested_device = (requested_device or "cuda").strip().lower()
    if requested_device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"[{component}] CUDA inference is required, but torch.cuda.is_available() is False. "
                "Fix the CUDA/PyTorch runtime instead of falling back to CPU."
            )
        return requested_device
    if requested_device == "cpu" and ALLOW_CPU_INFERENCE:
        return "cpu"
    raise RuntimeError(
        f"[{component}] Refusing to run inference on '{requested_device}'. "
        "Set ASD_MODEL_DEVICE=cuda, or explicitly set ASD_ALLOW_CPU_INFERENCE=1 for debugging only."
    )
PROCESS_TAP_ENABLE = bool(_get("PROCESS_TAP_ENABLE", False))
PROCESS_TAP_STAGE_NAME = str(_get("PROCESS_TAP_STAGE_NAME", "talknet")).strip() or "talknet"
PROCESS_TAP_ROOT = str(_get("PROCESS_TAP_ROOT", os.path.join(FAST_ASD_ROOT, "runs", "process_tap"))).strip()
PROCESS_TAP_RUN_NAME = str(_get("PROCESS_TAP_RUN_NAME", "")).strip()


# =========================
# TalkNet import (original)
# =========================
# 让 talknet 内部的相对 import (from model.xxx import ...) 能找到
_talknet_dir = os.path.join(FAST_ASD_ROOT, "talknet")
if _talknet_dir not in sys.path:
    sys.path.insert(0, _talknet_dir)

# 原始封装
from talkNet import talkNet  # noqa: E402


# =========================
# Utils
# =========================
def _to_gray112(frame: np.ndarray) -> np.ndarray:
    """frame: HxW or HxWx3 -> 112x112 gray float32(0~1)"""
    if frame.ndim == 3 and frame.shape[-1] == 3:
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        g = frame
    g = cv2.resize(g, (FACE_CROP_SIZE, FACE_CROP_SIZE), interpolation=cv2.INTER_LINEAR)
    g = g.astype(np.float32) / 255.0
    return g


def _pad_or_trim_time(x: np.ndarray, T: int) -> np.ndarray:
    """Pad last frame / trim to length T along axis=0."""
    if x.shape[0] == T:
        return x
    if x.shape[0] > T:
        return x[-T:]
    # pad
    pad_n = T - x.shape[0]
    last = x[-1:]
    pads = np.repeat(last, pad_n, axis=0)
    return np.concatenate([x, pads], axis=0)


def _ensure_1d_float32(x) -> np.ndarray:
    """把标量/列表/ndarray 统一成 1D float32 ndarray."""
    arr = np.asarray(x)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr.astype(np.float32, copy=False)


def _build_talknet_input_tap_decorator():
    if not PROCESS_TAP_ENABLE:
        return lambda func: func

    try:
        from process_tap import ArtifactMetricsPlugin, tap_processor
    except Exception as exc:
        print(f"[TalkNetASD] process_tap unavailable, skip tap integration: {exc}", flush=True)
        return lambda func: func

    try:
        plugin = ArtifactMetricsPlugin(
            root_dir=PROCESS_TAP_ROOT,
            run_name=PROCESS_TAP_RUN_NAME or None,
            default_audio_sample_rate=AUDIO_SAMPLE_RATE,
        )
    except Exception as exc:
        print(f"[TalkNetASD] failed to initialize process_tap plugin: {exc}", flush=True)
        return lambda func: func

    print(
        f"[TalkNetASD] process_tap enabled. stage={PROCESS_TAP_STAGE_NAME} root={PROCESS_TAP_ROOT}",
        flush=True,
    )
    return tap_processor(
        stage_name=PROCESS_TAP_STAGE_NAME,
        plugins=[plugin],
        output_selector=lambda result, args, kwargs: None,
    )


# =========================
# Main wrapper
# =========================
class TalkNetASDModel:
    def __init__(self):
        self.device = _require_inference_device(MODEL_DEVICE, "TalkNetASD")
        self._mfcc_device_fallback_warned = False
        self._mfcc_cache = {}
        self._mfcc_max_cache = 20
        self._mfcc_cache_lock = threading.RLock()

        # TRT branch
        self.use_trt = bool(TALKNET_USE_TRT)
        self.trt_engine_path = TALKNET_TRT_ENGINE_PATH
        self.trt_tv = int(TALKNET_TRT_TV)
        self.trt_ta = int(TALKNET_TRT_TA)
        self.trt_verbose = bool(TALKNET_TRT_VERBOSE)
        self.trt_fallback_to_torch = bool(TALKNET_TRT_FALLBACK_TO_TORCH)

        # 关键：TRT runner 做成 thread-local，避免 worker 线程 invalid device context
        self._trt_tls = threading.local()

        # 并行推理配置
        self.parallel_workers = TALKNET_PARALLEL_WORKERS
        self._executor = None  # 延迟创建
        
        if self.use_trt:
            if not self.trt_engine_path:
                raise ValueError("[TalkNetASD] TALKNET_TRT_ENGINE_PATH is empty but TALKNET_USE_TRT=True")
            if not os.path.isfile(self.trt_engine_path):
                raise FileNotFoundError(f"[TalkNetASD] TensorRT engine not found: {self.trt_engine_path}")
            trt_missing = []
            if importlib.util.find_spec("tensorrt") is None:
                trt_missing.append("tensorrt")
            if importlib.util.find_spec("pycuda") is None:
                trt_missing.append("pycuda")
            if trt_missing:
                msg = "[TalkNetASD] TensorRT disabled: missing %s" % ", ".join(trt_missing)
                if not self.trt_fallback_to_torch:
                    raise ImportError(msg + ". Set ASD_TALKNET_TRT_FALLBACK_TO_TORCH=1 or install the missing package(s).")
                print(msg + "; falling back to PyTorch CUDA.", flush=True)
                self.use_trt = False
            else:
                print(f"[TalkNetASD] TensorRT enabled. engine={self.trt_engine_path}", flush=True)
                print(f"[TalkNetASD] Parallel workers: {self.parallel_workers}", flush=True)

        # PyTorch fallback (可选)
        self.wrapper = None
        self.model = None
        self.loss_av = None

        if (not self.use_trt) or TALKNET_TRT_KEEP_TORCH_FALLBACK:
            self._init_torch()

        if self.use_trt and TALKNET_TRT_WARMUP_ENABLE:
            self._warmup_trt()

    def _get_executor(self) -> ThreadPoolExecutor:
        """获取或创建线程池执行器"""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self.parallel_workers,
                thread_name_prefix="talknet_worker"
            )
        return self._executor

    def _warmup_trt(self) -> None:
        """Pre-create TRT runners so new tracks do not pay this cost in image_cb."""
        try:
            audio = np.zeros((1, self.trt_ta, 13), dtype=np.float32)
            video = np.zeros((1, self.trt_tv, FACE_CROP_SIZE, FACE_CROP_SIZE), dtype=np.float32)

            runner = self._get_trt_runner()
            if runner is not None:
                _ = runner.infer(audio=audio, video=video)

            if self.parallel_workers > 1:
                executor = self._get_executor()
                barrier = threading.Barrier(self.parallel_workers)

                def _warm_worker():
                    barrier.wait(timeout=10.0)
                    return self._get_trt_runner().infer(audio=audio, video=video)

                futures = [executor.submit(_warm_worker) for _ in range(self.parallel_workers)]
                for future in as_completed(futures):
                    future.result()

            print(
                f"[TalkNetASD] TensorRT warmup completed (workers={self.parallel_workers})",
                flush=True,
            )
        except Exception as exc:
            if not self.trt_fallback_to_torch:
                raise
            print(f"[TalkNetASD] TensorRT warmup skipped: {exc}", flush=True)

    def _init_torch(self):
        self.wrapper = talkNet(lr=0.0001, lrDecay=0.95)
        if os.path.isfile(TALKNET_WEIGHTS_PATH):
            self.wrapper.loadParameters(TALKNET_WEIGHTS_PATH)
        else:
            print(f"[TalkNetASD] WARNING: weights not found: {TALKNET_WEIGHTS_PATH}", flush=True)

        self.model = self.wrapper.model.to(self.device).eval()
        self.loss_av = self.wrapper.lossAV.to(self.device).eval()
        print(f"[TalkNetASD] PyTorch ready on {self.device}, weights={TALKNET_WEIGHTS_PATH}", flush=True)

    def _get_trt_runner(self):
        """
        在当前线程获取/初始化 TalkNetTRT。
        注意：这里故意做延迟 import，让 pycuda.autoinit 在 worker 线程中建立 context。
        """
        runner = getattr(self._trt_tls, "runner", None)
        if runner is not None:
            return runner

        # 延迟导入（关键）
        try:
            from trt_infer.talknet_trt import TalkNetTRT  # noqa: E402
        except ImportError as exc:
            if not self.trt_fallback_to_torch:
                raise
            print(f"[TalkNetASD] TensorRT unavailable at runtime: {exc}; falling back to PyTorch CUDA.", flush=True)
            self.use_trt = False
            if self.model is None or self.loss_av is None:
                self._init_torch()
            return None

        sev = "INFO" if self.trt_verbose else "ERROR"
        try:
            runner = TalkNetTRT(self.trt_engine_path, logger_severity=sev)
        except Exception as exc:
            if not self.trt_fallback_to_torch:
                raise
            print(f"[TalkNetASD] TensorRT runner unavailable: {exc}; falling back to PyTorch CUDA.", flush=True)
            self.use_trt = False
            if self.model is None or self.loss_av is None:
                self._init_torch()
            return None
        self._trt_tls.runner = runner
        return runner

    def _extract_mfcc(self, mono_audio_16k: np.ndarray, target_ta: int = None) -> np.ndarray:
        """
        mono_audio_16k: (N,) float32
        return: (Ta, 13) float32
        """
        Ta = target_ta if target_ta is not None else (self.trt_ta if self.use_trt else None)

        if Ta is not None and len(mono_audio_16k) > 1000:
            required_samples = (Ta + 2) * 160 + 400
            if len(mono_audio_16k) > required_samples:
                dropped = len(mono_audio_16k) - required_samples
                dropped = (dropped // 160) * 160
                keep_samples = len(mono_audio_16k) - dropped
                mono_audio_16k = mono_audio_16k[-keep_samples:]

        audio_key = mono_audio_16k.tobytes()
        with self._mfcc_cache_lock:
            feat_cached = self._mfcc_cache.get(audio_key)
        if feat_cached is not None:
            Ta_final = Ta if Ta is not None else feat_cached.shape[0]
            return _pad_or_trim_time(feat_cached, Ta_final)

        if torchaudio is None or torchaudio_kaldi is None:
            raise ImportError(
                "[TalkNetASD] torchaudio is required for MFCC extraction. "
                "Please install torchaudio matching the current torch build."
            )

        min_samples = max(1, int(round(AUDIO_SAMPLE_RATE * MFCC_WINLEN)))
        if mono_audio_16k.size == 0:
            mono_audio_16k = np.zeros((min_samples,), dtype=np.float32)
        elif mono_audio_16k.shape[0] < min_samples:
            mono_audio_16k = np.pad(mono_audio_16k, (0, min_samples - mono_audio_16k.shape[0]), mode="edge")

        waveform = torch.from_numpy(np.ascontiguousarray(mono_audio_16k, dtype=np.float32)).unsqueeze(0)

        mfcc_kwargs = dict(
            sample_frequency=float(AUDIO_SAMPLE_RATE),
            frame_length=float(MFCC_WINLEN * 1000.0),
            frame_shift=float(MFCC_WINSTEP * 1000.0),
            num_ceps=int(MFCC_NUM_CEPS),
            num_mel_bins=int(MFCC_NUM_MELS),
            dither=0.0,
            preemphasis_coefficient=0.97,
            use_energy=True,
            cepstral_lifter=22.0,
            low_freq=0.0,
            high_freq=0.0,
            window_type="hamming",
            round_to_power_of_two=False,
            remove_dc_offset=True,
            snip_edges=True,
        )

        prefer_cuda_mfcc = self.device.startswith("cuda") and torch.cuda.is_available()
        if prefer_cuda_mfcc:
            try:
                feat_tensor = torchaudio_kaldi.mfcc(waveform.to(self.device), **mfcc_kwargs)
            except Exception as exc:
                if not self._mfcc_device_fallback_warned:
                    print(
                        f"[TalkNetASD] torchaudio MFCC CUDA path unavailable, fallback to CPU: {exc}",
                        flush=True,
                    )
                    self._mfcc_device_fallback_warned = True
                feat_tensor = torchaudio_kaldi.mfcc(waveform, **mfcc_kwargs)
        else:
            feat_tensor = torchaudio_kaldi.mfcc(waveform, **mfcc_kwargs)

        feat = feat_tensor.detach().cpu().numpy().astype(np.float32, copy=False)

        Ta_final = Ta if Ta is not None else feat.shape[0]
        feat = _pad_or_trim_time(feat, Ta_final)
        
        with self._mfcc_cache_lock:
            if len(self._mfcc_cache) >= self._mfcc_max_cache:
                self._mfcc_cache.pop(next(iter(self._mfcc_cache)))
            self._mfcc_cache[audio_key] = feat
        
        return feat

    def _prep_video_clip(self, clip) -> np.ndarray:
        """
        clip: (Tv, H, W, 3) or (Tv, H, W) or list of frames
        return: (Tv, 112,112) float32 0~1
        """
        if isinstance(clip, list):
            clip = np.stack(clip, axis=0)

        if clip.ndim == 3:
            frames = [_to_gray112(clip[i]) for i in range(clip.shape[0])]
        elif clip.ndim == 4:
            frames = [_to_gray112(clip[i]) for i in range(clip.shape[0])]
        else:
            raise ValueError(f"Unexpected clip shape: {getattr(clip, 'shape', None)}")

        v = np.stack(frames, axis=0)  # (Tv,112,112)
        Tv = self.trt_tv if self.use_trt else v.shape[0]
        v = _pad_or_trim_time(v, Tv)
        return v

    @torch.no_grad()
    def infer_batch(
        self,
        audio_clip: np.ndarray,
        video_clips: List[np.ndarray],
        fps: Optional[float] = None,  # 兼容旧 worker 的 fps=xxx
        **kwargs,
    ) -> List[np.ndarray]:
        """
        audio_clip: 1D mono audio (float32) at 16kHz
        video_clips: list length B, each is (Tv,112,112[,3]) face clip

        return:
          probs_list: List[np.ndarray], len=B
            each element is shape=(Tv,) raw logit/score sequence for that face (不做 sigmoid)
        
        重要说明：
        TalkNet 的 cross-attention 机制在 batch 处理时会导致不同人脸的特征相互干扰，
        使得先出现的 track 更容易被误识别为说话人。
        因此必须逐个人脸独立推理（batch_size=1）。
        
        优化：使用线程池并行处理多个人脸，提高吞吐量。
        每个线程有独立的 TRT runner 实例（thread-local）。
        """
        if audio_clip is None or len(audio_clip) == 0 or not video_clips:
            return []

        # 确保 audio_clip 是 numpy 数组
        if not isinstance(audio_clip, np.ndarray):
            audio_clip = np.asarray(audio_clip, dtype=np.float32)

        B = len(video_clips)
        
        # 如果只有一个人脸或禁用并行，直接串行处理
        if B == 1 or self.parallel_workers <= 1:
            probs_list = []
            for i in range(B):
                single_probs = self._infer_single(audio_clip, video_clips[i])
                probs_list.append(single_probs)
            return probs_list
        
        # 多人脸时使用线程池并行推理
        # 每个线程有独立的 TRT runner (thread-local)，可以真正并行执行
        executor = self._get_executor()
        futures = {}
        
        for i in range(B):
            future = executor.submit(self._infer_single, audio_clip, video_clips[i])
            futures[future] = i
        
        # 按原始顺序收集结果
        probs_list = [None] * B
        for future in as_completed(futures):
            idx = futures[future]
            try:
                probs_list[idx] = future.result()
            except Exception as e:
                print(f"[TalkNetASD] Parallel infer error for face {idx}: {e}", flush=True)
                if TALKNET_RAISE_ON_INFER_ERROR:
                    raise
                # 失败时返回零向量
                Tv = self.trt_tv if self.use_trt else 8
                probs_list[idx] = np.zeros(Tv, dtype=np.float32)
        
        return probs_list
    
    def _infer_single(
        self,
        audio_clip: np.ndarray,
        video_clip: np.ndarray,
    ) -> np.ndarray:
        """
        单个人脸的推理（batch_size=1），避免 cross-attention 干扰。
        """
        # 0) 预先计算需要的 Ta，避免 MFCC 在大量冗余音频上空转
        v_len = len(video_clip) if isinstance(video_clip, list) else video_clip.shape[0]
        if self.use_trt:
            ta_target = self.trt_ta
        else:
            ratio = 4.0
            if TALKNET_TRT_TA and TALKNET_TRT_TV:
                ratio = float(TALKNET_TRT_TA) / float(TALKNET_TRT_TV)
            ta_target = max(1, int(round(v_len * ratio))) if v_len > 0 else None

        # 1) features
        a = self._extract_mfcc(audio_clip.astype(np.float32, copy=False), target_ta=ta_target)  # (Ta,13)
        v = self._prep_video_clip(video_clip)  # (Tv,112,112)

        if not self.use_trt:
            Tv = int(v.shape[0])
            # a = _pad_or_trim_time(a, ta_target) # 在 extract_mfcc 里已经基于 target_ta 切过了

        # batch_size = 1 的输入
        audio_b = a[None, :, :].astype(np.float32)   # (1,Ta,13)
        video_b = v[None, :, :, :].astype(np.float32) if v.ndim == 3 else v[None, :, :].astype(np.float32)  # (1,Tv,112,112)

        # 2) TRT inference (raw logits)
        if self.use_trt:
            runner = self._get_trt_runner()
            if runner is None:
                return self._infer_single(audio_clip, video_clip)
            try:
                outs = runner.infer(audio=audio_b, video=video_b)
            except Exception as exc:
                if not self.trt_fallback_to_torch:
                    raise
                print(f"[TalkNetASD] TensorRT inference failed: {exc}; falling back to PyTorch CUDA.", flush=True)
                self.use_trt = False
                if self.model is None or self.loss_av is None:
                    self._init_torch()
                return self._infer_single(audio_clip, video_clip)

            out_name = list(outs.keys())[0]
            y = np.asarray(outs[out_name])

            Tv = self.trt_tv
            speak = self._parse_trt_output(y, B=1, Tv=Tv)
            return _ensure_1d_float32(speak[0])

        # 3) PyTorch fallback
        if self.model is None or self.loss_av is None:
            raise RuntimeError("[TalkNetASD] PyTorch fallback not initialized and TensorRT is disabled.")

        audio_t = torch.from_numpy(audio_b).to(self.device)
        video_t = torch.from_numpy(video_b).to(self.device)

        a_emb = self.model.forward_audio_frontend(audio_t)
        v_emb = self.model.forward_visual_frontend(video_t)
        a2, v2 = self.model.forward_cross_attention(a_emb, v_emb)

        outs_av = self.model.forward_audio_visual_backend(a2, v2)
        logits = self.loss_av.FC(outs_av) if hasattr(self.loss_av, "FC") else self.loss_av(outs_av)

        Tv = v.shape[0]
        if isinstance(logits, np.ndarray):
            speak_np = logits.reshape(1, Tv)
        else:
            if logits.ndim == 2 and logits.shape[1] >= 2:
                speak = logits[:, 1].view(1, Tv)
            else:
                speak = logits.view(1, Tv)
            speak_np = speak.detach().float().cpu().numpy()

        return _ensure_1d_float32(speak_np[0])
    
    def _parse_trt_output(self, y: np.ndarray, B: int, Tv: int) -> np.ndarray:
        """解析 TRT 输出为 (B, Tv) 的 speaking logits"""
        speak = None

        if y.ndim == 3 and y.shape[0] == B and y.shape[1] == Tv:
            if y.shape[-1] >= 2:
                speak = y[:, :, 1]
            elif y.shape[-1] == 1:
                speak = y[:, :, 0]
            else:
                speak = y

        elif y.ndim == 2:
            if y.shape[0] == B * Tv:
                if y.shape[1] >= 2:
                    speak = y[:, 1].reshape(B, Tv)
                else:
                    speak = y[:, 0].reshape(B, Tv)
            elif y.shape[0] == B and y.shape[1] == Tv:
                speak = y
            elif y.shape[0] == B and y.shape[1] == 1:
                speak = np.repeat(y, Tv, axis=1)

        elif y.ndim == 1:
            if y.shape[0] == B * Tv:
                speak = y.reshape(B, Tv)
            elif y.shape[0] == B:
                speak = np.repeat(y.reshape(B, 1), Tv, axis=1)

        if speak is None:
            raise RuntimeError(f"[TalkNetASD][TRT] Unexpected output shape: {y.shape}")
        
        return speak

    @_build_talknet_input_tap_decorator()
    def infer_batch_per_track(
        self,
        audio_clips: List[np.ndarray],
        video_clips: List[np.ndarray],
        fps: Optional[float] = None,
        **kwargs,
    ) -> List[np.ndarray]:
        """
        为每个 track 独立进行推理，每个 track 使用其独立的音频片段。
        
        这是解决 "track_id 靠前更容易被识别为说话人" 问题的关键方法。
        原因：之前所有 track 共用一段音频，但各自的视频时间窗口不同，
        导致音视频时间对齐不正确，先出现的 track 被误判。
        
        Args:
            audio_clips: List[np.ndarray], 每个 track 对应的独立音频片段 (1D float32 @ 16kHz)
            video_clips: List[np.ndarray], 每个 track 的视频片段 (Tv,112,112[,3])
            fps: 兼容参数
            
        Returns:
            probs_list: List[np.ndarray], 每个 track 的 ASD 预测得分 (raw logit)
        """
        if not audio_clips or not video_clips:
            return []
        
        if len(audio_clips) != len(video_clips):
            raise ValueError(
                f"[TalkNetASD] audio_clips({len(audio_clips)}) != video_clips({len(video_clips)})"
            )
        
        B = len(video_clips)
        Tv = self.trt_tv if self.use_trt else 8

        # 单 track 或禁用并行时，保留最直接的串行路径
        if B == 1 or self.parallel_workers <= 1:
            probs_list = []
            for i in range(B):
                audio_clip = audio_clips[i]
                video_clip = video_clips[i]

                if not isinstance(audio_clip, np.ndarray):
                    audio_clip = np.asarray(audio_clip, dtype=np.float32)

                if audio_clip is None or len(audio_clip) == 0:
                    probs_list.append(np.zeros(Tv, dtype=np.float32))
                    continue

                probs_list.append(self._infer_single(audio_clip, video_clip))
            return probs_list

        # 多 track 时并行提交，但每个 track 仍然独立推理，避免 TalkNet batch 串扰。
        executor = self._get_executor()
        futures = {}
        probs_list = [None] * B

        for i in range(B):
            audio_clip = audio_clips[i]
            video_clip = video_clips[i]

            if not isinstance(audio_clip, np.ndarray):
                audio_clip = np.asarray(audio_clip, dtype=np.float32)

            if audio_clip is None or len(audio_clip) == 0:
                probs_list[i] = np.zeros(Tv, dtype=np.float32)
                continue

            future = executor.submit(self._infer_single, audio_clip, video_clip)
            futures[future] = i

        for future in as_completed(futures):
            idx = futures[future]
            try:
                probs_list[idx] = future.result()
            except Exception as e:
                print(f"[TalkNetASD] Parallel per-track infer error for track {idx}: {e}", flush=True)
                if TALKNET_RAISE_ON_INFER_ERROR:
                    raise
                probs_list[idx] = np.zeros(Tv, dtype=np.float32)

        for i, probs in enumerate(probs_list):
            if probs is None:
                probs_list[i] = np.zeros(Tv, dtype=np.float32)

        return probs_list
