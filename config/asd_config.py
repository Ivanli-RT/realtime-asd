# config/asd_config.py
# -*- coding: utf-8 -*-
"""
实时主动说话人识别(ASD)配置文件
"""

import os


def _env_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return int(default)
    return int(value)


def _env_float(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return float(default)
    return float(value)


# ========= 路径相关 =========
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAST_ASD_ROOT = ROOT
MODEL_DIR = os.path.join(ROOT, "models")

# TalkNet预训练模型路径
TALKNET_WEIGHTS_PATH = os.path.join(MODEL_DIR, "pretrain_TalkSet.model")

# ========= ROS Topic & 同步 =========
VIDEO_TOPIC = "/zj_humanoid/sensor/realsense_head/color/image_raw"
AUDIO_TOPIC = "/zj_humanoid/audio/microphone/audio_data_raw"
AUDIO_MSG_TYPE = "audio/AudioData"
AUDIO_OFFSET_SEC = 0.151
AUDIO_AUTO_SELECT_DEVICE = False
AUDIO_DEVICE_NAME = ""
AUDIO_SELECT_DEVICE_SERVICE = "/zj_humanoid/audio/microphone/select_device"
AUDIO_GET_DEVICES_SERVICE = "/zj_humanoid/audio/microphone/get_devices_list"
AUDIO_SERVICE_WAIT_TIMEOUT_SEC = 5.0


# ========= 音频特征配置 =========
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 6
AUDIO_USE_CHANNEL = int(os.environ.get("ASD_AUDIO_USE_CHANNEL", "2"))
AUDIO_BUFFER_SECONDS = 5.0
AUDIO_ASYNC_PROCESSING = _env_bool("ASD_AUDIO_ASYNC_PROCESSING", True)
AUDIO_ASYNC_QUEUE_SIZE = _env_int("ASD_AUDIO_ASYNC_QUEUE_SIZE", 200)
AUDIO_ISOLATED_PROCESS = _env_bool("ASD_AUDIO_ISOLATED_PROCESS", True)
AUDIO_ISOLATED_BATCH_PACKETS = _env_int("ASD_AUDIO_ISOLATED_BATCH_PACKETS", 10)
AUDIO_ISOLATED_QUEUE_SIZE = _env_int("ASD_AUDIO_ISOLATED_QUEUE_SIZE", 50)
MFCC_NUM_CEPS = 13
MFCC_WINLEN = 0.025
MFCC_WINSTEP = 0.010
MFCC_NUM_MELS = 26
MFCC_NFFT = 512


# ========= 视频处理配置 =========
VIDEO_PROCESS_FPS = _env_float("ASD_VIDEO_PROCESS_FPS", 10)
FACE_CROP_SIZE = 112
MIN_FACE_SIZE = 20

# ========= 主说话人相机坐标估计 =========
SPEAKER_POSITION_ENABLE = _env_bool("ASD_SPEAKER_POSITION_ENABLE", True)
SPEAKER_POSITION_USE_DEPTH = _env_bool("ASD_SPEAKER_POSITION_USE_DEPTH", True)
SPEAKER_CAMERA_INFO_TOPIC = os.environ.get(
    "ASD_SPEAKER_CAMERA_INFO_TOPIC",
    "/zj_humanoid/sensor/realsense_head/depth/camera_info",
)
SPEAKER_DEPTH_TOPIC = os.environ.get(
    "ASD_SPEAKER_DEPTH_TOPIC",
    "/zj_humanoid/sensor/realsense_head/aligned_depth_to_color/image_raw",
)
SPEAKER_DEPTH_MIN_M = _env_float("ASD_SPEAKER_DEPTH_MIN_M", 0.30)
SPEAKER_DEPTH_MAX_M = _env_float("ASD_SPEAKER_DEPTH_MAX_M", 4.00)
SPEAKER_DEPTH_VALID_RATIO_MIN = _env_float("ASD_SPEAKER_DEPTH_VALID_RATIO_MIN", 0.20)
SPEAKER_DEPTH_ROI_SCALE = _env_float("ASD_SPEAKER_DEPTH_ROI_SCALE", 0.40)
SPEAKER_DEPTH_MAX_AGE_SEC = _env_float("ASD_SPEAKER_DEPTH_MAX_AGE_SEC", 0.50)
SPEAKER_DEPTH_UINT16_SCALE = _env_float("ASD_SPEAKER_DEPTH_UINT16_SCALE", 0.001)
SPEAKER_CAMERA_FX = _env_float("ASD_SPEAKER_CAMERA_FX", 0.0)
SPEAKER_CAMERA_FY = _env_float("ASD_SPEAKER_CAMERA_FY", 0.0)
SPEAKER_CAMERA_CX = _env_float("ASD_SPEAKER_CAMERA_CX", 0.0)
SPEAKER_CAMERA_CY = _env_float("ASD_SPEAKER_CAMERA_CY", 0.0)
SPEAKER_CAMERA_FRAME_ID = os.environ.get(
    "ASD_SPEAKER_CAMERA_FRAME_ID",
    "realsense_head_color_optical_frame",
)

# ========= 人脸检测配置 =========

# 人脸检测器选择: 当前瘦身版仅保留 yolo
FACE_DETECTOR_TYPE = "yolo"

# 通用人脸检测参数
FACE_DETECTION_CONF_THRESH = _env_float("ASD_FACE_DETECTION_CONF_THRESH", 0.6)  # 置信度阈值

# --- YOLO 专用配置 ---
_default_yolo_model_name = os.environ.get("ASD_YOLO_MODEL_NAME", "yolov11n-face.pt")
YOLO_MODEL_PATH = os.environ.get(
    "ASD_YOLO_MODEL_PATH",
    os.path.join(MODEL_DIR, _default_yolo_model_name),
)  # YOLO模型路径
YOLO_IMGSZ = _env_int("ASD_YOLO_IMGSZ", 416)  # YOLO输入图像尺寸
YOLO_IOU_THRESH = _env_float("ASD_YOLO_IOU_THRESH", 0.45)  # YOLO的NMS IoU阈值

# Jetson优化选项
YOLO_DISABLE_FP16 = _env_bool("ASD_YOLO_DISABLE_FP16", False)  # Jetson上禁用FP16(某些版本不稳定)
YOLO_DISABLE_TTA = _env_bool("ASD_YOLO_DISABLE_TTA", True)   # 禁用测试时增强(加速)
YOLO_USE_TRT = _env_bool("ASD_YOLO_USE_TRT", False)
YOLO_TRT_ENGINE_PATH = os.environ.get(
    "ASD_YOLO_TRT_ENGINE_PATH",
    os.path.join(MODEL_DIR, "yolov11n-face_416_fp16.engine"),
)
YOLO_TRT_DYNAMIC = _env_bool("ASD_YOLO_TRT_DYNAMIC", False)
YOLO_TRT_STRIDE = _env_int("ASD_YOLO_TRT_STRIDE", 32)
YOLO_FORCE_CONTIGUOUS_INPUT = _env_bool("ASD_YOLO_FORCE_CONTIGUOUS_INPUT", True)


# ========= ASD窗口 & 触发 =========
CLIP_SECONDS = 1.0
MIN_AUDIO_SECONDS = 0.3
MIN_VIDEO_FRAMES = 3

# TalkNet 调用步长（基于处理帧计数）:
# 1 = 每个处理帧都跑 TalkNet
# 2 = 每 2 个处理帧跑 1 次 TalkNet（Lip 仍每帧更新）
TALKNET_FRAME_STRIDE = _env_int("ASD_TALKNET_FRAME_STRIDE", 4)

# 兼容旧配置名
FRAME_STRIDE = TALKNET_FRAME_STRIDE

# 融合时允许使用的 Lip 状态最大时延（秒）
LIP_STATE_MAX_AGE_SEC = 0.5


# ========= Tracking 配置 =========
IOU_THRESHOLD = 0.3
MAX_TRACK_MISSED = 15
MAX_TRACKS = 5

# ========= ASD处理人脸数量限制 =========
# 与源项目保持一致：多人场景下最多同时处理 5 张人脸
MAX_FACES_FOR_ASD = _env_int("ASD_MAX_FACES_FOR_ASD", 5)

# ========= 模型设备 =========
MODEL_DEVICE = os.environ.get("ASD_MODEL_DEVICE", "cuda").strip() or "cuda"
ALLOW_CPU_INFERENCE = _env_bool("ASD_ALLOW_CPU_INFERENCE", False)


# ========= 输出平滑 & 说话判定阈值 =========
SMOOTH_ALPHA = 0.8
SPEAK_ON_THRESH = 0.45
SPEAK_OFF_THRESH = 0.3


# ========= 主动说话人显示模式配置 =========
# 显示模式选择:
#   "threshold" - 模式1: 所有超过阈值的人都显示为主动说话人 (适合多人同时说话)
#   "top_scorer" - 模式2: 只显示得分最高的人（严格只选一个）
#   "top_scorer_with_override" - 模式3: 只显示最高分,但超过强制阈值的人也显示
ACTIVE_SPEAKER_DISPLAY_MODE = "top_scorer"  # 仅输出一个主动说话人（得分最高）

# 模式1参数: 阈值模式
THRESHOLD_MODE_MIN_SCORE = 0.25  # 进一步提召回，减少“该亮不亮”
THRESHOLD_MODE_RELATIVE_MARGIN = 0.12  # 适度收紧，抑制双人同亮

# 模式2参数: 最高分模式
TOP_SCORER_MIN_SCORE = 0.35  # 最高分者的最低分数要求(避免无人说话时误判)

# 模式3参数: 最高分+强制显示模式
TOP_SCORER_OVERRIDE_THRESHOLD = 0.7  # 强制显示阈值

# Active speaker score source:
#   "smooth"  - use EMA-smoothed state["p"]
#   "raw"     - use latest state["p_raw"] (recommended for multi-person)
#   "history" - use mean of recent state["history"]
#   "fused"   - use fused score of ASD and Lip motion
ACTIVE_SPEAKER_SCORE_SOURCE = "fused"

# Spatial priority for active-speaker selection:
# A side/edge speaker cannot acquire the main-speaker lock immediately while a
# center-core user is visible. The system waits briefly for a center-core speech
# event; if none arrives, the earliest side candidate is allowed to acquire.
ACTIVE_SPEAKER_SPATIAL_PRIORITY_ENABLED = _env_bool("ASD_ACTIVE_SPEAKER_SPATIAL_PRIORITY_ENABLED", True)
ACTIVE_SPEAKER_CENTER_WAIT_SECONDS = _env_float("ASD_ACTIVE_SPEAKER_CENTER_WAIT_SECONDS", 3.0)
# Core box format: (x, y, w, h). By default these are normalized frame
# coordinates, so the same config works across camera resolutions.
ACTIVE_SPEAKER_CENTER_CORE_BOX = os.environ.get("ASD_ACTIVE_SPEAKER_CENTER_CORE_BOX", "0.35,0.10,0.30,0.80")
ACTIVE_SPEAKER_CENTER_CORE_BOX_NORMALIZED = _env_bool("ASD_ACTIVE_SPEAKER_CENTER_CORE_BOX_NORMALIZED", True)

# First-speaker lock:
# Once a speaker is selected, only this track is checked for the next window.
# If the locked track speaks again, the window is refreshed from that detection time.
# The lock is released after the refreshed window expires, or after the locked
# speaker is dropped by the tracker (controlled by MAX_TRACK_MISSED).
ACTIVE_SPEAKER_LOCK_ENABLED = _env_bool("ASD_ACTIVE_SPEAKER_LOCK_ENABLED", True)
ACTIVE_SPEAKER_LOCK_SECONDS = _env_float("ASD_ACTIVE_SPEAKER_LOCK_SECONDS", 5.0)
ACTIVE_SPEAKER_LOCK_REID_IOU = _env_float("ASD_ACTIVE_SPEAKER_LOCK_REID_IOU", 0.35)
ACTIVE_SPEAKER_LOCK_REID_IOU_MARGIN = _env_float("ASD_ACTIVE_SPEAKER_LOCK_REID_IOU_MARGIN", 0.08)
ACTIVE_SPEAKER_LOCK_REID_CENTER_RATIO = _env_float("ASD_ACTIVE_SPEAKER_LOCK_REID_CENTER_RATIO", 0.60)
ACTIVE_SPEAKER_LOCK_REID_SIZE_RATIO_MIN = _env_float("ASD_ACTIVE_SPEAKER_LOCK_REID_SIZE_RATIO_MIN", 0.45)
ACTIVE_SPEAKER_LOCK_REID_SIZE_RATIO_MAX = _env_float("ASD_ACTIVE_SPEAKER_LOCK_REID_SIZE_RATIO_MAX", 2.20)
ACTIVE_SPEAKER_LOCK_REFRESH_SCORE = _env_float(
    "ASD_ACTIVE_SPEAKER_LOCK_REFRESH_SCORE",
    min(TOP_SCORER_MIN_SCORE, SPEAK_OFF_THRESH),
)
ACTIVE_SPEAKER_LOCK_MISSING_GRACE_SEC = _env_float("ASD_ACTIVE_SPEAKER_LOCK_MISSING_GRACE_SEC", 0.80)
ACTIVE_SPEAKER_DEBUG = _env_bool("ASD_ACTIVE_SPEAKER_DEBUG", False)


# ========= 唇部运动检测配置 =========
# 启用唇部运动辅助检测 (闭嘴否决 + 融合 TalkNet 得分)
ENABLE_LIP_MOTION = _env_bool("ASD_ENABLE_LIP_MOTION", True)

_lip_weights_candidates = [
	os.path.join(MODEL_DIR, "resnet_50-epoch-724.pth"),
	os.path.join(os.path.dirname(ROOT), "model", "resnet_50-epoch-724.pth"),
	"/home/naviai/face_detection_unit/model/resnet_50-epoch-724.pth",
	"/home/naviai/face_detection_unit.bat_0121/model/resnet_50-epoch-724.pth",
]
LIP_LANDMARK_WEIGHTS_PATH = _lip_weights_candidates[0]
for _p in _lip_weights_candidates:
	if os.path.isfile(_p):
		LIP_LANDMARK_WEIGHTS_PATH = _p
		break

LIP_USE_TRT = _env_bool("ASD_LIP_USE_TRT", False)
LIP_TRT_ENGINE_PATH = os.environ.get("ASD_LIP_TRT_ENGINE_PATH", "")
LIP_WARMUP_ENABLE = _env_bool("ASD_LIP_WARMUP_ENABLE", True)

# 唇部检测等待机制（当前顺序模式保留参数兼容）
LIP_WAIT_FOR_RESULT = True
LIP_WAIT_TIMEOUT_MS = 200

# 唇部运动分析参数
LIP_MOTION_HISTORY_FRAMES = 10
LIP_LAR_CLOSED_THRESHOLD = 0.13
LIP_LAR_OPEN_THRESHOLD = 0.25
LIP_MOTION_STD_THRESHOLD = 0.05
LIP_CONSECUTIVE_CLOSED_FOR_PENALTY = 12
LIP_CONSECUTIVE_NO_DETECT_FOR_PENALTY = 10

# Lip/TalkNet 时域同步：根据 TalkNet 节拍和耗时动态调整 Lip 窗口与新鲜度阈值
LIP_SYNC_ADAPTIVE = True
LIP_SYNC_TARGET_MULTIPLIER = 1.0
LIP_SYNC_MIN_HISTORY_FRAMES = 4
LIP_SYNC_MAX_HISTORY_FRAMES = 18
LIP_SYNC_MIN_AGE_SEC = 0.10
LIP_SYNC_MAX_AGE_SEC = 0.60
LIP_SYNC_AGE_MARGIN_SEC = 0.05

# 融合参数
LIP_ASD_WEIGHT = 0.7
LIP_MOTION_WEIGHT = 0.2
LIP_VETO_THRESHOLD = -0.8
LIP_TALKNET_CONFIRM_THRESH = 0.28

LIP_MOTION_DEBUG = False


# ========= 调试开关 =========
PRINT_ASD_TIME = True
DEBUG_SHOW_WINDOW = os.environ.get("ASD_DEBUG_SHOW_WINDOW", "1") == "1"
DEBUG_WINDOW_NAME = "Realtime ASD"
DEBUG_SHOW_FPS = True
DEBUG_SHOW_SCORES = True  # 是否在框上显示分数
YOLO_PROFILE_ENABLE = _env_bool("ASD_YOLO_PROFILE_ENABLE", False)
YOLO_PROFILE_INTERVAL_SEC = _env_float("ASD_YOLO_PROFILE_INTERVAL_SEC", 3.0)

# 中间结果输出总开关：
# - production 默认关闭，避免部署环境写 manifest/frame_results/jsonl 等测试产物
# - test/debug/development 模式默认允许，再由具体开关决定写哪些内容
ASD_RUNTIME_PROFILE = os.environ.get("ASD_RUNTIME_PROFILE", "production").strip().lower()
INTERMEDIATE_RESULTS_ENABLE = os.environ.get(
	"ASD_INTERMEDIATE_RESULTS_ENABLE",
	"1" if ASD_RUNTIME_PROFILE in {"test", "debug", "development"} else "0",
) == "1"

CAPTURE_ONLY_MODE = os.environ.get(
	"ASD_STREAM_CAPTURE_ONLY",
	os.environ.get("ASD_CAPTURE_ONLY", "0"),
) == "1" and INTERMEDIATE_RESULTS_ENABLE
DETECTOR_ONLY_MODE = _env_bool("ASD_DETECTOR_ONLY", False)
DETECTOR_ONLY_SKIP_AUDIO = _env_bool("ASD_DETECTOR_ONLY_SKIP_AUDIO", True)
DETECTOR_SYNTHETIC_FRAME = _env_bool("ASD_DETECTOR_SYNTHETIC_FRAME", False)

TARGET_VIDEO_FRAMES = 8
AUDIO_MARGIN_SEC = 0.05


# Process tap: optional TalkNet input dump
PROCESS_TAP_ENABLE = INTERMEDIATE_RESULTS_ENABLE and os.environ.get("ASD_PROCESS_TAP_ENABLE", "0") == "1"
PROCESS_TAP_STAGE_NAME = os.environ.get("ASD_PROCESS_TAP_STAGE_NAME", "talknet")
PROCESS_TAP_ROOT = os.environ.get("ASD_PROCESS_TAP_ROOT", os.path.join(ROOT, "runs", "process_tap"))
PROCESS_TAP_RUN_NAME = os.environ.get("ASD_PROCESS_TAP_RUN_NAME", "")


# TalkNet TensorRT
TALKNET_USE_TRT = _env_bool("ASD_TALKNET_USE_TRT", True)
TALKNET_TRT_FALLBACK_TO_TORCH = _env_bool("ASD_TALKNET_TRT_FALLBACK_TO_TORCH", True)
TALKNET_TRT_KEEP_TORCH_FALLBACK = _env_bool("ASD_TALKNET_TRT_KEEP_TORCH_FALLBACK", False)
TALKNET_TRT_ENGINE = os.path.join(MODEL_DIR, "talknet_tv8_ta32_b1_8_fp16.engine")  # 按你的实际路径改
TALKNET_TRT_TV = 8
TALKNET_TRT_TA = 32
TALKNET_TRT_VERBOSE = False

TALKNET_PARALLEL_WORKERS = _env_int("ASD_TALKNET_PARALLEL_WORKERS", 2)
TALKNET_TRT_WARMUP_ENABLE = _env_bool("ASD_TALKNET_TRT_WARMUP_ENABLE", True)
TALKNET_RAISE_ON_INFER_ERROR = _env_bool("ASD_TALKNET_RAISE_ON_INFER_ERROR", True)

# ==========================
# TalkNet TensorRT (TRT)
# ==========================
# 你现在写的是 TALKNET_TRT_ENGINE，这里做一个兼容别名，避免代码读不到
TALKNET_USE_TRT = bool(globals().get("TALKNET_USE_TRT", True))
TALKNET_TRT_FALLBACK_TO_TORCH = bool(globals().get("TALKNET_TRT_FALLBACK_TO_TORCH", True))
TALKNET_TRT_KEEP_TORCH_FALLBACK = bool(globals().get("TALKNET_TRT_KEEP_TORCH_FALLBACK", False))

# 兼容：你可能用 TALKNET_TRT_ENGINE 或 TALKNET_TRT_ENGINE_PATH
TALKNET_TRT_ENGINE = globals().get("TALKNET_TRT_ENGINE", os.path.join(MODEL_DIR, "talknet_tv8_ta32_b1_8_fp16.engine"))
TALKNET_TRT_ENGINE_PATH = globals().get("TALKNET_TRT_ENGINE_PATH", TALKNET_TRT_ENGINE)

TALKNET_TRT_TV = int(globals().get("TALKNET_TRT_TV", 8))
TALKNET_TRT_TA = int(globals().get("TALKNET_TRT_TA", 32))
TALKNET_TRT_VERBOSE = bool(globals().get("TALKNET_TRT_VERBOSE", False))
TALKNET_TRT_WARMUP_ENABLE = bool(globals().get("TALKNET_TRT_WARMUP_ENABLE", True))

# ==========================
# 兼容历史写法：from config.asd_config import C
# 让 C.XXX 这种访问成立
# ==========================
from types import SimpleNamespace as _SimpleNamespace
C = _SimpleNamespace(**{k: v for k, v in globals().items() if k.isupper()})
