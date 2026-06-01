#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import importlib
import json
import multiprocessing as mp
import queue
import threading

# === 把工程根目录(realtime_asd_ros)加入 sys.path ===
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

ROS_PYTHON_PATHS = [
    os.path.join(ROOT_DIR, "audio", "catkin_ws", "devel", "lib", "python3", "dist-packages"),
    "/ros_noetic/catkin_ws/devel/lib/python3/dist-packages",
    "/ros_noetic/catkin_ws/install/lib/python3/dist-packages",
    "/opt/ros/noetic/lib/python3/dist-packages",
]
for _ros_python_path in ROS_PYTHON_PATHS + os.environ.get("PYTHONPATH", "").split(os.pathsep):
    if os.path.isdir(_ros_python_path) and _ros_python_path not in sys.path:
        sys.path.insert(0, _ros_python_path)

_pythonpath_parts = [p for p in sys.path if p and os.path.isdir(p)]
os.environ["PYTHONPATH"] = os.pathsep.join(
    dict.fromkeys(_pythonpath_parts + os.environ.get("PYTHONPATH", "").split(os.pathsep))
)

import rospy
import time
import numpy as np
from std_msgs.msg import String as RosString
from sensor_msgs.msg import Image, CameraInfo
from roslib.message import get_message_class
from rospy.msg import AnyMsg
import cv2

try:
    from cv_bridge import CvBridge  # type: ignore
except Exception:
    CvBridge = None


class _FallbackCvBridge:
    """Minimal Image->OpenCV converter for common ROS encodings when cv_bridge is unavailable."""

    @staticmethod
    def imgmsg_to_cv2(msg, desired_encoding="bgr8"):
        enc = (getattr(msg, "encoding", "") or "").lower()
        h = int(msg.height)
        w = int(msg.width)
        step = int(msg.step)
        raw = np.frombuffer(msg.data, dtype=np.uint8)

        if enc in ("bgr8", "rgb8"):
            row = raw.reshape((h, step))
            img = row[:, : w * 3].reshape((h, w, 3))
            if enc == "rgb8" and desired_encoding == "bgr8":
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return img

        if enc in ("mono8", "8uc1"):
            row = raw.reshape((h, step))
            gray = row[:, :w].reshape((h, w))
            if desired_encoding == "bgr8":
                return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            return gray

        raise RuntimeError(f"Unsupported ROS image encoding without cv_bridge: {enc}")

from config import asd_config as C
from asd.buffers import AudioBuffer, TrackBuffer
from asd.tracking import Tracker, iou as bbox_iou
from asd.ros_audio import extract_mono_audio, extract_raw_audio_packet, format_audio_meta
from asd import create_face_detector  # 使用工厂函数
import traceback
# 在文件开头,其他导入语句之后添加
from asd.active_speaker_utils import (
    determine_active_speakers,
    get_display_color,
    format_speaker_info
)

# SDK package path (source fallback when wheel is not installed in environment)
SDK_DIR = os.path.join(ROOT_DIR, "sdk")
if SDK_DIR not in sys.path:
    sys.path.insert(0, SDK_DIR)

from asd_sdk import ASDSDK, InferenceRequest, TrackInput

_PROCESS_TAP_FACTORY = None
_PROCESS_TAP_IMPORT_ERROR = None
_PROCESS_TAP_SRC = os.environ.get(
    "ASD_STREAM_CAPTURE_PLUGIN_SRC",
    "/home/naviai/Desktop/multimodal_process_tap/src",
)


def resolve_ros_message_class(type_name):
    """Resolve ROS message classes even when a package lacks rospack metadata."""
    type_name = (type_name or "").strip()
    if not type_name:
        return None

    if "/" not in type_name:
        return None
    package_name, message_name = type_name.split("/", 1)
    if not package_name or not message_name:
        return None

    candidate_paths = [
        os.path.join(ROOT_DIR, "audio", "catkin_ws", "devel", "lib", "python3", "dist-packages"),
        "/opt/ros/noetic/lib/python3/dist-packages",
    ]

    for attempt in range(2):
        audio_cls = get_message_class(type_name)
        if audio_cls is not None:
            return audio_cls

        try:
            module = importlib.import_module(f"{package_name}.msg")
            audio_cls = getattr(module, message_name, None)
            if audio_cls is not None and hasattr(audio_cls, "_type"):
                return audio_cls
        except Exception:
            pass

        if attempt == 0:
            for path in candidate_paths:
                if os.path.isdir(path) and path not in sys.path:
                    sys.path.insert(0, path)
    return None
try:
    from process_tap.integrations import maybe_create_asd_stream_capture as _PROCESS_TAP_FACTORY
except Exception as exc:
    _PROCESS_TAP_IMPORT_ERROR = exc
    if os.path.isdir(_PROCESS_TAP_SRC) and _PROCESS_TAP_SRC not in sys.path:
        sys.path.insert(0, _PROCESS_TAP_SRC)
    try:
        from process_tap.integrations import maybe_create_asd_stream_capture as _PROCESS_TAP_FACTORY
        _PROCESS_TAP_IMPORT_ERROR = None
    except Exception as exc2:
        _PROCESS_TAP_IMPORT_ERROR = exc2


def _audio_isolated_process_main(
    out_queue,
    audio_topic,
    audio_msg_type,
    fallback_channels,
    use_channel,
    target_sample_rate,
    batch_packets,
):
    import rospy as _rospy
    import threading as _threading

    _rospy.init_node("asd_audio_bridge", anonymous=True, disable_signals=True)
    audio_cls = resolve_ros_message_class(audio_msg_type)
    if audio_cls is None:
        out_queue.put(
            {
                "kind": "error",
                "message": f"Unknown audio message type in isolated process: {audio_msg_type}",
            }
        )
        return

    state = {
        "blocks": [],
        "first_t_raw": None,
        "pkt_count": 0,
        "sample_count": 0,
        "drop_count": 0,
        "invalid_count": 0,
        "empty_count": 0,
        "last_frame_count": None,
        "last_meta": None,
    }
    state_lock = _threading.Lock()

    def _flush():
        with state_lock:
            if not state["blocks"]:
                return
            blocks = list(state["blocks"])
            first_t_raw = state["first_t_raw"]
            last_meta = dict(state["last_meta"] or {})
            pkt_count = int(state["pkt_count"])
            sample_count = int(state["sample_count"])
            drop_count = int(state["drop_count"])
            invalid_count = int(state["invalid_count"])
            empty_count = int(state["empty_count"])
            state["blocks"] = []
            state["first_t_raw"] = None
            state["pkt_count"] = 0
            state["sample_count"] = 0
            state["drop_count"] = 0
            state["invalid_count"] = 0
            state["empty_count"] = 0

        mono = np.concatenate(blocks, axis=0).astype(np.float32, copy=False)
        payload = {
            "kind": "audio",
            "mono": mono,
            "t_raw": float(first_t_raw if first_t_raw is not None else _rospy.Time.now().to_sec()),
            "meta": last_meta,
            "pkt_count": pkt_count,
            "sample_count": sample_count,
            "drop_count": drop_count,
            "invalid_count": invalid_count,
            "empty_count": empty_count,
        }
        try:
            out_queue.put_nowait(payload)
        except Exception:
            pass

    def _cb(msg):
        is_valid = getattr(msg, "is_valid", True)
        if is_valid is False:
            with state_lock:
                state["invalid_count"] += 1
            return

        try:
            fc_raw = getattr(msg, "frame_count", None)
            cur_fc = int(fc_raw) if fc_raw is not None else None
        except Exception:
            cur_fc = None

        try:
            t_raw = msg.header.stamp.to_sec() if msg.header.stamp.to_nsec() > 0 else _rospy.Time.now().to_sec()
            mono, meta = extract_mono_audio(
                msg,
                fallback_channels=int(fallback_channels),
                use_channel=int(use_channel),
                target_sample_rate=int(target_sample_rate),
            )
        except Exception:
            with state_lock:
                state["empty_count"] += 1
            return

        should_flush = False
        with state_lock:
            if cur_fc is not None:
                last_fc = state["last_frame_count"]
                if last_fc is not None and cur_fc > (last_fc + 1):
                    state["drop_count"] += cur_fc - last_fc - 1
                state["last_frame_count"] = cur_fc

            if state["first_t_raw"] is None:
                state["first_t_raw"] = t_raw
            state["blocks"].append(mono)
            state["last_meta"] = meta
            state["pkt_count"] += 1
            state["sample_count"] += int(mono.shape[0])
            should_flush = state["pkt_count"] >= int(batch_packets)

        if should_flush:
            _flush()

    _rospy.Subscriber(audio_topic, audio_cls, _cb, queue_size=20, buff_size=2**23)
    try:
        out_queue.put_nowait(
            {
                "kind": "status",
                "message": f"subscribed audio topic {audio_topic} with type {audio_msg_type}",
            }
        )
    except Exception:
        pass
    _rospy.Timer(_rospy.Duration(0.1), lambda _evt: _flush())
    _rospy.spin()


_LIP_AVAILABLE = False
if getattr(C, "ENABLE_LIP_MOTION", False):
    try:
        from asd.lip_landmark_wrapper import LipLandmarkDetector
        from asd.lip_motion_analyzer import LipMotionAnalyzer
        _LIP_AVAILABLE = True
    except Exception as e:
        print(f"[ASD] Lip module not available: {e}")
        _LIP_AVAILABLE = False

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", module="torch")

os.environ["PYTHONUNBUFFERED"] = "1"
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass


class RealtimeASDNode:
    def __init__(self):
        rospy.init_node("realtime_asd_node")

        # ROS参数
        self.video_topic = rospy.get_param("~video_topic", C.VIDEO_TOPIC)
        self.audio_topic = rospy.get_param("~audio_topic", C.AUDIO_TOPIC)
        self.audio_msg_type = rospy.get_param("~audio_msg_type", C.AUDIO_MSG_TYPE)
        self.audio_offset_sec = rospy.get_param("~audio_offset_sec", C.AUDIO_OFFSET_SEC)
        self.audio_auto_select_device = bool(
            rospy.get_param("~audio_auto_select_device", getattr(C, "AUDIO_AUTO_SELECT_DEVICE", False))
        )
        self.audio_device_name = str(
            rospy.get_param("~audio_device_name", getattr(C, "AUDIO_DEVICE_NAME", ""))
        ).strip()
        self.audio_select_device_service = str(
            rospy.get_param(
                "~audio_select_device_service",
                getattr(C, "AUDIO_SELECT_DEVICE_SERVICE", ""),
            )
        ).strip()
        self.audio_get_devices_service = str(
            rospy.get_param(
                "~audio_get_devices_service",
                getattr(C, "AUDIO_GET_DEVICES_SERVICE", ""),
            )
        ).strip()
        self.audio_service_wait_timeout_sec = float(
            rospy.get_param(
                "~audio_service_wait_timeout_sec",
                getattr(C, "AUDIO_SERVICE_WAIT_TIMEOUT_SEC", 5.0),
            )
        )
        self.audio_use_channel = int(rospy.get_param("~audio_use_channel", getattr(C, "AUDIO_USE_CHANNEL", 0)))
        self.audio_async_processing = bool(
            rospy.get_param("~audio_async_processing", getattr(C, "AUDIO_ASYNC_PROCESSING", True))
        )
        self.audio_async_queue_size = max(
            1,
            int(rospy.get_param("~audio_async_queue_size", getattr(C, "AUDIO_ASYNC_QUEUE_SIZE", 200))),
        )
        self.audio_isolated_process = bool(
            rospy.get_param("~audio_isolated_process", getattr(C, "AUDIO_ISOLATED_PROCESS", True))
        )
        self.audio_isolated_batch_packets = max(
            1,
            int(rospy.get_param("~audio_isolated_batch_packets", getattr(C, "AUDIO_ISOLATED_BATCH_PACKETS", 10))),
        )
        self.audio_isolated_queue_size = max(
            1,
            int(rospy.get_param("~audio_isolated_queue_size", getattr(C, "AUDIO_ISOLATED_QUEUE_SIZE", 50))),
        )
        
        # 人脸检测器类型(可通过ROS参数覆盖配置文件)
        detector_type = rospy.get_param("~face_detector_type", C.FACE_DETECTOR_TYPE)
        self.display_mode = rospy.get_param("~active_speaker_mode", C.ACTIVE_SPEAKER_DISPLAY_MODE)
        self.active_speaker_lock_enabled = bool(
            rospy.get_param("~active_speaker_lock_enabled", getattr(C, "ACTIVE_SPEAKER_LOCK_ENABLED", True))
        )
        self.active_speaker_lock_seconds = max(
            0.0,
            float(rospy.get_param("~active_speaker_lock_seconds", getattr(C, "ACTIVE_SPEAKER_LOCK_SECONDS", 5.0))),
        )
        self.active_speaker_lock_reid_iou = max(
            0.0,
            float(rospy.get_param("~active_speaker_lock_reid_iou", getattr(C, "ACTIVE_SPEAKER_LOCK_REID_IOU", 0.35))),
        )
        self.active_speaker_lock_reid_iou_margin = max(
            0.0,
            float(
                rospy.get_param(
                    "~active_speaker_lock_reid_iou_margin",
                    getattr(C, "ACTIVE_SPEAKER_LOCK_REID_IOU_MARGIN", 0.08),
                )
            ),
        )
        self.active_speaker_lock_reid_center_ratio = max(
            0.0,
            float(
                rospy.get_param(
                    "~active_speaker_lock_reid_center_ratio",
                    getattr(C, "ACTIVE_SPEAKER_LOCK_REID_CENTER_RATIO", 0.60),
                )
            ),
        )
        self.active_speaker_lock_reid_size_ratio_min = max(
            0.0,
            float(
                rospy.get_param(
                    "~active_speaker_lock_reid_size_ratio_min",
                    getattr(C, "ACTIVE_SPEAKER_LOCK_REID_SIZE_RATIO_MIN", 0.45),
                )
            ),
        )
        self.active_speaker_lock_reid_size_ratio_max = max(
            self.active_speaker_lock_reid_size_ratio_min,
            float(
                rospy.get_param(
                    "~active_speaker_lock_reid_size_ratio_max",
                    getattr(C, "ACTIVE_SPEAKER_LOCK_REID_SIZE_RATIO_MAX", 2.20),
                )
            ),
        )
        self.active_speaker_lock_refresh_score = max(
            0.0,
            float(
                rospy.get_param(
                    "~active_speaker_lock_refresh_score",
                    getattr(C, "ACTIVE_SPEAKER_LOCK_REFRESH_SCORE", getattr(C, "TOP_SCORER_MIN_SCORE", 0.35)),
                )
            ),
        )
        self.active_speaker_lock_missing_grace_sec = max(
            0.0,
            float(
                rospy.get_param(
                    "~active_speaker_lock_missing_grace_sec",
                    getattr(C, "ACTIVE_SPEAKER_LOCK_MISSING_GRACE_SEC", 0.80),
                )
            ),
        )
        self.active_speaker_spatial_priority_enabled = bool(
            rospy.get_param(
                "~active_speaker_spatial_priority_enabled",
                getattr(C, "ACTIVE_SPEAKER_SPATIAL_PRIORITY_ENABLED", True),
            )
        )
        self.active_speaker_center_wait_seconds = max(
            0.0,
            float(
                rospy.get_param(
                    "~active_speaker_center_wait_seconds",
                    getattr(C, "ACTIVE_SPEAKER_CENTER_WAIT_SECONDS", 3.0),
                )
            ),
        )
        self.active_speaker_center_core_box = self._parse_core_box_param(
            rospy.get_param(
                "~active_speaker_center_core_box",
                getattr(C, "ACTIVE_SPEAKER_CENTER_CORE_BOX", "0.35,0.10,0.30,0.80"),
            )
        )
        self.active_speaker_center_core_box_normalized = bool(
            rospy.get_param(
                "~active_speaker_center_core_box_normalized",
                getattr(C, "ACTIVE_SPEAKER_CENTER_CORE_BOX_NORMALIZED", True),
            )
        )
        self.speaker_position_enable = bool(
            rospy.get_param("~speaker_position_enable", getattr(C, "SPEAKER_POSITION_ENABLE", True))
        )
        self.speaker_position_use_depth = bool(
            rospy.get_param("~speaker_position_use_depth", getattr(C, "SPEAKER_POSITION_USE_DEPTH", True))
        )
        self.speaker_camera_info_topic = str(
            rospy.get_param(
                "~speaker_camera_info_topic",
                getattr(C, "SPEAKER_CAMERA_INFO_TOPIC", "/zj_humanoid/sensor/realsense_head/depth/camera_info"),
            )
        ).strip()
        self.speaker_depth_topic = str(
            rospy.get_param(
                "~speaker_depth_topic",
                getattr(C, "SPEAKER_DEPTH_TOPIC", "/zj_humanoid/sensor/realsense_head/aligned_depth_to_color/image_raw"),
            )
        ).strip()
        self.speaker_depth_min_m = max(
            0.0,
            float(rospy.get_param("~speaker_depth_min_m", getattr(C, "SPEAKER_DEPTH_MIN_M", 0.30))),
        )
        self.speaker_depth_max_m = max(
            self.speaker_depth_min_m,
            float(rospy.get_param("~speaker_depth_max_m", getattr(C, "SPEAKER_DEPTH_MAX_M", 4.00))),
        )
        self.speaker_depth_valid_ratio_min = max(
            0.0,
            min(
                1.0,
                float(
                    rospy.get_param(
                        "~speaker_depth_valid_ratio_min",
                        getattr(C, "SPEAKER_DEPTH_VALID_RATIO_MIN", 0.20),
                    )
                ),
            ),
        )
        self.speaker_depth_roi_scale = max(
            0.05,
            min(
                1.0,
                float(rospy.get_param("~speaker_depth_roi_scale", getattr(C, "SPEAKER_DEPTH_ROI_SCALE", 0.40))),
            ),
        )
        self.speaker_depth_max_age_sec = max(
            0.0,
            float(rospy.get_param("~speaker_depth_max_age_sec", getattr(C, "SPEAKER_DEPTH_MAX_AGE_SEC", 0.50))),
        )
        self.speaker_depth_uint16_scale = max(
            1e-9,
            float(rospy.get_param("~speaker_depth_uint16_scale", getattr(C, "SPEAKER_DEPTH_UINT16_SCALE", 0.001))),
        )
        self.speaker_camera_fallback = {
            "fx": float(rospy.get_param("~speaker_camera_fx", getattr(C, "SPEAKER_CAMERA_FX", 0.0))),
            "fy": float(rospy.get_param("~speaker_camera_fy", getattr(C, "SPEAKER_CAMERA_FY", 0.0))),
            "cx": float(rospy.get_param("~speaker_camera_cx", getattr(C, "SPEAKER_CAMERA_CX", 0.0))),
            "cy": float(rospy.get_param("~speaker_camera_cy", getattr(C, "SPEAKER_CAMERA_CY", 0.0))),
            "frame_id": str(
                rospy.get_param(
                    "~speaker_camera_frame_id",
                    getattr(C, "SPEAKER_CAMERA_FRAME_ID", "realsense_head_color_optical_frame"),
                )
            ),
        }
        if self.active_speaker_lock_seconds <= 0.0:
            self.active_speaker_lock_enabled = False
        self.debug_show_window = rospy.get_param("~debug_show_window", C.DEBUG_SHOW_WINDOW)
        self.capture_only = bool(rospy.get_param("~capture_only", getattr(C, "CAPTURE_ONLY_MODE", False)))
        if self.capture_only:
            self.debug_show_window = False
        self.debug_window_name = rospy.get_param("~debug_window_name", C.DEBUG_WINDOW_NAME)
        self.debug_show_fps = rospy.get_param("~debug_show_fps", C.DEBUG_SHOW_FPS)
        self.debug_show_scores = rospy.get_param("~debug_show_scores", C.DEBUG_SHOW_SCORES)
        self.intermediate_results_enable = bool(
            rospy.get_param("~intermediate_results_enable", getattr(C, "INTERMEDIATE_RESULTS_ENABLE", False))
        )
        self.detector_only = bool(rospy.get_param("~detector_only", getattr(C, "DETECTOR_ONLY_MODE", False)))
        self.detector_only_skip_audio = bool(
            rospy.get_param("~detector_only_skip_audio", getattr(C, "DETECTOR_ONLY_SKIP_AUDIO", True))
        )
        self.detector_synthetic_frame = bool(
            rospy.get_param("~detector_synthetic_frame", getattr(C, "DETECTOR_SYNTHETIC_FRAME", False))
        )
        self.display_env = os.environ.get("DISPLAY", "")
        self.talknet_frame_stride = max(
            1,
            int(rospy.get_param("~talknet_frame_stride", getattr(C, "TALKNET_FRAME_STRIDE", getattr(C, "FRAME_STRIDE", 1))))
        )
        self.lip_state_max_age_sec = float(rospy.get_param("~lip_state_max_age_sec", getattr(C, "LIP_STATE_MAX_AGE_SEC", 0.5)))
        self.lip_sync_adaptive = bool(rospy.get_param("~lip_sync_adaptive", getattr(C, "LIP_SYNC_ADAPTIVE", True)))
        self.lip_sync_target_multiplier = float(
            rospy.get_param("~lip_sync_target_multiplier", getattr(C, "LIP_SYNC_TARGET_MULTIPLIER", 1.0))
        )
        self.lip_sync_min_history_frames = max(
            3,
            int(rospy.get_param("~lip_sync_min_history_frames", getattr(C, "LIP_SYNC_MIN_HISTORY_FRAMES", 4)))
        )
        self.lip_sync_max_history_frames = max(
            self.lip_sync_min_history_frames,
            int(rospy.get_param("~lip_sync_max_history_frames", getattr(C, "LIP_SYNC_MAX_HISTORY_FRAMES", 18)))
        )
        self.lip_sync_min_age_sec = float(rospy.get_param("~lip_sync_min_age_sec", getattr(C, "LIP_SYNC_MIN_AGE_SEC", 0.10)))
        self.lip_sync_max_age_sec = float(rospy.get_param("~lip_sync_max_age_sec", getattr(C, "LIP_SYNC_MAX_AGE_SEC", 0.60)))
        self.lip_sync_age_margin_sec = float(
            rospy.get_param("~lip_sync_age_margin_sec", getattr(C, "LIP_SYNC_AGE_MARGIN_SEC", 0.05))
        )
        self._lip_dynamic_history_frames = int(getattr(C, "LIP_MOTION_HISTORY_FRAMES", 15))

        if CvBridge is not None:
            self.bridge = CvBridge()
        else:
            rospy.logwarn("[ASD] cv_bridge not found, using fallback image converter")
            self.bridge = _FallbackCvBridge()
        self.audio_buf = AudioBuffer(
            max_sec=float(getattr(C, "AUDIO_BUFFER_SECONDS", C.CLIP_SECONDS * 4.0)),
            sample_rate=int(getattr(C, "AUDIO_SAMPLE_RATE", 16000)),
        )
        self.tracker = Tracker()
        self.track_buffers = {}
        self.asd_states = {}
        self.enable_lip_motion = bool(
            getattr(C, "ENABLE_LIP_MOTION", False)
            and _LIP_AVAILABLE
            and not self.capture_only
            and not self.detector_only
        )

        self.last_proc_frame_ts = None
        self.last_video_wall_ts = None
        self.last_audio_wall_ts = None
        self._proc_frame_count = 0
        self.active_speaker_lock_id = None
        self.active_speaker_lock_start_ts = None
        self.active_speaker_lock_last_active_ts = None
        self.active_speaker_lock_until_ts = None
        self.active_speaker_lock_last_bbox = None
        self.active_speaker_lock_last_visible_ts = None
        self.center_wait_started_ts = None
        self.center_wait_until_ts = None
        self.center_wait_side_candidates = []
        self._speaker_camera_info = None
        self._speaker_depth_image_m = None
        self._speaker_depth_stamp = None
        self._speaker_depth_frame_id = None
        self._speaker_depth_lock = threading.Lock()

        self.detector = None
        self.asd_sdk = None
        self.lip_landmark_detector = None
        self.lip_motion_analyzer = None

        if self.capture_only:
            rospy.loginfo("[ASD] capture-only mode: skip detector, lip motion, ASD SDK, and GUI")
        else:
            # 使用工厂函数创建检测器
            try:
                self.detector = create_face_detector(detector_type)
            except Exception as e:
                rospy.logfatal(f"[ASD] Failed to create face detector: {e}")
                raise

            # 唇部运动检测（闭嘴否决 + 融合得分）
            if self.enable_lip_motion:
                lip_weights = getattr(C, "LIP_LANDMARK_WEIGHTS_PATH", "")
                if lip_weights and os.path.isfile(lip_weights):
                    try:
                        self.lip_landmark_detector = LipLandmarkDetector(
                            weights_path=lip_weights,
                            device=str(getattr(C, "MODEL_DEVICE", "cuda")),
                            enable_debug=getattr(C, "LIP_MOTION_DEBUG", False),
                            use_trt=getattr(C, "LIP_USE_TRT", False),
                            trt_engine_path=getattr(C, "LIP_TRT_ENGINE_PATH", ""),
                        )
                        self.lip_motion_analyzer = LipMotionAnalyzer(
                            history_frames=getattr(C, "LIP_MOTION_HISTORY_FRAMES", 15),
                            lar_closed_threshold=getattr(C, "LIP_LAR_CLOSED_THRESHOLD", 0.15),
                            lar_open_threshold=getattr(C, "LIP_LAR_OPEN_THRESHOLD", 0.25),
                            motion_threshold=getattr(C, "LIP_MOTION_STD_THRESHOLD", 0.05),
                            consecutive_closed_for_penalty=getattr(C, "LIP_CONSECUTIVE_CLOSED_FOR_PENALTY", 8),
                            consecutive_no_detect_for_penalty=getattr(C, "LIP_CONSECUTIVE_NO_DETECT_FOR_PENALTY", 5),
                            enable_debug=getattr(C, "LIP_MOTION_DEBUG", False),
                        )
                        self._lip_dynamic_history_frames = int(getattr(C, "LIP_MOTION_HISTORY_FRAMES", 15))
                        rospy.loginfo("[ASD] Lip motion enabled. Weights: %s", lip_weights)
                    except Exception as e:
                        rospy.logwarn("[ASD] Failed to initialize lip motion detector: %s", e)
                        self.enable_lip_motion = False
                else:
                    rospy.logwarn("[ASD] Lip motion disabled: weights not found at %s", lip_weights)
                    self.enable_lip_motion = False

            if self.detector_only:
                self.enable_lip_motion = False
                rospy.logwarn("[ASD] detector-only mode: skip Lip updates, TalkNet, and ASD SDK")
            else:
                # 初始化 SDK（ASD 核心算法）
                # 注意：放在 lip 模块初始化之后，确保 enable_lip_motion 与实际可用性一致。
                self.asd_sdk = ASDSDK(
                    talknet_backend=ASDSDK.build_default_talknet_backend(),
                    mode=self.display_mode,
                    enable_lip_motion=self.enable_lip_motion,
                    smooth_alpha=float(getattr(C, "SMOOTH_ALPHA", 0.55)),
                    speak_on_thresh=float(getattr(C, "SPEAK_ON_THRESH", 0.52)),
                    speak_off_thresh=float(getattr(C, "SPEAK_OFF_THRESH", 0.38)),
                    lip_motion_weight=float(getattr(C, "LIP_MOTION_WEIGHT", 0.2)),
                    lip_veto_threshold=float(getattr(C, "LIP_VETO_THRESHOLD", -0.8)),
                    lip_talknet_confirm_thresh=float(getattr(C, "LIP_TALKNET_CONFIRM_THRESH", 0.50)),
                    threshold_min_score=float(getattr(C, "THRESHOLD_MODE_MIN_SCORE", 0.30)),
                    threshold_relative_margin=float(getattr(C, "THRESHOLD_MODE_RELATIVE_MARGIN", 0.12)),
                    top_scorer_min_score=float(getattr(C, "TOP_SCORER_MIN_SCORE", 0.40)),
                    top_scorer_override_threshold=float(getattr(C, "TOP_SCORER_OVERRIDE_THRESHOLD", 0.70)),
                )

        # 可视化状态
        self._last_vis_time = None
        self._talknet_calls = 0
        self._lip_calls = 0
        self._yolo_calls = 0
        self._talknet_fps = 0.0
        self._lip_fps = 0.0
        self._yolo_fps = 0.0
        self._talknet_ms_ema = 0.0
        self._lip_ms_ema = 0.0
        self._yolo_ms_ema = 0.0
        self._talknet_proc_fps = 0.0
        self._lip_proc_fps = 0.0
        self._yolo_proc_fps = 0.0
        self._fps_stat_last_ts = time.time()
        self._audio_last_frame_count = None
        self._audio_drop_total = 0
        self._audio_invalid_total = 0
        self._audio_empty_total = 0
        self._audio_queue_drop_total = 0
        self._audio_pkt_total = 0
        self._audio_sample_total = 0
        self._audio_monitor_last_ts = time.time()
        self._audio_monitor_last_pkt_total = 0
        self._audio_monitor_last_sample_total = 0
        self._audio_monitor_last_drop_total = 0
        self._audio_monitor_last_invalid_total = 0
        self._audio_monitor_last_empty_total = 0
        self._audio_monitor_last_queue_drop_total = 0
        self._last_audio_meta_desc = None
        self.stream_capture = None
        self._audio_queue = None
        self._audio_worker_stop = False
        self._audio_worker_thread = None
        self._audio_process_queue = None
        self._audio_process = None
        self._use_isolated_audio = bool(
            self.audio_isolated_process and not (self.detector_only and self.detector_only_skip_audio)
        )
        if self._use_isolated_audio:
            ctx = mp.get_context("spawn")
            self._audio_process_queue = ctx.Queue(maxsize=self.audio_isolated_queue_size)
            self._audio_process = ctx.Process(
                target=_audio_isolated_process_main,
                args=(
                    self._audio_process_queue,
                    self.audio_topic,
                    str(self.audio_msg_type),
                    int(getattr(C, "AUDIO_CHANNELS", 1)),
                    int(self.audio_use_channel),
                    int(getattr(C, "AUDIO_SAMPLE_RATE", 16000)),
                    int(self.audio_isolated_batch_packets),
                ),
                daemon=True,
            )
            self._audio_process.start()
            rospy.Timer(rospy.Duration(0.05), self._drain_audio_process_queue)
        elif self.audio_async_processing and not (self.detector_only and self.detector_only_skip_audio):
            self._audio_queue = queue.Queue(maxsize=self.audio_async_queue_size)
            self._audio_worker_thread = threading.Thread(
                target=self._audio_worker_loop,
                name="asd_audio_worker",
                daemon=True,
            )
            self._audio_worker_thread.start()

        # 自动探测音频消息类型
        if self._use_isolated_audio:
            AudioMsg = None
        else:
            AudioMsg = self._resolve_audio_msg_class()
            self.audio_msg_type = AudioMsg

        # 订阅
        self.sub_video = None
        if not self.capture_only:
            self.sub_video = rospy.Subscriber(
                self.video_topic, Image, self.image_cb,
                queue_size=2, buff_size=2**24
            )
        self.sub_speaker_camera_info = None
        self.sub_speaker_depth = None
        if self.speaker_position_enable and not self.capture_only:
            if self.speaker_camera_info_topic:
                self.sub_speaker_camera_info = rospy.Subscriber(
                    self.speaker_camera_info_topic,
                    CameraInfo,
                    self.speaker_camera_info_cb,
                    queue_size=1,
                    buff_size=2**20,
                )
            if self.speaker_position_use_depth and self.speaker_depth_topic:
                self.sub_speaker_depth = rospy.Subscriber(
                    self.speaker_depth_topic,
                    Image,
                    self.speaker_depth_cb,
                    queue_size=1,
                    buff_size=2**24,
                )
        self.sub_video_capture = None
        self.sub_audio = None
        self._audio_probe_timer = None
        if self.detector_only and self.detector_only_skip_audio:
            rospy.logwarn("[ASD] detector-only mode: skip audio subscriber")
        elif self._use_isolated_audio:
            rospy.loginfo("[ASD] isolated audio process starting for %s", self.audio_topic)
        elif AudioMsg is not None:
            self._create_audio_subscriber(AudioMsg)
        else:
            rospy.logwarn(
                "[ASD] audio msg type unresolved for topic %s; will retry in background. "
                "Set ~audio_msg_type to skip auto-detection.",
                self.audio_topic,
            )
            self._audio_probe_timer = rospy.Timer(rospy.Duration(2.0), self._retry_audio_subscribe)
        if not (self.detector_only and self.detector_only_skip_audio):
            self._maybe_select_microphone_device()

        # 输出
        self.pub_result = rospy.Publisher(
            "asd/active_speakers", RosString, queue_size=10
        )
        self.pub_main_speaker_pose = rospy.Publisher(
            "asd/main_speaker_pose", RosString, queue_size=10
        )

        rospy.Timer(rospy.Duration(5.0), self.watchdog_tick)
        rospy.loginfo("[ASD] realtime_asd_node started."
        f"Display mode: {self.display_mode}")
        rospy.loginfo(
            "[ASD] visualization: show_window=%s window_name=%s DISPLAY=%s",
            self.debug_show_window,
            self.debug_window_name,
            self.display_env or "<empty>",
        )
        rospy.loginfo("[ASD] capture_only=%s", self.capture_only)
        rospy.loginfo("[ASD] detector_only=%s", self.detector_only)
        rospy.loginfo("[ASD] detector_only_skip_audio=%s", self.detector_only_skip_audio)
        rospy.loginfo("[ASD] detector_synthetic_frame=%s", self.detector_synthetic_frame)
        rospy.loginfo("[ASD] intermediate_results_enable=%s", self.intermediate_results_enable)
        rospy.loginfo(
            "[ASD] cadence: talknet_frame_stride=%d, lip_state_max_age_sec=%.2f, lip_sync_adaptive=%s",
            self.talknet_frame_stride,
            self.lip_state_max_age_sec,
            self.lip_sync_adaptive,
        )
        rospy.loginfo(
            "[ASD] active speaker lock: enabled=%s seconds=%.2f reid_iou=%.2f "
            "reid_margin=%.2f center_ratio=%.2f size_ratio=[%.2f,%.2f] "
            "refresh_score=%.2f missing_grace=%.2f",
            self.active_speaker_lock_enabled,
            self.active_speaker_lock_seconds,
            self.active_speaker_lock_reid_iou,
            self.active_speaker_lock_reid_iou_margin,
            self.active_speaker_lock_reid_center_ratio,
            self.active_speaker_lock_reid_size_ratio_min,
            self.active_speaker_lock_reid_size_ratio_max,
            self.active_speaker_lock_refresh_score,
            self.active_speaker_lock_missing_grace_sec,
        )
        rospy.loginfo(
            "[ASD] active speaker spatial priority: enabled=%s center_wait=%.2f "
            "core_box=%s normalized=%s",
            self.active_speaker_spatial_priority_enabled,
            self.active_speaker_center_wait_seconds,
            self.active_speaker_center_core_box,
            self.active_speaker_center_core_box_normalized,
        )
        rospy.loginfo(
            "[ASD] speaker position: enabled=%s use_depth=%s camera_info=%s depth=%s "
            "depth_range=[%.2f,%.2f] valid_ratio_min=%.2f roi_scale=%.2f max_age=%.2f",
            self.speaker_position_enable,
            self.speaker_position_use_depth,
            self.speaker_camera_info_topic or "<none>",
            self.speaker_depth_topic or "<none>",
            self.speaker_depth_min_m,
            self.speaker_depth_max_m,
            self.speaker_depth_valid_ratio_min,
            self.speaker_depth_roi_scale,
            self.speaker_depth_max_age_sec,
        )
        rospy.loginfo(
            "[ASD] audio buffer: seconds=%.2f sample_rate=%d",
            float(getattr(C, "AUDIO_BUFFER_SECONDS", C.CLIP_SECONDS * 4.0)),
            int(getattr(C, "AUDIO_SAMPLE_RATE", 16000)),
        )
        rospy.loginfo("[ASD] audio selected channel: %d", self.audio_use_channel)
        rospy.loginfo(
            "[ASD] audio async processing: enabled=%s queue_size=%d",
            self.audio_async_processing,
            self.audio_async_queue_size,
        )
        rospy.loginfo(
            "[ASD] audio isolated process: enabled=%s batch_packets=%d queue_size=%d",
            self._use_isolated_audio,
            self.audio_isolated_batch_packets,
            self.audio_isolated_queue_size,
        )

        self._audio_monitor_enable = bool(getattr(C, "AUDIO_MONITOR_ENABLE", True))
        self._audio_monitor_interval_sec = max(1.0, float(getattr(C, "AUDIO_MONITOR_INTERVAL_SEC", 5.0)))
        if self._audio_monitor_enable:
            rospy.Timer(rospy.Duration(self._audio_monitor_interval_sec), self.audio_monitor_tick)
            rospy.loginfo("[ASD] audio monitor enabled, interval=%.1fs", self._audio_monitor_interval_sec)

        stream_capture_requested = os.environ.get("ASD_STREAM_CAPTURE_ENABLE", "0") == "1"
        if self.intermediate_results_enable and _PROCESS_TAP_FACTORY is not None:
            try:
                self.stream_capture = _PROCESS_TAP_FACTORY(
                    default_root=os.path.join(ROOT_DIR, "runs", "stream_capture"),
                    video_fps=float(getattr(C, "VIDEO_PROCESS_FPS", 10.0)),
                    audio_sample_rate=int(getattr(C, "AUDIO_SAMPLE_RATE", 16000)),
                    info_logger=lambda msg: rospy.loginfo("%s", msg),
                    warn_logger=lambda msg: rospy.logwarn("%s", msg),
                )
            except Exception as e:
                rospy.logwarn("[ASD] failed to initialize stream capture: %s", e)
                self.stream_capture = None
        elif stream_capture_requested and not self.intermediate_results_enable:
            rospy.logwarn(
                "[ASD] stream capture requested but intermediate results are disabled. "
                "Set ASD_INTERMEDIATE_RESULTS_ENABLE=1 or ASD_RUNTIME_PROFILE=test to allow test artifacts."
            )
        elif stream_capture_requested:
            rospy.logwarn(
                "[ASD] stream capture requested but process_tap integration unavailable: %s",
                _PROCESS_TAP_IMPORT_ERROR,
            )
        if self.stream_capture is not None:
            self._create_input_capture_subscriber()

        if self.debug_show_window:
            try:
                cv2.namedWindow(self.debug_window_name, cv2.WINDOW_NORMAL)
            except Exception as e:
                rospy.logwarn("[ASD] failed to create debug window: %s", e)

    def _create_input_capture_subscriber(self):
        if self.sub_video_capture is not None:
            return
        qsize = int(rospy.get_param("~capture_video_sub_queue_size", 20))
        bsize = int(rospy.get_param("~capture_video_sub_buff_size", 2**24))
        self.sub_video_capture = rospy.Subscriber(
            self.video_topic, Image, self.input_capture_image_cb,
            queue_size=qsize, buff_size=bsize
        )
        rospy.loginfo(
            "[ASD] raw input capture subscribed on %s (queue=%d buff_size=%d)",
            self.video_topic,
            qsize,
            bsize,
        )

    def _resolve_audio_msg_class(self):
        if isinstance(self.audio_msg_type, str) and self.audio_msg_type.strip():
            audio_cls = resolve_ros_message_class(self.audio_msg_type.strip())
            if audio_cls is None:
                raise RuntimeError(f"Unknown audio message type: {self.audio_msg_type}")
            return audio_cls

        try:
            from rostopic import get_topic_class as _get_topic_class
            cand, _, _ = _get_topic_class(self.audio_topic, blocking=False)
            if hasattr(cand, "_type"):
                return cand
            if isinstance(cand, str):
                return resolve_ros_message_class(cand)
        except Exception as e:
            rospy.logwarn_throttle(5.0, "[ASD] failed to probe audio msg type: %s", e)
        return None

    def _create_audio_subscriber(self, audio_cls):
        qsize = int(rospy.get_param("~audio_sub_queue_size", getattr(C, "AUDIO_SUB_QUEUE_SIZE", 20)))
        bsize = int(rospy.get_param("~audio_sub_buff_size", getattr(C, "AUDIO_SUB_BUFF_SIZE", 2**23)))
        self.sub_audio = rospy.Subscriber(
            self.audio_topic, audio_cls, self.audio_cb,
            queue_size=qsize, buff_size=bsize
        )
        cls_name = getattr(audio_cls, "_type", str(audio_cls))
        rospy.loginfo(
            "[ASD] subscribed audio topic %s with type %s (queue=%d buff_size=%d)",
            self.audio_topic,
            cls_name,
            qsize,
            bsize,
        )

    def _maybe_select_microphone_device(self):
        if not (self.audio_auto_select_device and self.audio_device_name):
            return
        if not self.audio_select_device_service:
            rospy.logwarn("[ASD] audio auto-select enabled but no select_device service configured")
            return

        try:
            from audio.srv import SetDevice, SetDeviceRequest
        except Exception as e:
            rospy.logwarn("[ASD] failed to import audio.srv.SetDevice: %s", e)
            return

        try:
            rospy.loginfo(
                "[ASD] waiting for microphone select_device service %s (timeout=%.1fs)",
                self.audio_select_device_service,
                self.audio_service_wait_timeout_sec,
            )
            rospy.wait_for_service(
                self.audio_select_device_service,
                timeout=self.audio_service_wait_timeout_sec,
            )
            client = rospy.ServiceProxy(self.audio_select_device_service, SetDevice)
            req = SetDeviceRequest(name=self.audio_device_name)
            resp = client.call(req)
            if getattr(resp, "success", False):
                rospy.loginfo("[ASD] microphone device selected: %s", self.audio_device_name)
            else:
                rospy.logwarn(
                    "[ASD] microphone select_device failed: device=%s status=%s message=%s",
                    self.audio_device_name,
                    getattr(resp, "status", "<unknown>"),
                    getattr(resp, "message", ""),
                )
        except Exception as e:
            rospy.logwarn(
                "[ASD] failed to select microphone device '%s' via %s: %s",
                self.audio_device_name,
                self.audio_select_device_service,
                e,
            )

    def _retry_audio_subscribe(self, _evt):
        if self.sub_audio is not None:
            if self._audio_probe_timer is not None:
                self._audio_probe_timer.shutdown()
                self._audio_probe_timer = None
            return

        audio_cls = self._resolve_audio_msg_class()
        if audio_cls is None:
            return

        self.audio_msg_type = audio_cls
        self._create_audio_subscriber(audio_cls)
        if self._audio_probe_timer is not None:
            self._audio_probe_timer.shutdown()
            self._audio_probe_timer = None

    def _update_runtime_fps_stats(self):
        now = time.time()
        dt = now - self._fps_stat_last_ts
        if dt < 0.5:
            return

        talk_inst = self._talknet_calls / max(dt, 1e-6)
        lip_inst = self._lip_calls / max(dt, 1e-6)
        yolo_inst = self._yolo_calls / max(dt, 1e-6)
        alpha = 0.3
        if self._talknet_fps <= 0:
            self._talknet_fps = talk_inst
        else:
            self._talknet_fps = alpha * talk_inst + (1 - alpha) * self._talknet_fps

        if self._lip_fps <= 0:
            self._lip_fps = lip_inst
        else:
            self._lip_fps = alpha * lip_inst + (1 - alpha) * self._lip_fps

        if self._yolo_fps <= 0:
            self._yolo_fps = yolo_inst
        else:
            self._yolo_fps = alpha * yolo_inst + (1 - alpha) * self._yolo_fps

        self._talknet_calls = 0
        self._lip_calls = 0
        self._yolo_calls = 0
        self._fps_stat_last_ts = now

    def _maybe_record_audio_chunk(self, audio_block: np.ndarray, t_start: float, meta: dict):
        if self.stream_capture is None:
            return
        try:
            self.stream_capture.record_audio_chunk(
                audio_block,
                t_start,
                sample_rate=int(meta.get("sample_rate") or getattr(C, "AUDIO_SAMPLE_RATE", 16000)),
                channels=1,
                bits_per_sample=int(meta.get("bits_per_sample") or 16),
                field_name="asd_input_audio",
                is_raw_source=False,
            )
        except Exception as e:
            rospy.logwarn_throttle(5.0, "[ASD] stream capture audio error: %s", e)

    def _maybe_record_raw_audio_chunk(self, audio_block: np.ndarray, t_start: float, meta: dict):
        if self.stream_capture is None:
            return
        try:
            self.stream_capture.record_raw_audio_chunk(
                audio_block,
                t_start,
                sample_rate=int(meta.get("sample_rate") or getattr(C, "AUDIO_RAW_SAMPLE_RATE", 0) or getattr(C, "AUDIO_SAMPLE_RATE", 16000)),
                channels=int(meta.get("channels") or getattr(C, "AUDIO_CHANNELS", 1)),
                bits_per_sample=int(meta.get("bits_per_sample") or 16),
                field_name=str(meta.get("field_name") or "raw_topic_audio"),
                is_raw_source=bool(meta.get("is_raw_source", True)),
            )
        except Exception as e:
            rospy.logwarn_throttle(5.0, "[ASD] stream capture raw audio error: %s", e)

    def _maybe_record_input_frame(self, frame_bgr, timestamp: float):
        if self.stream_capture is None:
            return
        try:
            self.stream_capture.record_input_frame(frame_bgr, timestamp)
        except Exception as e:
            rospy.logwarn_throttle(5.0, "[ASD] stream capture input frame error: %s", e)

    def speaker_camera_info_cb(self, msg):
        try:
            k = list(getattr(msg, "K", []) or [])
            if len(k) < 6:
                return
            fx = float(k[0])
            fy = float(k[4])
            cx = float(k[2])
            cy = float(k[5])
            if fx <= 0.0 or fy <= 0.0:
                return
            self._speaker_camera_info = {
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "width": int(getattr(msg, "width", 0) or 0),
                "height": int(getattr(msg, "height", 0) or 0),
                "frame_id": str(getattr(getattr(msg, "header", None), "frame_id", "") or self.speaker_camera_fallback["frame_id"]),
                "stamp": float(msg.header.stamp.to_sec()) if getattr(msg, "header", None) is not None else None,
            }
            if self.sub_speaker_camera_info is not None:
                try:
                    self.sub_speaker_camera_info.unregister()
                    self.sub_speaker_camera_info = None
                    rospy.loginfo(
                        "[ASD] speaker camera_info latched: fx=%.3f fy=%.3f cx=%.3f cy=%.3f frame=%s",
                        fx,
                        fy,
                        cx,
                        cy,
                        self._speaker_camera_info["frame_id"],
                    )
                except Exception:
                    pass
        except Exception as e:
            rospy.logwarn_throttle(5.0, "[ASD] speaker camera_info error: %s", e)

    def _depth_msg_to_meters(self, msg):
        enc = (getattr(msg, "encoding", "") or "").lower()
        try:
            if CvBridge is not None:
                arr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                arr = np.asarray(arr)
            else:
                arr = None
        except Exception:
            arr = None

        if arr is None:
            h = int(msg.height)
            w = int(msg.width)
            step = int(msg.step)
            if enc in ("16uc1", "mono16", "uint16"):
                dtype = np.dtype(np.uint16)
            elif enc in ("32fc1", "float32"):
                dtype = np.dtype(np.float32)
            else:
                raise RuntimeError(f"Unsupported depth image encoding: {enc}")
            row_elems = max(1, step // dtype.itemsize)
            arr = np.frombuffer(msg.data, dtype=dtype).reshape((h, row_elems))[:, :w]

        if arr.ndim == 3:
            arr = arr[:, :, 0]
        if arr.dtype == np.uint16 or enc in ("16uc1", "mono16", "uint16"):
            return arr.astype(np.float32) * float(self.speaker_depth_uint16_scale)
        if arr.dtype == np.float32 or arr.dtype == np.float64 or enc in ("32fc1", "float32"):
            return arr.astype(np.float32, copy=False)
        return arr.astype(np.float32) * float(self.speaker_depth_uint16_scale)

    def speaker_depth_cb(self, msg):
        if not (self.speaker_position_enable and self.speaker_position_use_depth):
            return
        try:
            depth_m = self._depth_msg_to_meters(msg)
            stamp = msg.header.stamp.to_sec() if msg.header.stamp.to_nsec() > 0 else rospy.Time.now().to_sec()
            frame_id = str(getattr(msg.header, "frame_id", "") or "")
            with self._speaker_depth_lock:
                self._speaker_depth_image_m = depth_m
                self._speaker_depth_stamp = float(stamp)
                self._speaker_depth_frame_id = frame_id
        except Exception as e:
            rospy.logwarn_throttle(5.0, "[ASD] speaker depth image error: %s", e)

    def _speaker_camera_info_for_frame(self, frame_shape=None):
        info = self._speaker_camera_info
        if info is not None:
            return dict(info)

        fx = float(self.speaker_camera_fallback.get("fx", 0.0))
        fy = float(self.speaker_camera_fallback.get("fy", 0.0))
        cx = float(self.speaker_camera_fallback.get("cx", 0.0))
        cy = float(self.speaker_camera_fallback.get("cy", 0.0))
        if fx <= 0.0 or fy <= 0.0:
            return None
        width = int(frame_shape[1]) if frame_shape is not None else 0
        height = int(frame_shape[0]) if frame_shape is not None else 0
        return {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "width": width,
            "height": height,
            "frame_id": str(self.speaker_camera_fallback.get("frame_id", "")),
            "stamp": None,
        }

    @staticmethod
    def _bbox_center_uv(bbox):
        x1, y1, x2, y2 = [float(v) for v in bbox]
        return (x1 + x2) * 0.5, (y1 + y2) * 0.5

    def _estimate_depth_for_bbox(self, bbox, timestamp=None, frame_shape=None, projection=None):
        if not (self.speaker_position_enable and self.speaker_position_use_depth):
            return {
                "ok": False,
                "z_m": None,
                "valid_ratio": 0.0,
                "source": "disabled",
                "reason": "depth_disabled",
            }

        with self._speaker_depth_lock:
            depth = None if self._speaker_depth_image_m is None else self._speaker_depth_image_m.copy()
            depth_stamp = self._speaker_depth_stamp
            depth_frame_id = self._speaker_depth_frame_id

        if depth is None:
            return {
                "ok": False,
                "z_m": None,
                "valid_ratio": 0.0,
                "source": "realsense_depth",
                "frame_id": depth_frame_id,
                "reason": "no_depth_frame",
            }

        if timestamp is not None and depth_stamp is not None:
            age_sec = abs(float(timestamp) - float(depth_stamp))
            if self.speaker_depth_max_age_sec > 0.0 and age_sec > float(self.speaker_depth_max_age_sec):
                return {
                    "ok": False,
                    "z_m": None,
                    "valid_ratio": 0.0,
                    "source": "realsense_depth",
                    "frame_id": depth_frame_id,
                    "stamp": depth_stamp,
                    "age_sec": age_sec,
                    "reason": "stale_depth_frame",
                }

        h, w = depth.shape[:2]
        x1, y1, x2, y2 = [float(v) for v in bbox]
        if frame_shape is not None and frame_shape[0] > 0 and frame_shape[1] > 0:
            sx = float(w) / float(frame_shape[1])
            sy = float(h) / float(frame_shape[0])
            x1 *= sx
            x2 *= sx
            y1 *= sy
            y2 *= sy

        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        bw = max(1.0, (x2 - x1) * float(self.speaker_depth_roi_scale))
        bh = max(1.0, (y2 - y1) * float(self.speaker_depth_roi_scale))
        rx1 = int(max(0, min(w - 1, round(cx - bw * 0.5))))
        rx2 = int(max(0, min(w, round(cx + bw * 0.5))))
        ry1 = int(max(0, min(h - 1, round(cy - bh * 0.5))))
        ry2 = int(max(0, min(h, round(cy + bh * 0.5))))
        if rx2 <= rx1 or ry2 <= ry1:
            return {
                "ok": False,
                "z_m": None,
                "valid_ratio": 0.0,
                "source": "realsense_depth",
                "frame_id": depth_frame_id,
                "stamp": depth_stamp,
                "reason": "empty_depth_roi",
            }

        roi = depth[ry1:ry2, rx1:rx2]
        total = int(roi.size)
        valid_mask = (
            np.isfinite(roi)
            & (roi >= float(self.speaker_depth_min_m))
            & (roi <= float(self.speaker_depth_max_m))
        )
        valid = roi[valid_mask]
        valid_ratio = float(valid.size) / float(max(total, 1))
        if valid.size == 0 or valid_ratio < float(self.speaker_depth_valid_ratio_min):
            return {
                "ok": False,
                "z_m": None,
                "valid_ratio": valid_ratio,
                "source": "realsense_depth",
                "frame_id": depth_frame_id,
                "stamp": depth_stamp,
                "roi": [rx1, ry1, rx2, ry2],
                "reason": "low_valid_depth_ratio",
            }

        z_m = float(np.median(valid))
        result = {
            "ok": True,
            "z_m": z_m,
            "valid_ratio": valid_ratio,
            "source": "realsense_depth",
            "frame_id": depth_frame_id,
            "stamp": depth_stamp,
            "roi": [rx1, ry1, rx2, ry2],
            "reason": "",
        }

        if projection is not None:
            try:
                valid_ys, valid_xs = np.nonzero(valid_mask)
                point_u = (valid_xs.astype(np.float32) + float(rx1))
                point_v = (valid_ys.astype(np.float32) + float(ry1))
                if frame_shape is not None and frame_shape[0] > 0 and frame_shape[1] > 0:
                    point_u *= float(frame_shape[1]) / float(w)
                    point_v *= float(frame_shape[0]) / float(h)

                z_vals = roi[valid_mask].astype(np.float32, copy=False)
                fx = float(projection["fx"])
                fy = float(projection["fy"])
                cx_proj = float(projection["cx"])
                cy_proj = float(projection["cy"])
                x_vals = ((point_u - cx_proj) / fx) * z_vals
                y_vals = ((point_v - cy_proj) / fy) * z_vals
                result["position_camera"] = {
                    "x": float(np.median(x_vals)),
                    "y": float(np.median(y_vals)),
                    "z": float(np.median(z_vals)),
                    "frame_id": str(projection.get("frame_id", "")),
                    "source": "realsense_depth_roi_median3d",
                    "valid_ratio": valid_ratio,
                    "point_count": int(z_vals.size),
                    "depth_stamp": depth_stamp,
                }
            except Exception as e:
                result["position_error"] = str(e)

        return result

    def _build_speaker_geometry_for_track(self, tr, timestamp=None, frame_shape=None):
        if not self.speaker_position_enable or tr is None:
            return {
                "speaker_bearing_camera": None,
                "speaker_depth": {"ok": False, "reason": "position_disabled"},
                "speaker_position_camera": None,
            }

        camera = self._speaker_camera_info_for_frame(frame_shape)
        if camera is None:
            return {
                "speaker_bearing_camera": None,
                "speaker_depth": {"ok": False, "reason": "no_camera_info"},
                "speaker_position_camera": None,
            }

        u, v = self._bbox_center_uv(tr.bbox)
        fx = float(camera["fx"])
        fy = float(camera["fy"])
        cx = float(camera["cx"])
        cy = float(camera["cy"])
        camera_width = int(camera.get("width", 0) or 0)
        camera_height = int(camera.get("height", 0) or 0)
        source_width = int(frame_shape[1]) if frame_shape is not None and frame_shape[1] > 0 else camera_width
        source_height = int(frame_shape[0]) if frame_shape is not None and frame_shape[0] > 0 else camera_height
        raw_camera_info = {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "width": camera_width,
            "height": camera_height,
        }
        if (
            frame_shape is not None
            and frame_shape[0] > 0
            and frame_shape[1] > 0
            and camera_width > 0
            and camera_height > 0
            and (camera_width != frame_shape[1] or camera_height != frame_shape[0])
        ):
            sx = float(frame_shape[1]) / float(camera_width)
            sy = float(frame_shape[0]) / float(camera_height)
            fx *= sx
            cx *= sx
            fy *= sy
            cy *= sy
        ray_x = (float(u) - cx) / fx
        ray_y = (float(v) - cy) / fy
        norm = float(np.sqrt(ray_x * ray_x + ray_y * ray_y + 1.0))
        frame_id = str(camera.get("frame_id", "") or self.speaker_camera_fallback.get("frame_id", ""))
        bearing = {
            "u": float(u),
            "v": float(v),
            "yaw_rad": float(np.arctan2(ray_x, 1.0)),
            "pitch_rad": float(np.arctan2(ray_y, 1.0)),
            "ray_unit": [float(ray_x / norm), float(ray_y / norm), float(1.0 / norm)],
            "frame_id": frame_id,
            "camera_info": {
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "width": source_width,
                "height": source_height,
            },
            "raw_camera_info": raw_camera_info,
        }

        depth_info = self._estimate_depth_for_bbox(
            tr.bbox,
            timestamp=timestamp,
            frame_shape=frame_shape,
            projection={
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "frame_id": frame_id,
            },
        )
        position = depth_info.pop("position_camera", None)
        if position is None and bool(depth_info.get("ok", False)):
            z_m = float(depth_info["z_m"])
            position = {
                "x": float(ray_x * z_m),
                "y": float(ray_y * z_m),
                "z": z_m,
                "frame_id": frame_id,
                "source": "center_ray_depth_median_fallback",
                "valid_ratio": float(depth_info.get("valid_ratio", 0.0)),
                "depth_stamp": depth_info.get("stamp"),
            }

        return {
            "speaker_bearing_camera": bearing,
            "speaker_depth": depth_info,
            "speaker_position_camera": position,
        }

    def _release_active_speaker_lock(self, reason: str, timestamp: float):
        locked_id = self.active_speaker_lock_id
        if locked_id is None:
            return

        self.active_speaker_lock_id = None
        self.active_speaker_lock_start_ts = None
        self.active_speaker_lock_last_active_ts = None
        self.active_speaker_lock_until_ts = None
        self.active_speaker_lock_last_bbox = None
        self.active_speaker_lock_last_visible_ts = None
        self._clear_center_wait_state()
        self.asd_states.pop(locked_id, None)
        rospy.loginfo(
            "[ASD][LOCK] release speaker id=%s reason=%s ts=%.3f",
            locked_id,
            reason,
            timestamp,
        )

    def _acquire_active_speaker_lock(self, track_id: int, timestamp: float, bbox=None):
        if not self.active_speaker_lock_enabled:
            return

        track_id = int(track_id)
        self.active_speaker_lock_id = track_id
        self.active_speaker_lock_start_ts = float(timestamp)
        self.active_speaker_lock_last_active_ts = float(timestamp)
        self.active_speaker_lock_until_ts = float(timestamp) + float(self.active_speaker_lock_seconds)
        self.active_speaker_lock_last_visible_ts = float(timestamp)
        self._clear_center_wait_state()
        if bbox is not None:
            self.active_speaker_lock_last_bbox = tuple(int(v) for v in bbox)
        rospy.loginfo(
            "[ASD][LOCK] acquire speaker id=%d until=%.3f",
            track_id,
            self.active_speaker_lock_until_ts,
        )

    def _refresh_active_speaker_lock_activity(self, timestamp: float, score=None):
        if not (self.active_speaker_lock_enabled and self.active_speaker_lock_id is not None):
            return

        self.active_speaker_lock_last_active_ts = float(timestamp)
        self.active_speaker_lock_until_ts = float(timestamp) + float(self.active_speaker_lock_seconds)
        rospy.loginfo_throttle(
            1.0,
            "[ASD][LOCK] refresh speaker id=%s until=%.3f score=%s",
            self.active_speaker_lock_id,
            self.active_speaker_lock_until_ts,
            "n/a" if score is None else f"{float(score):.3f}",
        )

    def _mark_active_speaker_lock_visible(self, timestamp: float, bbox=None):
        if self.active_speaker_lock_id is None:
            return
        self.active_speaker_lock_last_visible_ts = float(timestamp)
        if bbox is not None:
            self.active_speaker_lock_last_bbox = tuple(int(v) for v in bbox)

    @staticmethod
    def _bbox_reid_metrics(old_bbox, new_bbox):
        old_x1, old_y1, old_x2, old_y2 = old_bbox
        new_x1, new_y1, new_x2, new_y2 = new_bbox

        old_w = max(1.0, float(old_x2 - old_x1))
        old_h = max(1.0, float(old_y2 - old_y1))
        new_w = max(1.0, float(new_x2 - new_x1))
        new_h = max(1.0, float(new_y2 - new_y1))

        old_cx = float(old_x1 + old_x2) * 0.5
        old_cy = float(old_y1 + old_y2) * 0.5
        new_cx = float(new_x1 + new_x2) * 0.5
        new_cy = float(new_y1 + new_y2) * 0.5

        old_diag = max(1.0, float(np.hypot(old_w, old_h)))
        center_ratio = float(np.hypot(new_cx - old_cx, new_cy - old_cy) / old_diag)
        size_ratio = float((new_w * new_h) / max(1.0, old_w * old_h))
        return center_ratio, size_ratio

    def _try_reassign_active_speaker_lock(self, tracks, timestamp: float) -> bool:
        if self.active_speaker_lock_last_bbox is None:
            return False

        visible_tracks = [tr for tr in tracks if int(getattr(tr, "missed", 0)) == 0]
        if not visible_tracks:
            return False

        candidates = []
        for tr in visible_tracks:
            cur_iou = float(bbox_iou(self.active_speaker_lock_last_bbox, tr.bbox))
            center_ratio, size_ratio = self._bbox_reid_metrics(self.active_speaker_lock_last_bbox, tr.bbox)
            if cur_iou < float(self.active_speaker_lock_reid_iou):
                continue
            if center_ratio > float(self.active_speaker_lock_reid_center_ratio):
                continue
            if not (
                float(self.active_speaker_lock_reid_size_ratio_min)
                <= size_ratio
                <= float(self.active_speaker_lock_reid_size_ratio_max)
            ):
                continue
            candidates.append((cur_iou, center_ratio, size_ratio, tr))

        if not candidates:
            return False

        candidates.sort(key=lambda item: (-item[0], item[1], int(item[3].track_id)))
        best_iou, best_center_ratio, best_size_ratio, best_track = candidates[0]
        if len(candidates) > 1:
            second_iou = float(candidates[1][0])
            if (best_iou - second_iou) < float(self.active_speaker_lock_reid_iou_margin):
                rospy.loginfo_throttle(
                    1.0,
                    "[ASD][LOCK] reassign rejected: ambiguous best_iou=%.3f second_iou=%.3f margin=%.3f",
                    best_iou,
                    second_iou,
                    self.active_speaker_lock_reid_iou_margin,
                )
                return False

        old_id = int(self.active_speaker_lock_id)
        new_id = int(best_track.track_id)
        if new_id == old_id:
            self._mark_active_speaker_lock_visible(timestamp, best_track.bbox)
            return True

        self.active_speaker_lock_id = new_id
        self._mark_active_speaker_lock_visible(timestamp, best_track.bbox)

        old_state = self.asd_states.pop(old_id, None)
        if old_state is not None:
            self.asd_states[new_id] = old_state
        old_buffer = self.track_buffers.pop(old_id, None)
        if old_buffer is not None:
            self.track_buffers[new_id] = old_buffer
        if self.asd_sdk is not None and hasattr(self.asd_sdk, "remap_track_id"):
            self.asd_sdk.remap_track_id(old_id, new_id)
        if self.lip_motion_analyzer is not None and hasattr(self.lip_motion_analyzer, "remap_track_id"):
            self.lip_motion_analyzer.remap_track_id(old_id, new_id)

        rospy.loginfo(
            "[ASD][LOCK] reassign speaker id=%d->%d iou=%.3f center=%.3f size=%.3f until=%.3f ts=%.3f",
            old_id,
            new_id,
            best_iou,
            best_center_ratio,
            best_size_ratio,
            float(self.active_speaker_lock_until_ts or 0.0),
            timestamp,
        )
        return True

    def _refresh_active_speaker_lock(self, timestamp: float, tracks, check_timeout: bool = True):
        if not (self.active_speaker_lock_enabled and self.active_speaker_lock_id is not None):
            return

        locked_id = int(self.active_speaker_lock_id)
        visible_by_id = {
            int(tr.track_id): tr
            for tr in tracks
            if int(getattr(tr, "missed", 0)) == 0
        }
        locked_track = visible_by_id.get(locked_id)
        if locked_track is None:
            if self._try_reassign_active_speaker_lock(tracks, timestamp):
                return
            last_visible = self.active_speaker_lock_last_visible_ts
            if last_visible is None:
                last_visible = self.active_speaker_lock_start_ts
            if last_visible is None:
                last_visible = timestamp
            missing_sec = max(0.0, float(timestamp) - float(last_visible))
            if missing_sec < float(self.active_speaker_lock_missing_grace_sec):
                rospy.loginfo_throttle(
                    1.0,
                    "[ASD][LOCK] keep missing speaker id=%d missing=%.3fs grace=%.3fs",
                    locked_id,
                    missing_sec,
                    self.active_speaker_lock_missing_grace_sec,
                )
                return
            self._release_active_speaker_lock("left_frame", timestamp)
            return
        self._mark_active_speaker_lock_visible(timestamp, locked_track.bbox)

        if (
            check_timeout
            and self.active_speaker_lock_until_ts is not None
            and timestamp >= float(self.active_speaker_lock_until_ts)
        ):
            self._release_active_speaker_lock("timeout", timestamp)

    def _tracks_for_active_speaker_inference(self, tracks):
        tracks = [tr for tr in tracks if int(getattr(tr, "missed", 0)) == 0]
        if not (self.active_speaker_lock_enabled and self.active_speaker_lock_id is not None):
            return tracks
        locked_id = int(self.active_speaker_lock_id)
        return [tr for tr in tracks if int(tr.track_id) == locked_id]

    def _determine_effective_active_speakers(self):
        if self.active_speaker_lock_enabled and self.active_speaker_lock_id is not None:
            locked_id = int(self.active_speaker_lock_id)
            state = self.asd_states.get(locked_id)
            if state and bool(state.get("is_spk", False)):
                return [locked_id]
            return []

        return determine_active_speakers(
            self.asd_states,
            mode=self.display_mode,
        )

    @staticmethod
    def _parse_core_box_param(value):
        default_box = (0.35, 0.10, 0.30, 0.80)
        try:
            if isinstance(value, str):
                cleaned = value.strip().strip("()[]")
                parts = [p.strip() for p in cleaned.split(",") if p.strip()]
            elif isinstance(value, (list, tuple)):
                parts = list(value)
            else:
                return default_box
            if len(parts) != 4:
                return default_box
            box = tuple(float(v) for v in parts)
            if box[2] <= 0.0 or box[3] <= 0.0:
                return default_box
            return box
        except Exception:
            return default_box

    def _get_center_core_box_px(self, frame_shape):
        if frame_shape is None:
            return None
        try:
            frame_h = float(frame_shape[0])
            frame_w = float(frame_shape[1])
            if frame_h <= 0.0 or frame_w <= 0.0:
                return None
            x, y, w, h = [float(v) for v in self.active_speaker_center_core_box]
            if self.active_speaker_center_core_box_normalized:
                x *= frame_w
                w *= frame_w
                y *= frame_h
                h *= frame_h
            x1 = max(0.0, min(frame_w, x))
            y1 = max(0.0, min(frame_h, y))
            x2 = max(0.0, min(frame_w, x + w))
            y2 = max(0.0, min(frame_h, y + h))
            if x2 <= x1 or y2 <= y1:
                return None
            return (x1, y1, x2, y2)
        except Exception:
            return None

    def _track_in_center_core(self, tr, frame_shape):
        core_box = self._get_center_core_box_px(frame_shape)
        if core_box is None:
            return False
        try:
            x1, y1, x2, y2 = [float(v) for v in tr.bbox]
            cx = (x1 + x2) * 0.5
            cy = (y1 + y2) * 0.5
            bx1, by1, bx2, by2 = core_box
            return bx1 <= cx <= bx2 and by1 <= cy <= by2
        except Exception:
            return False

    def _clear_center_wait_state(self):
        self.center_wait_started_ts = None
        self.center_wait_until_ts = None
        self.center_wait_side_candidates = []

    def _center_wait_candidate_ids(self):
        return {int(c.get("track_id")) for c in self.center_wait_side_candidates}

    def _add_center_wait_side_candidates(self, side_ids, valid_tracks, timestamp, selection_meta):
        track_by_id = {int(tr.track_id): tr for tr in valid_tracks}
        existing_ids = self._center_wait_candidate_ids()
        for tid in side_ids:
            tid = int(tid)
            if tid in existing_ids:
                continue
            tr = track_by_id.get(tid)
            if tr is None:
                continue
            meta = selection_meta.get(tid, {})
            self.center_wait_side_candidates.append(
                {
                    "track_id": tid,
                    "timestamp": float(timestamp),
                    "bbox": tuple(int(v) for v in tr.bbox),
                    "selection_score": float(meta.get("selection_score", 0.0)),
                }
            )
            existing_ids.add(tid)

    def _pick_center_wait_side_candidate(self, valid_tracks):
        visible_ids = {int(tr.track_id) for tr in valid_tracks}
        for candidate in sorted(self.center_wait_side_candidates, key=lambda c: float(c.get("timestamp", 0.0))):
            tid = int(candidate.get("track_id"))
            if tid in visible_ids:
                return tid
        return None

    def _select_active_speakers_by_fused(self, sdk_result, valid_tracks, frame_shape):
        track_by_id = {int(tr.track_id): tr for tr in valid_tracks}
        score_items = []
        score_meta = {}

        for tr_res in sdk_result.tracks:
            tid = int(tr_res.track_id)
            fused_score = float(tr_res.fused_score)
            if not np.isfinite(fused_score):
                fused_score = -float("inf")

            tr = track_by_id.get(tid)
            is_center_core = bool(tr is not None and self._track_in_center_core(tr, frame_shape))
            score_items.append((tid, fused_score, 1 if is_center_core else 0, bool(tr_res.is_lip_vetoed)))
            score_meta[tid] = {
                "selection_score": fused_score,
                "center_prior": 0.0,
                "center_proximity": 1.0 if is_center_core else 0.0,
                "is_center_core": is_center_core,
                "spatial_zone": "center_core" if is_center_core else "outer",
                "center_wait_candidate": False,
            }

        valid_items = [
            item for item in score_items
            if np.isfinite(float(item[1])) and not bool(item[3])
        ]
        if not valid_items:
            return [], score_meta

        effective_threshold = float(self.active_speaker_lock_refresh_score)
        effective_items = [
            item for item in valid_items
            if float(item[1]) >= effective_threshold
        ]
        if not effective_items:
            return [], score_meta

        sorted_items = sorted(effective_items, key=lambda item: (-item[2], -item[1], item[0]))
        return [int(item[0]) for item in sorted_items], score_meta

    def _apply_spatial_priority_to_unlocked_selection(self, active_ids, selection_meta, valid_tracks, frame_shape, timestamp):
        if not (
            self.active_speaker_spatial_priority_enabled
            and self.active_speaker_center_wait_seconds > 0.0
        ):
            self._clear_center_wait_state()
            return [int(tid) for tid in active_ids]

        track_by_id = {int(tr.track_id): tr for tr in valid_tracks}
        center_visible = any(self._track_in_center_core(tr, frame_shape) for tr in valid_tracks)
        center_active_ids = [
            int(tid) for tid in active_ids
            if bool(selection_meta.get(int(tid), {}).get("is_center_core", False))
        ]
        side_active_ids = [
            int(tid) for tid in active_ids
            if int(tid) in track_by_id and int(tid) not in set(center_active_ids)
        ]

        if center_active_ids:
            self._clear_center_wait_state()
            return center_active_ids

        if side_active_ids and (center_visible or self.center_wait_started_ts is not None):
            if self.center_wait_started_ts is None:
                self.center_wait_started_ts = float(timestamp)
                self.center_wait_until_ts = float(timestamp) + float(self.active_speaker_center_wait_seconds)
                rospy.loginfo(
                    "[ASD][CENTER_WAIT] start side_candidate=%s until=%.3f",
                    side_active_ids,
                    self.center_wait_until_ts,
                )
            self._add_center_wait_side_candidates(side_active_ids, valid_tracks, timestamp, selection_meta)

        if self.center_wait_started_ts is not None:
            for candidate_id in self._center_wait_candidate_ids():
                if candidate_id in selection_meta:
                    selection_meta[candidate_id]["center_wait_candidate"] = True

            if timestamp >= float(self.center_wait_until_ts or 0.0):
                winner_id = self._pick_center_wait_side_candidate(valid_tracks)
                self._clear_center_wait_state()
                if winner_id is not None:
                    rospy.loginfo("[ASD][CENTER_WAIT] expired, acquire side candidate id=%d", winner_id)
                    return [int(winner_id)]
                rospy.loginfo("[ASD][CENTER_WAIT] expired, no visible side candidate")
                return []
            return []

        if not center_visible:
            return side_active_ids
        return []

    def _build_visualization(self, frame_bgr, tracks):
        vis = frame_bgr.copy()
        active_speakers = self._determine_effective_active_speakers()
        locked_id = None
        if self.active_speaker_lock_enabled and self.active_speaker_lock_id is not None:
            locked_id = int(self.active_speaker_lock_id)
        wait_candidate_ids = set()
        if locked_id is None:
            wait_candidate_ids = self._center_wait_candidate_ids()
        self._last_vis_time = time.time()

        if self.active_speaker_spatial_priority_enabled:
            core_box = self._get_center_core_box_px(frame_bgr.shape[:2])
            if core_box is not None:
                bx1, by1, bx2, by2 = [int(round(v)) for v in core_box]
                core_color = (255, 0, 255)
                cv2.rectangle(vis, (bx1, by1), (bx2, by2), core_color, 2)
                label = "CENTER CORE"
                if self.center_wait_started_ts is not None:
                    label = "CENTER CORE WAIT"
                cv2.putText(
                    vis,
                    label,
                    (bx1, max(18, by1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    core_color,
                    2,
                    cv2.LINE_AA,
                )

        for tr in tracks:
            x1, y1, x2, y2 = tr.bbox
            state = self.asd_states.get(tr.track_id, {"p": 0.0, "is_spk": False})
            is_active = tr.track_id in active_speakers
            is_locked = locked_id is not None and int(tr.track_id) == locked_id
            is_wait_candidate = locked_id is None and int(tr.track_id) in wait_candidate_ids
            color = get_display_color(tr.track_id, active_speakers)
            if is_locked and not is_active:
                color = (0, 255, 255)
            elif is_wait_candidate:
                color = (255, 255, 0)

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            asd_score = float(state.get("p", 0.0))
            lip_score = float(state.get("lip_score", 0.0))
            fused_score_raw = float(state.get("fused_score", asd_score))
            selection_score = float(state.get("selection_score", fused_score_raw))
            center_prior = float(state.get("center_prior", 0.0))
            spatial_zone = str(state.get("spatial_zone", ""))
            is_vetoed = bool(state.get("is_lip_vetoed", False))
            speaker_position = state.get("speaker_position_camera")
            speaker_depth = state.get("speaker_depth") or {}

            if self.debug_show_scores:
                lip_display = lip_score if lip_score > -100 else -1.0
                fused_display = fused_score_raw if fused_score_raw > -100 else -1.0

                status = "[SPK]" if is_active else ("[LOCK]" if is_locked else ("[WAIT]" if is_wait_candidate else ""))
                line1 = f"ID:{tr.track_id} {status}"
                line2 = f"ASD:{asd_score:+.2f} LIP:{lip_display:+.2f}"
                if self.active_speaker_spatial_priority_enabled:
                    line3 = (
                        f"FUSED:{fused_display:+.2f} SEL:{selection_score:+.2f} "
                        f"{spatial_zone}{'[VETO]' if is_vetoed else ''}"
                    )
                else:
                    line3 = f"FUSED:{fused_display:+.2f}{'[VETO]' if is_vetoed else ''}"

                y_offset = max(0, y1 - 5)
                cv2.putText(vis, line1, (x1, y_offset - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
                cv2.putText(vis, line2, (x1, y_offset - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(vis, line3, (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
            else:
                if is_locked and not is_active:
                    text = f"id={tr.track_id} LOCK"
                elif is_wait_candidate:
                    text = f"id={tr.track_id} WAIT"
                else:
                    text = format_speaker_info(tr.track_id, asd_score, is_active)
                cv2.putText(
                    vis,
                    text,
                    (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )

            if is_active or is_locked:
                pos_text = None
                pos_color = (255, 255, 255)
                if isinstance(speaker_position, dict):
                    try:
                        pos_text = (
                            "CAM x:{:+.2f} y:{:+.2f} z:{:.2f}m".format(
                                float(speaker_position.get("x", 0.0)),
                                float(speaker_position.get("y", 0.0)),
                                float(speaker_position.get("z", 0.0)),
                            )
                        )
                    except Exception:
                        pos_text = None
                if pos_text is None:
                    reason = str(speaker_depth.get("reason", "no_position"))
                    pos_text = f"CAM {reason}"
                    pos_color = (0, 128, 255)

                text_y = min(vis.shape[0] - 8, max(16, int(y2) + 18))
                cv2.putText(
                    vis,
                    pos_text,
                    (int(x1), text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    pos_color,
                    1,
                    cv2.LINE_AA,
                )

        if self.debug_show_fps:
            info_text_1 = (
                f"procFPS Y/TN/LIP: {self._yolo_proc_fps:.1f}/{self._talknet_proc_fps:.1f}/{self._lip_proc_fps:.1f}"
            )
            info_text_2 = (
                f"ms Y/TN/LIP: {self._yolo_ms_ema:.1f}/{self._talknet_ms_ema:.1f}/{self._lip_ms_ema:.1f}"
            )
            cv2.putText(
                vis,
                info_text_1,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                vis,
                info_text_2,
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        return vis, active_speakers

    def _build_frame_result(self, t_now: float, tracks, active_speakers, frame_shape):
        active_set = {int(tid) for tid in active_speakers}
        core_box = self._get_center_core_box_px(frame_shape)
        tracks_msg = []
        for tr in tracks:
            state = self.asd_states.get(tr.track_id, {})
            fused_score = state.get("fused_score")
            if fused_score == -float("inf"):
                fused_score = None
            speaker_bearing = state.get("speaker_bearing_camera")
            speaker_depth = state.get("speaker_depth")
            speaker_position = state.get("speaker_position_camera")
            if self.speaker_position_enable and speaker_bearing is None:
                speaker_geometry = self._build_speaker_geometry_for_track(
                    tr,
                    timestamp=t_now,
                    frame_shape=frame_shape,
                )
                speaker_bearing = speaker_geometry.get("speaker_bearing_camera")
                speaker_depth = speaker_geometry.get("speaker_depth")
                speaker_position = speaker_geometry.get("speaker_position_camera")
            tracks_msg.append(
                {
                    "track_id": int(tr.track_id),
                    "bbox": [int(v) for v in tr.bbox],
                    "is_active": int(tr.track_id) in active_set,
                    "talknet_smooth": float(state.get("p", 0.0)),
                    "talknet_raw": float(state.get("p_raw", 0.0)),
                    "lip_score": float(state.get("lip_score", 0.0)),
                    "fused_score": None if fused_score is None else float(fused_score),
                    "selection_score": float(state.get("selection_score", state.get("fused_score", 0.0))),
                    "center_prior": float(state.get("center_prior", 0.0)),
                    "center_proximity": float(state.get("center_proximity", 0.0)),
                    "is_center_core": bool(state.get("is_center_core", False)),
                    "spatial_zone": str(state.get("spatial_zone", "")),
                    "center_wait_candidate": int(tr.track_id) in self._center_wait_candidate_ids(),
                    "speaker_bearing_camera": speaker_bearing,
                    "speaker_depth": speaker_depth,
                    "speaker_position_camera": speaker_position,
                    "is_lip_vetoed": bool(state.get("is_lip_vetoed", False)),
                }
            )

        return {
            "frame_index": int(self._proc_frame_count),
            "timestamp": float(t_now),
            "mode": self.display_mode,
            "video_topic": self.video_topic,
            "audio_topic": self.audio_topic,
            "frame_height": int(frame_shape[0]),
            "frame_width": int(frame_shape[1]),
            "active_speaker_ids": sorted(active_set),
            "center_core_box": None if core_box is None else [int(round(v)) for v in core_box],
            "center_wait": {
                "active": self.center_wait_started_ts is not None,
                "started": self.center_wait_started_ts,
                "until": self.center_wait_until_ts,
                "candidate_ids": sorted(self._center_wait_candidate_ids()),
            },
            "tracks": tracks_msg,
        }

    def _maybe_record_frame_capture(self, output_frame_bgr, timestamp: float, frame_result):
        if self.stream_capture is None:
            return
        try:
            self.stream_capture.record_output_frame(output_frame_bgr, timestamp, frame_result=frame_result)
        except Exception as e:
            rospy.logwarn_throttle(5.0, "[ASD] stream capture output frame error: %s", e)

    def _get_dynamic_lip_history_frames(self):
        fps = max(float(C.VIDEO_PROCESS_FPS), 1e-6)
        cadence_sec = float(self.talknet_frame_stride) / fps
        talknet_sec = max(self._talknet_ms_ema, 0.0) / 1000.0
        target_sec = max(cadence_sec, talknet_sec) * max(self.lip_sync_target_multiplier, 0.1)
        target_frames = int(round(target_sec * fps))
        return max(self.lip_sync_min_history_frames, min(self.lip_sync_max_history_frames, target_frames))

    def _get_dynamic_lip_state_max_age_sec(self):
        if not self.lip_sync_adaptive:
            return self.lip_state_max_age_sec

        fps = max(float(C.VIDEO_PROCESS_FPS), 1e-6)
        cadence_sec = float(self.talknet_frame_stride) / fps
        talknet_sec = max(self._talknet_ms_ema, 0.0) / 1000.0
        target_age = max(cadence_sec, talknet_sec) + max(self.lip_sync_age_margin_sec, 0.0)
        return max(self.lip_sync_min_age_sec, min(self.lip_sync_max_age_sec, target_age))

    def _sync_lip_history_window(self):
        if not (self.lip_sync_adaptive and self.lip_motion_analyzer):
            return

        target_frames = self._get_dynamic_lip_history_frames()
        if target_frames == self._lip_dynamic_history_frames:
            return

        if self.lip_motion_analyzer.set_history_frames(target_frames):
            self._lip_dynamic_history_frames = target_frames
            rospy.loginfo_throttle(
                5.0,
                "[ASD] adaptive lip history_frames=%d (talknet_ms_ema=%.1f)",
                target_frames,
                self._talknet_ms_ema,
            )

    def _update_lip_states(self, tracks, frame_bgr, timestamp):
        if not (self.enable_lip_motion and self.lip_landmark_detector and self.lip_motion_analyzer):
            return
        if not tracks:
            self.lip_motion_analyzer.cleanup_stale_tracks([])
            return

        try:
            self._sync_lip_history_window()
            bboxes = [tr.bbox for tr in tracks]
            tids = [tr.track_id for tr in tracks]
            t_lip0 = time.time()
            lip_results = self.lip_landmark_detector.detect(frame_bgr, bboxes, tids)
            lip_ms = (time.time() - t_lip0) * 1000.0
            if self._lip_ms_ema <= 0:
                self._lip_ms_ema = lip_ms
            else:
                self._lip_ms_ema = 0.3 * lip_ms + 0.7 * self._lip_ms_ema
            self._lip_proc_fps = 1000.0 / max(self._lip_ms_ema, 1e-6)
            self._lip_calls += 1

            for lip_result in lip_results:
                if lip_result.is_valid:
                    self.lip_motion_analyzer.update_track(
                        track_id=lip_result.track_id,
                        landmarks_98=lip_result.landmarks_98,
                        timestamp=timestamp,
                    )
                else:
                    self.lip_motion_analyzer.update_track(
                        track_id=lip_result.track_id,
                        landmarks_98=None,
                        timestamp=timestamp,
                    )

            self.lip_motion_analyzer.cleanup_stale_tracks(tids)
        except Exception as e:
            rospy.logwarn_throttle(5.0, "[ASD] Lip detection error: %s", e)

    # ---------------- Watchdog ----------------
    def watchdog_tick(self, _evt):
        now = time.time()
        if self.last_video_wall_ts is None or (now - self.last_video_wall_ts) > 5.0:
            rospy.logwarn_throttle(5.0, "[ASD] 未检测到视频数据... (%s)", self.video_topic)
        if self.last_audio_wall_ts is None or (now - self.last_audio_wall_ts) > 5.0:
            rospy.logwarn_throttle(5.0, "[ASD] 未检测到音频数据... (%s)", self.audio_topic)

    # ---------------- 音频回调 ----------------
    def _drain_audio_process_queue(self, _evt=None):
        if self._audio_process_queue is None:
            return

        throttle_sec = float(getattr(C, "AUDIO_DROP_LOG_THROTTLE_SEC", 2.0))
        while True:
            try:
                item = self._audio_process_queue.get_nowait()
            except queue.Empty:
                break
            except Exception:
                break

            if item.get("kind") == "error":
                rospy.logerr_throttle(5.0, "[ASD] isolated audio process error: %s", item.get("message"))
                continue
            if item.get("kind") == "status":
                rospy.loginfo("[ASD] isolated audio process: %s", item.get("message"))
                continue
            if item.get("kind") != "audio":
                continue

            mono = item.get("mono")
            if mono is None or getattr(mono, "size", 0) == 0:
                continue

            t_raw = float(item.get("t_raw", rospy.Time.now().to_sec()))
            meta = dict(item.get("meta") or {})

            self.last_audio_wall_ts = time.time()
            self._audio_pkt_total += int(item.get("pkt_count", 0))
            self._audio_sample_total += int(item.get("sample_count", int(mono.shape[0])))
            self._audio_drop_total += int(item.get("drop_count", 0))
            self._audio_invalid_total += int(item.get("invalid_count", 0))
            self._audio_empty_total += int(item.get("empty_count", 0))

            meta_desc = format_audio_meta(meta)
            if meta_desc != self._last_audio_meta_desc:
                self._last_audio_meta_desc = meta_desc
                rospy.loginfo("[ASD] audio stream format: %s", meta_desc)

            source_sample_rate = meta.get("source_sample_rate")
            expected_sample_rate = int(getattr(C, "AUDIO_SAMPLE_RATE", 16000))
            if source_sample_rate and source_sample_rate != expected_sample_rate:
                rospy.logwarn_throttle(
                    throttle_sec,
                    "[ASD] audio sample_rate mismatch handled by resampling: src=%d target=%d",
                    source_sample_rate,
                    expected_sample_rate,
                )

            self._maybe_record_audio_chunk(mono, t_raw, meta)
            self.audio_buf.add_block(mono, t_raw + self.audio_offset_sec)

    def audio_cb(self, msg):
        if not self.audio_async_processing or self._audio_queue is None:
            self._process_audio_msg(msg)
            return

        if isinstance(msg, AnyMsg):
            return
        self.last_audio_wall_ts = time.time()
        try:
            self._audio_queue.put_nowait(msg)
        except queue.Full:
            self._audio_queue_drop_total += 1
            rospy.logwarn_throttle(
                float(getattr(C, "AUDIO_DROP_LOG_THROTTLE_SEC", 2.0)),
                "[ASD] audio async queue full, packet dropped (queue_drop_total=%d)",
                self._audio_queue_drop_total,
            )

    def _audio_worker_loop(self):
        while not self._audio_worker_stop and not rospy.is_shutdown():
            try:
                msg = self._audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self._process_audio_msg(msg)
            finally:
                try:
                    self._audio_queue.task_done()
                except Exception:
                    pass

    def _process_audio_msg(self, msg):
        try:
            if isinstance(msg, AnyMsg):
                return
            self.last_audio_wall_ts = time.time()
            self._audio_pkt_total += 1

            throttle_sec = float(getattr(C, "AUDIO_DROP_LOG_THROTTLE_SEC", 2.0))
            monitor_drop = bool(getattr(C, "AUDIO_ENABLE_DROP_MONITOR", True))

            # 上游若显式标注无效音频，直接跳过并计数
            is_valid = getattr(msg, "is_valid", True)
            if is_valid is False:
                self._audio_invalid_total += 1
                rospy.logwarn_throttle(
                    throttle_sec,
                    "[ASD] invalid audio packet skipped (invalid_total=%d)",
                    self._audio_invalid_total,
                )
                return

            # 利用 frame_count 监控上游/传输层丢包
            if monitor_drop:
                cur_fc = None
                try:
                    fc_raw = getattr(msg, "frame_count", None)
                    cur_fc = int(fc_raw) if fc_raw is not None else None
                except Exception:
                    cur_fc = None

                if cur_fc is not None:
                    if self._audio_last_frame_count is not None and cur_fc > (self._audio_last_frame_count + 1):
                        missed = cur_fc - self._audio_last_frame_count - 1
                        self._audio_drop_total += missed
                        rospy.logwarn_throttle(
                            throttle_sec,
                            "[ASD] audio frame_count jump: prev=%d cur=%d missed=%d total_missed=%d",
                            self._audio_last_frame_count,
                            cur_fc,
                            missed,
                            self._audio_drop_total,
                        )
                    elif self._audio_last_frame_count is not None and cur_fc <= self._audio_last_frame_count:
                        rospy.logwarn_throttle(
                            throttle_sec,
                            "[ASD] audio frame_count non-increasing: prev=%d cur=%d (source reset or reorder)",
                            self._audio_last_frame_count,
                            cur_fc,
                        )
                    self._audio_last_frame_count = cur_fc

            t_raw = msg.header.stamp.to_sec() if msg.header.stamp.to_nsec() > 0 else rospy.Time.now().to_sec()

            if self.stream_capture is not None:
                try:
                    raw_audio, raw_meta = extract_raw_audio_packet(
                        msg,
                        fallback_channels=int(getattr(C, "AUDIO_CHANNELS", 1)),
                        fallback_sample_rate=getattr(C, "AUDIO_RAW_SAMPLE_RATE", None),
                    )
                    self._maybe_record_raw_audio_chunk(raw_audio, t_raw, raw_meta)
                except ValueError as e:
                    rospy.logwarn_throttle(
                        throttle_sec,
                        "[ASD] raw rostopic audio not recorded: %s",
                        e,
                    )

            try:
                mono, meta = extract_mono_audio(
                    msg,
                    fallback_channels=int(getattr(C, "AUDIO_CHANNELS", 1)),
                    use_channel=self.audio_use_channel,
                    target_sample_rate=int(getattr(C, "AUDIO_SAMPLE_RATE", 16000)),
                )
            except ValueError as e:
                self._audio_empty_total += 1
                rospy.logwarn_throttle(
                    throttle_sec,
                    "[ASD] audio packet skipped: %s (empty_total=%d)",
                    e,
                    self._audio_empty_total,
                )
                return

            meta_desc = format_audio_meta(meta)
            if meta_desc != self._last_audio_meta_desc:
                self._last_audio_meta_desc = meta_desc
                rospy.loginfo("[ASD] audio stream format: %s", meta_desc)

            sample_rate = meta.get("sample_rate")
            source_sample_rate = meta.get("source_sample_rate")
            expected_sample_rate = int(getattr(C, "AUDIO_SAMPLE_RATE", 16000))
            if source_sample_rate and source_sample_rate != expected_sample_rate:
                rospy.logwarn_throttle(
                    throttle_sec,
                    "[ASD] audio sample_rate mismatch handled by resampling: src=%d target=%d",
                    source_sample_rate,
                    expected_sample_rate,
                )
            elif sample_rate and sample_rate != expected_sample_rate:
                rospy.logwarn_throttle(
                    throttle_sec,
                    "[ASD] audio sample_rate mismatch: msg=%d expected=%d",
                    sample_rate,
                    expected_sample_rate,
                )

            self._audio_sample_total += int(mono.shape[0])
            t_start = t_raw + self.audio_offset_sec

            # Record the ASD input audio samples on the raw topic timeline so the
            # saved preview audio stays aligned with the camera stream. The
            # runtime ASD buffer still uses the configured offset for inference.
            self._maybe_record_audio_chunk(mono, t_raw, meta)
            self.audio_buf.add_block(mono, t_start)

        except Exception as e:
            rospy.logerr("[ASD audio_cb] %s", e)

    def audio_monitor_tick(self, _evt):
        now = time.time()
        dt = max(now - self._audio_monitor_last_ts, 1e-6)

        pkt_total = int(self._audio_pkt_total)
        sample_total = int(self._audio_sample_total)
        drop_total = int(self._audio_drop_total)
        invalid_total = int(self._audio_invalid_total)
        empty_total = int(self._audio_empty_total)
        queue_drop_total = int(self._audio_queue_drop_total)

        d_pkt = pkt_total - self._audio_monitor_last_pkt_total
        d_sample = sample_total - self._audio_monitor_last_sample_total
        d_drop = drop_total - self._audio_monitor_last_drop_total
        d_invalid = invalid_total - self._audio_monitor_last_invalid_total
        d_empty = empty_total - self._audio_monitor_last_empty_total
        d_queue_drop = queue_drop_total - self._audio_monitor_last_queue_drop_total

        pkt_rate = d_pkt / dt
        mono_sample_rate = d_sample / dt

        status = "OK" if d_drop == 0 and d_invalid == 0 and d_empty == 0 and d_queue_drop == 0 else "WARNING"
        log_fn = rospy.loginfo if status == "OK" else rospy.logwarn
        log_fn(
            "[ASD][AUDIO_MON][%s] dt=%.1fs pkt=%d(%.1f/s) mono_samples=%d(%.1f/s) "
            "drop=%d(total=%d) invalid=%d(total=%d) empty=%d(total=%d) queue_drop=%d(total=%d)",
            status,
            dt,
            d_pkt,
            pkt_rate,
            d_sample,
            mono_sample_rate,
            d_drop,
            drop_total,
            d_invalid,
            invalid_total,
            d_empty,
            empty_total,
            d_queue_drop,
            queue_drop_total,
        )

        self._audio_monitor_last_ts = now
        self._audio_monitor_last_pkt_total = pkt_total
        self._audio_monitor_last_sample_total = sample_total
        self._audio_monitor_last_drop_total = drop_total
        self._audio_monitor_last_invalid_total = invalid_total
        self._audio_monitor_last_empty_total = empty_total
        self._audio_monitor_last_queue_drop_total = queue_drop_total

    # ---------------- 视频回调 ----------------
    def image_cb(self, msg):
        try:
            self.last_video_wall_ts = time.time()
            
            t_frame = msg.header.stamp.to_sec() if msg.header.stamp.to_nsec() > 0 else rospy.Time.now().to_sec()

            # 控制逻辑 FPS(跳帧)
            if self.last_proc_frame_ts is not None:
                if (t_frame - self.last_proc_frame_ts) < (1.0 / C.VIDEO_PROCESS_FPS):
                    return
            self.last_proc_frame_ts = t_frame

            if self.detector_synthetic_frame:
                frame_bgr = np.zeros((int(msg.height), int(msg.width), 3), dtype=np.uint8)
            else:
                frame_bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            h, w = frame_bgr.shape[:2]

            # 1) 人脸检测(统一接口)
            t_yolo0 = time.time()
            dets = self.detector.detect(frame_bgr)
            yolo_ms = (time.time() - t_yolo0) * 1000.0
            if self._yolo_ms_ema <= 0:
                self._yolo_ms_ema = yolo_ms
            else:
                self._yolo_ms_ema = 0.3 * yolo_ms + 0.7 * self._yolo_ms_ema
            self._yolo_proc_fps = 1000.0 / max(self._yolo_ms_ema, 1e-6)
            self._yolo_calls += 1

            # 2) IOU tracking
            all_tracks = self.tracker.update(dets)
            tracks = [
                tr for tr in all_tracks
                if int(getattr(tr, "missed", 0)) == 0
            ]
            self._proc_frame_count += 1
            self._refresh_active_speaker_lock(t_frame, all_tracks, check_timeout=False)
            inference_tracks = self._tracks_for_active_speaker_inference(tracks)

            # 3) 更新每个 track 的 buffer
            for tr in tracks:
                x1, y1, x2, y2 = tr.bbox
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(w, x2); y2 = min(h, y2)
                face_roi = frame_bgr[y1:y2, x1:x2, :].copy()
                face_roi = cv2.resize(face_roi, (C.FACE_CROP_SIZE, C.FACE_CROP_SIZE))

                if tr.track_id not in self.track_buffers:
                    self.track_buffers[tr.track_id] = TrackBuffer()
                self.track_buffers[tr.track_id].add_frame(face_roi, t_frame)

            # 4) 每个处理帧都先更新 Lip 状态（用于更高时域分辨率）。
            # 锁定人短暂漏检的宽限期内不要清空 Lip 历史，否则 ID 恢复后会冷启动。
            if inference_tracks or not (
                self.active_speaker_lock_enabled and self.active_speaker_lock_id is not None
            ):
                self._update_lip_states(inference_tracks, frame_bgr, t_frame)

            # 5) TalkNet 按独立步长触发，和 Lip 解耦
            if len(inference_tracks) > 0 and (self._proc_frame_count % self.talknet_frame_stride == 0):
                self.try_run_asd(t_frame, inference_tracks, frame_bgr.shape[:2])
            else:
                self._refresh_active_speaker_lock(t_frame, all_tracks, check_timeout=True)

            # 每个处理帧都更新一次运行频率统计，避免统计只跟随 TalkNet 调用节拍。
            self._update_runtime_fps_stats()

            vis = None
            active_speakers = None
            if self.debug_show_window:
                vis, active_speakers = self._build_visualization(frame_bgr, tracks)

            if self.stream_capture is not None:
                if active_speakers is None:
                    active_speakers = self._determine_effective_active_speakers()
                frame_result = self._build_frame_result(t_frame, tracks, active_speakers, frame_bgr.shape[:2])
                self._maybe_record_frame_capture(None, t_frame, frame_result)

            # 6) 实时可视化
            if self.debug_show_window and vis is not None:
                try:
                    cv2.imshow(self.debug_window_name, vis)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        rospy.loginfo("[ASD] 'q' pressed, shutting down...")
                        rospy.signal_shutdown("user quit")
                except cv2.error as e:
                    self.debug_show_window = False
                    rospy.logwarn("[ASD] disabled debug window after OpenCV highgui error: %s", e)

        except Exception as e:
           tb = traceback.format_exc()
           rospy.logerr("[ASD image_cb] Exception caught:\n%s", tb)

    def input_capture_image_cb(self, msg):
        if self.stream_capture is None:
            return
        try:
            self.last_video_wall_ts = time.time()
            t_frame = msg.header.stamp.to_sec() if msg.header.stamp.to_nsec() > 0 else rospy.Time.now().to_sec()
            frame_bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self._maybe_record_input_frame(frame_bgr, t_frame)
        except Exception as e:
            rospy.logwarn_throttle(5.0, "[ASD] raw input capture error: %s", e)

    # ---------------- ASD 触发逻辑 ----------------
    def try_run_asd(self, t_now: float, tracks, frame_shape=None):
        """ASD inference aligned with source sequential behavior."""
        if self.asd_sdk is None:
            return
        if len(tracks) == 0:
            return

        self._refresh_active_speaker_lock(t_now, tracks, check_timeout=False)
        tracks = self._tracks_for_active_speaker_inference(tracks)
        if len(tracks) == 0:
            self._refresh_active_speaker_lock(t_now, tracks, check_timeout=True)
            return
        
        valid_tracks = []
        video_clips = []
        audio_clips = []
        
        for tr in tracks:
            buf = self.track_buffers.get(tr.track_id)
            if not buf or not buf.has_enough():
                continue
            
            video_clip, ts_list = buf.get_last_n(C.TARGET_VIDEO_FRAMES)
            if video_clip.size == 0 or len(ts_list) < C.MIN_VIDEO_FRAMES:
                continue

            t_audio_start = min(ts_list) - C.AUDIO_MARGIN_SEC
            t_audio_end = max(ts_list) + C.AUDIO_MARGIN_SEC
            audio_clip = self.audio_buf.get_range(t_audio_start, t_audio_end)
            if len(audio_clip) == 0:
                continue
            
            valid_tracks.append(tr)
            video_clips.append(video_clip)
            audio_clips.append(audio_clip)
        
        if len(valid_tracks) == 0:
            self._refresh_active_speaker_lock(t_now, tracks, check_timeout=True)
            return

        max_faces = int(getattr(C, "MAX_FACES_FOR_ASD", 0))
        if max_faces > 0 and len(valid_tracks) > max_faces:
            keep_indices = sorted(range(len(valid_tracks)), key=lambda i: valid_tracks[i].track_id)[:max_faces]
            keep_indices.sort()
            valid_tracks = [valid_tracks[i] for i in keep_indices]
            video_clips = [video_clips[i] for i in keep_indices]
            audio_clips = [audio_clips[i] for i in keep_indices]

        # 收集每个 track 的 lip_score（由上游 lip 模块计算）
        lip_scores = {}
        lip_state_max_age = self._get_dynamic_lip_state_max_age_sec()
        if self.enable_lip_motion and self.lip_motion_analyzer:
            for tr in valid_tracks:
                state = self.lip_motion_analyzer.get_state(tr.track_id)
                if state is None or len(state.timestamps) == 0:
                    # 启用 lip 融合时，缺失值采用轻惩罚，避免 SDK 入参缺失
                    lip_scores[tr.track_id] = -0.5
                    continue
                latest_ts = float(state.timestamps[-1])
                if (t_now - latest_ts) > lip_state_max_age:
                    lip_scores[tr.track_id] = -0.5
                    continue
                lip_scores[tr.track_id] = float(state.lip_score)

        req_tracks = []
        for tr, a_clip, v_clip in zip(valid_tracks, audio_clips, video_clips):
            req_tracks.append(
                TrackInput(
                    track_id=int(tr.track_id),
                    audio_clip=a_clip,
                    video_clip=v_clip,
                    lip_score=lip_scores.get(tr.track_id) if self.enable_lip_motion else None,
                )
            )

        t0 = time.time()
        sdk_result = self.asd_sdk.infer(
            InferenceRequest(
                tracks=req_tracks,
                timestamp=t_now,
            )
        )
        talknet_ms = (time.time() - t0) * 1000.0
        if self._talknet_ms_ema <= 0:
            self._talknet_ms_ema = talknet_ms
        else:
            self._talknet_ms_ema = 0.3 * talknet_ms + 0.7 * self._talknet_ms_ema
        self._talknet_proc_fps = 1000.0 / max(self._talknet_ms_ema, 1e-6)
        self._talknet_calls += 1

        debug_rows = []

        sdk_active_ids, selection_meta = self._select_active_speakers_by_fused(
            sdk_result,
            valid_tracks,
            frame_shape,
        )
        valid_track_by_id = {int(tr.track_id): tr for tr in valid_tracks}
        sdk_results_by_id = {int(tr.track_id): tr for tr in sdk_result.tracks}
        locked_id = self.active_speaker_lock_id
        if self.active_speaker_lock_enabled and locked_id is not None:
            locked_id = int(locked_id)
            locked_result = sdk_results_by_id.get(locked_id)
            locked_score = None
            locked_has_speech = locked_id in sdk_active_ids
            if locked_result is not None:
                fused_score = float(locked_result.fused_score)
                fused_score = fused_score if np.isfinite(fused_score) else 0.0
                locked_score = fused_score

            if locked_has_speech:
                active_set = {locked_id}
                self._refresh_active_speaker_lock_activity(t_now, score=locked_score)
            else:
                active_set = set()
        elif self.active_speaker_lock_enabled and sdk_active_ids:
            selected_ids = self._apply_spatial_priority_to_unlocked_selection(
                sdk_active_ids,
                selection_meta,
                valid_tracks,
                frame_shape,
                t_now,
            )
            if selected_ids:
                first_id = int(selected_ids[0])
                first_bbox = None
                for tr in valid_tracks:
                    if int(tr.track_id) == first_id:
                        first_bbox = tr.bbox
                        break
                self._acquire_active_speaker_lock(first_id, t_now, bbox=first_bbox)
                active_set = {first_id}
            else:
                active_set = set()
        else:
            selected_ids = []
            if self.active_speaker_lock_enabled and self.center_wait_started_ts is not None:
                selected_ids = self._apply_spatial_priority_to_unlocked_selection(
                    [],
                    selection_meta,
                    valid_tracks,
                    frame_shape,
                    t_now,
                )
            if selected_ids:
                first_id = int(selected_ids[0])
                first_bbox = None
                for tr in valid_tracks:
                    if int(tr.track_id) == first_id:
                        first_bbox = tr.bbox
                        break
                self._acquire_active_speaker_lock(first_id, t_now, bbox=first_bbox)
                active_set = {first_id}
            else:
                active_set = set(sdk_active_ids)

        for tr_res in sdk_result.tracks:
            tid = int(tr_res.track_id)
            prev = self.asd_states.get(tid, {})
            hist = list(prev.get("history", []))
            hist.append(float(tr_res.talknet_raw))
            if len(hist) > 5:
                hist = hist[-5:]

            tr_selection_meta = selection_meta.get(tid, {})
            selection_score = float(tr_selection_meta.get("selection_score", tr_res.fused_score))
            center_prior = float(tr_selection_meta.get("center_prior", 0.0))
            center_proximity = float(tr_selection_meta.get("center_proximity", 0.0))
            is_center_core = bool(tr_selection_meta.get("is_center_core", False))
            spatial_zone = str(tr_selection_meta.get("spatial_zone", "center_core" if is_center_core else "outer"))
            center_wait_candidate = bool(tr_selection_meta.get("center_wait_candidate", False))
            speaker_geometry = self._build_speaker_geometry_for_track(
                valid_track_by_id.get(tid),
                timestamp=t_now,
                frame_shape=frame_shape,
            )

            self.asd_states[tid] = {
                "p": float(tr_res.talknet_smooth),
                "p_raw": float(tr_res.talknet_raw),
                "is_spk": tid in active_set,
                "history": hist,
                "fused_score": float(tr_res.fused_score),
                "selection_score": selection_score,
                "center_prior": center_prior,
                "center_proximity": center_proximity,
                "is_center_core": is_center_core,
                "spatial_zone": spatial_zone,
                "center_wait_candidate": center_wait_candidate,
                "speaker_bearing_camera": speaker_geometry.get("speaker_bearing_camera"),
                "speaker_depth": speaker_geometry.get("speaker_depth"),
                "speaker_position_camera": speaker_geometry.get("speaker_position_camera"),
                "lip_score": float(tr_res.lip_score),
                "is_lip_vetoed": bool(tr_res.is_lip_vetoed),
            }

            debug_rows.append(
                "id=%d raw=%.3f smooth=%.3f lip=%.3f fused=%s sel=%s zone=%s wait=%s veto=%s"
                % (
                    tid,
                    float(tr_res.talknet_raw),
                    float(tr_res.talknet_smooth),
                    float(tr_res.lip_score),
                    "-inf" if tr_res.fused_score == -float("inf") else f"{float(tr_res.fused_score):.3f}",
                    "-inf" if not np.isfinite(selection_score) else f"{selection_score:.3f}",
                    spatial_zone,
                    str(center_wait_candidate),
                    str(bool(tr_res.is_lip_vetoed)),
                )
            )

        if self.active_speaker_lock_enabled:
            if self.active_speaker_lock_id is not None:
                locked_id = int(self.active_speaker_lock_id)
                for tid in list(self.asd_states.keys()):
                    if int(tid) != locked_id:
                        self.asd_states.pop(tid, None)

        self._refresh_active_speaker_lock(t_now, tracks, check_timeout=True)

        if debug_rows:
            rospy.loginfo_throttle(1.0, "[ASD][FUSION_DBG] %s", " | ".join(debug_rows))
        
        self.publish_result(t_now)

    # ---------------- 发布 ASD 结果 ----------------
    def publish_result(self, t_now: float):
        active_speakers = self._determine_effective_active_speakers()
        tracks_msg = []
        for tid, st in self.asd_states.items():
            fused_score = st.get("fused_score")
            if fused_score is not None:
                fused_score = float(fused_score)
                if not np.isfinite(fused_score):
                    fused_score = None
            tracks_msg.append({
                "id": int(tid),
                "is_speaking": tid in active_speakers, 
                "prob": float(st.get("p", 0.0)),
                "talknet_smooth": float(st.get("p", 0.0)),
                "talknet_raw": float(st.get("p_raw", 0.0)),
                "lip_score": float(st.get("lip_score", 0.0)),
                "fused_score": fused_score,
                "selection_score": float(st.get("selection_score", st.get("fused_score", 0.0))),
                "center_prior": float(st.get("center_prior", 0.0)),
                "center_proximity": float(st.get("center_proximity", 0.0)),
                "is_center_core": bool(st.get("is_center_core", False)),
                "spatial_zone": str(st.get("spatial_zone", "")),
                "center_wait_candidate": int(tid) in self._center_wait_candidate_ids(),
                "speaker_bearing_camera": st.get("speaker_bearing_camera"),
                "speaker_depth": st.get("speaker_depth"),
                "speaker_position_camera": st.get("speaker_position_camera"),
                "is_lip_vetoed": bool(st.get("is_lip_vetoed", False)),
            })
        msg = {
            "stamp": float(t_now),
            "mode": self.display_mode,
            "active_speaker_ids": [int(tid) for tid in active_speakers],
            "lock": {
                "enabled": bool(self.active_speaker_lock_enabled),
                "id": None if self.active_speaker_lock_id is None else int(self.active_speaker_lock_id),
                "until": self.active_speaker_lock_until_ts,
                "last_active": self.active_speaker_lock_last_active_ts,
                "last_visible": self.active_speaker_lock_last_visible_ts,
            },
            "center_wait": {
                "active": self.center_wait_started_ts is not None,
                "started": self.center_wait_started_ts,
                "until": self.center_wait_until_ts,
                "candidate_ids": sorted(self._center_wait_candidate_ids()),
            },
            "tracks": tracks_msg,
        }
        self.pub_result.publish(RosString(data=json.dumps(msg, ensure_ascii=False)))
        self.pub_main_speaker_pose.publish(
            RosString(data=json.dumps(self._build_main_speaker_pose_msg(t_now, active_speakers), ensure_ascii=False))
        )

    @staticmethod
    def _finite_float_or_none(value):
        try:
            out = float(value)
            if np.isfinite(out):
                return out
        except Exception:
            pass
        return None

    def _speaker_pose_payload_from_state(self, speaker_id, state):
        if state is None:
            return {
                "id": None if speaker_id is None else int(speaker_id),
                "position_camera": None,
                "bearing_camera": None,
                "depth": {"ok": False, "reason": "no_track_state"},
                "scores": None,
            }

        fused_score = state.get("fused_score")
        if fused_score is not None:
            try:
                fused_score = float(fused_score)
                if not np.isfinite(fused_score):
                    fused_score = None
            except Exception:
                fused_score = None

        depth = state.get("speaker_depth")
        if not isinstance(depth, dict):
            depth = {"ok": False, "reason": "no_depth_state"}

        return {
            "id": int(speaker_id),
            "position_camera": state.get("speaker_position_camera"),
            "bearing_camera": state.get("speaker_bearing_camera"),
            "depth": depth,
            "scores": {
                "prob": float(state.get("p", 0.0)),
                "talknet_smooth": float(state.get("p", 0.0)),
                "talknet_raw": float(state.get("p_raw", 0.0)),
                "lip_score": float(state.get("lip_score", 0.0)),
                "fused_score": fused_score,
                "selection_score": self._finite_float_or_none(
                    state.get("selection_score", state.get("fused_score", 0.0))
                ),
            },
            "spatial": {
                "is_center_core": bool(state.get("is_center_core", False)),
                "spatial_zone": str(state.get("spatial_zone", "")),
                "center_wait_candidate": bool(state.get("center_wait_candidate", False)),
            },
        }

    def _build_center_wait_msg(self):
        candidate_ids = self._center_wait_candidate_ids()
        return {
            "active": self.center_wait_started_ts is not None,
            "started": self.center_wait_started_ts,
            "until": self.center_wait_until_ts,
            "candidate_ids": sorted(candidate_ids),
        }

    def _center_wait_speaker_id(self):
        for candidate in sorted(self.center_wait_side_candidates, key=lambda c: float(c.get("timestamp", 0.0))):
            tid = int(candidate.get("track_id"))
            if tid in self.asd_states:
                return tid
        candidate_ids = sorted(self._center_wait_candidate_ids())
        return candidate_ids[0] if candidate_ids else None

    def _build_main_speaker_pose_msg(self, t_now: float, active_speakers):
        active_ids = [int(tid) for tid in active_speakers]
        locked_id = None if self.active_speaker_lock_id is None else int(self.active_speaker_lock_id)
        is_center_waiting = self.center_wait_started_ts is not None

        state = "none"
        main_speaker_id = None
        is_speaking = False
        if active_ids:
            state = "active_speaking"
            main_speaker_id = active_ids[0]
            is_speaking = True
        elif locked_id is not None:
            state = "locked_5s"
            main_speaker_id = locked_id
        elif is_center_waiting:
            state = "center_waiting_3s"

        pending_speaker_id = self._center_wait_speaker_id() if is_center_waiting else None
        has_main_speaker = main_speaker_id is not None

        main_payload = self._speaker_pose_payload_from_state(
            main_speaker_id,
            self.asd_states.get(main_speaker_id) if main_speaker_id is not None else None,
        )
        if main_speaker_id is None:
            main_payload["depth"] = {"ok": False, "reason": state}

        main_speaker = None
        if main_speaker_id is not None:
            main_speaker = dict(main_payload)
            main_speaker.update(
                {
                    "status": state,
                    "is_speaking": bool(is_speaking),
                    "is_locked": locked_id is not None and main_speaker_id == locked_id,
                    "is_center_wait_candidate": False,
                }
            )

        pending_speaker = None
        if pending_speaker_id is not None:
            pending_speaker = self._speaker_pose_payload_from_state(
                pending_speaker_id,
                self.asd_states.get(pending_speaker_id),
            )
            pending_speaker.update(
                {
                    "status": "center_waiting_3s",
                    "is_speaking": False,
                    "is_locked": False,
                    "is_center_wait_candidate": True,
                    "center_wait_started": self.center_wait_started_ts,
                    "center_wait_until": self.center_wait_until_ts,
                }
            )

        return {
            "stamp": float(t_now),
            "state": state,
            "has_speaker": has_main_speaker,
            "speaker_id": main_speaker_id,
            "has_main_speaker": has_main_speaker,
            "main_speaker_id": main_speaker_id,
            "main_speaker": main_speaker,
            "has_pending_speaker": pending_speaker is not None,
            "pending_speaker_id": pending_speaker_id,
            "pending_speaker": pending_speaker,
            "is_speaking": bool(is_speaking),
            "is_locked": locked_id is not None and main_speaker_id == locked_id,
            "is_center_waiting": bool(is_center_waiting),
            "is_center_wait_candidate": False,
            "active_speaker_ids": active_ids,
            "position_camera": main_payload.get("position_camera"),
            "bearing_camera": main_payload.get("bearing_camera"),
            "depth": main_payload.get("depth"),
            "scores": main_payload.get("scores"),
            "spatial": main_payload.get("spatial"),
            "lock": {
                "enabled": bool(self.active_speaker_lock_enabled),
                "id": locked_id,
                "until": self.active_speaker_lock_until_ts,
                "last_active": self.active_speaker_lock_last_active_ts,
                "last_visible": self.active_speaker_lock_last_visible_ts,
            },
            "center_wait": self._build_center_wait_msg(),
        }

    def cleanup(self):
        rospy.loginfo("[ASD] cleanup...")
        self._audio_worker_stop = True
        if self._audio_worker_thread is not None:
            try:
                self._audio_worker_thread.join(timeout=1.0)
            except Exception:
                pass
        if self._audio_process is not None:
            try:
                self._audio_process.terminate()
                self._audio_process.join(timeout=1.0)
            except Exception:
                pass
        if self.sub_video_capture is not None:
            try:
                self.sub_video_capture.unregister()
            except Exception:
                pass
        if self.stream_capture is not None:
            try:
                self.stream_capture.finalize()
            except Exception as e:
                rospy.logwarn("[ASD] failed to finalize stream capture: %s", e)
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    try:
        node = RealtimeASDNode()
        rospy.on_shutdown(node.cleanup)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
