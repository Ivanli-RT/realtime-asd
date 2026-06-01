#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import importlib

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
from sensor_msgs.msg import Image
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
from asd.tracking import Tracker
from asd.ros_audio import extract_mono_audio, extract_raw_audio_packet, format_audio_meta
from asd import create_face_detector  # 使用工厂函数
import traceback
# 在文件开头,其他导入语句之后添加
from asd.active_speaker_utils import (
    determine_active_speakers,
    get_display_color,
    format_speaker_info
)

SDK_IMPORT_SOURCE = "installed"
try:
    from asd_sdk import ASDSDK, InferenceRequest, TrackInput
except Exception:
    # Fallback to local source tree when wheel is not installed in environment.
    SDK_DIR = os.path.join(ROOT_DIR, "sdk")
    if SDK_DIR not in sys.path:
        sys.path.insert(0, SDK_DIR)
    from asd_sdk import ASDSDK, InferenceRequest, TrackInput
    SDK_IMPORT_SOURCE = "source_fallback"

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
        rospy.init_node("realtime_asd_node_sdk")
        rospy.logwarn(
            "[ASD] realtime_asd_node_sdk_entry.py is a compatibility entry and does not "
            "include the current active-speaker lock/re-id logic. Use scripts/run_node.sh "
            "for realtime active-speaker testing."
        )

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
        
        # 人脸检测器类型(可通过ROS参数覆盖配置文件)
        detector_type = rospy.get_param("~face_detector_type", C.FACE_DETECTOR_TYPE)
        self.display_mode = rospy.get_param("~active_speaker_mode", C.ACTIVE_SPEAKER_DISPLAY_MODE)
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
        self._audio_pkt_total = 0
        self._audio_sample_total = 0
        self._audio_monitor_last_ts = time.time()
        self._audio_monitor_last_pkt_total = 0
        self._audio_monitor_last_sample_total = 0
        self._audio_monitor_last_drop_total = 0
        self._audio_monitor_last_invalid_total = 0
        self._audio_monitor_last_empty_total = 0
        self._last_audio_meta_desc = None
        self.stream_capture = None

        # 自动探测音频消息类型
        AudioMsg = self._resolve_audio_msg_class()
        self.audio_msg_type = AudioMsg

        # 订阅
        self.sub_video = None
        if not self.capture_only:
            self.sub_video = rospy.Subscriber(
                self.video_topic, Image, self.image_cb,
                queue_size=2, buff_size=2**24
            )
        self.sub_video_capture = None
        self.sub_audio = None
        self._audio_probe_timer = None
        if self.detector_only and self.detector_only_skip_audio:
            rospy.logwarn("[ASD] detector-only mode: skip audio subscriber")
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
            "[ASD] audio buffer: seconds=%.2f sample_rate=%d",
            float(getattr(C, "AUDIO_BUFFER_SECONDS", C.CLIP_SECONDS * 4.0)),
            int(getattr(C, "AUDIO_SAMPLE_RATE", 16000)),
        )
        rospy.loginfo("[ASD] audio selected channel: %d", self.audio_use_channel)
        rospy.loginfo("[ASD] SDK import source: %s", SDK_IMPORT_SOURCE)

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

    def _build_visualization(self, frame_bgr, tracks):
        vis = frame_bgr.copy()
        active_speakers = determine_active_speakers(
            self.asd_states,
            mode=self.display_mode,
        )
        self._last_vis_time = time.time()

        for tr in tracks:
            x1, y1, x2, y2 = tr.bbox
            state = self.asd_states.get(tr.track_id, {"p": 0.0, "is_spk": False})
            is_active = tr.track_id in active_speakers
            color = get_display_color(tr.track_id, active_speakers)

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            asd_score = float(state.get("p", 0.0))
            lip_score = float(state.get("lip_score", 0.0))
            fused_score_raw = float(state.get("fused_score", asd_score))
            is_vetoed = bool(state.get("is_lip_vetoed", False))

            if self.debug_show_scores:
                lip_display = lip_score if lip_score > -100 else -1.0
                fused_display = fused_score_raw if fused_score_raw > -100 else -1.0

                line1 = f"ID:{tr.track_id} {'[SPK]' if is_active else ''}"
                line2 = f"ASD:{asd_score:+.2f} LIP:{lip_display:+.2f}"
                line3 = f"FUSED:{fused_display:+.2f}{'[VETO]' if is_vetoed else ''}"

                y_offset = max(0, y1 - 5)
                cv2.putText(vis, line1, (x1, y_offset - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
                cv2.putText(vis, line2, (x1, y_offset - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(vis, line3, (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
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
        tracks_msg = []
        for tr in tracks:
            state = self.asd_states.get(tr.track_id, {})
            fused_score = state.get("fused_score")
            if fused_score == -float("inf"):
                fused_score = None
            tracks_msg.append(
                {
                    "track_id": int(tr.track_id),
                    "bbox": [int(v) for v in tr.bbox],
                    "is_active": int(tr.track_id) in active_set,
                    "talknet_smooth": float(state.get("p", 0.0)),
                    "talknet_raw": float(state.get("p_raw", 0.0)),
                    "lip_score": float(state.get("lip_score", 0.0)),
                    "fused_score": None if fused_score is None else float(fused_score),
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
    def audio_cb(self, msg):
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

        d_pkt = pkt_total - self._audio_monitor_last_pkt_total
        d_sample = sample_total - self._audio_monitor_last_sample_total
        d_drop = drop_total - self._audio_monitor_last_drop_total
        d_invalid = invalid_total - self._audio_monitor_last_invalid_total
        d_empty = empty_total - self._audio_monitor_last_empty_total

        pkt_rate = d_pkt / dt
        mono_sample_rate = d_sample / dt

        status = "OK" if d_drop == 0 and d_invalid == 0 and d_empty == 0 else "WARNING"
        log_fn = rospy.loginfo if status == "OK" else rospy.logwarn
        log_fn(
            "[ASD][AUDIO_MON][%s] dt=%.1fs pkt=%d(%.1f/s) mono_samples=%d(%.1f/s) "
            "drop=%d(total=%d) invalid=%d(total=%d) empty=%d(total=%d)",
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
        )

        self._audio_monitor_last_ts = now
        self._audio_monitor_last_pkt_total = pkt_total
        self._audio_monitor_last_sample_total = sample_total
        self._audio_monitor_last_drop_total = drop_total
        self._audio_monitor_last_invalid_total = invalid_total
        self._audio_monitor_last_empty_total = empty_total

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
            tracks = self.tracker.update(dets)
            self._proc_frame_count += 1

            # 2.5) Clean up stale history to prevent "ghost" speakers
            active_track_ids = {tr.track_id for tr in tracks}
            stale_ids = [tid for tid in list(self.track_buffers.keys()) + list(self.asd_states.keys()) if tid not in active_track_ids]
            for tid in set(stale_ids):
                self.track_buffers.pop(tid, None)
                self.asd_states.pop(tid, None)

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

            # 4) 每个处理帧都先更新 Lip 状态（用于更高时域分辨率）
            self._update_lip_states(tracks, frame_bgr, t_frame)

            # 5) TalkNet 按独立步长触发，和 Lip 解耦
            if len(tracks) > 0 and (self._proc_frame_count % self.talknet_frame_stride == 0):
                self.try_run_asd(t_frame, tracks)

            # 每个处理帧都更新一次运行频率统计，避免统计只跟随 TalkNet 调用节拍。
            self._update_runtime_fps_stats()

            vis = None
            active_speakers = None
            if self.debug_show_window:
                vis, active_speakers = self._build_visualization(frame_bgr, tracks)

            if self.stream_capture is not None:
                if active_speakers is None:
                    active_speakers = determine_active_speakers(
                        self.asd_states,
                        mode=self.display_mode,
                    )
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
    def try_run_asd(self, t_now: float, tracks):
        """ASD inference aligned with source sequential behavior."""
        if self.asd_sdk is None:
            return
        if len(tracks) == 0:
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

        active_set = set(sdk_result.active_speaker_ids)
        for tr_res in sdk_result.tracks:
            tid = int(tr_res.track_id)
            prev = self.asd_states.get(tid, {})
            hist = list(prev.get("history", []))
            hist.append(float(tr_res.talknet_raw))
            if len(hist) > 5:
                hist = hist[-5:]

            self.asd_states[tid] = {
                "p": float(tr_res.talknet_smooth),
                "p_raw": float(tr_res.talknet_raw),
                "is_spk": tid in active_set,
                "history": hist,
                "fused_score": float(tr_res.fused_score),
                "lip_score": float(tr_res.lip_score),
                "is_lip_vetoed": bool(tr_res.is_lip_vetoed),
            }

            debug_rows.append(
                "id=%d raw=%.3f smooth=%.3f lip=%.3f fused=%s veto=%s"
                % (
                    tid,
                    float(tr_res.talknet_raw),
                    float(tr_res.talknet_smooth),
                    float(tr_res.lip_score),
                    "-inf" if tr_res.fused_score == -float("inf") else f"{float(tr_res.fused_score):.3f}",
                    str(bool(tr_res.is_lip_vetoed)),
                )
            )

        if debug_rows:
            rospy.loginfo_throttle(1.0, "[ASD][FUSION_DBG] %s", " | ".join(debug_rows))
        
        self.publish_result(t_now)

    # ---------------- 发布 ASD 结果 ----------------
    def publish_result(self, t_now: float):
        active_speakers = determine_active_speakers(
        self.asd_states,
        mode=self.display_mode
    )
        tracks_msg = []
        for tid, st in self.asd_states.items():
            tracks_msg.append({
                "id": tid,
                "is_speaking": tid in active_speakers, 
                "prob": float(st["p"]),
            })
        msg = {
            "stamp": t_now,
            "mode": self.display_mode,
            "tracks": tracks_msg,
        }
        self.pub_result.publish(RosString(data=str(msg)))

    def cleanup(self):
        rospy.loginfo("[ASD] cleanup...")
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
