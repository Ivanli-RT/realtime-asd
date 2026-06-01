# asd/__init__.py
"""
实时主动说话人识别核心模块包

包含:
- buffers: 音频 / 视频轨迹缓冲(带时间戳)
- tracking: 人脸检测结果的 IOU 多目标追踪
- yolo_wrapper: YOLO 人脸检测器
- model_wrapper: TalkNet ASD 模型封装
- create_face_detector: 检测器工厂函数
"""

from .buffers import AudioBuffer, TrackBuffer
from .tracking import Tracker, FaceDet
from .yolo_wrapper import YOLOFaceDetector
from .model_wrapper import TalkNetASDModel
from .lip_landmark_wrapper import LipLandmarkDetector
from .lip_motion_analyzer import LipMotionAnalyzer, compute_fused_score
from .active_speaker_utils import (
    determine_active_speakers,
    get_display_color,
    format_speaker_info,
)

from config import asd_config as C

# 全局调试开关
DEBUG_ASD_SCORES = True  # 打印ASD分数调试信息
DEBUG_AUDIO_BUFFER = True  # 打印音频缓冲调试信息
DEBUG_TRACK_BUFFER = True  # 打印跟踪缓冲调试信息


def create_face_detector(detector_type: str = None):
    """
    人脸检测器工厂函数
    
    Args:
        detector_type: 当前瘦身版仅支持 "yolo"
        
    Returns:
        YOLOFaceDetector 实例
        
    Raises:
        ValueError: 不支持的检测器类型
    """
    if detector_type is None:
        detector_type = C.FACE_DETECTOR_TYPE
    
    detector_type = detector_type.lower()
    
    if detector_type == "yolo":
        print(f"[FaceDetector] Using YOLO (conf={C.FACE_DETECTION_CONF_THRESH})")
        return YOLOFaceDetector()
    
    else:
        raise ValueError(
            f"Unsupported face detector type: {detector_type}. "
            f"Choose 'yolo'"
        )


__all__ = [
    "AudioBuffer",
    "TrackBuffer",
    "Tracker",
    "FaceDet",
    "YOLOFaceDetector",
    "TalkNetASDModel",
    "LipLandmarkDetector",
    "LipMotionAnalyzer",
    "compute_fused_score",
    "create_face_detector",
    "determine_active_speakers",
    "get_display_color", 
    "format_speaker_info",
]