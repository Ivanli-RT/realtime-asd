from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class TrackInput:
    audio_clip: np.ndarray
    video_clip: np.ndarray
    track_id: Optional[int] = None
    lip_score: Optional[float] = None


@dataclass
class InferenceRequest:
    tracks: List[TrackInput]
    timestamp: float


@dataclass
class TrackResult:
    track_id: int
    talknet_raw: float
    talknet_smooth: float
    lip_score: float
    fused_score: float
    is_lip_vetoed: bool
    is_active: bool = False


@dataclass
class InferenceResult:
    mode: str
    timestamp: float
    active_speaker_ids: List[int]
    tracks: List[TrackResult]
    debug: Dict[str, float] = field(default_factory=dict)
