# asd/active_speaker_utils.py
# -*- coding: utf-8 -*-
"""
Active speaker selection helpers.

支持多种得分来源和判定模式:
- 原始 TalkNet 得分
- 融合唇部运动的得分
- 唇部运动一票否决机制
"""

from typing import Dict, List, Tuple

from config import asd_config as C


def _score_from_state(state: Dict, use_fused: bool = None) -> float:
    """
    Choose which score to use for active speaker selection.

    Args:
        state: track state dict
        use_fused: 是否使用融合得分 (None=自动检测)

    Score sources:
    - fused: state["fused_score"] (融合唇部运动后的得分, 推荐)
    - smooth: state["p"] (EMA-smoothed)
    - raw: state["p_raw"] (latest)
    - history: mean of recent raw history
    """
    # 如果启用了唇部运动检测，优先使用融合得分
    if use_fused is None:
        use_fused = getattr(C, 'ENABLE_LIP_MOTION', False)
    
    # 检查是否被唇部运动否决
    if state.get("is_lip_vetoed", False):
        return -float('inf')  # 被否决的 track 得分为负无穷
    
    # 如果有融合得分且启用了唇部运动，使用融合得分
    if use_fused and "fused_score" in state:
        fused = state["fused_score"]
        if fused == -float('inf'):
            return -float('inf')
        return float(fused)
    
    # 否则使用配置的得分来源
    source = str(getattr(C, "ACTIVE_SPEAKER_SCORE_SOURCE", "smooth")).lower()

    if source == "raw":
        return float(state.get("p_raw", state.get("p", 0.0)))

    if source == "history":
        history = state.get("history") or []
        if history:
            return float(sum(history) / len(history))
        return float(state.get("p", 0.0))

    return float(state.get("p", 0.0))


def determine_active_speakers(
    asd_states: Dict[int, Dict],
    mode: str = None,
    use_fused: bool = None,
) -> List[int]:
    """
    Decide which track_ids are active speakers.

    Args:
        asd_states: {track_id: {"p": float, "is_spk": bool, "fused_score": float, ...}}
        mode: display mode; if None, use config.
        use_fused: 是否使用融合得分 (None=自动)

    Returns:
        List[int]: active speaker track_ids.
    """
    if mode is None:
        mode = C.ACTIVE_SPEAKER_DISPLAY_MODE

    if not asd_states:
        return []

    # 过滤被唇部否决的 track
    valid_states = {
        tid: state for tid, state in asd_states.items()
        if not state.get("is_lip_vetoed", False)
    }
    
    tracks_with_scores = [
        (tid, _score_from_state(state, use_fused))
        for tid, state in valid_states.items()
    ]
    
    # 过滤掉得分为负无穷的 track
    tracks_with_scores = [
        (tid, score) for tid, score in tracks_with_scores
        if score != -float('inf')
    ]

    if not tracks_with_scores:
        return []

    debug_enabled = bool(getattr(C, "ACTIVE_SPEAKER_DEBUG", False))
    if debug_enabled:
        enable_lip = getattr(C, 'ENABLE_LIP_MOTION', False)
        if enable_lip:
            lip_info = [(tid, asd_states[tid].get("lip_score", 0.0)) for tid, _ in tracks_with_scores]
            print(
                f"[ActiveSpeaker] mode={mode}, lip_motion=ON, "
                f"candidates={[(t[0], f'{t[1]:.4f}') for t in tracks_with_scores]}, "
                f"lip_scores={[(t, f'{l:.3f}') for t, l in lip_info]}",
                flush=True
            )
        else:
            print(f"[ActiveSpeaker] mode={mode}, candidates={[(t[0], f'{t[1]:.4f}') for t in tracks_with_scores]}", flush=True)

    if mode == "threshold":
        result = _threshold_mode(tracks_with_scores)
    elif mode == "top_scorer":
        result = _top_scorer_mode(tracks_with_scores)
    elif mode == "top_scorer_with_override":
        result = _top_scorer_with_override_mode(tracks_with_scores)
    else:
        if debug_enabled:
            print(f"[ActiveSpeaker] Unknown mode: {mode}, fallback to threshold")
        result = _threshold_mode(tracks_with_scores)
    
    if debug_enabled:
        print(f"[ActiveSpeaker] Result: active_speakers={result}", flush=True)
    return result


def _threshold_mode(tracks_with_scores: List[Tuple[int, float]]) -> List[int]:
    """Mode 1: tracks above threshold are active, with optional relative-top gating."""
    if not tracks_with_scores:
        return []

    threshold = C.THRESHOLD_MODE_MIN_SCORE
    relative_margin = float(getattr(C, "THRESHOLD_MODE_RELATIVE_MARGIN", 1.0))
    top_score = max(score for _, score in tracks_with_scores)

    selected = [tid for tid, score in tracks_with_scores if score >= threshold]

    if relative_margin >= 0.0 and relative_margin < 1.0:
        selected = [
            tid for tid, score in tracks_with_scores
            if score >= threshold and score >= (top_score - relative_margin)
        ]

    return selected


def _top_scorer_mode(tracks_with_scores: List[Tuple[int, float]]) -> List[int]:
    """Mode 2 (top_scorer): 只返回得分最高的一个 track。
    
    逻辑：
    - 只取得分最大的一个人
    - 要求分数必须超过 TOP_SCORER_MIN_SCORE
    - 如果多个 track 得分相同，选择 track_id 最小的（确定性选择）
    """
    if not tracks_with_scores:
        return []

    max_score = max(score for _, score in tracks_with_scores)

    # 最高分必须超过 TOP_SCORER_MIN_SCORE
    if max_score < C.TOP_SCORER_MIN_SCORE:
        return []

    # 严格只返回一个：找到得分最高的，如果有多个相同最高分则选 track_id 最小的
    # 按 (score desc, track_id asc) 排序
    sorted_tracks = sorted(tracks_with_scores, key=lambda x: (-x[1], x[0]))
    winner = sorted_tracks[0]
    
    return [winner[0]]


def _top_scorer_with_override_mode(tracks_with_scores: List[Tuple[int, float]]) -> List[int]:
    """Mode 3 (top_scorer_with_override): 显示最高分者，以及所有超过 override 阈值的人。
    
    逻辑：
    - 取得分最大的作为主动说话人
    - 但如果某人得分超过 TOP_SCORER_OVERRIDE_THRESHOLD，即使不是最大分也显示为说话人
    - 最高分必须超过 TOP_SCORER_MIN_SCORE
    """
    if not tracks_with_scores:
        return []

    override_threshold = C.TOP_SCORER_OVERRIDE_THRESHOLD
    min_score = C.TOP_SCORER_MIN_SCORE
    max_score = max(score for _, score in tracks_with_scores)

    # 最高分必须超过 min_score
    if max_score < min_score:
        return []

    active_speakers = set()
    
    # 找到得分最高的（如果有多个相同最高分，选 track_id 最小的）
    sorted_tracks = sorted(tracks_with_scores, key=lambda x: (-x[1], x[0]))
    top_scorer = sorted_tracks[0]
    active_speakers.add(top_scorer[0])
    
    # 添加所有超过 override 阈值的
    for tid, score in tracks_with_scores:
        if score >= override_threshold:
            active_speakers.add(tid)

    return list(active_speakers)


def get_display_color(track_id: int, active_speakers: List[int]) -> Tuple[int, int, int]:
    """Return display color based on active status."""
    if track_id in active_speakers:
        return (0, 255, 0)
    return (0, 0, 255)


def format_speaker_info(
    track_id: int,
    score: float,
    is_active: bool,
    show_scores: bool = None,
) -> str:
    """Format speaker label text."""
    if show_scores is None:
        show_scores = C.DEBUG_SHOW_SCORES

    if show_scores:
        status = "ACTIVE" if is_active else ""
        return f"id={track_id} p={score:.2f} {status}"

    if is_active:
        return f"id={track_id} ACTIVE"
    return f"id={track_id}"
