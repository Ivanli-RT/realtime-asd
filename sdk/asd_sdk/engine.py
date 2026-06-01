from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional

import numpy as np

from .fusion import compute_fused_score
from .selector import select_active_speakers
from .types import InferenceRequest, InferenceResult, TrackResult


class ASDSDK:
    """
    Algorithm-only SDK:
    - No YOLO detector
    - No ROS dependence
    - No UI rendering

    Upstream should provide aligned per-track audio/video clips and optional lip scores.
    """

    def __init__(
        self,
        talknet_backend: Optional[Callable[[List[np.ndarray], List[np.ndarray]], List[np.ndarray]]] = None,
        mode: str = "top_scorer",
        enable_lip_motion: bool = True,
        smooth_alpha: float = 0.55,
        speak_on_thresh: float = 0.52,
        speak_off_thresh: float = 0.38,
        lip_motion_weight: float = 0.2,
        lip_veto_threshold: float = -0.8,
        lip_talknet_confirm_thresh: float = 0.50,
        threshold_min_score: float = 0.30,
        threshold_relative_margin: float = 0.12,
        top_scorer_min_score: float = 0.40,
        top_scorer_override_threshold: float = 0.70,
    ):
        self._talknet_backend = talknet_backend
        self.mode = mode
        self.enable_lip_motion = bool(enable_lip_motion)

        self.smooth_alpha = float(smooth_alpha)
        self.speak_on_thresh = float(speak_on_thresh)
        self.speak_off_thresh = float(speak_off_thresh)

        self.lip_motion_weight = float(lip_motion_weight)
        self.lip_veto_threshold = float(lip_veto_threshold)
        self.lip_talknet_confirm_thresh = float(lip_talknet_confirm_thresh)

        self.threshold_min_score = float(threshold_min_score)
        self.threshold_relative_margin = float(threshold_relative_margin)
        self.top_scorer_min_score = float(top_scorer_min_score)
        self.top_scorer_override_threshold = float(top_scorer_override_threshold)

        self._states: Dict[int, Dict[str, float]] = {}

    def remap_track_id(self, old_id: int, new_id: int) -> None:
        """Move temporal smoothing state when the upstream tracker changes IDs."""
        old_id = int(old_id)
        new_id = int(new_id)
        if old_id == new_id:
            return

        old_state = self._states.pop(old_id, None)
        if old_state is not None:
            self._states[new_id] = old_state

    @staticmethod
    def build_default_talknet_backend() -> Callable[[List[np.ndarray], List[np.ndarray]], List[np.ndarray]]:
        """Build backend adapter using runtime TalkNet wrapper when available."""
        from asd.model_wrapper import TalkNetASDModel

        model = TalkNetASDModel()

        def _backend(audio_clips: List[np.ndarray], video_clips: List[np.ndarray]) -> List[np.ndarray]:
            return model.infer_batch_per_track(audio_clips=audio_clips, video_clips=video_clips)

        return _backend

    def infer(self, request: InferenceRequest) -> InferenceResult:
        if not request.tracks:
            return InferenceResult(
                mode=self.mode,
                timestamp=request.timestamp,
                active_speaker_ids=[],
                tracks=[],
            )

        audio_clips = [t.audio_clip for t in request.tracks]
        video_clips = [t.video_clip for t in request.tracks]
        track_ids = [t.track_id if t.track_id is not None else i for i, t in enumerate(request.tracks)]
        lip_scores_in = {
            tid: request.tracks[i].lip_score
            for i, tid in enumerate(track_ids)
            if request.tracks[i].lip_score is not None
        }

        if self.enable_lip_motion:
            missing = [tid for tid in track_ids if tid not in lip_scores_in]
            if missing:
                raise ValueError(
                    f"lip_score is required when enable_lip_motion=True, missing track_ids={missing}"
                )

        backend = self._talknet_backend
        if backend is None:
            raise RuntimeError(
                "talknet_backend is not set. Pass a backend callable or use ASDSDK.build_default_talknet_backend()."
            )

        candidate_indices: List[int] = []
        skipped_indices: List[int] = []

        if self.enable_lip_motion:
            for idx, tid in enumerate(track_ids):
                lip_score = float(lip_scores_in[tid])
                if lip_score < self.lip_veto_threshold:
                    skipped_indices.append(idx)
                else:
                    candidate_indices.append(idx)
        else:
            candidate_indices = list(range(len(track_ids)))

        candidate_audio_clips = [audio_clips[idx] for idx in candidate_indices]
        candidate_video_clips = [video_clips[idx] for idx in candidate_indices]

        if candidate_indices:
            probs_list = backend(candidate_audio_clips, candidate_video_clips)
        else:
            probs_list = []

        probs_by_index = {
            candidate_indices[idx]: probs_list[idx]
            for idx in range(len(candidate_indices))
        }

        track_results: List[TrackResult] = []
        score_for_selection: Dict[int, float] = {}

        for idx, tid in enumerate(track_ids):
            if idx in skipped_indices:
                p_raw = 0.0
                p_smooth = 0.0
                lip_score = float(lip_scores_in[tid]) if self.enable_lip_motion else 0.0
                fused_score = -float("inf")
                is_vetoed = True
                is_spk = False
            else:
                probs = probs_by_index.get(idx)
                if probs is None or len(probs) == 0:
                    p_raw = 0.0
                else:
                    p_logit = float(np.mean(probs[-3:])) if len(probs) >= 3 else float(np.mean(probs))
                    try:
                        p_raw = 1.0 / (1.0 + math.exp(-p_logit))
                    except OverflowError:
                        p_raw = 0.0 if p_logit < 0 else 1.0

                st = self._states.get(tid)
                if st is None:
                    p_smooth = p_raw
                    is_spk = p_raw > self.speak_on_thresh
                else:
                    p_smooth = self.smooth_alpha * float(st.get("p", p_raw)) + (1.0 - self.smooth_alpha) * p_raw
                    is_spk = bool(st.get("is_spk", False))

                if self.enable_lip_motion:
                    lip_score = float(lip_scores_in[tid])
                    fused_score, is_vetoed = compute_fused_score(
                        asd_score=p_smooth,
                        lip_score=lip_score,
                        lip_weight=self.lip_motion_weight,
                        lip_veto_threshold=self.lip_veto_threshold,
                        talknet_confirm_thresh=self.lip_talknet_confirm_thresh,
                    )
                else:
                    lip_score = 0.0
                    fused_score, is_vetoed = p_smooth, False

            decision_score = 0.0 if fused_score == -float("inf") else float(fused_score)

            if not is_spk and decision_score > self.speak_on_thresh:
                is_spk = True
            elif is_spk and decision_score < self.speak_off_thresh:
                is_spk = False

            self._states[tid] = {
                "p": p_smooth,
                "p_raw": p_raw,
                "fused_score": decision_score,
                "is_spk": float(is_spk),
                "ts": float(request.timestamp),
            }

            score_for_selection[tid] = decision_score
            track_results.append(
                TrackResult(
                    track_id=tid,
                    talknet_raw=p_raw,
                    talknet_smooth=p_smooth,
                    lip_score=lip_score,
                    fused_score=fused_score,
                    is_lip_vetoed=is_vetoed,
                )
            )

        # Garbage collection for stale states to prevent "ghost track" holding high scores
        stale_tids = [tid for tid, st in self._states.items() if (request.timestamp - float(st.get("ts", request.timestamp))) > 2.0]
        for tid in stale_tids:
            self._states.pop(tid, None)

        active_ids = select_active_speakers(
            score_by_track=score_for_selection,
            mode=self.mode,
            threshold_min_score=self.threshold_min_score,
            threshold_relative_margin=self.threshold_relative_margin,
            top_scorer_min_score=self.top_scorer_min_score,
            top_scorer_override_threshold=self.top_scorer_override_threshold,
        )

        active_set = set(active_ids)
        for tr in track_results:
            tr.is_active = tr.track_id in active_set

        return InferenceResult(
            mode=self.mode,
            timestamp=request.timestamp,
            active_speaker_ids=active_ids,
            tracks=track_results,
            debug={
                "num_tracks": float(len(track_results)),
                "num_talknet_candidates": float(len(candidate_indices)),
                "num_lip_veto_tracks": float(len(skipped_indices)),
            },
        )
