from typing import Dict, Iterable, Tuple

import cv2


def draw_active_speaker_overlay(
    frame_bgr,
    bboxes_by_track: Dict[int, Tuple[int, int, int, int]],
    score_by_track: Dict[int, float],
    active_track_ids: Iterable[int],
    show_scores: bool = True,
):
    """Draw active speaker overlay independent from SDK core logic."""
    active_set = set(active_track_ids)

    for tid, bbox in bboxes_by_track.items():
        x1, y1, x2, y2 = bbox
        color = (0, 255, 0) if tid in active_set else (0, 0, 255)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)

        if show_scores:
            score = float(score_by_track.get(tid, 0.0))
            label = f"id={tid} p={score:.2f}" + (" ACTIVE" if tid in active_set else "")
        else:
            label = f"id={tid}" + (" ACTIVE" if tid in active_set else "")

        y_text = max(15, y1 - 8)
        cv2.putText(
            frame_bgr,
            label,
            (x1, y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    return frame_bgr
