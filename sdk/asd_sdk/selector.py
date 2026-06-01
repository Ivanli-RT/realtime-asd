from typing import Dict, List, Tuple


def select_active_speakers(
    score_by_track: Dict[int, float],
    mode: str = "top_scorer",
    threshold_min_score: float = 0.30,
    threshold_relative_margin: float = 0.12,
    top_scorer_min_score: float = 0.40,
    top_scorer_override_threshold: float = 0.70,
) -> List[int]:
    tracks_with_scores: List[Tuple[int, float]] = list(score_by_track.items())
    if not tracks_with_scores:
        return []

    if mode == "threshold":
        top_score = max(score for _, score in tracks_with_scores)
        selected = [tid for tid, score in tracks_with_scores if score >= threshold_min_score]
        if 0.0 <= threshold_relative_margin < 1.0:
            selected = [
                tid
                for tid, score in tracks_with_scores
                if score >= threshold_min_score and score >= (top_score - threshold_relative_margin)
            ]
        return selected

    if mode == "top_scorer_with_override":
        max_score = max(score for _, score in tracks_with_scores)
        if max_score < top_scorer_min_score:
            return []
        sorted_tracks = sorted(tracks_with_scores, key=lambda x: (-x[1], x[0]))
        active = {sorted_tracks[0][0]}
        for tid, score in tracks_with_scores:
            if score >= top_scorer_override_threshold:
                active.add(tid)
        return list(active)

    # default: top_scorer
    max_score = max(score for _, score in tracks_with_scores)
    if max_score < top_scorer_min_score:
        return []
    winner = sorted(tracks_with_scores, key=lambda x: (-x[1], x[0]))[0]
    return [winner[0]]
