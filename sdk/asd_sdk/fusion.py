from typing import Tuple


def compute_fused_score(
    asd_score: float,
    lip_score: float,
    lip_weight: float = 0.2,
    lip_veto_threshold: float = -0.8,
    talknet_confirm_thresh: float = 0.50,
) -> Tuple[float, bool]:
    if lip_score == -float("inf"):
        return -float("inf"), True

    if lip_score < lip_veto_threshold:
        return -float("inf"), True

    if lip_score < 0:
        penalty_factor = max(0.35, 1.0 + lip_score)
        fused = asd_score * penalty_factor
        return fused, False

    if asd_score > talknet_confirm_thresh:
        bonus = lip_score * max(0.0, min(lip_weight, 0.2))
        return min(1.0, asd_score + bonus), False

    base_bonus = lip_score * max(0.0, min(lip_weight, 0.2))
    low_conf_bonus = min(0.08, base_bonus * 0.45)
    return min(1.0, asd_score + low_conf_bonus), False
