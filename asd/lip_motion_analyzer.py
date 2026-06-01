# asd/lip_motion_analyzer.py
# -*- coding: utf-8 -*-
"""
唇部运动分析模块

通过分析唇部关键点的运动来辅助主动说话人检测：
1. 计算嘴唇开合度 (Lip Aperture Ratio, LAR)
2. 多帧运动分析 (运动幅度、变化率)
3. 生成唇部运动得分，用于融合 TalkNet ASD 得分

唇部关键点分布 (98点标准，0-based索引):
- 外嘴角: 76(左), 82(右)
- 内嘴角: 88(左), 92(右)
- 上唇外侧: 77, 78, 79, 80, 81
- 下唇外侧: 83, 84, 85, 86, 87
- 上唇内侧: 89, 90, 91
- 下唇内侧: 93, 94, 95

核心算法:
- LAR (Lip Aperture Ratio) = 嘴唇垂直开度 / 嘴唇水平宽度
- 运动得分 = f(LAR变化幅度, LAR均值)
- 多帧累积判断: 持续闭合 → 强制非说话人
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


# ============== 唇部关键点索引 (0-based) ==============
# 对应 98 点标准
LIP_OUTER_LEFT_CORNER = 76
LIP_OUTER_RIGHT_CORNER = 82
LIP_INNER_LEFT_CORNER = 88
LIP_INNER_RIGHT_CORNER = 92

LIP_UPPER_OUTER = [77, 78, 79, 80, 81]  # 上唇外侧 (左→右)
LIP_LOWER_OUTER = [83, 84, 85, 86, 87]  # 下唇外侧 (左→右)
LIP_UPPER_INNER = [89, 90, 91]          # 上唇内侧 (左→右)
LIP_LOWER_INNER = [93, 94, 95]          # 下唇内侧 (左→右)

# 上唇中点、下唇中点 (用于计算 LAR)
LIP_UPPER_CENTER = 90  # 上唇内侧中点
LIP_LOWER_CENTER = 94  # 下唇内侧中点
LIP_ALL_INDICES = list(range(76, 96))
NON_LIP_INDICES = [idx for idx in range(98) if idx not in set(LIP_ALL_INDICES)]


@dataclass
class LipMetrics:
    """单帧唇部测量指标"""
    lar: float                 # Lip Aperture Ratio (垂直/水平)
    vertical_dist: float       # 嘴唇垂直开度 (像素)
    horizontal_dist: float     # 嘴唇水平宽度 (像素)
    is_valid: bool             # 关键点是否有效
    timestamp: float = 0.0     # 时间戳


@dataclass
class LipMotionState:
    """单个 track 的唇部运动状态"""
    track_id: int
    history: deque = field(default_factory=lambda: deque(maxlen=30))  # 最近 N 帧的 LAR
    timestamps: deque = field(default_factory=lambda: deque(maxlen=30))
    
    # 运动统计
    lar_mean: float = 0.0
    lar_std: float = 0.0
    lar_range: float = 0.0  # max - min
    lar_toggle_rate: float = 0.0  # 唇部开合切换频率 (Hz)
    motion_score: float = 0.0
    
    # 状态标记
    is_lip_detected: bool = True
    is_lip_moving: bool = False
    is_lip_closed: bool = False
    consecutive_closed_frames: int = 0
    consecutive_no_detection_frames: int = 0
    last_landmarks_98: Optional[np.ndarray] = None
    head_motion_norm: float = 0.0
    lip_local_motion_norm: float = 0.0
    is_head_motion_dominant: bool = False
    
    # 最终得分 (用于融合)
    lip_score: float = 0.0


class LipMotionAnalyzer:
    """
    唇部运动分析器
    
    核心功能:
    1. 从 98 点关键点提取唇部指标
    2. 多帧累积分析运动模式
    3. 生成唇部运动得分 (用于融合 TalkNet 得分)
    
    得分设计:
    - lip_score > 0: 嘴唇在运动，可能在说话
    - lip_score = 0: 中性状态
    - lip_score < 0: 嘴唇闭合或未检测到，惩罚
    - lip_score = -inf: 强制判定为非说话人
    """
    
    def __init__(
        self,
        history_frames: int = 15,
        lar_closed_threshold: float = 0.15,
        lar_open_threshold: float = 0.25,
        motion_threshold: float = 0.05,
        consecutive_closed_for_penalty: int = 8,
        consecutive_no_detect_for_penalty: int = 5,
        enable_debug: bool = False,
    ):
        """
        Args:
            history_frames: 保留的历史帧数
            lar_closed_threshold: LAR 低于此值认为嘴唇闭合
            lar_open_threshold: LAR 高于此值认为嘴唇张开
            motion_threshold: LAR 变化标准差阈值，高于此值认为在运动
            consecutive_closed_for_penalty: 连续闭合帧数阈值，超过则强制惩罚
            consecutive_no_detect_for_penalty: 连续未检测帧数阈值
            enable_debug: 是否打印调试信息
        """
        self.history_frames = history_frames
        self.lar_closed_threshold = lar_closed_threshold
        self.lar_open_threshold = lar_open_threshold
        self.motion_threshold = motion_threshold
        self.consecutive_closed_for_penalty = consecutive_closed_for_penalty
        self.consecutive_no_detect_for_penalty = consecutive_no_detect_for_penalty
        self.enable_debug = enable_debug
        
        # 每个 track 的运动状态
        self.states: Dict[int, LipMotionState] = {}

        # Head-motion gate:
        # If the face as a whole moves more than the lip's local residual motion,
        # treat positive lip evidence as unreliable. This targets side-face/head
        # motion cases where projected landmark jitter can look like mouth motion.
        self.head_motion_gate_enable = True
        self.head_motion_min_norm = 0.015
        self.head_to_lip_motion_ratio = 1.8
    
    def extract_lip_metrics(
        self,
        landmarks_98: np.ndarray,
        timestamp: float = 0.0,
    ) -> Optional[LipMetrics]:
        """
        从 98 点关键点提取唇部指标
        
        Args:
            landmarks_98: (98, 2) 关键点坐标
            timestamp: 时间戳
            
        Returns:
            LipMetrics 或 None (如果关键点无效)
        """
        if landmarks_98 is None or landmarks_98.shape[0] < 96:
            return None
        
        try:
            # 提取关键点
            upper_center = landmarks_98[LIP_UPPER_CENTER]  # 上唇内侧中点
            lower_center = landmarks_98[LIP_LOWER_CENTER]  # 下唇内侧中点
            inner_left = landmarks_98[LIP_INNER_LEFT_CORNER]   # 内嘴角左
            inner_right = landmarks_98[LIP_INNER_RIGHT_CORNER] # 内嘴角右
            
            # 检查坐标有效性
            all_pts = [upper_center, lower_center, inner_left, inner_right]
            for pt in all_pts:
                if not np.isfinite(pt).all():
                    return None
            
            # 计算垂直距离 (上唇中点 → 下唇中点)
            vertical_dist = np.linalg.norm(lower_center - upper_center)
            
            # 计算水平距离 (左内嘴角 → 右内嘴角)
            horizontal_dist = np.linalg.norm(inner_right - inner_left)
            
            # 避免除零
            if horizontal_dist < 1e-6:
                return None
            
            # 计算 LAR (Lip Aperture Ratio)
            lar = vertical_dist / horizontal_dist
            
            return LipMetrics(
                lar=float(lar),
                vertical_dist=float(vertical_dist),
                horizontal_dist=float(horizontal_dist),
                is_valid=True,
                timestamp=timestamp,
            )
            
        except Exception as e:
            if self.enable_debug:
                print(f"[LipMotion] extract_lip_metrics error: {e}", flush=True)
            return None
    
    def update_track(
        self,
        track_id: int,
        landmarks_98: Optional[np.ndarray],
        timestamp: float,
    ) -> LipMotionState:
        """
        更新指定 track 的唇部运动状态
        
        Args:
            track_id: 跟踪 ID
            landmarks_98: 98 点关键点 (可为 None 表示未检测到)
            timestamp: 时间戳
            
        Returns:
            更新后的 LipMotionState
        """
        # 获取或创建状态
        if track_id not in self.states:
            self.states[track_id] = LipMotionState(
                track_id=track_id,
                history=deque(maxlen=self.history_frames),
                timestamps=deque(maxlen=self.history_frames),
            )
        
        state = self.states[track_id]
        
        # 提取唇部指标
        metrics = None
        if landmarks_98 is not None:
            metrics = self.extract_lip_metrics(landmarks_98, timestamp)
        
        if metrics is None or not metrics.is_valid:
            # 未检测到唇部
            state.is_lip_detected = False
            state.consecutive_no_detection_frames += 1
            state.consecutive_closed_frames = 0
            state.is_head_motion_dominant = False
            
            # 连续未检测采用分级惩罚：短时抖动/遮挡不直接一票否决。
            if state.consecutive_no_detection_frames >= (self.consecutive_no_detect_for_penalty * 3):
                state.lip_score = -float('inf')
            elif state.consecutive_no_detection_frames >= self.consecutive_no_detect_for_penalty:
                state.lip_score = -0.8
            else:
                state.lip_score = -0.5
            
            if self.enable_debug:
                print(
                    f"[LipMotion] track_id={track_id}: NO_DETECT "
                    f"(consecutive={state.consecutive_no_detection_frames})",
                    flush=True
                )
            
            return state
        
        # 有效检测
        state.is_lip_detected = True
        state.consecutive_no_detection_frames = 0
        head_motion_norm, lip_local_motion_norm = self._estimate_head_vs_lip_motion(
            state.last_landmarks_98,
            landmarks_98,
        )
        state.head_motion_norm = head_motion_norm
        state.lip_local_motion_norm = lip_local_motion_norm
        state.is_head_motion_dominant = self._is_head_motion_dominant(
            head_motion_norm,
            lip_local_motion_norm,
        )
        
        # 更新历史
        state.history.append(metrics.lar)
        state.timestamps.append(timestamp)
        
        # 判断嘴唇状态
        if metrics.lar < self.lar_closed_threshold:
            state.is_lip_closed = True
            state.is_lip_moving = False
            state.consecutive_closed_frames += 1
        else:
            state.is_lip_closed = False
            state.consecutive_closed_frames = 0
        
        # 计算运动统计
        if len(state.history) >= 3:
            lar_arr = np.array(list(state.history))
            state.lar_mean = float(np.mean(lar_arr))
            state.lar_std = float(np.std(lar_arr))
            state.lar_range = float(np.max(lar_arr) - np.min(lar_arr))
            
            # 判断是否在运动
            state.is_lip_moving = state.lar_std > self.motion_threshold
        
        # 计算唇部运动得分
        state.lip_score = self._compute_lip_score(state, metrics)
        state.last_landmarks_98 = np.asarray(landmarks_98, dtype=np.float32).copy()
        
        if self.enable_debug:
            print(
                f"[LipMotion] track_id={track_id}: LAR={metrics.lar:.3f}, "
                f"std={state.lar_std:.3f}, moving={state.is_lip_moving}, "
                f"closed={state.is_lip_closed}, head={state.head_motion_norm:.4f}, "
                f"lip_local={state.lip_local_motion_norm:.4f}, "
                f"head_gate={state.is_head_motion_dominant}, score={state.lip_score:.3f}",
                flush=True
            )
        
        return state
    
    def _compute_lip_score(self, state: LipMotionState, metrics: LipMetrics) -> float:
        """
        计算唇部运动得分
        
        得分范围设计:
        - 正分 (0 ~ 1): 嘴唇在运动，越高表示运动越明显
        - 零分: 中性状态
        - 负分 (-1 ~ 0): 嘴唇闭合或静止
        - -inf: 强制判定非说话人
        
        算法(频率主导):
        1. 估计唇部开合切换频率 (toggle_rate)
        2. 结合 LAR 方差/幅度判断是否存在稳定开合节律
        3. 常开或常闭(抖动率≈0)都视为非说话
        """
        # 连续闭口同样采用分级惩罚，避免短时误判直接 veto。
        if state.consecutive_closed_frames >= (self.consecutive_closed_for_penalty * 3):
            return -float('inf')
        if state.consecutive_closed_frames >= self.consecutive_closed_for_penalty:
            return -0.8

        # 历史不足时，不做激进正判，只给轻微非说话惩罚。
        # 这里必须与自适应窗口对齐(常见为4帧)，否则会长期输出固定分值。
        min_hist = 4
        if len(state.history) < min_hist or len(state.timestamps) < min_hist:
            return -0.2

        lar_arr = np.array(list(state.history), dtype=np.float32)
        ts_arr = np.array(list(state.timestamps), dtype=np.float32)

        toggle_rate = self._estimate_toggle_rate_hz(lar_arr, ts_arr)
        state.lar_toggle_rate = toggle_rate

        # 常开/常闭都归为非说话，但短窗(如4帧)里不能只靠单一条件就判死。
        # 否则会出现用户持续说话但长期卡在 -0.5 的情况。
        min_std = max(0.008, self.motion_threshold * 0.35)
        min_range = max(0.03, self.motion_threshold * 1.20)
        min_toggle_hz = 0.45

        # 一阶差分平均绝对值：补充短窗下“单向开/合”运动证据。
        d = np.diff(lar_arr)
        delta_mean_abs = float(np.mean(np.abs(d))) if d.size > 0 else 0.0

        # 静态判定使用组合条件，避免任一条件触发导致过度负判。
        likely_static_motion = (
            toggle_rate < min_toggle_hz
            and state.lar_std < (min_std * 1.25)
            and state.lar_range < (min_range * 1.25)
            and delta_mean_abs < 0.010
        )
        likely_closed_static = (
            state.lar_mean < self.lar_closed_threshold
            and state.lar_range < (min_range * 1.40)
            and delta_mean_abs < 0.012
        )

        if likely_static_motion or likely_closed_static:
            # 常闭给予更强惩罚；常开但无节律也视作不说话
            if state.lar_mean < self.lar_closed_threshold:
                return -0.8
            return -0.5

        # 有明显开合频率时，按频率与幅度联合给分
        # 频率达到 ~3Hz 和幅度达到 ~0.12 时接近满分
        freq_component = min(1.0, toggle_rate / 3.0)
        amp_component = min(1.0, state.lar_range / 0.12)
        stability_component = min(1.0, state.lar_std / 0.08)
        trend_component = min(1.0, delta_mean_abs / 0.02)

        raw_score = (
            0.45 * freq_component
            + 0.25 * amp_component
            + 0.15 * stability_component
            + 0.15 * trend_component
        )
        if state.is_head_motion_dominant:
            return min(0.0, float(raw_score) - 0.35)
        return float(np.clip(raw_score, -1.0, 1.0))

    def _estimate_head_vs_lip_motion(
        self,
        prev_landmarks: Optional[np.ndarray],
        cur_landmarks: Optional[np.ndarray],
    ) -> Tuple[float, float]:
        if prev_landmarks is None or cur_landmarks is None:
            return 0.0, 0.0
        if prev_landmarks.shape[0] < 96 or cur_landmarks.shape[0] < 96:
            return 0.0, 0.0

        prev = np.asarray(prev_landmarks, dtype=np.float32)
        cur = np.asarray(cur_landmarks, dtype=np.float32)
        finite_cur = np.isfinite(cur).all(axis=1)
        if not finite_cur.any():
            return 0.0, 0.0

        valid_cur = cur[finite_cur]
        span = np.max(valid_cur, axis=0) - np.min(valid_cur, axis=0)
        face_scale = float(max(np.max(span), 1.0))

        non_lip_idx = np.asarray(NON_LIP_INDICES, dtype=np.int32)
        non_lip_idx = non_lip_idx[non_lip_idx < min(prev.shape[0], cur.shape[0])]
        non_lip_valid = (
            np.isfinite(prev[non_lip_idx]).all(axis=1)
            & np.isfinite(cur[non_lip_idx]).all(axis=1)
        )
        if int(np.count_nonzero(non_lip_valid)) < 8:
            return 0.0, 0.0

        non_lip_delta = cur[non_lip_idx][non_lip_valid] - prev[non_lip_idx][non_lip_valid]
        global_delta = np.median(non_lip_delta, axis=0)
        head_motion = float(np.median(np.linalg.norm(non_lip_delta, axis=1)) / face_scale)

        lip_idx = np.asarray(LIP_ALL_INDICES, dtype=np.int32)
        lip_idx = lip_idx[lip_idx < min(prev.shape[0], cur.shape[0])]
        lip_valid = (
            np.isfinite(prev[lip_idx]).all(axis=1)
            & np.isfinite(cur[lip_idx]).all(axis=1)
        )
        if int(np.count_nonzero(lip_valid)) < 4:
            return head_motion, 0.0

        lip_delta = cur[lip_idx][lip_valid] - prev[lip_idx][lip_valid]
        lip_residual = lip_delta - global_delta
        lip_local_motion = float(np.median(np.linalg.norm(lip_residual, axis=1)) / face_scale)
        return head_motion, lip_local_motion

    def _is_head_motion_dominant(self, head_motion_norm: float, lip_local_motion_norm: float) -> bool:
        if not self.head_motion_gate_enable:
            return False
        if head_motion_norm < self.head_motion_min_norm:
            return False
        return head_motion_norm > (lip_local_motion_norm * self.head_to_lip_motion_ratio + 1e-6)

    def _estimate_toggle_rate_hz(self, lar_arr: np.ndarray, ts_arr: np.ndarray) -> float:
        """
        估计唇部开合切换频率(Hz):
        - 基于一阶差分符号变化计数(上升/下降切换)
        - 过滤掉微小抖动，避免噪声导致误判
        """
        if lar_arr.size < 4 or ts_arr.size < 4:
            return 0.0

        dt = float(ts_arr[-1] - ts_arr[0])
        if dt <= 1e-3:
            return 0.0

        d = np.diff(lar_arr)
        if d.size < 3:
            return 0.0

        # 按当前幅度自适应门限，过滤微小噪声
        delta_th = max(0.004, float(stateful_abs_mean(d)) * 0.35)
        active = np.abs(d) > delta_th
        signs = np.sign(d)

        toggles = 0
        for i in range(1, signs.size):
            if not (active[i - 1] and active[i]):
                continue
            if signs[i - 1] == 0 or signs[i] == 0:
                continue
            if signs[i - 1] != signs[i]:
                toggles += 1

        return float(toggles / dt)


    def get_state(self, track_id: int) -> Optional[LipMotionState]:
        """获取指定 track 的状态"""
        return self.states.get(track_id)

    def remap_track_id(self, old_id: int, new_id: int) -> None:
        """在 tracker ID 跳变时迁移唇部历史状态。"""
        old_id = int(old_id)
        new_id = int(new_id)
        if old_id == new_id:
            return

        state = self.states.pop(old_id, None)
        if state is not None:
            state.track_id = new_id
            self.states[new_id] = state
    
    def get_lip_score(self, track_id: int) -> float:
        """获取指定 track 的唇部运动得分"""
        state = self.states.get(track_id)
        if state is None:
            return 0.0  # 无历史数据，返回中性分
        return state.lip_score

    def cleanup_stale_tracks(self, active_track_ids: List[int]) -> None:
        """清理不再活跃的 track"""
        stale_ids = [tid for tid in self.states if tid not in active_track_ids]
        for tid in stale_ids:
            del self.states[tid]

    def reset(self) -> None:
        """重置所有状态"""
        self.states.clear()

    def set_history_frames(self, history_frames: int) -> bool:
        """
        动态调整历史窗口帧数。

        Returns:
            True: 窗口长度发生变化
            False: 与当前值相同，无需更新
        """
        new_frames = max(3, int(history_frames))
        if new_frames == self.history_frames:
            return False

        self.history_frames = new_frames
        for state in self.states.values():
            state.history = deque(state.history, maxlen=new_frames)
            state.timestamps = deque(state.timestamps, maxlen=new_frames)
        return True


def stateful_abs_mean(arr: np.ndarray) -> float:
    if arr.size == 0:
        return 0.0
    return float(np.mean(np.abs(arr)))


def compute_fused_score(
    asd_score: float,
    lip_score: float,
    asd_weight: float = 0.7,
    lip_weight: float = 0.3,
    lip_veto_threshold: float = -0.8,
    talknet_confirm_thresh: float = 0.55,
) -> Tuple[float, bool]:
    """
    融合 TalkNet ASD 得分和唇部运动得分
    
    Args:
        asd_score: TalkNet ASD 得分 (probability [0, 1])
        lip_score: 唇部运动得分 (-1 ~ 1, 或 -inf)
        asd_weight: ASD 得分权重
        lip_weight: 唇部得分权重
        lip_veto_threshold: 唇部得分低于此值时，触发一票否决
        talknet_confirm_thresh: 只有 TalkNet 分数超过此阈值才允许唇部奖励
        
    Returns:
        (fused_score, is_vetoed)
        - fused_score: 融合后的得分
        - is_vetoed: 是否被唇部运动一票否决
    """
    # ---- 一票否决逻辑 ----
    # 1. 唇部未检测到或持续闭合 → 强制非说话人
    if lip_score == -float('inf'):
        return -float('inf'), True
    
    # 2. 唇部得分极低 → 触发否决
    if lip_score < lip_veto_threshold:
        return -float('inf'), True
    
    # ---- 正常融合 (非线性门控融合) ----
    # asd_score 是 TalkNet 概率 [0, 1]
    # lip_score 是唇部运动得分 [-1, 1]
    
    # 1. 唇部静止但未触发绝对否决 (lip_score < 0)
    # 此时大概率没说话，TalkNet 的高分可能是噪音干扰，施加惩罚
    if lip_score < 0:
        # lip_score 越接近 -1，惩罚越重
        penalty_factor = max(0.35, 1.0 + lip_score)  # 保留惩罚但避免分数被压得过低
        fused = asd_score * penalty_factor
        
    # 2. 唇部在运动 (lip_score > 0)
    # 可能是说话，也可能是嚼口香糖。必须依赖 TalkNet 来确认是否有声音同步。
    else:
        if asd_score > talknet_confirm_thresh:
            # TalkNet 认为有声音，且嘴巴在动，互相印证，给予奖励
            # lip_score 越大，奖励越多，最高不超过 1.0
            bonus = lip_score * max(0.0, min(lip_weight, 0.2))
            fused = min(1.0, asd_score + bonus)
        else:
            # 嘴巴在动，但 TalkNet 认为没声音 (大概率在嚼口香糖或无声说话)
            # 低置信度时给一个受限增益，减少“差一点不过线”的漏检。
            base_bonus = lip_score * max(0.0, min(lip_weight, 0.2))
            low_conf_bonus = min(0.08, base_bonus * 0.45)
            fused = min(1.0, asd_score + low_conf_bonus)
            
    return fused, False


# ============== 测试代码 ==============
if __name__ == "__main__":
    import random
    
    analyzer = LipMotionAnalyzer(enable_debug=True)
    
    # 模拟 98 点关键点
    def make_landmarks(lar: float = 0.3) -> np.ndarray:
        pts = np.random.randn(98, 2) * 10 + 100
        # 设置唇部关键点模拟特定 LAR
        pts[LIP_INNER_LEFT_CORNER] = [80, 100]
        pts[LIP_INNER_RIGHT_CORNER] = [120, 100]
        horizontal = 40  # 水平距离
        vertical = lar * horizontal  # 垂直距离
        pts[LIP_UPPER_CENTER] = [100, 100 - vertical / 2]
        pts[LIP_LOWER_CENTER] = [100, 100 + vertical / 2]
        return pts
    
    # 模拟说话 (LAR 变化)
    print("=== 模拟说话 (LAR 变化) ===")
    for i in range(10):
        lar = 0.2 + 0.2 * np.sin(i * 0.5)  # 0.0 ~ 0.4 变化
        landmarks = make_landmarks(lar)
        state = analyzer.update_track(track_id=1, landmarks_98=landmarks, timestamp=i * 0.1)
    
    # 模拟嘴唇闭合
    print("\n=== 模拟嘴唇持续闭合 ===")
    for i in range(10):
        landmarks = make_landmarks(lar=0.05)  # LAR 很小
        state = analyzer.update_track(track_id=2, landmarks_98=landmarks, timestamp=i * 0.1)
    
    # 模拟未检测到嘴唇
    print("\n=== 模拟未检测到嘴唇 ===")
    for i in range(6):
        state = analyzer.update_track(track_id=3, landmarks_98=None, timestamp=i * 0.1)
