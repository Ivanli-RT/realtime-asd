from collections import deque
import threading
from typing import Deque, List, Tuple

import numpy as np
from config import asd_config as C

class AudioBuffer:
    """
    简单的时间窗口音频缓存：
    - 每次 add_block 追加一段 1D int16 / float32 音频（16kHz 单声道）
    - 内部用 deque 存 (block, t_block_start)
    - get_window 时按 [t_now - win_sec, t_now] 拼接出一段连续音频
    - get_range 时按 [t_start, t_end] 精确裁剪并拼接，便于对齐 ts_list
    """
    def __init__(self, max_sec: float = 10.0, sample_rate: int = 16000):
        self.blocks: Deque[np.ndarray] = deque()
        self.timestamps: Deque[float] = deque()  # 每个 block 的“起始时间”戳（秒）
        self.max_sec = max_sec  # 缓存上限（秒），防止无限增长
        self.sample_rate = sample_rate  # 采样率，默认 16kHz
        self._lock = threading.RLock()

    def add_block(self, audio_block: np.ndarray, t: float):
        """
        audio_block: 1D np.ndarray, int16 或 float32, 16kHz 单声道
        t: 这个 block 的“起始时间戳”（对应第 0 个采样点的时间）
        """
        if audio_block is None or audio_block.size == 0:
            return

        # 统一成 float32
        if audio_block.dtype != np.float32:
            audio_block = audio_block.astype(np.float32)

        with self._lock:
            self.blocks.append(audio_block)
            self.timestamps.append(t)
            # 只在这里做剪枝（while），不在 get_window / get_range 里边遍历边删
            self._prune_old(t)

    def _prune_old(self, t_now: float):
        """
        移除 (t_now - max_sec, t_now] 之前的数据
        注意：这里只用 while，不用 for-in deque，避免 iterator 被修改。
        """
        t_min = t_now - self.max_sec
        while self.timestamps and self.timestamps[0] < t_min:
            self.timestamps.popleft()
            self.blocks.popleft()

    def get_window(self, t_end: float, window_sec: float) -> np.ndarray:
        """
        返回 [t_end - window_sec, t_end] 内的音频拼接结果（float32）
        —— 旧接口保留，内部复用 get_range
        """
        t_start = t_end - window_sec
        return self.get_range(t_start, t_end)

    def get_range(self, t_start: float, t_end: float) -> np.ndarray:
        """
        返回 [t_start, t_end] 区间内的音频（float32, 单声道）
        - 使用 block 的起始时间 + 采样数 / 采样率 来计算每个 block 的时间范围
        - 只做快照，不在遍历时修改 deque
        """
        if t_end <= t_start:
            return np.zeros((0,), dtype=np.float32)
        sr = float(self.sample_rate)

        # 1) 对 (blocks, timestamps) 做快照，防止迭代时被外部回调修改
        with self._lock:
            if not self.timestamps:
                return np.zeros((0,), dtype=np.float32)
            blocks_snap = list(self.blocks)
            ts_snap = list(self.timestamps)

        selected_pieces: List[np.ndarray] = []

        for blk, t_blk_start in zip(blocks_snap, ts_snap):
            if blk is None or blk.size == 0:
                continue

            # 当前 block 覆盖的时间区间 [b_start, b_end]
            b_start = t_blk_start
            b_end = t_blk_start + len(blk) / sr

            # 和 [t_start, t_end] 没有交集就跳过
            if b_end <= t_start or b_start >= t_end:
                continue

            # 计算在当前 block 中应该截取的采样区间
            # 把时间差换算成采样点索引
            s = max(0, int((t_start - b_start) * sr)) if t_start > b_start else 0
            e = min(len(blk), int((t_end - b_start) * sr)) if t_end < b_end else len(blk)

            if e > s:
                piece = blk[s:e].astype(np.float32, copy=False)
                selected_pieces.append(piece)

        if not selected_pieces:
            return np.zeros((0,), dtype=np.float32)

        audio = np.concatenate(selected_pieces, axis=0).astype(np.float32)
        return audio



class TrackBuffer:
    """
    每个 track 独立的视频缓存：
    - 存人脸 ROI + 时间戳
    - 支持：
        * has_enough(): 是否积累够一段 clip
        * get_last_n(n): 取最近 n 帧 + 对应时间戳
        * get_window(): 老接口，按时间窗口取（兼容之前用法）
    """
    def __init__(self):
        self.frames: Deque[np.ndarray] = deque()
        self.timestamps: Deque[float] = deque()
        self.first_ts: float = None  # track 第一次出现的时间

    def add_frame(self, face_roi: np.ndarray, t: float):
        """
        face_roi: (H, W, 3) BGR
        t: 这一帧的时间戳（秒）
        """
        if face_roi is None or face_roi.size == 0:
            return

        self.frames.append(face_roi)
        self.timestamps.append(t)

        if self.first_ts is None:
            self.first_ts = t

        # 按时间限制缓存长度，这里用 CLIP_SECONDS 的 2 倍做上限
        if self.timestamps:
            t_now = self.timestamps[-1]
            while self.timestamps and (t_now - self.timestamps[0]) > C.CLIP_SECONDS * 2.0:
                self.timestamps.popleft()
                self.frames.popleft()

    def age(self, t_now: float) -> float:
        """
        这个 track 存在了多久（秒）
        """
        if self.first_ts is None:
            return 0.0
        return max(0.0, t_now - self.first_ts)

    def has_enough(self) -> bool:
        """
        是否已经积累够一段完整 clip：
        这里以 TARGET_VIDEO_FRAMES 为基准
        """
        return len(self.frames) >= C.TARGET_VIDEO_FRAMES

    def get_last_n(self, n: int) -> Tuple[np.ndarray, List[float]]:
        """
        取最近 n 帧 + 对应时间戳
        - 如果帧数不够，返回空数组和空列表
        """
        if len(self.frames) < n:
            return np.empty((0,)), []

        # deque -> list 之后切片，保证不会在遍历时被修改
        frames_list = list(self.frames)[-n:]
        ts_list = list(self.timestamps)[-n:]

        video = np.stack(frames_list, axis=0)  # (n, H, W, C)
        return video, ts_list

    def get_window(self, t_end: float, window_sec: float) -> Tuple[np.ndarray, List[float]]:
        """
        旧接口：按时间窗口 [t_end - window_sec, t_end] 取帧
        保留以防其它地方还在用
        """
        if not self.timestamps:
            return np.empty((0,)), []

        t_start = t_end - window_sec

        # 做快照，避免遍历时被外部修改
        frames_snap = list(self.frames)
        ts_snap = list(self.timestamps)

        selected = [
            (f, ts) for f, ts in zip(frames_snap, ts_snap)
            if t_start <= ts <= t_end
        ]

        if not selected:
            return np.empty((0,)), []

        frames, ts_list = zip(*selected)
        video = np.stack(frames, axis=0)  # (T, H, W, C)
        return video, list(ts_list)
