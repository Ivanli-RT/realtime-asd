# asd/tracking.py - 改进版(解决ID跳变)
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import numpy as np
from collections import deque
from config import asd_config as C

@dataclass
class FaceDet:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float

@dataclass
class Track:
    track_id: int
    bbox: Tuple[int, int, int, int]
    missed: int = 0
    age: int = 0
    velocity: Tuple[float, float] = (0.0, 0.0)
    bbox_history: deque = field(default_factory=lambda: deque(maxlen=5))
    
    def predict_bbox(self) -> Tuple[int, int, int, int]:
        """基于速度预测下一帧位置"""
        if self.velocity == (0.0, 0.0) or len(self.bbox_history) < 2:
            return self.bbox
        x1, y1, x2, y2 = self.bbox
        vx, vy = self.velocity
        return (
            int(x1 + vx), int(y1 + vy),
            int(x2 + vx), int(y2 + vy)
        )
    
    def update_velocity(self, new_bbox: Tuple[int, int, int, int]):
        """更新速度估计(EMA平滑)"""
        if self.bbox_history:
            old_cx = (self.bbox[0] + self.bbox[2]) / 2
            old_cy = (self.bbox[1] + self.bbox[3]) / 2
            new_cx = (new_bbox[0] + new_bbox[2]) / 2
            new_cy = (new_bbox[1] + new_bbox[3]) / 2
            
            vx = new_cx - old_cx
            vy = new_cy - old_cy
            
            alpha = 0.7
            self.velocity = (
                alpha * vx + (1 - alpha) * self.velocity[0],
                alpha * vy + (1 - alpha) * self.velocity[1]
            )

def iou(boxA, boxB) -> float:
    """计算IoU"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    if interArea == 0:
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(areaA + areaB - interArea + 1e-6)

def giou(boxA, boxB) -> float:
    """
    Generalized IoU (GIoU)
    - 比标准IoU更适合追踪(考虑框的相对位置)
    - 值域: [-1, 1], 越大越相似
    """
    iou_val = iou(boxA, boxB)
    
    # 计算包围盒
    xA = min(boxA[0], boxB[0])
    yA = min(boxA[1], boxB[1])
    xB = max(boxA[2], boxB[2])
    yB = max(boxA[3], boxB[3])
    
    convex_area = (xB - xA) * (yB - yA)
    union_area = (boxA[2]-boxA[0])*(boxA[3]-boxA[1]) + (boxB[2]-boxB[0])*(boxB[3]-boxB[1]) - iou_val*convex_area
    
    if convex_area == 0:
        return iou_val
    
    giou_val = iou_val - (convex_area - union_area) / convex_area
    return giou_val

class Tracker:
    def __init__(self):
        self.tracks: Dict[int, Track] = {}
        self.next_id = 0
        self.iou_threshold = getattr(C, 'IOU_THRESHOLD', 0.35)
        self.max_track_missed = getattr(C, 'MAX_TRACK_MISSED', 15)
        self.max_tracks = getattr(C, 'MAX_TRACKS', 8)

    def update(self, detections: List[FaceDet]) -> List[Track]:
        """
        改进的匹配策略:
        1. 预测track位置
        2. GIoU + 中心距离混合
        3. 贪心匹配(按得分排序)
        """
        # 预测位置
        predicted_bboxes = {}
        for tid, tr in self.tracks.items():
            predicted_bboxes[tid] = tr.predict_bbox()
        
        # 构建匹配得分
        matches = []
        for tid, pred_bbox in predicted_bboxes.items():
            for j, det in enumerate(detections):
                det_bbox = (det.x1, det.y1, det.x2, det.y2)
                
                # 混合得分: GIoU(主) + 检测置信度(辅)
                giou_score = giou(pred_bbox, det_bbox)
                conf_score = det.score
                
                # GIoU已经在[-1,1]范围,直接加权
                score = 0.8 * giou_score + 0.2 * conf_score
                
                matches.append((score, tid, j))
        
        # 按得分排序(贪心匹配)
        matches.sort(reverse=True)
        
        matched_tracks = set()
        matched_dets = set()
        final_matches = []
        
        for score, tid, j in matches:
            if score < 0.2:  # 低于阈值直接停止
                break
            if tid not in matched_tracks and j not in matched_dets:
                final_matches.append((tid, j))
                matched_tracks.add(tid)
                matched_dets.add(j)
        
        # 更新匹配的tracks
        for tid, j in final_matches:
            det = detections[j]
            new_bbox = (det.x1, det.y1, det.x2, det.y2)
            
            tr = self.tracks[tid]
            tr.update_velocity(new_bbox)
            tr.bbox_history.append(new_bbox)
            tr.bbox = new_bbox
            tr.missed = 0
            tr.age += 1
            self.tracks[tid] = tr
        
        # 处理未匹配的tracks
        for tid in list(self.tracks.keys()):
            if tid not in matched_tracks:
                tr = self.tracks[tid]
                tr.missed += 1
                if tr.missed > self.max_track_missed:
                    del self.tracks[tid]
                    print(f"[Tracker] delete track {tid} after {tr.missed} misses", flush=True)
        
        # 创建新tracks
        for j in range(len(detections)):
            if j not in matched_dets and len(self.tracks) < self.max_tracks:
                det = detections[j]
                tid = self.next_id
                self.next_id += 1
                new_bbox = (det.x1, det.y1, det.x2, det.y2)
                track = Track(track_id=tid, bbox=new_bbox, missed=0, age=1)
                track.bbox_history.append(new_bbox)
                self.tracks[tid] = track
                print(f"[Tracker] create new track {tid}", flush=True)
        
        return list(self.tracks.values())