#!/usr/bin/env python3
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.signal import resample_poly

for _legacy_name, _legacy_value in {
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "object": object,
}.items():
    if _legacy_name not in np.__dict__:
        setattr(np, _legacy_name, _legacy_value)

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import asd_config as C
from asd import create_face_detector
from asd.buffers import TrackBuffer
from asd.tracking import Tracker

try:
    from asd_sdk import ASDSDK, InferenceRequest, TrackInput
except Exception:
    SDK_DIR = ROOT_DIR / "sdk"
    if str(SDK_DIR) not in sys.path:
        sys.path.insert(0, str(SDK_DIR))
    from asd_sdk import ASDSDK, InferenceRequest, TrackInput


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=True, sort_keys=True))
            handle.write("\n")


def _resolve_path(run_dir: Path, value: Optional[str]) -> Optional[Path]:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else run_dir / path


def _load_wav_mono(path: Path) -> Tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        raw = handle.readframes(handle.getnframes())
    if sample_width != 2:
        raise ValueError(f"expected 16-bit PCM wav, got sample_width={sample_width}: {path}")
    pcm = np.frombuffer(raw, dtype="<i2")
    if channels > 1:
        pcm = pcm.reshape(-1, channels)[:, 0]
    return pcm.astype(np.float32, copy=False), int(sample_rate)


def _resample_audio(audio: np.ndarray, src_sample_rate: int, dst_sample_rate: int) -> np.ndarray:
    src_sample_rate = int(src_sample_rate)
    dst_sample_rate = int(dst_sample_rate)
    if src_sample_rate <= 0 or dst_sample_rate <= 0:
        raise ValueError(f"invalid sample rate conversion: src={src_sample_rate} dst={dst_sample_rate}")
    if src_sample_rate == dst_sample_rate:
        return audio.astype(np.float32, copy=False)
    gcd = math.gcd(src_sample_rate, dst_sample_rate)
    up = dst_sample_rate // gcd
    down = src_sample_rate // gcd
    return resample_poly(audio.astype(np.float32, copy=False), up, down).astype(np.float32, copy=False)


def _transcode_audio_to_wav(source: Path, output: Path, sample_rate: int) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("ffmpeg not found; direct offline video inference requires ffmpeg to extract audio")

    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(int(sample_rate)),
        "-c:a",
        "pcm_s16le",
        str(output),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        reason = completed.stderr.strip() or f"ffmpeg_exit_{completed.returncode}"
        raise SystemExit(f"failed to extract/normalize audio from {source}: {reason}")


def _audio_range(
    audio: np.ndarray,
    *,
    audio_start_ts: float,
    sample_rate: int,
    t_start: float,
    t_end: float,
) -> np.ndarray:
    if t_end <= t_start:
        return np.zeros((0,), dtype=np.float32)
    start = int(round((float(t_start) - float(audio_start_ts)) * float(sample_rate)))
    end = int(round((float(t_end) - float(audio_start_ts)) * float(sample_rate)))
    dst_len = max(end - start, 0)
    if dst_len <= 0:
        return np.zeros((0,), dtype=np.float32)
    out = np.zeros((dst_len,), dtype=np.float32)
    src_start = max(start, 0)
    src_end = min(end, int(audio.shape[0]))
    dst_start = max(-start, 0)
    copy_len = max(src_end - src_start, 0)
    if copy_len > 0:
        out[dst_start : dst_start + copy_len] = audio[src_start:src_end]
    return out


def _track_result_map(sdk_result) -> Dict[int, Any]:
    return {int(item.track_id): item for item in sdk_result.tracks}


def _serialize_track(track, sdk_track=None) -> Dict[str, Any]:
    row = {
        "track_id": int(track.track_id),
        "bbox": [int(v) for v in track.bbox],
        "used_for_inference": sdk_track is not None,
        "is_active": False,
        "talknet_raw": None,
        "talknet_smooth": None,
        "lip_score": None,
        "fused_score": None,
        "is_lip_vetoed": False,
    }
    if sdk_track is not None:
        row.update(
            {
                "is_active": bool(sdk_track.is_active),
                "talknet_raw": float(sdk_track.talknet_raw),
                "talknet_smooth": float(sdk_track.talknet_smooth),
                "lip_score": float(sdk_track.lip_score),
                "fused_score": float(sdk_track.fused_score),
                "is_lip_vetoed": bool(sdk_track.is_lip_vetoed),
            }
        )
    return row


class OfflineASDRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.detector = create_face_detector(args.face_detector_type)
        self.tracker = Tracker()
        self.track_buffers: Dict[int, TrackBuffer] = {}
        self.lip_landmark_detector = None
        self.lip_motion_analyzer = None
        self.enable_lip_motion = bool(args.enable_lip_motion)
        if self.enable_lip_motion:
            self.enable_lip_motion = self._init_lip_motion()
        self.asd_sdk = ASDSDK(
            talknet_backend=ASDSDK.build_default_talknet_backend(),
            mode=args.mode,
            enable_lip_motion=self.enable_lip_motion,
            smooth_alpha=float(getattr(C, "SMOOTH_ALPHA", 0.55)),
            speak_on_thresh=float(getattr(C, "SPEAK_ON_THRESH", 0.52)),
            speak_off_thresh=float(getattr(C, "SPEAK_OFF_THRESH", 0.38)),
            lip_motion_weight=float(getattr(C, "LIP_MOTION_WEIGHT", 0.2)),
            lip_veto_threshold=float(getattr(C, "LIP_VETO_THRESHOLD", -0.8)),
            lip_talknet_confirm_thresh=float(getattr(C, "LIP_TALKNET_CONFIRM_THRESH", 0.50)),
            threshold_min_score=float(getattr(C, "THRESHOLD_MODE_MIN_SCORE", 0.30)),
            threshold_relative_margin=float(getattr(C, "THRESHOLD_MODE_RELATIVE_MARGIN", 0.12)),
            top_scorer_min_score=float(getattr(C, "TOP_SCORER_MIN_SCORE", 0.40)),
            top_scorer_override_threshold=float(getattr(C, "TOP_SCORER_OVERRIDE_THRESHOLD", 0.70)),
        )

    def _init_lip_motion(self) -> bool:
        lip_weights = str(getattr(C, "LIP_LANDMARK_WEIGHTS_PATH", "") or "")
        if not lip_weights or not os.path.isfile(lip_weights):
            print(f"[offline_asd] lip motion disabled: weights not found: {lip_weights}", flush=True)
            return False
        try:
            from asd.lip_landmark_wrapper import LipLandmarkDetector
            from asd.lip_motion_analyzer import LipMotionAnalyzer

            device = self.args.lip_device or ("cuda" if getattr(C, "MODEL_DEVICE", "cpu") == "cuda" else "cpu")
            self.lip_landmark_detector = LipLandmarkDetector(
                weights_path=lip_weights,
                device=device,
                enable_debug=bool(getattr(C, "LIP_MOTION_DEBUG", False)),
                use_trt=bool(getattr(C, "LIP_USE_TRT", False)),
                trt_engine_path=str(getattr(C, "LIP_TRT_ENGINE_PATH", "") or ""),
            )
            self.lip_motion_analyzer = LipMotionAnalyzer(
                history_frames=int(getattr(C, "LIP_MOTION_HISTORY_FRAMES", 15)),
                lar_closed_threshold=float(getattr(C, "LIP_LAR_CLOSED_THRESHOLD", 0.15)),
                lar_open_threshold=float(getattr(C, "LIP_LAR_OPEN_THRESHOLD", 0.25)),
                motion_threshold=float(getattr(C, "LIP_MOTION_STD_THRESHOLD", 0.05)),
                consecutive_closed_for_penalty=int(getattr(C, "LIP_CONSECUTIVE_CLOSED_FOR_PENALTY", 8)),
                consecutive_no_detect_for_penalty=int(getattr(C, "LIP_CONSECUTIVE_NO_DETECT_FOR_PENALTY", 5)),
                enable_debug=bool(getattr(C, "LIP_MOTION_DEBUG", False)),
            )
            print(f"[offline_asd] lip motion enabled. weights={lip_weights}", flush=True)
            return True
        except Exception as exc:
            print(f"[offline_asd] lip motion disabled: {exc}", flush=True)
            self.lip_landmark_detector = None
            self.lip_motion_analyzer = None
            return False

    def _update_lip_states(self, tracks, frame_bgr: np.ndarray, timestamp: float) -> None:
        if not (self.enable_lip_motion and self.lip_landmark_detector and self.lip_motion_analyzer):
            return
        if not tracks:
            self.lip_motion_analyzer.cleanup_stale_tracks([])
            return
        try:
            bboxes = [track.bbox for track in tracks]
            track_ids = [int(track.track_id) for track in tracks]
            lip_results = self.lip_landmark_detector.detect(frame_bgr, bboxes, track_ids)
            for result in lip_results:
                self.lip_motion_analyzer.update_track(
                    track_id=int(result.track_id),
                    landmarks_98=result.landmarks_98 if result.is_valid else None,
                    timestamp=float(timestamp),
                )
            self.lip_motion_analyzer.cleanup_stale_tracks(track_ids)
        except Exception as exc:
            print(f"[offline_asd] lip detection warning: {exc}", flush=True)

    def _get_lip_score(self, track_id: int, timestamp: float) -> Optional[float]:
        if not (self.enable_lip_motion and self.lip_motion_analyzer):
            return None
        state = self.lip_motion_analyzer.get_state(int(track_id))
        if state is None or len(state.timestamps) == 0:
            return -0.5
        latest_ts = float(state.timestamps[-1])
        if (float(timestamp) - latest_ts) > float(self.args.lip_state_max_age_sec):
            return -0.5
        return float(state.lip_score)

    def _get_recent_track_window(self, track_id: int) -> Tuple[np.ndarray, List[float], bool]:
        buf = self.track_buffers.get(int(track_id))
        if buf is None or len(buf.frames) == 0:
            return np.empty((0,), dtype=np.float32), [], False

        target_frames = max(1, int(getattr(C, "TARGET_VIDEO_FRAMES", 8)))
        if self.args.require_full_window and not buf.has_enough():
            return np.empty((0,), dtype=np.float32), [], False

        frame_count = min(len(buf.frames), target_frames)
        frames = list(buf.frames)[-frame_count:]
        ts_list = list(buf.timestamps)[-frame_count:]
        if not frames or not ts_list:
            return np.empty((0,), dtype=np.float32), [], False
        return np.stack(frames, axis=0), ts_list, frame_count < target_frames

    def _process_frame(
        self,
        *,
        frame: np.ndarray,
        frame_ts: float,
        source_frame_index: int,
        image_path: Optional[str],
        audio: np.ndarray,
        sample_rate: int,
        audio_start_ts: float,
        source_name: str,
        video_fps: Optional[float],
    ) -> Dict[str, Any]:
        detections = self.detector.detect(frame)
        tracks = self.tracker.update(detections)

        active_ids = {int(track.track_id) for track in tracks}
        for tid in list(self.track_buffers.keys()):
            if tid not in active_ids:
                self.track_buffers.pop(tid, None)

        h, w = frame.shape[:2]
        for track in tracks:
            x1, y1, x2, y2 = [int(v) for v in track.bbox]
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(x1 + 1, min(w, x2))
            y2 = max(y1 + 1, min(h, y2))
            face_roi = frame[y1:y2, x1:x2, :].copy()
            face_roi = cv2.resize(face_roi, (int(C.FACE_CROP_SIZE), int(C.FACE_CROP_SIZE)))
            self.track_buffers.setdefault(int(track.track_id), TrackBuffer()).add_frame(face_roi, frame_ts)

        self._update_lip_states(tracks, frame, frame_ts)

        valid_tracks = []
        req_tracks = []
        short_window_track_ids: List[int] = []
        min_video_frames = int(getattr(C, "MIN_VIDEO_FRAMES", 1)) if self.args.require_full_window else 1
        for track in tracks:
            video_clip, ts_list, is_short_window = self._get_recent_track_window(int(track.track_id))
            if video_clip.size == 0 or len(ts_list) < min_video_frames:
                continue
            t_audio_start = min(ts_list) - float(C.AUDIO_MARGIN_SEC)
            t_audio_end = max(ts_list) + float(C.AUDIO_MARGIN_SEC)
            audio_clip = _audio_range(
                audio,
                audio_start_ts=audio_start_ts,
                sample_rate=sample_rate,
                t_start=t_audio_start,
                t_end=t_audio_end,
            )
            if audio_clip.size == 0:
                continue
            if is_short_window:
                short_window_track_ids.append(int(track.track_id))
            valid_tracks.append(track)
            req_tracks.append(
                TrackInput(
                    track_id=int(track.track_id),
                    audio_clip=audio_clip,
                    video_clip=video_clip,
                    lip_score=self._get_lip_score(int(track.track_id), frame_ts),
                )
            )

        max_faces = int(self.args.max_faces)
        if max_faces > 0 and len(valid_tracks) > max_faces:
            keep_indices = sorted(range(len(valid_tracks)), key=lambda i: int(valid_tracks[i].track_id))[:max_faces]
            keep_indices.sort()
            valid_tracks = [valid_tracks[i] for i in keep_indices]
            req_tracks = [req_tracks[i] for i in keep_indices]
            keep_ids = {int(track.track_id) for track in valid_tracks}
            short_window_track_ids = [tid for tid in short_window_track_ids if tid in keep_ids]

        sdk_track_map: Dict[int, Any] = {}
        status = "ok"
        debug: Dict[str, Any] = {
            "frame_source": source_name,
            "video_fps": None if video_fps is None else float(video_fps),
            "require_full_window": bool(self.args.require_full_window),
            "short_window_track_ids": short_window_track_ids,
        }
        if req_tracks:
            sdk_result = self.asd_sdk.infer(InferenceRequest(tracks=req_tracks, timestamp=frame_ts))
            sdk_track_map = _track_result_map(sdk_result)
            debug.update(dict(sdk_result.debug or {}))
        elif tracks:
            status = "warming_up_or_missing_audio" if self.args.require_full_window else "missing_audio_or_invalid_track"
        else:
            status = "no_tracks"

        return {
            "source_frame_index": int(source_frame_index),
            "timestamp": frame_ts,
            "image_path": image_path,
            "status": status,
            "mode": self.args.mode,
            "lip_motion_enabled": self.enable_lip_motion,
            "active_speaker_ids": [tid for tid, item in sdk_track_map.items() if bool(item.is_active)],
            "tracks": [_serialize_track(track, sdk_track_map.get(int(track.track_id))) for track in tracks],
            "debug": debug,
        }

    def _run_from_dataset(
        self,
        *,
        run_dir: Path,
        frame_rows: List[Dict[str, Any]],
        audio: np.ndarray,
        sample_rate: int,
        audio_start_ts: float,
    ) -> List[Dict[str, Any]]:
        predictions: List[Dict[str, Any]] = []
        frame_rows = sorted(frame_rows, key=lambda item: float(item["timestamp"]))
        if self.args.max_frames > 0:
            frame_rows = frame_rows[: int(self.args.max_frames)]

        for seq, frame_row in enumerate(frame_rows, start=1):
            frame_ts = float(frame_row["timestamp"])
            image_path = _resolve_path(run_dir, frame_row.get("path"))
            source_frame_index = int(frame_row.get("frame_index", seq))
            image_path_str = frame_row.get("path")
            if image_path is None or not image_path.exists():
                predictions.append(
                    {
                        "source_frame_index": source_frame_index,
                        "timestamp": frame_ts,
                        "image_path": image_path_str,
                        "status": "missing_image",
                        "tracks": [],
                        "debug": {"frame_source": "dataset"},
                    }
                )
                continue

            frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if frame is None:
                predictions.append(
                    {
                        "source_frame_index": source_frame_index,
                        "timestamp": frame_ts,
                        "image_path": image_path_str,
                        "status": "decode_failed",
                        "tracks": [],
                        "debug": {"frame_source": "dataset"},
                    }
                )
                continue

            predictions.append(
                self._process_frame(
                    frame=frame,
                    frame_ts=frame_ts,
                    source_frame_index=source_frame_index,
                    image_path=image_path_str,
                    audio=audio,
                    sample_rate=sample_rate,
                    audio_start_ts=audio_start_ts,
                    source_name="dataset",
                    video_fps=None,
                )
            )

            if seq % self.args.log_every == 0:
                print(f"[offline_asd] processed {seq}/{len(frame_rows)} frames", flush=True)

        return predictions

    def _run_from_video(
        self,
        *,
        input_video: Path,
        audio: np.ndarray,
        sample_rate: int,
        audio_start_ts: float,
    ) -> List[Dict[str, Any]]:
        cap = cv2.VideoCapture(str(input_video))
        if not cap.isOpened():
            raise SystemExit(f"failed to open input video: {input_video}")

        fps = float(self.args.video_fps) if self.args.video_fps and self.args.video_fps > 0 else float(
            cap.get(cv2.CAP_PROP_FPS) or 0.0
        )
        if fps <= 0:
            cap.release()
            raise SystemExit(
                f"failed to infer video fps from {input_video}; pass --video-fps explicitly"
            )

        predictions: List[Dict[str, Any]] = []
        frame_index = 0
        try:
            while True:
                if self.args.max_frames > 0 and frame_index >= int(self.args.max_frames):
                    break
                ok, frame = cap.read()
                if not ok:
                    break
                frame_ts = float(frame_index) / float(fps)
                predictions.append(
                    self._process_frame(
                        frame=frame,
                        frame_ts=frame_ts,
                        source_frame_index=frame_index,
                        image_path=f"{input_video}#frame={frame_index}",
                        audio=audio,
                        sample_rate=sample_rate,
                        audio_start_ts=audio_start_ts,
                        source_name="input_video",
                        video_fps=fps,
                    )
                )
                frame_index += 1
                if frame_index % self.args.log_every == 0:
                    print(f"[offline_asd] processed {frame_index} frames from {input_video}", flush=True)
        finally:
            cap.release()

        return predictions

    def run(self) -> List[Dict[str, Any]]:
        if self.args.input_video is not None:
            input_video = self.args.input_video.resolve()
            if not input_video.exists():
                raise SystemExit(f"input video not found: {input_video}")

            with tempfile.TemporaryDirectory(prefix="offline_asd_video_") as temp_dir_str:
                temp_dir = Path(temp_dir_str)
                audio_input = self.args.audio.resolve() if self.args.audio is not None else input_video
                if audio_input.suffix.lower() == ".wav":
                    audio, sample_rate = _load_wav_mono(audio_input)
                    expected_sample_rate = int(getattr(C, "AUDIO_SAMPLE_RATE", 16000))
                    if sample_rate != expected_sample_rate:
                        audio = _resample_audio(audio, sample_rate, expected_sample_rate)
                        sample_rate = expected_sample_rate
                else:
                    normalized_audio = temp_dir / "audio_16k.wav"
                    _transcode_audio_to_wav(audio_input, normalized_audio, int(getattr(C, "AUDIO_SAMPLE_RATE", 16000)))
                    audio, sample_rate = _load_wav_mono(normalized_audio)
                audio_start_ts = float(self.args.audio_start_ts) if self.args.audio_start_ts is not None else 0.0
                return self._run_from_video(
                    input_video=input_video,
                    audio=audio,
                    sample_rate=sample_rate,
                    audio_start_ts=audio_start_ts,
                )

        manifest = _read_json(self.args.manifest)
        run_dir = self.args.run_dir or Path(manifest.get("run_dir") or self.args.manifest.parent)
        paths = manifest.get("paths", {})

        frames_index = self.args.frames_index or _resolve_path(run_dir, paths.get("source_frames_index"))
        audio_path = self.args.audio or _resolve_path(run_dir, paths.get("asd_input_audio"))
        if frames_index is None or not frames_index.exists():
            raise SystemExit(f"source frame index not found: {frames_index}")
        if audio_path is None or not audio_path.exists():
            raise SystemExit(f"ASD input audio not found: {audio_path}")

        frame_rows = _read_jsonl(frames_index)
        audio, sample_rate = _load_wav_mono(audio_path)
        if sample_rate != int(getattr(C, "AUDIO_SAMPLE_RATE", 16000)):
            raise SystemExit(f"expected audio sample_rate={C.AUDIO_SAMPLE_RATE}, got {sample_rate}")

        audio_start_ts = float(self.args.audio_start_ts) if self.args.audio_start_ts is not None else float(
            manifest.get("audio", {}).get("first_timestamp", frame_rows[0]["timestamp"])
        )
        return self._run_from_dataset(
            run_dir=run_dir,
            frame_rows=frame_rows,
            audio=audio,
            sample_rate=sample_rate,
            audio_start_ts=audio_start_ts,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ASD offline on a captured dataset or a direct input video, with per-frame inference."
    )
    parser.add_argument("--input-video", type=Path, help="Direct input video path. If set, the script decodes every frame from the video.")
    parser.add_argument("--run-dir", type=Path, help="Capture run directory.")
    parser.add_argument("--manifest", type=Path, help="Defaults to <run-dir>/manifest.json.")
    parser.add_argument("--frames-index", type=Path, help="Defaults to manifest paths.source_frames_index.")
    parser.add_argument("--audio", type=Path, help="Defaults to manifest paths.asd_input_audio.")
    parser.add_argument("--audio-start-ts", type=float, default=None, help="Override audio timeline start timestamp.")
    parser.add_argument("--video-fps", type=float, default=None, help="Override decoded video FPS for direct video mode.")
    parser.add_argument("--output", type=Path, help="Defaults to <run-dir>/offline_eval/predictions.jsonl.")
    parser.add_argument("--mode", default=getattr(C, "ACTIVE_SPEAKER_DISPLAY_MODE", "top_scorer"))
    parser.add_argument("--face-detector-type", default=getattr(C, "FACE_DETECTOR_TYPE", "yolo"))
    parser.add_argument("--max-faces", type=int, default=int(getattr(C, "MAX_FACES_FOR_ASD", 0)))
    parser.add_argument("--max-frames", type=int, default=0, help="Optional frame limit for debugging. 0 means all frames.")
    parser.add_argument(
        "--require-full-window",
        action="store_true",
        help="Keep the old behavior: only run ASD after a full TalkNet window is accumulated.",
    )
    parser.add_argument("--enable-lip-motion", dest="enable_lip_motion", action="store_true", default=bool(getattr(C, "ENABLE_LIP_MOTION", False)))
    parser.add_argument("--disable-lip-motion", dest="enable_lip_motion", action="store_false")
    parser.add_argument("--lip-state-max-age-sec", type=float, default=float(getattr(C, "LIP_STATE_MAX_AGE_SEC", 0.5)))
    parser.add_argument("--lip-device", default="", help="Defaults to config MODEL_DEVICE.")
    parser.add_argument("--log-every", type=int, default=50)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.input_video is None:
        if args.manifest is None:
            if args.run_dir is None:
                raise SystemExit("--input-video or --run-dir/--manifest is required")
            args.manifest = args.run_dir / "manifest.json"
        if args.run_dir is None:
            manifest = _read_json(args.manifest)
            args.run_dir = Path(manifest.get("run_dir") or args.manifest.parent)
        if args.output is None:
            args.output = args.run_dir / "offline_eval" / "predictions.jsonl"
    else:
        if args.output is None:
            args.output = args.input_video.with_name(f"{args.input_video.stem}.offline_asd.jsonl")

    runner = OfflineASDRunner(args)
    predictions = runner.run()
    _write_jsonl(args.output, predictions)
    print(f"[offline_asd] wrote {args.output} frames={len(predictions)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
