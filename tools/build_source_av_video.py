#!/usr/bin/env python3
import argparse
import json
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2


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


def _resolve_path(run_dir: Path, value: Optional[str]) -> Optional[Path]:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else run_dir / path


def _estimate_fps(frame_rows: List[Dict[str, Any]], default_fps: float) -> float:
    timestamps = [float(row["timestamp"]) for row in frame_rows if row.get("timestamp") is not None]
    if len(timestamps) >= 2:
        span = timestamps[-1] - timestamps[0]
        if span > 1e-6:
            return max(1.0, min(120.0, float(len(timestamps) - 1) / span))
    return float(max(default_fps, 1e-3))


def _wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return float(handle.getnframes()) / float(handle.getframerate())


def _write_silent_video(
    *,
    run_dir: Path,
    frame_rows: List[Dict[str, Any]],
    output_path: Path,
    fps: float,
) -> int:
    writer = None
    frame_count = 0
    try:
        for row in frame_rows:
            frame_path = _resolve_path(run_dir, row.get("path"))
            if frame_path is None or not frame_path.exists():
                raise FileNotFoundError(f"source frame not found: {frame_path}")

            frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"failed to read source frame: {frame_path}")

            height, width = frame.shape[:2]
            if writer is None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(output_path), fourcc, float(fps), (int(width), int(height)))
                if not writer.isOpened():
                    raise RuntimeError(f"failed to open video writer: {output_path}")
            writer.write(frame)
            frame_count += 1
    finally:
        if writer is not None:
            writer.release()
    return frame_count


def _mux_audio(
    *,
    silent_video: Path,
    audio_path: Path,
    output_path: Path,
    audio_trim_sec: float,
    audio_delay_sec: float,
    audio_gain_db: float,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found; cannot mux audio into mp4")

    command = [ffmpeg, "-y", "-loglevel", "error", "-i", str(silent_video)]
    if audio_trim_sec > 1e-6:
        command.extend(["-ss", f"{audio_trim_sec:.6f}"])
    command.extend(["-i", str(audio_path)])

    command.extend(["-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy"])
    audio_filters: List[str] = []
    if audio_delay_sec > 1e-6:
        delay_ms = int(round(audio_delay_sec * 1000.0))
        audio_filters.append(f"adelay={delay_ms}")
    if abs(float(audio_gain_db)) > 1e-6:
        audio_filters.append(f"volume={float(audio_gain_db):.3f}dB")
    if audio_filters:
        command.extend(["-af", ",".join(audio_filters)])
    command.extend(["-c:a", "aac", "-shortest", str(output_path)])

    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        reason = completed.stderr.strip() or f"ffmpeg_exit_{completed.returncode}"
        raise RuntimeError(f"ffmpeg mux failed: {reason}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an MP4 with audio from ASD source_frames.jsonl and asd_input_audio.wav."
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="Capture run directory containing manifest.json.")
    parser.add_argument("--manifest", type=Path, help="Defaults to <run-dir>/manifest.json.")
    parser.add_argument("--frames-index", type=Path, help="Defaults to manifest paths.source_frames_index.")
    parser.add_argument("--audio", type=Path, help="Defaults to manifest paths.asd_input_audio.")
    parser.add_argument("--output", type=Path, help="Defaults to manifest paths.asd_input_video.")
    parser.add_argument("--fps", type=float, help="Override output FPS. By default it is estimated from frame timestamps.")
    parser.add_argument(
        "--audio-offset-sec",
        type=float,
        default=0.0,
        help="Additional offset applied to audio. Positive delays audio; negative makes audio earlier.",
    )
    parser.add_argument(
        "--audio-gain-db",
        type=float,
        default=0.0,
        help="Optional gain for the muxed preview audio only, e.g. 12.0. Does not modify asd_input_audio.wav.",
    )
    parser.add_argument("--keep-temp", action="store_true", help="Keep the temporary silent mp4.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    manifest_path = args.manifest or (run_dir / "manifest.json")
    manifest = _read_json(manifest_path)
    paths = manifest.get("paths", {})

    frames_index = args.frames_index or _resolve_path(run_dir, paths.get("source_frames_index"))
    audio_path = args.audio or _resolve_path(run_dir, paths.get("asd_input_audio"))
    output_path = args.output or _resolve_path(run_dir, paths.get("asd_input_video")) or (run_dir / "asd_input_video.mp4")

    if frames_index is None or not frames_index.exists():
        raise SystemExit(f"source frames index not found: {frames_index}")
    if audio_path is None or not audio_path.exists():
        raise SystemExit(f"audio file not found: {audio_path}")
    if output_path is None:
        raise SystemExit("output path is required")

    frame_rows = _read_jsonl(frames_index)
    if not frame_rows:
        raise SystemExit(f"no frame rows found in {frames_index}")

    fps = float(args.fps) if args.fps and args.fps > 0 else _estimate_fps(
        frame_rows,
        float(manifest.get("input_video_fps_config") or manifest.get("process_video_fps") or 30.0),
    )

    video_first_ts = float(frame_rows[0]["timestamp"])
    video_last_ts = float(frame_rows[-1]["timestamp"])
    audio_first_ts = manifest.get("audio", {}).get("first_timestamp")
    if audio_first_ts is None:
        audio_first_ts = video_first_ts
    audio_first_ts = float(audio_first_ts)

    # If audio starts earlier than video, trim the leading audio. If audio starts
    # later, delay the audio track so the first audible sample lands correctly.
    audio_delta = (video_first_ts - audio_first_ts) - float(args.audio_offset_sec)
    audio_trim_sec = max(audio_delta, 0.0)
    audio_delay_sec = max(-audio_delta, 0.0)

    temp_dir = Path(tempfile.mkdtemp(prefix="source_av_", dir=str(output_path.parent)))
    silent_video = temp_dir / f"{output_path.stem}_silent.mp4"
    try:
        frame_count = _write_silent_video(
            run_dir=run_dir,
            frame_rows=frame_rows,
            output_path=silent_video,
            fps=fps,
        )
        _mux_audio(
            silent_video=silent_video,
            audio_path=audio_path,
            output_path=output_path,
            audio_trim_sec=audio_trim_sec,
            audio_delay_sec=audio_delay_sec,
            audio_gain_db=float(args.audio_gain_db),
        )
    finally:
        if args.keep_temp:
            print(f"[build_source_av_video] kept temp_dir={temp_dir}")
        else:
            shutil.rmtree(temp_dir, ignore_errors=True)

    video_duration = float(frame_count) / float(fps)
    audio_duration = _wav_duration(audio_path)
    print(
        "[build_source_av_video] wrote %s frames=%d fps=%.6f video_duration=%.3fs "
        "audio_duration=%.3fs audio_trim=%.3fs audio_delay=%.3fs audio_gain=%.1fdB"
        % (
            output_path,
            frame_count,
            fps,
            video_duration,
            audio_duration,
            audio_trim_sec,
            audio_delay_sec,
            float(args.audio_gain_db),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
