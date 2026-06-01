#!/usr/bin/env python3
import argparse
import bisect
import json
import shutil
import subprocess
import tempfile
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


def _load_results_from_dir(results_dir: Path) -> Tuple[List[float], List[Dict[str, Any]]]:
    records: List[Dict[str, Any]] = []
    for path in sorted(results_dir.glob("*.json")):
        payload = _read_json(path)
        if "timestamp" not in payload:
            continue
        payload["_path"] = str(path)
        records.append(payload)

    records.sort(key=lambda item: float(item["timestamp"]))
    return [float(item["timestamp"]) for item in records], records


def _load_results_from_jsonl(predictions_path: Path) -> Tuple[List[float], List[Dict[str, Any]]]:
    records = [row for row in _read_jsonl(predictions_path) if row.get("timestamp") is not None]
    for row in records:
        row["_path"] = str(predictions_path)
    records.sort(key=lambda item: float(item["timestamp"]))
    return [float(item["timestamp"]) for item in records], records


def _load_results(path: Path) -> Tuple[List[float], List[Dict[str, Any]]]:
    if path.is_dir():
        return _load_results_from_dir(path)
    return _load_results_from_jsonl(path)


def _pick_result(
    frame_ts: Optional[float],
    timestamps: List[float],
    records: List[Dict[str, Any]],
    max_time_diff: float,
) -> Optional[Dict[str, Any]]:
    if not records:
        return None
    if frame_ts is None:
        return records[0]

    idx = bisect.bisect_right(timestamps, frame_ts) - 1
    if idx < 0:
        idx = 0

    candidate = records[idx]
    if abs(float(candidate["timestamp"]) - frame_ts) <= max_time_diff:
        return candidate
    return None


def _format_score(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        score = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(score):
        return str(value)
    return f"{score:+.2f}"


def _draw_text_box(frame, lines: List[str], origin: Tuple[int, int], color: Tuple[int, int, int]) -> None:
    if not lines:
        return
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.48
    thickness = 1
    padding = 5
    line_h = 18
    widths = []
    for line in lines:
        (w, _), _ = cv2.getTextSize(line, font, scale, thickness)
        widths.append(w)
    box_w = max(widths) + padding * 2
    box_h = line_h * len(lines) + padding * 2
    h, w = frame.shape[:2]
    x = max(0, min(x, max(0, w - box_w - 1)))
    y = max(box_h + 1, min(y, h - 1))
    top_left = (x, y - box_h)
    bottom_right = (x + box_w, y)
    overlay = frame.copy()
    cv2.rectangle(overlay, top_left, bottom_right, (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.rectangle(frame, top_left, bottom_right, color, 1)
    for i, line in enumerate(lines):
        text_y = top_left[1] + padding + 13 + i * line_h
        cv2.putText(frame, line, (x + padding, text_y), font, scale, (245, 245, 245), thickness, cv2.LINE_AA)


def _draw_header(frame, result: Optional[Dict[str, Any]], frame_ts: Optional[float]) -> None:
    if not result:
        return
    result_ts = float(result.get("timestamp", 0.0))
    delta = None if frame_ts is None else frame_ts - result_ts
    active_ids = result.get("active_speaker_ids", []) or []
    frame_index = result.get("frame_index", result.get("source_frame_index", "?"))
    lines = [
        f"result frame {frame_index}  mode={result.get('mode', 'n/a')}",
        f"active={active_ids}  dt={delta:+.3f}s" if delta is not None else f"active={active_ids}",
    ]
    _draw_text_box(frame, lines, (10, 44), (80, 180, 255))


def _draw_result(frame, result: Optional[Dict[str, Any]]) -> None:
    if not result:
        return

    active_ids = {int(tid) for tid in result.get("active_speaker_ids", [])}
    tracks = result.get("tracks", []) or []
    for track in tracks:
        bbox = track.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox]
        tid = int(track.get("track_id", -1))
        is_active = bool(track.get("is_active", tid in active_ids))
        is_vetoed = bool(track.get("is_lip_vetoed", False))
        color = (0, 255, 0) if is_active else ((80, 80, 255) if is_vetoed else (0, 180, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        state = "SPK" if is_active else ("VETO" if is_vetoed else "silent")
        lines = [
            f"ID {tid}  {state}",
            f"fused={_format_score(track.get('fused_score'))}  tn={_format_score(track.get('talknet_smooth'))}",
            f"raw={_format_score(track.get('talknet_raw'))}  lip={_format_score(track.get('lip_score'))}",
        ]
        _draw_text_box(frame, lines, (x1, max(58, y1 - 8)), color)


def _mux_audio(silent_video: Path, input_video: Path, output_video: Path) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        shutil.copy2(silent_video, output_video)
        return False

    base_command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(silent_video),
        "-i",
        str(input_video),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        "copy",
    ]
    last_error = ""
    for audio_codec in ("copy", "aac"):
        command = base_command + ["-c:a", audio_codec, "-shortest", str(output_video)]
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        if completed.returncode == 0:
            return True
        last_error = completed.stderr.strip() or f"ffmpeg_exit_{completed.returncode}"

    shutil.copy2(silent_video, output_video)
    print(f"[render_asd_overlay_video] ffmpeg mux failed, wrote silent video: {last_error}")
    return False


def _infer_predictions_default(run_dir: Path, input_video: Optional[Path]) -> Optional[Path]:
    candidates: List[Path] = []
    if input_video is not None:
        candidates.append(run_dir / f"{input_video.stem}.offline_asd.jsonl")
        candidates.append(run_dir / f"{input_video.stem}.offline_asd_100.jsonl")
    candidates.append(run_dir / "offline_eval" / "predictions.jsonl")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _infer_defaults(run_dir: Path, manifest: Dict[str, Any]) -> Tuple[Optional[Path], Optional[Path]]:
    paths = manifest.get("paths", {})
    input_video = _resolve_path(
        run_dir,
        paths.get("asd_input_video") or paths.get("raw_input_video") or paths.get("output_mp4"),
    )
    results_path = _resolve_path(run_dir, paths.get("frame_results_dir"))
    if results_path is None:
        results_path = _infer_predictions_default(run_dir, input_video)
    return input_video, results_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render ASD overlay video from captured input AV and offline ASD results."
    )
    parser.add_argument("--run-dir", type=Path, help="Capture run directory containing manifest.json.")
    parser.add_argument("--manifest", type=Path, help="Path to manifest.json. Defaults to <run-dir>/manifest.json.")
    parser.add_argument("--input", type=Path, help="Input AV file. Defaults to manifest paths.asd_input_video.")
    parser.add_argument("--results", type=Path, help="Directory with frame_results/*.json.")
    parser.add_argument("--predictions", type=Path, help="Offline predictions jsonl, such as record.offline_asd.jsonl.")
    parser.add_argument("--output", type=Path, help="Output overlay video path. Defaults to <run-dir>/asd_output_video.mp4.")
    parser.add_argument("--max-time-diff", type=float, default=0.35, help="Max seconds to hold a JSON result on nearby frames.")
    parser.add_argument("--no-audio", action="store_true", help="Do not copy audio from input video.")
    parser.add_argument("--keep-temp", action="store_true", help="Keep the temporary silent overlay video.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir
    manifest_path = args.manifest
    if manifest_path is None and run_dir is not None:
        manifest_path = run_dir / "manifest.json"
    manifest: Dict[str, Any] = {}
    if manifest_path is not None and manifest_path.exists():
        manifest = _read_json(manifest_path)
        run_dir = run_dir or Path(manifest.get("run_dir") or manifest_path.parent)
    elif run_dir is not None:
        run_dir = Path(run_dir)

    input_default = None
    results_default = None
    if run_dir is not None:
        input_default, results_default = _infer_defaults(run_dir, manifest)

    input_video = args.input or input_default
    results_path = args.predictions or args.results or results_default

    if args.output is not None:
        output_video = args.output
    elif run_dir is not None:
        output_video = run_dir / "asd_output_video.mp4"
    else:
        raise SystemExit("--output is required when neither --run-dir nor --manifest is provided")

    if input_video is None or not input_video.exists():
        raise SystemExit(f"input video not found: {input_video}")
    if results_path is None or not results_path.exists():
        raise SystemExit(f"results/predictions not found: {results_path}")

    timestamps, records = _load_results(results_path)
    if not records:
        raise SystemExit(f"no timestamped ASD results found in {results_path}")

    input_meta = manifest.get("input_video") or manifest.get("raw_input_video") or {}
    first_ts = input_meta.get("first_timestamp")
    manifest_fps = input_meta.get("fps_used")

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise SystemExit(f"failed to open input video: {input_video}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or manifest_fps or 25.0)
    if fps <= 0:
        fps = float(manifest_fps or 25.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_video.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix="asd_overlay_", dir=str(output_video.parent)))
    silent_video = temp_dir / f"{output_video.stem}_silent.mp4"
    writer = cv2.VideoWriter(str(silent_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise SystemExit(f"failed to open video writer: {silent_video}")

    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_ts = float(frame_index) / fps
            if first_ts is not None:
                frame_ts = float(first_ts) + frame_ts
            result = _pick_result(frame_ts, timestamps, records, args.max_time_diff)
            _draw_result(frame, result)
            _draw_header(frame, result, frame_ts)
            writer.write(frame)
            frame_index += 1
    finally:
        cap.release()
        writer.release()

    muxed = False
    if args.no_audio:
        shutil.copy2(silent_video, output_video)
    else:
        muxed = _mux_audio(silent_video, input_video, output_video)

    if not args.keep_temp:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print(
        "[render_asd_overlay_video] wrote %s frames=%d fps=%.3f audio=%s"
        % (output_video, frame_index, fps, "copied" if muxed else "silent")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
