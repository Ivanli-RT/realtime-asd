#!/usr/bin/env python3
import json
import queue
import shutil
import signal
import subprocess
import threading
import time
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import rospy
from roslib.message import get_message_class
from sensor_msgs.msg import Image

try:
    from cv_bridge import CvBridge
except Exception:
    CvBridge = None

try:
    from rostopic import get_topic_class as _get_topic_class
except Exception:
    _get_topic_class = None


def _stamp_to_sec(msg: Any) -> float:
    header = getattr(msg, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is not None:
        try:
            value = float(stamp.to_sec())
            if value > 0:
                return value
        except Exception:
            pass
    return float(rospy.Time.now().to_sec())


def _image_msg_to_bgr(msg: Image, bridge: Optional[Any]) -> np.ndarray:
    if bridge is not None:
        frame = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        return np.ascontiguousarray(frame)

    encoding = (msg.encoding or "").lower()
    channels_by_encoding = {
        "bgr8": 3,
        "rgb8": 3,
        "8uc3": 3,
        "bgra8": 4,
        "rgba8": 4,
        "mono8": 1,
        "8uc1": 1,
    }
    if encoding not in channels_by_encoding:
        raise ValueError(f"unsupported image encoding without cv_bridge: {msg.encoding}")

    channels = channels_by_encoding[encoding]
    row_bytes = int(msg.step)
    if row_bytes <= 0:
        raise ValueError(f"invalid image step: {msg.step}")

    arr = np.frombuffer(msg.data, dtype=np.uint8)
    expected = row_bytes * int(msg.height)
    if arr.nbytes < expected:
        raise ValueError(f"image data too short: got {arr.nbytes} bytes, expected {expected}")

    rows = arr[:expected].reshape((int(msg.height), row_bytes))
    rows = rows[:, : int(msg.width) * channels]
    image = rows.reshape((int(msg.height), int(msg.width), channels))

    if encoding in ("bgr8", "8uc3"):
        return np.ascontiguousarray(image)
    if encoding == "rgb8":
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if encoding == "bgra8":
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if encoding == "rgba8":
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def _pick_audio_payload(msg: Any) -> Tuple[Optional[str], Optional[Any]]:
    for field_name in ("audio_data", "data", "filtered_data"):
        payload = getattr(msg, field_name, None)
        if payload is None:
            continue
        try:
            if len(payload) == 0:
                continue
        except Exception:
            continue
        return field_name, payload
    return None, None


def _coerce_pcm16(payload: Any, field_name: str) -> np.ndarray:
    if field_name == "audio_data":
        arr = np.asarray(payload)
        if arr.dtype.kind == "u":
            return arr.astype(np.uint16, copy=False).view(np.int16)
        if arr.dtype.kind in ("i", "u") and arr.size > 0:
            min_value = int(arr.min())
            max_value = int(arr.max())
            if min_value >= 0 and max_value > np.iinfo(np.int16).max:
                return arr.astype(np.uint16, copy=False).view(np.int16)
        return arr.astype(np.int16, copy=False)
    return np.asarray(payload, dtype=np.int16)


def _extract_mono_pcm16(
    msg: Any,
    *,
    fallback_channels: int,
    fallback_sample_rate: int,
    audio_channel: int,
) -> Tuple[np.ndarray, int, Dict[str, Any]]:
    if getattr(msg, "is_valid", True) is False:
        raise ValueError("audio message marked invalid")

    field_name, payload = _pick_audio_payload(msg)
    if payload is None or field_name is None:
        raise ValueError("empty audio payload")

    arr = _coerce_pcm16(payload, field_name)
    total_samples = int(arr.size)
    if total_samples <= 0:
        raise ValueError("empty audio payload")

    channels = getattr(msg, "channels", None)
    if channels in (None, 0):
        channels = getattr(msg, "channel", None)
    if channels in (None, 0):
        channels = fallback_channels
    channels = int(channels)
    if channels <= 0:
        raise ValueError(f"invalid channel count: {channels}")
    if total_samples % channels != 0:
        raise ValueError(f"payload length {total_samples} is not divisible by channels={channels}")

    selected_channel = int(np.clip(int(audio_channel), 0, channels - 1))
    if channels == 1:
        mono = arr
        selected_channel = 0
    else:
        mono = arr.reshape((-1, channels))[:, selected_channel]

    sample_rate = getattr(msg, "sample_rate", None)
    sample_rate = int(sample_rate) if sample_rate not in (None, 0) else int(fallback_sample_rate)
    if sample_rate <= 0:
        raise ValueError(f"invalid sample rate: {sample_rate}")

    pcm = np.array(mono, dtype=np.int16, copy=True)
    return pcm, sample_rate, {
        "field_name": field_name,
        "channels": channels,
        "selected_channel": selected_channel,
        "sample_rate": sample_rate,
        "samples_per_packet": int(pcm.size),
    }


def _resolve_audio_msg_class(topic: str, explicit_type: str) -> Tuple[Any, str]:
    candidates: List[str] = []
    if explicit_type:
        candidates.append(explicit_type)

    if not candidates and _get_topic_class is not None:
        try:
            msg_class, _, _ = _get_topic_class(topic, blocking=False)
            if hasattr(msg_class, "_type"):
                return msg_class, str(msg_class._type)
        except Exception:
            pass

    candidates.extend(["audio/AudioData", "avvtn_msgs/AudioData"])
    for msg_type in candidates:
        msg_class = get_message_class(msg_type)
        if msg_class is not None:
            return msg_class, msg_type
    raise RuntimeError(
        "cannot resolve audio message type; pass _audio_msg_type:=audio/AudioData "
        "or the concrete type used by your topic"
    )


class RosAVRecorder:
    def __init__(self) -> None:
        rospy.init_node("ros_av_recorder", anonymous=True)

        self.video_topic = str(rospy.get_param("~video_topic", "/realsense_head/color/image_raw"))
        self.audio_topic = str(rospy.get_param("~audio_topic", "/zj_humanoid/audio/microphone/audio_data_raw"))
        self.audio_msg_type = str(rospy.get_param("~audio_msg_type", "audio/AudioData"))
        self.audio_channel = int(rospy.get_param("~audio_channel", 0))
        self.fallback_audio_channels = int(rospy.get_param("~fallback_audio_channels", 1))
        self.fallback_audio_rate = int(rospy.get_param("~fallback_audio_sample_rate", 48000))

        self.video_fps = float(rospy.get_param("~video_fps", 30.0))
        self.default_video_fps = float(rospy.get_param("~default_video_fps", 30.0))
        self.fps_probe_frames = max(2, int(rospy.get_param("~fps_probe_frames", 30)))
        self.audio_offset_sec = float(rospy.get_param("~audio_offset_sec", 0.0))
        self.audio_gain_db = float(rospy.get_param("~audio_gain_db", 0.0))
        self.max_duration_sec = float(rospy.get_param("~max_duration_sec", 0.0))

        root = Path(str(rospy.get_param("~out_dir", "/workspace/asd_runtime_slim/runs/av_capture")))
        run_name = str(rospy.get_param("~run_name", time.strftime("%Y%m%d_%H%M%S")))
        self.run_dir = root / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        out_mp4 = str(rospy.get_param("~out_mp4", ""))
        self.output_mp4 = Path(out_mp4) if out_mp4 else self.run_dir / "record.mp4"
        if not self.output_mp4.is_absolute():
            self.output_mp4 = self.run_dir / self.output_mp4
        self.output_mp4.parent.mkdir(parents=True, exist_ok=True)

        self.silent_mp4 = self.run_dir / "video_silent.mp4"
        self.audio_wav = self.run_dir / "audio.wav"
        self.manifest_path = self.run_dir / "manifest.json"

        self.bridge = CvBridge() if CvBridge is not None else None
        self.video_queue: "queue.Queue[Any]" = queue.Queue(maxsize=int(rospy.get_param("~video_queue_size", 60)))
        self.audio_queue: "queue.Queue[Any]" = queue.Queue(maxsize=int(rospy.get_param("~audio_queue_size", 1000)))
        self._sentinel = object()
        self._stop_event = threading.Event()
        self._finalized = False
        self._state_lock = threading.Lock()

        self.video_writer: Optional[cv2.VideoWriter] = None
        self.audio_writer: Optional[wave.Wave_write] = None

        self.video_width: Optional[int] = None
        self.video_height: Optional[int] = None
        self.video_fps_effective: Optional[float] = None
        self.video_frames = 0
        self.video_dropped_queue = 0
        self.video_convert_errors = 0
        self.first_video_ts: Optional[float] = None
        self.last_video_ts: Optional[float] = None

        self.audio_chunks = 0
        self.audio_samples = 0
        self.audio_sample_rate: Optional[int] = None
        self.audio_meta_first: Dict[str, Any] = {}
        self.audio_dropped_queue = 0
        self.audio_parse_errors = 0
        self.first_audio_ts: Optional[float] = None
        self.last_audio_ts: Optional[float] = None

        AudioMsg, resolved_type = _resolve_audio_msg_class(self.audio_topic, self.audio_msg_type)
        self.audio_msg_type = resolved_type

        self.video_thread = threading.Thread(target=self._video_loop, name="ros_av_video_writer", daemon=True)
        self.audio_thread = threading.Thread(target=self._audio_loop, name="ros_av_audio_writer", daemon=True)
        self.video_thread.start()
        self.audio_thread.start()

        self.video_sub = rospy.Subscriber(
            self.video_topic,
            Image,
            self._image_cb,
            queue_size=int(rospy.get_param("~video_sub_queue_size", 4)),
            buff_size=int(rospy.get_param("~video_sub_buff_size", 2**24)),
        )
        self.audio_sub = rospy.Subscriber(
            self.audio_topic,
            AudioMsg,
            self._audio_cb,
            queue_size=int(rospy.get_param("~audio_sub_queue_size", 200)),
            buff_size=int(rospy.get_param("~audio_sub_buff_size", 2**23)),
        )

        rospy.Timer(rospy.Duration(5.0), self._watchdog_tick)
        if self.max_duration_sec > 0:
            rospy.Timer(rospy.Duration(self.max_duration_sec), self._duration_reached, oneshot=True)

        signal.signal(signal.SIGINT, self._signal_shutdown)
        signal.signal(signal.SIGTERM, self._signal_shutdown)
        rospy.on_shutdown(self.cleanup)

        rospy.loginfo(
            "[ros_av_recorder] recording video=%s audio=%s(%s) out=%s run_dir=%s fps=%.3f gain=%.1fdB",
            self.video_topic,
            self.audio_topic,
            self.audio_msg_type,
            self.output_mp4,
            self.run_dir,
            self.video_fps,
            self.audio_gain_db,
        )

    def _signal_shutdown(self, *_: Any) -> None:
        rospy.signal_shutdown("signal received")

    def _duration_reached(self, _event: Any) -> None:
        rospy.signal_shutdown(f"max_duration_sec reached: {self.max_duration_sec}")

    def _image_cb(self, msg: Image) -> None:
        if self._stop_event.is_set():
            return
        item = (_stamp_to_sec(msg), msg)
        try:
            self.video_queue.put_nowait(item)
        except queue.Full:
            with self._state_lock:
                self.video_dropped_queue += 1
            rospy.logwarn_throttle(5.0, "[ros_av_recorder] video queue full; dropping frames")

    def _audio_cb(self, msg: Any) -> None:
        if self._stop_event.is_set():
            return
        timestamp = _stamp_to_sec(msg)
        try:
            pcm, sample_rate, meta = _extract_mono_pcm16(
                msg,
                fallback_channels=self.fallback_audio_channels,
                fallback_sample_rate=self.fallback_audio_rate,
                audio_channel=self.audio_channel,
            )
        except Exception as exc:
            with self._state_lock:
                self.audio_parse_errors += 1
            rospy.logwarn_throttle(5.0, "[ros_av_recorder] audio parse failed: %s", exc)
            return

        try:
            self.audio_queue.put_nowait((timestamp, pcm, sample_rate, meta))
        except queue.Full:
            with self._state_lock:
                self.audio_dropped_queue += 1
            rospy.logwarn_throttle(5.0, "[ros_av_recorder] audio queue full; dropping chunks")

    def _open_video_writer(self, frame: np.ndarray, fps: float) -> None:
        height, width = frame.shape[:2]
        fps = max(1.0, min(120.0, float(fps)))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(self.silent_mp4), fourcc, fps, (int(width), int(height)))
        if not writer.isOpened():
            raise RuntimeError(f"failed to open video writer: {self.silent_mp4}")
        self.video_writer = writer
        self.video_width = int(width)
        self.video_height = int(height)
        self.video_fps_effective = fps
        rospy.loginfo(
            "[ros_av_recorder] video writer ready: %sx%s @ %.6ffps",
            self.video_width,
            self.video_height,
            self.video_fps_effective,
        )

    def _write_video_frame(self, timestamp: float, frame: np.ndarray) -> None:
        if self.video_writer is None:
            raise RuntimeError("video writer is not open")
        if self.video_width and self.video_height and frame.shape[:2] != (self.video_height, self.video_width):
            frame = cv2.resize(frame, (self.video_width, self.video_height), interpolation=cv2.INTER_AREA)
        self.video_writer.write(frame)
        with self._state_lock:
            if self.first_video_ts is None:
                self.first_video_ts = timestamp
            self.last_video_ts = timestamp
            self.video_frames += 1

    def _video_loop(self) -> None:
        pending: List[Tuple[float, np.ndarray]] = []
        while True:
            item = self.video_queue.get()
            try:
                if item is self._sentinel:
                    break
                timestamp, msg = item
                try:
                    frame = _image_msg_to_bgr(msg, self.bridge)
                except Exception as exc:
                    with self._state_lock:
                        self.video_convert_errors += 1
                    rospy.logwarn_throttle(5.0, "[ros_av_recorder] image convert failed: %s", exc)
                    continue

                if self.video_writer is None:
                    pending.append((timestamp, frame))
                    if self.video_fps > 0:
                        self._open_video_writer(frame, self.video_fps)
                    elif len(pending) >= self.fps_probe_frames:
                        span = pending[-1][0] - pending[0][0]
                        fps = (len(pending) - 1) / span if span > 1e-6 else self.default_video_fps
                        self._open_video_writer(pending[0][1], fps)
                    else:
                        continue

                    for pending_ts, pending_frame in pending:
                        self._write_video_frame(pending_ts, pending_frame)
                    pending.clear()
                    continue

                self._write_video_frame(timestamp, frame)
            finally:
                self.video_queue.task_done()

        if self.video_writer is None and pending:
            fps = self.default_video_fps
            if len(pending) >= 2:
                span = pending[-1][0] - pending[0][0]
                if span > 1e-6:
                    fps = (len(pending) - 1) / span
            self._open_video_writer(pending[0][1], fps)
            for pending_ts, pending_frame in pending:
                self._write_video_frame(pending_ts, pending_frame)
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

    def _audio_loop(self) -> None:
        while True:
            item = self.audio_queue.get()
            try:
                if item is self._sentinel:
                    break
                timestamp, pcm, sample_rate, meta = item
                if self.audio_writer is None:
                    writer = wave.open(str(self.audio_wav), "wb")
                    writer.setnchannels(1)
                    writer.setsampwidth(2)
                    writer.setframerate(int(sample_rate))
                    self.audio_writer = writer
                    self.audio_sample_rate = int(sample_rate)
                    self.audio_meta_first = dict(meta)
                    rospy.loginfo(
                        "[ros_av_recorder] audio writer ready: %dHz mono from payload=%s channels=%s selected=%s",
                        self.audio_sample_rate,
                        meta.get("field_name"),
                        meta.get("channels"),
                        meta.get("selected_channel"),
                    )

                if int(sample_rate) != int(self.audio_sample_rate or sample_rate):
                    rospy.logwarn_throttle(
                        5.0,
                        "[ros_av_recorder] audio sample_rate changed: got=%s expected=%s; dropping chunk",
                        sample_rate,
                        self.audio_sample_rate,
                    )
                    continue

                self.audio_writer.writeframes(pcm.tobytes())
                with self._state_lock:
                    if self.first_audio_ts is None:
                        self.first_audio_ts = timestamp
                    self.last_audio_ts = timestamp
                    self.audio_chunks += 1
                    self.audio_samples += int(pcm.size)
            finally:
                self.audio_queue.task_done()

        if self.audio_writer is not None:
            self.audio_writer.close()
            self.audio_writer = None

    def _watchdog_tick(self, _event: Any) -> None:
        with self._state_lock:
            video_frames = self.video_frames
            audio_chunks = self.audio_chunks
            video_dropped = self.video_dropped_queue
            audio_dropped = self.audio_dropped_queue
        rospy.loginfo(
            "[ros_av_recorder] progress video_frames=%d audio_chunks=%d dropped(video=%d audio=%d)",
            video_frames,
            audio_chunks,
            video_dropped,
            audio_dropped,
        )

    def _stop_queues(self) -> None:
        self._stop_event.set()
        for q in (self.video_queue, self.audio_queue):
            while True:
                try:
                    q.put_nowait(self._sentinel)
                    break
                except queue.Full:
                    try:
                        q.get_nowait()
                        q.task_done()
                    except queue.Empty:
                        pass

    def _mux(self) -> Dict[str, Any]:
        if self.video_frames <= 0 or not self.silent_mp4.exists():
            return {"status": "skipped", "reason": "no_video"}
        if self.audio_chunks <= 0 or not self.audio_wav.exists():
            shutil.copyfile(str(self.silent_mp4), str(self.output_mp4))
            return {"status": "video_only", "reason": "no_audio"}

        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            return {"status": "failed", "reason": "ffmpeg_not_found"}

        first_video = float(self.first_video_ts or 0.0)
        first_audio = float(self.first_audio_ts or first_video)
        audio_delta = (first_video - first_audio) - self.audio_offset_sec
        audio_trim_sec = max(audio_delta, 0.0)
        audio_delay_sec = max(-audio_delta, 0.0)

        command = [ffmpeg, "-y", "-loglevel", "error", "-i", str(self.silent_mp4)]
        if audio_trim_sec > 1e-6:
            command.extend(["-ss", f"{audio_trim_sec:.6f}"])
        command.extend(["-i", str(self.audio_wav), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy"])

        audio_filters: List[str] = []
        if audio_delay_sec > 1e-6:
            delay_ms = int(round(audio_delay_sec * 1000.0))
            audio_filters.append(f"adelay={delay_ms}")
        if abs(self.audio_gain_db) > 1e-6:
            audio_filters.append(f"volume={self.audio_gain_db:.3f}dB")
        if audio_filters:
            command.extend(["-af", ",".join(audio_filters)])
        command.extend(["-c:a", "aac", "-shortest", str(self.output_mp4)])

        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            reason = completed.stderr.strip() or f"ffmpeg_exit_{completed.returncode}"
            return {"status": "failed", "reason": reason}
        return {
            "status": "ok",
            "audio_trim_sec": audio_trim_sec,
            "audio_delay_sec": audio_delay_sec,
            "audio_gain_db": self.audio_gain_db,
        }

    def _write_manifest(self, mux_result: Dict[str, Any]) -> None:
        with self._state_lock:
            fps = float(self.video_fps_effective or self.video_fps or self.default_video_fps)
            audio_rate = int(self.audio_sample_rate or self.fallback_audio_rate)
            manifest = {
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "topics": {
                    "video": self.video_topic,
                    "audio": self.audio_topic,
                    "audio_msg_type": self.audio_msg_type,
                },
                "paths": {
                    "run_dir": str(self.run_dir),
                    "output_mp4": str(self.output_mp4),
                    "silent_mp4": str(self.silent_mp4),
                    "audio_wav": str(self.audio_wav),
                    "manifest": str(self.manifest_path),
                },
                "video": {
                    "frames": self.video_frames,
                    "fps": fps,
                    "duration_sec": float(self.video_frames) / fps if fps > 0 else 0.0,
                    "width": self.video_width,
                    "height": self.video_height,
                    "first_timestamp": self.first_video_ts,
                    "last_timestamp": self.last_video_ts,
                    "dropped_queue": self.video_dropped_queue,
                    "convert_errors": self.video_convert_errors,
                },
                "audio": {
                    "chunks": self.audio_chunks,
                    "samples": self.audio_samples,
                    "sample_rate": audio_rate,
                    "duration_sec": float(self.audio_samples) / audio_rate if audio_rate > 0 else 0.0,
                    "first_timestamp": self.first_audio_ts,
                    "last_timestamp": self.last_audio_ts,
                    "dropped_queue": self.audio_dropped_queue,
                    "parse_errors": self.audio_parse_errors,
                    "first_packet_meta": self.audio_meta_first,
                },
                "mux": mux_result,
            }
        with self.manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)

    def cleanup(self) -> None:
        if self._finalized:
            return
        self._finalized = True
        rospy.loginfo("[ros_av_recorder] finalizing...")
        self._stop_queues()
        self.video_thread.join(timeout=20.0)
        self.audio_thread.join(timeout=20.0)
        mux_result = self._mux()
        self._write_manifest(mux_result)
        rospy.loginfo(
            "[ros_av_recorder] done output=%s mux=%s manifest=%s",
            self.output_mp4,
            mux_result.get("status"),
            self.manifest_path,
        )


def main() -> int:
    RosAVRecorder()
    rospy.spin()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
