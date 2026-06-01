import math

import numpy as np
from scipy.signal import resample_poly


def _pick_audio_payload(msg):
    for field_name in ("audio_data", "filtered_data", "data"):
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


def _pick_raw_audio_payload(msg):
    # Prefer the upstream raw PCM payload. Only fall back to filtered_data when
    # older message variants do not expose raw samples.
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


def _coerce_pcm_samples(raw_data, field_name):
    if field_name == "audio_data":
        # The audio middleware exposes uint16[] while payload bits are PCM s16le.
        return np.asarray(raw_data, dtype=np.uint16).view(np.int16)
    return np.asarray(raw_data, dtype=np.int16)


def _resample_mono_audio(mono, src_sample_rate, dst_sample_rate):
    src_sample_rate = int(src_sample_rate)
    dst_sample_rate = int(dst_sample_rate)
    if src_sample_rate <= 0 or dst_sample_rate <= 0:
        raise ValueError(
            f"invalid sample rate conversion: src={src_sample_rate} dst={dst_sample_rate}"
        )
    if src_sample_rate == dst_sample_rate:
        return mono.astype(np.float32, copy=False)

    gcd = math.gcd(src_sample_rate, dst_sample_rate)
    up = dst_sample_rate // gcd
    down = src_sample_rate // gcd
    resampled = resample_poly(mono.astype(np.float32, copy=False), up, down)
    return resampled.astype(np.float32, copy=False)


def extract_mono_audio(msg, fallback_channels, use_channel, target_sample_rate=None):
    field_name, raw_data = _pick_audio_payload(msg)
    if raw_data is None:
        raise ValueError("empty audio payload")

    arr = _coerce_pcm_samples(raw_data, field_name)
    total_samples = int(arr.size)
    if total_samples <= 0:
        raise ValueError("empty audio payload")

    channels = getattr(msg, "channel", None)
    if channels in (None, 0):
        channels = getattr(msg, "channels", None)
    if channels in (None, 0):
        channels = fallback_channels

    channels = int(channels)
    if channels <= 0:
        raise ValueError(f"invalid audio channel count: {channels}")
    if total_samples % channels != 0:
        raise ValueError(
            f"audio payload length {total_samples} is not divisible by channel count {channels}"
        )

    if channels == 1:
        selected_channel = 0
        mono = arr.astype(np.float32, copy=False)
    else:
        samples_per_packet = total_samples // channels
        interleaved = arr.reshape(samples_per_packet, channels)
        selected_channel = int(np.clip(use_channel, 0, channels - 1))
        mono = interleaved[:, selected_channel].astype(np.float32, copy=False)

    source_sample_rate = getattr(msg, "sample_rate", None)
    source_sample_rate = int(source_sample_rate) if source_sample_rate not in (None, 0) else None

    effective_sample_rate = source_sample_rate
    was_resampled = False
    if target_sample_rate is not None:
        target_sample_rate = int(target_sample_rate)
        if source_sample_rate is None:
            effective_sample_rate = target_sample_rate
        elif source_sample_rate != target_sample_rate:
            mono = _resample_mono_audio(mono, source_sample_rate, target_sample_rate)
            effective_sample_rate = target_sample_rate
            was_resampled = True

    return mono, {
        "field_name": field_name,
        "channels": channels,
        "selected_channel": selected_channel,
        "sample_rate": effective_sample_rate,
        "source_sample_rate": source_sample_rate,
        "target_sample_rate": int(target_sample_rate) if target_sample_rate is not None else None,
        "was_resampled": was_resampled,
        "samples_per_packet": int(mono.shape[0]),
    }


def extract_raw_audio_packet(msg, fallback_channels, fallback_sample_rate=None):
    field_name, raw_data = _pick_raw_audio_payload(msg)
    if raw_data is None:
        raise ValueError("empty audio payload")

    arr = _coerce_pcm_samples(raw_data, field_name)
    total_samples = int(arr.size)
    if total_samples <= 0:
        raise ValueError("empty audio payload")

    channels = getattr(msg, "channel", None)
    if channels in (None, 0):
        channels = getattr(msg, "channels", None)
    if channels in (None, 0):
        channels = fallback_channels

    channels = int(channels)
    if channels <= 0:
        raise ValueError(f"invalid audio channel count: {channels}")
    if total_samples % channels != 0:
        raise ValueError(
            f"audio payload length {total_samples} is not divisible by channel count {channels}"
        )

    sample_rate = getattr(msg, "sample_rate", None)
    sample_rate = int(sample_rate) if sample_rate not in (None, 0) else None
    if sample_rate is None and fallback_sample_rate is not None:
        sample_rate = int(fallback_sample_rate)

    bits_per_sample = getattr(msg, "bits_per_sample", None)
    bits_per_sample = int(bits_per_sample) if bits_per_sample not in (None, 0) else 16

    return np.ascontiguousarray(arr.astype(np.int16, copy=False)), {
        "field_name": field_name,
        "channels": channels,
        "sample_rate": sample_rate,
        "bits_per_sample": bits_per_sample,
        "samples_per_channel": int(total_samples // channels),
        "is_raw_source": field_name in {"audio_data", "data"},
    }


def format_audio_meta(meta):
    sample_rate = meta.get("sample_rate")
    source_sample_rate = meta.get("source_sample_rate")
    target_sample_rate = meta.get("target_sample_rate")
    if source_sample_rate and target_sample_rate and source_sample_rate != target_sample_rate:
        sample_rate_desc = f"{source_sample_rate}Hz->{target_sample_rate}Hz"
    elif sample_rate:
        sample_rate_desc = f"{sample_rate}Hz"
    else:
        sample_rate_desc = "sample_rate=unknown"
    return (
        f"payload={meta.get('field_name')} channels={meta.get('channels')} "
        f"selected_channel={meta.get('selected_channel')} "
        f"samples_per_packet={meta.get('samples_per_packet')} "
        f"{sample_rate_desc}"
    )
