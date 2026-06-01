# Realtime ASD 

Realtime ASD is an experimental active speaker detection runtime for robotics,
meeting, and multimodal video pipelines. It combines face detection, short
track buffers, audio/video alignment, TalkNet-style scoring, optional lip-motion
fusion, and ROS topic output for online active-speaker selection.

The current codebase is a source-first open release extracted from an internal
runtime. It is useful for developers who want to inspect, adapt, or reproduce a
real-time ASD pipeline, especially on ROS/Jetson-style deployments.

## Features

- ROS1 online node for synchronized camera and microphone topics.
- Track-level active speaker scoring with smoothing and top-speaker selection.
- Optional first-speaker lock and center-area priority logic for human-robot
  interaction scenarios.
- Optional RealSense depth-based main-speaker camera-coordinate output.
- Separate `asd_sdk` package boundary for algorithm-only inference.
- Offline tools for recording, rendering overlays, and dataset-style runs.

## Project Status

This repository is early-stage. The runtime code is available, but model weights
and deployment binaries are intentionally not committed. See
[`models/README.md`](models/README.md) for expected artifacts.

## Repository Layout

```text
asd/                 Core runtime helpers: buffers, tracking, audio parsing, scoring utilities.
config/              Runtime configuration defaults.
nodes/               ROS node entry points.
sdk/asd_sdk/         Algorithm-only SDK boundary.
tools/               Offline recording, rendering, export, and dataset utilities.
visualization/       OpenCV overlay rendering helpers.
scripts/             Container and ROS startup helpers.
docs/                Maintainer docs and roadmap.
models/              Placeholder directory for external model artifacts.
```

## Quick Start

Clone the repository:

```bash
git clone https://github.com/Dqiqi123/realtime-asd.git
cd realtime-asd
```

Install the common Python dependencies:

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -e sdk
```

For a ROS1 runtime, provide your ROS environment and model files, then run:

```bash
./scripts/run_node.sh \
  _video_topic:=/camera/color/image_raw \
  _audio_topic:=/microphone/audio_data_raw \
  _audio_msg_type:=audio/AudioData
```

The default internal deployment topics are kept in
[`config/asd_config.py`](config/asd_config.py). Override them with ROS
parameters or environment variables for your own setup.

## SDK Example

```python
import time
from asd_sdk import ASDSDK, InferenceRequest, TrackInput

backend = ASDSDK.build_default_talknet_backend()
sdk = ASDSDK(talknet_backend=backend, mode="top_scorer", enable_lip_motion=True)

request = InferenceRequest(
    timestamp=time.time(),
    tracks=[
        TrackInput(track_id=1, audio_clip=audio1, video_clip=clip1, lip_score=0.3),
        TrackInput(track_id=2, audio_clip=audio2, video_clip=clip2, lip_score=-0.2),
    ],
)

result = sdk.infer(request)
print(result.active_speaker_ids)
```

## Contributing

Real usage reports are welcome: installation failures, unsupported ROS message
types, model-loading issues, latency profiles, and documentation gaps are all
valuable. See [`CONTRIBUTING.md`](CONTRIBUTING.md) and
[`docs/ROADMAP.md`](docs/ROADMAP.md).

## License

MIT License.
