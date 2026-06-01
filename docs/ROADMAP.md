# Roadmap

## v0.1

- Publish a clean source release without model binaries.
- Document required model artifacts and runtime assumptions.
- Add issue templates and a basic compile check in CI.

## v0.2

- Add a small synthetic test suite for audio parsing, track buffers, IoU, and
  active-speaker selection.
- Provide an offline demo command that can run without ROS.
- Add latency and memory profiling notes for CPU, CUDA, and Jetson deployments.

## v0.3

- Improve model download/setup documentation.
- Add a stable CLI around offline ASD processing.
- Package the SDK with clearer backend extension points.

## Open Questions

- Which ROS message types should be supported out of the box?
- Which face detector backend should be the default for non-ROS users?
- How should model weights be distributed while respecting upstream licenses?
