# ASD SDK

ASD SDK 提供“仅算法层”的主动说话人评分与选择能力。

## 和实时节点的关系

实时运行不要直接启动 SDK 入口。当前完整在线逻辑在：

```text
nodes/realtime_asd_node_parallel.py
```

启用方式是：

```bash
cd /workspace/asd_runtime_slim
./scripts/run_node.sh
```

SDK 只负责算法推理和分数输出，不负责：

- ROS topic 发布/订阅
- 5 秒主对话人锁定
- 3 秒中心核心区等待
- ID 跳变重绑定
- RealSense 深度图和相机坐标输出
- OpenCV 可视化窗口

因此如果只修改 `nodes/realtime_asd_node_parallel.py`、`config/asd_config.py` 或 README，不需要重新打包 SDK wheel；重启 ASD 节点即可。只有修改 `sdk/asd_sdk/` 目录内的 SDK 代码，并且运行环境是通过已安装 wheel 导入 SDK 时，才需要重新打包并安装。

## 功能边界

- 包含：TalkNet 打分、Lip 融合、主动说话人选择、状态平滑。
- 不包含：人脸检测（YOLO）、跟踪、ROS 收发、界面渲染。

## 输入 / 输出约定

输入为 `InferenceRequest`，其中包含按 track 对齐的数据片段：

- `track_id`（可选）：上游跟踪器分配的 ID。未提供时，SDK 会按输入顺序自动分配（0,1,2...）。
- `audio_clip`：单声道浮点数组（16kHz）。
- `video_clip`：人脸时序帧，形状为 `(T, H, W, C)` 或 `(T, H, W)`。
- `lip_score`：外部提供的 lip 分数，范围通常为 `[-1, 1]` 或 `-inf`。
	- 当 `enable_lip_motion=True`（默认）时，`lip_score` 为必填。
	- 当 `enable_lip_motion=False` 时，`lip_score` 可省略，SDK 不参与 Lip 融合。

输出为 `InferenceResult`：

- `active_speaker_ids`：最终选中的主动说话人 track_id 列表。
- `tracks`：每个 track 的 `TrackResult`，包含 raw/smooth/fused 分数和相关标记。

## 使用示例

```python
import time
from asd_sdk import ASDSDK, InferenceRequest, TrackInput

# 方案 A：在本仓库环境中，使用默认 TalkNet 运行后端
backend = ASDSDK.build_default_talknet_backend()
sdk = ASDSDK(talknet_backend=backend, mode="top_scorer", enable_lip_motion=True)

req = InferenceRequest(
	timestamp=time.time(),
	tracks=[
		TrackInput(track_id=1, audio_clip=audio1, video_clip=clip1, lip_score=0.32),
		TrackInput(track_id=2, audio_clip=audio2, video_clip=clip2, lip_score=-0.2),
	],
)

res = sdk.infer(req)
print(res.active_speaker_ids)
```

说明：

- YOLO 检测与跟踪位于 SDK 外部。
- 可视化位于 SDK 外部（参考 `visualization/asd_visualizer.py`）。

## 打包 wheel

```bash
cd sdk
python3 -m pip install --upgrade build
python3 -m build
```

生成物位于 `sdk/dist/` 目录。

如果希望 Docker 和其他环境都统一从本地包目录安装，建议在仓库根目录执行：

```bash
cd /home/naviai/asd_runtime_slim
./scripts/refresh_local_packages.sh
```

这样会把最新的 `asd_sdk-*.whl` 同步到 `packages/python/`。

## 在 Docker 中安装 SDK

推荐流程：

1. 先在宿主机打包 wheel（只需在版本更新后重新打包）
2. 建议运行 `./scripts/refresh_local_packages.sh`，把 wheel 同步到 `packages/python/`
3. 再构建镜像，Dockerfile 会自动安装 `packages/python/` 下的 `asd_sdk-*.whl`

```bash
cd /home/naviai/asd_runtime_slim/sdk
python3 -m build --wheel

cd /home/naviai/asd_runtime_slim
./scripts/refresh_local_packages.sh
docker compose build
```
