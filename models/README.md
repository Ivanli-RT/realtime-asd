# Model Artifacts

Large runtime artifacts are not committed to this repository.

Expected files for the current runtime include:

- `pretrain_TalkSet.model`
- `yolov11n-face.pt` or another YOLO face detector
- Optional TensorRT engines such as `talknet_tv8_ta32_b1_8_fp16.engine`
- Optional lip landmark weights such as `resnet_50-epoch-724.pth`

Place artifacts in this directory or point the runtime to custom paths with the
environment variables documented in `config/asd_config.py`.

Before publishing model files, verify their upstream licenses and redistribution
rights.
