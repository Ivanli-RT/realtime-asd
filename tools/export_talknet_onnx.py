#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
TALKNET_DIR = ROOT / "talknet"
if str(TALKNET_DIR) not in sys.path:
    sys.path.insert(0, str(TALKNET_DIR))

from talkNet import talkNet  # noqa: E402


class TalkNetExportWrapper(nn.Module):
    def __init__(self, wrapper: talkNet):
        super().__init__()
        self.model = wrapper.model.eval()
        self.fc = wrapper.lossAV.FC.eval()

    def forward(self, audio: torch.Tensor, video: torch.Tensor) -> torch.Tensor:
        a_emb = self.model.forward_audio_frontend(audio)
        v_emb = self.model.forward_visual_frontend(video)
        a2, v2 = self.model.forward_cross_attention(a_emb, v_emb)
        outs_av = self.model.forward_audio_visual_backend(a2, v2)
        logits = self.fc(outs_av)
        b = audio.shape[0]
        t = video.shape[1]
        return logits.view(b, t, 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export TalkNet to ONNX for TensorRT rebuild")
    parser.add_argument("--weights", required=True, help="Path to TalkNet .model weights")
    parser.add_argument("--output", required=True, help="Output ONNX path")
    parser.add_argument("--batch", type=int, default=1, help="Dummy batch size for export")
    parser.add_argument("--ta", type=int, default=32, help="Audio time steps")
    parser.add_argument("--tv", type=int, default=8, help="Video time steps")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    weights = Path(args.weights).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    if not weights.is_file():
        raise FileNotFoundError(f"TalkNet weights not found: {weights}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wrapper = talkNet(lr=0.0001, lrDecay=0.95)
    wrapper.loadParameters(str(weights))
    wrapper = wrapper.to(device).eval()

    export_model = TalkNetExportWrapper(wrapper).to(device).eval()

    audio = torch.zeros((args.batch, args.ta, 13), dtype=torch.float32, device=device)
    video = torch.zeros((args.batch, args.tv, 112, 112), dtype=torch.float32, device=device)

    with torch.no_grad():
        _ = export_model(audio, video)

    torch.onnx.export(
        export_model,
        (audio, video),
        str(output),
        input_names=["audio", "video"],
        output_names=["logits"],
        dynamic_axes={
            "audio": {0: "batch"},
            "video": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=int(args.opset),
        do_constant_folding=True,
    )

    print(f"[export_talknet_onnx] Exported ONNX: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
