#!/usr/bin/env bash
# One-time environment setup (run inside an activated Python 3.10 environment that already has CUDA PyTorch).
#   bash setup.sh                       # python deps + NMS extension
#   bash weights/download_weights.sh    # then the checkpoints (~3.4 GB)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"
TP=code/third_party

echo "== python deps (the installed torch / torchvision are pinned so nothing replaces them)"
pip freeze | grep -i -E "^(torch|torchvision)==" > /tmp/t4_torch_pin.txt || true
pip install -r requirements.txt -c /tmp/t4_torch_pin.txt
pip install --no-deps mmengine "mmcv-lite==2.1.0"

echo "== 1-D NMS extension (ActionFormer)"
( cd $TP/actionformer_release/libs/utils && python setup.py install --user ) || \
  { echo "!! NMS build failed: install a C++ toolchain (build-essential) and re-run"; exit 1; }

echo "== sanity"
python - <<'PY'
import sys; sys.path.insert(0, "code")
import torch, open_clip, cv2, nms_1d_cpu
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
import config as C, ot_loader
ot_loader.setup(C.OPENTAD_DIR)
from opentad.models.backbones.vit_adapter import VisionTransformerAdapter
print("AdaTAD model code imports OK | weights present:", C._ADATAD_OK, "| detector:", C.DETECTOR)
PY
echo "setup done. Next: bash weights/download_weights.sh, then ./run.sh <video.mp4> <out.json>"
