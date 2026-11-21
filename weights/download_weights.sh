#!/usr/bin/env bash
# Download the large checkpoints into weights/ (they are too big for git).
#   bash weights/download_weights.sh
# Sources, in order: the original public releases, then an optional mirror (a URL prefix in weights/MIRROR_URL,
# e.g. this repository's GitHub release). Every file is checked against weights/SHA256SUMS.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
MIRROR=""; [ -s MIRROR_URL ] && MIRROR=$(tr -d ' \r\n' < MIRROR_URL)
CLIP_URL="https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K/resolve/main/open_clip_model.safetensors"

ok() { [ -s "$1" ] && { [ ! -s SHA256SUMS ] || grep " $1\$" SHA256SUMS | sha256sum -c --status - ; }; }
from_mirror() { [ -n "$MIRROR" ] && wget -q --show-progress -O "$1.part" "$MIRROR/$1" && mv "$1.part" "$1"; }
from_drive() { python -m pip install -q gdown && python -m gdown "$2" -O "$1.part" && mv "$1.part" "$1"; }
from_url() { wget -q --show-progress -O "$1.part" "$2" && mv "$1.part" "$1"; }

get() {   # get <file> <kind: drive|url> <id-or-url>
  local f="$1"
  if ok "$f"; then echo "[ok] $f"; return 0; fi
  rm -f "$f" "$f.part"
  echo "[..] $f"
  if [ "$2" = drive ]; then from_drive "$f" "$3"; else from_url "$f" "$3"; fi
  if ! ok "$f"; then rm -f "$f" "$f.part"; echo "     original source failed, trying mirror"; from_mirror "$f"; fi
  if ok "$f"; then echo "[ok] $f"; else echo "[FAILED] $f"; return 1; fi
}
rc=0
get adatad_epic_verb.pth drive 16Hq3sHu0S97Ge2AewHT6DOaHSo0TqIlx || rc=1        # OpenTAD model zoo (AdaTAD, EPIC verb)
get adatad_epic_noun.pth drive 17k3f6wirqniLTjKOsIXbfqJPA_iLb88E || rc=1        # OpenTAD model zoo (AdaTAD, EPIC noun)
get open_clip_vit_b32_laion2b_s34b_b79k.safetensors url "$CLIP_URL" || rc=1    # OpenCLIP ViT-B/32 laion2b_s34b_b79k
[ $rc -eq 0 ] && echo "all weights present and verified" || echo "some downloads failed; re-run, or set weights/MIRROR_URL"
exit $rc
