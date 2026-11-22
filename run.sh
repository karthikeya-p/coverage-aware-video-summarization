#!/usr/bin/env bash
# Track 04 inference entry point:  ./run.sh <video.mp4> <out.json>
# Env (optional): TEAM_ID, DEVICE=cuda|cpu, ADATAD_BATCH, FEAT_CACHE=<dir> (caches detections), DETECTOR=adatad|actionformer
set -euo pipefail
if [ $# -lt 2 ]; then
  echo "usage: $0 <video.mp4> <out.json>" >&2; exit 1
fi
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TP="$HERE/code/third_party"; [ -d "$TP" ] || TP="$HERE/third_party"
export PYTHONPATH="$HERE/code:$TP/actionformer_release:$TP/epic-kitchens-slowfast:${PYTHONPATH:-}"
exec python "$HERE/code/run_video.py" "$1" "$2" ${FEAT_CACHE:+--cache-dir "$FEAT_CACHE"}
