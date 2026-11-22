"""Parity check: our SlowFast features vs the public ActionFormer features for the same EPIC video.

python dev/check_features.py <video_id> <our.npz> [--public-dir weights/epic_kitchens_public]
Prints lengths and per-step cosine similarity. Target: length diff <= 2, mean cosine > 0.9.
If cosine is low, try FEAT_SAMPLING_RATE=2 (64-frame span, every 2nd frame) and/or check the crop/normalisation.
"""
import argparse, glob, os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "code"))  # inference code
import config as C

ap = argparse.ArgumentParser()
ap.add_argument("video_id"); ap.add_argument("ours")
ap.add_argument("--public-dir", default=os.path.join(C.WEIGHTS, "epic_kitchens_public"))
a = ap.parse_args()

hits = glob.glob(os.path.join(a.public_dir, "**", f"{a.video_id}.npz"), recursive=True)
if not hits:
    sys.exit(f"public features for {a.video_id} not found under {a.public_dir}")
pub = np.load(hits[0])["feats"].astype(np.float32)
ours = np.load(a.ours)["feats"].astype(np.float32)
if pub.shape[0] < pub.shape[1] and pub.shape[0] == 2304:
    pub = pub.T
print("public", pub.shape, "ours", ours.shape)
n = min(len(pub), len(ours))
best = None
for shift in range(-3, 4):   # allow a small temporal offset
    p = pub[max(0, shift):n + min(0, shift)]; o = ours[max(0, -shift):n + min(0, -shift)]
    m = min(len(p), len(o)); p, o = p[:m], o[:m]
    cos = (p * o).sum(1) / (np.linalg.norm(p, axis=1) * np.linalg.norm(o, axis=1) + 1e-9)
    if best is None or cos.mean() > best[1]:
        best = (shift, cos.mean(), np.percentile(cos, 10))
print(f"best shift {best[0]:+d} clips: mean cosine {best[1]:.3f}, p10 {best[2]:.3f}")
print("PASS" if best[1] > 0.9 and abs(len(pub) - len(ours)) <= 2 else "CHECK: features differ — see docstring")
