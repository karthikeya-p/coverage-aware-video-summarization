"""DEV ONLY — find the extraction settings that best reproduce the public ActionFormer features.
  python dev/parity_sweep.py <video.MP4> [<video2.MP4> ...]
Prints mean cosine similarity to the public feature of the same video for each variant, best first.
"""
import glob, itertools, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "code"))  # inference code
import config as C
from slowfast_extract import extract

def public(vid):
    h = glob.glob(os.path.join(C.WEIGHTS, "epic_kitchens_public", "**", vid + ".npz"), recursive=True)
    return np.load(h[0])["feats"].astype(np.float32)

def cos(a, b):
    n = min(len(a), len(b)); a, b = a[:n], b[:n]
    return float(((a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)).mean())

videos = sys.argv[1:]
variants = [dict(offset=o) for o in (0, 4, 8, 12, 16, -4, -8, -12, -16)]   # other settings: config defaults
res = []
for v in variants:
    t0 = time.time(); cs = []; lens = []
    for vp in videos:
        vid = os.path.splitext(os.path.basename(vp))[0]
        f = extract(vp, **v)["feats"]; p = public(vid)
        cs.append(cos(f, p)); lens.append((len(f), len(p)))
    res.append((float(np.mean(cs)), v, lens))
    print(f"cos {np.mean(cs):.4f}  {v}  len(ours,public)={lens}  [{time.time() - t0:.0f}s]", flush=True)
best = max(res, key=lambda r: r[0])
print("\nBEST:", f"{best[0]:.4f}", best[1])
