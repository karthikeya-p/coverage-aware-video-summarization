"""Extract SlowFast features for every fully downloaded manifest EPIC video into the feature cache
(shortest first, resumable, skips partial downloads). run.sh / batch_manifest.py reuse this cache via FEAT_CACHE.

  python dev/extract_all.py --video-root ../data/EPIC-KITCHENS --cache ../data/feat_cache [--limit 15]
"""
import argparse, csv, glob, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "code"))  # inference code
import config as C
from slowfast_extract import extract

ap = argparse.ArgumentParser()
ap.add_argument("--video-root", required=True); ap.add_argument("--cache", required=True)
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
os.makedirs(a.cache, exist_ok=True)
rows = sorted((r for r in csv.DictReader(open(os.path.join(C.WEIGHTS, "manifest.csv"), newline=""))
               if r["dataset"].startswith("epic")), key=lambda r: float(r["duration_sec"]))
todo = []
for r in rows:
    hits = glob.glob(os.path.join(a.video_root, "**", r["video_id"] + ".MP4"), recursive=True)
    if hits and not os.path.exists(hits[0] + ".aria2") and not os.path.exists(os.path.join(a.cache, r["video_id"] + ".npz")):
        todo.append((r["video_id"], float(r["duration_sec"]), hits[0]))
if a.limit:
    todo = todo[:a.limit]
total = sum(d for _, d, _ in todo); done = 0.0; t0 = time.time()
print(f"{len(todo)} videos to extract, {total / 3600:.2f} h of video; settings: spatial={C.FEAT_SPATIAL} "
      f"slow={C.FEAT_SLOW} bgr={C.FEAT_BGR} offset={C.FEAT_OFFSET} batch={C.FEAT_BATCH}", flush=True)
for vid, d, path in todo:
    try:
        res = extract(path)
        tmp = os.path.join(a.cache, vid + ".tmp.npz"); np.savez_compressed(tmp, **res)
        os.replace(tmp, os.path.join(a.cache, vid + ".npz"))
    except Exception as e:
        print(f"!! {vid} failed: {e}", flush=True); continue
    done += d; el = time.time() - t0
    print(f"{vid} {d:6.0f}s -> {res['feats'].shape[0]} clips | {done / el:.2f}x real-time, "
          f"ETA {(total - done) / max(done / el, 1e-6) / 3600:.1f} h", flush=True)
ids = sorted(os.path.splitext(f)[0] for f in os.listdir(a.cache) if f.endswith(".npz") and ".tmp" not in f)
open(os.path.join(a.cache, "_ids.txt"), "w").write("\n".join(ids) + "\n")
print(f"cache now holds {len(ids)} videos (list: {a.cache}/_ids.txt)")
