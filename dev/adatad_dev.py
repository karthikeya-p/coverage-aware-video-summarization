"""DEV ONLY — run AdaTAD verb + noun detection on the N shortest manifest EPIC videos, cache raw detections in
the same format as dev_epic_public.py (so pairing / selection / scoring can be reused), and report speed.

  python dev/adatad_dev.py --out ../data/dev_adatad --limit 20
"""
import argparse, csv, glob, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "code"))  # inference code
import config as C
from video_io import iter_frames
import adatad_infer as A

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--video-root", default="../data/EPIC-KITCHENS")
ap.add_argument("--adatad-dir", default=os.path.expanduser("~/xr_hackathon/adatad"))
ap.add_argument("--limit", type=int, default=20)
ap.add_argument("--ids", default=None)
ap.add_argument("--batch", type=int, default=8)
a = ap.parse_args()
raw_dir = os.path.join(a.out, "_raw"); os.makedirs(raw_dir, exist_ok=True)
rows = sorted((r for r in csv.DictReader(open(os.path.join(C.WEIGHTS, "manifest.csv"), newline=""))
               if r["dataset"].startswith("epic")), key=lambda r: float(r["duration_sec"]))
if a.ids:
    keep = set(open(a.ids).read().split()); rows = [r for r in rows if r["video_id"] in keep]
elif a.limit:
    rows = rows[:a.limit]
OT = os.path.join(a.adatad_dir, "OpenTAD")
models = {k: A.load_model(k, OT, os.path.join(a.adatad_dir, "ckpt", f"adatad_epic_{k}.pth"),
                          os.path.join(a.adatad_dir, "anno", "annotations", f"category_idx_{k}.txt")) for k in ("verb", "noun")}
print("models loaded; GPU MB:", round(torch.cuda.memory_allocated() / 1e6) if torch.cuda.is_available() else 0, flush=True)
tot_v = tot_t = 0.0; done = []
for r in rows:
    vid, dur = r["video_id"], float(r["duration_sec"])
    outp = os.path.join(raw_dir, vid + ".npz")
    if os.path.exists(outp):
        done.append(vid); continue
    hits = glob.glob(os.path.join(a.video_root, "**", vid + ".MP4"), recursive=True)
    if not hits:
        continue
    t0 = time.time()
    fr = []
    for _, f in iter_frames(hits[0], fps=15, short_side=160):          # every 2nd frame of the 30-fps stream
        x = (f.shape[1] - 160) // 2; y = (f.shape[0] - 160) // 2
        fr.append(f[y:y + 160, x:x + 160])
    fr = np.ascontiguousarray(np.stack(fr)); t1 = time.time()
    res = {}
    for k in ("verb", "noun"):
        res[k] = A.detect_video(models[k], fr, dur, batch=a.batch)
    t2 = time.time()
    np.savez_compressed(outp, vs=res["verb"][0], vc=res["verb"][1], vl=res["verb"][2],
                        ns=res["noun"][0], nc=res["noun"][1], nl=res["noun"][2])
    tot_v += dur; tot_t += t2 - t0; done.append(vid)
    peak = round(torch.cuda.max_memory_allocated() / 1e6) if torch.cuda.is_available() else 0
    print(f"{vid} {dur:6.1f}s frames={len(fr)} decode {t1 - t0:.1f}s model {t2 - t1:.1f}s | {len(res['verb'][1])} verb / "
          f"{len(res['noun'][1])} noun dets | overall {tot_v / tot_t:.2f}x real-time | peak GPU {peak} MB", flush=True)
open(os.path.join(a.out, "_ids.txt"), "w").write("\n".join(done) + "\n")
print(f"done: {len(done)} videos; ids -> {a.out}/_ids.txt")
