"""DEV ONLY — fast tuning loop for the EPIC branch using the publicly released ActionFormer features, so the
detector + pairing + selector can be measured before the 184 GB of video is downloaded.
These JSONs are NOT submission outputs: they lack the video-derived action-ness curve and do not come from
run.sh. Submission outputs must be produced from the MP4s.

  python dev/dev_epic_public.py --out ../data/dev_outputs [--limit 30] [--beta ... --pad ... --min-score ...]
  python dev/eval_local.py --outputs ../data/dev_outputs --manifest weights/manifest.csv \
         --epic-ann weights/epic-kitchens-100-annotations
"""
import argparse, csv, glob, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "code"))  # inference code
import config as C
from actionformer_infer import detect_actions, pair_actions, get_detectors
from coverage_selector import select_highlights, pack_windows

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--feat-dir", default=os.path.join(C.WEIGHTS, "epic_kitchens_public", "epic_kitchens", "features"))
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--ids", default=None, help="file with video_ids to run (one per line)")
ap.add_argument("--mode", choices=["coverage", "baseline"], default="coverage",
                help="baseline = importance-threshold summariser (sum of action scores), for the ablation table")
for name, default in (("beta", C.SEL_BETA), ("pad", C.SEL_PAD), ("min-score", C.SEL_MIN_SCORE),
                      ("repeat-w", C.SEL_REPEAT_WEIGHT), ("max-seg", C.SEL_MAX_SEG),
                      ("alpha", C.PAIR_ALPHA), ("pair-iou", C.PAIR_IOU)):
    ap.add_argument("--" + name, type=float, default=default)
a = ap.parse_args()

rows = [r for r in csv.DictReader(open(os.path.join(C.WEIGHTS, "manifest.csv"), newline=""))
        if r["dataset"].startswith("epic")]
if a.ids:
    keep = set(open(a.ids).read().split()); rows = [r for r in rows if r["video_id"] in keep]
if a.limit:
    rows = rows[:a.limit]
os.makedirs(a.out, exist_ok=True)
cache = os.path.join(a.out, "_raw"); os.makedirs(cache, exist_ok=True)
D = get_detectors(); t0 = time.time()
for i, r in enumerate(rows):
    vid, dur = r["video_id"], float(r["duration_sec"])
    rp = os.path.join(cache, vid + ".npz")
    if os.path.exists(rp):                        # raw detections are cached: re-tuning takes seconds
        z = np.load(rp); v = (z["vs"], z["vc"], z["vl"]); n = (z["ns"], z["nc"], z["nl"])
    else:
        f = np.load(os.path.join(a.feat_dir, vid + ".npz"))["feats"]
        v = D["verb"].detect(f, dur, vid); n = D["noun"].detect(f, dur, vid)
        np.savez_compressed(rp, vs=v[0], vc=v[1], vl=v[2], ns=n[0], nc=n[1], nl=n[2])
    actions = pair_actions(*v, *n, dur, alpha=a.alpha, iou_thr=a.pair_iou, cooc=D["cooc"],
                           verb_names=D["verb_names"], noun_names=D["noun_names"], vocab=D["vocab"])
    if a.mode == "coverage":
        hl = select_highlights(actions, dur, beta=a.beta, pad=a.pad, min_score=a.min_score,
                               repeat_w=a.repeat_w, max_seg=a.max_seg)
    else:
        grid = np.arange(0, dur, 0.5); imp = np.zeros(len(grid))
        for x in actions:
            s, e = x["segment"]; imp[(grid >= s) & (grid <= e)] += x["score"]
        hl = pack_windows(imp, grid, dur)
    out = {"team_id": C.TEAM_ID, "video_id": vid, "duration_sec": dur, "label": {"name": "cooking", "score": 0.9},
           "actions": [{k: x[k] for k in ("segment", "verb", "noun", "score")} for x in actions],
           "highlights": [{"segment": [float(s), float(e)]} for s, e in hl]}
    json.dump(out, open(os.path.join(a.out, vid + ".json"), "w"))
    if (i + 1) % 10 == 0 or i + 1 == len(rows):
        print(f"{i + 1}/{len(rows)}  {time.time() - t0:.0f}s", flush=True)
