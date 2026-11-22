"""DEV ONLY — diagnose and tune the EPIC branch from cached raw detections (run dev_epic_public.py first).

  python dev/sweep_selector.py --raw ../data/dev_cov/_raw            # diagnostics + selector grid
Prints: raw verb/noun mAP (wrapper sanity vs published ~23.4 / ~21.9), oracle coverage (selector fed ground truth =
ceiling under the 15 % / 4 s rules), then a grid over selector settings.
Settings are chosen on a fixed 30-video tune split (tune_ids.txt) and reported on the other 108.
"""
import argparse, csv, itertools, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "code"))  # inference code
import config as C
from actionformer_infer import pair_actions, _load_json
from coverage_selector import select_highlights
from eval_local import load_epic_gt, action_map, coverage

ap = argparse.ArgumentParser()
ap.add_argument("--raw", required=True)
ap.add_argument("--epic-ann", default=os.path.join(C.WEIGHTS, "epic-kitchens-100-annotations"))
ap.add_argument("--skip-map", action="store_true")
a = ap.parse_args()

man = {r["video_id"]: float(r["duration_sec"]) for r in csv.DictReader(open(os.path.join(C.WEIGHTS, "manifest.csv")))
       if r["dataset"].startswith("epic")}
gt = load_epic_gt(a.epic_ann)
vids = sorted(v for v in man if os.path.exists(os.path.join(a.raw, v + ".npz")) and v in gt)
tune = vids[::len(vids) // 30][:30]; held = [v for v in vids if v not in set(tune)]
open(os.path.join(C.ROOT, "tune_ids.txt"), "w").write("\n".join(tune) + "\n")
print(f"{len(vids)} videos: {len(tune)} tune / {len(held)} held-out (tune ids -> tune_ids.txt)")
vn = {int(k): v for k, v in _load_json(C.VERB_LABELS).items()}
nn = {int(k): v for k, v in _load_json(C.NOUN_LABELS).items()}
cooc = _load_json(C.COOC_PRIOR); vocab = _load_json(C.SUPPORTED_CLASSES)
raw = {}
for v in vids:
    z = np.load(os.path.join(a.raw, v + ".npz"))
    raw[v] = (z["vs"], z["vc"], z["vl"], z["ns"], z["nc"], z["nl"])

if not a.skip_map:
    t0 = time.time()
    for kind, names, o in (("verb", vn, 0), ("noun", nn, 3)):
        preds = {v: [dict(segment=list(map(float, s)), score=float(c), **{kind: names.get(int(l), "other")})
                     for s, c, l in zip(raw[v][o], raw[v][o + 1], raw[v][o + 2])] for v in vids}
        print(f"raw {kind} mAP (tIoU .1-.5): {100 * action_map(preds, {v: gt[v] for v in vids}, key=lambda x: x[kind]):.2f}"
              f"   [{time.time() - t0:.0f}s]", flush=True)

def cov(hl_by_vid, ids):
    c = [coverage([dict(segment=h) for h in hl_by_vid[v]], gt[v], man[v]) for v in ids]
    return float(np.mean([x["half"] * x["penalty"] for x in c]))

oracle = {v: select_highlights([dict(segment=g["segment"], verb=g["verb"], noun=g["noun"], score=1.0) for g in gt[v]],
                               man[v], beta=0.0) for v in vids}
print(f"ORACLE coverage (selector on ground truth): tune {cov(oracle, tune):.3f}  held-out {cov(oracle, held):.3f}")
print(f"GT stats: mean distinct pairs/video {np.mean([len({(g['verb'], g['noun']) for g in gt[v]}) for v in vids]):.1f}, "
      f"median GT action length {np.median([g['segment'][1] - g['segment'][0] for v in vids for g in gt[v]]):.2f}s")

print("\nalpha iou | min_score pad max_seg repeat_w | cov_tune")
best = None
for alpha, iou in ((0.5, 0.3), (0.5, 0.1), (0.4, 0.3), (0.6, 0.3)):
    acts = {v: pair_actions(*raw[v], man[v], alpha=alpha, iou_thr=iou, cooc=cooc, verb_names=vn, noun_names=nn,
                            vocab=vocab) for v in tune}
    for ms, pad, mx, rw in itertools.product((0.05, 0.1, 0.15, 0.2, 0.3), (0.0, 0.4), (4.0, 6.0, 12.0), (0.0, 0.05)):
        hl = {v: select_highlights(acts[v], man[v], beta=0.0, pad=pad, min_score=ms, repeat_w=rw, max_seg=mx)
              for v in tune}
        c = cov(hl, tune)
        if best is None or c > best[0]:
            best = (c, alpha, iou, ms, pad, mx, rw)
            print(f"{alpha:.1f}  {iou:.1f} |   {ms:.2f}    {pad:.1f}   {mx:4.1f}    {rw:.2f}   | {c:.3f}  *", flush=True)
c, alpha, iou, ms, pad, mx, rw = best
acts = {v: pair_actions(*raw[v], man[v], alpha=alpha, iou_thr=iou, cooc=cooc, verb_names=vn, noun_names=nn, vocab=vocab)
        for v in held}
hl = {v: select_highlights(acts[v], man[v], beta=0.0, pad=pad, min_score=ms, repeat_w=rw, max_seg=mx) for v in held}
print(f"\nBEST on tune: alpha={alpha} pair_iou={iou} min_score={ms} pad={pad} max_seg={mx} repeat_w={rw}"
      f"  -> tune {c:.3f}, HELD-OUT {cov(hl, held):.3f}")
