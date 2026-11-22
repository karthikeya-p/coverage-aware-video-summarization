"""Single-video entry point:  python run_video.py <video.mp4> <out.json> [--team-id X] [--cache-dir D]

Pipeline
  1. ffprobe duration
  2. CLIP frames @2 fps -> video label (router) + highlightness / action-ness curves
  3a. cooking  -> SlowFast features -> ActionFormer verb/noun -> paired actions -> coverage-aware highlights
  3b. sport    -> actions = []       -> importance knapsack highlights
  4. JSON in the organiser schema
"""
import argparse, json, os, sys, time, csv
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from video_io import probe
from coverage_selector import select_highlights, pack_windows


def _manifest_lookup(video_id):
    """Optional: if the organiser manifest is present next to the code, use its duration (must be within ±0.5 s)."""
    for p in [os.path.join(C.WEIGHTS, "manifest.csv"), os.path.join(C.ROOT, "..", "student_package", "manifest.csv")]:
        if os.path.exists(p):
            with open(p, newline="") as f:
                for r in csv.DictReader(f):
                    if r["video_id"] == video_id:
                        return float(r["duration_sec"]), r["dataset"]
    return None, None


def process(video_path, out_path, team_id=None, cache_dir=None, force_domain=None, verbose=True):
    t0 = time.time()
    team_id = team_id or C.TEAM_ID
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    info = probe(video_path)
    duration = round(info["duration"], 2)
    m_dur, m_ds = _manifest_lookup(video_id)
    if m_dur is not None and abs(m_dur - duration) > 0.5:
        print(f"[warn] probed duration {duration} differs from manifest {m_dur}; using manifest", file=sys.stderr)
        duration = m_dur

    # ---- 2. curves + label
    from highlightness import score_video
    hl = score_video(video_path)
    label, label_score = hl["label"], hl["label_score"]
    domain = force_domain or ("cooking" if label == "cooking" else "sport")
    if verbose:
        print(f"[{video_id}] {duration:.1f}s label={label} ({label_score:.2f}) domain={domain} "
              f"curves in {time.time() - t0:.1f}s", flush=True)

    actions, highlights, debug = [], [], {}
    if domain == "cooking":
        # ---- 3a. action detection (cached) -> verb-noun pairing -> coverage selection
        from actionformer_infer import pair_actions, get_maps
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        if C.DETECTOR == "adatad":
            import adatad_infer
            raw = None
            fp = os.path.join(cache_dir, f"{video_id}.adatad.npz") if cache_dir else None
            if fp and os.path.exists(fp):
                z = np.load(fp); raw = {k: z[k] for k in z.files}
            if raw is None:
                raw = adatad_infer.detect_verb_noun(video_path, duration, verbose=verbose)
                if fp:
                    np.savez_compressed(fp, **raw)
            if verbose:
                print(f"[{video_id}] AdaTAD: {len(raw['vc'])} verb / {len(raw['nc'])} noun detections at "
                      f"{time.time() - t0:.1f}s", flush=True)
            M = get_maps()
            actions = pair_actions(raw["vs"], raw["vc"], raw["vl"].astype(int), raw["ns"], raw["nc"], raw["nl"].astype(int),
                                   duration, cooc=M["cooc"], verb_names=M["verb_names"], noun_names=M["noun_names"],
                                   vocab=M["vocab"], cls_topk=0)
            highlights, stats = select_highlights(actions, duration, min_score=C.SEL_MIN_SCORE_ADATAD, return_stats=True)
        else:
            from slowfast_extract import extract
            from actionformer_infer import detect_actions
            res = None
            if cache_dir:
                fp = os.path.join(cache_dir, f"{video_id}.npz")
                if os.path.exists(fp):
                    z = np.load(fp); res = {k: z[k] for k in z.files}
            if res is None:
                res = extract(video_path, progress=verbose)
                if cache_dir:
                    np.savez_compressed(fp, **res)
            feats = res["feats"]
            if verbose:
                print(f"[{video_id}] features {feats.shape} at {time.time() - t0:.1f}s", flush=True)
            actions, _ = detect_actions(feats, duration, video_id, res.get("verb_prob"), res.get("noun_prob"))
            highlights, stats = select_highlights(actions, duration, return_stats=True)
        debug = dict(selector=stats, n_actions=len(actions))
        label = "cooking"
        label_score = max(label_score, 0.5) if hl["label"] == "cooking" else 0.5
    else:
        # ---- 3b. importance knapsack
        highlights = pack_windows(hl["highlight"], hl["t"], duration)

    out = {
        "team_id": team_id,
        "video_id": video_id,
        "duration_sec": duration,
        "label": {"name": label, "score": float(round(label_score, 4))},
        "actions": [{"segment": a["segment"], "verb": a["verb"], "noun": a["noun"], "score": a["score"]}
                    for a in actions[:C.MAX_ACTIONS]],
        "highlights": [{"segment": [float(s), float(e)]} for s, e in highlights],
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)
    if verbose:
        tot = sum(e - s for s, e in highlights)
        print(f"[{video_id}] {len(actions)} actions, {len(highlights)} highlights ({tot:.1f}s = "
              f"{100 * tot / duration:.1f}%), token reduction x{duration / max(tot, 1e-6):.1f}; "
              f"done in {time.time() - t0:.1f}s", flush=True)
    return out, debug


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video"); ap.add_argument("out")
    ap.add_argument("--team-id", default=None)
    ap.add_argument("--cache-dir", default=os.environ.get("FEAT_CACHE"))
    ap.add_argument("--domain", choices=["cooking", "sport"], default=None, help="override the router")
    a = ap.parse_args()
    process(a.video, a.out, a.team_id, a.cache_dir, a.domain)
