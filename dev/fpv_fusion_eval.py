"""DEV ONLY — offline highlight-F1 evaluation on FPVSum from the dumped signals (dev/dump_fpv.py),
optionally fused with per-window VLM scores (kaggle/vlm_highlight_scores.ipynb).

  python dev/fpv_fusion_eval.py --dump ../data/fpv_dump --ann ../data/fpvsum_repo/annotation [--vlm ../data/vlm_scores]

Recomputes the zero-shot curve exactly as code/highlightness.py does (from cached CLIP features), packs windows with
code/coverage_selector.pack_windows, and reports frame-level F1 for every fusion weight, overall and on two fixed
halves of the videos (sorted ids, alternating). A setting is adopted only if it beats the baseline on BOTH halves.
"""
import argparse, ast, glob, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(os.path.dirname(HERE), "code")
sys.path.insert(0, CODE); sys.path.insert(0, HERE)
import config as C
from coverage_selector import pack_windows
from eval_local import load_fpv_scores, highlight_f1


def _consts(path):
    """Read the prompt lists from highlightness.py without importing it (it needs torch / open_clip)."""
    out = {}
    for node in ast.parse(open(path).read()).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except Exception:
                pass
    return out


H = _consts(os.path.join(CODE, "highlightness.py"))


def rz(x):
    x = np.asarray(x, np.float64); med = np.median(x); mad = np.median(np.abs(x - med)) + 1e-6
    return np.clip((x - med) / (1.4826 * mad), -3, 3)


def smooth(x, s):
    if s <= 0 or len(x) < 3:
        return x
    r = int(3 * s); k = np.exp(-0.5 * (np.arange(-r, r + 1) / s) ** 2); k /= k.sum()
    return np.convolve(np.pad(x, r, mode="edge"), k, mode="valid")


def novelty(f, window=20):
    nov = np.zeros(len(f))
    for i in range(1, len(f)):
        nov[i] = 1.0 - float((f[max(0, i - window):i] @ f[i]).max())
    if len(nov) > 1:
        nov[0] = nov[1]
    return nov


def label_of(vid):
    return vid.split("_")[0].lower()          # dev only: FPVSum ids start with the category (router is 46/46 correct)


def base_curve(d, label, texts):
    t = d["t"]; F = d["feats"].astype(np.float32); fps = C.HL_FPS
    F /= np.linalg.norm(F, axis=1, keepdims=True) + 1e-8
    emb = lambda L: np.stack([texts[s] for s in L])
    pos = emb(H["POS_BY_LABEL"].get(label) or H["POS_GENERIC"]); neg = emb(H["NEG_GENERIC"])
    sem = (F @ pos.T).mean(1) - (F @ neg.T).mean(1)
    aud = d["audio"]
    audio = rz(np.interp(t, (np.arange(len(aud)) + 0.5) / fps, aud)) if len(aud) else np.zeros(len(t))
    w = C.HL_WEIGHTS
    hl = w["semantic"] * rz(sem) + w["motion"] * rz(d["diff"]) + w["audio"] * audio + w["novelty"] * rz(novelty(F))
    if not len(aud):
        hl = hl / (1 - w["audio"])
    return t, smooth(hl, C.HL_SMOOTH_SIGMA_S * fps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True); ap.add_argument("--ann", required=True); ap.add_argument("--vlm")
    a = ap.parse_args()
    tx = np.load(os.path.join(a.dump, "_texts.npz"), allow_pickle=True)
    texts = {str(s): e / np.linalg.norm(e) for s, e in zip(tx["texts"], tx["emb"].astype(np.float32))}

    vids = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(a.dump, "*.npz")) if "_texts" not in p)
    data = []
    for v in vids:
        gt = load_fpv_scores(a.ann, v)
        if gt is None:
            continue
        d = np.load(os.path.join(a.dump, v + ".npz"), allow_pickle=True)
        t, hl = base_curve(d, label_of(v), texts)
        vl = None
        if a.vlm and os.path.exists(os.path.join(a.vlm, v + ".npz")):
            q = np.load(os.path.join(a.vlm, v + ".npz"))
            vl = {f"p{j}": rz(np.interp(t, q["t"], q["score"][:, j])) for j in range(q["score"].shape[1])}
            vl["mean"] = rz(np.mean([vl[k] for k in list(vl)], axis=0))
        data.append((v, float(d["duration"]), t, hl, vl, gt))
    print(f"{len(data)} videos with ground truth; {sum(x[4] is not None for x in data)} with VLM scores")

    def run(fn):
        hl = lambda x: [h if isinstance(h, dict) else {"segment": h} for h in pack_windows(fn(x), x[2], x[1])]
        f = np.array([highlight_f1(hl(x), x[5], x[1]) for x in data])
        return f.mean(), f[0::2].mean(), f[1::2].mean()

    base = run(lambda x: x[3])
    print(f"{'setting':<34}{'all':>8}{'half A':>8}{'half B':>8}")
    print(f"{'baseline (current submission)':<34}{base[0]:8.4f}{base[1]:8.4f}{base[2]:8.4f}")
    if not any(x[4] is not None for x in data):
        return
    keys = list(next(x[4] for x in data if x[4] is not None))
    sig = C.HL_SMOOTH_SIGMA_S * C.HL_FPS
    best = None
    for k in keys:
        for w in [0.25, 0.5, 1.0, 2.0, 4.0, None]:          # None = VLM curve alone
            def fn(x, k=k, w=w):
                if x[4] is None:
                    return x[3]
                v = smooth(x[4][k], sig)
                return v if w is None else x[3] + w * v
            r = run(fn)
            both = r[1] > base[1] and r[2] > base[2]
            name = f"{k} {'alone' if w is None else f'w={w}'}"
            print(f"{name:<34}{r[0]:8.4f}{r[1]:8.4f}{r[2]:8.4f}  {'BOTH HALVES UP' if both else ''}")
            if both and (best is None or r[0] > best[1][0]):
                best = (name, r)
    print("\nADOPT:", f"{best[0]}  F1 {best[1][0]:.4f} (+{(best[1][0] - base[0]) * 25:.2f} total points)" if best
          else "nothing — no setting beats the baseline on both halves")


if __name__ == "__main__":
    main()
