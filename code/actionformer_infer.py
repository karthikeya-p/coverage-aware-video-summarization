"""ActionFormer (EPIC-KITCHENS-100 verb & noun checkpoints) inference on SlowFast features, plus verb–noun pairing
into action instances with a train-split co-occurrence prior.

Output action instance: dict(segment=[s, e], verb=str, noun=str, score=float, verb_id=int, noun_id=int)
"""
import json, os, sys, numpy as np, torch
import config as C

_DETECTORS = {}


def _load_json(path, default=None):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default


class ActionFormerDetector:
    def __init__(self, cfg_path, ckpt_path, device=None):
        sys.path.insert(0, C.ACTIONFORMER_DIR)
        from libs.core import load_config
        from libs.modeling import make_meta_arch
        self.device = device or (C.DEVICE if torch.cuda.is_available() else "cpu")
        cfg = load_config(cfg_path)
        cfg["model"]["test_cfg"]["max_seg_num"] = C.MAX_ACTIONS
        cfg["model"]["test_cfg"]["min_score"] = 0.001
        self.cfg = cfg
        model = make_meta_arch(cfg["model_name"], **cfg["model"])
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt.get("state_dict_ema", ckpt.get("state_dict"))
        sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
        model.load_state_dict(sd)
        self.model = model.to(self.device).eval()
        d = cfg["dataset"]
        self.feat_stride, self.num_frames, self.fps = d["feat_stride"], d["num_frames"], d["default_fps"]

    @torch.no_grad()
    def detect(self, feats, duration, video_id="vid"):
        """feats: (T, C) numpy. Returns arrays segments (N,2) in seconds, scores (N,), labels (N,)."""
        x = torch.from_numpy(np.ascontiguousarray(feats.T.astype(np.float32)))   # C x T
        item = {"video_id": video_id, "feats": x, "segments": None, "labels": None,
                "fps": float(self.fps), "duration": float(duration),
                "feat_stride": int(self.feat_stride), "feat_num_frames": int(self.num_frames)}
        out = self.model([item])[0]
        seg = out["segments"].cpu().numpy().astype(np.float64)
        sc = out["scores"].cpu().numpy().astype(np.float64)
        lb = out["labels"].cpu().numpy().astype(np.int64)
        seg = np.clip(seg, 0.0, duration)
        keep = (seg[:, 1] - seg[:, 0]) > 0.05
        return seg[keep], sc[keep], lb[keep]


_MAPS = {}


def get_maps():
    """Class-id -> name maps, co-occurrence prior and supported vocabulary (no model is loaded)."""
    if not _MAPS:
        _MAPS["verb_names"] = {int(k): v for k, v in _load_json(C.VERB_LABELS, {}).items()}
        _MAPS["noun_names"] = {int(k): v for k, v in _load_json(C.NOUN_LABELS, {}).items()}
        _MAPS["cooc"] = _load_json(C.COOC_PRIOR, None)
        _MAPS["vocab"] = _load_json(C.SUPPORTED_CLASSES, None)
    return _MAPS


def get_detectors():
    if not _DETECTORS:
        _DETECTORS["verb"] = ActionFormerDetector(C.AF_VERB_CFG, C.AF_VERB_CKPT)
        _DETECTORS["noun"] = ActionFormerDetector(C.AF_NOUN_CFG, C.AF_NOUN_CKPT)
        _DETECTORS["verb_names"] = {int(k): v for k, v in _load_json(C.VERB_LABELS, {}).items()}
        _DETECTORS["noun_names"] = {int(k): v for k, v in _load_json(C.NOUN_LABELS, {}).items()}
        _DETECTORS["cooc"] = _load_json(C.COOC_PRIOR, None)      # {"<verb_id>,<noun_id>": count}
        _DETECTORS["vocab"] = _load_json(C.SUPPORTED_CLASSES, None)
    return _DETECTORS


def tiou_matrix(a, b):
    """a: (N,2), b: (M,2) -> (N,M) temporal IoU."""
    inter = np.maximum(0.0, np.minimum(a[:, None, 1], b[None, :, 1]) - np.maximum(a[:, None, 0], b[None, :, 0]))
    union = (a[:, None, 1] - a[:, None, 0]) + (b[None, :, 1] - b[None, :, 0]) - inter
    return inter / np.maximum(union, 1e-6)


def _segment_class_probs(clip_probs, segs):
    """Mean class probability over the feature clips whose centre lies inside each segment (nearest clip if none).
    clip_probs: (T, K) per-clip softmax; segs: (N, 2) seconds -> (N, K)."""
    pm = np.asarray(clip_probs, dtype=np.float64)
    t = (np.arange(len(pm)) * C.FEAT_STRIDE + C.FEAT_NUM_FRAMES / 2.0) / C.FEAT_FPS
    cs = np.concatenate([np.zeros((1, pm.shape[1])), np.cumsum(pm, 0)])
    a = np.searchsorted(t, segs[:, 0], "left"); b = np.searchsorted(t, segs[:, 1], "right")
    mid = np.clip(np.searchsorted(t, 0.5 * (segs[:, 0] + segs[:, 1])), 0, len(pm) - 1)
    empty = b <= a
    a = np.where(empty, mid, a); b = np.where(empty, mid + 1, b)
    return (cs[b] - cs[a]) / (b - a)[:, None]


def pair_actions(vseg, vsc, vlb, nseg, nsc, nlb, duration, alpha=C.PAIR_ALPHA, iou_thr=C.PAIR_IOU,
                 cooc=None, verb_names=None, noun_names=None, vocab=None, max_actions=C.MAX_ACTIONS,
                 topk=C.PAIR_TOPK, per_verb=C.PAIR_PER_VERB, blend=C.PAIR_BLEND,
                 verb_prob=None, noun_prob=None, cls_topk=C.PAIR_CLS_TOPK):
    """Combine verb and noun detections into verb-noun action instances.

    Every verb segment is paired with its `per_verb` best-scoring noun segments among those with tIoU >= iou_thr;
    score = p_v^alpha * p_n^(1-alpha), boundaries = score-weighted average. Near-duplicates of the same pair are
    suppressed (tIoU > 0.5). Because at most `max_actions` may be submitted and mAP averages over classes, the
    list is cut with a class-balanced key  score / (1 + rank_within_pair)^blend  rather than by raw score.
    `sel_score` (score x train-split co-occurrence prior) is what the highlight selector uses.

    If the SlowFast clip classifier's per-clip verb / noun probabilities are given, two more candidate sources
    are added: each verb segment with the classifier's `cls_topk` most likely nouns over that segment, and each
    noun segment with its most likely verbs. This recovers actions where only one of the two detectors fired.
    (Settings chosen on tune_ids.txt; see sweep notes in README.)
    """
    verb_names = verb_names or {}; noun_names = noun_names or {}
    valid_verbs = set(vocab["verbs"]) if vocab else None
    valid_nouns = set(vocab["nouns"]) if vocab else None
    vo = np.argsort(-vsc)[:topk]; no = np.argsort(-nsc)[:topk]
    vseg, vsc, vlb = vseg[vo], vsc[vo], vlb[vo]
    nseg, nsc, nlb = nseg[no], nsc[no], nlb[no]
    cands = []
    if len(vseg) and len(nseg):
        M = tiou_matrix(vseg, nseg)
        ok = M >= iou_thr
        S = (vsc[:, None] ** alpha) * (nsc[None, :] ** (1 - alpha))
        js = np.argsort(-np.where(ok, nsc[None, :], -1.0), axis=1)[:, :per_verb]
        for i in range(len(vseg)):
            for j in js[i]:
                if not ok[i, j]:
                    continue
                w = vsc[i] / (vsc[i] + nsc[j] + 1e-9)
                seg = w * vseg[i] + (1 - w) * nseg[j]
                cands.append((float(S[i, j]), float(seg[0]), float(seg[1]), int(vlb[i]), int(nlb[j])))
    if verb_prob is not None and noun_prob is not None and cls_topk > 0:
        if len(vseg):
            pn = _segment_class_probs(noun_prob, vseg); top = np.argsort(-pn, axis=1)[:, :cls_topk]
            for i in range(len(vseg)):
                for j in top[i]:
                    cands.append((float(vsc[i] ** alpha * pn[i, j] ** (1 - alpha)), float(vseg[i, 0]),
                                  float(vseg[i, 1]), int(vlb[i]), int(j)))
        if len(nseg):
            pv = _segment_class_probs(verb_prob, nseg); top = np.argsort(-pv, axis=1)[:, :cls_topk]
            for i in range(len(nseg)):
                for j in top[i]:
                    cands.append((float(nsc[i] ** (1 - alpha) * pv[i, j] ** alpha), float(nseg[i, 0]),
                                  float(nseg[i, 1]), int(j), int(nlb[i])))
    cands.sort(key=lambda c: -c[0])
    kept, by_pair = [], {}
    for s, a, b, v, n in cands:                     # per-pair NMS, remembering the rank inside the pair
        lst = by_pair.setdefault((v, n), [])
        dup = False
        for (a2, b2) in lst:
            inter = max(0.0, min(b, b2) - max(a, a2)); union = (b - a) + (b2 - a2) - inter
            if union > 0 and inter / union > 0.5:
                dup = True; break
        if dup:
            continue
        kept.append((-s / (1.0 + len(lst)) ** blend, s, a, b, v, n)); lst.append((a, b))
    kept.sort(key=lambda c: c[0])
    kept = sorted(kept[:max_actions], key=lambda c: -c[1])

    actions = []
    for _, s, a, b, v, n in kept:
        verb = verb_names.get(v, "other"); noun = noun_names.get(n, "other")
        if valid_verbs is not None and verb not in valid_verbs:
            verb = "other"
        if valid_nouns is not None and noun not in valid_nouns:
            noun = "other"
        a = max(0.0, min(a, duration)); b = max(a + 0.1, min(b, duration))
        s = float(min(1.0, max(0.0, s)))
        prior = 1.0 if (cooc is None or cooc.get(f"{v},{n}", 0) > 0) else C.COOC_UNSEEN_PENALTY
        actions.append(dict(segment=[round(a, 2), round(b, 2)], verb=verb, noun=noun, score=round(s, 4),
                            sel_score=round(s * prior, 4), verb_id=v, noun_id=n))
    return actions


def detect_actions(feats, duration, video_id="vid", verb_prob=None, noun_prob=None):
    D = get_detectors()
    vseg, vsc, vlb = D["verb"].detect(feats, duration, video_id)
    nseg, nsc, nlb = D["noun"].detect(feats, duration, video_id)
    actions = pair_actions(vseg, vsc, vlb, nseg, nsc, nlb, duration, cooc=D["cooc"],
                           verb_names=D["verb_names"], noun_names=D["noun_names"], vocab=D["vocab"],
                           verb_prob=verb_prob, noun_prob=noun_prob)
    raw = dict(verb=(vseg, vsc, vlb), noun=(nseg, nsc, nlb))
    return actions, raw
