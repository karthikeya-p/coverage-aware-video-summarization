#!/usr/bin/env python3
"""Local re-implementation of the four Track 04 metrics for *evaluation only*.
Ground truth is never read by the inference code.

  python eval_local.py --outputs outputs/ --manifest ../student_package/manifest.csv \
      --epic-ann ../data/epic-kitchens-100-annotations --fpv-ann ../data/fpvsum_repo/annotation [--split-file tune_ids.txt]

Metrics (our best reading of TEAM_README.md — the organisers' exact code is not public):
  * Action mAP      : per-(verb,noun) class AP, tIoU 0.1..0.5, averaged over classes with GT (ActivityNet style)
  * Action coverage : distinct GT (verb,noun) pairs with >= 1 instance captured by highlights. We report two capture
                      rules — midpoint-inside and >=50 % overlap — plus the soft budget penalty min(1, budget/total).
  * Highlight F1    : frame-level F1 vs top-15 % consensus frames (mean of annotators).
  * Label accuracy  : video-level category.
"""
import argparse, csv, glob, json, os, sys
import numpy as np


# ----------------------------------------------------------------------------- EPIC GT
def _ts(s):
    h, m, sec = s.split(":"); return int(h) * 3600 + int(m) * 60 + float(sec)


def load_epic_gt(ann_dir):
    classes = {}
    for kind in ("verb", "noun"):
        with open(os.path.join(ann_dir, f"EPIC_100_{kind}_classes.csv"), newline="") as f:
            classes[kind] = {int(r["id"]): r["key"] for r in csv.DictReader(f)}
    gt = {}
    with open(os.path.join(ann_dir, "EPIC_100_validation.csv"), newline="") as f:
        for r in csv.DictReader(f):
            gt.setdefault(r["video_id"], []).append(dict(
                segment=[_ts(r["start_timestamp"]), _ts(r["stop_timestamp"])],
                verb=classes["verb"][int(r["verb_class"])], noun=classes["noun"][int(r["noun_class"])]))
    return gt


def tiou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    return inter / max(1e-9, (a[1] - a[0]) + (b[1] - b[0]) - inter)


def action_map(preds, gts, thresholds=np.linspace(0.1, 0.5, 5), key=lambda a: (a["verb"], a["noun"])):
    """preds/gts: {video_id: [action dicts]} -> mean AP over classes and thresholds."""
    by_cls_gt, by_cls_pred = {}, {}
    for vid, acts in gts.items():
        for a in acts:
            by_cls_gt.setdefault(key(a), []).append((vid, a["segment"]))
    for vid, acts in preds.items():
        if vid not in gts:
            continue
        for a in acts:
            by_cls_pred.setdefault(key(a), []).append((a["score"], vid, a["segment"]))
    aps = []
    for cls, g in by_cls_gt.items():
        p = sorted(by_cls_pred.get(cls, []), key=lambda x: -x[0])
        cls_ap = []
        for thr in thresholds:
            used = set(); tp = np.zeros(len(p)); fp = np.zeros(len(p))
            g_by_vid = {}
            for i, (vid, seg) in enumerate(g):
                g_by_vid.setdefault(vid, []).append((i, seg))
            for k, (_, vid, seg) in enumerate(p):
                best, bi = 0.0, -1
                for i, gseg in g_by_vid.get(vid, []):
                    if i in used:
                        continue
                    o = tiou(seg, gseg)
                    if o > best:
                        best, bi = o, i
                if best >= thr:
                    tp[k] = 1; used.add(bi)
                else:
                    fp[k] = 1
            if len(p) == 0:
                cls_ap.append(0.0); continue
            ctp, cfp = np.cumsum(tp), np.cumsum(fp)
            rec = ctp / len(g); prec = ctp / np.maximum(ctp + cfp, 1e-9)
            # interpolated AP
            mrec = np.r_[0, rec, 1]; mpre = np.r_[0, prec, 0]
            for i in range(len(mpre) - 2, -1, -1):
                mpre[i] = max(mpre[i], mpre[i + 1])
            idx = np.where(mrec[1:] != mrec[:-1])[0]
            cls_ap.append(float(((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).sum()))
        aps.append(np.mean(cls_ap))
    return float(np.mean(aps)) if aps else 0.0


def coverage(pred_hl, gt_acts, duration, budget_frac=0.15):
    pairs = {}
    for a in gt_acts:
        pairs.setdefault((a["verb"], a["noun"]), []).append(a["segment"])
    def captured(seg, rule):
        s, e = seg
        for h in pred_hl:
            hs, he = h["segment"]
            if rule == "mid":
                if hs <= 0.5 * (s + e) <= he:
                    return True
            else:
                if max(0.0, min(e, he) - max(s, hs)) >= 0.5 * (e - s):
                    return True
        return False
    res = {}
    for rule in ("mid", "half"):
        got = sum(any(captured(sg, rule) for sg in segs) for segs in pairs.values())
        res[rule] = got / max(1, len(pairs))
    total = sum(h["segment"][1] - h["segment"][0] for h in pred_hl)
    res["penalty"] = min(1.0, (budget_frac * duration) / max(total, 1e-9))
    res["total_frac"] = total / duration
    res["n_pairs"] = len(pairs)
    return res


# ----------------------------------------------------------------------------- FPVSum GT
_FPV = {}


def load_fpv_scores(ann_dir, video_id):
    """Per-frame consensus importance from FPVSum annotation/gt_data.mat (struct array: Name, Score, ...)."""
    if not _FPV:
        import scipy.io as sio
        m = sio.loadmat(os.path.join(ann_dir, "gt_data.mat"), squeeze_me=True, struct_as_record=False)
        for e in np.atleast_1d(m["gt_data"]):
            _FPV[str(e.Name)] = np.asarray(e.Score, dtype=np.float64)
    return _FPV.get(video_id)


def highlight_f1(pred_hl, scores, duration, top_frac=0.15):
    sc = np.asarray(scores, dtype=np.float64)
    if sc.ndim == 2:
        sc = sc.mean(0) if sc.shape[0] < sc.shape[1] else sc.mean(1)
    n = len(sc); t = (np.arange(n) + 0.5) * duration / n
    k = max(1, int(round(top_frac * n)))
    gt = np.zeros(n, bool); gt[np.argsort(-sc)[:k]] = True
    pred = np.zeros(n, bool)
    for h in pred_hl:
        s, e = h["segment"]; pred |= (t >= s) & (t <= e)
    tp = (gt & pred).sum(); p = tp / max(pred.sum(), 1); r = tp / max(gt.sum(), 1)
    return 2 * p * r / max(p + r, 1e-9)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", required=True); ap.add_argument("--manifest", required=True)
    ap.add_argument("--epic-ann", default=None); ap.add_argument("--fpv-ann", default=None)
    ap.add_argument("--split-file", default=None, help="only evaluate these video_ids (one per line)")
    a = ap.parse_args()
    man = {r["video_id"]: r for r in csv.DictReader(open(a.manifest, newline=""))}
    keep = set(open(a.split_file).read().split()) if a.split_file else None
    preds = {}
    for fp in glob.glob(os.path.join(a.outputs, "*.json")):
        d = json.load(open(fp)); preds[d["video_id"]] = d
    report = {}

    if a.epic_ann:
        gt = load_epic_gt(a.epic_ann)
        vids = [v for v in preds if v in gt and man.get(v, {}).get("dataset") == "epic_kitchens_100"
                and (keep is None or v in keep)]
        report["epic_videos"] = len(vids)
        report["action_mAP"] = action_map({v: preds[v]["actions"] for v in vids}, {v: gt[v] for v in vids})
        report["verb_mAP"] = action_map({v: preds[v]["actions"] for v in vids}, {v: gt[v] for v in vids},
                                        key=lambda x: x["verb"])
        report["noun_mAP"] = action_map({v: preds[v]["actions"] for v in vids}, {v: gt[v] for v in vids},
                                        key=lambda x: x["noun"])
        cov = [coverage(preds[v]["highlights"], gt[v], float(man[v]["duration_sec"])) for v in vids]
        for rule in ("mid", "half"):
            report[f"coverage_{rule}"] = float(np.mean([c[rule] for c in cov])) if cov else 0
            report[f"coverage_{rule}_penalised"] = float(np.mean([c[rule] * c["penalty"] for c in cov])) if cov else 0
        report["mean_highlight_frac"] = float(np.mean([c["total_frac"] for c in cov])) if cov else 0

    if a.fpv_ann:
        vids = [v for v in preds if man.get(v, {}).get("dataset") == "fpvsum" and (keep is None or v in keep)]
        f1s, acc, n = [], [], 0
        for v in vids:
            sc = load_fpv_scores(a.fpv_ann, v)
            if sc is None:
                continue
            n += 1
            f1s.append(highlight_f1(preds[v]["highlights"], sc, float(man[v]["duration_sec"])))
            acc.append(preds[v]["label"]["name"] == v.split("_")[0].lower())
        report["fpv_videos_with_gt"] = n
        report["highlight_F1"] = float(np.mean(f1s)) if f1s else None
        report["label_acc"] = float(np.mean(acc)) if acc else None

    if all(k in report for k in ("action_mAP", "coverage_half_penalised")) and report.get("highlight_F1") is not None:
        report["est_total_/100"] = 100 * (0.25 * report["action_mAP"] + 0.40 * report["coverage_half_penalised"]
                                          + 0.25 * report["highlight_F1"] + 0.10 * (report["label_acc"] or 0))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
