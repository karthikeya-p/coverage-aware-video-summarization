#!/usr/bin/env python3
"""Re-implementation of the organisers' validate.py from TEAM_README.md (the official file was missing from the
student package — replace this with the official one when you get it; keep CLI-compatible).

  python validate.py --file out.json [-v]
  python validate.py --dir outputs/ [--manifest manifest.csv] [--classes supported_classes.json]
Standard library only.
"""
import argparse, csv, json, math, os, sys

REQUIRED = ["team_id", "video_id", "duration_sec", "label", "actions", "highlights"]
MIN_SEG = 4.0
BUDGET = 0.15


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def validate_obj(d, fname=None, manifest=None, classes=None):
    errs, warns = [], []
    if not isinstance(d, dict):
        return ["root is not an object"], warns
    for k in REQUIRED:
        if k not in d:
            errs.append(f"missing key '{k}'")
    if errs:
        return errs, warns
    if not isinstance(d["team_id"], str) or not d["team_id"].strip():
        errs.append("team_id must be a non-empty string")
    vid = d["video_id"]
    if not isinstance(vid, str) or not vid:
        errs.append("video_id must be a non-empty string")
    elif fname and os.path.splitext(os.path.basename(fname))[0] != vid:
        errs.append(f"video_id '{vid}' != filename '{fname}'")
    dur = d["duration_sec"]
    if not _num(dur) or dur <= 0:
        errs.append("duration_sec must be a positive number"); dur = None
    elif manifest is not None:
        if vid not in manifest:
            errs.append(f"video_id '{vid}' not in manifest")
        elif abs(manifest[vid] - dur) > 0.5:
            errs.append(f"duration_sec {dur} differs from manifest {manifest[vid]} by > 0.5 s")
    lab = d["label"]
    if not (isinstance(lab, dict) and isinstance(lab.get("name"), str) and _num(lab.get("score"))
            and 0 <= lab["score"] <= 1):
        errs.append("label must be {name: str, score: number in [0,1]}")
    elif classes and lab["name"] not in classes["labels"]:
        warns.append(f"label '{lab['name']}' not in supported labels")
    acts = d["actions"]
    if not isinstance(acts, list) or len(acts) > 2000:
        errs.append("actions must be a list of 0–2000 items")
    else:
        for i, a in enumerate(acts):
            ok = (isinstance(a, dict) and isinstance(a.get("segment"), list) and len(a["segment"]) == 2
                  and all(_num(x) for x in a["segment"]) and isinstance(a.get("verb"), str)
                  and isinstance(a.get("noun"), str) and _num(a.get("score")) and 0 <= a["score"] <= 1)
            if not ok:
                errs.append(f"actions[{i}] malformed"); continue
            s, e = a["segment"]
            if not (0 <= s < e) or (dur is not None and e > dur + 0.5):
                errs.append(f"actions[{i}] segment [{s},{e}] out of range")
            if classes:
                if a["verb"] not in classes["verbs"]:
                    warns.append(f"actions[{i}] verb '{a['verb']}' not supported")
                if a["noun"] not in classes["nouns"]:
                    warns.append(f"actions[{i}] noun '{a['noun']}' not supported")
    hls = d["highlights"]
    if not isinstance(hls, list) or len(hls) < 1:
        errs.append("highlights must be a non-empty list")
    else:
        prev_end, total, any_long = -1.0, 0.0, False
        for i, h in enumerate(hls):
            ok = (isinstance(h, dict) and isinstance(h.get("segment"), list) and len(h["segment"]) == 2
                  and all(_num(x) for x in h["segment"]))
            if not ok:
                errs.append(f"highlights[{i}] malformed"); continue
            s, e = h["segment"]
            if not (0 <= s < e) or (dur is not None and e > dur + 0.5):
                errs.append(f"highlights[{i}] segment [{s},{e}] out of range")
            if s < prev_end:
                errs.append(f"highlights[{i}] not sorted / overlaps previous (start {s} < prev end {prev_end})")
            prev_end = e
            L = e - s; total += L
            if L >= MIN_SEG:
                any_long = True
            else:
                warns.append(f"highlights[{i}] shorter than {MIN_SEG}s (ignored in scoring)")
        if not any_long:
            errs.append(f"no highlight segment >= {MIN_SEG}s")
        if dur and total > BUDGET * dur + 1e-6:
            warns.append(f"total highlight length {total:.2f}s exceeds 15% budget ({BUDGET * dur:.2f}s)")
    return errs, warns


def load_manifest(path):
    if not path or not os.path.exists(path):
        return None
    with open(path, newline="") as f:
        return {r["video_id"]: float(r["duration_sec"]) for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file"); ap.add_argument("--dir")
    ap.add_argument("--manifest", default=None); ap.add_argument("--classes", default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    manifest = load_manifest(a.manifest)
    classes = json.load(open(a.classes)) if a.classes and os.path.exists(a.classes) else None
    files = [a.file] if a.file else sorted(os.path.join(a.dir, f) for f in os.listdir(a.dir) if f.endswith(".json"))
    bad, team_ids = 0, set()
    for fp in files:
        try:
            d = json.load(open(fp))
        except Exception as e:
            print(f"FAIL {fp}: invalid JSON ({e})"); bad += 1; continue
        errs, warns = validate_obj(d, fp, manifest, classes)
        if isinstance(d, dict) and "team_id" in d:
            team_ids.add(d["team_id"])
        if errs:
            bad += 1; print(f"FAIL {fp}")
            for e in errs: print("   ERROR:", e)
        elif a.verbose or a.file:
            print(f"OK   {fp}")
        if warns and (a.verbose or a.file):
            for w in warns[:20]: print("   warn:", w)
    if len(team_ids) > 1:
        print(f"ERROR: multiple team_ids across files: {sorted(team_ids)}"); bad += 1
    if manifest and a.dir:
        have = {os.path.splitext(os.path.basename(f))[0] for f in files}
        miss = sorted(set(manifest) - have)
        if miss:
            print(f"WARNING: {len(miss)} manifest videos have no JSON: {miss[:10]}{'...' if len(miss) > 10 else ''}")
    print(f"{len(files) - bad}/{len(files)} files valid")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
