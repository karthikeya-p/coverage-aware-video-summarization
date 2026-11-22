"""Run the pipeline over every manifest video whose MP4 can be found; skips existing JSONs.

python code/batch_manifest.py --manifest ../student_package/manifest.csv \
    --video-roots ../data/EPIC-KITCHENS ../data/FPVSum --out outputs --cache ../data/feat_cache [--only epic|fpvsum]
Re-run as downloads land. Failures are listed in submission/_batch_failures.json (kept out of outputs/).
"""
import argparse, csv, glob, json, os, sys, time, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from run_video import process


def find_video(root_dirs, video_id):
    for root in root_dirs:
        for ext in ("MP4", "mp4", "mkv", "webm"):
            hits = glob.glob(os.path.join(root, "**", f"{video_id}.{ext}"), recursive=True)
            if hits:
                return hits[0]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(C.ROOT, "..", "student_package", "manifest.csv"))
    ap.add_argument("--video-roots", nargs="+", required=True)
    ap.add_argument("--out", default=os.path.join(C.ROOT, "outputs"))
    ap.add_argument("--cache", default=None)
    ap.add_argument("--only", choices=["epic", "fpvsum"], default=None)
    ap.add_argument("--team-id", default=None)
    ap.add_argument("--redo", action="store_true")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.manifest, newline="")))
    if a.only:
        rows = [r for r in rows if r["dataset"].startswith("epic" if a.only == "epic" else "fpv")]
    os.makedirs(a.out, exist_ok=True)
    failures, done, missing = [], 0, []
    t0 = time.time()
    for i, r in enumerate(rows):
        vid = r["video_id"]; outp = os.path.join(a.out, f"{vid}.json")
        if os.path.exists(outp) and not a.redo:
            done += 1; continue
        vp = find_video(a.video_roots, vid)
        if vp is None:
            missing.append(vid); continue
        try:
            domain = None   # the router decides from pixels, exactly as run.sh does
            process(vp, outp, team_id=a.team_id, cache_dir=a.cache, force_domain=domain)
            done += 1
        except Exception as e:  # keep going; report at the end
            traceback.print_exc(); failures.append(dict(video_id=vid, error=str(e)))
        print(f"--- {done}/{len(rows)} done, {len(missing)} missing, {len(failures)} failed, "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)
    json.dump(dict(failures=failures, missing=missing), open(os.path.join(a.out, "..", "_batch_failures.json"), "w"), indent=1)
    print(f"done={done} missing={len(missing)} failed={len(failures)}")


if __name__ == "__main__":
    main()
