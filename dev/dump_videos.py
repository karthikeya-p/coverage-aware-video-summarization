"""DEV ONLY — dump per-frame signals (CLIP image features at 2 fps, frame-difference, optical flow, audio RMS)
for every .mp4 in a folder. Used to train / test a highlight scorer on a public third-party dataset (TVSum).

  python dev/dump_videos.py --video-dir ../data/TVSum --out ../data/tvsum_dump
"""
import argparse, glob, os, sys, time
import numpy as np, cv2
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "code"))  # inference code
import config as C
from video_io import iter_frames, audio_rms, probe
from clip_utils import embed_images
import highlightness as H

ap = argparse.ArgumentParser()
ap.add_argument("--video-dir", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
files = sorted(glob.glob(os.path.join(a.video_dir, "*.mp4")))
t0 = time.time()
for i, p in enumerate(files):
    vid = os.path.splitext(os.path.basename(p))[0]; outp = os.path.join(a.out, vid + ".npz")
    if os.path.exists(outp):
        continue
    try:
        frames, small = [], []
        for _, f in iter_frames(p, fps=C.HL_FPS, short_side=224):
            frames.append(f); small.append(cv2.resize(f, (128, 96), interpolation=cv2.INTER_AREA))
        diff, flow = H.motion_curves(small)
        aud = audio_rms(p, hop_s=1.0 / C.HL_FPS)
        np.savez_compressed(outp, t=(np.arange(len(frames)) + 0.5) / C.HL_FPS, feats=embed_images(frames).astype(np.float16),
                            diff=diff, flow=flow, audio=aud if aud is not None else np.zeros(0), duration=probe(p)["duration"])
        print(f"{i + 1}/{len(files)} {vid} {len(frames)} frames  [{time.time() - t0:.0f}s]", flush=True)
    except Exception as e:
        print(f"!! {vid} failed: {e}", flush=True)
