"""DEV ONLY — dump per-frame signals for the FPVSum videos so the zero-shot highlight scorer and label router can
be tuned offline (no ground truth is read here).

  DEVICE=cpu python dev/dump_fpv.py --video-root ../data/FPVSum --out ../data/fpv_dump
Writes <video_id>.npz (t, CLIP image feats, frame-diff, optical-flow, audio RMS) and _texts.npz (prompt bank embeddings).
"""
import argparse, csv, glob, os, sys, time
import numpy as np, cv2
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "code"))  # inference code
import config as C
from video_io import iter_frames, audio_rms, probe
from clip_utils import embed_images, embed_texts, LABEL_DESC, TEMPLATES
import highlightness as H

ap = argparse.ArgumentParser()
ap.add_argument("--video-root", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

bank = set(H.POS_GENERIC + H.NEG_GENERIC + H.HOI_POS + H.HOI_NEG)
for v in H.POS_BY_LABEL.values(): bank.update(v)
for name, descs in LABEL_DESC.items():
    bank.update(t.format(d) for d in descs for t in TEMPLATES)
    bank.update([f"a first-person video of {name}", f"a GoPro point-of-view video of {name}", name])
bank.update([
    "bicycle handlebars seen from the rider's point of view", "riding a bicycle on a street or trail, pedalling",
    "a motorcycle dashboard and handlebars with a speedometer", "the cockpit of a small aeroplane with flight instruments",
    "view out of an aeroplane window over land", "skydiving in free fall above the ground", "wingsuit flying or base jumping off a cliff",
    "bungee jumping", "riding a horse, the horse's ears and mane in front", "a boxing ring, gloves in front of the camera",
    "an interesting, eventful moment", "a boring, uneventful moment", "a spectacular view", "fast movement", "slow movement",
    "a close interaction with another person or animal", "a wide scenic landscape", "the start of the activity, getting ready",
    "the end of the activity, stopping", "a crash, fall or near miss", "an obstacle ahead", "a sharp turn", "in mid-air",
    "people cheering or celebrating", "a person waving at the camera", "an empty static scene", "the camera pointing at the ground",
    "the camera pointing at the sky", "a shaky, blurred frame", "text or a logo on screen",
])
bank = sorted(bank)
np.savez_compressed(os.path.join(a.out, "_texts.npz"), texts=np.array(bank, dtype=object), emb=embed_texts(bank))
print(len(bank), "prompts embedded", flush=True)

rows = [r for r in csv.DictReader(open(os.path.join(C.WEIGHTS, "manifest.csv"), newline="")) if r["dataset"] == "fpvsum"]
t0 = time.time()
for i, r in enumerate(rows):
    vid = r["video_id"]; outp = os.path.join(a.out, vid + ".npz")
    hits = glob.glob(os.path.join(a.video_root, "**", vid + ".mp4"), recursive=True)
    if not hits or os.path.exists(outp):
        continue
    frames, small = [], []
    for _, f in iter_frames(hits[0], fps=C.HL_FPS, short_side=224):
        frames.append(f); small.append(cv2.resize(f, (128, 96), interpolation=cv2.INTER_AREA))
    diff, flow = H.motion_curves(small)
    aud = audio_rms(hits[0], hop_s=1.0 / C.HL_FPS)
    gray = np.array([cv2.cvtColor(s, cv2.COLOR_RGB2GRAY).mean() for s in small]); sharp = np.array([cv2.Laplacian(cv2.cvtColor(s, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var() for s in small])
    np.savez_compressed(outp, t=(np.arange(len(frames)) + 0.5) / C.HL_FPS, feats=embed_images(frames).astype(np.float16),
                        diff=diff, flow=flow, audio=aud if aud is not None else np.zeros(0), bright=gray, sharp=sharp,
                        duration=float(r["duration_sec"]), probed=probe(hits[0])["duration"])
    print(f"{i + 1}/{len(rows)} {vid} {len(frames)} frames  [{time.time() - t0:.0f}s]", flush=True)
