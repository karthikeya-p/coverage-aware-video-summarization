"""DEV ONLY — compare CLIP backbones for the video-label router. Uses the category in the FPVSum file name as the
reference (evaluation only; the pipeline never reads file names for labels).

  python dev/label_test.py --fpv ../data/FPVSum --epic ../data/EPIC-KITCHENS
"""
import argparse, csv, glob, os, sys, time
import numpy as np, torch
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "code"))  # inference code
import config as C
from video_io import sample_frames_uniform
from clip_utils import LABEL_DESC, TEMPLATES

ap = argparse.ArgumentParser()
ap.add_argument("--fpv", required=True); ap.add_argument("--epic", default=None)
ap.add_argument("--models", nargs="+", default=["ViT-B-32:laion2b_s34b_b79k", "ViT-B-16:laion2b_s34b_b88k",
                                                 "ViT-L-14:laion2b_s32b_b82k", "ViT-L-14:openai"])
ap.add_argument("--frames", type=int, default=32)
a = ap.parse_args()
import open_clip
dev = "cuda" if torch.cuda.is_available() else "cpu"
rows = [r for r in csv.DictReader(open(os.path.join(C.WEIGHTS, "manifest.csv"), newline=""))]
vids = []
for r in rows:
    if r["dataset"] == "fpvsum":
        h = glob.glob(os.path.join(a.fpv, "**", r["video_id"] + ".mp4"), recursive=True)
        if h: vids.append((r["video_id"], r["video_id"].split("_")[0].lower(), h[0]))
if a.epic:
    ep = sorted((r for r in rows if r["dataset"].startswith("epic")), key=lambda r: float(r["duration_sec"]))[20:140:8]
    for r in ep:
        h = glob.glob(os.path.join(a.epic, "**", r["video_id"] + ".MP4"), recursive=True)
        if h: vids.append((r["video_id"], "cooking", h[0]))
print(len(vids), "videos; decoding", a.frames, "frames each ...", flush=True)
t0 = time.time(); frames = {v: sample_frames_uniform(p, a.frames) for v, _, p in vids}; print(f"decoded in {time.time() - t0:.0f}s", flush=True)
names = list(LABEL_DESC)
for spec in a.models:
    name, pre = spec.split(":")
    try:
        model, _, prep = open_clip.create_model_and_transforms(name, pretrained=pre, device=dev); tok = open_clip.get_tokenizer(name); model.eval()
    except Exception as e:
        print(spec, "could not load:", str(e)[:120]); continue
    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=dev == "cuda"):
        T = []
        for k in names:
            e = torch.nn.functional.normalize(model.encode_text(tok([t.format(d) for d in LABEL_DESC[k] for t in TEMPLATES]).to(dev)).float(), dim=-1).mean(0)
            T.append(e / e.norm())
        T = torch.stack(T); ok = 0; wrong = []; t0 = time.time()
        for v, true, _ in vids:
            x = torch.stack([prep(Image.fromarray(f)) for f in frames[v]]).to(dev)
            f = torch.nn.functional.normalize(model.encode_image(x).float(), dim=-1)
            p = (100 * f @ T.T).softmax(1).mean(0); pred = names[int(p.argmax())]
            ok += pred == true
            if pred != true: wrong.append(f"{v}->{pred}({p.max():.2f})")
    nf = sum(1 for _, t, _ in vids if t != "cooking")
    fo = sum(1 for (v, t, _) in vids if t != "cooking" and not any(w.startswith(v + "->") for w in wrong))
    print(f"{spec}: FPVSum {fo}/{nf} = {fo / max(nf, 1):.3f}; all {ok}/{len(vids)}; {time.time() - t0:.0f}s; wrong: {wrong}", flush=True)
    del model; torch.cuda.empty_cache()
