"""Shared open_clip model + zero-shot label router (video-level category)."""
import numpy as np, torch
from PIL import Image
import config as C

_MODEL = None


def get_clip(device=None):
    global _MODEL
    if _MODEL is None:
        import open_clip
        device = device or (C.DEVICE if torch.cuda.is_available() else "cpu")
        name, pre = C.CLIP_MODEL
        model, _, preprocess = open_clip.create_model_and_transforms(name, pretrained=pre, device=device)
        tok = open_clip.get_tokenizer(name)
        model.eval()
        _MODEL = (model, preprocess, tok, device)
    return _MODEL


@torch.no_grad()
def embed_images(frames, batch=64):
    """frames: list of HxWx3 uint8 -> L2-normalised features (N, D) float32 numpy."""
    model, preprocess, _, device = get_clip()
    feats = []
    for i in range(0, len(frames), batch):
        x = torch.stack([preprocess(Image.fromarray(f)) for f in frames[i:i + batch]]).to(device)
        with torch.autocast(device_type="cuda", enabled=device.startswith("cuda")):
            f = model.encode_image(x)
        feats.append(torch.nn.functional.normalize(f.float(), dim=-1).cpu())
    return torch.cat(feats).numpy() if feats else np.zeros((0, 512), np.float32)


@torch.no_grad()
def embed_texts(texts):
    model, _, tok, device = get_clip()
    t = model.encode_text(tok(texts).to(device))
    return torch.nn.functional.normalize(t.float(), dim=-1).cpu().numpy()


# ------------------------------------------------------------------ label router
LABEL_DESC = {
    "cooking": ["someone cooking in a kitchen", "hands preparing food on a kitchen counter",
                "washing dishes in a kitchen sink", "a stove, pans and kitchen cupboards"],
    "bike": ["mountain biking on a trail", "riding a bicycle, handlebars visible", "downhill mountain bike racing seen over the handlebars", "BMX racing on a dirt or asphalt pump track"],
    "bikepolo": ["playing bike polo with a mallet on a court", "cyclists hitting a ball with mallets"],
    "boxing": ["boxing sparring with gloves", "punching a heavy bag in a boxing gym"],
    "horse": ["horseback riding, the horse's head and mane in front", "riding a horse through a field"],
    "jumping": ["skydiving or base jumping, free fall from the sky", "jumping from a height with a parachute", "wingsuit flying past cliffs and mountains", "BASE jumping off a mountain, looking down at the valley"],
    "longboarding": ["longboarding downhill on a road", "riding a longboard skateboard on asphalt"],
    "motor": ["riding a motorcycle on a road, speedometer and mirrors visible", "motocross dirt bike riding with an engine"],
    "parkour": ["parkour, running and jumping over urban obstacles", "free running on rooftops"],
    "plane": ["a camera mounted on a radio-controlled model airplane flying low over fields", "aerial view from a small model aircraft with part of its wing or tail visible", "flying a small airplane, cockpit and instruments"],
    "rockclimbing": ["rock climbing on a cliff with a rope", "bouldering, hands gripping rock"],
    "scuba": ["scuba diving underwater with fish and coral", "underwater diving with bubbles"],
    "skate": ["skateboarding tricks at a skate park", "riding a skateboard, doing a kickflip"],
    "ski": ["skiing down a snowy mountain slope", "snow skiing with poles and ski tips visible"],
    "surfing": ["surfing on an ocean wave", "riding a surfboard in the sea"],
}
TEMPLATES = ["a first-person GoPro video of {}.", "a photo of {}.", "egocentric footage of {}."]

_TEXT_CACHE = None


def _label_text_feats():
    global _TEXT_CACHE
    if _TEXT_CACHE is None:
        names, feats = [], []
        for name, descs in LABEL_DESC.items():
            emb = embed_texts([t.format(d) for d in descs for t in TEMPLATES]).mean(0)
            feats.append(emb / np.linalg.norm(emb)); names.append(name)
        _TEXT_CACHE = (names, np.stack(feats))
    return _TEXT_CACHE


def predict_label(frame_feats, temperature=100.0):
    """frame_feats: (N, D) CLIP image feats. Returns (label_name, prob)."""
    names, tf = _label_text_feats()
    logits = temperature * frame_feats @ tf.T                # N x L
    logits = logits - logits.max(1, keepdims=True)
    probs = np.exp(logits); probs /= probs.sum(1, keepdims=True)
    video_prob = probs.mean(0)                               # average of per-frame posteriors
    i = int(video_prob.argmax())
    p = float(video_prob[i])
    # never emit 'other': every scored FPVSum video belongs to one of the 14 classes, so the
    # argmax is always the better bet for label accuracy; the score carries our confidence.
    return names[i], round(max(p, 1e-4), 4)
