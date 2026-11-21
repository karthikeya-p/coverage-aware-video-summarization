"""Zero-shot per-second importance curves.

  * `score_video()` samples frames at HL_FPS, computes CLIP image features once, and returns:
      - frame features (for the label router)
      - a combined highlightness curve  (FPVSum branch -> pack_windows)
      - an action-ness / hand-object-interaction proxy curve (EPIC branch -> coverage selector's BETA term)
  Components: CLIP semantic contrast (positive vs negative prompts), motion energy (frame difference +
  Farneback optical flow at 96 px), audio RMS, and visual novelty (1 - max cosine to a trailing window).
  No ground truth of any kind is used.
"""
import numpy as np, cv2
import config as C
from video_io import iter_frames, audio_rms, probe
from clip_utils import embed_images, embed_texts, predict_label

POS_GENERIC = ["an exciting action moment", "a fast, intense first-person shot", "a jump, trick or stunt",
               "the peak of the action", "a dramatic view during the activity"]
NEG_GENERIC = ["a person standing still and waiting", "walking slowly, nothing happening",
               "talking to the camera", "a blurry or dark frame", "a title screen or black frame",
               "adjusting equipment", "looking at the ground"]
POS_BY_LABEL = {
    "bike": ["speeding down a mountain bike trail", "jumping a bike over an obstacle"],
    "bikepolo": ["hitting the ball with the mallet", "scoring a goal in bike polo"],
    "boxing": ["throwing punches at an opponent", "landing a punch"],
    "horse": ["the horse galloping fast", "the horse jumping a fence"],
    "jumping": ["leaping off the edge in free fall", "the parachute opening in the sky"],
    "longboarding": ["carving fast downhill on a longboard", "sliding a longboard around a corner"],
    "motor": ["accelerating fast on a motorcycle", "a motorbike jump or wheelie"],
    "parkour": ["jumping across a rooftop gap", "vaulting over a wall"],
    "plane": ["the airplane taking off or landing", "a steep turn seen from the cockpit"],
    "rockclimbing": ["reaching for a hold high on the cliff", "the climber at the exposed top"],
    "scuba": ["a big fish, turtle or shark close to the camera", "swimming through a coral reef"],
    "skate": ["landing a skateboard trick", "dropping into a skate ramp"],
    "ski": ["skiing fast down a steep slope", "a ski jump in the air"],
    "surfing": ["riding inside the barrel of a wave", "standing up on the surfboard on a wave"],
    "cooking": ["hands actively cutting, stirring or pouring food", "manipulating a kitchen tool with both hands"],
}
HOI_POS = ["hands manipulating an object on a kitchen counter", "a hand holding a knife, spoon or pan",
           "opening a cupboard or fridge with a hand", "hands washing something in the sink"]
HOI_NEG = ["an empty kitchen with no hands visible", "walking through a room", "looking at a wall or floor",
           "a blurry frame", "standing still looking at the counter"]

_TXT = {}


def _txt(key, texts):
    if key not in _TXT:
        _TXT[key] = embed_texts(texts)
    return _TXT[key]


def _robust_z(x):
    x = np.asarray(x, np.float64)
    med = np.median(x); mad = np.median(np.abs(x - med)) + 1e-6
    return np.clip((x - med) / (1.4826 * mad), -3, 3)


def _smooth(x, sigma_steps):
    if sigma_steps <= 0 or len(x) < 3:
        return x
    r = int(3 * sigma_steps); k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma_steps) ** 2); k /= k.sum()
    return np.convolve(np.pad(x, r, mode="edge"), k, mode="valid")


def motion_curves(frames_small):
    """frames_small: list of HxWx3 uint8 (~96 px). Returns (frame_diff, flow_mag) per frame."""
    diff = np.zeros(len(frames_small)); flow = np.zeros(len(frames_small))
    prev = None
    for i, f in enumerate(frames_small):
        g = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
        if prev is not None:
            diff[i] = np.abs(g.astype(np.int16) - prev.astype(np.int16)).mean()
            fl = cv2.calcOpticalFlowFarneback(prev, g, None, 0.5, 2, 9, 2, 5, 1.1, 0)
            flow[i] = np.linalg.norm(fl, axis=2).mean()
        prev = g
    if len(diff) > 1:
        diff[0] = diff[1]; flow[0] = flow[1]
    return diff, flow


def novelty_curve(feats, window=20):
    nov = np.zeros(len(feats))
    for i in range(1, len(feats)):
        past = feats[max(0, i - window):i]
        nov[i] = 1.0 - float((past @ feats[i]).max())
    if len(nov) > 1:
        nov[0] = nov[1]
    return nov


def score_video(path, fps=C.HL_FPS, label=None):
    info = probe(path); D = info["duration"]
    frames, small = [], []
    for _, f in iter_frames(path, fps=fps, short_side=224):
        frames.append(f)
        small.append(cv2.resize(f, (128, 96), interpolation=cv2.INTER_AREA))
    n = len(frames)
    if n == 0:
        raise RuntimeError(f"no frames decoded from {path}")
    t = (np.arange(n) + 0.5) / fps
    feats = embed_images(frames)                                # n x 512
    if label is None:
        label, label_score = predict_label(feats)
    else:
        label_score = 1.0

    # --- semantic highlightness: mean sim to positives (generic + label-specific) minus negatives
    pos = _txt("pos_" + label, POS_BY_LABEL.get(label) or POS_GENERIC)     # sport-specific "peak moment" prompts
    neg = _txt("neg", NEG_GENERIC)
    sem = (feats @ pos.T).mean(1) - (feats @ neg.T).mean(1)
    # --- hand-object interaction proxy (used as action-ness on EPIC)
    hoi = (feats @ _txt("hoi_pos", HOI_POS).T).mean(1) - (feats @ _txt("hoi_neg", HOI_NEG).T).mean(1)
    # --- motion
    diff, flow = motion_curves(small)
    motion = 0.5 * _robust_z(diff) + 0.5 * _robust_z(flow)
    # --- audio
    aud = audio_rms(path, hop_s=1.0 / fps)
    audio = _robust_z(np.interp(t, (np.arange(len(aud)) + 0.5) / fps, aud)) if aud is not None else np.zeros(n)
    # --- novelty
    nov = _robust_z(novelty_curve(feats))

    w = C.HL_WEIGHTS
    hl = w["semantic"] * _robust_z(sem) + w["motion"] * _robust_z(diff) + w["audio"] * audio + w["novelty"] * nov
    if aud is None:  # re-normalise weights without audio
        hl = hl / (1 - w["audio"])
    sigma = C.HL_SMOOTH_SIGMA_S * fps
    highlight = _smooth(hl, sigma)
    # action-ness for EPIC: HOI semantics + moderate motion (too much motion = walking around)
    actionness = _smooth(0.6 * _robust_z(hoi) + 0.4 * motion - 0.3 * np.clip(_robust_z(flow) - 2.0, 0, None), sigma)

    return dict(t=t, feats=feats, label=label, label_score=label_score, highlight=highlight,
                actionness=actionness, components=dict(semantic=sem, motion=motion, audio=audio, novelty=nov,
                                                        hoi=hoi), duration=D, n_frames=n)
