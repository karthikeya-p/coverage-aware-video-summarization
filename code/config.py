"""Central configuration. Paths are relative to the submission root so run.sh works from anywhere."""
import os

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CODE_DIR)                       # submission/
WEIGHTS = os.path.join(ROOT, "weights")
# bundled inside code/ in the submission package; a sibling folder in a development checkout
THIRD_PARTY = os.path.join(CODE_DIR, "third_party") if os.path.isdir(os.path.join(CODE_DIR, "third_party")) \
    else os.path.join(ROOT, "third_party")
ACTIONFORMER_DIR = os.path.join(THIRD_PARTY, "actionformer_release")
SLOWFAST_DIR = os.path.join(THIRD_PARTY, "epic-kitchens-slowfast")

TEAM_ID = os.environ.get("TEAM_ID", "K99")   # written into every output JSON
DEVICE = os.environ.get("DEVICE", "cuda")

# ---------------- vocabulary ----------------
SUPPORTED_CLASSES = os.path.join(WEIGHTS, "supported_classes.json")
FPV_LABELS = ["bike", "bikepolo", "boxing", "horse", "jumping", "longboarding", "motor", "parkour",
              "plane", "rockclimbing", "scuba", "skate", "ski", "surfing"]

# ---------------- highlight budget ----------------
BUDGET_FRAC = 0.15
MIN_SEG = 4.0           # segments < 4 s are ignored by the scorer
MAX_SEG = 12.0          # soft cap on a single highlight (coverage prefers many short windows)

# ---------------- CLIP router / highlightness ----------------
# open_clip (name, pretrained). The bundled checkpoint is used when present, so no network access is needed.
_CLIP_LOCAL = os.path.join(WEIGHTS, "open_clip_vit_b32_laion2b_s34b_b79k.safetensors")
CLIP_MODEL = ("ViT-B-32", _CLIP_LOCAL if os.path.exists(_CLIP_LOCAL) else "laion2b_s34b_b79k")
ROUTER_FRAMES = 32
HL_FPS = 2.0            # sampling rate for FPVSum highlightness scoring
# weights and smoothing: coarse grid, checked by split-half on FPVSum (see README "Tuning disclosure")
HL_WEIGHTS = dict(semantic=0.25, motion=0.25, audio=0.25, novelty=0.25)   # equal weights, not tuned
HL_SMOOTH_SIGMA_S = 1.0

# ---------------- EPIC branch ----------------
SLOWFAST_CFG = os.path.join(SLOWFAST_DIR, "configs", "EPIC-KITCHENS", "SLOWFAST_8x8_R50.yaml")
SLOWFAST_CKPT = os.path.join(WEIGHTS, "SlowFast.pyth")
FEAT_FPS = 30
FEAT_NUM_FRAMES = 32
FEAT_STRIDE = 16
FEAT_SAMPLING_RATE = int(os.environ.get("FEAT_SAMPLING_RATE", "1"))  # verify with check_features.py (1 or 2)
FEAT_CROP = 256          # short side of decoded frames
FEAT_SPATIAL = os.environ.get("FEAT_SPATIAL", "full")   # center256 | center224 | full  (chosen by parity_sweep.py)
FEAT_SLOW = os.environ.get("FEAT_SLOW", "linspace")            # stride | linspace
FEAT_BGR = int(os.environ.get("FEAT_BGR", "1"))              # channel order the checkpoint was trained with
FEAT_OFFSET = int(os.environ.get("FEAT_OFFSET", "0"))        # temporal alignment of clip 0, in frames
FEAT_AMP = int(os.environ.get("FEAT_AMP", "1"))              # fp16 autocast
FEAT_BATCH = int(os.environ.get("FEAT_BATCH", "4"))   # clips per forward pass; 2–4 for a 4 GB GPU, 16 for a T4

AF_VERB_CFG = os.path.join(ACTIONFORMER_DIR, "configs", "epic_slowfast_verb.yaml")
AF_NOUN_CFG = os.path.join(ACTIONFORMER_DIR, "configs", "epic_slowfast_noun.yaml")


def _find_ckpt(sub):
    """Newest-epoch *.pth.tar anywhere under weights/<sub>/ (archive layouts vary)."""
    import glob
    hits = sorted(glob.glob(os.path.join(WEIGHTS, sub, "**", "*.pth.tar"), recursive=True))
    return hits[-1] if hits else os.path.join(WEIGHTS, sub, "MISSING.pth.tar")


AF_VERB_CKPT = _find_ckpt("af_verb")
AF_NOUN_CKPT = _find_ckpt("af_noun")
VERB_LABELS = os.path.join(WEIGHTS, "verb_labels.json")   # {label_id: class_key}
NOUN_LABELS = os.path.join(WEIGHTS, "noun_labels.json")
COOC_PRIOR = os.path.join(WEIGHTS, "verb_noun_cooc.json")  # built from EPIC *train* split

# ---------------- AdaTAD detector (OpenTAD; VideoMAE-L + adapters, end to end from raw video) ----------------
def _first(*paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return paths[0]


_ADATAD_DEV = os.path.expanduser("~/xr_hackathon/adatad")          # development checkout (not part of the package)
OPENTAD_DIR = _first(os.path.join(THIRD_PARTY, "OpenTAD"), os.path.join(_ADATAD_DEV, "OpenTAD"))
ADATAD_CKPT = {k: _first(os.path.join(WEIGHTS, f"adatad_epic_{k}.pth"), os.path.join(_ADATAD_DEV, "ckpt", f"adatad_epic_{k}.pth"))
               for k in ("verb", "noun")}
ADATAD_CLASSMAP = {k: _first(os.path.join(WEIGHTS, f"adatad_category_idx_{k}.txt"),
                             os.path.join(_ADATAD_DEV, "anno", "annotations", f"category_idx_{k}.txt")) for k in ("verb", "noun")}
ADATAD_BATCH = int(os.environ.get("ADATAD_BATCH", "8"))            # 16-frame chunks per forward pass
_ADATAD_OK = os.path.isdir(OPENTAD_DIR) and all(os.path.exists(p) for p in list(ADATAD_CKPT.values()) + list(ADATAD_CLASSMAP.values()))
# "adatad" (default when its weights are present) or "actionformer" (SlowFast features + ActionFormer)
DETECTOR = os.environ.get("DETECTOR", "adatad" if _ADATAD_OK else "actionformer")

PAIR_ALPHA = 0.5        # p(action) = p(verb)^a * p(noun)^(1-a)
PAIR_IOU = 0.3          # min tIoU between a verb and a noun segment to pair them
PAIR_TOPK = 2000        # verb / noun detections considered per video
PAIR_PER_VERB = 8       # noun candidates kept per verb segment
PAIR_CLS_TOPK = 3       # classes taken from the SlowFast clip classifier per detected segment (0 = off)
PAIR_BLEND = 0.3        # class-balancing exponent when cutting to MAX_ACTIONS
MAX_ACTIONS = 2000
COOC_UNSEEN_PENALTY = 0.35

# coverage selector
SEL_MAX_SEG = 4.0      # coverage windows: many minimum-length segments beat fewer long ones (tuned on tune_ids.txt)
SEL_MIN_SCORE = 0.15    # ignore action instances below this score (ActionFormer detections)
SEL_MIN_SCORE_ADATAD = 0.25   # same, for AdaTAD detections (two-fold check over the 138 validation videos)
SEL_PAD = 0.0            # seconds added each side of an action when building a candidate window
SEL_BETA = 0.0          # weight of actionness (per second) vs. new-pair coverage
SEL_REPEAT_WEIGHT = 0.0   # value of covering a pair already covered (diminishing returns)
