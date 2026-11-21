"""AdaTAD (OpenTAD) EPIC-KITCHENS verb / noun detection from raw video, with low-memory chunked inference.

Reference pipeline (configs/adatad/epic/e2e_epic_videomae_l_ft_768x8_160_adapter_*.py):
  video at 30 fps -> every 2nd frame (15 fps), short side 160, centre crop 160
  window = 768 feature steps (stride 16 frames @30 fps = 204.8 s) = 6144 sampled frames = 384 chunks of 16 frames
  each chunk -> VideoMAE-L with temporal adapters -> 8 temporal tokens (mean over space) -> 3072 per window
  -> linear interpolation to 768 -> ActionFormer head; windows overlap 50 %; soft-NMS over the merged windows.

The reference feeds all 384 chunks through the backbone at once (many GB). Here attention/MLP run in small
batches of chunks; only the light adapters (a temporal conv across the whole window) see all chunks together,
which is exactly what the reference computes. Padded tails of short windows are skipped except for a margin that
covers the adapters' receptive field.
"""
import os, sys, time
import numpy as np, torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
WINDOW = 768            # feature steps per window
SCALE = 8               # sampled frames per feature step
STRIDE30 = 16           # 30-fps frames per feature step
CLIP = 16               # sampled frames per backbone chunk
TOK_T = CLIP // 2       # temporal tokens per chunk (tubelet 2)
MARGIN = 4              # padded chunks kept after the valid part (adapter receptive field = 24 tokens = 3 chunks)
FPS = 30.0
_MODELS = {}


def _cfg_path(opentad_dir, kind):
    return os.path.join(opentad_dir, "configs", "adatad", "epic", f"e2e_epic_videomae_l_ft_768x8_160_adapter_{kind}.py")


def load_model(kind, opentad_dir, ckpt_path, class_map_path, device="cuda", half=True, verbose=True):
    key = (kind, device, half)
    if key in _MODELS:
        return _MODELS[key]
    import ot_loader
    ot_loader.setup(opentad_dir)
    from mmengine.config import Config
    from opentad.models import build_detector
    from opentad.models.backbones.vit_adapter import VisionTransformerAdapter
    cwd = os.getcwd(); os.chdir(opentad_dir)
    try:
        cfg = Config.fromfile(_cfg_path(opentad_dir, kind))
    finally:
        os.chdir(cwd)
    bcfg = dict(cfg.model.backbone.backbone); bcfg.pop("type")
    pre = cfg.model.backbone.data_preprocessor
    backbone = VisionTransformerAdapter(**bcfg)
    mcfg = cfg.model.copy(); mcfg.pop("backbone")
    detector = build_detector(mcfg)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict_ema", sd.get("state_dict", sd))
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    P = "backbone.model.backbone."
    rb = backbone.load_state_dict({k[len(P):]: v for k, v in sd.items() if k.startswith(P)}, strict=False)
    rd = detector.load_state_dict({k: v for k, v in sd.items() if not k.startswith("backbone.")}, strict=False)
    other = [k for k in sd if k.startswith("backbone.") and not k.startswith(P)]
    if verbose:
        print(f"[adatad:{kind}] backbone missing={len(rb.missing_keys)} unexpected={len(rb.unexpected_keys)} | "
              f"detector missing={len(rd.missing_keys)} unexpected={len(rd.unexpected_keys)} | "
              f"other backbone tensors={len(other)} {other[:3]}", flush=True)
        for r in (rb, rd):
            if r.missing_keys or r.unexpected_keys:
                print("   e.g. missing", r.missing_keys[:4], "unexpected", r.unexpected_keys[:4], flush=True)
    backbone.eval().to(device); detector.eval().to(device)
    if half and device.startswith("cuda"):
        backbone.half()
    with open(class_map_path) as f:
        class_ids = [int(x.strip().split("_")[-1]) for x in f.read().split("\n") if x.strip()]
    m = dict(kind=kind, backbone=backbone, detector=detector, device=device, half=half and device.startswith("cuda"),
             mean=torch.tensor(pre["mean"]).view(1, 3, 1, 1, 1), std=torch.tensor(pre["std"]).view(1, 3, 1, 1, 1),
             class_ids=np.array(class_ids), post=cfg.post_processing)
    _MODELS[key] = m
    return m


def activate(m):
    """Keep only one backbone on the GPU at a time (verb and noun backbones are ~0.6 GB each in half precision)."""
    for other in _MODELS.values():
        if other is not m and other["device"] == m["device"] and other.get("on_gpu", True):
            other["backbone"].cpu(); other["on_gpu"] = False
    if not m.get("on_gpu", True):
        m["backbone"].to(m["device"]); m["on_gpu"] = True


@torch.no_grad()
def backbone_tokens(m, frames, n_chunks, batch=8):
    """frames: uint8 array [n_chunks*16, H, W, 3] (RGB). Returns [n_chunks*8, C] float32 on CPU:
    one feature per temporal token, averaged over space (the reference's 'b n c t h w -> b c t')."""
    bb, dev = m["backbone"], m["device"]; dt = torch.half if m["half"] else torch.float32
    mean, std = m["mean"].to(dev), m["std"].to(dev)
    H, W = frames.shape[1:3]; h, w = H // bb.patch_size, W // bb.patch_size
    if (h, w) != bb.grid_size:
        pe = bb.pos_embed.reshape(-1, *bb.grid_size, bb.embed_dims).permute(0, 3, 1, 2)
        pe = F.interpolate(pe.float(), size=(h, w), mode="bicubic", align_corners=False)
        pe = pe.permute(0, 2, 3, 1).flatten(1, 2).reshape(1, -1, bb.embed_dims)
    else:
        pe = bb.pos_embed
    pe = pe.to(dev, dt)
    X = torch.empty((n_chunks, TOK_T * h * w, bb.embed_dims), device=dev, dtype=dt)
    for s in range(0, n_chunks, batch):
        e = min(n_chunks, s + batch)
        x = torch.from_numpy(frames[s * CLIP:e * CLIP]).to(dev)                   # [(e-s)*16, H, W, 3]
        x = x.view(e - s, CLIP, H, W, 3).permute(0, 4, 1, 2, 3).float()          # [b, 3, 16, H, W]
        x = ((x - mean) / std).to(dt)
        X[s:e] = bb.patch_embed(x)[0] + pe
    for blk in bb.blocks:
        for s in range(0, n_chunks, batch):
            x = X[s:s + batch]
            x = x + blk.attn(blk.norm1(x))
            X[s:s + batch] = x + blk.mlp(blk.norm2(x))
        if blk.use_adapter:
            _adapter_inplace(blk.adapter, X, h, w, batch)
    X = bb.norm(X)
    feat = X.view(n_chunks, TOK_T, h * w, bb.embed_dims).float().mean(2)          # [n_chunks, 8, C]
    return feat.reshape(n_chunks * TOK_T, bb.embed_dims).cpu()


def _adapter_inplace(ad, X, h, w, batch):
    """Adapter.forward, updating X in place. Equivalent to
        y = act(down_proj(X));  a = conv(dwconv(y over the temporal axis of the whole window))
        X = up_proj(y + a) * gamma + X
    The reference builds several full-size temporaries (each the size of X); here only the 4x narrower
    down-projected tensor is held for the whole window and the up-projection is applied in small batches,
    which keeps a 205-second window within a 4 GB card."""
    n = X.shape[0]
    y = torch.cat([ad.act(ad.down_proj(X[s:s + batch])) for s in range(0, n, batch)])      # [n, N, C/4]
    B, N, C = y.shape
    a = y.reshape(1, -1, h, w, C).permute(0, 2, 3, 4, 1).flatten(0, 2)                      # [h*w, C/4, T]
    a = ad.conv(ad.dwconv(a))
    y += a.unflatten(0, (-1, h, w)).permute(0, 4, 1, 2, 3).reshape(B, N, C)
    del a
    for s in range(0, n, batch):
        X[s:s + batch] += ad.up_proj(y[s:s + batch]) * ad.gamma


def windows_for(duration):
    """(feature_start, feature_end) of every sliding window, as SlidingWindowDataset.split_video_to_windows."""
    n30 = int(duration * FPS)
    snip = len(np.arange(0, n30, STRIDE30)); stride = WINDOW // 2
    out = []
    for idx in range(max(1, snip // stride)):
        ws, we = idx * stride, idx * stride + WINDOW
        last = we > snip
        if last:
            we = snip; ws = max(0, we - WINDOW)
        out.append((ws, we))
        if last:
            break
    return out


@torch.no_grad()
def detect_video(m, frames15, duration, batch=8, verbose=False):
    """frames15: uint8 [N,160,160,3], the video sampled at 15 fps (= every 2nd frame of the 30-fps stream).
    Returns (segments [K,2] seconds, scores [K], class_ids [K]) after sliding-window merging and soft-NMS."""
    det, dev = m["detector"], m["device"]; post = m["post"]
    activate(m)
    n = len(frames15)
    segs, scores, labels = [], [], []
    for ws, we in windows_for(duration):
        idx = np.arange(ws * SCALE, min(we * SCALE, n))                           # sampled-frame indices in the window
        if len(idx) == 0:
            idx = np.array([n - 1])
        valid = len(idx) // SCALE                                                 # valid feature steps (reference rule)
        need = WINDOW * SCALE
        n_chunks = min(need // CLIP, int(np.ceil(len(idx) / CLIP)) + MARGIN)
        idx = np.pad(idx, (0, n_chunks * CLIP - len(idx)), mode="edge") if len(idx) < n_chunks * CLIP else idx[:n_chunks * CLIP]
        tok = backbone_tokens(m, frames15[idx], n_chunks, batch)                  # [n_chunks*8, C]
        full = torch.zeros((WINDOW * SCALE // 2, tok.shape[1]))                   # 3072 temporal tokens per window
        full[:len(tok)] = tok
        feats = F.interpolate(full.t().unsqueeze(0), size=WINDOW, mode="linear", align_corners=False)   # [1,C,768]
        masks = torch.zeros(1, WINDOW, dtype=torch.bool); masks[0, :valid] = True
        feats = (feats * masks.unsqueeze(1).float()).to(dev)
        props, scs = det.forward_test(feats, masks.to(dev))
        p = props[0].detach().float().cpu(); s = scs[0].detach().float().cpu()    # [K,2], [K,num_classes]
        ncls = s.shape[-1]; flat = s.flatten()
        keep = flat > getattr(post, "pre_nms_thresh", 0.001)
        prob = flat[keep]; ids = keep.nonzero(as_tuple=True)[0]
        k = min(getattr(post, "pre_nms_topk", 2000), ids.numel())
        prob, order = prob.sort(descending=True); prob = prob[:k]; ids = ids[order[:k]]
        seg = p[torch.div(ids, ncls, rounding_mode="floor")]
        seg = (seg * STRIDE30 + ws * STRIDE30) / FPS
        seg = seg.clamp(min=0.0, max=float(duration))
        segs.append(seg); scores.append(prob); labels.append(torch.fmod(ids, ncls))
        if verbose:
            print(f"   window [{ws},{we}) chunks={n_chunks} valid={valid} dets={k}", flush=True)
    segs, scores, labels = torch.cat(segs), torch.cat(scores), torch.cat(labels)
    nms = dict(post.nms)
    segs, scores, labels = _batched_nms(segs, scores, labels, nms)
    return segs.numpy().astype(np.float64), scores.numpy().astype(np.float64), m["class_ids"][labels.numpy().astype(int)]


def _batched_nms(segs, scores, labels, nms):
    """Soft-NMS + voting via ActionFormer's compiled 1-D NMS (same algorithm as OpenTAD's, Gaussian method)."""
    from libs.utils.nms import batched_nms
    return batched_nms(segs.contiguous(), scores.contiguous(), labels.contiguous(), iou_threshold=0.0,
                       min_score=nms.get("min_score", 0.001), max_seg_num=nms.get("max_seg_num", 2000),
                       use_soft_nms=nms.get("use_soft_nms", True), multiclass=nms.get("multiclass", True),
                       sigma=nms.get("sigma", 0.4), voting_thresh=nms.get("voting_thresh", 0.75))


def load_frames15(video_path):
    """The video sampled at 15 fps (every 2nd frame of the 30-fps stream), short side 160, centre crop 160x160."""
    from video_io import iter_frames
    fr = []
    for _, f in iter_frames(video_path, fps=15, short_side=160):
        x = (f.shape[1] - 160) // 2; y = (f.shape[0] - 160) // 2
        fr.append(f[y:y + 160, x:x + 160])
    return np.ascontiguousarray(np.stack(fr))


def detect_verb_noun(video_path, duration, device=None, batch=None, verbose=False):
    """Verb and noun detections for one video, in the raw-detection layout used by the pairing code:
    dict(vs, vc, vl, ns, nc, nl) = segments [K,2] (s), scores [K], EPIC class ids [K] for verbs / nouns."""
    import config as C
    device = device or (C.DEVICE if torch.cuda.is_available() else "cpu")
    batch = batch or C.ADATAD_BATCH
    frames = load_frames15(video_path)
    out = {}
    for kind, p in (("verb", "v"), ("noun", "n")):
        m = load_model(kind, C.OPENTAD_DIR, C.ADATAD_CKPT[kind], C.ADATAD_CLASSMAP[kind], device=device, verbose=verbose)
        seg, sc, lab = detect_video(m, frames, duration, batch=batch)
        out[p + "s"], out[p + "c"], out[p + "l"] = seg, sc, lab.astype(np.int64)
    return out

