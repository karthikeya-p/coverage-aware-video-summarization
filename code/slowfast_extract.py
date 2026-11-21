"""SlowFast (EPIC-KITCHENS-100 pretrained, SLOWFAST_8x8_R50) clip features that match the ActionFormer
EPIC checkpoints: 32-frame windows at 30 fps, stride 16 frames, 2304-d (2048 slow + 256 fast, global-avg-pooled
features that feed the classification head).

Also returns the head's verb/noun softmax per window (free extra signal).

Usage (CLI):  python slowfast_extract.py video.mp4 out.npz [--sampling-rate 1]
Verify against the public ActionFormer features with check_features.py before trusting the pipeline.
"""
import argparse, os, sys, numpy as np, torch
import config as C
from video_io import iter_frames, probe

MEAN = torch.tensor([0.45, 0.45, 0.45]).view(1, 3, 1, 1, 1)
STD = torch.tensor([0.225, 0.225, 0.225]).view(1, 3, 1, 1, 1)
_MODEL = None


def load_model(device):
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    sys.path.insert(0, os.path.join(C.SLOWFAST_DIR))          # repo root contains slowfast/
    from slowfast.config.defaults import get_cfg
    from slowfast.models.build import MODEL_REGISTRY
    import slowfast.models.video_model_builder  # noqa: registers SlowFast
    import slowfast.utils.checkpoint as cu
    cfg = get_cfg()
    cfg.merge_from_file(C.SLOWFAST_CFG)
    cfg.NUM_GPUS = 1 if device.startswith("cuda") else 0
    cfg.TRAIN.ENABLE = False
    cfg.TEST.ENABLE = True
    cfg.DATA_LOADER.NUM_WORKERS = 0
    # build without the repo's build_model(): that helper unconditionally calls .cuda()
    model = MODEL_REGISTRY.get(cfg.MODEL.MODEL_NAME)(cfg)
    # load directly (torch>=2.6 defaults to weights_only=True, which rejects this older checkpoint)
    ckpt = torch.load(C.SLOWFAST_CKPT, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    res = model.load_state_dict(sd, strict=False)
    if res.missing_keys or res.unexpected_keys:
        print(f"[slowfast] missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)} "
              f"e.g. {res.missing_keys[:3]} {res.unexpected_keys[:3]}", flush=True)
    model.to(device).eval()
    # capture the pooled pathway features entering the head
    cache = {}

    def hook(module, inputs):
        pooled = []
        for x in inputs[0]:                                    # [B,2048,T,H,W], [B,256,T,H,W]
            pooled.append(x.mean(dim=(2, 3, 4)))
        cache["feat"] = torch.cat(pooled, 1)                   # B x 2304
    model.head.register_forward_pre_hook(hook)
    _MODEL = (model, cache, cfg)
    return _MODEL


def _center_crop(frames, size):
    # frames: N x H x W x 3 uint8 (short side already == size or larger)
    H, W = frames.shape[1:3]
    y = (H - size) // 2; x = (W - size) // 2
    return frames[:, y:y + size, x:x + size]


def _spatial(frames, mode):
    """'full' = whole frame at short side 256 (fully convolutional); 'center256' / 'center224' = centre crop."""
    if mode == "full":
        return frames
    return _center_crop(frames, int(mode.replace("center", "")))


@torch.no_grad()
def extract(video_path, device=None, sampling_rate=None, batch=None, progress=False,
            spatial=None, slow_mode=None, amp=None, bgr=None, offset=None):
    device = device or (C.DEVICE if torch.cuda.is_available() else "cpu")
    sr = sampling_rate or C.FEAT_SAMPLING_RATE
    batch = batch or C.FEAT_BATCH
    spatial = spatial or C.FEAT_SPATIAL
    slow_mode = slow_mode or C.FEAT_SLOW
    amp = C.FEAT_AMP if amp is None else amp
    bgr = C.FEAT_BGR if bgr is None else bgr
    offset = C.FEAT_OFFSET if offset is None else offset   # >0: repeat the first frame this many times; <0: skip frames
    model, cache, cfg = load_model(device)
    alpha = cfg.SLOWFAST.ALPHA
    T = C.FEAT_NUM_FRAMES           # fast-pathway frames per clip (32)
    span = T * sr                   # frames covered by one clip
    stride = C.FEAT_STRIDE

    feats, verb_p, noun_p = [], [], []
    buf = []                        # rolling frame buffer (list of HxWx3)
    pending = []                    # clips waiting to be batched: list of (T,H,W,3) uint8 arrays
    n_frames = 0

    def flush():
        if not pending:
            return
        x = torch.from_numpy(np.stack(pending)).to(device)           # B,T,H,W,3
        x = x.permute(0, 4, 1, 2, 3).float() / 255.0                 # B,3,T,H,W
        if bgr:   # the EPIC SlowFast loader reads frames with cv2.imread and never converts: the model expects BGR
            x = x.flip(1)
        x = (x - MEAN.to(device)) / STD.to(device)
        fast = x
        if slow_mode == "linspace":       # PySlowFast pack_pathway_output: linspace over the clip, not a fixed stride
            idx = torch.linspace(0, x.shape[2] - 1, x.shape[2] // alpha).long().to(device)
            slow = x.index_select(2, idx)
        else:
            slow = x[:, :, ::alpha]
        with torch.autocast(device_type="cuda", enabled=bool(amp) and device.startswith("cuda")):
            out = model([slow, fast])
        feats.append(cache["feat"].float().cpu().numpy())
        if isinstance(out, (list, tuple)) and len(out) == 2:
            verb_p.append(out[0].float().cpu().numpy()); noun_p.append(out[1].float().cpu().numpy())
        pending.clear()

    n_clips = 0
    # clip i covers frames [i*stride, i*stride + span); the buffer always holds the frames of the next clip
    for idx, frame in iter_frames(video_path, fps=C.FEAT_FPS, short_side=C.FEAT_CROP):
        if offset < 0 and idx < -offset:
            continue
        if idx == 0 and offset > 0:
            buf.extend([frame] * offset); n_frames += offset
        buf.append(frame); n_frames += 1
        if len(buf) == span:
            clip = np.stack(buf[::sr])                                 # T,H,W,3
            pending.append(_spatial(clip, spatial)); n_clips += 1
            del buf[:stride]
            if len(pending) == batch:
                flush()
        if progress and n_frames % 3000 == 0:
            print(f"  {n_frames / C.FEAT_FPS:.0f}s decoded, {n_clips} clips", flush=True)

    # tail: if frames remain beyond the last full clip (or the video is shorter than one clip),
    # pad by repeating the final frame so every second of video is covered by some feature
    if n_frames > 0 and (n_clips == 0 or len(buf) > span - stride):
        tail = buf + [buf[-1]] * (span - len(buf))
        pending.append(_spatial(np.stack(tail[::sr]), spatial)); n_clips += 1
    flush()

    F = np.concatenate(feats, 0) if feats else np.zeros((0, 2304), np.float32)
    out = dict(feats=F.astype(np.float32), fps=np.float32(C.FEAT_FPS), stride=np.int32(stride),
               num_frames=np.int32(span), n_decoded=np.int32(n_frames))
    if verb_p:
        out["verb_prob"] = np.concatenate(verb_p, 0).astype(np.float16)
        out["noun_prob"] = np.concatenate(noun_p, 0).astype(np.float16)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video"); ap.add_argument("out")
    ap.add_argument("--sampling-rate", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    a = ap.parse_args()
    res = extract(a.video, sampling_rate=a.sampling_rate, batch=a.batch, progress=True)
    np.savez_compressed(a.out, **res)
    print(a.out, res["feats"].shape, "clips; video", probe(a.video)["duration"], "s")
