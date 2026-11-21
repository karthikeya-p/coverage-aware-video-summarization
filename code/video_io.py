"""ffmpeg/ffprobe based video + audio access. No codec libraries to compile."""
import json, subprocess, numpy as np


def probe(path):
    out = subprocess.check_output(["ffprobe", "-v", "error", "-print_format", "json",
                                   "-show_format", "-show_streams", path]).decode()
    info = json.loads(out)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    a = [s for s in info["streams"] if s["codec_type"] == "audio"]
    num, den = v.get("avg_frame_rate", "30/1").split("/")
    fps = float(num) / float(den) if float(den) else 30.0
    duration = float(info["format"].get("duration") or v.get("duration") or 0.0)
    rot = 0
    for sd in v.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(sd["rotation"])
    w, h = int(v["width"]), int(v["height"])
    if rot % 180 != 0:
        w, h = h, w
    return dict(duration=duration, fps=fps, width=w, height=h, has_audio=bool(a))


def _scale_expr(short_side):
    # scale so the short side == short_side, keep aspect, even dims
    return f"scale='if(gt(iw,ih),-2,{short_side})':'if(gt(iw,ih),{short_side},-2)'"


def iter_frames(path, fps, short_side=256, start=None, end=None, chunk=64):
    """Yield (frame_index, RGB uint8 HxWx3) at a constant `fps`, short side resized to `short_side`."""
    info = probe(path)
    # we need fixed output dims to parse raw bytes; compute them
    w, h = info["width"], info["height"]
    if w >= h:
        H = short_side; W = int(round(w * short_side / h / 2)) * 2
    else:
        W = short_side; H = int(round(h * short_side / w / 2)) * 2
    cmd = ["ffmpeg", "-v", "error", "-nostdin"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", path]
    if end is not None:
        cmd += ["-to", f"{(end - (start or 0)):.3f}"]
    cmd += ["-vf", f"fps={fps},scale={W}:{H}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10 ** 8)
    frame_bytes = W * H * 3
    idx = 0
    try:
        while True:
            buf = proc.stdout.read(frame_bytes * chunk)
            if not buf:
                break
            n = len(buf) // frame_bytes
            arr = np.frombuffer(buf[: n * frame_bytes], dtype=np.uint8).reshape(n, H, W, 3)
            for i in range(n):
                yield idx, arr[i]
                idx += 1
    finally:
        proc.stdout.close(); proc.wait()


def sample_frames_uniform(path, n, short_side=224):
    """n frames spread uniformly over the video (for the label router)."""
    info = probe(path)
    dur = max(info["duration"], 1e-3)
    fps = n / dur
    frames = [f for _, f in iter_frames(path, fps=fps, short_side=short_side)]
    return frames[:n] if frames else []


def audio_rms(path, hop_s=0.5, sr=16000):
    """RMS energy per hop (seconds). Returns None if the file has no audio."""
    if not probe(path)["has_audio"]:
        return None
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", path, "-vn", "-ac", "1", "-ar", str(sr),
           "-f", "s16le", "-"]
    raw = subprocess.run(cmd, stdout=subprocess.PIPE).stdout
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    hop = int(hop_s * sr)
    if len(x) < hop:
        return None
    n = len(x) // hop
    x = x[: n * hop].reshape(n, hop)
    return np.sqrt((x ** 2).mean(1) + 1e-9)
