# dev/ — development and evaluation scripts (not used by `run.sh`)

These scripts were used to measure and tune the system. Several of them read public ground truth
(EPIC-KITCHENS-100 validation annotations, FPVSum importance scores) **for scoring only**; nothing in `code/`
reads ground truth, and `run.sh` never imports anything from this folder.

| Script | Purpose |
|---|---|
| `eval_local.py` | Our re-implementation of the four Track 04 metrics |
| `adatad_dev.py` | AdaTAD detections for many videos (raw, cached) + speed / memory report |
| `dev_epic_public.py` | Pairing + selection from cached raw detections, for fast tuning |
| `sweep_selector.py` | Diagnostics (oracle coverage) and selector grid; writes `tune_ids.txt` |
| `build_priors.py` | Builds the class-name maps and the verb–noun prior from the EPIC-100 **train** split |
| `extract_all.py`, `check_features.py`, `parity_sweep.py` | SlowFast feature extraction and parity check (older detector) |
| `dump_fpv.py`, `dump_videos.py`, `label_test.py` | Signal dumps and label-router tests for FPVSum / TVSum |
| `fpv_fusion_eval.py` | Highlight F1 on FPVSum from cached signals, optionally fused with VLM scores; reports two fixed halves |
| `kaggle/vlm_highlight_scores.ipynb` | Qwen2.5-VL-7B per-window highlight scores (Kaggle, 2× T4); negative result |
| `kaggle/vlm_context_scores.ipynb` | Context-aware variant: captions → video summary → per-window scores + ranking; negative result |
| `tune_ids.txt` | The fixed 30-video EPIC tuning subset |
