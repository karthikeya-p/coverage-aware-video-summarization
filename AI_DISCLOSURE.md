# AI disclosure

This file lists every AI tool used to build this submission and what it was used for.

## AI assistants used during development

| Tool | What it was used for |
|---|---|
| **Claude Opus 5.5 (Anthropic)**, AI assistant | Discussing the approach and the scoring rules; writing and debugging much of the code and helper scripts (the low-memory AdaTAD inference, the coverage selector, the evaluation, packaging and setup scripts); analysing experiment results; searching related papers; drafting the README, the slides and this file; building the demo video (title cards, overlays, editing). |
| **Microsoft Edge neural text-to-speech** (voice `en-IN-PrabhatNeural`) | The narration in the demo video. |

## AI models inside the system

These are pretrained models that the pipeline itself runs. They are part of the method and are credited in the README.

| Model | Role |
|---|---|
| AdaTAD (VideoMAE-L backbone), OpenTAD authors | Finds verb and noun actions in cooking videos |
| OpenCLIP ViT-B/32 | Decides cooking vs. sport and scores highlight-worthiness in sports videos |

## AI models used only in experiments (not in `run.sh`)

| Model | Role |
|---|---|
| Qwen2.5-VL-7B-Instruct | Two highlight-scoring experiments on Kaggle. Both were worse than our method, so it is not used (see "Things we tried that didn't help" in the README). |

## Human work and checking

- Choosing the track and the goal, and deciding what to try, what to keep and what to drop.
- Setting up the environment and running everything on our own hardware (a 4 GB laptop GPU) and on Kaggle:
  downloading the datasets, running the detectors, the experiments and the full output generation.
- Reviewing the outputs, the README, the slides and the video before submission.
- Every number in the README and slides comes from running the code (`dev/eval_local.py`,
  `dev/fpv_fusion_eval.py`, the logs of the full runs); none were written by hand or by an AI tool.
- The repository build runs a reproducibility check: a fresh run from the repository, offline, reproduces the
  submitted outputs.
