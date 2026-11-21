# weights/

Small files are in git; the three checkpoints are downloaded by `bash weights/download_weights.sh`
(checked against `SHA256SUMS`).

| File | In git | Source |
|---|---|---|
| `adatad_epic_verb.pth`, `adatad_epic_noun.pth` (1.4 GB each) | no | AdaTAD EPIC-KITCHENS-100 checkpoints, OpenTAD model zoo (github.com/sming256/OpenTAD) |
| `open_clip_vit_b32_laion2b_s34b_b79k.safetensors` (0.6 GB) | no | OpenCLIP ViT-B/32 `laion2b_s34b_b79k` (huggingface.co/laion) |
| `adatad_category_idx_{verb,noun}.txt` | yes | OpenTAD class index files (EPIC class ids) |
| `verb_labels.json`, `noun_labels.json` | yes | EPIC-100 class id → class key |
| `verb_noun_cooc.json` | yes | Verb–noun pair counts from the EPIC-100 **train** split (`dev/build_priors.py`) |
| `supported_classes.json`, `manifest.csv` | yes | Organisers' student kit |
