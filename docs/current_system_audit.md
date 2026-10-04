# Current system audit — 2026-10-04

## Repository state

- Git HEAD: `509911177213b88da5bf8e6227ceff085df5527e`
- Branch: `main`
- Modified tracked files: `README.md`, `RESEARCH_PLAN.md`, `requirements.txt`, `results/README.md`, `scripts/make_laptop_table_rgb_sheet.py`, `scripts/run_inference.py`
- Untracked research work: present and preserved; not committed, not stashed, not reset.

## Environment

| Item | Value |
|------|-------|
| Python (system) | 3.14.7 |
| Python (project) | `.venv` |
| PyTorch (system) | 2.12.1+cpu |
| PyTorch (`.venv`) | `2.14.0+cu130` |
| Torchvision (`.venv`) | `0.29.0+cu130` |
| Transformers | 5.18.0 |
| NumPy | 2.5.3 |
| OpenCV | 5.0.0.93 |
| SciPy | 1.18.1 |
| Pillow | 12.3.0 |
| CUDA available (`.venv`) | **True** |
| CUDA version | 13.0 |
| GPU hardware | NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB |
| Driver / CUDA UMD | 616.56 / 13.4 |

## Checkpoints

| Checkpoint | Path | SHA-256 prefix |
|------------|------|----------------|
| Depth Anything V2 Metric Hypersim Small | `checkpoints/depth_anything_v2_metric_hypersim_vits.pth` | `b782898d8a3e8be1f` |
| YOLO11n | `checkpoints/yolo11n.pt` | `0ebbc80d4a7680d1` |
| YOLO11n-seg | `checkpoints/yolo11n-seg.pt` | `55ed65c56c91713d` |
| Grounding DINO Tiny | `checkpoints/grounding-dino-tiny/` | directory |
| SAM ViT-B | `checkpoints/sam-vit-base/` | directory |
| SmolVLM2-2.2B-Instruct | `checkpoints/smolvlm2-2.2b-instruct/` | directory |

## Dataset paths

- `data/sunrgbd_official/` — official SUN RGB-D scenes with annotations and calibration
- `data/sunrgbd_sample/` — 10 community-repackaged test images with depth PNGs

## Test status

- **36/36 tests pass** on current env
- Missing deps discovered and installed during Phase 0: `numpy`, `opencv-python`, `scipy`, `pillow`, `transformers`, `accelerate`, `safetensors`, `huggingface-hub`

## CLI verification

- `python scripts/run_inference.py --help` — OK
- `python scripts/run_baseline.py --help` — OK
- `python scripts/run_benchmark.py --help` — OK

## Reproduction results

### Positive scene: `sunrgbd_007816`

- Input: `data/sunrgbd_official/radial_desk7/sunrgbd_007816/`
- Command used: Grounding DINO Tiny + SAM ViT-B + Depth Anything V2 Metric Hypersim Small + `--radial-proposed-support --support-method child-ray-soft --sunrgbd-calibration calib.txt --device cpu`
- Output: `audit/phase0/007816_layout.json`
- Detected: desk `gdino_0` (confidence 0.456), cup `gdino_1` (confidence 0.402); duplicate cup `gdino_2` suppressed by SAM mask IoU ≥ 0.7
- Proposal: cup on desk, depth gap 0.023 m
- Correction applied: `child_ray_support_correction` with `radial_shift_m = -0.23745 m`, `residual_m = -0.14335 m`, `offset_m = 0.4 m`
- Profile: total wall 28.68 s; detection 27.48 s, depth 1.09 s, geometry 0.014 s
- Comparison to stored CUDA reference (`outputs/child_ray_desk_demo_007816.json`): stored run used box regions + supplied pair on CUDA, `radial_shift_m = -0.27143 m`, residual `-0.16367 m`, total 25.12 s. Differences are attributable to SAM masks vs boxes, heuristic proposal vs supplied pair, and CPU vs CUDA. Both runs successfully applied child-ray correction.

### Abstaining scene: `sunrgbd_004139`

- Input: `data/sunrgbd_official/bottle_table6/sunrgbd_004139/`
- Command: same pipeline as above
- Expected behavior: abstain because multiple proposals survive the mask-contact gate
- Observed: `ValueError: Radial correction requires exactly one proposal; found 2.`
- Two proposals passed gate: `gdino_1`/`gdino_0` (contact fraction 0.487) and `gdino_2`/`gdino_0` (contact fraction 1.0)
- CLI correctly abstained; no `layout.json` written
- Matches stored frozen evaluation behavior for this scene.

## Blockers

1. **CUDA inference blocked by torch/torchvision incompatibility**
   - Error: `RuntimeError: operator torchvision::nms does not exist` when importing torchvision with nightly CUDA torch
   - Affects: all model inference stages that require torchvision (depth_anything, SAM, Grounding DINO)
   - Workaround: CPU-only inference works but is ~5× slower for detection stage
   - Required for: milestones 2–5 (model integration, VRAM measurement, CUDA profiling)

2. **`ultralytics` package not installed**
   - Affects: YOLO11n detection/segmentation paths
   - Workaround: Grounding DINO + SAM path does not require ultralytics
   - Required for: detector coverage benchmarks

## Reproducible baseline command

```powershell
.\.venv\Scripts\python.exe scripts\run_inference.py `
  --image data\sunrgbd_official\radial_desk7\sunrgbd_007816\image.jpg `
  --grounding-dino-checkpoint checkpoints\grounding-dino-tiny `
  --sam-checkpoint checkpoints\sam-vit-base `
  --checkpoint checkpoints\depth_anything_v2_metric_hypersim_vits.pth `
  --fx 520.532 --fy 520.7444 --cx 277.9258 --cy 215.115 `
  --radial-proposed-support --support-method child-ray-soft `
  --sunrgbd-calibration data\sunrgbd_official\radial_desk7\sunrgbd_007816\calib.txt `
  --masks-output audit\phase0\007816_masks.npz `
  --depth-output audit\phase0\007816_depth.npy `
  --boxes-preview audit\phase0\007816_boxes.png `
  --output audit\phase0\007816_layout.json `
  --profile-output audit\phase0\007816_profile.json `
  --device cpu
```
