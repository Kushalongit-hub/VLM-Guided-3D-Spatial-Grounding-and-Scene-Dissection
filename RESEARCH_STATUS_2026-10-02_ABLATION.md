# Stronger baseline audit — 2 October 2026

**Deadline: 27 October 2026.** The working image-to-3D model now offers a simpler child-only camera-ray correction, and a post-hoc ablation has changed the research direction. The paired camera-ray method previously beat a height-only exact-contact baseline, but it does **not** beat this stronger child-only ray baseline on the existing table and desk samples.

## Comparison on already scored scenes

All methods use RGB-predicted depth, official camera intrinsics, oracle 2D boxes and visually annotated positive support pairs. Both ray methods use the same manually supplied 0.4 m child-center height offset, 0.4 m child radial uncertainty, 0.2 m relation uncertainty and 0.5 m radial shift cap. The paired method also moves the parent with a 0.2 m radial uncertainty; the child-only method keeps the parent fixed. These controls were devised **after seeing the paired method's results**, so this is an exploratory ablation, not a fresh holdout test.

| Sample | Positive pairs | No correction | Exact height | Paired ray | Child-only ray, soft | Child-only ray, exact with cap |
|---|---:|---:|---:|---:|---:|---:|
| Table | 12 | 0.717 | 0.714 | 0.635 | **0.633** | **0.604** |
| Desk | 6 | 0.444 | 0.434 | 0.286 | **0.273** | **0.244** |

Values are mean child 3D centroid errors in metres. Paired minus child-only soft is **+0.003 m** on table pairs (scene-bootstrap 95% interval −0.007 to +0.013 m) and **+0.014 m** on desk pairs (+0.00002 to +0.029 m). Positive means the paired method is worse. The child-only exact variant also has a lower mean in both samples, but its capped shift does not guarantee exact contact. The original table sample contains nearby frames, so a naive scene bootstrap understates dependence. The desk sample has only six positive pairs. No method here establishes a broad improvement or novelty claim.

The paired method increased parent centroid error and height-gap error against the height-only baseline in the prior evaluation. Child-only leaves the parent estimate untouched, which is desirable when child placement is the target. It still relies on a fixed 0.4 m offset and a visually supplied or heuristic support pair; its contact residual and failure rate need separate measurement. The [ablation script](scripts/analyze_radial_ablation.py) and [table](results/sunrgbd_support_radial15_ablation.json) and [desk](results/sunrgbd_radial_desk7_ablation.json) per-scene reports contain the calculations.

## Runnable model and measured execution

`scripts/run_inference.py` now supports `--support-method child-ray-soft` with `--radial-support-pair` or the unique-proposal mode. It writes the original and corrected child center, radial shift and support residual. The original `paired-radial` method remains available for comparison. On SUN RGB-D scene `sunrgbd_007816`, Grounding DINO and the indoor metric-depth model ran on the local CUDA GPU; the child-only output is [here](outputs/child_ray_desk_demo_007816.json).

One cold local run recorded **25.12 s total**, with **24.54 s** in Grounding DINO loading/inference, **0.58 s** in metric-depth loading/inference and **0.007 s** in geometry/support. PyTorch peak allocated GPU memory was **1.86 GB** during detection and **0.39 GB** during depth; the full [profile](results/single_image_cuda_profile.json) also records reserved memory. Stage peak values are reset between stages and exclude non-PyTorch CUDA allocations and other processes. This single scene and run do not establish a worst-case VRAM budget or typical latency.

```powershell
.\.venv\Scripts\python.exe scripts\run_inference.py `
  --image data\sunrgbd_official\radial_desk7\sunrgbd_007816\image.jpg `
  --grounding-dino-checkpoint checkpoints\grounding-dino-tiny `
  --checkpoint checkpoints\depth_anything_v2_metric_hypersim_vits.pth `
  --fx 520.532 --fy 520.7444 --cx 277.9258 --cy 215.115 `
  --radial-support-pair gdino_1 gdino_0 --support-method child-ray-soft `
  --sunrgbd-calibration data\sunrgbd_official\radial_desk7\sunrgbd_007816\calib.txt `
  --profile-output results\single_image_cuda_profile.json `
  --output outputs\child_ray_desk_demo_007816.json --device cuda
```

The detector emitted two overlapping cup candidates in this image, so the supplied `gdino_1`/`gdino_0` IDs select the intended cup/desk pair after inspecting proposals. The demonstration is therefore not a fully automatic support-relation result. The frozen desk evaluation's automatic proposal matched 5/6 visually positive pairs, but it did not resolve duplicates in this specific image.

## Research decision

The paired-ray gain over height-only contact is **insufficient as a novelty claim** now that a simpler ray-constrained control performs as well or better. The immediate research question is whether child-only ray correction improves placement against strong controls on new, independent rooms and categories **without sacrificing contact plausibility or relying on a tuned offset**. Freeze a new protocol and relation labels before looking at reference 3D centers, include a no-correction baseline, exact-height contact, paired ray and child-only ray, and group uncertainty estimates by room/source. There are few untouched uniquely paired cup/mug/table scenes left in the current SUN RGB-D selection pool; obtaining a broader dataset or new annotations is necessary for a credible paper. Prior semantic support and scene optimization work such as [Diorama](https://arxiv.org/html/2411.19492) must be addressed before asserting methodological novelty.
