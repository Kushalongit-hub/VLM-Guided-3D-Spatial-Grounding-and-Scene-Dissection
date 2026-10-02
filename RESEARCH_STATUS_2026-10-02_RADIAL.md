# Camera-ray support correction milestone — 2 October 2026

**Deadline: 27 October 2026.** The image-to-layout prototype now runs the frozen camera-ray correction in the normal CLI. A separate desk-category transfer evaluation shows a substantial child-centroid gain over exact-height contact, but the small and partly correlated samples do not yet establish publication novelty or broad generalization.

## Method

The earlier height-only weighted optimizer tied with exact-height contact on a disjoint 15-scene table sample (see [earlier status](RESEARCH_STATUS_2026-10-02.md)). A local depth-strip contact estimate was tested on development scenes and did not beat exact contact. The new method moves child and parent centers along their individual camera rays, minimizing squared radial shifts plus a soft 0.4 m child-minus-parent camera-up height-gap residual. Development settings were frozen before the two evaluations below: child radial sigma 0.4 m, parent radial sigma 0.2 m, relation sigma 0.2 m, each shift capped at 0.5 m. The 0.4 m gap is a manual prior, not inferred object size.

## Frozen evaluations

All error values use RGB-predicted metric depth, official camera intrinsics and oracle 2D boxes for the primary metric; the support relation comes from RGB-only visual annotation. This isolates placement correction from detector failure. The automatic matched-pair result uses detector boxes and the unchanged support proposal rule, with oracle boxes used only for scoring alignment. Counts refer to annotated pairs, not exhaustive scene-level relation precision.

| Sample | Visual positives / negatives / uncertain | Proposed positives | Mean child error, no / exact / radial | Radial minus exact | Improved pairs |
|---|---:|---:|---:|---:|---:|
| Table, 15 selected scenes | 12 / 2 / 1 | 10/12 | 0.717 / 0.714 / **0.635 m** | **−0.079 m** | 9/12 |
| Desk transfer, 7 selected scenes | 6 / 0 / 1 | 5/6 | 0.444 / 0.434 / **0.286 m** | **−0.147 m** | 5/6 |

The desk sample transferred the previously fixed table-development method without parameter tuning. Its 10,000-resample scene-bootstrap interval for radial minus exact is **−0.213 to −0.062 m**. Excluding the two b3dodata desk positives still gives a mean **−0.105 m** on four scenes. On the five automatically detected and proposed desk positives, the conditional child mean is **0.242 m radial** versus **0.369 m exact**. These are encouraging, not proof of a population effect; the desk positive count is six and there are no clear negative desk pairs.

The table sample has a critical clustering sensitivity. Twelve visually positive pairs fall into four coarse metadata-neighborhood groups. An equal-cluster bootstrap interval for radial minus exact is **−0.183 to +0.00006 m** and includes zero; excluding six b3dodata positives makes the mean **+0.009 m**. The table gain therefore cannot be reported as robust across sources. The samples are official SUN RGB-D training-split images outside earlier project selections, not an untouched standard test-set benchmark. A single person labeled relations from RGB and 2D boxes.

Radial correction also **increased parent centroid error**: table 0.468 versus 0.417 m exact, desk 0.432 versus 0.387 m exact. It **increased height-gap error**: table 0.155 versus 0.054 m exact, desk 0.250 versus 0.128 m exact. The method targets child and pair centroid error, not exact physical contact. A deployment that requires correct support-surface contact should not assume this correction supplies it.

## Runnable result

`scripts/run_inference.py` accepts `--radial-support-pair CHILD_ID PARENT_ID` and either `--up-camera X Y Z` or `--sunrgbd-calibration calib.txt`. It applies the frozen parameters, retains `uncorrected_center_point_3d` for each changed object, and exports shifts, residual, relation source, and up vector. `--radial-proposed-support` uses the heuristic relation only when it emits exactly one proposal. It raises an error for zero or multiple proposals so the system does not silently choose a relation. The image `sunrgbd_007816` produced two overlapping detected cup proposals; a detected-box run using `gdino_1` and `gdino_0` succeeded on CUDA and wrote [`outputs/radial_desk_demo_007816_detected_corrected.json`](outputs/radial_desk_demo_007816_detected_corrected.json). This output is a demonstration, not an independently measured accuracy result.

```powershell
.\.venv\Scripts\python.exe scripts\run_inference.py `
  --image data\sunrgbd_official\radial_desk7\sunrgbd_007816\image.jpg `
  --grounding-dino-checkpoint checkpoints\grounding-dino-tiny `
  --checkpoint checkpoints\depth_anything_v2_metric_hypersim_vits.pth `
  --fx 520.532 --fy 520.7444 --cx 277.9258 --cy 215.115 `
  --radial-support-pair gdino_1 gdino_0 `
  --sunrgbd-calibration data\sunrgbd_official\radial_desk7\sunrgbd_007816\calib.txt `
  --output outputs\radial_desk_demo_007816_detected_corrected.json --device cuda
```

## Research claim boundary and next work

The current claim is **provisional evidence that camera-ray correction can improve cup/mug child-center placement over height-only contact under the stated calibration and prior**. It is not yet a validated novel algorithm. [Diorama](https://arxiv.org/html/2411.19492) already performs semantic-aware support hierarchy and 3D scene layout optimization; a paper must distinguish this lightweight ray-restricted formulation from prior placement optimization and compare against strong alternatives. Before 27 October: independently annotate more separated scenes and object/support categories with negatives; evaluate calibration and offset sensitivity; add an uncertainty or relation validation mechanism that limits parent/contact degradation; benchmark VRAM and latency; compare against a comparable prior method. Freeze new protocols before reading reference geometry.

## Evidence

- Table: [selection](results/sunrgbd_support_radial15_selection.json), [pre-image protocol](results/sunrgbd_support_radial15_protocol.json), [RGB-only labels](results/sunrgbd_support_radial15_labels.json), [3D report](results/sunrgbd_support_radial15_3d_evaluation.json), [cluster sensitivity](results/sunrgbd_support_radial15_cluster_sensitivity.json).
- Desk: [selection](results/sunrgbd_radial_desk7_selection.json), [pre-image protocol](results/sunrgbd_radial_desk7_protocol.json), [RGB-only labels](results/sunrgbd_radial_desk7_labels.json), [3D report](results/sunrgbd_radial_desk7_3d_evaluation.json).
- [Frozen correction code](src/pipeline/radial_support.py), [evaluation](scripts/evaluate_frozen_support_3d.py), [development diagnostic](results/local_contact_development.json). The protocols contain source and checkpoint checksums. Local model and dataset files are ignored by Git.
