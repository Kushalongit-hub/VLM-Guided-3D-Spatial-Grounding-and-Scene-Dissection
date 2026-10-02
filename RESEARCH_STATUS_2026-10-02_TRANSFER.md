# Project status after frozen category transfer — 2 October 2026

**Deadline: 27 October 2026.** The prototype is runnable from an RGB image to an object-level 3D JSON layout on the local CUDA GPU. The research claim remains unproven. A fresh, frozen lamp/table transfer test shows that the current fixed-offset support correction does not improve average lamp placement.

## What works now

- `scripts/run_inference.py` runs Grounding DINO Tiny or YOLO boxes, Depth Anything V2 Metric Hypersim Small, calibrated camera back-projection, optional cup/mug support proposals, and either paired-ray or child-only soft camera-ray correction. The image-to-layout run on SUN RGB-D scene 007816 succeeded on CUDA and produced [child-only output](outputs/child_ray_desk_demo_007816.json).
- The [single cold-run profile](results/single_image_cuda_profile.json) recorded 25.12 s total; PyTorch peak allocated memory was 1.86 GB during detection and 0.39 GB during depth. These are one-scene measurements and exclude non-PyTorch allocations. The detector emitted duplicate overlapping cups in that example, requiring an explicit pair choice.
- Calibrated geometry, detection, depth, correction, benchmark, and evaluation scripts have automated tests. Neither a VLM stage nor a visualization UI is implemented. The JSON visible extents are observed-point spreads, not complete hidden object sizes.

## Research evidence

The earlier disjoint cup/mug table and desk evaluations showed that camera-ray corrections can reduce child-centroid error relative to height-only exact contact. A [post-hoc stronger-baseline audit](RESEARCH_STATUS_2026-10-02_ABLATION.md) found the simpler **child-only** ray correction matched or beat paired-ray correction: table mean 0.633 versus 0.635 m over 12 positive pairs; desk mean 0.273 versus 0.286 m over 6 positive pairs. Those samples were already viewed before devising child-only correction, so they are development evidence.

For a new test, eight uniquely paired lamp/table candidates were selected from the **official SUN RGB-D test split**, each at least 30 metadata indices from prior project selections. The [selection](results/sunrgbd_lamp_table8_selection.json), [protocol](results/sunrgbd_lamp_table8_protocol.json), code/checkpoint hashes, fixed cup-derived 0.4 m center-height offset, and primary metric were saved before image or 3D reference inspection. One RGB-only review then saved [labels](results/sunrgbd_lamp_table8_labels.json): four clear positive pairs, three clear negative pairs, and one uncertain. Only after saving labels was predicted depth run and reference 3D read. No correction settings were changed.

| Lamp/table positive pairs, mean child 3D error | Result |
|---|---:|
| No correction | 0.799 m |
| Height-only exact contact | 0.796 m |
| Paired camera-ray correction | 0.800 m |
| Child-only soft camera-ray correction | 0.804 m |

Child-only minus paired is **+0.004 m**, a slight worsening. Child-only improved two of four individual pairs and worsened two; the scene-bootstrap 95% interval for its difference from paired is **−0.018 to +0.024 m**. The sample is too small for a population claim. Two lamp scenes had already large uncorrected child errors of about 1.5 m and 1.2 m, and both ray methods worsened them. The other two scenes improved. Across all eight candidate scenes, predicted depth plus oracle boxes had a 0.392 m median center error over 40 annotated objects, which is a separate all-object diagnostic. See the full [per-scene report](results/sunrgbd_lamp_table8_3d_evaluation.json) and [depth benchmark](results/sunrgbd_lamp_table8_depth.json).

**Interpretation:** the fixed 0.4 m cup-derived center-height gap does not generalize to lamps in this sample. The method currently requires known camera intrinsics, a credible camera-up direction, a valid support pair, and an object-appropriate offset. Do not describe it as a validated generic support-correction method or as established novel research. Negative pair labels here evaluate selection prevalence; automatic lamp relation detection was outside the frozen protocol and was not scored.

## Next research gate before 27 October

Use development data to replace the fixed offset with an object-scale/contact estimate and uncertainty, and test a relation-validity gate that can abstain when the image or geometry does not support correction. Freeze both against another genuinely untouched set with room/source grouping and independent relation annotation. Report child and parent 3D error, gap error, relation precision/coverage, memory, and latency. Compare against a child-only ray control and prior semantic scene optimization (for example [Diorama](https://arxiv.org/html/2411.19492)). More diverse ground-truth scenes are needed: the current uniquely paired cup/mug table/desk pool is nearly exhausted.

The lamp result is a useful falsification milestone. It establishes a concrete failure mode and prevents the earlier small-sample cup/desk gain from being overstated as a general solution.
