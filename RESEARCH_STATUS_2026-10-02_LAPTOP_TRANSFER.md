# Frozen laptop/table category transfer — 2 October 2026

The [selection](results/sunrgbd_laptop_table6_selection.json) and [protocol](results/sunrgbd_laptop_table6_protocol.json) were saved before viewing these images or 3D references. Six SUN RGB-D official test scenes each contained one annotated laptop and table. Prior project selections were excluded with a 30-metadata-index gap. The sample contains six distinct recorded sequences, but index/sequence separation alone does not guarantee statistically independent rooms. One RGB-only review of the image and 2D boxes then saved [five visibly positive paired-table relations and one negative](results/sunrgbd_laptop_table6_rgb_labels.json), before predicted/sensor depth or 3D center inspection.

The frozen test used Grounding DINO Tiny prompted for `laptop` and `table` at 0.25 box/text thresholds, SAM ViT-B masks, Depth Anything V2 Metric Hypersim Small, and official camera intrinsics. Detector boxes were matched to the annotated pair at class-aware IoU ≥ 0.5 **only for scoring**. The box-foot/depth-gap proposal, visible-mask adjacency gate, and fixed 0.4 m child-only camera-ray correction ran from predictions. No parameter was changed after viewing the six scenes.

| Measure | Result |
|---|---:|
| Both annotated objects detected and matched | 5/6 scenes |
| Visible positive paired-table proposal before / after mask gate | 5/5 / 5/5 |
| Positive matched laptop 3D center error, box region | 0.637 m mean |
| Positive matched laptop 3D center error, SAM mask | 0.606 m mean |
| Positive matched laptop 3D center error, SAM mask + correction | **0.346 m mean** |
| Positive matched table 3D center error, box / SAM mask | 0.435 / 0.331 m mean |

All five positive matched laptop errors improved under the frozen correction. The paired masked improvement was **0.261 m mean**. Per-scene errors and detections are in the [machine-readable report](results/sunrgbd_laptop_table6_transfer.json). Six scenes and one annotator cannot establish a general effect or robust confidence interval. The fixed offset might still fail on other laptop poses or scales.

The one RGB-negative **annotated pair** is `sunrgbd_003328`: the annotated table is the left surface, while the laptop rests on a right workstation surface. Grounding DINO matched the laptop but its table box (IoU below 0.5 with the annotated table) covered the right surface. It proposed and corrected that detected laptop/table pair; visible mask adjacency was 0.411. Thus paired-table candidate count was 0/1, while **a proposal existed in the scene both before and after the gate**. The proposal's physical correctness is **unresolved** because the alternate support surface is not the annotated target and the current labels do not exhaust all support surfaces. It would be misleading to report this as either a verified false positive or a successful negative rejection. This exposes an evaluation gap: pairwise labels are insufficient for whole-scene relation precision.

This is an encouraging category-transfer stress test for 3D placement, not proof of a novel method. Next, annotate *all* plausible parent surfaces in newly selected scenes before scoring, include clear negatives with no plausible support candidate, and compare against category-size and no-correction controls. The original VLM reasoning stage remains unimplemented; a paper-level novelty claim remains open.
