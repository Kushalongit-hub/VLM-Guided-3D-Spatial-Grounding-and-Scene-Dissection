# Research and implementation plan — deadline 2026-10-27

## Target

Deliver a reproducible single RGB indoor scene pipeline with object-level 3D
layout and a tested method for correcting geometry using support relations.
Claim novelty only after comparison with closely related methods, particularly
SpatialRGPT and Diorama. Show a measurable gain over the same pipeline without
correction.

## Current baseline

`scripts/run_baseline.py` takes metric depth, pixel boxes, and known camera
intrinsics. It exports camera-space centroids and visible extents. This isolates
the geometry stage and gives a repeatable ablation before model integration.
`scripts/run_benchmark.py` evaluates a frozen manifest of scenes against
reference 3D centers. `scripts/prepare_sunrgbd_scene.py` converts one SUN RGB-D
v2 depth PNG plus VoteNet-extracted label and calibration files into the project
    manifest format. The current `.venv` has Python 3.14 and CUDA PyTorch
2.14.0+cu130. End-to-end inference has run on the RTX 5060 Laptop GPU, which
reports 8151 MiB; peak pipeline VRAM has not yet been measured.
The official indoor metric Small checkpoint is cached locally and
`scripts/infer_depth.py` completed CPU inference on one example image.
`scripts/run_inference.py` now combines predicted depth with supplied boxes
and camera intrinsics to write a 3D layout. The bundled outdoor example is an
execution check only because its intrinsics are assumed and the indoor model
is outside its intended domain.
The same script now supports local YOLO11n COCO detection weights and writes
raw detections plus a box overlay. This is the automated detection baseline;
its category set and boxes should be compared with later VLM or mask-based
regions on real indoor scenes.
YOLO11n-seg now provides an optional mask-based lifting path with saved masks
and overlays. Evaluate it against the box path on the same annotated indoor
images; synthetic tests only establish the expected background exclusion.
Same-class mask-IoU filtering removed a duplicate chair detection on the
bundled indoor example. A fixed ten-scene SUN RGB-D depth smoke benchmark now
reports 0.399 m mean per-scene MAE and 0.164 AbsRel against sensor depth;
see `results/README.md`. This does not evaluate 3D object placement.
An initial calibrated 3D center benchmark now uses three official SUN RGB-D
scenes and 14 unambiguous object pairs. Median center error is 0.442 m with
sensor depth and 0.415 m with predicted depth, both using oracle 2D boxes.
This is a small geometry sanity check, not a representative or held-out result.
A frozen 20-scene subset from the SUN RGB-D test list now covers four sensor
types and 93 uniquely paired objects. Its median center error is 0.301 m
with sensor depth versus 0.422 m with predicted depth (oracle 2D boxes in
both). The larger subset reverses the earlier tiny-sample impression and
identifies metric-depth generalization as a concrete bottleneck.
The full RGB-to-layout path has now run on these same 20 scenes. YOLO11n-seg
has a class-aware IoU≥0.5 match for 15 of 22 conservatively COCO-mappable
objects (15 of all 93 paired annotations). Category coverage is a major
bottleneck. After warmup, measured pipeline wall time averaged 0.193 s/image;
PyTorch's inference peak was 403 MiB allocated and 522 MiB reserved, excluding
CUDA context and non-PyTorch allocations. See `results/README.md`.
A cup/mug support proposal heuristic now emits candidate relations from box
footpoints and median predicted depth. It agrees with 18 labeled positive/
negative pair decisions on scenes already inspected during rule design, so
this is only a sanity check. It does not solve the detector's missing table
regions or establish relation accuracy on unseen scenes.
On a separately frozen cup/mug-and-desk set, the unchanged heuristic found
9/9 visually positive pairs with supplied boxes and predicted depth; the
sample had no clear negatives. The YOLO11n-seg path found only 2/9 complete
annotated pairs, confirming that broader indoor object detection is the
immediate end-to-end bottleneck. This still does not validate relation
specificity or a novel correction method.

## Completed
- Data adapter for SUN RGB-D v2 (`src/pipeline/sunrgbd.py`, `scripts/prepare_sunrgbd_scene.py`).
- Manifest-based benchmark runner (`src/pipeline/benchmark.py`, `scripts/run_benchmark.py`).
- Centroid evaluation with coverage, median, and P90 error (`src/pipeline/evaluation.py`, `scripts/evaluate_layout.py`).
- `build_layout` now preserves `instance_id` for evaluation.
- Unit tests cover depth decoding, label conversion (identity and non-identity rotation), calibration parsing, manifest aggregation, and geometry baseline validation (9 tests passing).

## Next milestones

1. **By October 4:** Select a dataset with RGB, metric depth, calibrated camera,
    object regions, and support annotations or a defensible support-label protocol.
    [A 20-scene balanced test subset is frozen and measured. Next: scale to
    more scenes and handle repeated object classes with a validated matcher.]
    Do not infer full 3D dimensions from visible depth points
    without explicit shape assumptions.
2. **By October 11:** Install GPU-enabled PyTorch, integrate the metric Depth
    Anything V2 indoor checkpoint and an object detector or VLM with validated
    pixel boxes. Save raw model outputs for reproducibility. Measure centroid and
    relation accuracy, runtime, and peak VRAM on this machine. [GPU inference
    and a small real-depth benchmark complete; a first 20-scene detector/region
    coverage and peak PyTorch-memory evaluation is complete. Broader indoor
    category coverage and total device VRAM accounting remain.]
3. **By October 18:** Specify and implement uncertainty-weighted support
   constraints. Separate physical contact from generic relations such as
   “above.” Compare with no correction and a simple hard-constraint baseline.
   [Height-only weighted solver and hard baseline implemented; one exploratory
   manually labeled mug-on-table case measured. A ten-scene support pilot
   exposed over-trusting table centers; a predeclared nine-scene confirmation
   improved child center error by 0.014 m on five positives, but hard contact
   performed nearly identically. Relation detection, calibrated uncertainty,
   and a larger independent support benchmark remain.]
4. **By October 23:** Run held-out evaluation and ablations, inspect failure
   cases, implement the visualization, and make the inference command repeatable.
5. **By October 27:** Freeze code and results; write the report with an honest
   comparison to prior work and measured limits.

## Metrics and gates

- Median and 90th-percentile 3D centroid error in metres; relation precision,
  recall, and F1 for contact and support; report confidence intervals by scene.
- Peak CUDA allocated/reserved memory and wall-clock latency at batch size one.
- Compare oracle depth/boxes and predicted depth/boxes to separate model errors
  from the proposed correction. Report per-scene failures.
- Only call the system “metric accurate” when real-scene measurement supports it.
  A predicted metric checkpoint yields metres but does not guarantee accuracy.

## Decisions to validate early

Determine whether the available dataset actually provides support labels and
object-level 3D ground truth. NYU Depth v2 offers RGB-D data but should not be
assumed to supply all labels needed for this evaluation. If labels are absent,
annotate a small fixed test set or choose a better suited dataset before claiming
support-relation improvements.

For the first geometry benchmark, use SUN RGB-D: its RGB-D data, calibration,
and 2D/3D box annotations suit object placement evaluation. Convert source
annotations into the manifest schema documented in README, carefully checking
the camera-coordinate transformation on a visualized sample. This benchmark
alone cannot establish support-relation performance. Use a separately labelled
set for physical support and contact, or a dataset with verified relation labels.
