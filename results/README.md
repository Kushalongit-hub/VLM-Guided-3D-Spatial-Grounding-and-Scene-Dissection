# Research results and reproduction

The [frozen six-scene laptop/table transfer](../RESEARCH_STATUS_2026-10-02_LAPTOP_TRANSFER.md)
found 5/5 visually positive annotated pairs detected and proposed; mean
matched laptop center error fell from 0.606 m with SAM masks to 0.346 m after
the fixed child-only correction. One annotated negative pair had an alternate
detected table surface, so whole-scene relation correctness was unresolved.
See the selection, RGB labels, protocol and per-scene report linked there.

The latest [Grounding DINO + SAM milestone](../RESEARCH_STATUS_2026-10-02_MASKS.md)
documents a fully automatic CUDA demo and paired box-versus-mask 3D checks.
On 15 previously scored lamp/table detections, mean 3D center error was
**0.537 m with boxes** and **0.398 m with SAM masks**. On 11 previously scored
positive cup/mug/table pairs, mask-based child-ray correction averaged
**0.458 m child error** versus **0.566 m** with box-based correction. These are
post-hoc exploratory comparisons on inspected scenes; see the report for
per-category failures and relation-oracle limits.
The optional visible-mask contact gate retained 13/18 matched positive pairs
and 0/5 matched negative pairs across two previously inspected sets; see the
[diagnostic summary](sunrgbd_mask_contact_transfer_summary.json). It is not a
held-out or whole-scene relation score.

## First real indoor depth benchmark (2026-10-01)

This is a **pipeline milestone**, not a 3D layout or research-novelty result.
It measures Depth Anything V2 Metric Hypersim Small on ten SUN RGB-D indoor
test images from the community [SUNRGBD_seg repackaging](https://huggingface.co/datasets/wyrx/SUNRGBD_seg).
The frozen source revision, row indices, and image/depth checksums are saved in
the local ignored `data/sunrgbd_sample/source_manifest.json`; the source rows
are test indices 0–9. The packaged depth PNGs retain SUN RGB-D's 3-bit rotated
millimetre encoding, decoded by `src/pipeline/sunrgbd.py`.

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe scripts\fetch_sunrgbd_samples.py --length 10
.\.venv\Scripts\python.exe scripts\benchmark_metric_depth.py `
  --checkpoint checkpoints\depth_anything_v2_metric_hypersim_vits.pth `
  --output outputs\sunrgbd_10_depth_benchmark.json --device cuda
```

The saved [JSON result](sunrgbd_10_depth_benchmark.json) reports the mean of
scene-level metrics: **0.399 m MAE**, **0.164 AbsRel**, and **0.769 delta-1**
over **2,775,294 valid sensor pixels**. Per-scene MAE spans 0.123–0.579 m.
These numbers use reference depths from 0.1 to 7.99 m, excluding zero/invalid
and 8 m clipped sensor pixels. Inference averaged 0.223 s per image on an RTX
5060 Laptop GPU, after model loading; this is not whole-pipeline latency.
PyTorch reported a depth-stage peak of 384 MiB allocated and 426 MiB reserved
after model loading. These figures exclude the detector and other GPU processes.

The subset is small and selected by row order, so it is a smoke benchmark,
not a representative SUN RGB-D result. Sensor depth has noise and missing
values. The camera calibration and object-level 3D annotations are absent from
this repackaging, so these numbers do not validate 3D centroids or support
relations. A separate first object-level check using official SUN RGB-D
annotations and calibrated camera data follows below.

## First calibrated object-center benchmark (2026-10-01)

Three fixed scenes (official metadata indices 0, 2, and 3) were fetched from
the [official SUN RGB-D archive](https://rgbd.cs.princeton.edu/data/) using
HTTP byte ranges. We paired 2D and 3D boxes only when each class was unique
in the scene and the 2D record marked a 3D box as present. This yielded 14
paired objects; unmatched/ambiguous entries were excluded. Source filenames,
SHA-256 checksums, and metadata indices are saved in
[`sunrgbd_official_3_source.json`](sunrgbd_official_3_source.json).

| Depth source | Median 3D center error | P90 3D center error | Objects |
|---|---:|---:|---:|
| SUN RGB-D sensor depth | 0.442 m | 1.187 m | 14 |
| Predicted RGB depth | 0.415 m | 1.228 m | 14 |

The [sensor-depth result](sunrgbd_official_3_sensor_depth.json) and
[predicted-depth result](sunrgbd_official_3_predicted_depth.json) use the same
**oracle 2D boxes and official camera calibration**. The statistic is the
Euclidean distance between the median of visible points back-projected inside
an annotated 2D box and the annotated **amodal** 3D box center. It therefore
mixes surface-to-center bias, box background, sensor/model depth error, and
annotation noise. The 100% match coverage is by construction, not detector
recall. Predicted depth improved 9 of 14 individual objects but worsened 5;
the tiny aggregate difference cannot establish superiority.

The three selected scenes are neighboring entries with similar bedroom
content, chosen for unambiguous 2D/3D pairing. This is a geometry sanity
benchmark, not a held-out or representative result. It does not measure the
YOLO detector, support relations, full 3D size, or paper-level novelty.

Reproduction after downloading the two official MATLAB metadata files into
`data/sunrgbd_official/`:

```powershell
.\.venv\Scripts\python.exe scripts\fetch_sunrgbd_official_subset.py
.\.venv\Scripts\python.exe scripts\run_benchmark.py `
  --manifest data\sunrgbd_official\subset\manifest.json `
  --output outputs\sunrgbd_official_3_oracle_benchmark.json
.\.venv\Scripts\python.exe scripts\benchmark_sunrgbd_rgb.py `
  --manifest data\sunrgbd_official\subset\manifest.json `
  --checkpoint checkpoints\depth_anything_v2_metric_hypersim_vits.pth `
  --output outputs\sunrgbd_official_3_predicted_depth_benchmark.json --device cuda
```

## Frozen 20-scene test subset across four sensors (2026-10-01)

The [selection file](sunrgbd_heldout20_selection.json) froze five scenes each
from Kinect v1, Kinect v2, Xtion, and RealSense **before inference**. The
selection uses the 5,050-image [SUN RGB-D test list](https://github.com/ankurhanda/sunrgbd-meta-data/blob/master/sunrgbd_testing_images.txt),
random seed 27, and requires at least three uniquely pairable 2D/3D objects
per scene. It is a balanced **convenience subset**, not a random sample of all
test scenes. Repeated-class ambiguities are excluded. The [source record](sunrgbd_heldout20_source.json)
stores the official archive member paths and checksums.

| Depth source | Median 3D center error | P90 error | Objects |
|---|---:|---:|---:|
| Sensor depth | 0.301 m | 0.746 m | 93 |
| Predicted RGB depth | 0.422 m | 1.176 m | 93 |

The [paired summary](sunrgbd_heldout20_summary.json) shows **60 of 93 objects
worsened** with predicted depth. The mean of scene-level error differences
(predicted minus sensor) is **+0.209 m**; a scene bootstrap gives a 95% interval
of **+0.082 to +0.356 m**. Five scenes per sensor are too few for a reliable
sensor-specific ranking, but the Kinect v2 group has the largest observed
median gap here (0.422 m sensor versus 0.893 m predicted). The worst inspected
scene shows depth overestimation across furniture and the back wall.

Both [sensor](sunrgbd_heldout20_sensor_depth.json) and
[predicted](sunrgbd_heldout20_predicted_depth.json) results retain **oracle 2D
boxes and official calibration**. They therefore expose depth-stage error
without conflating it with detector misses. They still compare visible-point
medians with amodal object centers, and do not measure the complete RGB-to-3D
system or support reasoning. These numbers should not be compared directly
with 3D detection mAP in published work.

Reproduce the frozen subset after placing the two official `.mat` metadata
files and the test split list in `data/sunrgbd_official/`:

```powershell
.\.venv\Scripts\python.exe scripts\fetch_sunrgbd_official_subset.py `
  --selection results\sunrgbd_heldout20_selection.json `
  --output-dir data\sunrgbd_official\heldout20
.\.venv\Scripts\python.exe scripts\run_benchmark.py `
  --manifest data\sunrgbd_official\heldout20\manifest.json `
  --output outputs\sunrgbd_heldout20_sensor_depth.json
.\.venv\Scripts\python.exe scripts\benchmark_sunrgbd_rgb.py `
  --manifest data\sunrgbd_official\heldout20\manifest.json `
  --checkpoint checkpoints\depth_anything_v2_metric_hypersim_vits.pth `
  --output outputs\sunrgbd_heldout20_predicted_depth.json --device cuda
```

## Exploratory support correction case (2026-10-01)

`src/pipeline/support_constraints.py` adds a separate height-only solver. It
uses object center uncertainty and a supplied physical support relation to
fit child/parent center offsets. The support relation, offset, and uncertainty
must be supplied; the current RGB pipeline does not predict them. The solver
also has an exact-contact mode for an ablation. Original centers are retained
as `uncorrected_center_point_3d`.

The [single case report](support_case_002591.json) uses a visually checked mug
on a table in `sunrgbd_002591`, predicted RGB depth, official oracle boxes and
calibration. A provisional **0.40 m** mug-center-above-table-center prior with
**0.20 m** uncertainty was chosen without reading the 3D reference centers.
Center uncertainties are provisional too. Errors against official amodal 3D
centers are:

| Method | Table | Mug |
|---|---:|---:|
| No correction | 0.458 m | 0.394 m |
| Weighted contact | 0.446 m | 0.330 m |
| Exact contact | 0.444 m | 0.330 m |

This is an integration check, **not a held-out support result**. The offset is
a manually specified prior, its transfer to other scenes is untested, and the
weighted and exact modes perform nearly alike here. The 20-scene set was
already used to inspect failures, so any broad support claim needs a separate
frozen relation-labeled evaluation set and priors selected without its 3D
labels. The method is not yet established as novel relative to semantic scene
optimization in prior work.

Reproduce after generating `predicted_depth.npy` in the frozen subset:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_support_case.py `
  --scene data\sunrgbd_official\heldout20\sunrgbd_002591 `
  --output results\support_case_002591.json
```

## Support correction pilot and confirmation (2026-10-01)

The [exploratory selection](sunrgbd_support_candidates_selection.json) froze
ten mug/cup plus table candidates outside the 20-scene depth benchmark. RGB
images and 2D boxes were inspected to create [relation labels](sunrgbd_support_candidates_labels.json)
before depth inference: eight contact positives and two negatives. With a
0.40 m center-offset prior, 0.20 m table-center uncertainty, and 0.40 m
cup/mug-center uncertainty, correction reduced mean positive-pair upright
height-gap error from **0.261 m to 0.067 m** but increased mean child 3D center
error from **0.214 m to 0.238 m**. Four children improved and four worsened.
The [per-scene pilot report](sunrgbd_support_candidates_report.json) records
the full result.

Inspection showed that cups were sometimes already close to their reference
centers while table centers were badly biased. In the separate 20-scene depth
benchmark, four table centers averaged **0.793 m** 3D error. A post hoc pilot
trial with 0.80 m table uncertainty improved the aggregate, but is not a
valid held-out result. The [confirmation protocol](support_confirmation_protocol.json)
then froze 0.80 m table uncertainty, the original 0.40 m child uncertainty,
and the original 0.40 ± 0.20 m offset **before viewing or inferring on the
remaining nine candidates**.

The [confirmation selection](sunrgbd_support_confirmation_selection.json)
and [RGB-only labels](sunrgbd_support_confirmation_labels.json) contain five
positive pairs, three clear negatives, and one uncertain cropped pair. The
[confirmation report](sunrgbd_support_confirmation_report.json) shows:

| Method | Mean child center error | Mean pair center error | Mean upright gap error |
|---|---:|---:|---:|
| No correction | 0.345 m | 0.323 m | 0.241 m |
| Weighted contact | 0.331 m | 0.310 m | 0.050 m |
| Exact contact | 0.331 m | 0.310 m | 0.046 m |

All five positive children improved, but each gain was only **0.005–0.027 m**.
The two correction methods were nearly identical; these results do not show
that uncertainty weighting is superior to exact contact. The relations were
manually labeled by one annotator, oracle 2D boxes and official calibration
were supplied, the sample is small, and some scenes are nearby frames of the
same room. The confirmation set therefore does not establish independent
generalization, relation-detection accuracy, an end-to-end model result, or
novelty relative to prior support optimization work.

Reproduce after fetching the official scenes (the selection files contain
metadata hashes) and supplying the local depth checkpoint:

```powershell
.\.venv\Scripts\python.exe scripts\fetch_sunrgbd_official_subset.py `
  --selection results\sunrgbd_support_confirmation_selection.json `
  --output-dir data\sunrgbd_official\support_confirmation
.\.venv\Scripts\python.exe scripts\benchmark_sunrgbd_rgb.py `
  --manifest data\sunrgbd_official\support_confirmation\manifest.json `
  --checkpoint checkpoints\depth_anything_v2_metric_hypersim_vits.pth `
  --output outputs\sunrgbd_support_confirmation_predicted_depth.json --device cuda
.\.venv\Scripts\python.exe scripts\evaluate_support_candidates.py `
  --manifest data\sunrgbd_official\support_confirmation\manifest_predicted_depth.json `
  --labels results\sunrgbd_support_confirmation_labels.json `
  --protocol results\support_confirmation_protocol.json `
  --output results\sunrgbd_support_confirmation_report.json
```

## RGB detector, depth, and lifting benchmark (2026-10-01)

The [full RGB benchmark](sunrgbd_heldout20_rgb_pipeline.json) ran YOLO11n-seg,
Depth Anything V2 Metric Hypersim Small, and both box-based and mask-based 3D
lifting on the existing frozen 20-scene set. Official camera calibration was
supplied. Matching is one-to-one 2D-box IoU ≥ 0.5 against the 93 uniquely
paired SUN RGB-D annotations. These annotations omit many scene objects, so
unmatched predictions are **not** counted as false positives; this is not AP.

| Coverage measure | Matched / eligible |
|---|---:|
| Box overlap, any class | 27 / 93 (29.0%) |
| Matching COCO class among conservatively mappable labels | 15 / 22 (68.2%) |
| Matching COCO class among all paired annotations | 15 / 93 (16.1%) |

Only 22 of 93 SUN RGB-D labels have the conservative class mapping listed in
the JSON report. Desk, generic table, monitor, and many other categories are
excluded because they lack an exact COCO counterpart in this evaluation.
Eight mappable objects were missed at the chosen confidence and IoU thresholds;
keyboard and mouse had no matches in this small set. Class-agnostic overlap
does not establish correct recognition.

For the **same 15 class-aware matched objects**, the median 3D center errors
were **0.538 m** with oracle boxes, **0.419 m** with predicted boxes, and
**0.516 m** with predicted masks. Their respective means were **0.561 m**,
**0.585 m**, and **0.526 m**. These conditional, small-sample metrics do not
show consistent box or mask superiority, and must not be compared with the
93-object oracle benchmark as if they covered the same objects. Visible-point
medians are compared against amodal 3D box centers, so the oracle box need
not give the lowest error for every object.

After model loading, the first-image wall time was **1.89 s**, and subsequent
images averaged **0.193 s** for detection, image read, depth inference, 3D
lifting, and matching on this RTX 5060 Laptop GPU. Model construction/loading
took **4.07 s**. PyTorch's peak during inference was **403 MiB allocated** and
**522 MiB reserved** with both models resident; these counters exclude CUDA
context, non-PyTorch allocations, and any loading peak before the counter was
reset. Matching and reference evaluation add work that the deployment command
does not need, while file output is not included in the timed path.

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_rgb_pipeline.py `
  --manifest data\sunrgbd_official\heldout20\manifest.json `
  --depth-checkpoint checkpoints\depth_anything_v2_metric_hypersim_vits.pth `
  --segmenter-weights checkpoints\yolo11n-seg.pt `
  --output results\sunrgbd_heldout20_rgb_pipeline.json --device cuda
```

## Optional corrected inference output

The regular RGB inference CLI accepts `--support-constraints` with an explicit
camera up vector, object uncertainties, and physical-contact relations. The
scene-specific [example](../examples/sunrgbd_002591_support.example.json)
uses official SUN RGB-D object IDs and calibration. It exercises depth
prediction, lifting, and support correction in one command:

```powershell
.\.venv\Scripts\python.exe scripts\run_inference.py `
  --image data\sunrgbd_official\heldout20\sunrgbd_002591\image.jpg `
  --objects data\sunrgbd_official\heldout20\sunrgbd_002591\objects.json `
  --checkpoint checkpoints\depth_anything_v2_metric_hypersim_vits.pth `
  --fx 520.532 --fy 520.7444 --cx 277.9258 --cy 215.115 `
  --support-constraints examples\sunrgbd_002591_support.example.json `
  --output outputs\sunrgbd_002591_support_layout.json --device cuda
```

This produced three objects on the local GPU. With the example's 0.80 m table
uncertainty, the mug-center error was **0.394 m before** and **0.366 m after**
correction; the table changed from **0.458 m** to **0.445 m**. The JSON retains
uncorrected centers and the applied height shifts. This is a development-scene
execution check with supplied boxes and relation, not an independent result or
an automatically inferred relation.

## Heuristic support proposals (2026-10-01)

`src/pipeline/support_proposals.py` now proposes a cup/mug support surface
when the child's box-bottom center falls inside a table/desk/counter box and
the median predicted depths differ by at most 0.75 m. It emits the IDs, depth
gap, and footpoint as inspectable evidence. `--propose-support` adds these
proposals to the inference JSON **without applying a correction**; explicit
constraints remain required for that stage.

The [diagnostic report](sunrgbd_support_proposal_diagnostic.json) checked the
19 previously viewed cup/mug-table candidate scenes: 13 labeled positives,
five negatives, and one uncertain. All 13 positive and five negative *paired
table decisions* agreed with the rule. **This is resubstitution, not a test
score:** the boxes and depth gaps were inspected while designing the heuristic,
oracle object boxes were supplied, and the labels cover only selected pairs.
The heuristic may fail in clutter, at image boundaries, with occlusion, or
when median box depth is contaminated by background. It has no calibrated
confidence and does not prove physical contact.

On scene `sunrgbd_002591`, the GPU inference command with `--propose-support`
proposed mug `object_2` supported by table `object_1` with a 0.283 m median
depth gap. The YOLO11n-seg benchmark for this image detected the cup but did
not detect the table, so the automatic detector path cannot produce that pair
yet. Broader indoor object coverage and a separately frozen relation set are
necessary before reporting automatic relation accuracy.

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_support_proposals.py `
  --pilot-root data\sunrgbd_official\support_candidates `
  --pilot-labels results\sunrgbd_support_candidates_labels.json `
  --confirmation-root data\sunrgbd_official\support_confirmation `
  --confirmation-labels results\sunrgbd_support_confirmation_labels.json `
  --output results\sunrgbd_support_proposal_diagnostic.json
```

## Frozen cup/mug-on-desk proposal check (2026-10-01)

The [selection](sunrgbd_support_desk_selection.json) chose ten test-split
scenes with uniquely paired cup/mug and desk annotations, excluding earlier
scenes and nearby metadata indices. The [protocol](sunrgbd_support_desk_protocol.json)
saved a SHA-256 of the unchanged proposal code and selection **before these
images were viewed or RGB depth was inferred**. RGB images and 2D boxes were
then used to save [labels](sunrgbd_support_desk_labels.json): nine visibly
supported pairs and one cropped uncertain pair. No clear negative pair was
present in this selection.

With **oracle 2D boxes**, the frozen heuristic proposed all **9/9** visible
desk support pairs. The [desk proposal report](sunrgbd_support_desk_report.json)
records each decision. This is a useful new-category recall check, but gives
no false-positive or precision estimate. One annotator labeled the images;
metadata-index spacing does not guarantee room independence.

With **YOLO11n-seg boxes** on the same images, the full RGB pipeline produced
only **2/9** correct annotated cup/mug-and-desk support pairs, requiring IoU
≥ 0.5 for both child and parent. The two matched pairs had child/parent IoUs
of 0.832/0.821 and 0.944/0.839. The [detected-pair report](sunrgbd_support_desk_detected_report.json)
and [full pipeline output](sunrgbd_support_desk_rgb_pipeline.json) preserve
the per-scene evidence. The proposal rule succeeds when the relevant boxes
are supplied; **indoor object coverage is the practical bottleneck** in the
automatic path. The 2/9 figure measures pair coverage on this selected set,
not relation precision or end-to-end 3D placement accuracy.

```powershell
.\.venv\Scripts\python.exe scripts\fetch_sunrgbd_official_subset.py `
  --selection results\sunrgbd_support_desk_selection.json `
  --output-dir data\sunrgbd_official\support_desk
.\.venv\Scripts\python.exe scripts\benchmark_sunrgbd_rgb.py `
  --manifest data\sunrgbd_official\support_desk\manifest.json `
  --checkpoint checkpoints\depth_anything_v2_metric_hypersim_vits.pth `
  --output outputs\sunrgbd_support_desk_predicted_depth.json --device cuda
.\.venv\Scripts\python.exe scripts\evaluate_support_desk.py `
  --root data\sunrgbd_official\support_desk `
  --labels results\sunrgbd_support_desk_labels.json `
  --protocol results\sunrgbd_support_desk_protocol.json `
  --selection results\sunrgbd_support_desk_selection.json `
  --output results\sunrgbd_support_desk_report.json
.\.venv\Scripts\python.exe scripts\benchmark_rgb_pipeline.py `
  --manifest data\sunrgbd_official\support_desk\manifest.json `
  --depth-checkpoint checkpoints\depth_anything_v2_metric_hypersim_vits.pth `
  --segmenter-weights checkpoints\yolo11n-seg.pt `
  --output results\sunrgbd_support_desk_rgb_pipeline.json --device cuda
.\.venv\Scripts\python.exe scripts\evaluate_detected_support.py `
  --root data\sunrgbd_official\support_desk `
  --labels results\sunrgbd_support_desk_labels.json `
  --detections results\sunrgbd_support_desk_rgb_pipeline.json `
  --output results\sunrgbd_support_desk_detected_report.json
```

## Grounding DINO Tiny on the same desk scenes (2026-10-01)

The official [Grounding DINO Tiny checkpoint](https://huggingface.co/IDEA-Research/grounding-dino-tiny)
was run with the fixed prompt `a cup. a mug. a desk. a table.`, box
threshold 0.4, text threshold 0.3, and IoU threshold 0.5. The prompts and
thresholds are defined in `scripts/benchmark_grounding_dino.py`. The
[detector result](sunrgbd_support_desk_grounding_dino.json) contains all
per-image boxes, scores, label strings, timing, and best annotated-object
overlaps. One of nine visually positive cup/mug-on-desk pairs had both
required detections. The [unchanged support-proposal evaluation](sunrgbd_support_desk_grounding_dino_pairs.json)
also recovered **1/9**; YOLO11n-seg recovered **2/9** on these scenes.

Median detector inference after the first image was **0.347 s/image** on CUDA.
PyTorch's peak allocated/reserved counters were **1,870/2,208 MiB**. These
are not total GPU usage and omit depth-estimation runtime. The positive-only
selection cannot establish precision, specificity, or a general detector
ranking. This run used one prompt and no threshold search.

**Exploratory follow-up:** after inspecting the 0.4 result, a run with
explicit separate category phrases and 0.25 box/text thresholds detected
both annotated regions in **8/9** positive desk scenes. The same unchanged
predicted-depth proposal rule recovered **8/9** complete pairs. The
[detections](sunrgbd_support_desk_grounding_dino_exploratory025.json) and
[pair evaluation](sunrgbd_support_desk_grounding_dino_exploratory025_pairs.json)
preserve each scene. A separate-phrases run at 0.4/0.3 still found 1/9,
so lowering the threshold explains the gain. Because the threshold was tried
after seeing this set, this is a development result, not an independent
generalization estimate. The detector adapter also suppresses highly
overlapping cup/mug and desk/table synonym boxes at IoU ≥ 0.7.

The full single-image CLI now supports `--grounding-dino-checkpoint`. This
example ran on CUDA and wrote four 3D object entries with support candidates:

```powershell
.\.venv\Scripts\python.exe scripts\run_inference.py `
  --image data\sunrgbd_official\support_desk\sunrgbd_002532\image.jpg `
  --grounding-dino-checkpoint checkpoints\grounding-dino-tiny `
  --checkpoint checkpoints\depth_anything_v2_metric_hypersim_vits.pth `
  --fx 520.532 --fy 520.7444 --cx 277.9258 --cy 215.115 `
  --output outputs\sunrgbd_002532_grounding_dino_layout.json `
  --boxes-preview outputs\sunrgbd_002532_grounding_dino_boxes.png `
  --propose-support --device cuda
```

The same 0.25/0.25 detector setting was then applied to the earlier
cup/mug-on-table confirmation set with no threshold change. The
[detections](sunrgbd_support_confirmation_grounding_dino025.json) covered
both annotated objects in **5/5** visually positive scenes, and the
[paired proposal report](sunrgbd_support_confirmation_grounding_dino025_pairs.json)
matched **5/5** positive and **0/3** annotated negative pairs. One negative
scene emitted three *other* proposals without exhaustive relation labels;
therefore these numbers are not scene-level precision or specificity for
all output relations. These scenes were previously inspected for geometry
work, so this is a transfer diagnostic, not a new untouched research test.

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_grounding_dino.py `
  --root data\sunrgbd_official\support_confirmation `
  --labels results\sunrgbd_support_confirmation_labels.json `
  --checkpoint checkpoints\grounding-dino-tiny `
  --box-threshold 0.25 --text-threshold 0.25 `
  --output results\sunrgbd_support_confirmation_grounding_dino025.json
.\.venv\Scripts\python.exe scripts\evaluate_detected_support.py `
  --root data\sunrgbd_official\support_confirmation `
  --labels results\sunrgbd_support_confirmation_labels.json `
  --detections results\sunrgbd_support_confirmation_grounding_dino025.json `
  --output results\sunrgbd_support_confirmation_grounding_dino025_pairs.json `
  --detector-name "Grounding DINO Tiny 0.25/0.25"
```

## Camera-ray correction and desk transfer (2026-10-02)

The latest [frozen cross-category transfer](../RESEARCH_STATUS_2026-10-02_TRANSFER.md)
tested the fixed correction on eight previously untouched lamp/table candidates
from the official SUN RGB-D test split. Four visually positive pairs averaged
**0.799 m uncorrected**, **0.800 m paired-ray**, and **0.804 m child-only-ray**
child-centroid error. This is a negative small-sample result for generic
transfer. The [protocol](sunrgbd_lamp_table8_protocol.json),
[RGB-only labels](sunrgbd_lamp_table8_labels.json), and
[per-scene scores](sunrgbd_lamp_table8_3d_evaluation.json) are saved.

The later [stronger-baseline audit](../RESEARCH_STATUS_2026-10-02_ABLATION.md)
found that a post-hoc child-only camera-ray control matched or beat the
paired-ray method on both scored samples. It is available in the inference
CLI as `--support-method child-ray-soft`. Treat the older paired-ray gain
below as a comparison against height-only contact, not evidence that the
paired method is best.

The [current research report](../RESEARCH_STATUS_2026-10-02_RADIAL.md) records
the frozen table and desk evaluations, a cluster-sensitivity check, and a
runnable CUDA example. The table sample found a mean child-centroid reduction
of 0.079 m versus exact contact, but this gain does not survive exclusion of
one source group. The desk transfer sample found a 0.147 m reduction across
six positive pairs, with 5/6 automatic proposals recovered. Parent and
height-gap errors worsened under radial correction. These are exploratory
small-sample results, not a standard SUN RGB-D test-set benchmark.

Reproduce the desk scoring from the already downloaded official subset and
saved detector predictions:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_frozen_support_3d.py `
  --root data\sunrgbd_official\radial_desk7 `
  --selection results\sunrgbd_radial_desk7_selection.json `
  --protocol results\sunrgbd_radial_desk7_protocol.json `
  --labels results\sunrgbd_radial_desk7_labels.json `
  --detections results\sunrgbd_radial_desk7_grounding_dino.json `
  --output results\sunrgbd_radial_desk7_3d_evaluation.json
```

## Disjoint 15-scene 3D support evaluation (2026-10-02)

The [selection](sunrgbd_support_train15_selection.json) samples fifteen
uniquely paired cup/mug-plus-table candidates from the **complement of the
official SUN RGB-D test-image list**. All earlier selected project scenes
were excluded. The [protocol](sunrgbd_support_train15_protocol.json) records
the frozen settings and code/checkpoint hashes before image inspection.
RGB and 2D boxes alone were used to save [10 positive and 5 negative pair
labels](sunrgbd_support_train15_labels.json), prior to looking at predicted
depth or 3D references.

The [per-scene 3D report](sunrgbd_support_train15_3d_evaluation.json) shows
**7/10** positive pairs with both detector boxes at IoU ≥ 0.5, **7/10**
with the annotated pair proposed by the unchanged depth heuristic, and
**1/5** annotated negative pairs incorrectly proposed. Other emitted proposals lack
exhaustive labels. Using oracle boxes on the ten positive pairs, mean child
3D centroid error was **0.557 m uncorrected**, **0.549 m weighted**, and
**0.549 m exact contact**. The mean weighted-minus-uncorrected difference was
**−0.0075 m** (scene-bootstrap 95% interval **−0.0145 to +0.0003 m**).
Weighted and exact contact were effectively tied. On the seven automatically
matched and proposed positives, means were **0.643 / 0.636 / 0.636 m**;
unmatched positives remain part of the coverage denominator. This does not
support a distinct uncertainty-weighting novelty claim.

```powershell
.\.venv\Scripts\python.exe scripts\fetch_sunrgbd_official_subset.py `
  --selection results\sunrgbd_support_train15_selection.json `
  --output-dir data\sunrgbd_official\support_train15
.\.venv\Scripts\python.exe scripts\benchmark_sunrgbd_rgb.py `
  --manifest data\sunrgbd_official\support_train15\manifest.json `
  --checkpoint checkpoints\depth_anything_v2_metric_hypersim_vits.pth `
  --output results\sunrgbd_support_train15_depth.json --device cuda
.\.venv\Scripts\python.exe scripts\benchmark_grounding_dino.py `
  --root data\sunrgbd_official\support_train15 `
  --labels results\sunrgbd_support_train15_labels.json `
  --checkpoint checkpoints\grounding-dino-tiny `
  --box-threshold 0.25 --text-threshold 0.25 `
  --output results\sunrgbd_support_train15_grounding_dino.json
.\.venv\Scripts\python.exe scripts\evaluate_frozen_support_3d.py `
  --root data\sunrgbd_official\support_train15 `
  --selection results\sunrgbd_support_train15_selection.json `
  --protocol results\sunrgbd_support_train15_protocol.json `
  --labels results\sunrgbd_support_train15_labels.json `
  --detections results\sunrgbd_support_train15_grounding_dino.json `
  --output results\sunrgbd_support_train15_3d_evaluation.json
```

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_grounding_dino.py `
  --root data\sunrgbd_official\support_desk `
  --labels results\sunrgbd_support_desk_labels.json `
  --checkpoint checkpoints\grounding-dino-tiny `
  --prompt-format combined `
  --output results\sunrgbd_support_desk_grounding_dino.json
.\.venv\Scripts\python.exe scripts\evaluate_detected_support.py `
  --root data\sunrgbd_official\support_desk `
  --labels results\sunrgbd_support_desk_labels.json `
  --detections results\sunrgbd_support_desk_grounding_dino.json `
  --output results\sunrgbd_support_desk_grounding_dino_pairs.json `
  --detector-name "Grounding DINO Tiny"
```
