# Research status report — 1 October 2026

**Project:** VLM-Guided 3D Spatial Grounding and Scene Dissection  
**Deadline:** 27 October 2026  
**Status:** Runnable RGB-to-3D geometry prototype with a small support-correction study; research novelty and end-to-end accuracy remain unproven.

## Objective

Produce object-level 3D layout from a single indoor RGB image on an 8 GB GPU, then test whether physical support constraints improve 3D placement under uncertain monocular depth. The intended research contribution is a specific, measured correction mechanism. The existing detector, depth model, and 3D lifting path constitute an engineering baseline.

## System implemented

The runnable `scripts/run_inference.py` accepts an image and camera intrinsics. It can use supplied 2D boxes, YOLO11n detections, or YOLO11n-seg instance masks; Depth Anything V2 Metric Hypersim Small predicts depth; `src/pipeline/geometry.py` back-projects valid depth pixels and exports camera-frame object centers and visible extents in JSON. The script can save depth arrays, detections, masks, and an overlay. This path has run on the RTX 5060 Laptop GPU. Its visible extents are **not** full amodal object dimensions.

`src/pipeline/support_constraints.py` adjusts center heights through weighted least squares. An exact-contact setting provides an ablation. The image-to-JSON CLI can call this stage through `--support-constraints`, using an explicit `child supported_by parent` relation, center-height offset, per-object uncertainties, and camera up direction. A separate `--propose-support` option now emits **heuristic candidates** for cup/mug support surfaces from box position and predicted depth. It does not apply a correction or provide calibrated contact probabilities. Supplied correction IDs must match the object regions.

## Measured results

| Experiment | Setting | Result |
|---|---|---|
| Depth smoke test | 10 SUN RGB-D images, sensor depth reference | Mean scene MAE **0.399 m**; AbsRel **0.164**; depth inference **0.223 s/image** after loading. |
| Object-center benchmark | 20 frozen SUN RGB-D test scenes, 93 uniquely paired objects, oracle 2D boxes and official calibration | Median 3D center error **0.301 m** with sensor depth versus **0.422 m** with RGB-predicted depth. Predicted depth worsened 60/93 objects. |
| Initial support pilot | 10 separate candidates; 8 visually labeled positives; fixed 0.20 m table-center uncertainty | Mean positive child center error **rose from 0.214 m to 0.238 m** after weighted correction, despite a smaller height-gap error. |
| Predeclared support confirmation | 9 remaining candidates; 5 positives, 3 negatives, 1 uncertain; table uncertainty fixed to 0.80 m before viewing confirmation images | Mean positive child center error **fell from 0.345 m to 0.331 m**. All five improved by **0.005–0.027 m**. Exact contact was essentially tied (**0.331 m**). |
| Full RGB path | Same frozen 20 scenes; YOLO11n-seg + metric depth + mask lifting; official calibration | **27/93** annotations had a class-agnostic box-overlap match; **15/22** conservatively COCO-mappable annotations had a class-aware match. This is **15/93** class-aware matches overall. |
| New desk support check | 10 previously unseen cup/mug-plus-desk candidates; 9 visually labeled positive, 1 uncertain; proposal code hashed before viewing images | **9/9** supported pairs proposed with supplied boxes and predicted depth. With YOLO11n-seg detections, only **2/9** complete annotated pairs were found. No clear negatives, so specificity is unmeasured. |
| Open-vocabulary detector check | Same 10 desk scenes; Grounding DINO Tiny with fixed prompt `a cup. a mug. a desk. a table.`, box threshold 0.4, text threshold 0.3 | **1/9** positive pairs detected at IoU ≥ 0.5 for both objects; **1/9** reached the support proposal stage. Median later-image inference **0.347 s**; PyTorch peak **1,870 MiB allocated**, **2,208 MiB reserved**. |
| Exploratory lower detector threshold | Same desk scenes; Grounding DINO Tiny with separate cup/mug/desk/table phrases and 0.25/0.25 thresholds, examined after the 0.4 result | **8/9** positive pairs detected and **8/9** passed the unchanged support proposal. This is a development result, not an untouched final test. |
| Table transfer check | Nine earlier table confirmation scenes; same Grounding DINO Tiny 0.25/0.25 setting and unchanged depth proposal; 5 positive, 3 negative, 1 uncertain | **5/5** positive paired boxes and support proposals recovered; **0/3** annotated negative pairs matched. Other proposals in one negative scene lack exhaustive labels. |

The 20-scene depth comparison used a balanced convenience subset across Kinect v1, Kinect v2, Xtion, and RealSense. The mean scene-level predicted-minus-sensor center-error difference was **+0.209 m** with a scene-bootstrap 95% interval of **+0.082 to +0.356 m**. This establishes depth generalization as a current bottleneck; it does not measure detector performance. The separate 10-image depth test was a smoke benchmark, not a representative estimate.

## Interpretation

The first support pilot exposed a real failure mode: broad table boxes can yield less reliable centers than cups, yet the initial uncertainty values made the optimizer move the cup more than the table. Four table centers in the separate 20-scene development benchmark averaged 0.793 m 3D error. The revised 0.80 m table uncertainty was frozen before the nine-scene confirmation and modestly improved placement there.

The support result remains weak evidence. The positive confirmation sample has only five cases; some scenes are nearby frames of the same room; labels were visually assigned by one annotator; oracle object boxes and official camera calibration were used; the 0.40 m center-height offset is a manual prior. The weighted and exact-contact methods perform almost identically. There is no demonstrated advantage specific to uncertainty weighting, no automatic relation-detection score, and no fair published-method comparison. **Do not claim a novel or metric-accurate final model yet.**

The full RGB path is operational, but category coverage is now an observed bottleneck. Only 22/93 benchmark annotations map conservatively to a COCO class. Among the same 15 class-aware matched objects, mean 3D center error was **0.585 m** using predicted boxes and **0.526 m** using predicted masks; oracle-box mean on that matched subset was **0.561 m**. The ordering is mixed by mean versus median, so masks are not yet established as superior. After loading, the first-image wall time was **1.89 s** and later images averaged **0.193 s**; PyTorch reported **403 MiB peak allocated** and **522 MiB peak reserved** during inference with both small models resident. These GPU counters are narrower than total device VRAM use.

The heuristic support proposal agrees with all 13 positive and five negative cup/mug-table pair labels on the 19 scenes already inspected. This is a design-set sanity check only: the rule was chosen after viewing the same boxes and depth gaps. It uses oracle boxes in this diagnostic. In a representative development image, YOLO11n-seg found the cup but missed its table, so a correct relation proposal cannot be emitted from the detector output. A new, uninspected relation set and better indoor category coverage are required before any automatic-relation claim.

A subsequent cup/mug-on-desk set was selected and the proposal code hashed before the images were opened. The unchanged rule recovered all nine visually positive pairs using supplied boxes and predicted depth; one cropped pair was marked uncertain. There were no clear negative desk pairs, so this only supports *recall on visible examples*, not relation precision or physical-contact reliability. When boxes came from YOLO11n-seg, only two of nine positive pairs were correctly proposed. This directly isolates missing indoor object regions as the dominant blocker for an automatic final-stage result.

An official Grounding DINO Tiny checkpoint was also tested on these same frozen scenes. A single fixed four-category prompt at the documented 0.4/0.3 thresholds produced complete cup/mug-plus-desk detections for **1/9** positive pairs; that pair also passed the unchanged geometry proposal. This is weaker than the **2/9** YOLO result in this setting. Prompt phrasing and thresholds can affect this detector, so this is a fixed-setting comparison, not a general ranking of open-vocabulary detectors. The Grounding DINO inference measurement excludes depth estimation, 3D lifting, and model download; the YOLO timing cited above measured a different full path and is not a strict speed comparison.

After observing this result, a second run used 0.25/0.25 thresholds and the processor's explicit list of category phrases. It recovered **8/9** positive pairs at IoU ≥ 0.5 for both boxes, and all eight passed the unchanged metric-depth support proposal. The threshold change is the material difference: a separate-phrases 0.4/0.3 run still recovered 1/9. This is a substantial *engineering* milestone for the cup/mug-on-desk path. It is exploratory because the same desk set was inspected before lowering the threshold. It does not measure false positives, 3D placement accuracy, or a novel correction mechanism. The single-image CLI now runs this detector, predicted metric depth, 3D lifting, and support proposals together.

Without another detector setting change, the same 0.25/0.25 pipeline recovered all five visibly supported cup/mug-on-table pairs in the earlier confirmation set. None of the three annotated negative *pairs* matched, though one negative scene generated three other unverified proposals. The confirmation scenes had been used in prior geometry work, so this is a transfer diagnostic for the detector and integrated path rather than a fresh final research test. The remaining research gap is to demonstrate an actual correction method that improves 3D placement over exact contact and no correction on a genuinely independent set.

## Reproducibility and evidence

- Full numerical results and reproduction commands: [`results/README.md`](results/README.md).
- Frozen 20-scene selection and paired summary: [`results/sunrgbd_heldout20_selection.json`](results/sunrgbd_heldout20_selection.json), [`results/sunrgbd_heldout20_summary.json`](results/sunrgbd_heldout20_summary.json).
- Pilot labels and per-scene result: [`results/sunrgbd_support_candidates_labels.json`](results/sunrgbd_support_candidates_labels.json), [`results/sunrgbd_support_candidates_report.json`](results/sunrgbd_support_candidates_report.json).
- Predeclared confirmation protocol, RGB-only relation labels, and per-scene result: [`results/support_confirmation_protocol.json`](results/support_confirmation_protocol.json), [`results/sunrgbd_support_confirmation_labels.json`](results/sunrgbd_support_confirmation_labels.json), [`results/sunrgbd_support_confirmation_report.json`](results/sunrgbd_support_confirmation_report.json).
- Evaluation code: [`scripts/evaluate_support_candidates.py`](scripts/evaluate_support_candidates.py).
- Full RGB path benchmark, with per-scene detections, matching, latency, and memory: [`results/sunrgbd_heldout20_rgb_pipeline.json`](results/sunrgbd_heldout20_rgb_pipeline.json).
- Scene-specific corrected inference example: [`examples/sunrgbd_002591_support.example.json`](examples/sunrgbd_002591_support.example.json), with command in [`results/README.md`](results/README.md).
- Support proposal diagnostic: [`results/sunrgbd_support_proposal_diagnostic.json`](results/sunrgbd_support_proposal_diagnostic.json).
- Frozen desk protocol and relation labels: [`results/sunrgbd_support_desk_protocol.json`](results/sunrgbd_support_desk_protocol.json), [`results/sunrgbd_support_desk_labels.json`](results/sunrgbd_support_desk_labels.json).
- Desk proposal and detected-pair results: [`results/sunrgbd_support_desk_report.json`](results/sunrgbd_support_desk_report.json), [`results/sunrgbd_support_desk_detected_report.json`](results/sunrgbd_support_desk_detected_report.json).
- Grounding DINO Tiny fixed-setting detections and paired proposal results: [`results/sunrgbd_support_desk_grounding_dino.json`](results/sunrgbd_support_desk_grounding_dino.json), [`results/sunrgbd_support_desk_grounding_dino_pairs.json`](results/sunrgbd_support_desk_grounding_dino_pairs.json).
- Exploratory lower-threshold detections and proposal results: [`results/sunrgbd_support_desk_grounding_dino_exploratory025.json`](results/sunrgbd_support_desk_grounding_dino_exploratory025.json), [`results/sunrgbd_support_desk_grounding_dino_exploratory025_pairs.json`](results/sunrgbd_support_desk_grounding_dino_exploratory025_pairs.json).
- Table transfer detections and paired-relation results: [`results/sunrgbd_support_confirmation_grounding_dino025.json`](results/sunrgbd_support_confirmation_grounding_dino025.json), [`results/sunrgbd_support_confirmation_grounding_dino025_pairs.json`](results/sunrgbd_support_confirmation_grounding_dino025_pairs.json).
- The local unit suite passed **23 tests** on 1 October 2026.

Model checkpoints and downloaded SUN RGB-D data are ignored by Git. The frozen selections include metadata checksums and source identifiers so the dataset can be reconstructed. Result files are saved in the repository workspace but have not been committed or independently replicated.

## Work required by 27 October

1. **By 4 October:** Define a relation-labeling protocol with a second reviewer or an independently labeled support dataset; avoid neighboring frames across development and test. Preserve uncertain cases rather than silently treating them as negatives.
2. **By 11 October:** The first YOLO box/mask coverage and GPU timing benchmark is complete. Validate on more scenes and categories, test a detector with broader indoor vocabulary, and distinguish model inference memory from total process/device VRAM. Report automatic RGB-to-3D object-center accuracy separately from oracle-box geometry accuracy.
3. **By 18 October:** Replace the fixed center-height offset and hand-set uncertainties with a clearly specified geometric/contact estimator and calibrated uncertainty. Test its failure cases on development data only; freeze it before final evaluation.
4. **By 23 October:** Run a larger independent evaluation with no correction, exact contact, and the proposed method; include per-scene outcomes and an appropriate uncertainty interval. Compare the actual mechanism with closely related published scene-optimization systems.
5. **By 27 October:** Freeze code and data protocol, write the final report, and limit claims to results that survive the independent evaluation.

**Decision gate:** If the proposed method cannot outperform exact contact on a genuinely separate set, present the work as a reproducible system and negative/diagnostic study rather than a validated novel algorithm.
