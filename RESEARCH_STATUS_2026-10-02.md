# Research status — 2 October 2026

**Deadline:** 27 October 2026. **Current outcome:** working RGB-to-3D support pipeline, with a newly disjoint 15-scene evaluation. The proposed weighted correction is still **not shown to outperform exact contact**.

## New evaluation

Fifteen cup/mug-plus-table candidates were selected from the complement of SUN RGB-D's official test-image list, excluding every scene used in the earlier project selections. The random seed, minimum metadata-index gap of 20, selection checksum, detector checkpoint checksum, support code checksums, 0.25/0.25 detector settings, 0.4 m offset prior, uncertainty values, and evaluation metrics were saved in [`results/sunrgbd_support_train15_protocol.json`](results/sunrgbd_support_train15_protocol.json) before opening images, depth, or 3D reference values. One visual annotator labeled RGB images and 2D boxes: **10 supported pairs, 5 pairs on other surfaces**. Labels were saved before running depth inference or reading 3D reference centers.

The protocol calls this an evaluation sample, not a standard SUN RGB-D test-set benchmark. Dataset training-split membership does not imply that this project trained on these scenes. Index spacing and distinct sequence names do not establish room independence. The conditional automatic 3D errors use oracle boxes to align an emitted proposal with the annotated pair for **scoring only**; the detector and proposal do not receive oracle boxes at inference.

| Metric | Result |
|---|---:|
| Positive paired boxes at IoU ≥ 0.5 | **7/10** |
| Positive paired support proposals after detector and predicted depth | **7/10** |
| Annotated negative pairs incorrectly proposed | **1/5** |
| Oracle-box mean child 3D error, no correction / weighted / exact contact | **0.557 / 0.549 / 0.549 m** |
| Automatic matched-pair mean child 3D error, no correction / weighted / exact contact | **0.643 / 0.636 / 0.636 m** on the **7/10** positives reaching correction |

For the oracle-box primary metric, weighted minus no correction was **−0.0075 m**; a 10,000-resample scene bootstrap interval was **−0.0145 to +0.0003 m**. Weighted minus exact contact was **+0.00015 m** with an interval of **−0.00022 to +0.00050 m**. The signs mean weighted correction improved relative to no correction on average, but the interval includes no effect; exact contact was essentially tied. The automatic subset is conditional on detection and proposal success and cannot stand in for all ten positives. The heuristic falsely proposed the annotated negative pair in **sunrgbd_005307**, where the cup is on a foreground bin instead of the rear table. Other proposals were not exhaustively labeled, so this is not scene-level precision.

Predicted depth alone gave a median 3D center error of **0.504 m** across 100 uniquely paired objects in these 15 images using oracle boxes and official intrinsics. This broader object figure is separate from the 10 supported-pair child result.

## Interpretation and next research decision

This is a stronger engineering milestone than the earlier desk-only check: the frozen detector/proposal settings were carried to disjoint images, and the 3D correction was compared against both required controls. It is a **negative result for the current novelty claim**. The weighted optimizer cannot yet claim a distinct placement advantage over exact contact; both use a manually chosen 0.4 m center-height offset and supplied official intrinsics. The three automatic positive misses lacked adequate paired boxes; the false positive demonstrates that the support rule also needs a stronger contact check.

The next research work should happen on development scenes: derive a contact offset from image geometry or depth with an uncertainty estimate, then freeze that mechanism and evaluate on another untouched sample. The just-scored 15 scenes must not be used to tune the method while still being called a held-out test. A second relation annotator and wider object categories would strengthen any paper claim.

## Evidence and reproduction

- [Selection](results/sunrgbd_support_train15_selection.json), [pre-image protocol](results/sunrgbd_support_train15_protocol.json), and [RGB-only labels](results/sunrgbd_support_train15_labels.json).
- [Depth benchmark](results/sunrgbd_support_train15_depth.json), [detector outputs](results/sunrgbd_support_train15_grounding_dino.json), and [per-scene 3D evaluation](results/sunrgbd_support_train15_3d_evaluation.json).
- [Evaluation code](scripts/evaluate_frozen_support_3d.py); commands are in [results/README.md](results/README.md).
- The unit suite passed **24 tests** on 2 October 2026; the frozen protocol's selection, code, and checkpoint hashes were validated during scoring.
- Checkpoint and SUN RGB-D image files remain local and ignored by Git; report files are present in the workspace but not committed or independently replicated.
