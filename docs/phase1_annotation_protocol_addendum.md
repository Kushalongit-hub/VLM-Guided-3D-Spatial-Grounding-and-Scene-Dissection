# Phase 1 protocol addendum: split and label safeguards

Recorded 2026-10-04 before development labels or threshold tuning. The original
selection files and protocol remain unchanged; this documents corrections to
their interpretation.

## Split interpretation

`scripts/audit_phase1_split.py` finds no shared scene IDs between the 20-scene
development set and the 57-scene final set. It finds seven shared parent paths
of `sequence_name`. This path is a conservative source-group proxy, not a
verified room identifier. Therefore, the existing selections do **not** support
a room-disjoint generalization claim. Keep the frozen memberships, report
source-group overlap, and stratify or qualify uncertainty accordingly. Do not
choose test scenes using observed method performance. A metadata-only derived
subset, `results/sunrgbd_phase1_final_source_disjoint30_selection.json`,
retains 30 final scenes with zero overlapping source-group proxies. The
original 57-scene selection remains frozen for a broader descriptive analysis;
use the 30-scene subset for the stronger transfer claim and disclose its
smaller per-category counts (bottle+desk 14, laptop+table 7, bottle+table 6,
bowl+table 3). This derivation occurred before final image inspection or
scoring.

## Label order and evidence

For the first annotation pass, show RGB with neutral object IDs and candidate
classes. Do not show predicted depth, predicted relations, correction outputs,
or annotated 3D centers. A second pass may inspect sensor depth for uncertain
contact, recording `evidence_stage=rgbd_review`; keep the first-pass judgment.
Use SUN RGB-D `reference.json` only after relation labels are locked, for
association and scoring. Its 3D centers cannot be evidence for a supposed
independent contact label.

Physical support is a scene-level relation. Adjacent predicted masks are
support evidence, but are neither necessary nor sufficient for physical
contact. Occlusion, segmentation gaps, and projection can separate masks;
simple mask overlap can also arise without support.

Distinguish `unsupported`, `uncertain_contact`, `multiple_plausible_parents`,
and `unjudgeable`. An annotator's uncertainty is not a true negative, and a
pipeline abstention is still a coverage outcome on every scene. Report relation
accuracy only where the reference relation is judgeable, with explicit
denominators; report all-scene abstention separately. Do not award a free
correct prediction when both annotator and pipeline abstain.

Because one agent preparing this packet is also inspecting the data, any labels
it supplies are provisional. Independent human review is needed before calling
the relation labels independent reference annotations.
