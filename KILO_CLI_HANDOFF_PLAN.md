# Kilo CLI handoff: 3D support correction research

Prepared 4 October 2026. Final delivery: 27 October 2026, Asia/Kolkata.

## Copy this assignment into Kilo CLI

Work in `C:\AIML\Research\VLM-Guided-3D-Spatial-Grounding-and-Scene-Dissection`. Execute the staged plan below. Your objective is a reproducible, defensible study of support-aware correction for single-RGB object-level 3D placement using supplied camera calibration. Preserve existing results and user changes. Read repository instructions, current research reports, and source code before editing. Complete independent work autonomously, but request human input when annotation or an inaccessible resource is genuinely required. Do not claim annotation is independent if you generated and checked it yourself.

The main question is whether an observation-dependent support correction with explicit abstention improves object-center accuracy over no correction and the existing fixed 0.4 m child-ray control, on fresh room-aware scenes with exhaustive support labels. A positive result is desirable, not guaranteed. A reproducible negative result with a clear analysis is an acceptable final outcome. Do not touch the separate MUTE repository or the portfolio website. MUTE integration is outside the required scope.

Start with Phase 0 and create `KILO_PROGRESS.md`. After every phase, record completed tasks, changed files, exact commands, evidence, unresolved risks, and the next action. Mark a task complete only when its acceptance criteria are met. Use separate run directories, save raw outputs, and never overwrite historical protocols or results. Do not stop at a proposed plan: implement, validate, evaluate, and produce the final report within the decision gates below.

## 1. Scope and claim boundaries

Required deliverable: a working RGB-to-layout pipeline, one focused correction hypothesis, frozen evaluation, strong controls, failure analysis, measured resource use, and a final research report.

The model input is RGB plus supplied camera intrinsics; gravity/up direction may also be externally calibrated. State that requirement wherever the system is described. Sensor depth and annotated boxes are diagnostic/oracle inputs only. Ground-truth 3D centers, support labels, and annotation-derived dimensions must not enter automatic inference. Visible depth extents are not complete object dimensions; back-projected visible-point centers are not automatically amodal object centers.

No training a new VLM, replacing every backbone, full mesh reconstruction, learned hidden dimensions, outdoor generalization, or MUTE integration is required. Keep the current models and cached checkpoints unless a reproducibility failure forces a documented change. Keep VLM advisory as the default until evidence supports another policy.

## 2. Read these before implementation

- `README.md`, `RESEARCH_PLAN.md`, and `results/README.md`.
- `RESEARCH_STATUS_2026-10-02_ABLATION.md`, `RESEARCH_STATUS_2026-10-02_RADIAL.md`, and `RESEARCH_STATUS_2026-10-02_TRANSFER.md`.
- `RESEARCH_STATUS_2026-10-02_MASKS.md`, `RESEARCH_STATUS_2026-10-02_LAPTOP_TRANSFER.md`, `RESEARCH_STATUS_2026-10-04_BOTTLE_TRANSFER.md`, and `RESEARCH_STATUS_2026-10-04_VLM.md`.
- `scripts/run_inference.py` and the frozen support evaluators, including `evaluate_frozen_support_3d.py`, `evaluate_laptop_table6.py`, and `evaluate_bottle_table6.py`.
- `src/pipeline/child_ray_support.py`, `radial_support.py`, `support_constraints.py`, `contact_surface.py`, `generic_support_proposals.py`, `mask_contact.py`, `mask_support_gate.py`, and `vlm_support.py`.
- Geometry, SUN RGB-D adapter, detector matching, benchmark, and evaluation modules; relevant tests and viewer.

Current evidence to preserve accurately:

| Observation | Implication |
|---|---|
| On 20 scenes with oracle boxes, median center error was 0.301 m with sensor depth and 0.422 m with predicted depth | Predicted metric depth is a bottleneck; do not extrapolate tiny earlier samples |
| Weighted and hard height contact were effectively tied on the 15-scene study | Existing weights do not establish an uncertainty-method contribution |
| A simpler child-only ray control matched or beat the paired-ray method in the reported ablation | The fixed child-ray method is a mandatory strong baseline |
| Lamp transfer did not improve mean child-center error | Preserve negative category evidence |
| Two matched positive bottle pairs improved from 0.652 to 0.486 m mean error | Encouraging descriptive evidence, not general validation |
| VLM YES gating accepted 5/7 positives versus geometry's 7/7, with both accepting 0/2 negatives on the matched sample | Do not present VLM gating as an established improvement |
| Some negative scenes emitted other unverified support proposals | Paired specificity is not whole-scene precision |

Verify figures against their underlying JSON before copying them into the final report.

## 3. Phase 0: preserve and reproduce — October 4–5

- [ ] Read applicable `AGENTS.md` files. Record Git HEAD, branch, status, and initial diff summary. The checkout already contains modified and untracked research work; do not reset, discard, auto-stash, or sweep it into an unrelated commit.
- [ ] Record Python, package versions, CUDA, GPU, checkpoint revisions/hashes, available RAM/disk, and working dataset paths. Use the existing `.venv`; inspect before changing dependencies.
- [ ] Run `.\.venv\Scripts\python.exe -m unittest discover -s tests -v`. Save the result. Verify actual CLI arguments using `--help` before constructing commands.
- [ ] Reproduce one positive end-to-end scene and one ambiguous scene that should abstain, saving results to a new audit directory. Compare the positive radial shift to the stored frozen evaluation.
- [ ] Verify units, coordinate conventions, camera transformations, projected centers, mask resolution, instance IDs, and detection matching on real overlays.
- [ ] Create an inventory of historical selection manifests and previously inspected scenes. Include room/sequence/source metadata where available.
- [ ] Create `KILO_PROGRESS.md` and `docs/current_system_audit.md` with blockers and a reproducible baseline command.

Acceptance: existing tests pass or failures are diagnosed; the current pipeline reproduces its stated behavior; no historical work is lost; all input requirements are explicit. Synthetic tests validate code, not real-world accuracy.

## 4. Phase 1: freeze data and annotation protocol — October 5–8

Build development and final evaluation cohorts from eligible SUN RGB-D scenes. Existing inspected scenes may be development data but must not be represented as new final evidence.

- [ ] Select using metadata and a saved random seed before inspecting predictions or 3D errors. Target roughly 20 development scenes and 40 fresh final scenes, subject to availability and annotation time. These are planning targets, not a statistical power guarantee.
- [ ] Prefer room/sequence-disjoint grouping and verify it from source metadata. Index spacing alone does not demonstrate independence. Record unresolved grouping and use conservative source groups for analysis.
- [ ] Cover cups/mugs, laptops, and bottles with tables/desks/counters/shelves where annotations allow. Include real negatives, multiple plausible parents, occluded contact, and no eligible support. Keep lamp failures as historical context; add a fresh category-transfer cohort only if time permits.
- [ ] Freeze membership, eligibility rules, exclusions, matching rules, category vocabulary, and missing-data policy. Never exclude a scene because a method fails.
- [ ] Create an RGB annotation packet showing all detected candidates and annotated eligible objects. Label every emitted candidate relation within the declared vocabulary and every eligible ground-truth support pair needed for recall, not just one paired table. If new predictions introduce unlabeled candidates, flag them for blinded adjudication rather than treating them as negatives.
- [ ] Use explicit labels: visible physical support/contact, unsupported, uncertain/occluded, and unjudgeable. Record alternate supporting surfaces and ambiguity. A generic 'above' relation is insufficient.
- [ ] Keep annotators blind to predicted depth, correction outputs, method identity, and 3D error. Request human review; if possible, obtain a second review of at least a stratified subset and all disagreements. Save provisional automated labels separately and state their provenance.
- [ ] Separate physical parent/contact labeling from SUN RGB-D 3D center references. Validate repeated-class association; ambiguous matches remain documented, not silently resolved in a method's favor.
- [ ] Freeze and hash selection, label schema, labels, and scoring protocol. Any later amendment gets a version and reason.

Suggested artifacts: `results/support_v2/dev_selection.json`, `test_selection.json`, `annotation_protocol.md`, `labels.json`, and `evaluation_protocol.json`.

Acceptance: the final set is disjoint from development under the strongest verified grouping; eligible supports and emitted proposals have a defined scoring policy; test 3D references have not influenced method design. If exhaustive annotation is unavailable, explicitly narrow evaluation to annotated pairs and abandon whole-scene precision claims.

## 5. Phase 2: specify and implement one candidate — October 8–12

Choose one candidate on development data. Recommended direction: local support-surface evidence plus observation-dependent reliability, with child-only ray correction and conservative abstention. Inspect the existing local-contact implementation first; extending it is preferable to duplicating it.

Before coding, write `docs/support_method_v2.md` specifying:

- Exact target being corrected and its relation to the annotated amodal center.
- Inputs, equations, assumptions, reliability features, clipping, thresholds, and abstention reasons.
- Which quantities come from RGB/predicted depth versus development annotations.
- How perspective, tilted cameras, partial visibility, and child size affect the target.
- Why a parent centroid plus universal 0.4 m gap is inadequate, and what observed evidence replaces it.

Implementation tasks:

- [ ] Estimate the local parent surface from valid masked depth near the proposed contact; avoid background and child pixels. Use robust statistics or a locally fitted plane only where sufficient geometry exists.
- [ ] Keep size estimation honest: an image-height proxy is perspective-sensitive and does not recover hidden full height. Do not substitute test 3D dimensions. If amodal center offset cannot be inferred defensibly, retain a development-only prior with clear limitations or narrow the contribution.
- [ ] Build reliability from observables such as valid sample count, robust depth spread, mask/contact evidence, plane residual if used, and parent ambiguity. Depth spread is a heuristic reliability signal until calibration is validated.
- [ ] Fit any scale-to-uncertainty mapping or category prior on development data only. If there is insufficient calibration evidence, call the rule reliability-weighted rather than probabilistically calibrated.
- [ ] Reuse child-ray optimization, keeping the parent fixed. Log original/corrected centers, target, weights, residual, shift, evidence, and abstention reason. Guard rays nearly orthogonal to gravity and shifts behind the camera.
- [ ] Abstain on insufficient evidence or ambiguous parents. Do not silently fall back to unreported arbitrary correction.
- [ ] Add meaningful tests for tilted-camera geometry, positive depth, ray preservation, parent invariance, invalid depth, insufficient samples, ambiguity, caps, and uncertainty behavior.
- [ ] Demonstrate CLI/evaluator consistency on the same cached inputs. Correction disabled must reproduce the unchanged baseline.

Acceptance: a deterministic, finite, documented method runs on development scenes; no reference leakage; ablation switches isolate its components; the proposal is frozen before final scoring. If it fails development checks, retain the fixed control and prepare a diagnostic study instead of multiplying untested variants.

## 6. Phase 3: frozen evaluation and ablations — October 13–18

Required comparisons on identical inputs and matching:

| ID | Condition | Purpose |
|---|---|---|
| A | SAM lifting, no correction | Main baseline |
| B | Existing fixed 0.4 m child-ray correction | Strong simple control |
| C | Existing hard height/contact control | Prior control, with its assumptions explicit |
| D | Candidate local/size-aware target with constant reliability | Isolate target construction |
| E | Candidate with observation-dependent reliability | Main proposed condition |
| F | E without ambiguity/reliability abstention, where mathematically valid | Identify safety/coverage tradeoff |

Also retain box versus SAM lifting as a diagnostic. Compare geometry-only, VLM advisory, and VLM hard gating on the same candidate pool where budget permits. Advisory should leave geometry unchanged; verify that invariant. Reuse cached predictions so comparisons do not confound detector variation or loading behavior.

- [ ] Freeze configs, code/checkpoint hashes, seeds, primary metric, bootstrap unit, and all hyperparameters before looking at final 3D references.
- [ ] Diagnostic tracks: sensor depth + supplied boxes; predicted depth + supplied boxes; predicted depth + detected boxes/masks. Keep them separate. Do not imply oracle relations are automatically inferred.
- [ ] Primary estimate: paired scene-level difference in mean child-center error, candidate minus B, on the fixed baseline-matched eligible positive cohort. Abstention keeps the uncorrected center and remains in the cohort. Report A as an additional comparator. Average within scene before averaging scenes.
- [ ] Supplement with object-level median/P90 error, per-category results, parent-center error, harm rate and magnitude, contact residuals, and improvements/worsening counts. Contact residual alone cannot establish center accuracy.
- [ ] Report eligible-object coverage, complete-pair coverage, proposal precision/recall/F1, false proposals per scene, correction coverage, and abstention coverage/reasons. Publish denominators and uncertainty-label handling. Do not mix conditional centroid accuracy with all-scene detection success.
- [ ] Score negatives and ambiguous cases even if the method abstains. Report risk versus coverage using thresholds selected on development; final curves must not choose a new production threshold.
- [ ] Bootstrap paired differences by verified room/sequence group, or scene with an explicit dependence caveat. Save intervals and per-scene values. Define undefined metrics when denominators are zero.
- [ ] Produce every final score from machine-readable outputs with a summary script. Keep per-scene audit trails and partial-run failure counts.

October 18 decision: claim a correction gain only if evidence supports an improvement over B and A without concealing adverse categories, harm, or lost coverage. An interval crossing zero means the study is inconclusive at that sample size. A clear average gain driven by one source group needs qualification. If the candidate loses, report that and identify the strongest baseline; do not tune on the final set and relabel it held out.

## 7. Phase 4: resource profile, viewer, and reproducibility — October 19–22

- [ ] Profile a fixed multi-scene sample after warmup: per-stage and total latency, model-loading time separately, PyTorch allocated/reserved peaks, failures, and cold versus warm operation.
- [ ] Synchronize CUDA around GPU timings. Also sample total device/process memory when supported, documenting polling interval, other processes, and measurement limits. Do not call reserved PyTorch memory total VRAM.
- [ ] Preserve sequential loading on the approximately 8 GB GPU. Coordinate GPU use with the user/MUTE work; schedule long batches and save resumable artifacts. If no schedule exists, request one before a long competing run.
- [ ] Extend the existing Streamlit viewer only as needed: RGB, masks, depth, initial/corrected centers, support edges, alternate parents, raw VLM answers, confidence/reliability evidence, and abstention reason.
- [ ] Clearly distinguish visible point extents from inferred full boxes. Ground-truth overlays belong to evaluation mode.
- [ ] Provide one example command and one final evaluation command with actual supported flags, relative data paths, configs, and expected output schemas. Record software versions and checkpoint hashes; avoid unnecessary dependency upgrades.
- [ ] Test artifact reproduction from cached inputs without rerunning model inference. Do not claim fresh-install reproduction unless actually performed.

Acceptance: another operator can rerun scoring and inspect failures; figures agree with JSON; runtime claims specify exactly what is included.

## 8. Phase 5: final report and freeze — October 23–27

- [ ] Verify related work from primary papers/project repositories, particularly SpatialRGPT, Diorama, and relevant support-constrained reconstruction/refinement. Produce `docs/related_work_matrix.md` separating task, inputs, calibration, learned components, geometry, constraints, and evaluation. Conceptual comparisons must not become fabricated quantitative head-to-head results.
- [ ] Create `RESEARCH_REPORT_2026-10-27.md`: question, related work, method, dataset/annotation, protocol, main results, ablations, resource profile, failures, limitations, and reproduction commands.
- [ ] Include historical negative findings and distinguish exploratory, development, and final results. Never pool their samples into a single independent benchmark.
- [ ] Audit README for obsolete 'pending' statements and illustrative proposal claims. Update status and results references while preserving historical reports.
- [ ] If the VLM adds no value, describe the implemented system as geometry-led support correction with optional VLM diagnostics; explain the project's historical title. Do not invent a VLM-guided contribution.
- [ ] Freeze method by October 18 and final substantive experiments by October 22. After October 23, permit only correctness repairs with versioned reruns of affected results.
- [ ] Rerun appropriate tests, verify CLI examples, validate schemas and figure provenance, and create `FINAL_DELIVERY_CHECKLIST.md` with links to all required artifacts.

Final completion criteria: reproducible code and scoring; frozen sample and documented annotations; primary comparison against the strong child-ray baseline; separate accuracy/coverage/relation metrics; evidence-supported claims; measured resource limits; failure analysis; clear limitations. A positive novelty claim is not a completion requirement.

## 9. Fallbacks and escalation

- Annotation unavailable by October 8: continue code/development work, request human review, and shrink scope transparently. No independent relation-accuracy claim from unreviewed model labels.
- Too few new groups: publish a pilot with sample/independence limits; do not promise statistical certainty.
- Size/local-surface target unreliable by October 12: focus on conservative abstention and characterize fixed-prior failures. Treat novelty as unproven.
- VLM too slow or unhelpful: retain it as advisory and a diagnostic ablation; spend compute on the main geometry study.
- No final candidate gain by October 18: write an honest study of when support constraints help or harm. Preserve the baseline comparison and avoid a last-minute model sweep.
- Implementation blocker: record exact reproduction, affected phases, and independent work that can continue. Ask for missing access or decisions only when required.
- MUTE integration request: export a documented optional interface for object IDs, pixel boxes, masks, calibration, depth, centers, relations, and reliability. Integration itself must not delay the 3D deliverable.

## 10. Progress record template

For each work session, append to `KILO_PROGRESS.md`:

1. Date and phase.
2. Completed work and acceptance evidence.
3. Files changed and user changes preserved.
4. Exact commands and artifact paths.
5. Results with denominators and whether data were development or final.
6. Failures, assumptions, and required human input.
7. Next concrete task and deadline risk.

The first response to this assignment should report the repository audit and the first reproducibility result, then continue with the next authorized work. Do not replace implementation with repeated planning summaries.
