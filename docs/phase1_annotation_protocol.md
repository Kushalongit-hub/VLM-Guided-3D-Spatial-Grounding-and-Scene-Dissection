# Phase 1 annotation protocol

## Goal

Produce relation labels that are independent of the proposed camera-ray correction, so the correction can be evaluated against an unbiased oracle. The protocol emphasizes reproducibility, explicit abstentions, and traceability back to the original SUN RGB-D annotation source.

## Scope

- **Development set**: `results/sunrgbd_phase1_development20_selection.json` — 20 laptop+desk scenes used for pipeline tuning and failure-mode inspection.
- **Final test set**: `results/sunrgbd_phase1_final40_selection.json` — 57 scenes (bottle+table 9, bottle+desk 17, bowl+table 14, laptop+table 20) frozen before any model inspection. These are disjoint from the development set and from all previously inspected project scenes.
- **Negative examples**: Scenes with multiple plausible parents, occluded contacts, or no eligible support are included as negative/abstention cases. They are not excluded from the test set.

## Definitions

- **Support relation**: The child object is physically contacted and structurally supported by the parent. This is stricter than spatial proximity.
- **Contact**: At least one pixel of the child mask touches a pixel of the parent mask in the projected segmentation. Contact through a third object does not count.
- **Footpoint**: The image-plane point on the parent mask nearest to the child centroid, used for camera-ray correction.
- **Abstention**: The annotator or pipeline cannot determine a unique, physically correct support relation. Record `support_label = "none"` or `"ambiguous"` with a reason.
- **Oracle**: The manually verified relation label. Predicted relations are compared against this label, not against heuristic proposals.

## Annotation procedure

1. **Load scene metadata**
   - Read the scene's `image.jpg`, `depth.npy`, `calib.txt`, `objects.json`, and `reference.json`.
   - Verify that the 3D annotation centers and camera intrinsics agree with the coordinate system documented in `src/pipeline/sunrgbd.py`.

2. **Identify candidate pairs**
   - For each scene, the selection JSON provides `candidate_child_class` and `candidate_parent_class`.
   - Use the SUN RGB-D 3D bounding boxes to identify all instances of these classes.
   - If multiple child instances exist, annotate each child separately.
   - If multiple parent instances exist, record all plausible parents with relative plausibility ranking.

3. **Record spatial relations**
   - For each candidate child, inspect the RGB-D image and decide:
     - `supported_by(parent_id)`: child is contacted and structurally supported by parent.
     - `supported_by(parent_id, occluded)`: contact is likely but occluded in the RGB view; mark `occluded = true`.
     - `none`: no support relation exists.
     - `ambiguous`: multiple plausible parents or unclear physical contact; record the reason.
   - Record the relation in the scene-level annotation file as a list of objects with `support_label` and optional `occluded` flag.

4. **Verify contact (optional depth check)**
   - Project the parent and child 3D boxes to image plane using `src/utils/camera.py`.
   - Check whether the child mask and parent mask overlap in the image.
   - If masks do not overlap but the relation is still claimed, mark as `occluded = true` and require additional justification.

5. **Record abstentions explicitly**
   - If the scene has multiple candidate parents and the annotator cannot decide, record all candidates with confidence estimates and mark the scene as `ambiguous`.
   - If no eligible parent exists, record `support_label = "none"` and `reason = "no_eligible_parent"`.
   - Abstentions are not failures; they are valid labels for relation precision/recall evaluation.

6. **Cross-check with original SUN RGB-D annotations**
   - Compare the annotated relations with the original `label.txt` 3D centers and extents.
   - Flag any inconsistency between the claimed support relation and the 3D geometry (e.g., child center above parent center without overlap).

## Output format

Each annotated scene produces a file `annotation.json` with the following schema:

```json
{
  "scene_id": "sunrgbd_XXXXXX",
  "selection_seed": 202,
  "candidate_child_class": "laptop",
  "candidate_parent_class": "desk",
  "relations": [
    {
      "child_id": "obj_0",
      "parent_id": "obj_1",
      "support_label": "supported_by",
      "occluded": false,
      "contact_evidence": "visible_mask_contact",
      "annotator_notes": ""
    }
  ],
  "abstained": false,
  "abstention_reason": "",
  "metadata_index": 1234,
  "sensor": "xtion",
  "sequence_name": "SUNRGBD/xtion/..."
}
```

Fields:
- `support_label`: one of `supported_by`, `none`, `ambiguous`.
- `occluded`: boolean; `true` if contact is inferred but not visible.
- `contact_evidence`: one of `visible_mask_contact`, `occluded_depth_gap`, `none`.
- `abstained`: boolean; `true` if the annotator could not produce any relation label.
- `abstention_reason`: one of `no_eligible_parent`, `multiple_plausible_parents`, `occluded_unverifiable`, `inconsistent_geometry`.

## Quality control

- **Two-pass annotation**: First pass records raw judgments. Second pass reviews edge cases (occluded contacts, multiple parents) without changing frozen labels.
- **Calibration**: Before annotating, the annotator reviews 5 pre-annotated scenes with known labels to calibrate the support definition.
- **Disagreement resolution**: If two annotators disagree on a scene, the scene is flagged for review and the final label is decided by a third annotator or by consensus.

## Relation to evaluation

- The `src/pipeline/evaluation.py` module compares predicted relations against these oracle labels.
- Relation precision/recall/F1 are computed only on scenes where `abstained = false`.
- Abstentions are counted separately and reported as `abstention_rate` per split.
- The evaluation does not penalize the pipeline for abstaining on scenes where the annotation protocol also records `abstained = true` or `support_label = "ambiguous"`.

## Phase 1 timeline

1. **Week 1**: Annotate the 20 development scenes. Use these to tune the contact-surface and support-gate thresholds.
2. **Week 2**: Freeze the final test annotations. No changes to test labels are permitted after this point.
3. **Week 3**: Run the full pipeline on the final test set with the frozen oracle. Record per-scene abstentions and relation errors.

## Notes

- The oracle labels are **not** derived from the proposed camera-ray correction. They are independent human judgments based on RGB-D inspection.
- Scenes with `abstained = true` or `support_label = "ambiguous"` are valid test cases. The pipeline should abstain on these, and the evaluation should reflect that.
- Do not infer full 3D object dimensions from visible depth points without explicit shape assumptions. The annotation records only relations, not full 3D extents.
