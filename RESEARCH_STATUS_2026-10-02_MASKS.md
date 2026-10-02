# Grounding DINO + SAM milestone — 2 October 2026

**Deadline: 27 October 2026.** The prototype now runs an automatic indoor RGB-to-3D support path on CUDA: Grounding DINO Tiny finds cup/desk boxes, SAM ViT-B segments each box, mask overlap suppresses duplicate detections, Depth Anything V2 Metric predicts depth, calibrated projection produces object centers, the conservative support heuristic proposes a pair, and child-only camera-ray correction adjusts the cup. A Streamlit viewer inspects the resulting masks, depth and 3D layout. This is a working pipeline milestone; a broadly valid research contribution has not yet been demonstrated.

## End-to-end run

On SUN RGB-D scene `sunrgbd_007816`, the initial detector returned one desk and two overlapping cup boxes. The cup masks had **0.809 intersection-over-union**. Mask deduplication kept the higher-confidence cup, leaving one automatically proposed cup/desk relation. The [output layout](outputs/grounded_sam_auto_desk_layout.json), [mask overlay](outputs/grounded_sam_auto_desk_preview.png), and [stage profile](results/grounded_sam_auto_cuda_profile.json) are saved. The cup's error against the annotated 3D center was **0.344 m before correction** and **0.107 m after correction**; the desk center error was **0.204 m**. These figures describe this selected demonstration image and were read after building the path. They are not a held-out accuracy estimate.

The run took **8.12 s total** on the local CUDA GPU. The detection plus SAM stage peaked at **2.10 GB PyTorch allocated** and **2.49 GB reserved**; the depth stage peaked at **0.39 GB allocated**. These are one-run PyTorch measurements, excluding non-PyTorch allocations and other GPU processes. Model execution is sequential; peak stage memory is not the sum of stage peaks.

```powershell
.\.venv\Scripts\python.exe scripts\run_inference.py `
  --image data\sunrgbd_official\radial_desk7\sunrgbd_007816\image.jpg `
  --grounding-dino-checkpoint checkpoints\grounding-dino-tiny `
  --sam-checkpoint checkpoints\sam-vit-base `
  --checkpoint checkpoints\depth_anything_v2_metric_hypersim_vits.pth `
  --fx 520.532 --fy 520.7444 --cx 277.9258 --cy 215.115 `
  --radial-proposed-support --support-method child-ray-soft `
  --sunrgbd-calibration data\sunrgbd_official\radial_desk7\sunrgbd_007816\calib.txt `
  --masks-output outputs\grounded_sam_auto_desk_masks.npz `
  --depth-output outputs\grounded_sam_auto_desk_depth.npy `
  --boxes-preview outputs\grounded_sam_auto_desk_preview.png `
  --output outputs\grounded_sam_auto_desk_layout.json `
  --profile-output results\grounded_sam_auto_cuda_profile.json --device cuda
```

To inspect the saved output interactively, run
`.\.venv\Scripts\python.exe -m streamlit run src\viz\streamlit_app.py`
from the repository root. The viewer loads the demo paths by default and
provides an RGB mask overlay, a depth preview, a rotatable 3D center/visible
point plot, object coordinates, and a JSON download. The exported
[3D plot](outputs/grounded_sam_auto_desk_3d.png) is a static example. Its
point cloud contains only surfaces visible in the input image.

The local `checkpoints/sam-vit-base` directory contains the `facebook/sam-vit-base` Hugging Face model files. It is ignored by Git; another machine must download those weights separately. SAM masks are clipped to each detector box during 3D lifting. An empty SAM mask inside a box falls back to the box region and records the instance ID. Same-label masks with IoU ≥ 0.7 are deduplicated by detector confidence; any suppressed IDs are recorded in the layout JSON.

## Controlled, exploratory geometry comparisons

The previously run YOLO11n-seg 20-scene benchmark already contains matched box and mask center errors. A [paired analysis](results/sunrgbd_heldout20_mask_ablation.json) of its **15 class-aware matches across 13 scenes** found mean error **0.585 m with boxes** and **0.526 m with masks**; masks improved 10/15. The image-bootstrap interval for mask minus box was **−0.230 to +0.059 m**. Chair masks improved considerably in four matches, while bed masks worsened on average. This is a post-hoc analysis of previously inspected data.

On the eight previously scored lamp/table scenes, Grounding DINO matched **8 lamps and 7 tables** at box IoU ≥ 0.5. With the **same detections and predicted depth** in both arms, mean object-center error was **0.537 m with boxes** and **0.398 m with SAM masks**; masks improved 11/15 matched regions. By class, mask minus box was **−0.291 m for lamps** and **+0.033 m for tables**. The [per-scene report](results/sunrgbd_lamp_table8_grounded_sam_ablation.json) shows two lamps with very large box errors that drive much of the gain. The scene-bootstrap interval for all 15 objects is **−0.255 to −0.037 m**, but the scenes had already been viewed for correction work and independence is not established.

The interaction with support correction differs by category. On **3 of 4** positive lamp/table pairs with both objects detected, mean lamp error fell from **1.023 m with boxes** to **0.443 m with SAM masks**. The fixed 0.4 m child-ray correction then slightly worsened mask-based lamp error to **0.453 m**. See [lamp interaction](results/sunrgbd_lamp_table8_mask_support_interaction.json). On **11 previously scored** positive cup/mug/table pairs with both objects detected, mask-based child-only correction averaged **0.458 m** child error versus **0.566 m** with box-based correction, **0.650 m** with masks and no correction, and **0.661 m** with boxes and no correction. The [per-scene table results](results/sunrgbd_support_radial15_sam_pairs.json) and [paired summary](results/sunrgbd_support_radial15_sam_summary.json) are post-hoc; nearby frames in the table sample make naive image bootstraps optimistic. These comparisons use visually annotated positive relations for correction and oracle boxes only to match predictions for scoring, not for inference.

## Research interpretation

This result narrows the research problem. Better object masks can remove a large source of region-depth contamination, but the fixed support correction remains category dependent. An optional **visible mask contact gate** now filters candidate pairs before correction. It checks whether parent-mask pixels occur within 8 vertical pixels of the child's lower contour; contact fraction must be greater than zero. This is image adjacency, not proof of physical support. Run `--mask-contact-gate` with `--radial-proposed-support` and `--sam-checkpoint`. When no unique proposal survives, the layout records `abstained_no_unique_visible_contact_proposal` and skips correction.

The [diagnostic transfer summary](results/sunrgbd_mask_contact_transfer_summary.json) combines two previously inspected 15-scene sets. Among annotation-matched pairs with both boxes detected, the fixed rule retained **13/18 positive** pairs and **0/5 negative** pairs. Five true supports were missed. This denominator excludes missing detections, other proposals in the same image, and pairs without a clear annotation. It is not a held-out precision or recall estimate. The [protocol](results/mask_contact_transfer_protocol.json) records the rule and code hash before applying it to the second set. In the full end-to-end run on scene `sunrgbd_005307`, the known false mug/desk candidate was rejected, but a different cup/desk candidate was accepted and corrected; the gate therefore does **not** establish whole-scene abstention or relation correctness. The [gated layout](outputs/grounded_sam_gated_negative_005307.json) records both candidates and their mask evidence.

A research contribution would require a specific correction or uncertainty mechanism that improves on these heuristics under a newly frozen, room-disjoint test with positive and negative relation labels, missed-detection accounting, and object/parent/contact metrics. Grounding DINO + SAM itself is established prior work ([Grounded SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything)). The VLM reasoning stage from the original proposal remains unimplemented.

An annotation-metadata-only audit after excluding prior selections found remaining test-split candidates in other categories: 10 uniquely paired laptop/table frames across roughly 9 index groups, 32 lamp/night-stand frames across roughly 8 groups, and 8 bottle/table frames across roughly 7 groups. These are **candidate counts**, not confirmed support relations or room-disjoint sample counts. The current detector prompts and support heuristic are tuned to cup/mug and table/desk, so a credible next transfer test needs a predeclared category expansion and RGB-only relation labels before any 3D scoring. The audit read metadata only; those candidate images were not used in this report.
