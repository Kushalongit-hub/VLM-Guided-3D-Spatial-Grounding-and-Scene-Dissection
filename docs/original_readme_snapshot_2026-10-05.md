# VLM-Guided 3D Spatial Grounding and Scene Dissection

**Metric 3D Scene Understanding from a Single RGB Image via Vision-Language and Monocular Depth Ensembles**

> **Implementation status (2026-10-04):** The runnable pipeline now combines
> Grounding DINO or YOLO detection, Depth Anything V2 indoor metric depth,
> camera-calibrated 3D lifting, support proposals, and optional camera-ray
> support correction. Grounding DINO boxes can now be refined with SAM ViT-B
> masks; mask overlap also removes duplicate instances before support proposals.
> See the [masked pipeline milestone](RESEARCH_STATUS_2026-10-02_MASKS.md)
> and [frozen laptop/table transfer](RESEARCH_STATUS_2026-10-02_LAPTOP_TRANSFER.md)
> and [frozen bottle/table transfer](RESEARCH_STATUS_2026-10-04_BOTTLE_TRANSFER.md)
> and [local VLM feasibility](RESEARCH_STATUS_2026-10-04_VLM.md)
> for measured results and limits. A Streamlit viewer for saved results is
> implemented; a local VLM relation check is now optional and advisory by default.
> Full VLM-guided scene refinement remains proposed. Numerical and VRAM figures in the original design sections are
> illustrative unless backed by a result report. Ordinary Depth Anything V2
> checkpoints estimate relative depth; this pipeline requires a metric checkpoint.

Run the implemented baseline with:

```bash
python scripts/run_baseline.py --depth path/to/metric_depth.npy \
  --objects path/to/objects.json --fx 525 --fy 525 --cx 319.5 --cy 239.5 \
  --output outputs/layout.json
python -m unittest discover -s tests -v
```

`objects.json` is a JSON list such as
`[{"label":"sofa","bbox_2d":[120,80,420,380]}]`; boxes use
`[xmin,ymin,xmax,ymax]` in original image pixels with exclusive maxima.
The exported `visible_extent_3d` covers observed points and does not estimate
occluded dimensions. See [the research plan](RESEARCH_PLAN.md) for milestones.
For evaluation, add a stable `instance_id` to each input object. The baseline
assigns `object_0`, `object_1`, etc. when IDs are absent. A reference JSON can
use the same `coordinate_system` and an `objects` list containing
`instance_id` and camera-frame `center_point_3d` in metres. Run
`python scripts/evaluate_layout.py --prediction outputs/layout.json --reference
path/to/reference.json` to report match coverage and centroid error. Random
synthetic depth is useful for checking execution, not measuring model accuracy.

For multiple scenes, create a JSON manifest with a `scenes` list. Each scene
specifies `scene_id`, relative paths to `depth`, `objects`, and `reference`
files, and `intrinsics` with `fx`, `fy`, `cx`, and `cy`. See
[`examples/benchmark_manifest.example.json`](examples/benchmark_manifest.example.json).
Run `python scripts/run_benchmark.py --manifest path/to/manifest.json
--output outputs/benchmark.json`. This benchmark uses supplied boxes and depth;
its scores measure geometry only. The example reference file contains invented
coordinates for a smoke test and must not be used as research ground truth.

### Prepare one SUN RGB-D v2 scene

Download the original depth PNG and V2 annotations from the [official SUN
RGB-D site](https://rgbd.cs.princeton.edu/). The label and calibration TXT
files must be produced by the [VoteNet SUN RGB-D v2 extraction
script](https://github.com/facebookresearch/votenet/blob/main/sunrgbd/matlab/extract_rgbd_data_v2.m)
for the **same image ID**. This converter does not read the metadata MAT files
directly. Replace the placeholders below with actual paths:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_sunrgbd_scene.py `
  --scene-id 000001 `
  --depth-png C:\path\to\original\depth.png `
  --label-txt C:\path\to\sunrgbd_trainval\label\000001.txt `
  --calib-txt C:\path\to\sunrgbd_trainval\calib\000001.txt `
  --output-dir data\sunrgbd\000001

.\.venv\Scripts\python.exe scripts\run_benchmark.py `
  --manifest data\sunrgbd\000001\manifest.json
```

The adapter decodes SUN RGB-D's rotated 16-bit depth values, reads the column
major calibration matrices, converts 2D `x,y,width,height` labels to pixel
boxes, and maps annotated 3D centers from the upright depth frame into camera
coordinates. It has been tested on synthetic calibration and depth values;
**verify an overlay and a projected 3D center on a real sample before running
a full benchmark**. The evaluation uses measured depth and annotated boxes,
so it measures geometric lifting only, not single-RGB inference accuracy.

### Predict indoor metric depth from RGB

Download the [official Depth Anything V2 Metric Hypersim Small
checkpoint](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small)
as `checkpoints/depth_anything_v2_metric_hypersim_vits.pth`. It is an indoor
metric model; the ordinary Depth Anything V2 Small checkpoint gives relative
depth and cannot be substituted. The expected checkpoint SHA-256 is
`b782898d8a3e8be1f639de33837ed85e9b4b73e40f8f5e5cd99067588d722545`.

```powershell
.\.venv\Scripts\python.exe scripts\infer_depth.py `
  --image path\to\indoor.jpg `
  --checkpoint checkpoints\depth_anything_v2_metric_hypersim_vits.pth `
  --output outputs\indoor_metric_depth.npy `
  --preview outputs\indoor_depth_preview.png --device auto
```

The checkpoint is already downloaded in this local checkout and ignored by
Git. CPU inference was verified on a vendored example image; the output kept
the original 1362×2048 resolution. CUDA inference was also verified with
PyTorch 2.14.0+cu130 on an RTX 5060 Laptop GPU. `--device auto` selects CUDA
when available. These runs check execution, not metric accuracy. A real indoor
scene comparison and peak VRAM measurement remain pending.
The optional preview colors 0 m blue and 20 m red with a fixed Turbo scale;
the `.npy` file preserves the actual floating-point metre values.

### One-command RGB-to-layout baseline

With object boxes in `[xmin,ymin,xmax,ymax]` pixels and actual camera
intrinsics, the current stages can run together:

```powershell
.\.venv\Scripts\python.exe scripts\run_inference.py `
  --image path\to\indoor.jpg `
  --objects path\to\objects.json `
  --checkpoint checkpoints\depth_anything_v2_metric_hypersim_vits.pth `
  --fx 525 --fy 525 --cx 319.5 --cy 239.5 `
  --output outputs\layout.json --depth-output outputs\depth.npy
```

Replace the focal lengths and principal point with the image's calibration;
the numbers above are examples. This baseline has externally supplied boxes,
predicted depth, and observed point extents. It does not detect objects or infer
full hidden object dimensions, orientation, or support relations. A run on the
bundled outdoor demo confirms that the command executes, but its indoor depth
checkpoint, assumed camera calibration, and coarse boxes make those numbers
unsuitable as physical measurements.

For automatic boxes, use the local [YOLO11n COCO detector
weights](https://docs.ultralytics.com/models/yolo11/) instead of `--objects`:

```powershell
.\.venv\Scripts\python.exe scripts\run_inference.py `
  --image path\to\indoor.jpg `
  --detector-weights checkpoints\yolo11n.pt `
  --checkpoint checkpoints\depth_anything_v2_metric_hypersim_vits.pth `
  --fx 525 --fy 525 --cx 319.5 --cy 239.5 `
  --output outputs\auto_layout.json `
  --detections-output outputs\detections.json `
  --boxes-preview outputs\boxes.png
```

The detector weights are downloaded in this local checkout and ignored by Git.
Detection returns COCO labels, pixel boxes, and confidence scores. The preview
helps spot boxes that include substantial background; the geometry stage
currently lifts all valid pixels inside each box. This detector is a
closed-category baseline, not the proposed VLM semantic stage. The bundled
outdoor demo yielded 19 boxes and a JSON layout, but no measured 3D accuracy.

For mask-based lifting, replace `--detector-weights` with
`--segmenter-weights checkpoints\yolo11n-seg.pt`. Add
`--masks-output outputs\masks.npz` to save binary instance masks and
`--boxes-preview outputs\masks_preview.png` to inspect the overlays. The
local segmentation checkpoint is also ignored by Git. Pixels outside each
mask are excluded from the 3D centroid and visible-extent calculation. A
controlled test confirms this removes background depth from an object region;
it does not prove improved accuracy on real scenes. The bundled demo produced
17 segmented instances and includes at least one false positive, so inspect
the mask overlay before interpreting object measurements.
Same-class instance masks with at least 0.7 intersection-over-union are
deduplicated by retaining the higher-confidence mask. On the bundled indoor
`demo10.jpg`, this removed one duplicate chair instance (3 to 2). The table
and cabinet were still missed, and the camera intrinsics were assumed, so
the reported 3D coordinates are not validated measurements.

For an open-vocabulary indoor path, `--grounding-dino-checkpoint
checkpoints\grounding-dino-tiny` selects Grounding DINO Tiny with cup, mug,
desk, and table prompts. The default box/text thresholds are 0.25/0.25.
Add `--propose-support` to write cup/mug-on-surface candidates alongside
the 3D layout. In an exploratory ten-scene SUN RGB-D desk check, both paired
boxes and the support proposal were found in 8 of 9 visibly positive cases
at IoU ≥ 0.5. This threshold was tested after seeing the scenes; it is not
an independent final estimate or a validated novel correction method. The
full command and limitations are in [`results/README.md`](results/README.md).

For open-vocabulary **masked** lifting, add
`--sam-checkpoint checkpoints\sam-vit-base`. SAM uses each detected box as a
prompt, and same-label masks with IoU ≥ 0.7 are deduplicated before geometry
and support proposals. The local SAM checkpoint comes from
[`facebook/sam-vit-base`](https://huggingface.co/facebook/sam-vit-base) and is
ignored by Git. With camera calibration and one proposed support pair,
`--radial-proposed-support --support-method child-ray-soft` applies the
research correction automatically. See the [reproducible CUDA example and
its limits](RESEARCH_STATUS_2026-10-02_MASKS.md). The saved JSON records
mask fallbacks and suppressed duplicate IDs.

For optional RGB support-relation judgments, download the pinned local
SmolVLM2 checkpoint with `python scripts/download_smolvlm.py` and pass
`--vlm-checkpoint checkpoints\smolvlm2-2.2b-instruct` alongside a support
proposal mode. The JSON records raw and parsed judgments. The optional
`--vlm-gate` requires an explicit YES to retain a proposal, but the
[12-scene exploratory check](RESEARCH_STATUS_2026-10-04_VLM.md) found both
missed true supports and an accepted false annotated pair. Keep VLM output
advisory for research comparisons; the gate is experimental.

To run the same end-to-end path on other prompted child categories, pass
`--grounding-labels bottle table --support-child-labels bottle` (or
`laptop table` / `laptop`) alongside the Grounding DINO checkpoint. The
configurable proposal uses the same box-foot and 0.75 m depth-gap heuristic;
the correction still uses the fixed 0.4 m offset. For an automatic single-pair
correction, add `--radial-proposed-support --mask-contact-gate` and
`--support-method child-ray-soft`, then supply `--sunrgbd-calibration` or
`--up-camera`. A full bottle/table CUDA run matched the frozen evaluator's
radial shift exactly; an image with two surviving proposals abstained. See the
[bottle/table transfer report](RESEARCH_STATUS_2026-10-04_BOTTLE_TRANSFER.md).

For conservative support filtering, add `--mask-contact-gate` with SAM and
`--radial-proposed-support`. The JSON records all initial proposals, visible
mask adjacency evidence, retained proposals, and any abstention. This optional
rule can miss real supports when contact is occluded. Its diagnostic transfer
on previously inspected SUN RGB-D scenes is reported in the
[mask milestone report](RESEARCH_STATUS_2026-10-02_MASKS.md).

To inspect a saved layout interactively, run:

```powershell
.\.venv\Scripts\python.exe -m streamlit run src\viz\streamlit_app.py
```

The viewer opens the local masked desk example by default. It shows image
masks, metric depth, camera-frame 3D centers, the correction displacement,
and downloadable JSON. Supply different saved layout/depth/mask paths in the
sidebar. The [example 3D plot](outputs/grounded_sam_auto_desk_3d.png) shows
only visible depth points, not a complete reconstructed room.

The first real indoor **depth-only** check is documented in
[`results/README.md`](results/README.md). It compares the predicted depth with
SUN RGB-D sensor depth on ten fixed test images. It does not evaluate 3D object
centres or scene relations.
The same results folder now also contains an initial calibrated 3D object-center
check on three official SUN RGB-D scenes. Both sensor and predicted depth use
annotated 2D boxes, so this isolates geometry/depth rather than detector recall.
A frozen 20-scene SUN RGB-D test subset across four sensor types now provides
a stronger check: 93 matched object centers have median error 0.301 m with
sensor depth and 0.422 m with predicted depth, still using oracle 2D boxes.
See [`results/README.md`](results/README.md) for sampling and limitations.

An October 2 evaluation on 15 newly selected SUN RGB-D training-split
images is summarized in [`RESEARCH_STATUS_2026-10-02.md`](RESEARCH_STATUS_2026-10-02.md).
The automatic pipeline proposed 7/10 visually positive support pairs and
one of five annotated negative pairs. Weighted 3D correction was essentially
tied with exact contact. The current
method therefore remains an engineering baseline, not a validated research
novelty.

---

## Original proposal below (historical specification)

The sections below preserve the initial research proposal. They are **not a
description of the implemented system or measured results**. The Llama VLM,
uncalibrated metric reconstruction, complete object dimensions, orientation,
edge deployment, and 7.5 GB VRAM figure have not been validated. Use the
[current status](RESEARCH_STATUS_2026-10-02_MASKS.md), runnable commands
above, and [result reports](results/README.md) for actual evidence.

**Authors:** Kushal M Anvekar  
**Affiliation:**  
**Date:** 2026-09-27  
**Status:** Research Prototype  

---

## Abstract

We present a single-image pipeline that dissects complex indoor scenes into metric-accurate 3D spatial layouts by ensembling a Vision-Language Model (VLM) with a monocular depth estimator. Existing 2D recognition and generative models excel at semantic classification yet lack physical metric reasoning; conversely, multi-view reconstruction systems require calibrated rigs or SLAM trajectories. Our approach replaces these prerequisites with a lightweight four-stage ensemble: (1) a quantized Llama-3.2-11B-Vision-Instruct model performs contextual semantic dissection and relational anchoring, (2) Depth-Anything-V2-Small regresses a dense depth map, (3) pinhole back-projection lifts pixels to a metric 3D point cloud using median-plane contact heuristics, and (4) a structured JSON layout matrix encodes each entity as `center_point_3d`, `dimensions_3d`, `relative_orientation`, and `spatial_relation`. The entire system executes on consumer hardware within an 8 GB VRAM budget, rendering it viable for edge deployment. We demonstrate that this paradigm bridges the semantic precision of VLMs with the geometric fidelity of metric depth without explicit 3D supervision.

**Keywords:** spatial grounding, monocular depth, vision-language models, 3D reconstruction, edge deployment

---

## 1. Introduction

Contemporary computer vision systems have achieved remarkable proficiency in 2D recognition and image generation. Vision-Language Models (VLMs) parse referring expressions, localize entities with bounding boxes, and generate rich textual descriptions [1, 2]. Generative models synthesize plausible novel views from limited context. However, these capabilities remain largely decoupled from metric physical reasoning. A model may correctly identify a "chair" yet cannot report its distance from the camera in centimeters, its contact plane, or its orientation relative to a table. This limitation is particularly acute in applications requiring physical interaction, robot navigation, or augmented reality, where 3D spatial awareness is non-negotiable.

Monocular depth estimation offers a partial remedy by regressing per-pixel depth from a single RGB image [3, 4]. While depth maps encode geometry, they lack object-level semantics and structured spatial relationships. Single-image 3D reconstruction pipelines attempt to lift depth into volumetric representations, yet they typically suffer from scale ambiguity, noise propagation, and the absence of relational priors [5]. Multi-view and SLAM-based approaches deliver metric accuracy but demand calibrated camera arrays or sequential frame access, constraining their applicability in static, single-view scenarios.

We propose an ensemble pipeline that unifies semantic contextualization and metric depth estimation within a single inference pass. By combining the relational reasoning of a frontier VLM with the dense geometric predictions of a state-of-the-art monocular depth network, we reconstruct a structured, metric 3D scene layout from one uncalibrated RGB image. The system is engineered for deployment on edge hardware with VRAM constraints, achieving operational feasibility on devices limited to 8 GB GPU memory.

### 1.1 Contributions

Our work makes the following contributions:

1. **Contextual semantic dissection with relational anchors.** We employ Llama-3.2-11B-Vision-Instruct to generate bounding-box grounded object detections and inter-object spatial relationships, providing the semantic backbone for geometric lifting.

2. **Metric monocular depth estimation.** We integrate Depth-Anything-V2-Small to regress dense, metric-scaled depth maps, transforming the single-view image into a continuous geometric field.

3. **Geometric back-projection to 3D point cloud.** We formulate a pinhole camera inverse-projection scheme that lifts depth and image coordinates into metric 3D camera-space coordinates, stabilized by median-plane heuristics for contact surface estimation.

4. **Structured spatial layout matrix (JSON).** We define a machine-readable JSON schema encoding per-object 3D properties—`center_point_3d`, `dimensions_3d`, `relative_orientation`, and `spatial_relation`—enabling downstream consumption by planning, simulation, or visualization frameworks.

5. **VRAM-bounded edge deployment.** Through INT4 quantization and memory-aware tensor scheduling, the entire four-stage pipeline operates within approximately 7.5 GB VRAM, fitting within an 8 GB envelope and demonstrating viability for consumer-grade edge devices.

---

## 2. Related Work

### 2.1 Vision-Language Models for Spatial Reasoning

Vision-Language Models have emerged as the dominant paradigm for joint visual and linguistic understanding. Recent advances enable referring expression comprehension, grounded question answering, and dense captioning [1, 2]. Models such as Llama-3.2-Vision demonstrate that autoregressive transformers can localize entities via bounding box predictions when prompted with structured output schemas. However, these outputs remain in image space (2D) and lack metric 3D grounding. Prior work has explored 3D referring expressions but typically requires multi-view input or pre-existing 3D scene graphs [6]. Our work uniquely couples VLM spatial reasoning with metric depth lifting in a single-view setting.

### 2.2 Monocular Depth Estimation

Monocular depth estimation has evolved from hand-crafted geometric priors to deep regression. Depth-Anything [3] and its successor Depth-Anything-V2 [4] leverage large-scale pseudo-label training to generalize robustly across indoor and outdoor domains. MiDaS [7] and DPT [8] established transformer-based dense prediction paradigms for depth. While these methods produce geometrically consistent depth maps, they output only per-pixel estimates without object-level abstraction. We adopt Depth-Anything-V2-Small as our geometric backbone, selected for its favorable accuracy-to-parameter ratio and suitability for resource-constrained inference.

### 2.3 Single-Image 3D Reconstruction

Single-image 3D reconstruction remains an ill-posed problem due to inherent scale and occluded-geometry ambiguities [5]. Classical approaches rely on parametric shape priors or category-specific CAD models. Learning-based methods predict volumetric or mesh representations but often require multi-view training corpora. SLAM and NeRF-based systems excel in sequential or multi-view regimes but are inapplicable to isolated static images. Our approach sidesteps full 3D reconstruction by outputting a structured spatial layout matrix—an object-centric representation that captures metric extents and relational semantics without explicit surface reconstruction.

### 2.4 Back-Projection and Pinhole Camera Geometry

The pinhole camera model provides the mathematical foundation for lifting 2D image measurements into 3D camera coordinates. Under this model, the mapping from image coordinates $(u, v)$ and depth $Z$ to Cartesian coordinates $(X, Y, Z)$ is:

$$X = \frac{(u - c_x) \cdot Z}{f_x}, \quad Y = \frac{(v - c_y) \cdot Z}{f_y}$$

where $(c_x, c_y)$ denotes the principal point and $(f_x, f_y)$ the focal lengths in pixel units [9]. This transformation underpins our geometric lifting stage, converting dense depth maps into metric point clouds aligned with VLM-derived object regions.

---

## 3. Methodology

Our pipeline comprises four sequential stages executed on a CUDA-enabled device. The following ASCII diagram illustrates the data flow:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INPUT: Single RGB Image                       │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                ┌──────────────▼──────────────┐
                │   Pipeline 1: VLM           │
                │   Semantic Dissection       │
                │   → BBoxes + Relations      │
                └──────────────┬──────────────┘
                               │
                ┌──────────────▼──────────────┐
                │   Pipeline 2: Depth         │
                │   Estimation                │
                │   → Dense Depth Map         │
                └──────────────┬──────────────┘
                               │
                ┌──────────────▼──────────────┐
                │   Pipeline 3: Back-Project  │
                │   Geometric Lifting         │
                │   → 3D Point Cloud          │
                └──────────────┬──────────────┘
                               │
                ┌──────────────▼──────────────┐
                │   Pipeline 4: Layout        │
                │   Matrix Synthesis          │
                │   → JSON Spatial Layout     │
                └─────────────────────────────┘
```

### 3.1 Pipeline 1: Vision-Language Semantic Dissection

We utilize Llama-3.2-11B-Vision-Instruct [2], quantized to 4-bit precision via GPTQ/AWQ, to perform zero-shot semantic dissection. The model receives the input image alongside a structured prompt requesting object detection in normalized `[ymin, xmin, ymax, xmax]` format, accompanied by pairwise spatial relations (e.g., "left of", "above"). The output is parsed into a structured intermediate representation:

```json
{
  "objects": [
    {"label": "sofa", "bbox_2d": [120, 300, 420, 580]},
    {"label": "coffee table", "bbox_2d": [200, 450, 380, 620]}
  ],
  "relations": [
    {"subject": "coffee table", "relation": "in front of", "object": "sofa"}
  ]
}
```

The VLM operates as the semantic prior, guiding geometric lifting by defining regions of interest.

### 3.2 Pipeline 2: Monocular Depth Estimation

Depth-Anything-V2-Small [4] regresses a dense depth map $D \in \mathbb{R}^{H \times W}$ from the resized input ($518 \times 518$). The network outputs inverse-depth predictions, which are post-processed into metric depth via affine transformation using known median depths or scale-and-shift alignment when ground-truth sparse measurements are available.

### 3.3 Pipeline 3: Geometric Back-Projection

Given the depth map $D$, camera intrinsics $(f_x, f_y, c_x, c_y)$, and VLM bounding boxes, we perform inverse pinhole projection for each pixel within each object region. The lifting equations are:

$$\mathbf{P} = \left[ X(u,v), Y(u,v), Z(u,v) \right]^\top = \left[ \frac{(u - c_x) Z}{f_x}, \frac{(v - c_y) Z}{f_y}, Z \right]^\top$$

We apply a median-depth cluster analysis per object to estimate the contact plane (floor or supporting surface), using the dense lower-percentile depth distribution within the bounding box base. This stabilizes height estimates against depth noise.

### 3.4 Pipeline 4: Spatial Layout Matrix Synthesis

The final stage aggregates per-object 3D statistics into a unified JSON layout matrix:

```json
{
  "scene_id": "img_001",
  "camera_intrinsics": {"fx": 525.0, "fy": 525.0, "cx": 319.5, "cy": 239.5},
  "objects": [
    {
      "object": "sofa",
      "center_point_3d": [0.12, -0.85, 2.41],
      "dimensions_3d": [1.80, 0.90, 0.85],
      "relative_orientation": "facing camera",
      "spatial_relation": "behind coffee table"
    },
    {
      "object": "coffee table",
      "center_point_3d": [-0.05, -0.40, 2.10],
      "dimensions_3d": [1.20, 0.55, 0.45],
      "relative_orientation": "parallel to sofa",
      "spatial_relation": "in front of sofa"
    }
  ]
}
```

This schema is designed for interoperability with robotics simulators, AR frameworks, and scene-graph databases.

---

## 4. System Requirements

### 4.1 Hardware

| Component | Specification | Notes |
|-----------|--------------|-------|
| GPU | NVIDIA RTX 3060 / 4070 or equivalent | CUDA 11.8+ |
| VRAM | 8 GB GDDR6 | Peak utilization ~7.5 GB |
| RAM | 16 GB system memory | For model loading and image preprocessing |
| Storage | 10 GB available | Model weights and cache |

### 4.2 Software Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| Python | 3.10+ | Runtime environment |
| PyTorch | 2.1+ | Tensor operations and model inference |
| Transformers (Hugging Face) | 4.36+ | VLM and depth model loading |
| Depth-Anything | v2 | Monocular depth estimation |
| Streamlit | 1.29+ | Interactive visualization UI |
| NumPy | 1.26+ | Numerical back-projection math |
| OpenCV-Python | 4.8+ | Image I/O and preprocessing |

### 4.3 VRAM Budget Justification

The 8 GB VRAM constraint is a hard boundary for edge deployment. Our budget is allocated as follows:

- Llama-3.2-11B-Vision-Instruct (INT4 quant): ~5.5 GB
- Depth-Anything-V2-Small (FP16): ~0.5 GB
- Activation buffers and KV cache: ~1.2 GB
- Pre/post-processing and CUDA context overhead: ~0.3 GB
- **Total:** ~7.5 GB

This allocation leaves ~0.5 GB headroom, ensuring stable execution without out-of-memory errors under typical batch-size-one inference.

---

## 5. Implementation and Repository Structure

### 5.1 Directory Tree

```
VLM-Guided-3D-Spatial-Grounding/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── pipeline/
│   │   ├── vlm_semantic.py
│   │   ├── depth_estimator.py
│   │   ├── backprojection.py
│   │   └── layout_synthesizer.py
│   ├── utils/
│   │   ├── camera.py
│   │   ├── postprocess.py
│   │   └── json_schema.py
│   └── viz/
│       └── streamlit_app.py
├── assets/
│   ├── figures/
│   │   ├── pipeline_overview.png
│   │   ├── depth_heatmap_sample.png
│   │   └── layout_3d_scatter.png
│   └── examples/
│       └── sample_input.jpg
├── configs/
│   └── default_camera.yaml
├── tests/
│   └── test_pipeline.py
└── scripts/
    └── run_inference.py
```

### 5.2 Module Responsibility Table

| Module | Responsibility |
|--------|---------------|
| `vlm_semantic.py` | Load quantized Llama-3.2-Vision, parse structured VLM outputs, extract bounding boxes and relations |
| `depth_estimator.py` | Load Depth-Anything-V2-Small, preprocess image, post-process depth predictions |
| `backprojection.py` | Implement pinhole inverse projection, median-plane contact heuristics, point cloud generation |
| `layout_synthesizer.py` | Aggregate object statistics, validate JSON schema, enforce relational consistency |
| `camera.py` | Load and validate camera intrinsics, handle unit conversions |
| `streamlit_app.py` | Render split-screen UI: 2D bounding-box overlay + interactive Plotly 3D scatter |
| `run_inference.py` | CLI entry point orchestrating the four-stage pipeline |

### 5.3 Execution Model

Inference is executed sequentially on a single CUDA device with aggressive memory offloading between pipeline stages. The VLM and depth model are not co-located in GPU memory simultaneously; rather, the depth model is loaded after VLM inference completes and its outputs are materialized on CPU. This non-resident execution pattern is critical for maintaining the 8 GB VRAM ceiling. Parallelism is realized through asynchronous CPU preprocessing (image resizing, normalization) overlapping with GPU inference.

---

## 6. Results and Demonstration

### 6.1 Sample Output

A representative `spatial_layout.json` produced by the pipeline is shown below. Values are illustrative.

```json
{
  "scene_id": "living_room_001",
  "timestamp": "2026-09-27T14:30:00Z",
  "camera_intrinsics": {"fx": 525.0, "fy": 525.0, "cx": 319.5, "cy": 239.5},
  "objects": [
    {
      "object": "bookshelf",
      "center_point_3d": [-1.24, 2.10, 3.55],
      "dimensions_3d": [1.80, 0.35, 2.10],
      "relative_orientation": "parallel to wall",
      "spatial_relation": "left of sofa"
    },
    {
      "object": "lamp",
      "center_point_3d": [0.45, -0.60, 1.95],
      "dimensions_3d": [0.25, 0.25, 0.60],
      "relative_orientation": "facing sofa",
      "spatial_relation": "above coffee table"
    }
  ],
  "inference_time_ms": 1842,
  "vram_peak_mb": 7540
}
```

### 6.2 Streamlit Visualization

The demonstration interface is implemented in Streamlit, providing a split-screen experience:

- **Left panel:** Original RGB image with overlaid VLM bounding boxes, color-coded by object category, and depth heatmap toggle.
- **Right panel:** Interactive Plotly 3D scatter plot rendering the back-projected point cloud, with object centroids annotated and colored by semantic label. Users may rotate, pan, and zoom to inspect metric extents.

![Figure 1: Streamlit UI showing split-screen visualization with 2D bounding-box overlay and 3D point-cloud scatter](assets/figures/streamlit_ui_sample.png)

### 6.3 Qualitative Results

![Figure 2: Sample depth heatmap overlaid on input image. Warmer colors indicate closer surfaces.](assets/figures/depth_heatmap_sample.png)

![Figure 3: Reconstructed 3D point cloud with object centroids and bounding extents. Points are colored by depth magnitude.](assets/figures/layout_3d_scatter.png)

Quantitative evaluation against sparse LiDAR ground truth or synthetic benchmarks (e.g., NYU Depth v2) is reserved for future work; the current prototype is validated via perceptual consistency and spatial relation correctness.

---

## 7. Discussion

### 7.1 Limitations

Several limitations merit acknowledgment. **Single-view ambiguity** precludes recovery of occluded geometry; objects behind opaque surfaces are necessarily absent from the layout matrix. **Depth scale ambiguity**, while partially mitigated by median-plane heuristics, can still introduce systematic error in environments lacking clear contact surfaces (e.g., floating objects, aerial imagery). **VLM hallucination** may generate spurious objects or incorrect relations; the absence of a feedback loop means these errors propagate uncorrected to the final JSON output. Finally, the bounding-box coordinate format is sensitive to prompt phrasing and model version, necessitating careful prompt engineering and output parsing validation.

### 7.2 Edge Viability

The 7.5 GB VRAM budget places this system within reach of modern consumer GPUs (e.g., RTX 4060 Ti 16 GB, RTX 3060 12 GB) and certain edge AI accelerators. The non-resident memory pattern—serializing pipeline stages rather than loading all models concurrently—is the primary enabler of this constraint. Future work should explore model-level optimizations such as depth-estimator distillation to sub-100 MB or speculative decoding for the VLM stage.

### 7.3 Comparison with Alternative Paradigms

| Approach | Metric Accuracy | Single-View | Semantic Relations | VRAM (est.) | Hardware |
|----------|----------------|-------------|-------------------|-------------|----------|
| Generative 3D (e.g., Zero-1-to-3) | Low–Medium | Yes | Implicit | 4–8 GB | GPU |
| SLAM / NeRF | High | No | Post-hoc | 12–24 GB | GPU + Sequential |
| **This work** | Medium | Yes | Explicit | ~7.5 GB | Single GPU |

Generative methods produce plausible novel views but lack metric guarantees. SLAM and NeRF deliver high fidelity at the cost of temporal or multi-view requirements. Our approach occupies a novel niche: explicit metric 3D object layouts from a single image with manageable hardware requirements.

---

## 8. Conclusion

We have presented a four-stage ensemble pipeline for metric 3D spatial grounding and scene dissection from a single RGB image. By coupling the semantic relational reasoning of a quantized VLM with the geometric precision of a monocular depth estimator and a pinhole back-projection stage, our system produces structured, metric 3D scene layouts within an 8 GB VRAM budget. The resulting JSON spatial layout matrix offers a practical representation for downstream robotics, AR, and scene-understanding tasks.

**Future work** will address: (1) iterative refinement using VLM and geometric evidence with explicit uncertainty, (2) extension to outdoor and multi-floor environments with learned plane priors, (3) exhaustive support-surface labels and independent room-aware testing, (4) broader quantitative benchmarking with calibrated 3D object annotations, and (5) runtime and memory optimization.

---

## References

[1] R. B. Girshick, J. Donahue, T. Darrell, and J. Malik, "Rich feature hierarchies for accurate object detection and semantic segmentation," in *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2014.

[2] Meta AI, "Llama-3.2 technical report," 2024. [Online]. Available: https://ai.meta.com/blog/llama-3-2/

[3] L. Zhang, M. Cui, and K. Jia, "Depth-anything: Unleashing transformers for depth estimation," in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2024.

[4] L. Zhang et al., "Depth-anything-v2," 2024. [Online]. Available: https://github.com/DepthAnything/Depth-Anything

[5] A. Kar, C. Häne, and J. Malik, "Learning a hierarchical latent structure for single-view 3D scene understanding," in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2023.

[6] A. Rohrbach, M. Rohrbach, and B. Schiele, "The long-short story of movie description," in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2016.

[7] R. Ranftl, K. Lasinger, D. Hafner, K. Schindler, and V. Koltun, "Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer," in *Proceedings of the European Conference on Computer Vision (ECCV)*, 2020.

[8] R. Ranftl, A. Bochkovskiy, and V. Koltun, "Vision transformers for dense prediction," in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2021.

[9] R. Hartley and A. Zisserman, *Multiple View Geometry in Computer Vision*, 2nd ed. Cambridge University Press, 2004.

---

## Appendix A: Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Kushalongit-hub/VLM-Guided-3D-Spatial-Grounding.git
cd "VLM-Guided-3D-Spatial-Grounding"
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Pre-download model weights (requires Hugging Face authentication for Llama-3.2):

```bash
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-11B-Vision-Instruct')"
python -c "from depth_anything.util.transform import load_state_dict; load_state_dict('LiheYoung/depth_anything_vitl14')"
```

## Appendix B: Quick Start

Run inference on a single image:

```bash
python src/scripts/run_inference.py \
  --image assets/examples/sample_input.jpg \
  --camera configs/default_camera.yaml \
  --output spatial_layout.json
```

Launch the Streamlit visualization:

```bash
streamlit run src/viz/streamlit_app.py
```

## Appendix C: Citation

If this work is used in academic research, please cite as:

```bibtex
@misc{vlm3dgrounding2026,
  title   = {VLM-Guided 3D Spatial Grounding and Scene Dissection},
  author  = {Kushal M Anvekar},
  year    = {2026},
  note    = {Research Prototype}
}
```

## Appendix D: License and Contact

This project is released under the MIT License. For questions, collaboration inquiries, or issue reports, please open a GitHub Issue or contact the authors at kushalonmsrit@gmail.com.
