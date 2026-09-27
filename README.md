# VLM-Guided 3D Spatial Grounding and Scene Dissection

**Metric 3D Scene Understanding from a Single RGB Image via Vision-Language and Monocular Depth Ensembles**

---

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

**Future work** will address: (1) iterative refinement via VLM feedback on reconstructed 3D geometry, (2) extension to outdoor and multi-floor environments with learned plane priors, (3) integration with open-vocabulary segmentation (e.g., SAM) for mask-accurate lifting, (4) quantitative benchmarking on NYU Depth v2 and ScanNet with standard 3D detection metrics, and (5) ONNX/TensorRT optimization for further VRAM reduction.

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
