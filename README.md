# VLM Quant — Document OCR Pipeline (Part 2)

**Shrink Florence-2 for document OCR without breaking quality** — five post-training quantization methods implemented from scratch in plain PyTorch.

![Phases A→D — complete function journey](https://github.com/Gaurav14cs17/VLM_Quant/raw/main/assets/nb02/wb_phases_abcd_journey.png)

---

## Overview

This repository is **Part 2 of 5** in the [Document OCR Pipeline](https://github.com/Gaurav14cs17/Document-OCR-Pipeline) series. It takes the Florence-2 OCR setup from Part 1 and walks through post-training quantization (PTQ) end to end — from naive int4 baselines to mixed-precision plans that beat them on real OCR tasks.

Everything is written **by hand in PyTorch**. No `auto-gptq`, `awq`, `bitsandbytes`, or `optimum-quanto`. You implement the math, not hide it behind APIs.

| Method | Paper | What it optimizes |
|--------|-------|-------------------|
| **GPTQ** | Frantar et al., 2023 | Output reconstruction via Hessian-aware column-wise quant |
| **AWQ** | Lin et al., 2023 | Activation-aware per-channel scaling before quant |
| **SmoothQuant** | Xiao et al., 2023 | Outlier migration between weights and activations |
| **SpinQuant** | Liu et al., ICLR 2025 | Learned orthogonal rotations before quant |
| **ConvRot** | Huang et al., arXiv 2512.03673 | Group-wise Regular Hadamard Transform (RHT) |

**Model:** [`microsoft/Florence-2-base-ft`](https://huggingface.co/microsoft/Florence-2-base-ft)  
**Task:** Document OCR (`<OCR_WITH_REGION>` detect)

---

## Why quantize?

A linear layer stores weight matrix **W ∈ ℝ^(O×I)**. In fp16 that is `2OI` bytes. With 4-bit symmetric quantization, storage drops roughly **4×** vs fp16 (plus small scale overhead). The hard part is choosing **Q** so the reconstruction error stays small **in the directions calibration data actually uses**.

This notebook pairs with [01 — Document OCR Pipeline](https://github.com/Gaurav14cs17/Document-OCR-Pipeline/blob/main/01_document_ocr_pipeline.ipynb). Same Florence-2 OCR setup — but here we shrink the model the right way.

---

## Pipeline

Run **Phases A → D** in order. Each phase has markdown (what/why) then code (do it).

| Phase | Stages | What you learn |
|-------|--------|----------------|
| **A — Baseline** | 8a–8c | Naive int4 everywhere → OCR baseline scorecard |
| **B — Survey** | 9a–9b | Layer inventory: shapes, dtypes, parameter counts |
| **C — Sensitivity** | 10a–10b | Rank layers by output MSE under calibration data |
| **D — Smart quant** | 11–14 | Mixed-precision plan → apply → beat Phase A |
| **E — Deep dives** | 15–16 | Side-by-side comparison of all five methods |

**Theory & building blocks (Stages 1–6):** symmetric quant, `QuantState`, drop-in `nn.Linear` replacements, and one stage per method.

**Load model (Stage 7):** Florence-2 + calibration document page.

![Phases A→D — complete function journey](https://github.com/Gaurav14cs17/VLM_Quant/raw/main/assets/nb02/wb_phases_abcd_journey.png)

---

## Quick start

### Local (recommended)

1. Clone this repo and open `ocr_pipeline_quant.ipynb`.
2. In the **config cell**, set:
   - `QUANT_METHOD` — `gptq` | `awq` | `smoothquant` | `spinquant` | `convrot`
   - `MIXED_PRECISION = True` — auto-pick int4 / int8 / fp16 per layer (recommended)
3. Run the install cell, then **Run all** (restart kernel if prompted after `transformers` install).

### Google Colab

The full series is also available on Colab via the [Document OCR Pipeline](https://github.com/Gaurav14cs17/Document-OCR-Pipeline) repo.

```python
QUANT_METHOD = "gptq"          # or awq, smoothquant, spinquant, convrot
MIXED_PRECISION = True         # int4/int8/fp16 per layer
TASK = "detect"                # detect | ocr
MODEL_ID = "microsoft/Florence-2-base-ft"
```

Turn on `RUN_ALL_METHODS_OCR = True` in Stage 16 to compare all five methods on full OCR.

---

## Requirements

Installed automatically by the notebook's first cell:

- Python 3.10+
- PyTorch, `transformers==4.49.0`, `accelerate`, `numpy`, `scipy`, `scikit-learn`, `pillow`, `matplotlib`
- **GPU recommended** for calibration and quantization passes

No external quantization libraries — all quant code is plain PyTorch.

---

## Project structure

```
VLM_Quant/
├── ocr_pipeline_quant.ipynb      # Main notebook (Phases A–E)
├── https://github.com/Gaurav14cs17/VLM_Quant/raw/main/assets/nb02/                  # Diagrams and walkthrough images
├── LICENSE                       # Apache 2.0
└── README.md
```

---

## Series

| Part | Topic |
|------|-------|
| [01 OCR](https://github.com/Gaurav14cs17/Document-OCR-Pipeline/blob/main/01_document_ocr_pipeline.ipynb) | Document OCR with Florence-2 |
| **02 Quant** (this repo) | Post-training quantization |
| [03 Mobile export](https://github.com/Gaurav14cs17/Document-OCR-Pipeline/blob/main/03_ocr_pipeline_mobile.ipynb) | Export for mobile |
| [04 Mobile complete](https://github.com/Gaurav14cs17/Document-OCR-Pipeline/blob/main/04_ocr_pipeline_mobile_complete.ipynb) | End-to-end mobile pipeline |
| [05 Production issues](https://github.com/Gaurav14cs17/Document-OCR-Pipeline/blob/main/05_mobile_production_issues.ipynb) | Production pitfalls |

## Diagrams

All images live in [`https://github.com/Gaurav14cs17/VLM_Quant/raw/main/assets/nb02/`](https://github.com/Gaurav14cs17/VLM_Quant/raw/main/assets/nb02/). The notebook renders **25 stage overview images** inline; **73 line-by-line walkthrough diagrams** are clickable links inside the notebook (look for 📎).

| Category | Examples |
|----------|----------|
| **Pipeline** | [Phases A→D journey](https://github.com/Gaurav14cs17/VLM_Quant/raw/main/assets/nb02/wb_phases_abcd_journey.png) · [All classes map](https://github.com/Gaurav14cs17/VLM_Quant/raw/main/assets/nb02/wb_all_classes_map.png) |
| **Stages 1–7** | [Building blocks](https://github.com/Gaurav14cs17/VLM_Quant/raw/main/assets/nb02/nb02_stage01_building_blocks.png) · [GPTQ](https://github.com/Gaurav14cs17/VLM_Quant/raw/main/assets/nb02/nb02_stage02_gptq.png) · [Load model](https://github.com/Gaurav14cs17/VLM_Quant/raw/main/assets/nb02/nb02_stage07_load_model.png) |
| **Phases A–E** | [Phase A](https://github.com/Gaurav14cs17/VLM_Quant/raw/main/assets/nb02/wb_phase_a_overview.png) · [Phase B](https://github.com/Gaurav14cs17/VLM_Quant/raw/main/assets/nb02/wb_phase_b_overview.png) · [Phase C](https://github.com/Gaurav14cs17/VLM_Quant/raw/main/assets/nb02/wb_phase_c_overview.png) · [Phase D](https://github.com/Gaurav14cs17/VLM_Quant/raw/main/assets/nb02/wb_phase_d_overview.png) |
| **Walkthroughs** | Browse all `wb_*_lines.png` files in [https://github.com/Gaurav14cs17/VLM_Quant/raw/main/assets/nb02/](https://github.com/Gaurav14cs17/VLM_Quant/raw/main/assets/nb02/) |

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).
