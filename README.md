# HinglishNano-V5: An Ultra-Compact Multi-Task Encoder for Code-Mixed Conversational Memory on Edge Devices

[![arXiv](https://img.shields.io/badge/arXiv-2026.xxxxx-b31b1b.svg)](https://github.com/unshakensoul17/HinglishNano_Reserach_paper)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/unsahkensoul18/Hinglishnano_V5)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-INT8%20%7C%20FP32-blue)](https://github.com/microsoft/onnxruntime)

**Author**: Akash Yaduwanshi (`aakashyaduwanshi0470@gmail.com`)  
**Hugging Face Hub**: [`unsahkensoul18/Hinglishnano_V5`](https://huggingface.co/unsahkensoul18/Hinglishnano_V5)  
**Paper PDF**: [`main.pdf`](main.pdf)

---

## 📌 Abstract & Research Question

**Research Question**: *How much conversational-memory functionality can be compressed into a single edge encoder under strict size and latency constraints without catastrophic retrieval degradation?*

**HinglishNano-V5** is a 7.77M-parameter multi-task transformer architecture (7.70 MB dynamic INT8 ONNX, 9.2 ms single-thread CPU $P_{50}$) that produces, in a single forward pass:
1. **384-Dimensional L2-Normalized Dense Retrieval Embedding**
2. **Memory Gating Head** (Store vs. Discard)
3. **Domain Classification Head** (6 conversation domains)
4. **Temporal Scope Head** (Past, Present, Future)
5. **BIO Named Entity Recognition Head** (8 target categories)

On a held-out full-corpus retrieval evaluation (300 queries against 1,245 candidate memories), HinglishNano INT8 retrieves the correct memory within Top-10 for **99.67% of queries** (299/300)—statistically indistinguishable from an `all-MiniLM-L6-v2` teacher fine-tuned on the identical split (100.0%; paired $\Delta = 1/300$ queries, bootstrap CI $[-1.0, 0.0]$)—while requiring **$2.9\times$ fewer parameters**, a **$2.8\times$ smaller INT8 footprint**, and **$1.9\times$ higher single-thread throughput** ($P_{50}$ 9.2 vs. 17.0 ms).

---

## 📊 Canonical Master Benchmark Table

All latency numbers: **Intel Xeon CPU @ 2.0 GHz (2 physical cores), single-thread (torch=1, ORT=1 pinned), batch=1, seq=64, tokenization excluded, warm-up 50 followed by $3 \times 300$ iterations**.

| Model Architecture | Params (M) | Format | Size (MB) | RAM RSS (MB) | CPU $P_{50}$ (ms) | CPU $P_{95}$ (ms) | Serial QPS | GLUE STS-B ($\rho$) | Full-Corpus R@1 | Full-Corpus R@10 | In-Domain Margin ($\Delta$) | Multi-Task Heads |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **HinglishNano V5 (INT8)** | **7.77** | **INT8** | **7.70** | **18.5** | **9.20** | **10.80** | **108.7** | **65.33** | **88.0%** | **99.7%** | **+0.175** | **5 Unified** |
| HinglishNano V5 (FP32) | 7.77 | FP32 | 29.83 | 42.1 | 8.30 | 9.20 | 120.5 | 65.30 | 88.0% | 99.3% | +0.176 | 5 Unified |
| `all-MiniLM-L6-v2` (FT-30ep) | 22.7 | INT8 | 21.90 | 35.8 | 17.00 | 19.20 | 58.8 | 86.20 | 92.0% | 100.0% | +0.439 | 1 (Emb only) |
| `all-MiniLM-L6-v2` (Zero-Shot) | 22.7 | INT8 | 21.90 | 35.8 | 16.80 | 19.00 | 59.5 | 86.72 | 59.3% | 86.7% | +0.089 | 1 (Emb only) |
| `BAAI/bge-small-en-v1.5` (Zero-Shot) | 33.4 | INT8 | 32.30 | 48.0 | 33.50 | 38.10 | 29.8 | 88.92 | 66.7% | 92.3% | +0.060 | 1 (Emb only) |
| `l3cube-pune/hing-roberta` | 278.0 | INT8 | 265.90 | 312.0 | 108.90 | 124.50 | 9.2 | 64.93 | 57.0% | 74.0% | +0.032 | 1 (Encoder) |
| `TinyBERT-4L-312D` | 14.4 | INT8 | 13.90 | 24.2 | 8.60 | 9.80 | 116.3 | 62.94 | 2.3% | 7.7% | -0.016 | 1 (Encoder) |

---

## 🔬 Multi-Task Head Performance (Held-Out Split: 1,648 Samples)

| Head / Sub-Task | Evaluated Metric | Measured Score | 95% Bootstrap CI | Notes |
| :--- | :--- | :---: | :---: | :--- |
| **Memory Gate Head** | Classification Accuracy / F1 | **90.47% / 93.70%** | [89.1, 91.8] / [92.5, 94.8] | Precision: 93.9%, Recall: 93.4% |
| **Chatter Probe** | False-Positive Rate (FPR) | **20.34%** | [12.0, 32.3] | Evaluative/affective chatter filter probe ($n=59$) |
| **Domain Head** | 6-Way Categorization Accuracy | **65.41%** | [63.1, 67.7] | Evaluated on memory-bearing samples |
| **Temporal Scope** | 3-Way Classification Accuracy | **86.71%** | [85.0, 88.4] | Past vs. Present vs. Future facts |
| **BIO NER Head** | In-Distribution Span-F1 | **65.30%** | [62.8, 67.8] | Template-bound disclaimer disclosed in paper |

---

## 🏛️ Model Architecture & Footprint Definitions

- **Parameter Count**: $7.77\text{ M}$ parameters
- **Layers**: 5 Transformer Blocks, $d=384$
- **Attention**: Grouped-Query Attention (GQA 6 Query heads : 2 KV heads)
- **Feed-Forward**: SwiGLU ($d_{\text{ffn}} = 640$)
- **Positional Encoding**: Rotary Position Embeddings (RoPE)
- **Normalization**: Pre-RMSNorm
- **Embedding Table**: Factorized Rank-96 projection table ($1.95\text{ M}$ parameters)
- **Vocabulary**: 9,022 BPE tokens (in a 20,000-entry provisioned embedding table; $\approx 1\text{ MB}$ unused capacity disclosed in paper)
- **Disk Artifacts**:
  - `unified_memory_engine_int8.onnx`: **7.70 MB** (Dynamic INT8: MatMul + Gather)
  - `unified_memory_engine_fp32.onnx`: **29.83 MB** (FP32 baseline)
  - `memory_models.zip`: **6.51 MB** (Compressed edge deployment bundle)

---

## 📁 Repository Structure

```
.
├── main.pdf                               # 📄 10-page compiled research paper (single source of truth)
├── main.tex                               # 📝 LaTeX source manuscript
├── LICENSE                                # ⚖️ MIT Open Source License
├── reproduce_all.py                       # 🚀 Master one-command reproduction pipeline
├── experiments/                           # 🔬 Modular experiment reproduction scripts
│   ├── 01_dataset_statistics.py          # 📊 Dataset distribution and train/test leakage audit
│   ├── 02_train.py                       # 🧠 Model architecture & PyTorch training definitions
│   ├── 03_multitask_eval.py              # 🎯 Multi-task head evaluation on held-out split
│   ├── 04_retrieval_eval.py              # 🔍 In-domain full-corpus retrieval & margin benchmark
│   ├── 05_stsb_eval.py                   # 🌐 GLUE STS-B zero-shot generalization test
│   ├── 06_latency.py                     # ⚡ Standardized single-thread CPU & RAM profiling
│   ├── 07_quantization.py                # 🗜️ Dynamic INT8 quantization parity & compression
│   ├── 08_ablation.py                    # 🧩 Multi-task vs separate models & component ablations
│   └── 09_scaling.py                     # 📈 Candidate pool scaling (20 to 1,245) & margin analysis
├── fig_frontier.pdf                       # 📈 Quality-efficiency frontier figure
├── fig_degradation.pdf                    # 📉 Scaling robustness figure
├── fig_margin.pdf                         # 📊 Hard-negative margin correlation figure
├── fig_latency.pdf                        # ⚡ Single-thread CPU latency figure
└── README.md                              # 📖 Comprehensive repository documentation
```

---

## 🚀 One-Command Reproducibility

To run the complete benchmark suite and reproduce all numbers reported in the paper:

```bash
python3 reproduce_all.py
```

Results are saved to `./results/*.json`.

---

## 📄 Compiling the Paper Locally

To compile `main.tex` into `main.pdf` using Tectonic:

```bash
tectonic main.tex
```

---

## 📚 Citation

```bibtex
@article{yaduwanshi2026hinglishnano,
  title   = {HinglishNano-V5: An Ultra-Compact Multi-Task Encoder for Code-Mixed Conversational Memory on Edge Devices},
  author  = {Yaduwanshi, Akash},
  journal = {arXiv preprint},
  year    = {2026},
  url     = {https://github.com/unshakensoul17/HinglishNano_Reserach_paper}
}
```
