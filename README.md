# HinglishNano-V5: A 7.7 MB Multi-Task Encoder with Deployment-Grade Retrieval Fidelity for Code-Mixed Hindi–English

[![arXiv](https://img.shields.io/badge/arXiv-2026.xxxxx-b31b1b.svg)](https://github.com/unshakensoul17/HinglishNano_Reserach_paper)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/unsahkensoul18/Hinglishnano_V5)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-INT8%20%7C%20FP32-blue)](https://github.com/microsoft/onnxruntime)

**Author**: Akash Yaduwanshi (`aakashyaduwanshi0470@gmail.com`)  
**Hugging Face Hub**: [`unsahkensoul18/Hinglishnano_V5`](https://huggingface.co/unsahkensoul18/Hinglishnano_V5)  
**Paper PDF**: [`main.pdf`](main.pdf)

---

## 📌 Abstract

**HinglishNano-V5** is a 7.77M-parameter multi-task transformer architecture (7.70 MB dynamic INT8, 8.76 ms single-thread CPU $P_{50}$) that produces, in a single forward pass:
1. **384-Dimensional L2-Normalized Dense Retrieval Embedding**
2. **Memory Gating Head** (Store vs. Discard)
3. **Domain Classification Head** (6 conversation domains)
4. **Temporal Scope Head** (Past, Present, Future)
5. **BIO Named Entity Recognition Head** (8 target categories)

On a held-out full-corpus retrieval evaluation (300 queries against 1,245 candidate memories), HinglishNano INT8 retrieves the correct memory within Top-10 for **99.7% of queries**—statistically indistinguishable from a fine-tuned `all-MiniLM-L6-v2` (100.0%)—while using **$2.9\times$ fewer parameters**, a **$2.8\times$ smaller footprint**, and **$1.9\times$ higher single-thread throughput**, executing all 5 tasks simultaneously on-device without cloud LLM dependencies.

---

## 📊 Comprehensive Benchmark Results

### 1. In-Domain & Cross-Lingual Retrieval vs. Industry Baselines (Single-Thread CPU)

| Model Architecture | Parameters | Precision | Size (MB) | CPU P50 (ms) | QPS | GLUE STS-B ($\rho$) | In-Domain Margin ($\Delta$) | Multi-Task Heads |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `l3cube-pune/hing-roberta` | 278.0M | FP32 | 1,060.7 MB | 146.79 ms | 6.8 | 64.93% | $+0.339$ | 1 (Encoder only) |
| `BAAI/bge-small-en-v1.5` | 33.4M | FP32 | 127.3 MB | 41.85 ms | 23.7 | **88.92%** | $+0.255$ | 1 (Embedding only) |
| `all-MiniLM-L6-v2` (SBERT) | 22.7M | FP32 | 86.6 MB | 20.09 ms | 49.3 | 86.72% | $+0.344$ | 1 (Embedding only) |
| `TinyBERT-4L-312D` | 14.4M | FP32 | 54.7 MB | 9.77 ms | 100.4 | 62.94% | $+0.215$ | 1 (Encoder only) |
| **HinglishNano V5 (FP32)** | 7.77M | FP32 | 29.83 MB | **7.88 ms** | **125.8** | 65.30% | **+0.489** | **5 Heads Unified** |
| **HinglishNano V5 (INT8)** | **7.77M** | **INT8** | **7.70 MB** 🎯 | **8.76 ms** ⚡ | **111.4** 🚀 | **65.33%** | **+0.485** 🏆 | **5 Heads Unified** 🏆 |

### 2. Multi-Task Memory Head Accuracy (1,648 Held-Out Samples)

| Sub-Task | Metric | V4 Baseline | HinglishNano V5 (Ours) | Delta ($\Delta$) |
| :--- | :---: | :---: | :---: | :---: |
| **Memory Gate** | Accuracy / F1 | 72.0% / 79.6% | **90.47% / 93.7%** | $+18.47\%$ |
| **Domain Categorization** | Top-1 Accuracy | 34.9% | **65.41%** | $+30.51\%$ |
| **Temporal Scope** | Top-1 Accuracy | 64.3% | **86.71%** | $+22.41\%$ |
| **Dense Vector Retrieval** | Top-10 Recall / MRR@10 | 58.0% | **99.67% / 92.5%** | $+41.67\%$ |
| **In-Domain Margin** | Separation Margin | $-0.023$ | **$+0.485$** | $+0.508$ |

---

## 🏛️ Model Architecture Specifications

- **Layers**: 5 Transformer Blocks
- **Hidden Dimension ($d$)**: 384
- **Attention**: Grouped-Query Attention (GQA 6 Query heads : 2 KV heads)
- **Feed-Forward Network**: SwiGLU ($d_{\text{ffn}} = 640$)
- **Positional Encoding**: Rotary Position Embeddings (RoPE)
- **Normalization**: Pre-RMSNorm
- **Embedding Table**: Factorized Rank-96 Projection ($1.95\text{ M}$ parameters)
- **Vocabulary**: 9,022 BPE tokens trained specifically on code-mixed Hinglish
- **Export Format**: ONNX (INT8 dynamic quantization: MatMul + Gather)

---

## 📁 Repository Structure

```
.
├── main.pdf                               # 📄 10-page compiled research paper
├── main.tex                               # 📝 LaTeX source manuscript
├── fig_frontier.pdf                       # 📈 Quality-efficiency frontier figure
├── fig_degradation.pdf                    # 📉 Scaling robustness figure
├── fig_margin.pdf                         # 📊 Hard-negative margin correlation figure
├── fig_latency.pdf                        # ⚡ Single-thread CPU latency figure
├── run_kaggle_empirical_benchmark.py       # 🔬 Complete reproducibility benchmark script
└── README.md                              # 📖 Repository documentation
```

---

## 🚀 Compiling the Paper Locally

To compile `main.tex` into `main.pdf` using Tectonic or XeLaTeX:

```bash
# Using Tectonic (zero-config, auto-downloads packages)
tectonic main.tex

# Or using XeLaTeX
xelatex main.tex
```

---

## 📚 Citation

```bibtex
@article{yaduwanshi2026hinglishnano,
  title   = {HinglishNano-V5: A 7.7 MB Multi-Task Encoder with Deployment-Grade Retrieval Fidelity for Code-Mixed Hindi--English},
  author  = {Yaduwanshi, Akash},
  journal = {arXiv preprint},
  year    = {2026},
  url     = {https://github.com/unshakensoul17/HinglishNano_Reserach_paper}
}
```
