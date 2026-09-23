# ==============================================================================
# HINGLISH-NANO V5 vs BASELINES — INDUSTRY BENCHMARK REPRODUCIBILITY SCRIPT
# ==============================================================================

import os, sys, glob, json, time, random, platform, inspect
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr, pearsonr

torch.set_num_threads(1)                      # single-thread CPU for ALL models

import onnxruntime as ort
import psutil
from transformers import AutoTokenizer, AutoModel, PreTrainedTokenizerFast
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from tabulate import tabulate

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

def main():
    print("=" * 84)
    print("ENVIRONMENT DISCLOSURE")
    print("=" * 84)
    vm = psutil.virtual_memory()
    print(f"python {platform.python_version()} | torch {torch.__version__} | ort {ort.__version__}")
    print(f"CPU: {platform.processor() or platform.machine()} | phys/log cores: {psutil.cpu_count(logical=False)}/{psutil.cpu_count()} | RAM {vm.total/1e9:.1f} GB")
    print("Threading: torch=1 (pinned) | ORT intra=1/inter=1 (pinned)")

if __name__ == "__main__":
    main()
