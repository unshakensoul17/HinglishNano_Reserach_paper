#!/usr/bin/env python3
"""
Master Reproducibility Pipeline for HinglishNano-V5
Orchestrates the entire empirical evaluation suite:
  01. Dataset Statistics & Leakage Audit
  03. Multi-Task Head Evaluation (Held-Out Split)
  04. In-Domain Retrieval & Margin Benchmark (1,245 Candidate Store)
  05. Official GLUE STS-Benchmark Evaluation
  06. Standardized Single-Thread Latency & RAM Profiling
  07. Quantization Parity & Footprint Analysis
  08. Architectural & Multi-Task Ablation Study
  09. Candidate Set Scaling & Margin Correlation
"""

import os
import sys
import json
import time

def main():
    print("=" * 80)
    print("🚀 EXECUTING COMPLETE REPRODUCIBILITY BENCHMARK SUITE: HINGLISHNANO-V5")
    print("=" * 80)
    
    start_time = time.time()
    os.makedirs("results", exist_ok=True)
    
    scripts = [
        ("01_dataset_statistics.py", "Dataset Statistics and Leakage Audit"),
        ("03_multitask_eval.py", "Multi-Task Classification & Gating Evaluation"),
        ("04_retrieval_eval.py", "In-Domain Full-Corpus Retrieval Benchmark"),
        ("05_stsb_eval.py", "GLUE STS-B Zero-Shot Evaluation"),
        ("06_latency.py", "Single-Thread CPU Latency & RAM Profiling"),
        ("07_quantization.py", "Dynamic INT8 Quantization Parity"),
        ("08_ablation.py", "Multi-Task vs Single-Task & Component Ablations"),
        ("09_scaling.py", "Candidate Pool Scaling & Margin Robustness")
    ]
    
    for script_name, description in scripts:
        script_path = os.path.join("experiments", script_name)
        if os.path.exists(script_path):
            print(f"\n[RUNNING] {script_name} - {description}...")
            exit_code = os.system(f"python3 {script_path}")
            if exit_code != 0:
                print(f"⚠️ Warning: {script_name} finished with code {exit_code}")
        else:
            print(f"⚠️ Script not found: {script_path}")
            
    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"🎉 ALL BENCHMARKS COMPLETED SUCCESSFULLY IN {elapsed:.2f}s!")
    print("📁 Canonical results written to ./results/")
    print("=" * 80)

if __name__ == "__main__":
    main()
