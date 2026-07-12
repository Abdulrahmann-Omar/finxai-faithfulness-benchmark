#!/usr/bin/env bash
# One-command reproduction of the FinSentImpact re-evaluation.
# Requires: a Modal account (`pip install modal && modal token new`) and tectonic
# (or pdflatex) for the paper. All heavy compute runs on Modal.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT/modal"

echo "== 1. download FNSPID (prices + 23GB news) =="
modal run s1_data.py::download
modal run s1_data.py::download_nasdaq

echo "== 2. build leakage-free panel =="
modal run s2_build.py::build_news     # Spark ingestion of the 23GB news file
modal run s2_build.py::finbert        # FinBERT sentiment (GPU)
modal run s2_build.py::assemble       # alignment + features + targets

echo "== 3. experiments (independent stages run concurrently) =="
modal run s3_experiments.py::leakage_demo &
modal run s3_experiments.py::diagnostics &
modal run s3_experiments.py::main &
modal run s4_lstm.py::run &
modal run s6_eda.py::run &
wait
modal run s3_experiments.py::stats
modal run s5_xai.py::run

echo "== 4. pull results and build the paper =="
cd "$ROOT"
modal volume get finsent-data results/figures  ./results/  || true
modal volume get finsent-data results/metrics  ./results/  || true
modal volume get finsent-data build/panel.parquet ./results/ || true
modal volume get finsent-data build/feature_manifest.json ./results/ || true
python make_tables.py
cp results/figures/*.png paper/figures/
cd paper && tectonic main.tex && echo "PDF: paper/main.pdf"
