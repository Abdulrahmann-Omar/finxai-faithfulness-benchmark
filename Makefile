# FinSentImpact - reproducible pipeline (MLOps-style single entry point).
# Heavy compute runs on Modal.com; the paper builds with tectonic.
#
#   make reproduce     # EVERYTHING from scratch: data -> panel -> experiments -> paper
#   make all           # experiments -> results -> paper (assumes data+panel built)
#   make help          # list targets
#
# Granular targets let you re-run one stage without redoing the rest; the Modal
# Volume caches data and artifacts between runs, so stages are idempotent.

SHELL := /bin/bash
MODAL := modal run
VOL   := finsent-data
.DEFAULT_GOAL := help

.PHONY: help setup data build experiments results paper all reproduce clean check

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	 awk 'BEGIN{FS=":.*?## "}{printf "  \033[1m%-12s\033[0m %s\n",$$1,$$2}'

check:  ## Verify prerequisites (modal auth, tectonic)
	@command -v modal >/dev/null || { echo "modal not found: pip install modal && modal token new"; exit 1; }
	@modal app list >/dev/null 2>&1 || { echo "modal not authenticated: modal token new"; exit 1; }
	@command -v tectonic >/dev/null || echo "warning: tectonic not found (paper target needs it)"
	@echo "prerequisites OK"

setup:  ## Install local Python deps (for notebooks/paper tooling)
	pip install -r requirements.txt

data: check  ## Download FNSPID (prices + 23GB news) to the Modal Volume
	cd modal && $(MODAL) s1_data.py::download
	cd modal && $(MODAL) s1_data.py::download_nasdaq

build: check  ## Build the leakage-free panel (Spark ingest -> FinBERT -> features)
	cd modal && $(MODAL) s2_build.py::build_news
	cd modal && $(MODAL) s2_build.py::finbert
	cd modal && $(MODAL) s2_build.py::assemble

experiments: check  ## Run all experiments (independent stages concurrently) + stats/XAI
	cd modal && \
	  $(MODAL) s3_experiments.py::leakage_demo & \
	  $(MODAL) s3_experiments.py::diagnostics & \
	  $(MODAL) s3_experiments.py::main & \
	  $(MODAL) s4_lstm.py::run & \
	  $(MODAL) s4_lstm.py::run_lstm & \
	  $(MODAL) s4_lstm.py::run_cnn1d & \
	  $(MODAL) s4_lstm.py::run_tcn & \
	  $(MODAL) s4_lstm.py::run_transformer & \
	  $(MODAL) s4b_gnn.py::run & \
	  $(MODAL) s6_eda.py::run & \
	  wait
	cd modal && $(MODAL) s3_experiments.py::stats
	cd modal && $(MODAL) s5_xai.py::run
	cd modal && \
	  $(MODAL) s5_xai.py::treeshap_by_model & \
	  $(MODAL) s5_xai.py::agnostic_by_model & \
	  $(MODAL) s5_xai.py::ig_by_model & \
	  $(MODAL) s5b_rule_methods.py::run & \
	  $(MODAL) s5c_concept_intrinsic.py::run & \
	  $(MODAL) s5d_gradient_cam.py::run & \
	  $(MODAL) s5e_mechanistic.py::run & \
	  $(MODAL) s5f_graph.py::run & \
	  wait
	cd modal && $(MODAL) s5_xai.py::consolidate
	cd modal && $(MODAL) s5_xai.py::agreement_matrix
	cd modal && $(MODAL) s7_faithfulness.py::run
	cd modal && $(MODAL) s5e_mechanistic.py::run_seed_fold_sensitivity
	cd modal && \
	  $(MODAL) s5g_tcav_null.py::run & \
	  $(MODAL) s5h_mechanistic_null.py::run & \
	  $(MODAL) s7b_faithfulness_robustness.py::run & \
	  wait

results:  ## Pull metrics/figures/panel from the Modal Volume and regenerate tables
	modal volume get $(VOL) results/figures ./results/ || true
	modal volume get $(VOL) results/metrics ./results/ || true
	modal volume get $(VOL) build/panel.parquet ./results/ || true
	modal volume get $(VOL) build/feature_manifest.json ./results/ || true
	python make_tables.py
	cp results/figures/*.png paper/figures/

paper: results  ## Compile the anonymized PDF
	cd paper && tectonic main.tex && echo "PDF -> paper/main.pdf"

notebooks: results  ## Execute the notebook suite on Modal (not locally) with outputs embedded
	cd modal && $(MODAL) s8_notebooks.py::run_all

all: experiments results paper notebooks  ## experiments -> results -> paper -> notebooks

reproduce: data build all  ## FULL reproduction from scratch (one command)
	@echo "Done. Paper at paper/main.pdf, notebooks executed in notebooks/"

clean:  ## Remove local build artifacts (keeps Modal Volume + source)
	rm -f paper/*.aux paper/*.log paper/*.out paper/*.bbl paper/*.blg
	rm -rf results/figures results/metrics
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
