# FinXAI Faithfulness Benchmark

**Explaining the Unexplainable? A Ground-Truth Benchmark of Classical, Concept-Based, and Mechanistic XAI on Financial Forecasters**

A fully reproducible benchmark that measures whether XAI explanations of
financial forecasting models can be *trusted*, by injecting features with
known ground-truth importance into a real, leakage-free 57-stock panel and
scoring 23 XAI methods across 16 ML/DL architectures against that ground
truth. The compiled paper is at [`paper/main.pdf`](paper/main.pdf).

> **TL;DR of the findings.** Of 23 catalogued XAI methods (classical
> attribution, rule/example-based, concept-based and intrinsic,
> gradient/propagation, graph, and mechanistic interpretability), 16 are
> scoreable against planted ground truth, and all 16 correctly rank the
> linear, nonlinear, and near-noise-floor weak planted signals above pure
> noise. TreeSHAP, KernelSHAP, and LIME agree strongly with each other but
> only weakly with permutation importance, which paired per-stock tests
> (McNemar, Wilcoxon) confirm is statistically significantly less reliable
> under this data's collinearity. Methods disagree sharply on *how* they
> split credit between a feature and its duplicate: roughly a quarter of the
> credit (TreeSHAP/KernelSHAP, ~0.24-0.25) versus a near-exact 50/50 split
> (GNNExplainer, 0.500), a measured instance of the disagreement problem.
> Every claim is stock-clustered (5000 whole-symbol bootstrap resamples) and
> reports an effective sample size (Neff = 2.55 at 57 co-moving stocks,
> mean pairwise rho = 0.381); results replicate across 2021/2022/2023 test
> years and a leave-10-stocks-out jackknife.

**Design highlights.** The test-bed's real predictive signal is validated to
be near zero (persistence and majority-class baselines are essentially
unbeaten), and this is used deliberately as a stress test: a faithful
explainer should report "no real signal here" while an unfaithful one
hallucinates importance. The ground-truth grid spans linear and nonlinear
planted signals, each at a strong and a near-noise-floor strength, plus an
exact and a partially collinear (rho = 0.8) duplicate feature. Null
baselines (TCAV vs. 100 random-concept CAVs; SAE and ablation-circuit
feature-shuffle nulls) separate real structure from artifacts of the
methods themselves.

## Repository layout
```
modal/                 # compute pipeline (runs on Modal.com)
  _common.py           # shared image, Volume, config-as-code (config.xai.yaml, profile switch)
  s1_data.py           # download FNSPID (prices + 23GB news) to a Volume
  s2_build.py          # Spark ingestion -> FinBERT+LM sentiment -> leakage-free panel + ground-truth features
  s3_experiments.py    # leakage demo, baselines, walk-forward, ablation, HAC/bootstrap stats
  s4_lstm.py           # GRU / LSTM / 1D-CNN / TCN / Transformer sequence models (GPU)
  s4b_gnn.py           # GNN over a cross-stock correlation graph
  s4c_gt_artifacts.py  # ground-truth construction audit artifacts
  s5_xai.py            # TreeSHAP / KernelSHAP / LIME / permutation / block-perm / ALE / Integrated Gradients
  s5b_rule_methods.py  # Anchors, counterfactuals, CEM, RISE
  s5c_concept_intrinsic.py  # TCAV, EBM shape functions, Global Attribution Mapping, rule surrogates
  s5d_gradient_cam.py  # LRP, Guided Backprop, DeconvNet, Grad-CAM family
  s5e_mechanistic.py   # sparse autoencoders + ablation-based circuit analysis (+ seed/fold sensitivity)
  s5f_graph.py         # GNNExplainer
  s5g_tcav_null.py     # TCAV vs. 100 random-concept CAVs (significance test)
  s5h_mechanistic_null.py  # SAE + ablation feature-shuffle null baselines
  s6_eda.py            # stationarity, autocorrelation, fat tails, cross-sectional correlation
  s7_faithfulness.py   # ground-truth faithfulness benchmark, stock-clustered CI + effective N
  s7b_faithfulness_robustness.py  # year-robustness (2021/2022/2023) + leave-10-stocks jackknife
  s8_notebooks.py      # execute the notebook suite on Modal with outputs embedded
make_tables.py         # metrics -> LaTeX tables + numbers.tex (single source of paper numbers)
notebooks/             # 27 executed notebooks: data/EDA, one per model, one per analysis
paper/                 # LaTeX source (main.tex, refs.bib, generated tables/figures)
results/               # metrics (JSON/CSV) + figures pulled from the Volume
```

## Reproducing (one command)
Requires a (free) [Modal](https://modal.com) account for compute and `tectonic`
for the paper. The public [FNSPID](https://huggingface.co/datasets/Zihan1004/FNSPID)
dataset is downloaded automatically. All parameters live in `config.xai.yaml`
(`profile: reviewer` for a fast small run, `profile: full` for the paper).

```bash
pip install modal && modal token new     # one-time compute auth
make reproduce                           # data -> panel -> experiments -> paper
```

`make help` lists granular targets; the Modal Volume caches data/artifacts so
each stage is idempotent and independently re-runnable:

```bash
make data          # download FNSPID (prices + 23GB news)
make build         # Spark ingest -> FinBERT sentiment -> leakage-free panel
make experiments   # model zoo, XAI suite, faithfulness + robustness, nulls
make results       # pull metrics/figures, regenerate LaTeX tables
make paper         # compile the anonymized PDF
make notebooks     # execute the notebook suite on Modal, outputs embedded
```

Every number in the paper is generated by `make_tables.py` into
`paper/numbers.tex` and `paper/tables/`; nothing is hand-typed.

## Methodological guarantees
- **Ground truth by construction.** Planted features with known importance
  (`gt_noise`, `gt_signal`, `gt_nonlinear`, weak variants, `gt_redundant_copy`,
  `gt_redundant_partial`), construction logged in `results/feature_manifest.json`.
- **No leakage.** Strictly chronological, embargoed walk-forward evaluation;
  all preprocessing fit on training folds only; the "R2 = 0.99" level-prediction
  artifact is reproduced and refuted as a labeled diagnostic, not a result.
- **Dependence-aware inference.** Stock-days are never treated as independent:
  symbol-clustered bootstrap (5000 whole-symbol resamples), effective sample
  size reported alongside every interval, Newey-West HAC where applicable.
- **Robustness.** Findings replicate across three held-out test years
  (2021/2022/2023) and a 200-draw leave-10-stocks-out jackknife; TCAV, SAE,
  and circuit-ablation results are tested against shuffle/random nulls.
- **Universe selection by data availability, not outcome.** 57 large-cap
  stocks chosen from 81 sector-stratified candidates with a minimum
  news-coverage floor (`results/metrics/universe_selection.json`).

## Project history
This repository grew out of an earlier project (FinSentImpact) that
reproduced and refuted a "R2 = 0.99" next-day forecasting claim as an
autocorrelation/leakage artifact and measured the marginal value of
financial-news sentiment under honest evaluation. That analysis survives as
the leakage demo and sentiment ablation inside the current pipeline
(`modal/s3_experiments.py`), and the near-zero-signal test-bed it exposed is
now the deliberate stress test at the heart of the benchmark. Planning
documents from both phases are kept in `_planning/`.
