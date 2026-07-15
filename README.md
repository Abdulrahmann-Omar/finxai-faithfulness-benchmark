# FinSentImpact — Leakage-Free, Interpretable Re-evaluation of News-Driven Multi-Stock Forecasting

A fully reproducible pipeline that (1) reproduces and refutes the "R²≈0.99"
next-day forecasting claim as an autocorrelation/leakage artifact, and (2)
rigorously measures the marginal value of financial-news sentiment under honest,
dependence-aware evaluation.

> **TL;DR of the findings.** On FNSPID, predicting next-day *price levels* with a
> random split gives R²≈0.999 — but a naive persistence baseline scores the same,
> and the identical model on *returns* gets R²<0. Under embargoed walk-forward
> evaluation across 12 large-cap stocks, no model beats naive baselines on
> returns, directional accuracy is at chance, and adding FinBERT / Loughran–McDonald
> sentiment changes accuracy by a statistically insignificant amount. Four XAI
> methods (SHAP, permutation, ALE, Integrated Gradients) confirm sentiment's
> negligible contribution.

> **Repo status.** This repository is being extended into an XAI-centric
> benchmark paper (ground-truth faithfulness + agreement across ~26 XAI methods
> on 15 ML/DL models); see `PAPER_MISSION_BRIEF.md`. The `profile: reviewer`
> pipeline below (6 models, 7 XAI methods, 6 stocks) is green end-to-end on
> Modal; `profile: full` (15 models + GNN, ~26 methods, 60 stocks) is in
> progress. The TL;DR above reflects the earlier leakage-free re-evaluation and
> will be superseded once the XAI benchmark is written up.

## Repository layout
```
modal/                 # compute pipeline (runs on Modal.com)
  _common.py           # shared image, Volume, config-as-code (config.xai.yaml, profile switch)
  s1_data.py           # download FNSPID (prices + 23GB news) to a Volume
  s2_build.py          # Spark ingestion -> FinBERT+LM sentiment -> leakage-free panel + ground-truth features
  s3_experiments.py    # leakage demo, baselines, walk-forward, ablation, HAC/bootstrap stats
  s4_lstm.py           # GRU + 1D-CNN sequence models (GPU)
  s5_xai.py            # TreeSHAP / KernelSHAP / LIME / permutation / block-perm / ALE / Integrated Gradients
  s6_eda.py            # stationarity, autocorrelation, fat tails, cross-sectional correlation
  s7_faithfulness.py   # ground-truth faithfulness benchmark, stock-clustered CI + effective N
  s5g_tcav_null.py             # R4: TCAV vs. 100 random-concept CAVs (significance test)
  s5h_mechanistic_null.py      # R4: SAE + ablation feature-shuffle null baselines
  s7b_faithfulness_robustness.py  # R4: year-robustness (2021/2022/2023) + leave-10-stocks jackknife
make_tables.py         # metrics -> LaTeX tables + numbers.tex
notebooks/             # runnable analysis notebooks (from the built panel)
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
make experiments   # leakage demo, models, walk-forward, ablation, XAI, EDA
make results       # pull metrics/figures, regenerate LaTeX tables
make paper         # compile the anonymized PDF
```

The three notebooks in `notebooks/` reproduce the analysis interactively from
`results/panel.parquet` (they are shipped executed, with outputs).

## Design choices that fix the reviewed pipeline
- **No level prediction as the headline.** Primary tasks are next-day *return* and
  *direction*; level RMSE is reported only as a labeled leakage diagnostic.
- **Strictly chronological, embargoed walk-forward** (test years 2021–2023); all
  preprocessing is fit on training folds only.
- **Strong baselines** (persistence, historical-mean, majority) in every table.
- **Controlled sentiment ablation** (price-only vs. +sentiment vs. sentiment-only).
- **Reproducible sentiment**: FinBERT + Loughran–McDonald, replacing the original
  opaque `Sentiment_gpt`.
- **Correlation handled explicitly**: stationary features (VIF cut ~17×), ALE
  instead of PDP, symbol-clustered bootstrap and Newey–West HAC Diebold–Mariano.
- **Symbol** is one-hot encoded (not ordinal `LabelEncoder`).

## Data note
FNSPID's `full_history` truncates some tickers at 2020 and omits a few; the 12
tickers used here (AAPL, MSFT, AMZN, INTC, JPM, XOM, CVX, WMT, PFE, JNJ, KO, DIS)
have full 2018–2023 OHLCV coverage and dense news coverage in the 23 GB news file.
