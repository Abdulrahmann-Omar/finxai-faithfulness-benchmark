# Reproducibility & Methodological Checklist

Status: `profile: full` (16 ML/DL models including a GNN, 23 catalogued XAI
methods, 57-stock universe) is implemented end-to-end and is what the paper
(`docs/main.pdf`) reports. `profile: reviewer` remains available in
`config.xai.yaml` as a fast small-scale run. Items below marked `[ ]` were
written when `full` was still in progress and are now covered by the full
pipeline.

## Data and problem definition
- [x] Public dataset (FNSPID) with exact files documented (`modal/s1_data.py`).
- [x] X/Y stated formally: features use only information at the close of day t;
      targets are next-day return / direction / (diagnostic) level.
- [x] Time-aware news to trading-session alignment (after-close news -> next
      session); sentiment aggregated per (stock, session).
- [x] Universe selection documented (`reviewer` profile: 6 of the 12 tickers
      with full 2018-2023 OHLCV coverage and adequate news coverage; selection
      is by data availability, not outcome).
- [ ] `full` profile: 60-stock yfinance universe (2015-2024), diverse
      sectors/caps, for higher effective N (brief P5).
- [x] Ground-truth faithfulness features (brief P10, `modal/s2_build.py::assemble`):
      `gt_noise` (true importance 0), `gt_signal` (planted rho=0.15 leak of
      `ret_next`), `gt_redundant_copy` (exact duplicate of `vol_z20`, tests
      credit-splitting under collinearity). Construction logged in
      `build/feature_manifest.json`.

## Data integrity (found by EDA, `modal/s6_eda.py`)
- [x] ADF stationarity: price levels non-stationary, returns stationary.
- [x] Split/adjust artifact caught and fixed: returns from Adj_close, plus
      masking of impossible |r|>0.5 splice returns (e.g. AMZN -294% on 2020-07-06).
- [x] Post-clean sanity: realistic kurtosis and restored volatility clustering.
- [x] Empirical leakage screen: no backward-looking feature correlates > 0.06
      with the next-day return.

## Evaluation protocol
- [x] Strictly chronological, embargoed (10-day) walk-forward; test years
      2021, 2022, 2023.
- [x] All preprocessing (scalers) fit on training folds only.
- [x] Strong baselines in every table: persistence, historical-mean (returns),
      majority-class / always-up (direction).
- [x] Primary tasks are returns and direction; price level is a labeled
      diagnostic only.
- [x] `Symbol` one-hot encoded (never ordinal LabelEncoder) in the honest models.

## Correlation handled as a first-class threat
- [x] Feature multicollinearity: stationary features (max VIF ~5 vs ~29,000 for
      raw levels), redundant linear-combination features pruned.
- [x] Interpretation robust to correlation: ALE (not PDP), grouped attribution,
      and grouped block-permutation (permute whole sentiment block jointly).
- [x] Temporal autocorrelation: Newey-West HAC Diebold-Mariano test.
- [x] Cross-sectional co-movement: symbol-clustered bootstrap (resample whole
      stocks); stocks, not stock-days, are the effective unit.

## Models and ablation
- [x] `reviewer` profile model zoo (6/15): Ridge/Logistic, RF, XGBoost, LightGBM
      (bonus, beyond the profile spec), SVM, EBM (`interpret`, glass-box/GA2M),
      GRU, 1D-CNN (`modal/s4_lstm.py::run`, `::run_cnn1d`).
- [ ] `full` profile model zoo (15 + GNN): + Decision Tree, CatBoost, k-NN, MLP,
      LSTM, TCN, Transformer, GNN (brief S12).
- [x] Controlled ablation: price-only vs sentiment-only vs price+sentiment.
- [x] Reproducible sentiment (FinBERT + Loughran-McDonald), replacing the
      original opaque `Sentiment_gpt`.
- [x] Fixed seed (42); metrics as JSON/CSV; figures as PNG.

## Explainability (interpreted carefully)
- [x] `reviewer` profile XAI (7/26): TreeSHAP, KernelSHAP, LIME, permutation,
      grouped block-permutation, ALE, Integrated Gradients (GRU) (`modal/s5_xai.py`).
- [ ] `full` profile XAI (~26): + Anchors, counterfactuals, CEM, RISE, GAM
      aggregation, MUSE, BRCG, CAV/TCAV, LRP, Guided Backprop, DeconvNet,
      Grad-CAM family (CNN), GNNExplainer, SAE, ACDC, actionable recourse
      (brief S14, gated by the applicability matrix per model class).
- [x] Distinguishes in-sample attribution from out-of-sample predictive value.
- [x] **Ground-truth faithfulness benchmark** (`modal/s7_faithfulness.py`, the
      paper's headline): every method above scored on whether it ranks
      `gt_signal` above `gt_noise`, and how it splits credit between
      `gt_redundant_copy` and its source feature -- computed PER SYMBOL, then
      stock-clustered (5000/1000-draw bootstrap per profile) with effective N
      (`N/(1+(N-1)*rho_bar)`) and an approximate MDES reported alongside every
      claim. `results/metrics/faithfulness_benchmark.json`.
- [x] **R4 revision: null baselines and robustness checks**, added in response
      to peer review:
      - TCAV vs. 100 random-concept CAVs, empirical significance test
        (`modal/s5g_tcav_null.py` -> `xai_tcav_significance.json`).
      - SAE interpretable-latent count and ablation notable-component count vs.
        a 20-shuffle feature-permutation null, plus the exact recurrence
        probability for the `layer1_full` ablation finding
        (`modal/s5h_mechanistic_null.py` -> `xai_sae_null.json`,
        `xai_ablation_null.json`).
      - Faithfulness benchmark re-scored with 2021/2022 (not just 2023) as the
        held-out year, and a leave-10-stocks-out jackknife on the 2023 result
        (`modal/s7b_faithfulness_robustness.py` -> `faithfulness_by_year.json`,
        `faithfulness_jackknife.json`).
      All three write NEW metric files and never modify the outputs of the
      stages they check.

## Manuscript & artifact
- [x] Anonymized for double-blind review.
- [x] Numbers auto-generated from metrics (`make_tables.py`) -> internal
      consistency by construction.
- [x] Notebooks reproduce the analysis from `results/panel.parquet`.
- [x] `requirements.txt` pinned; `make reproduce` one-command reproduction
      (Makefile; supersedes the older `run_all.sh`, kept for reference but not
      maintained in lockstep with new pipeline stages).
