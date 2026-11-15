# FinXAI Planted-Signal Recovery Benchmark

**A Planted-Signal Recovery Benchmark for XAI Methods on Financial Forecasters**

The compiled paper is at [`docs/main.pdf`](docs/main.pdf). This repository
holds the pipeline, the configuration, the stored results, and the script
that generates every table and number in the paper.

## What the benchmark measures

Six synthetic features with a known relation to the target are injected into
a real panel of 57 US large-cap stocks (2015-2023, FNSPID prices and news).
Each XAI method is then scored on whether it ranks the planted signals above
a planted noise feature on held-out data.

This is a test of **signal recovery**, not of faithfulness. The relation
between a planted feature and the target is known by construction, but how
much a fitted model relies on that feature is not. Recovery is therefore a
necessary condition for a faithful explanation: a method that fails it cannot
be considered faithful, and a method that passes it has not been shown to be
faithful.

| Planted feature | Column | Construction |
|---|---|---|
| Noise | `gt_noise` | i.i.d. N(0,1) |
| Linear signal | `gt_signal` | correlation 0.15 with the next-day return |
| Weak linear signal | `gt_signal_weak` | correlation 0.05 with the next-day return |
| Sign-coded signal | `gt_signal_nl` | built from the sign of the next-day return; realized correlation 0.12 with the direction label |
| Exact copy | `gt_redundant_copy` | copy of the real feature `vol_z20` |
| Partial copy | `gt_redundant_partial` | correlation 0.80 with `vol_z20` |

Every planted signal is the effect of a single feature. No interaction
between distinct inputs is planted, and no target value is defined for the
credit that a method should give to a duplicated feature.

## Findings

- 22 XAI methods are catalogued across 16 models. The 13 methods that yield a
  feature-importance vector are scored on their own output; all 13 rank the
  linear and the sign-coded signal above the noise feature on every model on
  which they were scored. The other 9 are not scored on feature recovery
  (rule and example outputs, the three Grad-CAM variants, TCAV, and the two
  mechanistic tools).
- Stock by stock (57 stocks, one XGBoost model, scored on 2023), TreeSHAP,
  KernelSHAP, and LIME rank all three planted signals above noise in every
  stock. Permutation importance does so for the linear signal in 78.9% of
  stocks and does not recover the weak signal (57.9% of stocks; the interval
  of its margin contains zero).
- The credit share given to an exact copy of a feature is 0.19 (TreeSHAP),
  0.21 (KernelSHAP), and 0.36 (LIME); for a partial copy it is 0.45, 0.48,
  and 0.37. No share is taken to be the correct one.
- Rankings of TreeSHAP, KernelSHAP, and LIME agree with one another (Spearman
  0.71-0.93) and only weakly with permutation importance (0.10-0.27).
- Two TCAV scores of 1.00 are not distinguishable from the scores of 100
  random concept vectors (empirical p = 1.00). Sparse-autoencoder and
  component-ablation outputs are reported as exploratory only.

Scope of the statistics: intervals come from 5,000 resamples of whole stocks
with the model fitted once and each explanation computed once. They do not
include variation between training seeds or the sampling of the explainers,
and the stocks are not independent (mean pairwise return correlation 0.381,
about 2.55 independent units under equal correlation). They are reported as
descriptive summaries.

## Evaluation protocol

- **Model zoo (substrate check):** expanding-window walk-forward with four
  test years, 2020-2023. For test year Y the training rows are dated more
  than 10 calendar days before 1 January of Y. Scalers are fitted on training
  rows only. Hyperparameters are fixed in the code; there is no search and no
  validation split for model selection.
- **Recovery scoring:** models are trained on all rows before 2023 and every
  method is scored on the rows of 2023. No embargo is applied at this stage.
- **Stability checks:** the stock-level analysis is repeated with 2021, 2022,
  and 2023 as the scored year. In these repetitions the model is trained on
  all other years, so for 2021 and 2022 the training set includes later
  years; they test dependence on the scored year and are not out-of-time
  evaluations. A leave-10-stocks-out check (200 draws) is also stored.

## Terminology

File names and result keys keep the wording of the first version of the
project. The paper uses the terms on the right.

| In file names and result keys | In the paper |
|---|---|
| `faithfulness`, `ground truth` (`gt_*`) | planted-signal recovery, planted features |
| `gt_signal_nl`, "interaction" in comments | sign-coded signal: a nonlinear effect of one feature, not an interaction between inputs |
| `muse_brcg` | a depth-3 decision-tree surrogate of the XGBoost model; it is not an implementation of MUSE or BRCG |
| `gam_aggregation` | rank aggregation of local TreeSHAP values |
| `acdc` | a manual single-component ablation sweep; it is not the ACDC algorithm |
| `gradcam`, `gradcam_pp`, `score_cam` scores | a per-feature score from Integrated Gradients on the same CNN; counted under Integrated Gradients, not as a score of the CAM methods |
| `mdes_approx`, `p_binom_one`, `p_union` | stored but not used in the paper (the first treats the 57 stocks as independent; the other two assume that a whole layer and a single head are equally likely to show the largest ablation effect) |

Anchors, counterfactual search, the contrastive explanation method, RISE,
Grad-CAM++, and Score-CAM are simplified re-implementations in this
repository, adapted to tabular or one-dimensional inputs. SHAP, LIME,
permutation importance, Integrated Gradients, LRP, guided backpropagation,
deconvnet, Grad-CAM, the EBM, and GNNExplainer come from reference libraries.

The notebooks in `notebooks/` and the figures in `results/figures/` are the
executed records of the first version. Their numbers are the stored results,
but their narrative text and titles use the earlier wording. Where the
wording differs, the paper is authoritative. `paper/tables/` also holds
supplementary tables that the six-page paper does not include.

## Repository layout
```
modal/                 # compute pipeline (runs on Modal.com)
  _common.py           # shared image, Volume, config loader (config.xai.yaml, profile switch)
  s1_data.py           # download FNSPID (prices + 23GB news) to a Volume
  s2_build.py          # Spark ingestion -> FinBERT+LM sentiment -> panel + planted features
  s3_experiments.py    # leakage demo, baselines, walk-forward model zoo, sentiment ablation
  s4_lstm.py           # GRU / LSTM / 1D-CNN / TCN / Transformer sequence models (GPU)
  s4b_gnn.py           # GNN over a cross-stock correlation graph
  s4c_gt_artifacts.py  # sequence models retrained on inputs with the planted features
  s5_xai.py            # TreeSHAP / KernelSHAP / LIME / permutation / block-perm / ALE / Integrated Gradients
  s5b_rule_methods.py  # Anchors, counterfactuals, CEM, RISE (simplified implementations)
  s5c_concept_intrinsic.py  # TCAV, EBM term importance, rank-aggregated TreeSHAP, decision-tree surrogate
  s5d_gradient_cam.py  # LRP, guided backpropagation, deconvnet, Grad-CAM family
  s5e_mechanistic.py   # sparse autoencoder + single-component ablation (+ seed/year sensitivity)
  s5f_graph.py         # GNNExplainer
  s5g_tcav_null.py     # TCAV vs. 100 random concept vectors
  s5h_mechanistic_null.py  # sparse-autoencoder and ablation shuffle nulls
  s6_eda.py            # stationarity, autocorrelation, fat tails, cross-sectional correlation
  s7_faithfulness.py   # stock-level recovery of the four core methods, stock-resampled intervals
  s7b_faithfulness_robustness.py  # scored-year repetitions (2021/2022/2023) + leave-10-stocks-out
  s8_notebooks.py      # execute the notebook suite on Modal with outputs embedded
config.xai.yaml        # universe, dates, test years, embargo, seeds, planted-feature settings
make_tables.py         # results -> paper/tables/*.tex, paper/numbers.tex, paper/NUMBERS_AUDIT.md
notebooks/             # 27 executed notebooks (first version; see Terminology)
paper/                 # LaTeX source (main.tex, refs.bib, generated tables and numbers)
results/               # metrics (JSON/CSV) and figures pulled from the Volume
```

## Reproducing
Compute runs on [Modal](https://modal.com) and the paper is compiled with
`tectonic`. The public [FNSPID](https://huggingface.co/datasets/Zihan1004/FNSPID)
data set is downloaded by the pipeline and is not redistributed here. All
parameters are in `config.xai.yaml` (`profile: reviewer` for a small fast
run, `profile: full` for the paper).

```bash
pip install modal && modal token new     # one-time compute auth
make reproduce                           # data -> panel -> experiments -> paper
```

`make help` lists the individual targets. The Modal Volume caches data and
artifacts, so each stage can be re-run on its own:

```bash
make data          # download FNSPID (prices + 23GB news)
make build         # Spark ingest -> FinBERT sentiment -> panel
make experiments   # model zoo, XAI suite, recovery scoring, stability checks, nulls
make results       # pull metrics/figures, regenerate LaTeX tables
make paper         # compile the PDF into docs/
make notebooks     # execute the notebook suite on Modal, outputs embedded
```

The tables and numbers of the paper can be regenerated from the stored
results without Modal:

```bash
pip install pandas numpy pyarrow
python make_tables.py
```

Every result quoted in the paper is a macro in `paper/numbers.tex`, written
by `make_tables.py` from `results/`. [`paper/NUMBERS_AUDIT.md`](paper/NUMBERS_AUDIT.md)
lists each macro with its value and source file, together with the protocol
constants that the text restates from the pipeline code.

## Project history
This repository grew out of an earlier project (FinSentImpact) that
reproduced a "R2 = 0.99" next-day forecasting claim, showed it to be an
autocorrelation artifact, and measured the marginal value of financial-news
sentiment under walk-forward evaluation. That analysis survives as the
leakage demo and the sentiment ablation in `modal/s3_experiments.py`, and the
near-zero-signal panel it exposed is the substrate into which the planted
features are injected. The repository name keeps the earlier project title.
