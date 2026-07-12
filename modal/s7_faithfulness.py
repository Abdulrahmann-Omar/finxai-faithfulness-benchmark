"""Stage 7 -- the ground-truth faithfulness benchmark, stock-clustered (the
paper's headline; brief S4/S9/S13/S14).

  modal run s7_faithfulness.py::run

s5_xai.py already checks, once, on a single global model fit, whether each
method ranks gt_signal above gt_noise. That is not yet a claim the brief lets
us make: every importance/faithfulness claim must be stock-clustered and state
an effective N (S9's decision rule, S13). This stage:

  1. Computes each method's global importance vector PER SYMBOL (the same
     model, but attribution/permutation restricted to that symbol's rows) --
     TreeSHAP, KernelSHAP, LIME, permutation.
  2. Cluster-bootstraps over symbols (never stock-days) for a 95% CI on the
     mean (gt_signal - gt_noise) importance margin and on the credit-splitting
     share for gt_redundant_copy vs. its source feature.
  3. Reports effective N (N/(1+(N-1)*rho_bar), rho_bar = mean pairwise return
     correlation across THIS run's universe) and an approximate MDES, so a
     null or a positive faithfulness claim can both be read honestly.

Known limitation (stated, not hidden): with a 6-ticker reviewer universe the
per-symbol samples for KernelSHAP/LIME are thin and effective N is small --
CIs will be wide. That is the correct, honest behavior at this profile; the
`full` profile's 60-stock universe is what gives this benchmark real power.
"""
import modal
from _common import image, data_vol, DATA, SEED, GROUND_TRUTH, N_BOOTSTRAP

app = modal.App("finsent-s7-faithfulness", image=image)

BUILD = f"{DATA}/build"
RES = f"{DATA}/results"
MET = f"{RES}/metrics"
FIG = f"{RES}/figures"

PRICE_FEATS = ["ret_lag1", "ret_lag2", "ret_lag3", "ret_lag5",
               "vol_roll5", "vol_roll20", "rsi14", "ma_gap20",
               "hl_range", "co_ret", "logvol", "vol_z20"]
SENT_FEATS = ["fb_score", "fb_pos", "fb_neg", "lm_score", "news_count",
              "has_news", "fb_roll3", "fb_roll5", "news_roll5", "fb_chg"]
GT_FEATS = [c for c, on in [("gt_noise", GROUND_TRUTH.get("noise_feature")),
                            ("gt_signal", GROUND_TRUTH.get("signal_feature")),
                            ("gt_signal_nl", GROUND_TRUTH.get("nonlinear_signal_feature")),
                            ("gt_signal_weak", GROUND_TRUTH.get("weak_signal_feature")),
                            ("gt_redundant_copy", GROUND_TRUTH.get("redundant_copy")),
                            ("gt_redundant_partial", GROUND_TRUTH.get("partial_redundant_feature"))] if on]


def _mpl():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 150, "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False})
    return plt


def _cluster_one_margin(imp_by_symbol, signal_col, rng, n_bootstrap):
    """Cluster-bootstrap the mean (imp[signal_col] - imp['gt_noise']) margin
    over symbols. Shared by the linear (gt_signal), nonlinear (gt_signal_nl),
    and weak (gt_signal_weak, M5) faithfulness checks so all get identical
    treatment. Also cluster-bootstraps the %symbols-correct PROPORTION (M2)
    with the same whole-symbol resamples, and returns the raw per-symbol
    margins (keyed by symbol) so callers can run paired cross-method tests."""
    import numpy as np
    margins_by_sym = {s: float(v[signal_col] - v["gt_noise"]) for s, v in imp_by_symbol.items()
                       if signal_col in v.index and "gt_noise" in v.index}
    if not margins_by_sym:
        return None
    syms = list(margins_by_sym.keys())
    margins = np.array([margins_by_sym[s] for s in syms])
    n = len(margins)
    idx_draws = [rng.choice(n, n, replace=True) for _ in range(n_bootstrap)]
    boot_mean = np.array([margins[d].mean() for d in idx_draws])
    boot_prop = np.array([(margins[d] > 0).mean() for d in idx_draws])
    lo, hi = np.percentile(boot_mean, [2.5, 97.5])
    plo, phi = np.percentile(boot_prop, [2.5, 97.5])
    se = float(margins.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return {
        "n_symbols_scored": int(n),
        "mean_margin": float(margins.mean()),
        "ci95_clustered": [float(lo), float(hi)],
        "pct_symbols_signal_above_noise": float((margins > 0).mean()),
        "pct_symbols_ci95_clustered": [float(plo), float(phi)],
        "mdes_approx": float(2.8 * se) if se == se else None,   # 80% power, two-sided, cluster-mean SE
        "margins_by_symbol": margins_by_sym,
    }


def _cluster_faithfulness(imp_by_symbol, method_name, rng, n_bootstrap):
    """imp_by_symbol: dict[symbol] -> pd.Series (per-symbol |importance|).
    Cluster-bootstraps the mean (gt_signal - gt_noise) margin AND, if present,
    the (gt_signal_nl - gt_noise) and (gt_signal_weak - gt_noise, M5) margins
    over symbols (the clustering unit), and reports the exact-copy AND
    partial-collinearity (M5) redundant-feature credit splits."""
    import numpy as np
    credit_shares, credit_shares_partial = [], []
    for v in imp_by_symbol.values():
        if "gt_redundant_copy" in v.index and "vol_z20" in v.index:
            a, b = float(v["gt_redundant_copy"]), float(v["vol_z20"])
            if (a + b) > 0:
                credit_shares.append(a / (a + b))
        if "gt_redundant_partial" in v.index and "vol_z20" in v.index:
            ap, bp = float(v["gt_redundant_partial"]), float(v["vol_z20"])
            if (ap + bp) > 0:
                credit_shares_partial.append(ap / (ap + bp))

    linear = _cluster_one_margin(imp_by_symbol, "gt_signal", rng, n_bootstrap)
    nonlinear = _cluster_one_margin(imp_by_symbol, "gt_signal_nl", rng, n_bootstrap)
    weak = _cluster_one_margin(imp_by_symbol, "gt_signal_weak", rng, n_bootstrap)

    out = {"method": method_name,
           "n_symbols_scored": linear["n_symbols_scored"] if linear else 0}
    if linear is None:
        return out
    # keep the linear fields at top level (backward-compatible with existing
    # consumers: make_tables.py, s5_xai.py's summary printer, the paper table)
    out.update({
        "mean_signal_minus_noise_margin": linear["mean_margin"],
        "ci95_clustered": linear["ci95_clustered"],
        "pct_symbols_signal_above_noise": linear["pct_symbols_signal_above_noise"],
        "pct_symbols_ci95_clustered": linear["pct_symbols_ci95_clustered"],
        "mdes_approx": linear["mdes_approx"],
    })
    # per-symbol margins kept for cross-method paired tests (M2), not written
    # to the final JSON (popped by run() after pairing) to keep the file small
    out["_margins_by_symbol"] = linear["margins_by_symbol"]
    if nonlinear is not None:
        out["nonlinear"] = {
            "mean_signal_minus_noise_margin": nonlinear["mean_margin"],
            "ci95_clustered": nonlinear["ci95_clustered"],
            "pct_symbols_signal_above_noise": nonlinear["pct_symbols_signal_above_noise"],
            "mdes_approx": nonlinear["mdes_approx"],
            "n_symbols_scored": nonlinear["n_symbols_scored"],
        }
    if weak is not None:
        # R3 revision (M5): weak (rho=0.05) planted-signal recovery -- the
        # near-noise-floor condition reviewers asked for.
        out["weak"] = {
            "mean_signal_minus_noise_margin": weak["mean_margin"],
            "ci95_clustered": weak["ci95_clustered"],
            "pct_symbols_signal_above_noise": weak["pct_symbols_signal_above_noise"],
            "pct_symbols_ci95_clustered": weak["pct_symbols_ci95_clustered"],
            "mdes_approx": weak["mdes_approx"],
            "n_symbols_scored": weak["n_symbols_scored"],
        }
    if credit_shares:
        out["mean_redundant_copy_credit_share"] = float(np.mean(credit_shares))
        out["n_symbols_credit_split"] = len(credit_shares)
    if credit_shares_partial:
        # R3 revision (M5): partial (rho=0.8) collinearity credit split.
        out["mean_redundant_partial_credit_share"] = float(np.mean(credit_shares_partial))
        out["n_symbols_credit_split_partial"] = len(credit_shares_partial)
    return out


def _paired_tests(results):
    """M2: pairwise per-stock paired tests of 'permutation' vs each of
    treeshap/kernelshap/lime on the gt_signal-vs-gt_noise recovery outcome,
    using the per-symbol margins stashed by _cluster_faithfulness.
      - McNemar's test on the binary outcome (margin>0), paired by symbol
        (only symbols scored by BOTH methods enter the contingency table).
      - Wilcoxon signed-rank test on the continuous margins, same pairing.
    Both are legitimate paired tests here because clustering unit == pairing
    unit == symbol (never stock-day)."""
    import numpy as np
    from scipy.stats import wilcoxon
    from statsmodels.stats.contingency_tables import mcnemar

    by_method = {r["method"]: r for r in results if "_margins_by_symbol" in r}
    if "permutation" not in by_method:
        return []
    perm_m = by_method["permutation"]["_margins_by_symbol"]
    out = []
    for other in ("treeshap", "kernelshap", "lime"):
        if other not in by_method:
            continue
        other_m = by_method[other]["_margins_by_symbol"]
        common = sorted(set(perm_m) & set(other_m))
        if len(common) < 5:
            out.append({"method_a": "permutation", "method_b": other,
                        "n_common_symbols": len(common),
                        "note": "too few common symbols for a paired test"})
            continue
        a_bin = np.array([perm_m[s] > 0 for s in common], dtype=int)
        b_bin = np.array([other_m[s] > 0 for s in common], dtype=int)
        a_cont = np.array([perm_m[s] for s in common])
        b_cont = np.array([other_m[s] for s in common])
        # McNemar contingency table: [[both wrong/right agree], [disagree]]
        n00 = int(np.sum((a_bin == 0) & (b_bin == 0)))
        n01 = int(np.sum((a_bin == 0) & (b_bin == 1)))   # perm wrong, other right
        n10 = int(np.sum((a_bin == 1) & (b_bin == 0)))   # perm right, other wrong
        n11 = int(np.sum((a_bin == 1) & (b_bin == 1)))
        table = [[n00, n01], [n10, n11]]
        try:
            mc = mcnemar(table, exact=(n01 + n10) < 25, correction=True)
            mc_p = float(mc.pvalue)
        except Exception as e:
            mc_p = None
        try:
            diff = a_cont - b_cont
            if np.allclose(diff, 0):
                w_p = 1.0
            else:
                w_p = float(wilcoxon(a_cont, b_cont).pvalue)
        except Exception:
            w_p = None
        out.append({
            "method_a": "permutation", "method_b": other,
            "n_common_symbols": len(common),
            "mcnemar_table_perm_wrong_other_right__perm_right_other_wrong": [n01, n10],
            "mcnemar_p": mc_p,
            "wilcoxon_signed_rank_p": w_p,
            "interpretation": ("permutation and " + other + " differ significantly "
                "in per-stock signal-vs-noise recovery (p<0.05)") if (mc_p is not None and mc_p < 0.05)
                else "no significant paired difference detected (p>=0.05)",
        })
    return out


@app.function(volumes={DATA: data_vol}, cpu=8.0, memory=32768, timeout=60 * 60)
def run():
    import os, json, numpy as np, pandas as pd
    import shap
    from sklearn.preprocessing import StandardScaler
    from sklearn.inspection import permutation_importance
    import xgboost as xgb
    plt = _mpl()
    os.makedirs(FIG, exist_ok=True); os.makedirs(MET, exist_ok=True)
    rng = np.random.RandomState(SEED)

    feats = PRICE_FEATS + SENT_FEATS + GT_FEATS
    df = pd.read_parquet(os.path.join(BUILD, "panel.parquet")).copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Date", "Symbol"]).reset_index(drop=True)
    tr = df[df["Date"].dt.year < 2023]
    te = df[df["Date"].dt.year == 2023].reset_index(drop=True)
    sc = StandardScaler().fit(tr[feats].values)
    Xtr = pd.DataFrame(sc.transform(tr[feats].values), columns=feats)
    Xte = pd.DataFrame(sc.transform(te[feats].values), columns=feats)
    ytr, yte = tr["dir_next"].values, te["dir_next"].values
    sym = te["Symbol"].values   # positionally aligned with Xte/yte rows

    # ---- effective N: mean pairwise cross-sectional return correlation, THIS universe ----
    ret_piv = df.pivot_table(index="Date", columns="Symbol", values="ret")
    corr = ret_piv.corr()
    n_sym = corr.shape[0]
    off_diag = corr.values[~np.eye(n_sym, dtype=bool)]
    rho_bar = float(np.nanmean(off_diag))
    n_eff = float(n_sym / (1 + (n_sym - 1) * rho_bar))
    print(f"universe: {n_sym} symbols, rho_bar={rho_bar:.3f}, effective N={n_eff:.2f}")

    model = xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.03,
             subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, n_jobs=-1,
             random_state=SEED, eval_metric="logloss")
    model.fit(Xtr.values, ytr)

    symbols = sorted(set(sym))
    results = []

    # ---------- TreeSHAP, per symbol ----------
    try:
        expl = shap.TreeExplainer(model)
        # cap total rows explained for reviewer-profile wall-clock; sample per symbol
        max_total = 3000
        per_sym_cap = max(50, max_total // len(symbols))
        imp_by_symbol = {}
        for s in symbols:
            idx = np.where(sym == s)[0]
            if len(idx) > per_sym_cap:
                idx = rng.choice(idx, per_sym_cap, replace=False)
            if len(idx) < 10:
                continue
            sv = expl.shap_values(Xte.iloc[idx])
            if isinstance(sv, list):
                sv = sv[1]
            imp_by_symbol[s] = pd.Series(np.abs(sv).mean(0), index=feats)
        results.append(_cluster_faithfulness(imp_by_symbol, "treeshap", rng, N_BOOTSTRAP))
        print("treeshap per-symbol done:", len(imp_by_symbol), "symbols")
    except Exception as e:
        print("treeshap per-symbol step failed:", e)

    # ---------- KernelSHAP, per symbol (model-agnostic; independent of TreeSHAP) ----------
    try:
        bg = shap.sample(Xtr, 50, random_state=SEED)
        kexpl = shap.KernelExplainer(lambda X: model.predict_proba(X)[:, 1], bg)
        per_sym_cap_k = 40
        imp_by_symbol = {}
        for s in symbols:
            idx = np.where(sym == s)[0]
            if len(idx) > per_sym_cap_k:
                idx = rng.choice(idx, per_sym_cap_k, replace=False)
            if len(idx) < 10:
                continue
            ksv = kexpl.shap_values(Xte.iloc[idx], nsamples=150, silent=True)
            imp_by_symbol[s] = pd.Series(np.abs(ksv).mean(0), index=feats)
        results.append(_cluster_faithfulness(imp_by_symbol, "kernelshap", rng, N_BOOTSTRAP))
        print("kernelshap per-symbol done:", len(imp_by_symbol), "symbols")
    except Exception as e:
        print("kernelshap per-symbol step failed:", e)

    # ---------- LIME, per symbol (aggregated local weights) ----------
    try:
        from lime.lime_tabular import LimeTabularExplainer
        lexpl = LimeTabularExplainer(Xtr.values, feature_names=feats, class_names=["down", "up"],
                                     mode="classification", discretize_continuous=True,
                                     random_state=SEED)
        per_sym_n = 10
        imp_by_symbol = {}
        for s in symbols:
            idx = np.where(sym == s)[0]
            if len(idx) < 10:
                continue
            pick = rng.choice(idx, min(per_sym_n, len(idx)), replace=False)
            wsum = np.zeros(len(feats))
            for i in pick:
                exp = lexpl.explain_instance(Xte.values[i], model.predict_proba,
                                             num_features=len(feats), labels=(1,))
                for fi, w in exp.as_map()[1]:
                    wsum[fi] += abs(w)
            imp_by_symbol[s] = pd.Series(wsum / len(pick), index=feats)
        results.append(_cluster_faithfulness(imp_by_symbol, "lime", rng, N_BOOTSTRAP))
        print("lime per-symbol done:", len(imp_by_symbol), "symbols")
    except Exception as e:
        print("lime per-symbol step failed:", e)

    # ---------- Permutation importance, per symbol ----------
    try:
        imp_by_symbol = {}
        for s in symbols:
            idx = np.where(sym == s)[0]
            if len(idx) < 10:
                continue
            pi = permutation_importance(model, Xte.values[idx], yte[idx], n_repeats=10,
                                        random_state=SEED, scoring="accuracy")
            imp_by_symbol[s] = pd.Series(pi.importances_mean, index=feats)
        results.append(_cluster_faithfulness(imp_by_symbol, "permutation", rng, N_BOOTSTRAP))
        print("permutation per-symbol done:", len(imp_by_symbol), "symbols")
    except Exception as e:
        print("permutation per-symbol step failed:", e)

    # M2: paired per-stock tests (permutation vs treeshap/kernelshap/lime),
    # computed from the raw per-symbol margins BEFORE they're stripped below.
    paired = _paired_tests(results)
    for r in results:
        r.pop("_margins_by_symbol", None)

    summary = {
        "universe_n_symbols": int(n_sym),
        "cross_sectional_rho_bar": rho_bar,
        "effective_n": n_eff,
        "n_bootstrap": N_BOOTSTRAP,
        "cluster_unit": "symbol",
        "methods": results,
        "paired_tests_permutation_vs_shap_lime": paired,
        "note": ("KernelSHAP/LIME per-symbol samples are thin at the reviewer profile "
                 "(6 tickers); CIs are correspondingly wide. This is expected and honestly "
                 "reported, not hidden -- see effective_n and mdes_approx per method. "
                 "paired_tests_permutation_vs_shap_lime (M2, R3 revision): McNemar + "
                 "Wilcoxon signed-rank tests pairing each method's per-symbol "
                 "signal-vs-noise outcome with permutation's, on the common scored symbols."),
    }
    with open(os.path.join(MET, "faithfulness_benchmark.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    # ---- headline figure: mean signal-noise margin with clustered 95% CI ----
    scored = [r for r in results if "mean_signal_minus_noise_margin" in r]
    if scored:
        methods = [r["method"] for r in scored]
        means = [r["mean_signal_minus_noise_margin"] for r in scored]
        los = [means[i] - r["ci95_clustered"][0] for i, r in enumerate(scored)]
        his = [r["ci95_clustered"][1] - means[i] for i, r in enumerate(scored)]
        fig, ax = plt.subplots(figsize=(7.5, 4.3))
        y = np.arange(len(methods))
        ax.errorbar(means, y, xerr=[los, his], fmt="o", color="#2F4B94",
                    capsize=4, lw=2, markersize=7)
        ax.axvline(0, color="#C0362C", ls="--", lw=1.2)
        ax.set_yticks(y); ax.set_yticklabels(methods)
        ax.set_xlabel("mean(|imp(gt_signal)| - |imp(gt_noise)|), stock-clustered 95% CI")
        ax.set_title(f"Ground-truth faithfulness benchmark (N={n_sym} stocks, "
                     f"effective N={n_eff:.1f})", fontweight="bold", fontsize=10.5)
        fig.savefig(os.path.join(FIG, "fig_faithfulness_benchmark.png"), bbox_inches="tight")
        print("saved fig_faithfulness_benchmark.png")
    data_vol.commit()
