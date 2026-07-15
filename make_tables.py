#!/usr/bin/env python3
"""Turn the experiment metrics (results/metrics/*) into LaTeX tables and a
numbers.tex macro file consumed by paper/main.tex. Run locally after pulling
results from the Modal volume. Keeps every number in the paper consistent and
reproducible from a single source of truth.
"""
import json, os, math
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
MET = os.path.join(ROOT, "results", "metrics")
PAP = os.path.join(ROOT, "paper")
TAB = os.path.join(PAP, "tables")
os.makedirs(TAB, exist_ok=True)


def jload(name):
    with open(os.path.join(MET, name)) as f:
        return json.load(f)


def tex_escape(s):
    """Escape LaTeX special characters in free-text table cells (method
    names, notes) so raw underscores/percent signs/etc. from JSON/CSV don't
    break compilation."""
    s = str(s)
    for a, b in [("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"),
                 ("#", r"\#"), ("&", r"\&"), ("$", r"\$")]:
        s = s.replace(a, b)
    return s


def fmt(x, d=3):
    try:
        if x != x:  # nan
            return "--"
        return f"{x:.{d}f}"
    except Exception:
        return str(x)


def fmt_p(x):
    """p-value formatter: '<0.001' instead of a misleadingly-exact '0.000'."""
    try:
        if x != x:
            return "--"
        return "$<$0.001" if x < 0.001 else f"{x:.3f}"
    except Exception:
        return str(x)


macros = {}

# ---------------- leakage ----------------
lk = jload("leakage_demo.json")
macros["RleakRand"] = fmt(lk["A1_random_split_RF_levels"]["R2"], 3)
macros["RleakChrono"] = fmt(lk["A2_chrono_split_RF_levels"]["R2"], 3)
macros["RleakPersist"] = fmt(lk["A3_persistence_levels_TESTonly"]["R2"], 3)
macros["RleakReturns"] = fmt(lk["A4_chrono_RF_returns"]["R2"], 3)

with open(os.path.join(TAB, "tab_leakage.tex"), "w") as f:
    f.write("\\begin{tabular}{@{}llrr@{}}\n\\toprule\n")
    f.write("Protocol & Target & $R^2$ & RMSE \\\\\n\\midrule\n")
    rows = [
        ("RF, random split", "level", lk["A1_random_split_RF_levels"]),
        ("RF, chronological", "level", lk["A2_chrono_split_RF_levels"]),
        ("Persistence $\\hat P_{t+1}{=}P_t$", "level", lk["A3_persistence_levels_TESTonly"]),
        ("RF, chronological", "\\textbf{return}", lk["A4_chrono_RF_returns"]),
    ]
    for name, tgt, d in rows:
        f.write(f"{name} & {tgt} & {fmt(d['R2'],3)} & {fmt(d['RMSE'],3)} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# ---------------- main results ----------------
mr = pd.read_csv(os.path.join(MET, "main_results.csv"))
agg = mr.groupby(["task", "model", "features"]).mean(numeric_only=True).reset_index()

# return table: R2, RMSE, DirAcc for each model x {price_only, price+sentiment}
ret = agg[agg.task == "return"]
dic = agg[agg.task == "direction"]

reg_models = ["Ridge", "RandomForest", "XGBoost", "LightGBM", "SVR"]
clf_models = ["Logistic", "RandomForest", "XGBoost", "LightGBM", "SVC"]


def get(df, model, feat, col):
    r = df[(df.model == model) & (df.features == feat)]
    return r[col].iloc[0] if len(r) else float("nan")


def base(df, name, col):
    r = df[df.model == name]
    return r[col].mean() if len(r) else float("nan")


with open(os.path.join(TAB, "tab_main.tex"), "w") as f:
    f.write("\\begin{tabular}{@{}l|rrr|rrr@{}}\n\\toprule\n")
    f.write("& \\multicolumn{3}{c|}{Return ($R^2$ / RMSE / DirAcc)} & "
            "\\multicolumn{3}{c}{Direction (Acc / F1 / MCC)} \\\\\n")
    f.write("Model & \\multicolumn{3}{c|}{} & \\multicolumn{3}{c}{} \\\\\n\\midrule\n")
    # baselines
    f.write("\\multicolumn{7}{@{}l}{\\emph{Baselines}}\\\\\n")
    f.write(f"~~persistence/zero & {fmt(base(ret,'baseline:zero','R2'))} & "
            f"{fmt(base(ret,'baseline:zero','RMSE'))} & -- & -- & -- & -- \\\\\n")
    f.write(f"~~hist.\\ mean / majority & {fmt(base(ret,'baseline:train_mean','R2'))} & "
            f"{fmt(base(ret,'baseline:train_mean','RMSE'))} & -- & "
            f"{fmt(base(dic,'baseline:majority','Acc'))} & "
            f"{fmt(base(dic,'baseline:majority','F1'))} & "
            f"{fmt(base(dic,'baseline:majority','MCC'))} \\\\\n")
    f.write("\\midrule\n")
    for i, rm in enumerate(reg_models):
        cm = clf_models[i]
        for feat, tag in [("price_only", " (price)"), ("price+sentiment", " (+sent)")]:
            name = rm + tag
            f.write(
                f"{name} & {fmt(get(ret,rm,feat,'R2'))} & {fmt(get(ret,rm,feat,'RMSE'))} & "
                f"{fmt(get(ret,rm,feat,'DirAcc'))} & {fmt(get(dic,cm,feat,'Acc'))} & "
                f"{fmt(get(dic,cm,feat,'F1'))} & {fmt(get(dic,cm,feat,'MCC'))} \\\\\n")
        f.write("\\addlinespace\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# best directional accuracy (price+sentiment or price_only) and majority baseline
best_acc = dic[dic.model.isin(clf_models)]["Acc"].max()
maj = base(dic, "baseline:majority", "Acc")
macros["bestDirAcc"] = fmt(best_acc * 100, 1)
macros["majAcc"] = fmt(maj * 100, 1)

# ---------------- ablation stats ----------------
ab = jload("ablation_stats.json")
da = ab["direction_sentiment_ablation"]
# pick XGBoost (or first) for headline numbers
head_model = "XGBoost" if "XGBoost" in da else list(da.keys())[0]
hm = da[head_model]
macros["sentDeltaAcc"] = fmt(hm["mean_delta_acc"] * 100, 2)
macros["sentCIlo"] = fmt(hm["ci95_clustered"][0] * 100, 2)
macros["sentCIhi"] = fmt(hm["ci95_clustered"][1] * 100, 2)

with open(os.path.join(TAB, "tab_ablation.tex"), "w") as f:
    f.write("\\begin{tabular}{@{}lrrrr@{}}\n\\toprule\n")
    f.write("Model & $\\Delta$Acc (pp) & 95\\% CI & Wilcoxon $p$ & DM $t$ (HAC) \\\\\n\\midrule\n")
    dm = ab.get("return_diebold_mariano_HAC", {})
    for m, d in da.items():
        dmt = dm.get(m, {}).get("median_HAC_dm_t", float("nan"))
        ci = d["ci95_clustered"]
        f.write(f"{m} & {fmt(d['mean_delta_acc']*100,2)} & "
                f"$[{fmt(ci[0]*100,2)},{fmt(ci[1]*100,2)}]$ & "
                f"{fmt(d['wilcoxon_p_by_symbol'],2)} & {fmt(dmt,2)} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# ---------------- XAI ----------------
xai = jload("xai_summary.json")
macros["shapSentShare"] = fmt(xai["shap_sentiment_attribution_share"] * 100, 1)
macros["igSentShare"] = fmt(xai.get("ig_sentiment_share", float("nan")) * 100, 1)
macros["permSentShare"] = fmt(xai.get("perm_sentiment_share", 0.0) * 100, 1)
macros["blockPermSent"] = fmt(xai.get("block_perm_sentiment_acc_drop", 0.0) * 100, 2)
macros["blockPermPrice"] = fmt(xai.get("block_perm_price_acc_drop", 0.0) * 100, 2)

# sentiment-only skill (to show it is ~0 in isolation)
so = mr[(mr.task == "direction") & (mr.features == "sentiment_only") &
        (~mr.model.str.startswith("baseline"))]
macros["sentOnlyMaxMCC"] = fmt(so["MCC"].abs().max(), 3)

# ---------------- VIF ----------------
vif = jload("vif.json")
macros["vifClose"] = f"{max(vif['raw_OHLCV_levels'].values()):,.0f}"
macros["vifStatMax"] = f"{max(vif['stationary_features_used'].values()):,.0f}"

# ---------------- EDA ----------------
try:
    eda = jload("eda_report.json")
    macros["edaKurtosis"] = f"{eda['distribution']['excess_kurtosis']:,.1f}"
    macros["edaCrossCorr"] = fmt(eda["cross_sectional_return_corr"]["mean"], 2)
    macros["edaAcfClose"] = fmt(eda["autocorrelation"]["acf_close_lag1"], 3)
    macros["edaAcfReturn"] = fmt(eda["autocorrelation"]["acf_return_lag1"], 3)
    macros["edaFeatTargetMax"] = fmt(eda["feature_target_corr"]["max_abs"], 3)
    macros["edaSentTargetMax"] = fmt(eda["feature_target_corr"]["sentiment_max_abs"], 3)
    macros["edaLevelsNonstat"] = fmt(eda["stationarity"]["levels_pct_nonstationary_p>0.05"] * 100, 0)
    macros["edaReturnsStat"] = fmt(eda["stationarity"]["returns_pct_stationary_p<0.05"] * 100, 0)
    macros["edaUpRate"] = fmt(eda["direction_up_rate"] * 100, 1)
except Exception as e:
    print("EDA macros skipped:", e)

# ---------------- dataset facts (from panel manifest if present) ----------------
man_path = os.path.join(ROOT, "results", "feature_manifest.json")
if os.path.exists(man_path):
    man = json.load(open(man_path))
    macros["nTickers"] = str(man["n_tickers"])
    macros["nObs"] = f"{man['n_rows']:,}"
    macros["newsCov"] = fmt(man["news_coverage"] * 100, 1)
    macros["dateMin"] = man["date_min"][:4]
    macros["dateMax"] = man["date_max"][:4]

# ============================================================================
# XAI-centric outputs (brief PAPER_MISSION_BRIEF.md): model zoo, ground-truth
# faithfulness benchmark, and the XAI method catalogue. Each number below
# traces to a committed results/metrics/*.json|csv file from a real Modal run.
# ============================================================================

def jload_opt(name):
    p = os.path.join(MET, name)
    return json.load(open(p)) if os.path.exists(p) else None


def _faith_fields(d):
    """Extract the common faithfulness sub-dict from any of the XAI JSON
    shapes used across s5/s5b-f (top-level fields, nested under
    'faithfulness', or nested under 'feature_axis_via_integrated_gradients'
    for the CAM family, which only has a per-timestep curve natively and
    borrows a per-feature IG vector on the same GT-augmented model)."""
    if d is None:
        return None
    if "signal_rank" in d:
        return d
    if "faithfulness" in d and isinstance(d["faithfulness"], dict):
        return d["faithfulness"]
    if "feature_axis_via_integrated_gradients" in d and isinstance(d["feature_axis_via_integrated_gradients"], dict):
        return d["feature_axis_via_integrated_gradients"]
    return None


# ---------------- model zoo (RQ3: model-class dependence) ----------------
seq_files = {
    "GRU": "lstm_results.csv", "LSTM": "lstm_model_results.csv",
    "CNN1d": "cnn1d_results.csv", "TCN": "tcn_results.csv",
    "Transformer": "transformer_results.csv", "GNN": "gnn_results.csv",
}
zoo_rows = []
mr_dir = mr[(mr.task == "direction") & (mr.features == "price+sentiment") &
            (~mr.model.str.startswith("baseline"))]
for m in sorted(mr_dir.model.unique()):
    sub = mr_dir[mr_dir.model == m]
    zoo_rows.append({"model": m, "family": "classical/ML", "Acc": sub["Acc"].mean(),
                     "F1": sub["F1"].mean(), "MCC": sub["MCC"].mean(), "n_folds": len(sub)})
seq_family = {"GRU": "recurrent", "LSTM": "recurrent", "CNN1d": "conv",
              "TCN": "conv", "Transformer": "attention", "GNN": "graph"}
for name, fname in seq_files.items():
    p = os.path.join(MET, fname)
    if not os.path.exists(p):
        continue
    sdf = pd.read_csv(p)
    sub = sdf[(sdf.task == "direction") & (sdf.features == "price+sentiment")]
    if len(sub) == 0:
        continue
    zoo_rows.append({"model": name, "family": seq_family[name], "Acc": sub["Acc"].mean(),
                     "F1": sub["F1"].mean(), "MCC": sub["MCC"].mean(), "n_folds": len(sub)})
zoo = pd.DataFrame(zoo_rows).sort_values(["family", "model"])

# ---- M6 (R3 revision): clustered (by-symbol) 95% CIs on Acc/F1/MCC, where the
# per-row predictions needed for it exist. predictions.parquet only caches
# per-observation predictions for 3 of the 16 zoo models (XGBoost, LightGBM,
# Logistic -- the subset s3_experiments.py::main saves for significance
# testing); for those 3, bootstrap real clustered CIs matching the same
# protocol as Table VI (5000 resamples, resample whole symbols, percentile
# 95%). For the remaining 13 models we do not have per-symbol predictions
# cached and do not fabricate CIs for them -- instead Table IV states the
# caveat explicitly (main.tex) rather than implying uniform treatment.
import numpy as np

CI_SEED, CI_N_BOOTSTRAP = 42, 5000
ci_by_model = {}
pred_path = os.path.join(MET, "predictions.parquet")
if os.path.exists(pred_path):
    pred = pd.read_parquet(pred_path)
    pd_dir = pred[(pred.task == "direction") & (pred.features == "price+sentiment")]
    rng = np.random.RandomState(CI_SEED)
    for m in sorted(pd_dir.model.unique()):
        sub = pd_dir[pd_dir.model == m]
        # per-symbol confusion counts, precomputed once -- a bootstrap draw is
        # just a resampled SUM of these counts (vectorized below), not a
        # per-resample sklearn call, since y/pred are binary (0/1).
        tp, fp, tn, fn = [], [], [], []
        for s, g in sub.groupby("Symbol"):
            y, p = g["y"].values.astype(int), g["pred"].values.astype(int)
            tp.append(int(((y == 1) & (p == 1)).sum())); fp.append(int(((y == 0) & (p == 1)).sum()))
            tn.append(int(((y == 0) & (p == 0)).sum())); fn.append(int(((y == 1) & (p == 0)).sum()))
        tp, fp, tn, fn = (np.array(a, dtype=np.float64) for a in (tp, fp, tn, fn))
        n = len(tp)
        if n < 5:
            continue
        draws = rng.choice(n, size=(CI_N_BOOTSTRAP, n), replace=True)   # (n_bootstrap, n_symbols)
        TP, FP, TN, FN = tp[draws].sum(1), fp[draws].sum(1), tn[draws].sum(1), fn[draws].sum(1)
        total = TP + FP + TN + FN
        acc = np.divide(TP + TN, total, out=np.zeros_like(total), where=total > 0)
        f1_denom = 2 * TP + FP + FN
        f1 = np.divide(2 * TP, f1_denom, out=np.zeros_like(f1_denom), where=f1_denom > 0)
        mcc_denom = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        mcc = np.divide(TP * TN - FP * FN, mcc_denom, out=np.zeros_like(mcc_denom), where=mcc_denom > 0)
        boot = {"Acc": acc, "F1": f1, "MCC": mcc}
        ci_by_model[m] = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
                          for k, v in boot.items()}
        ci_by_model[m]["n_symbols"] = n
    with open(os.path.join(MET, "model_zoo_clustered_ci.json"), "w") as f:
        json.dump(ci_by_model, f, indent=2)
    print(f"model zoo clustered CIs (M6): computed for {list(ci_by_model)} "
          f"({CI_N_BOOTSTRAP} resamples, whole-symbol, matching Table VI's protocol)")
else:
    print("model zoo clustered CIs (M6) skipped: predictions.parquet not found")

def _ci_cell(model_name, metric_col, point_val):
    d = ci_by_model.get(model_name)
    if d is None:
        return f"{fmt(point_val,3)}"
    lo, hi = d[metric_col]
    half_width = (hi - lo) / 2   # compact +/- notation instead of full bracket range
    return f"{fmt(point_val,3)}$\\pm${fmt(half_width,3)}"

zoo.to_csv(os.path.join(TAB, "model_zoo_summary.csv"), index=False)
with open(os.path.join(TAB, "tab_model_zoo.tex"), "w") as f:
    f.write("\\begin{tabular}{@{}llrrr@{}}\n\\toprule\n")
    f.write("Model & Family & Acc & F1 & MCC \\\\\n\\midrule\n")
    for _, r in zoo.iterrows():
        f.write(f"{r['model']} & {r['family']} & {_ci_cell(r['model'],'Acc',r['Acc'])} & "
                f"{_ci_cell(r['model'],'F1',r['F1'])} & {_ci_cell(r['model'],'MCC',r['MCC'])} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
macros["nModelsZooCiScored"] = str(len(ci_by_model))
macros["nModelsZoo"] = str(len(zoo))
macros["modelZooMaxMCC"] = fmt(zoo["MCC"].abs().max(), 3)
macros["modelZooMeanAcc"] = fmt(zoo["Acc"].mean(), 3)
print(f"model zoo: {len(zoo)} models, mean Acc {zoo['Acc'].mean():.3f}, "
      f"max |MCC| {zoo['MCC'].abs().max():.3f} (near-chance across the board "
      f"is the expected, honest result on this near-zero-signal forecaster)")

# ---------------- ground-truth faithfulness benchmark (the headline) ----------------
fb = jload_opt("faithfulness_benchmark.json")
if fb:
    macros["faithUniverseN"] = str(fb["universe_n_symbols"])
    macros["faithRhoBar"] = fmt(fb["cross_sectional_rho_bar"], 3)
    macros["faithEffN"] = fmt(fb["effective_n"], 2)
    with open(os.path.join(TAB, "tab_faithfulness_benchmark.tex"), "w") as f:
        f.write("\\begin{tabular}{@{}lrrrrrr@{}}\n\\toprule\n")
        f.write("Method & Margin (l.)$\\pm$95\\%CI & \\%Sym (l.) & MDES (l.) & "
                "Margin (nl) & \\%Sym (nl) & Credit (ex./part.) \\\\\n\\midrule\n")
        for m in fb["methods"]:
            if "mean_signal_minus_noise_margin" not in m:
                continue
            ci = m["ci95_clustered"]
            half_width = (ci[1] - ci[0]) / 2
            nl = m.get("nonlinear")
            nl_margin = fmt(nl["mean_signal_minus_noise_margin"], 3) if nl else "--"
            nl_pct = fmt(nl["pct_symbols_signal_above_noise"] * 100, 1) if nl else "--"
            credit_exact = fmt(m.get("mean_redundant_copy_credit_share", float("nan")), 2)
            credit_partial = fmt(m.get("mean_redundant_partial_credit_share", float("nan")), 2)
            f.write(f"{m['method']} & {fmt(m['mean_signal_minus_noise_margin'],3)}$\\pm${fmt(half_width,3)} & "
                    f"{fmt(m['pct_symbols_signal_above_noise']*100,1)} & "
                    f"{fmt(m.get('mdes_approx', float('nan')),3)} & "
                    f"{nl_margin} & {nl_pct} & "
                    f"{credit_exact}/{credit_partial} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    # M5 (R3 revision): a second compact table for the broadened ground-truth
    # grid -- weak (rho=0.05) linear-signal recovery, the harder near-noise-
    # floor condition, alongside the strong-signal numbers already above for
    # direct comparison ("floor, not ceiling").
    weak_rows = [m for m in fb["methods"] if m.get("weak")]
    if weak_rows:
        with open(os.path.join(TAB, "tab_faithfulness_weak_signal.tex"), "w") as f:
            f.write("\\begin{tabular}{@{}lrrrr@{}}\n\\toprule\n")
            f.write("Method & Margin & 95\\% CI & \\%Sym & MDES \\\\\n\\midrule\n")
            for m in weak_rows:
                w = m["weak"]
                ci = w["ci95_clustered"]
                f.write(f"{m['method']} & {fmt(w['mean_signal_minus_noise_margin'],3)} & "
                        f"$[{fmt(ci[0],3)},{fmt(ci[1],3)}]$ & "
                        f"{fmt(w['pct_symbols_signal_above_noise']*100,1)} & "
                        f"{fmt(w.get('mdes_approx', float('nan')),3)} \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n")
        macros["nWeakSignalScored"] = str(len(weak_rows))

    # per-method MDES-vs-margin multiple, for the prose ("clears its MDES by Nx")
    for m in fb["methods"]:
        if "mean_signal_minus_noise_margin" in m and m.get("mdes_approx"):
            key = "mdesRatio" + "".join(w.capitalize() for w in m["method"].split("_"))
            macros[key] = fmt(m["mean_signal_minus_noise_margin"] / m["mdes_approx"], 1)
            macros["margin" + "".join(w.capitalize() for w in m["method"].split("_"))] = fmt(m["mean_signal_minus_noise_margin"], 3)
            macros["mdes" + "".join(w.capitalize() for w in m["method"].split("_"))] = fmt(m["mdes_approx"], 4)
        if "pct_symbols_signal_above_noise" in m:
            macros["pctSym" + "".join(w.capitalize() for w in m["method"].split("_"))] = fmt(m["pct_symbols_signal_above_noise"] * 100, 1)
            pci = m.get("pct_symbols_ci95_clustered")
            if pci:
                macros["pctSymCiLo" + "".join(w.capitalize() for w in m["method"].split("_"))] = fmt(pci[0] * 100, 1)
                macros["pctSymCiHi" + "".join(w.capitalize() for w in m["method"].split("_"))] = fmt(pci[1] * 100, 1)

    # M2: paired-test macros (permutation vs treeshap/kernelshap/lime)
    paired = fb.get("paired_tests_permutation_vs_shap_lime", [])
    with open(os.path.join(TAB, "tab_paired_tests.tex"), "w") as f:
        f.write("\\begin{tabular}{@{}llrrr@{}}\n\\toprule\n")
        f.write("A & B & $n$ & McNemar $p$ & Wilcoxon $p$ \\\\\n\\midrule\n")
        for pt in paired:
            if "mcnemar_p" not in pt:
                continue
            f.write(f"{pt['method_a']} & {pt['method_b']} & {pt['n_common_symbols']} & "
                    f"{fmt_p(pt['mcnemar_p'])} & {fmt_p(pt['wilcoxon_signed_rank_p'])} \\\\\n")
            key = "".join(w.capitalize() for w in pt["method_b"].split("_"))
            macros[f"mcnemarP{key}"] = fmt_p(pt["mcnemar_p"])
            macros[f"wilcoxonP{key}"] = fmt_p(pt["wilcoxon_signed_rank_p"])
        f.write("\\bottomrule\n\\end{tabular}\n")

    print(f"faithfulness benchmark: N={fb['universe_n_symbols']} stocks, "
          f"rho_bar={fb['cross_sectional_rho_bar']:.3f}, effective N={fb['effective_n']:.2f}")
    for m in fb["methods"]:
        if "pct_symbols_signal_above_noise" in m:
            print(f"  {m['method']}: {m['pct_symbols_signal_above_noise']*100:.1f}% of symbols "
                  f"rank signal above noise, CI excludes 0: "
                  f"{m['ci95_clustered'][0] > 0 or m['ci95_clustered'][1] < 0}")
            if m.get("nonlinear"):
                nl = m["nonlinear"]
                print(f"    nonlinear: {nl['pct_symbols_signal_above_noise']*100:.1f}% of symbols, "
                      f"CI excludes 0: {nl['ci95_clustered'][0] > 0 or nl['ci95_clustered'][1] < 0}")
            if m.get("weak"):
                w = m["weak"]
                print(f"    weak: {w['pct_symbols_signal_above_noise']*100:.1f}% of symbols, "
                      f"CI excludes 0: {w['ci95_clustered'][0] > 0 or w['ci95_clustered'][1] < 0}")

# ---------------- M4 (R3 revision): RQ2 pairwise agreement matrix ----------------
agr = jload_opt("xai_agreement_matrix.json")
if agr and agr.get("methods_in_matrix"):
    methods = agr["methods_in_matrix"]
    macros["nAgreementMethods"] = str(len(methods))
    rhos = agr["mean_spearman_rho"]
    off_diag = [rhos[m1][m2] for m1 in methods for m2 in methods
                if m1 != m2 and rhos[m1][m2] is not None]
    if off_diag:
        macros["agreementRhoMin"] = fmt(min(off_diag), 2)
        macros["agreementRhoMax"] = fmt(max(off_diag), 2)
        macros["agreementRhoMean"] = fmt(sum(off_diag) / len(off_diag), 2)
    with open(os.path.join(TAB, "tab_agreement_matrix.tex"), "w") as f:
        f.write("\\begin{tabular}{@{}l" + "r" * len(methods) + "@{}}\n\\toprule\n")
        f.write("Method & " + " & ".join(tex_escape(m) for m in methods) + " \\\\\n\\midrule\n")
        for m1 in methods:
            row = [fmt(rhos[m1][m2], 2) if rhos[m1][m2] is not None else "--" for m2 in methods]
            f.write(f"{tex_escape(m1)} & " + " & ".join(row) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    macros["nAgreementExcluded"] = str(len(agr.get("excluded_no_shared_model_class", [])))
    print(f"agreement matrix (M4/RQ2): {len(methods)} methods, mean off-diag rho "
          f"{macros.get('agreementRhoMean','?')}, range [{macros.get('agreementRhoMin','?')},"
          f"{macros.get('agreementRhoMax','?')}]")

# ---------------- XAI method catalogue (all ~26 methods, one row each) ----------------
# family, source file(s), applicable model(s) -- brief S14 taxonomy.
catalogue_spec = [
    ("treeshap", "attribution", "xai_by_model_treeshap.json", "5 tree models"),
    ("kernelshap", "attribution", "xai_by_model_agnostic.json", "Logistic/RF/XGBoost/EBM/SVM"),
    ("lime", "attribution", "xai_by_model_agnostic.json", "Logistic/RF/XGBoost/EBM/SVM"),
    ("permutation", "attribution", "xai_by_model_agnostic.json", "Logistic/RF/XGBoost/EBM/SVM"),
    ("integrated_gradients", "gradient", "xai_by_model_ig.json", "GRU/LSTM/CNN1d/TCN/Transformer"),
    ("rise", "attribution (adapted)", "xai_rule_methods.json", "XGBoost"),
    ("ebm_intrinsic", "intrinsic", "xai_concept_intrinsic.json", "EBM"),
    ("gam_aggregation", "global aggregation", "xai_concept_intrinsic.json", "XGBoost (TreeSHAP-derived)"),
    ("muse_brcg", "rule surrogate", "xai_concept_intrinsic.json", "XGBoost"),
    ("lrp", "gradient/propagation", "xai_gradient_cam.json", "MLP (GRU: N/A, no supported layers)"),
    ("guided_backprop", "gradient/propagation", "xai_gradient_cam.json", "MLP"),
    ("deconvnet", "gradient/propagation", "xai_gradient_cam.json", "MLP"),
    ("gnnexplainer", "graph", "xai_graph.json", "GNN"),
]
cat_rows = []
for method, family, fname, models in catalogue_spec:
    d = jload_opt(fname)
    scored = []   # entries that actually carry a real ground-truth verdict
    if d is not None:
        if isinstance(d, list):
            scored = [x for x in d if x.get("method") == method
                      and "ranks_signal_above_noise" in x]
        elif method in d:
            fe = _faith_fields(d[method])
            if fe is not None and "ranks_signal_above_noise" in fe:
                scored = [fe]
        elif "feature_importance" in d and d["feature_importance"].get("method") == method:
            fe = d["feature_importance"]
            if "ranks_signal_above_noise" in fe:
                scored = [fe]
        elif d.get("method") == method and "ranks_signal_above_noise" in d:
            scored = [d]
    if scored:
        n_yes = sum(1 for e in scored if e["ranks_signal_above_noise"])
        verdict = f"{n_yes}/{len(scored)} yes" if len(scored) > 1 else ("yes" if n_yes else "no")
        credits = [e["redundant_copy_credit_share"] for e in scored
                   if e.get("redundant_copy_credit_share") == e.get("redundant_copy_credit_share")]  # drop NaN
        # M3 (R3 revision): every row in THIS table (including treeshap/kernelshap/
        # lime/permutation) is a single-fit credit share, averaged across the
        # method's applicable MODELS (not stocks) -- see xai_by_model_*.json,
        # written by s5_xai.py's per-model _faithfulness(), never s7_faithfulness.py's
        # stock-clustered bootstrap. None of this table's numbers are stock-
        # clustered; the REAL stock-clustered credit share for the core 4 methods
        # is reported separately in Table VI. Tag accordingly so the two axes of
        # averaging (across models here vs. across stocks there) are never
        # conflated.
        credit = (f"{fmt(sum(credits) / len(credits), 2)} (avg./{len(credits)} models)"
                  if len(credits) > 1 else f"{fmt(credits[0], 2)}" if credits else "--")
        nl_scored = [e for e in scored if "ranks_nonlinear_signal_above_noise" in e]
        if nl_scored:
            nl_yes = sum(1 for e in nl_scored if e["ranks_nonlinear_signal_above_noise"])
            nl_verdict = (f"{nl_yes}/{len(nl_scored)} yes" if len(nl_scored) > 1
                         else ("yes" if nl_yes else "no"))
        else:
            nl_verdict = "--"
    else:
        verdict, credit, nl_verdict = "n/a (no gt feats in artifact)", "--", "--"
    cat_rows.append({"method": method, "family": family, "models": models,
                     "ranks_signal_above_noise": verdict, "redundant_copy_credit_share": credit,
                     "nl_faithful": nl_verdict})
# Tier-2/N-A and qualitative-verdict methods (no single global importance vector,
# or explicitly evaluated on other criteria per brief S14 -- reported, not hidden).
rule_d = jload_opt("xai_rule_methods.json") or {}
if "anchors" in rule_d:
    a = rule_d["anchors"]
    cat_rows.append({"method": "anchors", "family": "rule/example", "models": "XGBoost",
                     "ranks_signal_above_noise": f"precision={fmt(a['mean_precision'],2)}",
                     "redundant_copy_credit_share": f"coverage={fmt(a['mean_coverage'],3)}"})
if "counterfactual" in rule_d:
    c = rule_d["counterfactual"]
    cat_rows.append({"method": "counterfactual", "family": "rule/example", "models": "XGBoost",
                     "ranks_signal_above_noise": f"validity={fmt(c['validity'],2)}",
                     "redundant_copy_credit_share": f"sparsity={fmt(c['mean_sparsity_features_changed'],2)}"})
if "cem" in rule_d:
    c = rule_d["cem"]
    cat_rows.append({"method": "cem", "family": "rule/example", "models": "XGBoost",
                     "ranks_signal_above_noise": f"PP size={fmt(c['pertinent_positive_mean_size'],2)}",
                     "redundant_copy_credit_share": "--"})
concept_d = jload_opt("xai_concept_intrinsic.json") or {}
if "tcav" in concept_d and isinstance(concept_d["tcav"], list):
    for c in concept_d["tcav"]:
        cname = c.get("concept", c.get("name", "concept"))
        cat_rows.append({"method": f"tcav ({cname})", "family": "concept", "models": "GRU",
                         "ranks_signal_above_noise": f"TCAV={fmt(c.get('tcav_score', float('nan')),2)}",
                         "redundant_copy_credit_share": f"CAV acc={fmt(c.get('cav_train_accuracy', float('nan')),2)}"})
gc_d = jload_opt("xai_gradient_cam.json") or {}
for cam in ("gradcam", "gradcam_pp", "score_cam"):
    if cam in gc_d:
        fe = _faith_fields(gc_d[cam])
        if fe is not None and "ranks_signal_above_noise" in fe:
            verdict = "yes" if fe["ranks_signal_above_noise"] else "no"
            credit = fmt(fe.get("redundant_copy_credit_share", float("nan")), 2)
        else:
            verdict, credit = "n/a (no gt feats in artifact)", "--"
        cat_rows.append({"method": cam, "family": "CAM (vision-native)", "models": "CNN1d only",
                         "ranks_signal_above_noise": verdict,
                         "redundant_copy_credit_share": credit})
mech_d = jload_opt("xai_mechanistic.json") or {}
if "sae" in mech_d:
    s = mech_d["sae"]
    cat_rows.append({"method": "sae", "family": "mechanistic", "models": "Transformer",
                     "ranks_signal_above_noise": f"{s['n_interpretable_latents']}/{s.get('hidden_dim','?')} interp.",
                     "redundant_copy_credit_share": f"thr.\\ $|r|{{>}}${s.get('correlation_threshold','?')}"})
if "acdc_ablation" in mech_d:
    a = mech_d["acdc_ablation"]
    cat_rows.append({"method": "acdc (ablation)", "family": "mechanistic", "models": "Transformer",
                     "ranks_signal_above_noise": f"{a.get('n_notable','?')} notable/10",
                     "redundant_copy_credit_share": f"z{{>}}{a.get('notable_z_threshold','?')}"})

# ---------------- M7 (R3 revision): mechanistic seed/fold sensitivity ----------------
sens = jload_opt("xai_mechanistic_sensitivity.json")
if sens:
    macros["nSensRuns"] = str(len(sens["runs"]))
    pi = sens["pct_interpretable_latents_across_runs"]
    macros["sensPctInterpMin"] = fmt(pi["min"], 1)
    macros["sensPctInterpMax"] = fmt(pi["max"], 1)
    macros["sensPctInterpMean"] = fmt(pi["mean"], 1)
    nz = sens["n_notable_ablation_components_across_runs"]
    macros["nSensRunsWithNotable"] = str(sum(1 for v in nz if v > 0))
    macros["sensMaxZMin"] = fmt(min(r["max_ablation_z_score"] for r in sens["runs"]), 2)
    macros["sensMaxZMax"] = fmt(sens["max_ablation_z_across_runs"]["max"], 2)
    macros["nSensDistinctTop"] = str(sens["n_distinct_top_components"])
    tops = sens["top_component_per_run"]
    from collections import Counter
    top_counts = Counter(tops)
    majority_component, majority_n = top_counts.most_common(1)[0]
    macros["sensMajorityComponent"] = tex_escape(majority_component)
    macros["sensMajorityN"] = str(majority_n)
    macros["sensTotalRuns"] = str(len(tops))
    print(f"mechanistic sensitivity (M7): {len(sens['runs'])} runs, "
          f"interpretable-latent share {pi['min']:.1f}-{pi['max']:.1f}%, "
          f"{sum(1 for v in nz if v>0)}/{len(nz)} runs found a notable component, "
          f"majority top component: {majority_component} ({majority_n}/{len(tops)} runs)")

catalogue = pd.DataFrame(cat_rows)
if "nl_faithful" not in catalogue.columns:
    catalogue["nl_faithful"] = "--"
catalogue["nl_faithful"] = catalogue["nl_faithful"].fillna("--")
catalogue.to_csv(os.path.join(TAB, "xai_method_catalogue.csv"), index=False)
# B5 (R4 revision): the dense 6-column catalogue table was split into two --
# an applicability matrix (method/family/applicable models) and a results
# table (faithfulness verdicts + note) -- per reviewer feedback that mixing
# both axes in one table was too dense to read.
with open(os.path.join(TAB, "tab_method_applicability.tex"), "w") as f:
    f.write("\\begin{tabular}{@{}lll@{}}\n\\toprule\n")
    f.write("Method & Family & Applicable model(s) \\\\\n\\midrule\n")
    for _, r in catalogue.iterrows():
        f.write(f"{tex_escape(r['method'])} & {tex_escape(r['family'])} & {tex_escape(r['models'])} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
with open(os.path.join(TAB, "tab_method_catalogue.tex"), "w") as f:
    f.write("\\begin{tabular}{@{}llll@{}}\n\\toprule\n")
    f.write("Method & Faithful (linear) & Faithful (nonlinear) & Note \\\\\n\\midrule\n")
    for _, r in catalogue.iterrows():
        f.write(f"{tex_escape(r['method'])} & "
                f"{r['ranks_signal_above_noise']} & {r['nl_faithful']} & {r['redundant_copy_credit_share']} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
macros["nXaiMethodsCatalogued"] = str(len(catalogue))
def _is_scoreable(v):
    return v == "yes" or v == "no" or v.endswith(" yes")


def _is_fully_faithful(v):
    if v == "yes":
        return True
    if v.startswith("no") or v.startswith("n/a"):
        return False
    # "k/n yes" -- count as faithful only if ALL scored entries agreed
    k, n = v.split("/")[0], v.split("/")[1].split()[0]
    return k == n


scoreable = catalogue["ranks_signal_above_noise"].map(_is_scoreable)
n_scoreable = int(scoreable.sum())
n_faithful = int(catalogue.loc[scoreable, "ranks_signal_above_noise"].map(_is_fully_faithful).sum())
macros["nXaiFaithfulOfScoreable"] = f"{n_faithful}/{n_scoreable}"
print(f"XAI method catalogue: {len(catalogue)} methods; "
      f"{n_faithful}/{n_scoreable} scoreable methods rank gt_signal above gt_noise")

# ---------------- R4 revision: A1 TCAV null, A2 SAE/ablation null, ----------------
# ---------------- A3 year-robustness, A4 jackknife -------------------------------
def _sci(x):
    """Format a small p-value as LaTeX scientific notation, e.g. 5.5e-05 ->
    '5.5\\times10^{-5}'. Used inline in math mode ($p=\\recurPBinomOne$)."""
    if x == 0:
        return "0"
    exp = math.floor(math.log10(abs(x)))
    mant = x / (10 ** exp)
    return f"{mant:.1f}\\times10^{{{exp}}}"

tcav_sig = jload_opt("xai_tcav_significance.json")
if tcav_sig:
    ps = [c["p_empirical"] for c in tcav_sig["concepts"]]
    means = sorted(c["random_mean"] for c in tcav_sig["concepts"])
    macros["tcavPEmp"] = fmt(max(ps), 2)
    macros["tcavNullMeanRange"] = f"{fmt(means[0], 2)}--{fmt(means[-1], 2)}"
    print(f"A1 TCAV null: p_empirical={ps}, random means={means}")

sae_null = jload_opt("xai_sae_null.json")
if sae_null:
    macros["saeNullMean"] = fmt(sae_null["null_interpretable_mean"], 2)
    macros["saeNullPHi"] = fmt(sae_null["null_interpretable_p95"], 2)
    macros["saeNullMax"] = str(int(sae_null["null_interpretable_max"]))
    print(f"A2 SAE null: observed={sae_null['observed_interpretable']}/{sae_null['n_latents']}, "
          f"null_mean={sae_null['null_interpretable_mean']}, null_p95={sae_null['null_interpretable_p95']}")

abl_null = jload_opt("xai_ablation_null.json")
if abl_null:
    macros["ablNullMean"] = fmt(abl_null["null_notable_count_mean"], 2)
    macros["ablNullPHi"] = fmt(abl_null["null_notable_count_p95"], 2)
    rec = abl_null["recurrence"]
    macros["recurC"] = str(rec["C"])
    macros["recurK"] = str(rec["k"])
    macros["recurR"] = str(rec["R"])
    macros["recurPBinomOne"] = _sci(rec["p_binom_one"])
    macros["recurPUnion"] = _sci(rec["p_union"])
    print(f"A2 ablation null: null_notable_mean={abl_null['null_notable_count_mean']}, "
          f"recurrence k={rec['k']}/{rec['R']} of C={rec['C']}, "
          f"p_binom_one={rec['p_binom_one']}, p_union={rec['p_union']}")

by_year = jload_opt("faithfulness_by_year.json")
if by_year:
    sp_ts = by_year["spread"]["treeshap"]
    macros["yearMarginMinTreeshap"] = fmt(sp_ts["margin_min"], 3)
    macros["yearMarginMaxTreeshap"] = fmt(sp_ts["margin_max"], 3)
    macros["yearCopyCreditMinTreeshap"] = fmt(sp_ts["copy_credit_min"], 3)
    macros["yearCopyCreditMaxTreeshap"] = fmt(sp_ts["copy_credit_max"], 3)
    sp_perm = by_year["spread"]["permutation"]
    macros["yearPctSymMinPermutation"] = fmt(sp_perm["pct_symbols_correct_min"] * 100, 1)
    macros["yearPctSymMaxPermutation"] = fmt(sp_perm["pct_symbols_correct_max"] * 100, 1)
    print(f"A3 year robustness: treeshap margin range [{sp_ts['margin_min']:.3f},{sp_ts['margin_max']:.3f}], "
          f"permutation pct-correct range [{sp_perm['pct_symbols_correct_min']:.3f},"
          f"{sp_perm['pct_symbols_correct_max']:.3f}]")

jack = jload_opt("faithfulness_jackknife.json")
if jack:
    jts = jack["methods"]["treeshap"]
    margin_delta_pct = abs(jts["jackknife_mean_margin"] - jts["full_margin"]) / jts["full_margin"] * 100
    mdes_delta_pct = (jts["jackknife_mdes_mean"] - jts["full_mdes"]) / jts["full_mdes"] * 100
    macros["jackMarginDeltaTreeshap"] = fmt(margin_delta_pct, 2)
    macros["jackMdesDeltaTreeshap"] = fmt(mdes_delta_pct, 1)
    print(f"A4 jackknife: treeshap margin delta {margin_delta_pct:.3f}%, MDES delta {mdes_delta_pct:.1f}%")

# ---------------- rewrite numbers.tex with the full macro set ----------------
with open(os.path.join(PAP, "numbers.tex"), "w") as f:
    f.write("% auto-generated by make_tables.py -- do not edit\n")
    for k, v in macros.items():
        f.write(f"\\renewcommand{{\\{k}}}{{{v}}}\n")

print("\nWrote numbers.tex and tables. Key macros:")
for k, v in macros.items():
    print(f"  \\{k} = {v}")
