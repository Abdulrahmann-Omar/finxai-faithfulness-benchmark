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
paper_cat = {}   # method -> per-model recovery counts, for the compact table in the paper
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
        paper_cat[method] = {
            "models": [e.get("model") for e in scored if e.get("model")],
            "n": len(scored), "lin_yes": n_yes,
            "nl_n": len(nl_scored),
            "nl_yes": sum(1 for e in nl_scored if e["ranks_nonlinear_signal_above_noise"]),
        }
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
# The Grad-CAM family's native output is a per-time-step curve, which cannot
# rank a planted FEATURE against the noise feature. The stored per-feature
# score for these rows comes from an Integrated-Gradients pass on the same
# CNN: it is IG's result, not a result of the CAM methods, so the CAM rows are
# reported as "not scored natively" and are NOT counted among the scored
# methods (IG is already counted once, in its own row).
n_cam_proxy = 0
for cam in ("gradcam", "gradcam_pp", "score_cam"):
    if cam in gc_d:
        fe = _faith_fields(gc_d[cam])
        if fe is not None and "ranks_signal_above_noise" in fe:
            n_cam_proxy += 1
            note = (f"IG pass on the same CNN: signal "
                    f"{'above' if fe['ranks_signal_above_noise'] else 'not above'} noise, "
                    f"credit {fmt(fe.get('redundant_copy_credit_share', float('nan')), 2)}")
        else:
            note = "--"
        cat_rows.append({"method": cam, "family": "CAM (vision-native)", "models": "CNN1d only",
                         "ranks_signal_above_noise": "not scored natively",
                         "redundant_copy_credit_share": note})
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
    f.write("Method & Signal $>$ noise (linear) & Signal $>$ noise (sign-coded) & Note \\\\\n\\midrule\n")
    for _, r in catalogue.iterrows():
        f.write(f"{tex_escape(r['method'])} & "
                f"{r['ranks_signal_above_noise']} & {r['nl_faithful']} & {r['redundant_copy_credit_share']} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
# Distinct methods: TCAV appears once per concept in the catalogue rows, so
# rows are de-duplicated on the method name before counting.
import re
distinct_methods = sorted({re.sub(r"\s*\(.*\)$", "", m) for m in catalogue["method"]})
macros["nXaiMethodsCatalogued"] = str(len(distinct_methods))
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
n_nl_recovered = int(catalogue.loc[scoreable, "nl_faithful"].map(
    lambda v: _is_scoreable(v) and _is_fully_faithful(v)).sum())
macros["nXaiNativeScored"] = str(n_scoreable)        # methods scored on their own output
macros["nXaiRecoveredLinear"] = str(n_faithful)      # ...that rank gt_signal above gt_noise on every model
macros["nXaiRecoveredNl"] = str(n_nl_recovered)      # ...and gt_signal_nl above gt_noise on every model
macros["nXaiNotScored"] = str(len(distinct_methods) - n_scoreable)
macros["nXaiCamProxy"] = str(n_cam_proxy)
print(f"XAI method catalogue: {len(distinct_methods)} distinct methods ({len(catalogue)} rows); "
      f"{n_scoreable} scored on their own output; {n_faithful} rank gt_signal above gt_noise "
      f"on every model, {n_nl_recovered} do so for gt_signal_nl; "
      f"{n_cam_proxy} CAM rows carry only an IG proxy and are not counted")

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
    macros["ablNComponents"] = str(rec["C"])
    macros["ablNShuffles"] = str(abl_null["n_shuffles"])
    macros["ablThreshold"] = fmt(abl_null["notable_z_threshold"], 0)
    # The stored recurrence p-values (p_binom_one, p_union) assume that every
    # ablated component -- a single attention head or a whole encoder layer --
    # is equally likely to show the largest effect. A whole-layer ablation
    # removes all of that layer's heads, so the assumption does not hold and
    # the p-values are not exported to the paper.
    print(f"A2 ablation null: null_notable_mean={abl_null['null_notable_count_mean']}, "
          f"top component recurs {rec['k']}/{rec['R']} runs of C={rec['C']} candidates "
          f"(uniform-null p-values stored in the JSON are not used in the paper)")

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

# ============================================================================
# Conference-paper outputs. Everything below is derived from the stored
# results loaded above (no new computation on data): the compact tables used
# by paper/main.tex and one macro per number quoted in its text, so that no
# result is typed into the manuscript by hand.
# ============================================================================
import datetime as _dt


def _camel(name):
    return "".join(w.capitalize() for w in name.split("_"))


def _day(d):          # 3 Feb 2015
    return f"{d.day} {d:%b} {d.year}"


def _p_expr(x):       # used in math mode: $\macro$
    return "p<0.001" if x < 0.001 else f"p={x:.3f}"


def _ci(ci, d=3):
    return f"[{fmt(ci[0], d)},\\,{fmt(ci[1], d)}]"


# ---- evaluation protocol: the config, the stored folds, the panel manifest ----
cfg_text = open(os.path.join(ROOT, "config.xai.yaml"), encoding="utf-8").read()
embargo_days = int(re.search(r"^embargo_days:\s*(\d+)", cfg_text, re.M).group(1))
folds = sorted(int(y) for y in mr["fold"].unique())
macros["nFolds"] = str(len(folds))
macros["foldFirst"] = str(folds[0])
macros["foldLast"] = str(folds[-1])
macros["embargoDays"] = str(embargo_days)
macros["seedCfg"] = re.search(r"^seed:\s*(\d+)", cfg_text, re.M).group(1)
macros["returnClip"] = re.search(r"^return_clip:\s*([\d.]+)", cfg_text, re.M).group(1)
macros["minNewsDays"] = re.search(r"^min_news_days:\s*(\d+)", cfg_text, re.M).group(1)
macros["lookback"] = re.search(r"^lookback:\s*(\d+)", cfg_text, re.M).group(1)

gta = jload_opt("gt_artifacts_summary.json")   # the planted-feature models: feature list + scoring year
score_year = int(gta["test_year"]) if gta else folds[-1]
macros["scoreYear"] = str(score_year)
d_min = _dt.date.fromisoformat(man["date_min"])
d_max = _dt.date.fromisoformat(man["date_max"])
macros["panelStart"] = f"{d_min.day} {d_min:%B} {d_min.year}"
macros["panelEnd"] = f"{d_max.day} {d_max:%B} {d_max.year}"
with open(os.path.join(TAB, "tab_folds.tex"), "w") as f:
    f.write("\\begin{tabular}{@{}llll@{}}\n\\toprule\n")
    f.write("Fold & Training rows & Embargo & Test rows \\\\\n\\midrule\n")
    for i, y in enumerate(folds, 1):
        # s3_experiments.py: train = Date < (1 Jan y - embargo), test = calendar year y
        cutoff = _dt.date(y, 1, 1) - _dt.timedelta(days=embargo_days)
        train_end = cutoff - _dt.timedelta(days=1)
        emb_end = _dt.date(y - 1, 12, 31)
        f.write(f"{i} & {_day(d_min)} -- {_day(train_end)} & "
                f"{cutoff.day}--{emb_end.day} {emb_end:%b} {emb_end.year} & {y} \\\\\n")
    # s5*.py / s7_faithfulness.py: train = year < score_year, test = score_year, no embargo
    f.write("\\midrule\n")
    f.write(f"Recovery & {_day(d_min)} -- {_day(_dt.date(score_year - 1, 12, 31))} & none & {score_year} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

usel = jload_opt("universe_selection.json")
if usel:
    macros["nCandidates"] = str(usel["n_candidates"])

# ---- planted features (results/feature_manifest.json) ----
gtc = man["ground_truth_construction"]
macros["gtRhoStrong"] = fmt(gtc["gt_signal"]["target_correlation"], 2)
macros["gtRhoWeak"] = fmt(gtc["gt_signal_weak"]["target_correlation"], 2)
macros["gtRhoNl"] = fmt(gtc["gt_signal_nl"]["nonlinear_signal_strength_param"], 2)
macros["gtNlCorrDir"] = fmt(gtc["gt_signal_nl"]["realized_pointbiserial_corr_with_dir_next"], 2)
macros["gtNlCorrRet"] = fmt(gtc["gt_signal_nl"]["realized_pearson_corr_with_ret_next"], 2)
macros["gtPartialRho"] = fmt(gtc["gt_redundant_partial"]["target_corr"], 1)
macros["gtPartialRealized"] = fmt(gtc["gt_redundant_partial"]["realized_corr_with_source"], 2)
macros["gtSource"] = tex_escape(gtc["gt_redundant_copy"]["source_feature"])
if gta:
    feats_all = gta["features"]
    n_planted = sum(1 for c in feats_all if c.startswith("gt_"))
    n_sent = sum(1 for c in feats_all if c in set(man["sentiment_features"]))
    macros["nFeatTotal"] = str(len(feats_all))
    macros["nFeatPlanted"] = str(n_planted)
    macros["nFeatSent"] = str(n_sent)
    macros["nFeatPrice"] = str(len(feats_all) - n_planted - n_sent)

# ---- substrate: sentiment ablation per model, zoo accuracy as a percentage ----
for m_name, d in da.items():
    macros[f"sentDelta{m_name}"] = fmt(d["mean_delta_acc"] * 100, 2)
    macros[f"sentCIlo{m_name}"] = fmt(d["ci95_clustered"][0] * 100, 2)
    macros[f"sentCIhi{m_name}"] = fmt(d["ci95_clustered"][1] * 100, 2)
macros["nSentModels"] = str(len(da))
macros["nSentCiInclZero"] = str(sum(1 for d in da.values()
                                    if d["ci95_clustered"][0] < 0 < d["ci95_clustered"][1]))
macros["sentDeltaMin"] = fmt(min(d["mean_delta_acc"] for d in da.values()) * 100, 2)
macros["sentDeltaMax"] = fmt(max(d["mean_delta_acc"] for d in da.values()) * 100, 2)
macros["modelZooMeanAccPct"] = fmt(zoo["Acc"].mean() * 100, 1)

# ---- stock-level recovery of the four core methods (faithfulness_benchmark.json) ----
DISPLAY = {"treeshap": "TreeSHAP", "kernelshap": "KernelSHAP", "lime": "LIME",
           "permutation": "Permutation"}
if fb:
    core = [m for m in fb["methods"] if "mean_signal_minus_noise_margin" in m]
    macros["nCoreMethods"] = str(len(core))
    macros["nBootstrap"] = f"{fb['n_bootstrap']:,}".replace(",", "{,}")
    any_partial_credit = False
    with open(os.path.join(TAB, "tab_core_recovery.tex"), "w") as f:
        f.write("\\begin{tabular}{@{}l rl rl rl cc@{}}\n\\toprule\n")
        f.write(f" & \\multicolumn{{2}}{{c}}{{Linear signal ($\\rho{{=}}{macros['gtRhoStrong']}$)}} & "
                f"\\multicolumn{{2}}{{c}}{{Sign-coded signal}} & "
                f"\\multicolumn{{2}}{{c}}{{Weak linear signal ($\\rho{{=}}{macros['gtRhoWeak']}$)}} & "
                f"\\multicolumn{{2}}{{c}}{{Credit share of copy}} \\\\\n")
        f.write("\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(l){8-9}\n")
        f.write("Method & Stocks & Margin [95\\% interval] & Stocks & Margin [95\\% interval] & "
                "Stocks & Margin [95\\% interval] & Exact & Partial \\\\\n\\midrule\n")
        for m in core:
            k = _camel(m["method"])
            nl, wk = m["nonlinear"], m["weak"]
            n_all = m["n_symbols_scored"]
            n_cred = m.get("n_symbols_credit_split", n_all)
            n_cred_p = m.get("n_symbols_credit_split_partial", n_all)
            partial = n_cred < n_all or n_cred_p < n_all
            any_partial_credit = any_partial_credit or partial
            mark = "$^{\\dagger}$" if partial else ""
            f.write(
                f"{DISPLAY[m['method']]} & "
                f"{fmt(m['pct_symbols_signal_above_noise'] * 100, 1)}\\% & "
                f"${fmt(m['mean_signal_minus_noise_margin'], 3)}\\ {_ci(m['ci95_clustered'])}$ & "
                f"{fmt(nl['pct_symbols_signal_above_noise'] * 100, 1)}\\% & "
                f"${fmt(nl['mean_signal_minus_noise_margin'], 3)}\\ {_ci(nl['ci95_clustered'])}$ & "
                f"{fmt(wk['pct_symbols_signal_above_noise'] * 100, 1)}\\% & "
                f"${fmt(wk['mean_signal_minus_noise_margin'], 3)}\\ {_ci(wk['ci95_clustered'])}$ & "
                f"{fmt(m['mean_redundant_copy_credit_share'], 2)}{mark} & "
                f"{fmt(m['mean_redundant_partial_credit_share'], 2)}{mark} \\\\\n")
            macros["marginCiLo" + k] = fmt(m["ci95_clustered"][0], 3)
            macros["marginCiHi" + k] = fmt(m["ci95_clustered"][1], 3)
            macros["pctSymNl" + k] = fmt(nl["pct_symbols_signal_above_noise"] * 100, 1)
            macros["pctSymWeak" + k] = fmt(wk["pct_symbols_signal_above_noise"] * 100, 1)
            macros["pctSymWeakCiLo" + k] = fmt(wk["pct_symbols_ci95_clustered"][0] * 100, 1)
            macros["pctSymWeakCiHi" + k] = fmt(wk["pct_symbols_ci95_clustered"][1] * 100, 1)
            macros["marginWeak" + k] = fmt(wk["mean_signal_minus_noise_margin"], 3)
            macros["marginWeakCiLo" + k] = fmt(wk["ci95_clustered"][0], 3)
            macros["marginWeakCiHi" + k] = fmt(wk["ci95_clustered"][1], 3)
            macros["creditExact" + k] = fmt(m["mean_redundant_copy_credit_share"], 2)
            macros["creditPartial" + k] = fmt(m["mean_redundant_partial_credit_share"], 2)
            macros["nCredit" + k] = str(n_cred)
        f.write("\\bottomrule\n\\end{tabular}\n")
    # recovered = the stock-resampled interval of the margin lies above zero
    macros["nLinearRecovered"] = str(sum(1 for m in core if m["ci95_clustered"][0] > 0))
    macros["nNlRecovered"] = str(sum(1 for m in core if m["nonlinear"]["ci95_clustered"][0] > 0))
    macros["nWeakRecovered"] = str(sum(1 for m in core if m["weak"]["ci95_clustered"][0] > 0))
    macros["nAllStocksMethods"] = str(sum(
        1 for m in core if min(m["pct_symbols_signal_above_noise"],
                               m["nonlinear"]["pct_symbols_signal_above_noise"],
                               m["weak"]["pct_symbols_signal_above_noise"]) == 1.0))
    paired = [pt for pt in fb.get("paired_tests_permutation_vs_shap_lime", []) if "mcnemar_p" in pt]
    if paired:
        discord = {tuple(pt["mcnemar_table_perm_wrong_other_right__perm_right_other_wrong"]) for pt in paired}
        if len(discord) == 1:      # identical discordant counts against all three comparison methods
            perm_only_wrong, other_only_wrong = discord.pop()
            macros["mcnemarPermOnlyWrong"] = str(perm_only_wrong)
            macros["mcnemarOtherOnlyWrong"] = str(other_only_wrong)
        macros["mcnemarPExpr"] = _p_expr(max(pt["mcnemar_p"] for pt in paired))
        macros["wilcoxonPExpr"] = _p_expr(max(pt["wilcoxon_signed_rank_p"] for pt in paired))
        macros["nPairedComparisons"] = str(len(paired))

# ---- stability: other scoring years, leave-stocks-out, and the second scoring of the same year ----
if by_year:
    macros["yearRobustFirst"] = str(by_year["years"][0])
    macros["yearRobustLast"] = str(by_year["years"][-1])
    macros["nYearsRobust"] = str(len(by_year["years"]))
    for name, sp in by_year["spread"].items():
        k = _camel(name)
        macros["yearMarginMin" + k] = fmt(sp["margin_min"], 3)
        macros["yearMarginMax" + k] = fmt(sp["margin_max"], 3)
        macros["yearCopyCreditMin" + k] = fmt(sp["copy_credit_min"], 2)
        macros["yearCopyCreditMax" + k] = fmt(sp["copy_credit_max"], 2)
        macros["yearPctSymMin" + k] = fmt(sp["pct_symbols_correct_min"] * 100, 1)
        macros["yearPctSymMax" + k] = fmt(sp["pct_symbols_correct_max"] * 100, 1)
    if fb:
        # s7b re-scores the benchmark year with a different random draw of explained rows
        rerun = {}
        for m in core:
            b = by_year["by_year"].get(str(score_year), {}).get(m["method"])
            if b and "mean_signal_minus_noise_margin" in b:
                ref = m["mean_signal_minus_noise_margin"]
                rerun[m["method"]] = abs(b["mean_signal_minus_noise_margin"] - ref) / ref * 100
        for name, v in rerun.items():
            macros["rerunDelta" + _camel(name)] = fmt(v, 1)
        if rerun:
            macros["rerunDeltaMax"] = fmt(max(rerun.values()), 1)
if jack:
    macros["jackNDrop"] = str(jack["n_drop"])
    macros["jackNIter"] = str(jack["n_iter"])
    macros["jackMarginDeltaMax"] = fmt(max(
        abs(v["jackknife_mean_margin"] - v["full_margin"]) / abs(v["full_margin"]) * 100
        for v in jack["methods"].values()), 1)

# ---- agreement between rankings (xai_agreement_matrix.json) ----
if agr and agr.get("methods_in_matrix"):
    rh, nc = agr["mean_spearman_rho"], agr["n_common_models_per_pair"]
    names = agr["methods_in_matrix"]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            macros["agr" + _camel(a) + _camel(b)] = fmt(rh[a][b], 2)
    perm_pairs = [rh["permutation"][o] for o in names if o != "permutation"]
    other_pairs = [rh[a][b] for i, a in enumerate(names) for b in names[i + 1:]
                   if "permutation" not in (a, b)]
    macros["agrPermMin"] = fmt(min(perm_pairs), 2)
    macros["agrPermMax"] = fmt(max(perm_pairs), 2)
    macros["agrOthersMin"] = fmt(min(other_pairs), 2)
    macros["agrOthersMax"] = fmt(max(other_pairs), 2)
    common = [nc[a][b] for a in names for b in names if a != b]
    macros["agrCommonMin"] = str(min(common))
    macros["agrCommonMax"] = str(max(common))

# ---- concept-based and mechanistic outputs (exploratory section) ----
if tcav_sig:
    scores = [c["tcav_score"] for c in tcav_sig["concepts"]]
    macros["nTcavConcepts"] = str(len(scores))
    macros["tcavScoreMin"] = fmt(min(scores), 2)
    macros["tcavScoreMax"] = fmt(max(scores), 2)
    macros["tcavNRandom"] = str(tcav_sig["n_random_cavs"])
    macros["nTcavSignificant"] = str(sum(1 for c in tcav_sig["concepts"] if c["significant"]))
if sae_null:
    macros["saeObserved"] = str(sae_null["observed_interpretable"])
    macros["saeLatents"] = str(sae_null["n_latents"])
    macros["saeThreshold"] = fmt(sae_null["correlation_threshold"], 1)
    macros["saeNShuffles"] = str(sae_null["n_shuffles"])
if sens:
    tops = sens["top_component_per_run"]
    notable = [c for r in sens["runs"] for c in r["notable_components"]]
    macros["sensTopFullLayerN"] = str(sum(1 for t in tops if t.endswith("_full")))
    macros["sensNotableHeadN"] = str(sum(1 for c in notable if not c.endswith("_full")))
    macros["sensNotablePerRunMin"] = str(min(sens["n_notable_ablation_components_across_runs"]))
    macros["sensNotablePerRunMax"] = str(max(sens["n_notable_ablation_components_across_runs"]))
    macros["sensNSeeds"] = str(len({r["seed"] for r in sens["runs"]}))
    macros["sensNYears"] = str(len({r["test_year"] for r in sens["runs"]}))
    macros["sensMinorityN"] = str(len(tops) - top_counts.most_common(1)[0][1])
if "acdc_ablation" in mech_d:
    macros["ablOrigNotable"] = str(mech_d["acdc_ablation"].get("n_notable", "?"))

# ---- compact method table for the paper: scored on own output vs. not scored ----
ABBR = {"DecisionTree": "DT", "RandomForest": "RF", "XGBoost": "XGB", "LightGBM": "LGBM",
        "CatBoost": "CatB", "Logistic": "LR", "EBM": "EBM", "SVM": "SVM", "GRU": "GRU",
        "LSTM": "LSTM", "CNN1d": "CNN", "TCN": "TCN", "Transformer": "Trf"}
SCORED_SPEC = [   # catalogue key, name in the paper, cite key(s), model if a single fixed one
    ("treeshap", "TreeSHAP", "lundberg2020treeshap", None),
    ("kernelshap", "KernelSHAP", "lundberg2017shap", None),
    ("lime", "LIME", "ribeiro2016lime", None),
    ("permutation", "Permutation importance", "breiman2001random", None),
    ("integrated_gradients", "Integrated Gradients", "sundararajan2017ig", None),
    ("rise", "RISE, feature masking", "petsiuk2018rise", "XGB"),
    ("ebm_intrinsic", "EBM term importance", "nori2019interpretml", "EBM"),
    ("gam_aggregation", "Rank-aggregated TreeSHAP", None, "XGB"),
    # result key "muse_brcg": s5c_concept_intrinsic.py fits a depth-3 decision tree to the
    # XGBoost predictions; it is a tree surrogate, not an implementation of MUSE or BRCG.
    ("muse_brcg", "Decision-tree surrogate", None, "XGB"),
    ("lrp", "LRP", "bach2015lrp", "MLP"),
    ("guided_backprop", "Guided backpropagation", "springenberg2015guided", "MLP"),
    ("deconvnet", "Deconvnet", "zeiler2014deconv", "MLP"),
    ("gnnexplainer", "GNNExplainer", "ying2019gnnexplainer", "GNN"),
]
UNSCORED_SPEC = [  # catalogue key, name, cite key, model, why no recovery score
    ("anchors", "Anchors", "ribeiro2018anchors", "XGB", "rules"),
    ("counterfactual", "Counterfactual search", "wachter2018counterfactual", "XGB", "examples"),
    ("cem", "Contrastive explanation", "dhurandhar2018cem", "XGB", "examples"),
    ("gradcam", "Grad-CAM", "selvaraju2017gradcam", "CNN", "time-step curve"),
    ("gradcam_pp", "Grad-CAM++", "chattopadhay2018gradcampp", "CNN", "time-step curve"),
    ("score_cam", "Score-CAM", "wang2020scorecam", "CNN", "time-step curve"),
    ("tcav", "TCAV, two concepts", "kim2018tcav", "GRU", "Sec.~\\ref{sec:exploratory}"),
    ("sae", "Sparse autoencoder", "cunningham2024sae", "Trf", "Sec.~\\ref{sec:exploratory}"),
    # result key "acdc": a manual single-component ablation sweep, not the ACDC algorithm
    ("acdc", "Component ablation", None, "Trf", "Sec.~\\ref{sec:exploratory}"),
]
spec_keys = {s[0] for s in SCORED_SPEC} | {s[0] for s in UNSCORED_SPEC}
assert spec_keys == set(distinct_methods), (
    "paper method table is out of sync with the catalogue: "
    f"{sorted(spec_keys ^ set(distinct_methods))}")
assert {s[0] for s in SCORED_SPEC} == set(paper_cat), (
    "methods scored on their own output changed: "
    f"{sorted({s[0] for s in SCORED_SPEC} ^ set(paper_cat))}")


def _cite(keys):
    return f"~\\cite{{{keys}}}" if keys else ""


with open(os.path.join(TAB, "tab_method_recovery.tex"), "w") as f:
    f.write("\\begin{tabular}{@{}llcc@{}}\n\\toprule\n")
    f.write("Method & Model(s) & Linear & Sign-coded \\\\\n\\midrule\n")
    f.write(f"\\multicolumn{{4}}{{@{{}}l}}{{\\emph{{Scored on own output "
            f"({len(SCORED_SPEC)} methods)}}}} \\\\\n")
    for key, name, cite, fixed_model in SCORED_SPEC:
        pc = paper_cat[key]
        models = ", ".join(ABBR.get(m, m) for m in pc["models"]) if pc["models"] else fixed_model
        nl_cell = f"{pc['nl_yes']}/{pc['nl_n']}" if pc["nl_n"] else "--"
        f.write(f"{name}{_cite(cite)} & {models} & {pc['lin_yes']}/{pc['n']} & {nl_cell} \\\\\n")
    f.write("\\midrule\n")
    f.write(f"\\multicolumn{{4}}{{@{{}}l}}{{\\emph{{Not scored on feature recovery "
            f"({len(UNSCORED_SPEC)} methods)}}}} \\\\\n")
    for key, name, cite, model, why in UNSCORED_SPEC:
        f.write(f"{name}{_cite(cite)} & {model} & \\multicolumn{{2}}{{l}}{{{why}}} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# ---------------- paper/NUMBERS_AUDIT.md: the source of every number in main.tex ----------------
MACRO_SOURCES = [   # (regex on the macro name, where its value comes from)
    (r"^(nTickers|nObs|newsCov|dateMin|dateMax|panelStart|panelEnd)$", "results/feature_manifest.json"),
    (r"^gt", "results/feature_manifest.json (ground_truth_construction)"),
    (r"^(nFeat|scoreYear)", "results/metrics/gt_artifacts_summary.json"),
    (r"^(nFolds|foldFirst|foldLast)$", "results/metrics/main_results.csv (fold column)"),
    (r"^(embargoDays|returnClip|minNewsDays|lookback|seedCfg)$", "config.xai.yaml"),
    (r"^nCandidates$", "results/metrics/universe_selection.json"),
    (r"^(nModelsZoo|modelZoo|majAcc|bestDirAcc)", "results/metrics/main_results.csv and the per-model *_results.csv"),
    (r"^(sent|nSent)", "results/metrics/ablation_stats.json"),
    (r"^(faith|pctSym|margin|credit|nCredit|nCoreMethods|nBootstrap|nLinearRecovered|nNlRecovered|"
     r"nWeakRecovered|nAllStocksMethods|mcnemar|wilcoxon|nPairedComparisons)",
     "results/metrics/faithfulness_benchmark.json"),
    (r"^rerun", "results/metrics/faithfulness_by_year.json compared with faithfulness_benchmark.json"),
    (r"^(year|nYearsRobust)", "results/metrics/faithfulness_by_year.json"),
    (r"^jack", "results/metrics/faithfulness_jackknife.json"),
    (r"^agr", "results/metrics/xai_agreement_matrix.json"),
    (r"^nXai", "method catalogue built from results/metrics/xai_*.json (see xai_method_catalogue.csv)"),
    (r"^(tcav|nTcav)", "results/metrics/xai_tcav_significance.json"),
    (r"^sae", "results/metrics/xai_sae_null.json"),
    (r"^(sens|nSens)", "results/metrics/xai_mechanistic_sensitivity.json"),
    (r"^abl", "results/metrics/xai_ablation_null.json"),
]
# Numbers that are protocol constants of the pipeline code, not results. They
# are typed in main.tex and listed here with the line of code they restate.
LITERAL_SOURCES = {
    "14": "14-day RSI window: modal/s2_build.py (rsi14)",
    "20": "20-day windows of the moving-average gap and the volume z-score: modal/s2_build.py (ma_gap20, vol_z20)",
    "1": "'1 January' of the test year: modal/s3_experiments.py (test_start); '1D' convolutional network",
    "400": "XGBoost n_estimators: modal/s7_faithfulness.py (XGBClassifier)",
    "4": "XGBoost max_depth: modal/s7_faithfulness.py (XGBClassifier)",
    "0.03": "XGBoost learning_rate: modal/s7_faithfulness.py (XGBClassifier)",
    "0.8": "XGBoost subsample and colsample_bytree: modal/s7_faithfulness.py (XGBClassifier)",
    "52": "TreeSHAP rows per stock, max(50, 3000 // 57): modal/s7_faithfulness.py (per_sym_cap)",
    "40": "KernelSHAP rows per stock: modal/s7_faithfulness.py (per_sym_cap_k)",
    "50": "KernelSHAP background rows: modal/s7_faithfulness.py (shap.sample)",
    "150": "KernelSHAP perturbation samples per row: modal/s7_faithfulness.py (nsamples)",
    "10": "LIME rows per stock and permutation repeats: modal/s7_faithfulness.py (per_sym_n, n_repeats)",
    "95": "level of the percentile intervals (2.5th to 97.5th percentile): modal/s7_faithfulness.py",
}
tex_path = os.path.join(PAP, "main.tex")
if os.path.exists(tex_path):
    tex = open(tex_path, encoding="utf-8").read()
    body = tex[tex.index("\\maketitle"):]                                    # skip preamble and author block
    body = re.sub(r"(?<!\\)%.*", "", body)                                   # comments
    used = [m for m in dict.fromkeys(re.findall(r"\\([A-Za-z]+)", body)) if m in macros]

    def _source(name):
        for pat, src in MACRO_SOURCES:
            if re.search(pat, name):
                return src
        raise SystemExit(f"NUMBERS_AUDIT: macro \\{name} is used in main.tex but has no documented source")

    # literal numbers in running text: drop math, tables, references and the bibliography commands
    text = re.sub(r"\\begin\{(table\*?|tabular)\}.*?\\end\{\1\}", " ", body, flags=re.S)
    text = re.sub(r"\$[^$]*\$", " ", text)
    text = re.sub(r"\\(label|ref|cite|input|url|bibliography|bibliographystyle|section\*?|subsection)\{[^}]*\}", " ", text)
    text = re.sub(r"\\[A-Za-z]+", " ", text)
    literals = list(dict.fromkeys(re.findall(r"(?<![A-Za-z\d.])\d+(?:\.\d+)?(?![\d]*pt)", text)))
    unreviewed = [x for x in literals if x not in LITERAL_SOURCES]
    with open(os.path.join(PAP, "NUMBERS_AUDIT.md"), "w", encoding="utf-8") as f:
        f.write("# Where every number in the paper comes from\n\n"
                "Generated by `make_tables.py`; do not edit by hand. The paper quotes results only\n"
                "through the macros below, which are written to `paper/numbers.tex` from the stored\n"
                "results. Table bodies (`paper/tables/*.tex`) are generated by the same script.\n\n"
                "## Results (macros used in `main.tex`)\n\n| Macro | Value | Source |\n|---|---|---|\n")
        for m in used:
            f.write(f"| `\\{m}` | `{macros[m]}` | {_source(m)} |\n")
        f.write("\n## Protocol constants typed in the text\n\n"
                "These restate settings of the pipeline code; they are not results.\n\n"
                "| Number | What it is and where it is set |\n|---|---|\n")
        for x in literals:
            f.write(f"| {x} | {LITERAL_SOURCES.get(x, '**UNREVIEWED: not documented**')} |\n")
    print(f"NUMBERS_AUDIT.md: {len(used)} result macros used in main.tex, "
          f"{len(literals)} literal numbers ({len(unreviewed)} unreviewed: {unreviewed})")

# ---------------- rewrite numbers.tex with the full macro set ----------------
# \providecommand first, so main.tex needs no placeholder list for new macros.
with open(os.path.join(PAP, "numbers.tex"), "w") as f:
    f.write("% Generated by make_tables.py from results/ -- do not edit by hand.\n")
    for k, v in macros.items():
        f.write(f"\\providecommand{{\\{k}}}{{}}\\renewcommand{{\\{k}}}{{{v}}}\n")

print("\nWrote numbers.tex and tables. Key macros:")
for k, v in macros.items():
    print(f"  \\{k} = {v}")
