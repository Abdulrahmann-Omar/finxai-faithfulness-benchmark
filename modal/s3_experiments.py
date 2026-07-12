"""Stage 3 — the core experiments (classical models + baselines).

  modal run s3_experiments.py::leakage_demo   # reproduce & debunk the R2~0.99
  modal run s3_experiments.py::main           # honest return/direction, walk-forward
  modal run s3_experiments.py::stats          # sentiment ablation CIs + significance

Everything writes metrics JSON + figures to /data/results on the Volume.
"""
import modal
from _common import image, data_vol, DATA, SEED, EMBARGO_DAYS, TEST_YEARS

app = modal.App("finsent-s3-exp", image=image)

BUILD = f"{DATA}/build"
RES = f"{DATA}/results"
FIG = f"{RES}/figures"
MET = f"{RES}/metrics"

# ---- feature groups ------------------------------------------------------
PRICE_FEATS = ["ret_lag1", "ret_lag2", "ret_lag3", "ret_lag5",
               "vol_roll5", "vol_roll20", "rsi14", "ma_gap20",
               "hl_range", "co_ret", "logvol", "vol_z20"]
SENT_FEATS = ["fb_score", "fb_pos", "fb_neg", "lm_score", "news_count",
              "has_news", "fb_roll3", "fb_roll5", "news_roll5", "fb_chg"]
LEVEL_FEATS = ["Open", "High", "Low", "Close", "Adj_close", "Volume",
               "hl_range", "co_ret", "fb_score", "news_count"]


def _prep_fig_dirs():
    import os
    for d in (FIG, MET):
        os.makedirs(d, exist_ok=True)


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 150, "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "figure.autolayout": True,
    })
    return plt


# ==========================================================================
@app.function(volumes={DATA: data_vol}, cpu=8.0, memory=32768, timeout=60 * 40)
def leakage_demo():
    """Show that the reported R2~0.99 is an autocorrelation artifact of
    (random split + level targets), not forecasting skill."""
    import os, json, numpy as np, pandas as pd
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error
    _prep_fig_dirs(); plt = _mpl()

    df = pd.read_parquet(os.path.join(BUILD, "panel.parquet")).copy()
    df = df.sort_values(["Date", "Symbol"]).reset_index(drop=True)
    feats = LEVEL_FEATS.copy()
    X = df[feats].copy()
    X["Symbol_le"] = LabelEncoder().fit_transform(df["Symbol"])   # the flawed encoding
    y = df["Close_next"].values

    out = {}

    # --- A1: original flawed protocol (RANDOM shuffle split) ---
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)
    rf = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=SEED)
    rf.fit(Xtr, ytr); pr = rf.predict(Xte)
    out["A1_random_split_RF_levels"] = {
        "R2": float(r2_score(yte, pr)), "RMSE": float(mean_squared_error(yte, pr) ** .5)}

    # --- A2: same features/target but CHRONOLOGICAL split ---
    n = len(df); cut = int(n * 0.8)
    Xtr2, Xte2 = X.iloc[:cut], X.iloc[cut:]
    ytr2, yte2 = y[:cut], y[cut:]
    rf2 = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=SEED)
    rf2.fit(Xtr2, ytr2); pr2 = rf2.predict(Xte2)
    out["A2_chrono_split_RF_levels"] = {
        "R2": float(r2_score(yte2, pr2)), "RMSE": float(mean_squared_error(yte2, pr2) ** .5)}

    # --- A3: naive persistence baseline (predict Close_next = Close_today) ---
    persist = df["Close"].values
    out["A3_persistence_levels_ALL"] = {
        "R2": float(r2_score(y, persist)), "RMSE": float(mean_squared_error(y, persist) ** .5)}
    out["A3_persistence_levels_TESTonly"] = {
        "R2": float(r2_score(yte2, df["Close"].values[cut:])),
        "RMSE": float(mean_squared_error(yte2, df["Close"].values[cut:]) ** .5)}

    # --- A4: the SAME model, but predicting RETURNS (chronological) ---
    yr = df["ret_next"].values
    Xr = df[PRICE_FEATS].copy()
    sc = StandardScaler().fit(Xr.iloc[:cut])
    Xr_s = sc.transform(Xr)
    rf3 = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=SEED)
    rf3.fit(Xr_s[:cut], yr[:cut]); pr3 = rf3.predict(Xr_s[cut:])
    out["A4_chrono_RF_returns"] = {
        "R2": float(r2_score(yr[cut:], pr3)),
        "RMSE": float(mean_squared_error(yr[cut:], pr3) ** .5)}

    with open(os.path.join(MET, "leakage_demo.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))

    # --- figure: the collapse ---
    labels = ["RF\nrandom split\n(levels)", "RF\ntemporal split\n(levels)",
              "Persistence\nbaseline\n(levels)", "RF\ntemporal split\n(RETURNS)"]
    vals = [out["A1_random_split_RF_levels"]["R2"],
            out["A2_chrono_split_RF_levels"]["R2"],
            out["A3_persistence_levels_TESTonly"]["R2"],
            out["A4_chrono_RF_returns"]["R2"]]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#C0362C", "#B4740F", "#566B86", "#2C7A55"]
    bars = ax.bar(labels, vals, color=colors, width=0.62)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + (0.02 if v >= 0 else -0.06),
                f"{v:.3f}", ha="center", fontweight="bold", fontsize=10)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Test $R^2$")
    ax.set_title("The reported $R^2\\approx0.99$ is autocorrelation, not skill", fontweight="bold")
    ax.set_ylim(min(-0.15, min(vals) - 0.1), 1.05)
    fig.savefig(os.path.join(FIG, "fig_leakage.png"), bbox_inches="tight")
    print("saved fig_leakage.png")
    data_vol.commit()


# ==========================================================================
def _make_models(task, seed):
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.svm import SVR, SVC
    from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
    from sklearn.neural_network import MLPRegressor, MLPClassifier
    import xgboost as xgb, lightgbm as lgb
    from catboost import CatBoostRegressor, CatBoostClassifier
    from interpret.glassbox import ExplainableBoostingRegressor, ExplainableBoostingClassifier
    if task == "reg":
        return {
            "Ridge": Ridge(alpha=1.0),
            "DecisionTree": DecisionTreeRegressor(max_depth=6, min_samples_leaf=50, random_state=seed),
            "RandomForest": RandomForestRegressor(n_estimators=300, max_depth=8,
                             n_jobs=-1, random_state=seed),
            "XGBoost": xgb.XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.03,
                        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                        n_jobs=-1, random_state=seed),
            "LightGBM": lgb.LGBMRegressor(n_estimators=400, num_leaves=31, max_depth=5,
                         learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                         reg_lambda=1.0, n_jobs=-1, random_state=seed, verbose=-1),
            "CatBoost": CatBoostRegressor(iterations=400, depth=5, learning_rate=0.03,
                         random_state=seed, verbose=False, allow_writing_files=False),
            "SVR": SVR(C=1.0, gamma="scale"),
            "KNN": KNeighborsRegressor(n_neighbors=25, n_jobs=-1),
            "EBM": ExplainableBoostingRegressor(random_state=seed, n_jobs=-1),
            "MLP": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=300,
                    early_stopping=True, random_state=seed),
        }
    return {
        "Logistic": LogisticRegression(C=1.0, max_iter=2000),
        "DecisionTree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=50,
                         random_state=seed, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=8,
                         n_jobs=-1, random_state=seed, class_weight="balanced"),
        "XGBoost": xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.03,
                    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                    n_jobs=-1, random_state=seed, eval_metric="logloss"),
        "LightGBM": lgb.LGBMClassifier(n_estimators=400, num_leaves=31, max_depth=5,
                     learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                     reg_lambda=1.0, n_jobs=-1, random_state=seed, verbose=-1),
        "CatBoost": CatBoostClassifier(iterations=400, depth=5, learning_rate=0.03,
                     random_state=seed, verbose=False, allow_writing_files=False),
        "SVC": SVC(C=1.0, gamma="scale", probability=False),
        "KNN": KNeighborsClassifier(n_neighbors=25, n_jobs=-1),
        "EBM": ExplainableBoostingClassifier(random_state=seed, n_jobs=-1),
        "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300,
                early_stopping=True, random_state=seed),
    }


def _reg_metrics(y, p, y_train_mean):
    import numpy as np
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    return {
        "R2": float(r2_score(y, p)),
        "RMSE": float(mean_squared_error(y, p) ** 0.5),
        "MAE": float(mean_absolute_error(y, p)),
        "DirAcc": float((np.sign(p) == np.sign(y)).mean()),
    }


def _clf_metrics(y, p, proba=None):
    from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
    d = {"Acc": float(accuracy_score(y, p)),
         "F1": float(f1_score(y, p, zero_division=0)),
         "MCC": float(matthews_corrcoef(y, p))}
    if proba is not None:
        try: d["AUC"] = float(roc_auc_score(y, proba))
        except Exception: d["AUC"] = float("nan")
    return d


@app.function(volumes={DATA: data_vol}, cpu=32.0, memory=131072, timeout=60 * 240)
def main():
    """Honest tasks (return regression, direction classification) with
    chronological walk-forward across regimes; baselines; feature-set ablation.
    Saves per-fold predictions for downstream significance testing."""
    import os, json, numpy as np, pandas as pd
    from sklearn.preprocessing import StandardScaler
    _prep_fig_dirs()

    df = pd.read_parquet(os.path.join(BUILD, "panel.parquet")).copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Date", "Symbol"]).reset_index(drop=True)
    sym_oh = pd.get_dummies(df["Symbol"], prefix="sym").astype(float)
    base_feats = list(sym_oh.columns)
    df = pd.concat([df, sym_oh], axis=1)

    feature_sets = {
        "price_only": base_feats + PRICE_FEATS,
        "sentiment_only": base_feats + SENT_FEATS,
        "price+sentiment": base_feats + PRICE_FEATS + SENT_FEATS,
    }

    # walk-forward: expanding window; test on each calendar year 2021-2023
    test_years = TEST_YEARS
    rows = []                     # metric rows
    pred_records = []             # per-observation predictions for stats

    EMBARGO = pd.Timedelta(days=EMBARGO_DAYS)   # gap so train targets can't overlap test window
    for ty in test_years:
        test_start = pd.Timestamp(year=ty, month=1, day=1)
        tr = df[df["Date"] < (test_start - EMBARGO)]   # embargoed expanding window
        te = df[df["Date"].dt.year == ty]
        if len(te) == 0 or len(tr) < 500:
            continue
        # ---- baselines ----
        # returns
        # persistence uses today's realized return `ret`, which can be NaN on a
        # RETURN_CLIP-masked outlier day (the row still survives the panel's
        # price_feats dropna, since ret_lag1..5 reference PRIOR days, not this
        # one) -- rare with 6-12 tickers, real at 57 tickers/9 years. Evaluate
        # persistence only on rows where it's actually defined.
        pmask = te["ret"].notna().values
        r2_base = {
            "zero": _reg_metrics(te["ret_next"].values, np.zeros(len(te)), 0.0),
            "train_mean": _reg_metrics(te["ret_next"].values,
                          np.full(len(te), tr["ret_next"].mean()), 0.0),
            "persistence_ret": _reg_metrics(te["ret_next"].values[pmask], te["ret"].values[pmask], 0.0),
        }
        if (~pmask).sum():
            print(f"  [note] fold {ty}: {(~pmask).sum()} row(s) with masked same-day "
                  f"return excluded from the persistence baseline only")
        for name, m in r2_base.items():
            rows.append({"task": "return", "fold": ty, "model": f"baseline:{name}",
                         "features": "-", **m})
        # direction
        maj = int(tr["dir_next"].mode().iloc[0])
        rows.append({"task": "direction", "fold": ty, "model": "baseline:majority",
                     "features": "-", **_clf_metrics(te["dir_next"].values,
                     np.full(len(te), maj), np.full(len(te), 0.5))})
        rows.append({"task": "direction", "fold": ty, "model": "baseline:always_up",
                     "features": "-", **_clf_metrics(te["dir_next"].values,
                     np.ones(len(te), dtype=int), np.full(len(te), 0.5))})

        # ---- models x feature sets ----
        for fs_name, feats in feature_sets.items():
            sc = StandardScaler().fit(tr[feats].values)
            Xtr, Xte = sc.transform(tr[feats].values), sc.transform(te[feats].values)

            # regression (returns)
            for mname, model in _make_models("reg", SEED).items():
                if mname in ("SVR", "KNN") and len(Xtr) > 9000:
                    idx = np.random.RandomState(SEED).choice(len(Xtr), 9000, replace=False)
                    model.fit(Xtr[idx], tr["ret_next"].values[idx])
                else:
                    model.fit(Xtr, tr["ret_next"].values)
                p = model.predict(Xte)
                rows.append({"task": "return", "fold": ty, "model": mname,
                             "features": fs_name,
                             **_reg_metrics(te["ret_next"].values, p, tr["ret_next"].mean())})
                if mname in ("XGBoost", "LightGBM", "Ridge"):
                    for sym, yt, pp in zip(te["Symbol"].values, te["ret_next"].values, p):
                        pred_records.append({"task": "return", "fold": ty, "model": mname,
                                             "features": fs_name, "Symbol": sym,
                                             "y": float(yt), "pred": float(pp)})

            # classification (direction)
            for mname, model in _make_models("clf", SEED).items():
                if mname in ("SVC", "KNN") and len(Xtr) > 9000:
                    idx = np.random.RandomState(SEED).choice(len(Xtr), 9000, replace=False)
                    model.fit(Xtr[idx], tr["dir_next"].values[idx])
                else:
                    model.fit(Xtr, tr["dir_next"].values)
                p = model.predict(Xte)
                proba = None
                if hasattr(model, "predict_proba"):
                    try: proba = model.predict_proba(Xte)[:, 1]
                    except Exception: proba = None
                rows.append({"task": "direction", "fold": ty, "model": mname,
                             "features": fs_name,
                             **_clf_metrics(te["dir_next"].values, p, proba)})
                if mname in ("XGBoost", "LightGBM", "Logistic"):
                    for sym, yt, pp in zip(te["Symbol"].values, te["dir_next"].values, p):
                        pred_records.append({"task": "direction", "fold": ty, "model": mname,
                                             "features": fs_name, "Symbol": sym,
                                             "y": int(yt), "pred": int(pp)})
        print(f"fold {ty}: train={len(tr):,} test={len(te):,} done")

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(MET, "main_results.csv"), index=False)
    pd.DataFrame(pred_records).to_parquet(os.path.join(MET, "predictions.parquet"), index=False)

    # aggregate across folds
    agg = (res.groupby(["task", "model", "features"])
           .agg(["mean", "std"]).reset_index())
    agg.columns = ["_".join([c for c in col if c]).strip("_") for col in agg.columns]
    agg.to_csv(os.path.join(MET, "main_results_agg.csv"), index=False)
    print(res.to_string())
    print("\nsaved main_results.csv, main_results_agg.csv, predictions.parquet")
    _figures_main(res)
    data_vol.commit()


def _figures_main(res):
    import os, numpy as np, pandas as pd
    plt = _mpl()

    # --- returns R2: models vs baselines (mean over folds) ---
    r = res[res.task == "return"]
    piv = r.groupby(["model", "features"])["R2"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    models = [m for m in ["Ridge", "RandomForest", "XGBoost", "LightGBM", "SVR"]]
    fss = ["price_only", "price+sentiment", "sentiment_only"]
    cmap = {"price_only": "#566B86", "price+sentiment": "#2F4B94", "sentiment_only": "#B4740F"}
    w = 0.26
    x = np.arange(len(models))
    for i, fs in enumerate(fss):
        vals = [piv[(piv.model == m) & (piv.features == fs)]["R2"].mean() for m in models]
        ax.bar(x + (i - 1) * w, vals, w, label=fs, color=cmap[fs])
    base = res[(res.task == "return") & (res.model.str.startswith("baseline"))]
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(models)
    ax.set_ylabel("Test $R^2$ on next-day return")
    ax.set_title("Next-day RETURN prediction: near-zero skill for every model", fontweight="bold")
    ax.legend(fontsize=9, loc="lower left")
    fig.savefig(os.path.join(FIG, "fig_returns_r2.png"), bbox_inches="tight")

    # --- direction MCC ---
    d = res[res.task == "direction"]
    piv2 = d.groupby(["model", "features"])["MCC"].mean().reset_index()
    models2 = ["Logistic", "RandomForest", "XGBoost", "LightGBM", "SVC"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(models2))
    for i, fs in enumerate(fss):
        vals = [piv2[(piv2.model == m) & (piv2.features == fs)]["MCC"].mean() for m in models2]
        ax.bar(x + (i - 1) * w, vals, w, label=fs, color=cmap[fs])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(models2)
    ax.set_ylabel("Test MCC on next-day direction")
    ax.set_title("Next-day DIRECTION prediction: MCC near 0 (chance)", fontweight="bold")
    ax.legend(fontsize=9)
    fig.savefig(os.path.join(FIG, "fig_direction_mcc.png"), bbox_inches="tight")

    # --- walk-forward direction accuracy across regimes ---
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for fs in ["price_only", "price+sentiment"]:
        sub = d[(d.model == "XGBoost") & (d.features == fs)].sort_values("fold")
        ax.plot(sub["fold"], sub["Acc"], marker="o", label=f"XGBoost ({fs})",
                color=cmap[fs], lw=2)
    bmaj = d[d.model == "baseline:majority"].sort_values("fold")
    ax.plot(bmaj["fold"], bmaj["Acc"], marker="s", ls="--", color="#566B86",
            label="majority baseline")
    ax.set_xticks([2021, 2022, 2023])
    ax.set_ylabel("Directional accuracy"); ax.set_xlabel("Test year (expanding window)")
    ax.set_title("Walk-forward accuracy across regimes", fontweight="bold")
    ax.legend(fontsize=9)
    fig.savefig(os.path.join(FIG, "fig_walkforward.png"), bbox_inches="tight")
    print("saved fig_returns_r2.png, fig_direction_mcc.png, fig_walkforward.png")


# ==========================================================================
@app.function(volumes={DATA: data_vol}, cpu=8.0, memory=32768, timeout=60 * 40)
def stats():
    """Quantify the marginal value of sentiment while respecting the data's
    dependence structure (the TA's concern):
      * direction: symbol-CLUSTERED bootstrap CI (cross-sectional correlation)
        + Wilcoxon signed-rank on per-symbol effects.
      * returns: Diebold-Mariano with Newey-West HAC standard errors, computed
        per symbol (temporal autocorrelation), aggregated across symbols.
    """
    import os, json, numpy as np, pandas as pd
    from scipy import stats as st
    _prep_fig_dirs(); plt = _mpl()

    def newey_west_t(d, maxlags=5):
        """HAC (Newey-West, Bartlett kernel) t-stat for H0: mean(d)=0.
        Robust to serial correlation in the loss differential."""
        d = np.asarray(d, float)
        n = len(d)
        dbar = d.mean()
        e = d - dbar
        gamma0 = np.dot(e, e) / n
        S = gamma0
        for l in range(1, min(maxlags, n - 1) + 1):
            w = 1.0 - l / (maxlags + 1.0)
            cov = np.dot(e[l:], e[:-l]) / n
            S += 2.0 * w * cov
        var_mean = S / n
        return float(dbar / np.sqrt(var_mean)) if var_mean > 0 else 0.0

    pred = pd.read_parquet(os.path.join(MET, "predictions.parquet"))
    rng = np.random.RandomState(SEED)
    results = {}

    # ---- direction: per (model,fold,Symbol) accuracy; symbol-clustered bootstrap ----
    dd = pred[pred.task == "direction"].copy()
    dd["correct"] = (dd["y"] == dd["pred"]).astype(int)
    grp = dd.groupby(["model", "fold", "Symbol", "features"])["correct"].mean().reset_index()
    ablation = {}
    for model in dd["model"].unique():
        a = grp[(grp.model == model) & (grp.features == "price+sentiment")]
        b = grp[(grp.model == model) & (grp.features == "price_only")]
        m = a.merge(b, on=["model", "fold", "Symbol"], suffixes=("_all", "_price"))
        m["diff"] = m["correct_all"] - m["correct_price"]
        if len(m) < 3:
            continue
        # per-symbol mean effect -> the clustering unit
        per_sym = m.groupby("Symbol")["diff"].mean()
        syms = per_sym.index.values
        # CLUSTERED bootstrap: resample whole symbols (respects cross-sectional dep.)
        boot = []
        for _ in range(5000):
            pick = rng.choice(len(syms), len(syms), replace=True)
            boot.append(per_sym.values[pick].mean())
        lo, hi = np.percentile(boot, [2.5, 97.5])
        try: w_p = float(st.wilcoxon(per_sym.values).pvalue)
        except Exception: w_p = float("nan")
        ablation[model] = {
            "mean_delta_acc": float(per_sym.mean()),
            "ci95_clustered": [float(lo), float(hi)],
            "wilcoxon_p_by_symbol": w_p,
            "n_symbols": int(len(syms)),
            "pct_symbols_positive": float((per_sym.values > 0).mean()),
        }
    results["direction_sentiment_ablation"] = ablation

    # ---- returns: Diebold-Mariano with Newey-West HAC, per symbol ----
    rr = pred[pred.task == "return"].copy()
    dm = {}
    for model in rr["model"].unique():
        a = rr[(rr.model == model) & (rr.features == "price+sentiment")]
        b = rr[(rr.model == model) & (rr.features == "price_only")]
        merged = a.merge(b, on=["fold", "Symbol", "y"], suffixes=("_all", "_price"))
        stats_by_sym, reductions = [], []
        for sym, g in merged.groupby("Symbol"):
            d = ((g["y"] - g["pred_price"]) ** 2 - (g["y"] - g["pred_all"]) ** 2).values
            if len(d) < 30:
                continue
            stats_by_sym.append(newey_west_t(d, maxlags=5))
            reductions.append(float(d.mean()))
        stats_by_sym = np.array(stats_by_sym)
        dm[model] = {
            "mean_mse_reduction": float(np.mean(reductions)) if reductions else 0.0,
            "median_HAC_dm_t": float(np.median(stats_by_sym)) if len(stats_by_sym) else 0.0,
            "pct_symbols_sig_improve": float((stats_by_sym > 1.96).mean()) if len(stats_by_sym) else 0.0,
            "pct_symbols_sig_worse": float((stats_by_sym < -1.96).mean()) if len(stats_by_sym) else 0.0,
            "n_symbols": int(len(stats_by_sym)),
        }
    results["return_diebold_mariano_HAC"] = dm

    with open(os.path.join(MET, "ablation_stats.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))

    # ---- figure: sentiment Δaccuracy with 95% CI ----
    if ablation:
        models = list(ablation.keys())
        means = [ablation[m]["mean_delta_acc"] for m in models]
        los = [ablation[m]["mean_delta_acc"] - ablation[m]["ci95_clustered"][0] for m in models]
        his = [ablation[m]["ci95_clustered"][1] - ablation[m]["mean_delta_acc"] for m in models]
        fig, ax = plt.subplots(figsize=(7.5, 4.3))
        y = np.arange(len(models))
        ax.errorbar([x * 100 for x in means], y, xerr=[[l * 100 for l in los], [h * 100 for h in his]],
                    fmt="o", color="#2F4B94", capsize=4, lw=2, markersize=7)
        ax.axvline(0, color="#C0362C", ls="--", lw=1.2)
        ax.set_yticks(y); ax.set_yticklabels(models)
        ax.set_xlabel("Δ directional accuracy from adding sentiment (percentage points)")
        ax.set_title("Marginal value of news sentiment (95% bootstrap CI)", fontweight="bold")
        fig.savefig(os.path.join(FIG, "fig_ablation.png"), bbox_inches="tight")
        print("saved fig_ablation.png")
    data_vol.commit()


# ==========================================================================
@app.function(volumes={DATA: data_vol}, cpu=8.0, memory=32768, timeout=60 * 30)
def diagnostics():
    """Collinearity diagnostics (the TA's concern): show that raw OHLCV levels
    are near-perfectly collinear (huge VIF) while the stationary return-based
    features we actually use for forecasting are well-conditioned. Also emit a
    correlation heatmap."""
    import os, json, numpy as np, pandas as pd
    import seaborn as sns
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    _prep_fig_dirs(); plt = _mpl()

    df = pd.read_parquet(os.path.join(BUILD, "panel.parquet")).copy()
    level_block = ["Open", "High", "Low", "Close", "Adj_close", "Volume"]
    stat_block = PRICE_FEATS

    def vif_of(cols):
        X = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
        Xz = (X - X.mean()) / (X.std() + 1e-9)
        out = {}
        for i, c in enumerate(cols):
            try:
                v = float(variance_inflation_factor(Xz.values, i))
            except Exception:
                v = float("inf")
            out[c] = v
        return out

    vif = {"raw_OHLCV_levels": vif_of(level_block),
           "stationary_features_used": vif_of(stat_block)}
    with open(os.path.join(MET, "vif.json"), "w") as f:
        json.dump(vif, f, indent=2)
    print("VIF (raw levels):", json.dumps(vif["raw_OHLCV_levels"], indent=2))
    print("max VIF stationary:", max(vif["stationary_features_used"].values()))

    # --- VIF comparison bar (log scale) ---
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    lv = pd.Series(vif["raw_OHLCV_levels"]).sort_values(ascending=False)
    sv = pd.Series(vif["stationary_features_used"]).sort_values(ascending=False).head(8)
    labels = list(lv.index) + ["|"] + list(sv.index)
    vals = list(lv.clip(upper=1e6).values) + [np.nan] + list(sv.values)
    colors = ["#C0362C"] * len(lv) + ["white"] + ["#2C7A55"] * len(sv)
    xs = np.arange(len(labels))
    ax.bar(xs, [v if not np.isnan(v) else 0 for v in vals], color=colors)
    ax.set_yscale("log")
    ax.axhline(10, color="black", ls="--", lw=1, label="VIF=10 (usual threshold)")
    ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Variance Inflation Factor (log)")
    ax.set_title("Raw OHLCV levels are severely collinear (red); the stationary\n"
                 "features we forecast with (green) are well-conditioned", fontweight="bold", fontsize=11)
    ax.legend()
    fig.savefig(os.path.join(FIG, "fig_vif.png"), bbox_inches="tight")

    # --- correlation heatmap of features actually used ---
    allf = PRICE_FEATS + SENT_FEATS
    corr = df[allf].corr()
    fig, ax = plt.subplots(figsize=(10, 8.5))
    sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1, square=True,
                cbar_kws={"shrink": .7}, ax=ax, linewidths=.3)
    ax.set_title("Correlation of modelling features (price/technical + sentiment)", fontweight="bold")
    fig.savefig(os.path.join(FIG, "fig_corr_heatmap.png"), bbox_inches="tight")
    print("saved fig_vif.png, fig_corr_heatmap.png, vif.json")
    data_vol.commit()
