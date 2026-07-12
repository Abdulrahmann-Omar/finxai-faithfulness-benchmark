"""Stage 5 — modern, professional XAI, used to *quantify* (not just display)
how little the news-sentiment features contribute, and to measure faithfulness
against the P10 synthetic ground-truth features (brief S8/S9/S14).

  modal run s5_xai.py::run

  * SHAP TreeExplainer: beeswarm + global bar; sentiment attribution SHARE.
  * SHAP KernelExplainer (model-agnostic; independent of TreeSHAP).
  * LIME (local surrogate, aggregated to a global importance vector).
  * Permutation importance on the held-out test set.
  * Grouped block-permutation (correlation-robust).
  * ALE (Accumulated Local Effects) — the modern, correlation-robust successor
    to PDP — for a top price feature, the leading sentiment feature, and the
    ground-truth signal/noise features.
  * Integrated Gradients (Captum) on the GRU sequence model.
  * Ground-truth faithfulness: for every method that yields a global importance
    vector, does it rank gt_signal above gt_noise, and how much credit does it
    put on gt_redundant_copy vs. its source feature (credit-splitting test)?
"""
import modal
from _common import image, data_vol, DATA, SEED, GROUND_TRUTH

app = modal.App("finsent-s5-xai", image=image)

BUILD = f"{DATA}/build"
RES = f"{DATA}/results"
MET = f"{RES}/metrics"
FIG = f"{RES}/figures"
ART = f"{RES}/artifacts"

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


def _faithfulness(imp, method_name):
    """Score one method's global |importance| vector against the P10 ground
    truth: does it rank gt_signal > gt_noise (should), and how does it split
    credit between gt_redundant_copy and its source feature 'vol_z20'?"""
    out = {"method": method_name}
    n = len(imp)
    ranked = imp.sort_values(ascending=False)
    rank_of = {f: int(ranked.index.get_loc(f)) + 1 for f in imp.index}
    if "gt_signal" in imp.index and "gt_noise" in imp.index:
        out["signal_rank"] = rank_of["gt_signal"]
        out["noise_rank"] = rank_of["gt_noise"]
        out["n_features"] = n
        out["ranks_signal_above_noise"] = bool(rank_of["gt_signal"] < rank_of["gt_noise"])
    if "gt_signal_nl" in imp.index and "gt_noise" in imp.index:
        out["signal_nl_rank"] = rank_of["gt_signal_nl"]
        out["ranks_nonlinear_signal_above_noise"] = bool(rank_of["gt_signal_nl"] < rank_of["gt_noise"])
    if "gt_signal_weak" in imp.index and "gt_noise" in imp.index:
        # R3 revision (M5): a weak (rho=0.05) planted signal near the noise
        # floor -- the harder recovery test than gt_signal's rho=0.15.
        out["signal_weak_rank"] = rank_of["gt_signal_weak"]
        out["ranks_weak_signal_above_noise"] = bool(rank_of["gt_signal_weak"] < rank_of["gt_noise"])
    if "gt_redundant_copy" in imp.index and "vol_z20" in imp.index:
        a, b = float(imp["gt_redundant_copy"]), float(imp["vol_z20"])
        out["redundant_copy_importance"] = a
        out["source_feature_importance"] = b
        out["redundant_copy_credit_share"] = float(a / (a + b)) if (a + b) > 0 else float("nan")
    if "gt_redundant_partial" in imp.index and "vol_z20" in imp.index:
        # R3 revision (M5): partial (rho=0.8), not exact, collinearity --
        # tests whether the exact-duplicate credit-split pattern generalizes.
        ap, bp = float(imp["gt_redundant_partial"]), float(imp["vol_z20"])
        out["redundant_partial_importance"] = ap
        out["redundant_partial_credit_share"] = float(ap / (ap + bp)) if (ap + bp) > 0 else float("nan")
    return out


def _mpl():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 150, "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False})
    return plt


@app.function(volumes={DATA: data_vol}, gpu="a10g", cpu=8.0, memory=32768, timeout=60 * 60)
def run():
    import os, json, numpy as np, pandas as pd
    import shap
    from sklearn.preprocessing import StandardScaler
    from sklearn.inspection import permutation_importance
    import xgboost as xgb
    plt = _mpl()
    os.makedirs(FIG, exist_ok=True); os.makedirs(MET, exist_ok=True)

    feats = PRICE_FEATS + SENT_FEATS + GT_FEATS
    df = pd.read_parquet(os.path.join(BUILD, "panel.parquet")).copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Date", "Symbol"]).reset_index(drop=True)
    tr = df[df["Date"].dt.year < 2023]
    te = df[df["Date"].dt.year == 2023]
    sc = StandardScaler().fit(tr[feats].values)
    Xtr = pd.DataFrame(sc.transform(tr[feats].values), columns=feats)
    Xte = pd.DataFrame(sc.transform(te[feats].values), columns=feats)
    ytr, yte = tr["dir_next"].values, te["dir_next"].values

    model = xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.03,
             subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, n_jobs=-1,
             random_state=SEED, eval_metric="logloss")
    model.fit(Xtr.values, ytr)
    summary = {}
    faithfulness = []   # one entry per method that yields a global importance vector

    # ---------- SHAP (TreeExplainer) ----------
    expl = shap.TreeExplainer(model)
    Xs = Xte.sample(min(2000, len(Xte)), random_state=SEED)
    sv = expl.shap_values(Xs)
    if isinstance(sv, list):
        sv = sv[1]
    mean_abs = np.abs(sv).mean(0)
    imp = pd.Series(mean_abs, index=feats).sort_values(ascending=False)
    sent_share = float(imp[SENT_FEATS].sum() / imp.sum())
    summary["shap_sentiment_attribution_share"] = sent_share
    summary["shap_top10"] = imp.head(10).round(5).to_dict()
    summary["shap_sentiment_ranks"] = {f: int(imp.index.get_loc(f)) + 1 for f in SENT_FEATS}
    faithfulness.append(_faithfulness(imp, "treeshap"))
    print(f"SHAP sentiment attribution share: {sent_share:.4f}")

    plt.figure()
    shap.summary_plot(sv, Xs, plot_type="bar", max_display=18, show=False)
    plt.title(f"SHAP global importance — sentiment = {sent_share*100:.1f}% of total", fontsize=11)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_shap_bar.png"), bbox_inches="tight"); plt.close()

    plt.figure()
    shap.summary_plot(sv, Xs, max_display=18, show=False)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_shap_beeswarm.png"), bbox_inches="tight"); plt.close()

    # colored global bar highlighting sentiment vs price
    plt.figure(figsize=(7.5, 6))
    order = imp.sort_values()
    colors = ["#B4740F" if f in SENT_FEATS else "#2F4B94" for f in order.index]
    plt.barh(order.index, order.values, color=colors)
    plt.xlabel("mean(|SHAP value|)")
    plt.title(f"Feature attribution (orange = sentiment): {sent_share*100:.1f}% of total", fontsize=11)
    import matplotlib.patches as mp
    plt.legend(handles=[mp.Patch(color="#2F4B94", label="price/technical"),
                        mp.Patch(color="#B4740F", label="news sentiment")], loc="lower right")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_shap_grouped.png"), bbox_inches="tight"); plt.close()

    # ---------- KernelSHAP (model-agnostic; independent of TreeSHAP) ----------
    # Expensive (sampling-based): small background set + a modest test sample.
    try:
        bg = shap.sample(Xtr, 50, random_state=SEED)
        kexpl = shap.KernelExplainer(lambda X: model.predict_proba(X)[:, 1], bg)
        Xk = Xte.sample(min(200, len(Xte)), random_state=SEED)
        ksv = kexpl.shap_values(Xk, nsamples=200, silent=True)
        kimp = pd.Series(np.abs(ksv).mean(0), index=feats).sort_values(ascending=False)
        summary["kernelshap_top10"] = kimp.head(10).round(5).to_dict()
        summary["kernelshap_sentiment_share"] = float(kimp[SENT_FEATS].sum() / kimp.sum())
        faithfulness.append(_faithfulness(kimp, "kernelshap"))
        plt.figure(figsize=(7.5, 6))
        order = kimp.sort_values()
        colors = ["#B4740F" if f in SENT_FEATS else "#2F4B94" for f in order.index]
        plt.barh(order.index, order.values, color=colors)
        plt.xlabel("mean(|KernelSHAP value|)"); plt.title("KernelSHAP global importance", fontsize=11)
        plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_kernelshap.png"), bbox_inches="tight"); plt.close()
        print("KernelSHAP done")
    except Exception as e:
        print("KernelSHAP step failed:", e)

    # ---------- LIME (local surrogate, aggregated to a global vector) ----------
    # LIME is inherently local; the global importance vector is the mean
    # |per-feature weight| across a random sample of explained test instances.
    try:
        from lime.lime_tabular import LimeTabularExplainer
        lexpl = LimeTabularExplainer(Xtr.values, feature_names=feats, class_names=["down", "up"],
                                     mode="classification", discretize_continuous=True,
                                     random_state=SEED)
        rng = np.random.RandomState(SEED)
        rows_i = rng.choice(len(Xte), min(30, len(Xte)), replace=False)
        wsum = np.zeros(len(feats))
        for i in rows_i:
            exp = lexpl.explain_instance(Xte.values[i], model.predict_proba,
                                         num_features=len(feats), labels=(1,))
            for fi, w in exp.as_map()[1]:
                wsum[fi] += abs(w)
        lime_imp = pd.Series(wsum / len(rows_i), index=feats).sort_values(ascending=False)
        summary["lime_top10"] = lime_imp.head(10).round(5).to_dict()
        summary["lime_sentiment_share"] = float(lime_imp[SENT_FEATS].sum() / lime_imp.sum())
        faithfulness.append(_faithfulness(lime_imp, "lime"))
        plt.figure(figsize=(7.5, 6))
        order = lime_imp.sort_values()
        colors = ["#B4740F" if f in SENT_FEATS else "#2F4B94" for f in order.index]
        plt.barh(order.index, order.values, color=colors)
        plt.xlabel("mean |LIME weight| over sampled instances")
        plt.title(f"LIME global importance (aggregated over {len(rows_i)} instances)", fontsize=11)
        plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_lime.png"), bbox_inches="tight"); plt.close()
        print("LIME done")
    except Exception as e:
        print("LIME step failed:", e)

    # ---------- Permutation importance ----------
    pi = permutation_importance(model, Xte.values, yte, n_repeats=10,
                                random_state=SEED, scoring="accuracy")
    pis = pd.Series(pi.importances_mean, index=feats).sort_values()
    pos = pis.clip(lower=0)   # sentiment feats often <=0 (permuting them doesn't hurt)
    summary["perm_sentiment_share"] = float(pos[SENT_FEATS].sum() / max(pos.sum(), 1e-9))
    summary["perm_sentiment_max"] = float(pis[SENT_FEATS].max())
    faithfulness.append(_faithfulness(pis, "permutation"))
    plt.figure(figsize=(7.5, 6))
    colors = ["#B4740F" if f in SENT_FEATS else "#2F4B94" for f in pis.index]
    plt.barh(pis.index, pis.values, color=colors)
    plt.axvline(0, color="black", lw=0.8)
    plt.xlabel("Decrease in accuracy when permuted")
    plt.title("Permutation importance (test 2023)", fontsize=11)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_perm_importance.png"), bbox_inches="tight"); plt.close()

    # ---------- GROUPED block-permutation (correlation-robust) ----------
    # Permute an entire feature block jointly: within-block correlation is
    # preserved, so redundancy among correlated features cannot hide the block's
    # true joint contribution. This is the key test against the multicollinearity
    # critique -- it asks "does the sentiment GROUP carry signal, robustly?"
    from sklearn.metrics import accuracy_score
    def block_perm(block, n=25):
        base = accuracy_score(yte, model.predict(Xte.values))
        drops = []
        rr = np.random.RandomState(SEED)
        cols = [feats.index(c) for c in block]
        for _ in range(n):
            Xp = Xte.values.copy()
            perm = rr.permutation(len(Xp))
            Xp[:, cols] = Xp[perm][:, cols]     # permute whole block with one perm
            drops.append(base - accuracy_score(yte, model.predict(Xp)))
        return float(np.mean(drops)), float(np.std(drops))
    sent_drop = block_perm(SENT_FEATS)
    price_drop = block_perm(PRICE_FEATS)
    summary["block_perm_sentiment_acc_drop"] = sent_drop[0]
    summary["block_perm_price_acc_drop"] = price_drop[0]
    print(f"block-perm sentiment Δacc={sent_drop[0]:.4f}±{sent_drop[1]:.4f} | "
          f"price Δacc={price_drop[0]:.4f}±{price_drop[1]:.4f}")
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.barh(["news sentiment\nblock", "price/technical\nblock"],
            [sent_drop[0], price_drop[0]], xerr=[sent_drop[1], price_drop[1]],
            color=["#B4740F", "#2F4B94"], capsize=4)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Accuracy drop when whole block is permuted (test 2023)")
    ax.set_title("Grouped block-permutation importance\n(correlation-robust)", fontsize=10, fontweight="bold")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_block_perm.png"), bbox_inches="tight"); plt.close()

    # ---------- ALE (modern PDP successor) ----------
    try:
        from PyALE import ale
        class ProbaWrap:
            def __init__(self, m): self.m = m
            def predict(self, X): return self.m.predict_proba(np.asarray(X))[:, 1]
        wrap = ProbaWrap(model)
        ale_targets = [("fb_score", "fig_ale_sentiment.png"),
                       ("vol_z20", "fig_ale_price.png"),
                       ("fb_roll5", "fig_ale_sentiment2.png")]
        if "gt_signal" in feats:
            ale_targets.append(("gt_signal", "fig_ale_gt_signal.png"))
        if "gt_noise" in feats:
            ale_targets.append(("gt_noise", "fig_ale_gt_noise.png"))
        for feat, fname in ale_targets:
            res = ale(X=Xte[feats], model=wrap, feature=[feat], grid_size=40,
                      include_CI=True, C=0.95)
            fig = plt.gcf()
            fig.suptitle(f"ALE — {feat}", fontsize=11)
            fig.savefig(os.path.join(FIG, fname), bbox_inches="tight"); plt.close("all")
        print("saved ALE figures")
    except Exception as e:
        print("ALE step failed:", e)

    # ---------- Integrated Gradients on the GRU ----------
    try:
        import torch, torch.nn as nn
        from captum.attr import IntegratedGradients
        bundle = np.load(os.path.join(ART, "gru_ig_bundle.npz"), allow_pickle=True)
        Xt = torch.tensor(bundle["X_test"]).float()
        fn = list(bundle["feat_names"])
        class GRUNet(nn.Module):
            def __init__(self, nf, hidden=64):
                super().__init__()
                self.gru = nn.GRU(nf, hidden, num_layers=2, batch_first=True, dropout=0.2)
                self.head = nn.Linear(hidden, 1)
            def forward(self, x):
                o, _ = self.gru(x); return self.head(o[:, -1]).squeeze(-1)
        gru = GRUNet(Xt.shape[-1]); gru.load_state_dict(torch.load(os.path.join(ART, "gru_dir_2023.pt"), map_location="cpu"))
        gru.eval()
        ig = IntegratedGradients(gru)
        attr = ig.attribute(Xt, baselines=torch.zeros_like(Xt), n_steps=32)
        a = attr.abs().mean(dim=(0, 1)).detach().numpy()   # per-feature
        ig_imp = pd.Series(a, index=fn).sort_values()
        ig_sent_share = float(ig_imp[[f for f in SENT_FEATS if f in fn]].sum() / ig_imp.sum())
        summary["ig_sentiment_share"] = ig_sent_share
        # note: the GRU (s4_lstm.py) is trained on PRICE_FEATS/SENT_FEATS only,
        # not GT_FEATS, so this faithfulness entry has no signal/noise ranks yet
        # -- a known gap, closed once the sequence models train on gt_* too.
        faithfulness.append(_faithfulness(ig_imp, "integrated_gradients"))
        plt.figure(figsize=(7.5, 6))
        colors = ["#B4740F" if f in SENT_FEATS else "#2F4B94" for f in ig_imp.index]
        plt.barh(ig_imp.index, ig_imp.values, color=colors)
        plt.xlabel("mean |Integrated Gradients attribution|")
        plt.title(f"GRU attribution (orange = sentiment): {ig_sent_share*100:.1f}% of total", fontsize=11)
        plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_ig_gru.png"), bbox_inches="tight"); plt.close()
        print(f"IG sentiment share: {ig_sent_share:.4f}")
    except Exception as e:
        print("IG step failed:", e)

    # ---------- Ground-truth faithfulness summary (the headline spine) ----------
    summary["ground_truth_faithfulness"] = faithfulness
    n_signal_over_noise = sum(1 for f in faithfulness if f.get("ranks_signal_above_noise"))
    n_scored = sum(1 for f in faithfulness if "ranks_signal_above_noise" in f)
    print(f"ground-truth faithfulness: {n_signal_over_noise}/{n_scored} methods "
          f"ranked gt_signal above gt_noise")

    with open(os.path.join(MET, "xai_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    data_vol.commit()


# ==========================================================================
# ---- XAI-method x model-class matrix -------------------------------------
# The functions below extend run() (left untouched above) to run each XAI
# method against every model class it applies to, per the applicability
# matrix: TreeSHAP -> tree ensembles/glass-box trees; KernelSHAP/LIME/
# permutation/block-permutation/ALE -> a model-agnostic slate; Integrated
# Gradients -> the sequence NN zoo. Each producer function writes its own
# partial JSON to MET; consolidate() merges them into the final
# xai_attribution_by_model.json. Run producers first, consolidate() last:
#
#   modal run s5_xai.py::treeshap_by_model
#   modal run s5_xai.py::agnostic_by_model
#   modal run s5_xai.py::ig_by_model
#   modal run s5_xai.py::consolidate
# ==========================================================================

def _load_split():
    """Shared train/test split for the model-x-method matrix: features =
    PRICE_FEATS+SENT_FEATS+GT_FEATS, target = dir_next (classification),
    train = years < 2023, test = year 2023 -- same protocol as run()."""
    import os, pandas as pd
    from sklearn.preprocessing import StandardScaler
    feats = PRICE_FEATS + SENT_FEATS + GT_FEATS
    df = pd.read_parquet(os.path.join(BUILD, "panel.parquet")).copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Date", "Symbol"]).reset_index(drop=True)
    tr = df[df["Date"].dt.year < 2023]
    te = df[df["Date"].dt.year == 2023]
    sc = StandardScaler().fit(tr[feats].values)
    Xtr = pd.DataFrame(sc.transform(tr[feats].values), columns=feats)
    Xte = pd.DataFrame(sc.transform(te[feats].values), columns=feats)
    ytr, yte = tr["dir_next"].values, te["dir_next"].values
    return feats, Xtr, Xte, ytr, yte


@app.function(volumes={DATA: data_vol}, cpu=16.0, memory=65536, timeout=55 * 60)
def treeshap_by_model():
    """TreeSHAP against every tree-ensemble / glass-box tree model in the
    zoo (RandomForest, XGBoost, LightGBM, CatBoost, DecisionTree), retrained
    here with s3_experiments.py::_make_models' exact classification
    hyperparameters, on dir_next using PRICE_FEATS+SENT_FEATS+GT_FEATS.
    Ground-truth faithfulness is scored with the shared _faithfulness()
    helper (does gt_signal outrank gt_noise; how is credit split between
    gt_redundant_copy and its source feature vol_z20)."""
    import os, json, numpy as np, pandas as pd
    import shap
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    import xgboost as xgb, lightgbm as lgb
    from catboost import CatBoostClassifier
    os.makedirs(MET, exist_ok=True)

    feats, Xtr, Xte, ytr, yte = _load_split()

    models = {
        "DecisionTree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=50,
                         random_state=SEED, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=8,
                         n_jobs=-1, random_state=SEED, class_weight="balanced"),
        "XGBoost": xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.03,
                    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                    n_jobs=-1, random_state=SEED, eval_metric="logloss"),
        "LightGBM": lgb.LGBMClassifier(n_estimators=400, num_leaves=31, max_depth=5,
                     learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                     reg_lambda=1.0, n_jobs=-1, random_state=SEED, verbose=-1),
        "CatBoost": CatBoostClassifier(iterations=400, depth=5, learning_rate=0.03,
                     random_state=SEED, verbose=False, allow_writing_files=False),
    }

    results = []
    vectors = []   # M4: full per-feature importance vectors for the RQ2 agreement matrix
    for name, model in models.items():
        print(f"[treeshap] fitting {name} ...")
        model.fit(Xtr.values, ytr)
        expl = shap.TreeExplainer(model)
        Xs = Xte.sample(min(2000, len(Xte)), random_state=SEED)
        sv = expl.shap_values(Xs)
        if isinstance(sv, list):
            sv = sv[1]
        if hasattr(sv, "ndim") and sv.ndim == 3:   # (n, features, classes) in some shap versions
            sv = sv[:, :, 1]
        mean_abs = np.abs(sv).mean(0)
        imp = pd.Series(mean_abs, index=feats).sort_values(ascending=False)
        row = {"model": name, "task": "direction",
               "sentiment_attribution_share": float(imp[SENT_FEATS].sum() / imp.sum())}
        row.update(_faithfulness(imp, "treeshap"))
        results.append(row)
        vectors.append({"method": "treeshap", "model": name,
                         "importance": {f: float(imp[f]) for f in feats}})
        print(f"[treeshap] {name}: signal_rank={row.get('signal_rank')} "
              f"noise_rank={row.get('noise_rank')} "
              f"redundant_copy_credit_share={row.get('redundant_copy_credit_share')}")

    with open(os.path.join(MET, "xai_by_model_treeshap.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(MET, "xai_vectors_treeshap.json"), "w") as f:
        json.dump(vectors, f, indent=2)
    print(json.dumps(results, indent=2))
    data_vol.commit()


@app.function(volumes={DATA: data_vol}, cpu=16.0, memory=32768, timeout=55 * 60)
def agnostic_by_model():
    """Model-agnostic XAI methods -- KernelSHAP, LIME, permutation
    importance, block-permutation, ALE -- against a representative
    model-agnostic slate: Logistic, RandomForest, XGBoost, EBM, SVM.

    KNN is EXCLUDED from every method here, not just KernelSHAP/LIME:
    KernelSHAP and LIME each require thousands of predict_proba evaluations,
    and KNN's O(n_train) prediction cost makes that combination too slow at
    this data scale (>100k training rows after the embargoed expanding
    window). To keep an identical 5-model set across all five methods for a
    clean comparison, KNN is skipped for permutation/block-permutation/ALE
    too rather than silently swapping in a sixth model only for the cheap
    methods."""
    import os, json, numpy as np, pandas as pd
    import shap
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score
    from sklearn.inspection import permutation_importance
    from interpret.glassbox import ExplainableBoostingClassifier
    import xgboost as xgb
    from lime.lime_tabular import LimeTabularExplainer
    os.makedirs(MET, exist_ok=True)

    feats, Xtr, Xte, ytr, yte = _load_split()
    rng = np.random.RandomState(SEED)

    # SVC needs probability=True for KernelSHAP/LIME to call predict_proba
    # (s3_experiments.py's SVC uses probability=False since it only needs
    # .predict there) -- a deliberate deviation from _make_models, stated
    # here rather than silently changed. Also subsample its training set
    # like s3_experiments.py::main does for SVC/KNN on large folds, since
    # SVC training is O(n^2)-O(n^3).
    if len(Xtr) > 9000:
        svc_idx = rng.choice(len(Xtr), 9000, replace=False)
    else:
        svc_idx = np.arange(len(Xtr))

    models = {
        "Logistic": LogisticRegression(C=1.0, max_iter=2000),
        "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=8,
                         n_jobs=-1, random_state=SEED, class_weight="balanced"),
        "XGBoost": xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.03,
                    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                    n_jobs=-1, random_state=SEED, eval_metric="logloss"),
        "EBM": ExplainableBoostingClassifier(random_state=SEED, n_jobs=-1),
        "SVM": SVC(C=1.0, gamma="scale", probability=True, random_state=SEED),
    }

    results = [
        {"method": "kernelshap", "model": "KNN", "task": "direction", "status": "N/A",
         "reason": "KNN predict_proba is O(n_train) per call; KernelSHAP needs "
                   "thousands of such calls, prohibitively slow at this data scale."},
        {"method": "lime", "model": "KNN", "task": "direction", "status": "N/A",
         "reason": "same as kernelshap -- LIME perturbs ~5000 samples per explained "
                   "instance, each needing a KNN predict_proba call."},
        {"method": "permutation", "model": "KNN", "task": "direction", "status": "N/A",
         "reason": "excluded to keep an identical 5-model set across all agnostic "
                   "methods in this run (see function docstring)."},
        {"method": "block_permutation", "model": "KNN", "task": "direction", "status": "N/A",
         "reason": "same as permutation."},
        {"method": "ale", "model": "KNN", "task": "direction", "status": "N/A",
         "reason": "same as permutation."},
    ]

    # small shared eval subsample for permutation/block-permutation/ALE --
    # keeps SVM prediction cost bounded (modest scope per instructions; SVM
    # predict scales with #support-vectors x n_eval).
    Xte_eval = Xte.sample(min(3000, len(Xte)), random_state=SEED)
    idx_eval = Xte_eval.index.values
    yte_eval = yte[idx_eval]
    gt_relevant = [f for f in ["gt_signal", "gt_noise", "gt_redundant_copy", "vol_z20"] if f in feats]

    vectors = []   # M4: full per-feature importance vectors for the RQ2 agreement matrix
    for name, model in models.items():
        print(f"[agnostic] fitting {name} ...")
        if name == "SVM":
            model.fit(Xtr.values[svc_idx], ytr[svc_idx])
        else:
            model.fit(Xtr.values, ytr)

        # ---- KernelSHAP ----
        try:
            bg = shap.sample(Xtr, 30, random_state=SEED)
            kexpl = shap.KernelExplainer(lambda X: model.predict_proba(X)[:, 1], bg)
            Xk = Xte.sample(min(60, len(Xte)), random_state=SEED)
            ksv = kexpl.shap_values(Xk, nsamples=100, silent=True)
            kimp = pd.Series(np.abs(ksv).mean(0), index=feats)
            row = {"model": name, "task": "direction",
                   "sentiment_attribution_share": float(kimp[SENT_FEATS].sum() / kimp.sum())}
            row.update(_faithfulness(kimp, "kernelshap"))
            results.append(row)
            vectors.append({"method": "kernelshap", "model": name,
                             "importance": {f: float(kimp[f]) for f in feats}})
            print(f"[kernelshap] {name} done")
        except Exception as e:
            results.append({"method": "kernelshap", "model": name, "task": "direction",
                             "status": "ERROR", "reason": str(e)})
            print(f"[kernelshap] {name} failed: {e}")

        # ---- LIME ----
        try:
            lexpl = LimeTabularExplainer(Xtr.values, feature_names=feats,
                                         class_names=["down", "up"], mode="classification",
                                         discretize_continuous=True, random_state=SEED)
            rows_i = rng.choice(len(Xte), min(15, len(Xte)), replace=False)
            wsum = np.zeros(len(feats))
            for i in rows_i:
                exp = lexpl.explain_instance(Xte.values[i], model.predict_proba,
                                             num_features=len(feats), labels=(1,))
                for fi, w in exp.as_map()[1]:
                    wsum[fi] += abs(w)
            lime_imp = pd.Series(wsum / len(rows_i), index=feats)
            row = {"model": name, "task": "direction",
                   "sentiment_attribution_share": float(lime_imp[SENT_FEATS].sum() / lime_imp.sum())}
            row.update(_faithfulness(lime_imp, "lime"))
            results.append(row)
            vectors.append({"method": "lime", "model": name,
                             "importance": {f: float(lime_imp[f]) for f in feats}})
            print(f"[lime] {name} done")
        except Exception as e:
            results.append({"method": "lime", "model": name, "task": "direction",
                             "status": "ERROR", "reason": str(e)})
            print(f"[lime] {name} failed: {e}")

        # ---- permutation importance ----
        try:
            pi = permutation_importance(model, Xte_eval.values, yte_eval, n_repeats=5,
                                        random_state=SEED, scoring="accuracy")
            pis = pd.Series(pi.importances_mean, index=feats)
            row = {"model": name, "task": "direction"}
            row.update(_faithfulness(pis, "permutation"))
            results.append(row)
            vectors.append({"method": "permutation", "model": name,
                             "importance": {f: float(pis[f]) for f in feats}})
            print(f"[permutation] {name} done")
        except Exception as e:
            results.append({"method": "permutation", "model": name, "task": "direction",
                             "status": "ERROR", "reason": str(e)})
            print(f"[permutation] {name} failed: {e}")

        # ---- block-permutation ----
        # Block-permutation is fundamentally a BLOCK-level method (as used in
        # run(), which reports one accuracy drop for the whole sentiment
        # block and one for the whole price block). To still get a
        # per-feature faithfulness score, the 4 ground-truth-relevant
        # features are each treated as their own singleton block here
        # (equivalent to permuting that one feature alone) instead of
        # computing a full 22-feature vector, which would just duplicate the
        # permutation-importance step above at ~4x the cost for no extra
        # faithfulness signal.
        try:
            base_acc = accuracy_score(yte_eval, model.predict(Xte_eval.values))
            drops = {}
            for f in gt_relevant:
                col = feats.index(f)
                rr = np.random.RandomState(SEED)
                d = []
                for _ in range(15):
                    Xp = Xte_eval.values.copy()
                    perm = rr.permutation(len(Xp))
                    Xp[:, col] = Xp[perm, col]
                    d.append(base_acc - accuracy_score(yte_eval, model.predict(Xp)))
                drops[f] = float(np.mean(d))
            bimp = pd.Series(drops)
            row = {"model": name, "task": "direction",
                   "note": "singleton-block permutation restricted to the 4 "
                           "ground-truth-relevant features (see comment above)."}
            row.update(_faithfulness(bimp, "block_permutation"))
            results.append(row)
            print(f"[block_permutation] {name} done")
        except Exception as e:
            results.append({"method": "block_permutation", "model": name, "task": "direction",
                             "status": "ERROR", "reason": str(e)})
            print(f"[block_permutation] {name} failed: {e}")

        # ---- ALE ----
        # Restricted to the 4 ground-truth-relevant features: a full
        # 22-feature x 5-model ALE grid would be by far the most expensive
        # method in this matrix for little extra faithfulness signal beyond
        # what's needed to score gt_signal/gt_noise/gt_redundant_copy --
        # modest scope per instructions.
        try:
            from PyALE import ale
            class ProbaWrap:
                def __init__(self, m): self.m = m
                def predict(self, X): return self.m.predict_proba(np.asarray(X))[:, 1]
            wrap = ProbaWrap(model)
            aimp = {}
            for f in gt_relevant:
                res = ale(X=Xte_eval[feats], model=wrap, feature=[f], grid_size=20,
                          include_CI=False)
                col = "eff" if "eff" in res.columns else res.columns[0]
                aimp[f] = float(res[col].max() - res[col].min())
            aimp = pd.Series(aimp)
            row = {"model": name, "task": "direction",
                   "note": "ALE range (max-min effect) restricted to the 4 "
                           "ground-truth-relevant features (see comment above)."}
            row.update(_faithfulness(aimp, "ale"))
            results.append(row)
            print(f"[ale] {name} done")
        except Exception as e:
            results.append({"method": "ale", "model": name, "task": "direction",
                             "status": "ERROR", "reason": str(e)})
            print(f"[ale] {name} failed: {e}")

    with open(os.path.join(MET, "xai_by_model_agnostic.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(MET, "xai_vectors_agnostic.json"), "w") as f:
        json.dump(vectors, f, indent=2)
    print(json.dumps(results, indent=2))
    data_vol.commit()


@app.function(volumes={DATA: data_vol}, gpu="a10g", cpu=4.0, memory=16384, timeout=30 * 60)
def ig_by_model():
    """Integrated Gradients (Captum) against every sequence NN model in the
    zoo: GRU, LSTM, CNN1d, TCN, Transformer. Reloads each model's saved
    weights + held-out IG bundle from s4c_gt_artifacts.py (ART on the
    Volume) -- gap-fix round 2: these are now GROUND-TRUTH-AUGMENTED
    artifacts (PRICE_FEATS+SENT_FEATS+GT_FEATS, trained separately from the
    honest model-zoo results in s4_lstm.py so the predictive-performance
    table is never polluted by the synthetic leak), so IG now gets a real
    faithfulness score for every sequence model, not just an N/A. The net
    class definitions below are copied verbatim from s4_lstm.py."""
    import os, json, math, numpy as np, pandas as pd, torch, torch.nn as nn
    from captum.attr import IntegratedGradients
    os.makedirs(MET, exist_ok=True)

    class GRUNet(nn.Module):
        def __init__(self, nf, hidden=64):
            super().__init__()
            self.gru = nn.GRU(nf, hidden, num_layers=2, batch_first=True, dropout=0.2)
            self.head = nn.Linear(hidden, 1)
        def forward(self, x):
            o, _ = self.gru(x); return self.head(o[:, -1]).squeeze(-1)

    class LSTMNet(nn.Module):
        def __init__(self, nf, hidden=64):
            super().__init__()
            self.lstm = nn.LSTM(nf, hidden, num_layers=2, batch_first=True, dropout=0.2)
            self.head = nn.Linear(hidden, 1)
        def forward(self, x):
            o, _ = self.lstm(x); return self.head(o[:, -1]).squeeze(-1)

    class CNN1dNet(nn.Module):
        def __init__(self, nf, channels=32):
            super().__init__()
            self.conv1 = nn.Conv1d(nf, channels, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
            self.act = nn.ReLU()
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.drop = nn.Dropout(0.2)
            self.head = nn.Linear(channels, 1)
        def forward(self, x):
            x = x.transpose(1, 2)
            x = self.act(self.conv1(x))
            x = self.act(self.conv2(x))
            x = self.drop(self.pool(x).squeeze(-1))
            return self.head(x).squeeze(-1)

    class TCNBlock(nn.Module):
        def __init__(self, in_ch, out_ch, kernel_size, dilation):
            super().__init__()
            self.pad = (kernel_size - 1) * dilation
            self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=self.pad, dilation=dilation)
            self.norm = nn.BatchNorm1d(out_ch)
            self.act = nn.ReLU()
        def forward(self, x):
            out = self.conv(x)
            out = out[:, :, :-self.pad] if self.pad > 0 else out
            return self.act(self.norm(out))

    class TCNNet(nn.Module):
        def __init__(self, nf, channels=32, levels=3, kernel_size=3):
            super().__init__()
            layers, in_ch = [], nf
            for i in range(levels):
                layers.append(TCNBlock(in_ch, channels, kernel_size, dilation=2 ** i))
                in_ch = channels
            self.net = nn.Sequential(*layers)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.drop = nn.Dropout(0.2)
            self.head = nn.Linear(channels, 1)
        def forward(self, x):
            x = x.transpose(1, 2)
            x = self.net(x)
            x = self.drop(self.pool(x).squeeze(-1))
            return self.head(x).squeeze(-1)

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=64):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len).unsqueeze(1).float()
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe.unsqueeze(0))
        def forward(self, x):
            return x + self.pe[:, :x.size(1)]

    class TransformerNet(nn.Module):
        def __init__(self, nf, d_model=64, nhead=4, num_layers=2):
            super().__init__()
            self.proj = nn.Linear(nf, d_model)
            self.pos = PositionalEncoding(d_model)
            layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128,
                                               dropout=0.2, batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
            self.head = nn.Linear(d_model, 1)
        def forward(self, x):
            x = self.proj(x); x = self.pos(x); x = self.encoder(x)
            x = x[:, -1]
            return self.head(x).squeeze(-1)

    specs = [
        ("GRU", GRUNet, "gru"),
        ("LSTM", LSTMNet, "lstm"),
        ("CNN1d", CNN1dNet, "cnn1d"),
        ("TCN", TCNNet, "tcn"),
        ("Transformer", TransformerNet, "transformer"),
    ]

    results = []
    for label, cls, prefix in specs:
        bpath = os.path.join(ART, f"{prefix}_gt_ig_bundle.npz")
        wpath = os.path.join(ART, f"{prefix}_gt_dir_2023.pt")
        if not (os.path.exists(bpath) and os.path.exists(wpath)):
            results.append({"method": "integrated_gradients", "model": label, "task": "direction",
                             "status": "N/A", "reason": f"artifact bundle missing: {bpath} / {wpath} "
                             "(run s4c_gt_artifacts.py::run first)"})
            print(f"[ig] {label}: artifact missing, skipping")
            continue
        try:
            bundle = np.load(bpath, allow_pickle=True)
            Xt = torch.tensor(bundle["X_test"]).float()
            fn = list(bundle["feat_names"])
            net = cls(Xt.shape[-1])
            net.load_state_dict(torch.load(wpath, map_location="cpu"))
            net.eval()
            ig = IntegratedGradients(net)
            attr = ig.attribute(Xt, baselines=torch.zeros_like(Xt), n_steps=32)
            a = attr.abs().mean(dim=(0, 1)).detach().numpy()
            imp = pd.Series(a, index=fn)
            sent_share = float(imp[[f for f in SENT_FEATS if f in fn]].sum() / imp.sum())
            row = {"model": label, "task": "direction",
                   "sentiment_attribution_share": sent_share,
                   "note": "ground-truth-augmented artifact (s4c_gt_artifacts.py); "
                           "trained separately from the honest model-zoo results."}
            row.update(_faithfulness(imp, "integrated_gradients"))
            results.append(row)
            print(f"[ig] {label}: sentiment_share={sent_share:.4f}, "
                  f"ranks_signal_above_noise={row.get('ranks_signal_above_noise')}")
        except Exception as e:
            results.append({"method": "integrated_gradients", "model": label, "task": "direction",
                             "status": "ERROR", "reason": str(e)})
            print(f"[ig] {label} failed: {e}")

    with open(os.path.join(MET, "xai_by_model_ig.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    data_vol.commit()


@app.function(volumes={DATA: data_vol}, cpu=2.0, memory=4096, timeout=10 * 60)
def consolidate():
    """Merge the per-method-family partial outputs (treeshap_by_model,
    agnostic_by_model, ig_by_model) into the single consolidated
    xai_attribution_by_model.json this stage's job requires, plus a summary
    figure of ground-truth faithfulness across the whole method x model
    matrix. Run this AFTER the three producer functions have each committed
    their partial file to the Volume."""
    import os, json, numpy as np, pandas as pd
    plt = _mpl()
    os.makedirs(MET, exist_ok=True); os.makedirs(FIG, exist_ok=True)

    parts = ["xai_by_model_treeshap.json", "xai_by_model_agnostic.json", "xai_by_model_ig.json"]
    combined = []
    missing = []
    for p in parts:
        fp = os.path.join(MET, p)
        if os.path.exists(fp):
            with open(fp) as f:
                combined.extend(json.load(f))
        else:
            missing.append(p)
    if missing:
        print(f"[consolidate] warning: missing partial files (run their producer "
              f"functions first): {missing}")

    with open(os.path.join(MET, "xai_attribution_by_model.json"), "w") as f:
        json.dump(combined, f, indent=2)
    print(f"[consolidate] wrote {len(combined)} entries to xai_attribution_by_model.json")

    scored = [r for r in combined if "ranks_signal_above_noise" in r]
    n_ok = sum(1 for r in scored if r["ranks_signal_above_noise"])
    print(f"[consolidate] {n_ok}/{len(scored)} scoreable (method, model) cells "
          f"ranked gt_signal above gt_noise")

    if scored:
        df = pd.DataFrame(scored)
        df["label"] = df["method"] + " / " + df["model"]
        df = df.sort_values("signal_rank")
        fig, ax = plt.subplots(figsize=(8, max(3, 0.32 * len(df))))
        colors = ["#2C7A55" if v else "#C0362C" for v in df["ranks_signal_above_noise"]]
        ax.barh(df["label"], df["signal_rank"], color=colors)
        ax.axvline(df["noise_rank"].median(), color="black", ls="--", lw=1,
                   label="median gt_noise rank")
        ax.invert_yaxis()
        ax.set_xlabel("rank of gt_signal (1 = most important; lower is better)")
        ax.set_title("Ground-truth faithfulness across the XAI method x model matrix\n"
                     "(green = correctly ranked gt_signal above gt_noise)", fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG, "fig_xai_by_model_faithfulness.png"), bbox_inches="tight")
        plt.close()
        print("saved fig_xai_by_model_faithfulness.png")

    data_vol.commit()


@app.function(volumes={DATA: data_vol}, cpu=2.0, memory=4096, timeout=10 * 60)
def agreement_matrix():
    """M4 (R3 revision) -- delivers RQ2 properly: a full pairwise rank-
    correlation (Spearman) agreement matrix across the XAI methods that each
    produce a complete per-feature importance ranking over the SAME feature
    set (PRICE_FEATS+SENT_FEATS+GT_FEATS). Scope, stated honestly rather than
    silently: this covers TreeSHAP, KernelSHAP, LIME, and permutation
    importance -- the four methods with a full-vector ranking on shared model
    classes (xai_vectors_treeshap.json / xai_vectors_agnostic.json, written
    by treeshap_by_model / agnostic_by_model). block_permutation and ALE only
    score the 4 ground-truth-relevant features (not a full ranking, see their
    comments in agnostic_by_model) and integrated_gradients/LRP/Grad-CAM/
    GNNExplainer run on disjoint (neural-only or graph-only) model classes
    with no model in common with the classical methods here, so none of
    those five can enter a same-model rank-correlation comparison; they are
    excluded from this matrix by construction, not by oversight, and that
    scope is reported in the output rather than hidden.

    Run this AFTER treeshap_by_model and agnostic_by_model have committed.
      modal run s5_xai.py::agreement_matrix
    """
    import os, json, itertools
    import numpy as np, pandas as pd
    from scipy.stats import spearmanr
    plt = _mpl()
    import seaborn as sns
    os.makedirs(MET, exist_ok=True); os.makedirs(FIG, exist_ok=True)

    vecs = []
    excluded_methods = []
    for fname, methods_here in [("xai_vectors_treeshap.json", {"treeshap"}),
                                 ("xai_vectors_agnostic.json", {"kernelshap", "lime", "permutation"})]:
        fp = os.path.join(MET, fname)
        if os.path.exists(fp):
            with open(fp) as f:
                vecs.extend(json.load(f))
        else:
            excluded_methods.extend(sorted(methods_here))
            print(f"[agreement] warning: {fname} missing -- run its producer function first")

    methods = sorted(set(v["method"] for v in vecs))
    if len(methods) < 2:
        print("[agreement] fewer than 2 methods with vectors available -- nothing to compare")
        summary = {"methods": methods, "excluded_no_shared_model_class":
                   ["block_permutation", "ale", "integrated_gradients", "lrp", "guided_backprop",
                    "deconvnet", "gradcam", "gradcam_pp", "score_cam", "gnnexplainer"],
                   "note": "insufficient vector files present; rerun treeshap_by_model / agnostic_by_model"}
        with open(os.path.join(MET, "xai_agreement_matrix.json"), "w") as f:
            json.dump(summary, f, indent=2)
        data_vol.commit()
        return

    # group importance vectors by model, so pairs are compared on the SAME model
    by_model = {}
    for v in vecs:
        by_model.setdefault(v["model"], {})[v["method"]] = v["importance"]

    pair_rhos = {m1: {m2: [] for m2 in methods} for m1 in methods}
    pair_models = {m1: {m2: [] for m2 in methods} for m1 in methods}
    for model_name, per_method in by_model.items():
        present = [m for m in methods if m in per_method]
        for m1, m2 in itertools.combinations(present, 2):
            v1, v2 = per_method[m1], per_method[m2]
            common_feats = sorted(set(v1) & set(v2))
            if len(common_feats) < 5:
                continue
            x = [v1[f] for f in common_feats]
            y = [v2[f] for f in common_feats]
            rho = float(spearmanr(x, y).statistic) if np.std(x) > 0 and np.std(y) > 0 else float("nan")
            if rho == rho:
                pair_rhos[m1][m2].append(rho); pair_rhos[m2][m1].append(rho)
                pair_models[m1][m2].append(model_name); pair_models[m2][m1].append(model_name)

    matrix = pd.DataFrame(index=methods, columns=methods, dtype=float)
    n_models_used = pd.DataFrame(index=methods, columns=methods, dtype=float)
    for m1 in methods:
        for m2 in methods:
            if m1 == m2:
                matrix.loc[m1, m2] = 1.0
                n_models_used.loc[m1, m2] = np.nan
            else:
                vals = pair_rhos[m1][m2]
                matrix.loc[m1, m2] = float(np.mean(vals)) if vals else float("nan")
                n_models_used.loc[m1, m2] = len(vals)

    summary = {
        "methods_in_matrix": methods,
        "excluded_no_shared_model_class": sorted(set(
            ["block_permutation", "ale", "integrated_gradients", "lrp", "guided_backprop",
             "deconvnet", "gradcam", "gradcam_pp", "score_cam", "gnnexplainer"]) - set(excluded_methods)) +
            excluded_methods,
        "mean_spearman_rho": {m1: {m2: (None if pd.isna(matrix.loc[m1, m2]) else float(matrix.loc[m1, m2]))
                                    for m2 in methods} for m1 in methods},
        "n_common_models_per_pair": {m1: {m2: int(n_models_used.loc[m1, m2]) if not pd.isna(n_models_used.loc[m1, m2]) else 0
                                           for m2 in methods} for m1 in methods},
        "note": ("Each cell is the mean Spearman rank correlation between two methods' full "
                 "per-feature importance vectors, averaged over every model where BOTH methods "
                 "were run (the pairing/clustering unit here is the MODEL, not the stock -- a "
                 "different axis of agreement than the stock-clustered faithfulness benchmark). "
                 "Methods restricted to a 4-ground-truth-feature subset (block_permutation, ALE) "
                 "or run on disjoint model classes (integrated_gradients and every neural/graph-"
                 "only method) cannot enter a same-model rank correlation and are excluded from "
                 "this matrix by construction; see excluded_no_shared_model_class."),
    }
    with open(os.path.join(MET, "xai_agreement_matrix.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    sns.heatmap(matrix.astype(float), cmap="RdBu_r", vmin=-1, vmax=1, center=0,
                annot=True, fmt=".2f", square=True, cbar_kws={"label": "mean Spearman rho"}, ax=ax)
    ax.set_title("RQ2: pairwise XAI method agreement\n(rank correlation, averaged across shared models)",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_agreement_matrix.png"), bbox_inches="tight")
    print("saved fig_agreement_matrix.png")
    data_vol.commit()
