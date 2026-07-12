"""Stage 5c -- concept-based and intrinsic/global-aggregation XAI methods
(brief S8/S9/S14 families not yet covered by s5_xai.py):

  modal run s5c_concept_intrinsic.py::run

  * CAV/TCAV: linear Concept Activation Vectors on the trained GRU's last
    hidden state, for two concepts defined from the panel's own semantics
    ("high attention" = news_count > p75 vs < p50; "positive sentiment" =
    fb_score > 0 vs < 0). TCAV sensitivity = fraction of a test sample where
    the directional derivative of the GRU's scalar output w.r.t. the CAV
    direction is positive.
  * EBM intrinsic explanations: retrain an ExplainableBoostingClassifier on
    PRICE_FEATS+SENT_FEATS+GT_FEATS and read its native global importances
    via explain_global().
  * GAM (Global Attribution Mapping) aggregation: TreeSHAP local |SHAP|
    vectors on a sample of test instances, rank-normalized within each
    instance and averaged into one global ranking.
  * MUSE/BRCG global rule surrogate: a shallow decision tree fit to a
    black-box XGBoost's own PREDICTIONS (fidelity, not accuracy against
    labels), scored against the ground-truth spine too.

  All four methods are scored against the P10 synthetic ground-truth
  features (gt_signal should outrank gt_noise; gt_redundant_copy should
  split credit with its source feature vol_z20) using the same
  `_faithfulness` helper as s5_xai.py.
"""
import modal
from _common import image, data_vol, DATA, SEED, GROUND_TRUTH, LOOKBACK

app = modal.App("finsent-s5c-concept-intrinsic", image=image)

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
    ranked = imp.sort_values(ascending=False)
    rank_of = {f: int(ranked.index.get_loc(f)) + 1 for f in imp.index}
    if "gt_signal" in imp.index and "gt_noise" in imp.index:
        out["signal_rank"] = rank_of["gt_signal"]
        out["noise_rank"] = rank_of["gt_noise"]
        out["n_features"] = len(imp)
        out["ranks_signal_above_noise"] = bool(rank_of["gt_signal"] < rank_of["gt_noise"])
    if "gt_signal_nl" in imp.index and "gt_noise" in imp.index:
        out["signal_nl_rank"] = rank_of["gt_signal_nl"]
        out["ranks_nonlinear_signal_above_noise"] = bool(rank_of["gt_signal_nl"] < rank_of["gt_noise"])
    if "gt_redundant_copy" in imp.index and "vol_z20" in imp.index:
        a, b = float(imp["gt_redundant_copy"]), float(imp["vol_z20"])
        out["redundant_copy_importance"] = a
        out["source_feature_importance"] = b
        out["redundant_copy_credit_share"] = float(a / (a + b)) if (a + b) > 0 else float("nan")
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
    plt = _mpl()
    os.makedirs(FIG, exist_ok=True); os.makedirs(MET, exist_ok=True)

    out = {}

    df = pd.read_parquet(os.path.join(BUILD, "panel.parquet")).copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Symbol", "Date"]).reset_index(drop=True)

    # ======================================================================
    # 1) CAV / TCAV on the trained GRU's last hidden state
    # ======================================================================
    try:
        import torch, torch.nn as nn
        from sklearn.linear_model import LogisticRegression

        gru_feats = list(np.load(os.path.join(ART, "gru_ig_bundle.npz"),
                                  allow_pickle=True)["feat_names"])

        class GRUNet(nn.Module):
            def __init__(self, nf, hidden=64):
                super().__init__()
                self.gru = nn.GRU(nf, hidden, num_layers=2, batch_first=True, dropout=0.2)
                self.head = nn.Linear(hidden, 1)
            def forward(self, x):
                o, _ = self.gru(x)
                return self.head(o[:, -1]).squeeze(-1)

        class GRUHiddenWrap(nn.Module):
            """Runs self.gru(x) and returns the LAST hidden state o[:, -1]
            directly (before the head) -- used to build CAVs and to compute
            the directional derivative of the head output w.r.t. that hidden
            representation."""
            def __init__(self, gru_submodule):
                super().__init__()
                self.gru = gru_submodule
            def forward(self, x):
                o, _ = self.gru(x)
                return o[:, -1]

        full_gru = GRUNet(len(gru_feats))
        full_gru.load_state_dict(torch.load(os.path.join(ART, "gru_dir_2023.pt"), map_location="cpu"))
        full_gru.eval()
        hidden_wrap = GRUHiddenWrap(full_gru.gru)
        hidden_wrap.eval()
        head = full_gru.head

        # ---- build sequences exactly like s4_lstm.py::build_sequences ----
        def build_sequences(feats):
            Xs, meta_news, meta_fb, meta_dates, meta_syms = [], [], [], [], []
            for sym, g in df.groupby("Symbol"):
                g = g.sort_values("Date")
                F = g[feats].values.astype(np.float32)
                nc = g["news_count"].values.astype(np.float32)
                fb = g["fb_score"].values.astype(np.float32)
                dt = g["Date"].values
                for i in range(LOOKBACK - 1, len(g)):
                    Xs.append(F[i - LOOKBACK + 1:i + 1])
                    meta_news.append(nc[i]); meta_fb.append(fb[i])
                    meta_dates.append(dt[i]); meta_syms.append(sym)
            return (np.stack(Xs), np.array(meta_news), np.array(meta_fb),
                    np.array(meta_dates), np.array(meta_syms))

        X_all, news_all, fb_all, dates_all, syms_all = build_sequences(gru_feats)
        yy = pd.to_datetime(dates_all).year.values
        te_mask = yy == 2023
        tr_mask = yy < 2023

        # standardize with train-only stats, exactly like s4_lstm.py
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(X_all[tr_mask].reshape(-1, X_all.shape[-1]))
        Xn = ((X_all.reshape(-1, X_all.shape[-1]) - sc.mean_) / np.sqrt(sc.var_ + 1e-9)
              ).reshape(X_all.shape).astype(np.float32)

        Xte = Xn[te_mask]
        news_te = news_all[te_mask]
        fb_te = fb_all[te_mask]

        # ---- concept definitions on TEST rows (unstandardized raw values) ----
        news_p75 = float(np.percentile(news_te, 75))
        news_p50 = float(np.percentile(news_te, 50))
        concepts = {
            "high_attention": {
                "pos": np.where(news_te > news_p75)[0],
                "neg": np.where(news_te < news_p50)[0],
            },
            "positive_sentiment": {
                "pos": np.where(fb_te > 0)[0],
                "neg": np.where(fb_te < 0)[0],
            },
        }

        rng = np.random.RandomState(SEED)
        tcav_results = []
        for cname, sets in concepts.items():
            pos_idx, neg_idx = sets["pos"], sets["neg"]
            n_pos = min(300, len(pos_idx))
            n_neg = min(300, len(neg_idx))
            if n_pos < 20 or n_neg < 20:
                print(f"concept {cname}: too few examples (pos={len(pos_idx)}, neg={len(neg_idx)}), skipping")
                continue
            pos_sample = rng.choice(pos_idx, n_pos, replace=False)
            neg_sample = rng.choice(neg_idx, n_neg, replace=False)

            with torch.no_grad():
                h_pos = hidden_wrap(torch.tensor(Xte[pos_sample])).numpy()
                h_neg = hidden_wrap(torch.tensor(Xte[neg_sample])).numpy()

            Hc = np.concatenate([h_pos, h_neg], axis=0)
            yc = np.concatenate([np.ones(len(h_pos)), np.zeros(len(h_neg))])
            cav_clf = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
            cav_clf.fit(Hc, yc)
            cav_train_acc = float(cav_clf.score(Hc, yc))
            cav_dir = cav_clf.coef_[0]
            cav_dir = cav_dir / (np.linalg.norm(cav_dir) + 1e-12)
            cav_dir_t = torch.tensor(cav_dir, dtype=torch.float32)

            # ---- TCAV sensitivity: directional derivative of the scalar
            # model output w.r.t. the hidden representation, dotted with the
            # CAV direction, on a sample of test examples ----
            n_tcav = min(400, len(Xte))
            tcav_sample = rng.choice(len(Xte), n_tcav, replace=False)
            sens = []
            for i in tcav_sample:
                x = torch.tensor(Xte[i:i + 1], requires_grad=False)
                h = hidden_wrap(x)
                h.retain_grad()
                out_scalar = head(h).squeeze()
                out_scalar.backward()
                grad_h = h.grad[0].numpy()
                s = float(np.dot(grad_h, cav_dir))
                sens.append(s)
            sens = np.array(sens)
            tcav_score = float((sens > 0).mean())
            tcav_results.append({
                "concept": cname,
                "n_pos_examples": int(n_pos),
                "n_neg_examples": int(n_neg),
                "cav_train_accuracy": cav_train_acc,
                "n_tcav_test_examples": int(n_tcav),
                "tcav_score": tcav_score,
                "mean_directional_derivative": float(sens.mean()),
            })
            print(f"TCAV[{cname}]: score={tcav_score:.4f} (cav_train_acc={cav_train_acc:.4f}, "
                  f"n_pos={n_pos}, n_neg={n_neg})")

        out["tcav"] = tcav_results

        if tcav_results:
            fig, ax = plt.subplots(figsize=(6, 3.4))
            names = [r["concept"] for r in tcav_results]
            scores = [r["tcav_score"] for r in tcav_results]
            ax.barh(names, scores, color="#2F4B94")
            ax.axvline(0.5, color="black", lw=0.8, ls="--", label="chance (0.5)")
            ax.set_xlim(0, 1)
            ax.set_xlabel("TCAV score (fraction of examples with positive directional derivative)")
            ax.set_title("TCAV concept sensitivity (GRU hidden state)", fontsize=11)
            ax.legend(fontsize=8)
            plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_tcav.png"), bbox_inches="tight"); plt.close()
        print("CAV/TCAV done")
    except Exception as e:
        import traceback; traceback.print_exc()
        print("CAV/TCAV step failed:", e)
        out["tcav"] = []

    # ======================================================================
    # shared tabular setup for EBM / GAM / MUSE (classical, PRICE+SENT+GT)
    # ======================================================================
    feats = PRICE_FEATS + SENT_FEATS + GT_FEATS
    from sklearn.preprocessing import StandardScaler
    tr = df[df["Date"].dt.year < 2023]
    te = df[df["Date"].dt.year == 2023]
    sc2 = StandardScaler().fit(tr[feats].values)
    Xtr = pd.DataFrame(sc2.transform(tr[feats].values), columns=feats)
    Xte2 = pd.DataFrame(sc2.transform(te[feats].values), columns=feats)
    ytr, yte = tr["dir_next"].values, te["dir_next"].values

    # ======================================================================
    # 2) EBM intrinsic explanations
    # ======================================================================
    try:
        from interpret.glassbox import ExplainableBoostingClassifier
        ebm = ExplainableBoostingClassifier(random_state=SEED, n_jobs=-1)
        # fit on the DataFrame (not .values) so interpret picks up real column
        # names -- fitting on a bare ndarray makes it label terms generically
        # as 'feature_0000', 'feature_0001', ... which silently breaks any
        # name-based lookup against `feats` downstream.
        ebm.fit(Xtr, ytr)
        ebm_acc = float(ebm.score(Xte2, yte))

        glob = ebm.explain_global()
        gdata = glob.data()
        print("EBM explain_global().data() keys:", list(gdata.keys()) if isinstance(gdata, dict) else type(gdata))
        names = list(gdata["names"])
        scores = list(gdata["scores"])
        # EBM's global explanation includes pairwise interaction terms
        # (named "featA & featB"); keep only single-feature terms so the
        # importance vector aligns 1:1 with `feats`.
        single = {n: abs(float(s)) for n, s in zip(names, scores) if n in feats}
        # some interpret versions omit a feature if it was never split on;
        # fill with 0 so every column in `feats` is represented.
        for f in feats:
            single.setdefault(f, 0.0)
        ebm_imp = pd.Series(single).reindex(feats)

        ebm_faith = _faithfulness(ebm_imp, "ebm_intrinsic")
        ebm_faith["test_accuracy"] = ebm_acc
        ebm_faith["top10"] = ebm_imp.sort_values(ascending=False).head(10).round(5).to_dict()
        out["ebm_intrinsic"] = ebm_faith
        print("EBM intrinsic:", json.dumps(ebm_faith, indent=2, default=str))

        fig, ax = plt.subplots(figsize=(7.5, 6))
        order = ebm_imp.sort_values()
        colors = ["#B4740F" if f in SENT_FEATS else ("#C0362C" if f in GT_FEATS else "#2F4B94")
                  for f in order.index]
        ax.barh(order.index, order.values, color=colors)
        ax.set_xlabel("EBM native global term importance (|score|)")
        ax.set_title("EBM intrinsic global explanation", fontsize=11)
        plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_ebm_intrinsic.png"), bbox_inches="tight"); plt.close()
        print("EBM intrinsic done")
    except Exception as e:
        import traceback; traceback.print_exc()
        print("EBM intrinsic step failed:", e)
        out["ebm_intrinsic"] = {"method": "ebm_intrinsic", "error": str(e)}

    # ======================================================================
    # 3) GAM (Global Attribution Mapping) aggregation via rank-normalized TreeSHAP
    # ======================================================================
    try:
        import shap
        import xgboost as xgb
        xgb_model = xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.03,
                    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, n_jobs=-1,
                    random_state=SEED, eval_metric="logloss")
        xgb_model.fit(Xtr.values, ytr)

        expl = shap.TreeExplainer(xgb_model)
        n_gam = min(500, len(Xte2))
        Xg = Xte2.sample(n_gam, random_state=SEED)
        sv = expl.shap_values(Xg)
        if isinstance(sv, list):
            sv = sv[1]
        abs_sv = np.abs(sv)   # (n_instances, n_feats)

        # rank-normalize WITHIN each instance: rank 1 = most important feature
        # for that instance (largest |SHAP|); average ranks across instances.
        n_feats = abs_sv.shape[1]
        ranks = np.zeros_like(abs_sv, dtype=float)
        for i in range(abs_sv.shape[0]):
            order = np.argsort(-abs_sv[i])          # descending |SHAP|
            r = np.empty(n_feats, dtype=float)
            r[order] = np.arange(1, n_feats + 1)
            ranks[i] = r
        avg_rank = ranks.mean(axis=0)                # lower = more consistently important
        gam_imp = pd.Series(1.0 / avg_rank, index=feats).sort_values(ascending=False)

        gam_faith = _faithfulness(gam_imp, "gam_aggregation")
        gam_faith["n_instances"] = int(n_gam)
        gam_faith["avg_rank_top10"] = pd.Series(avg_rank, index=feats).sort_values().head(10).round(3).to_dict()
        gam_faith["pseudo_importance_top10"] = gam_imp.head(10).round(5).to_dict()
        out["gam_aggregation"] = gam_faith
        print("GAM aggregation:", json.dumps(gam_faith, indent=2, default=str))

        fig, ax = plt.subplots(figsize=(7.5, 6))
        order = gam_imp.sort_values()
        colors = ["#B4740F" if f in SENT_FEATS else ("#C0362C" if f in GT_FEATS else "#2F4B94")
                  for f in order.index]
        ax.barh(order.index, order.values, color=colors)
        ax.set_xlabel("GAM pseudo-importance (1 / mean within-instance SHAP rank)")
        ax.set_title(f"Global Attribution Mapping (rank-aggregated TreeSHAP, n={n_gam})", fontsize=11)
        plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_gam_aggregation.png"), bbox_inches="tight"); plt.close()
        print("GAM aggregation done")
    except Exception as e:
        import traceback; traceback.print_exc()
        print("GAM aggregation step failed:", e)
        out["gam_aggregation"] = {"method": "gam_aggregation", "error": str(e)}

    # ======================================================================
    # 4) MUSE/BRCG global rule surrogate (shallow decision tree on the
    #    black-box XGBoost's own PREDICTIONS -- fidelity, not label accuracy)
    # ======================================================================
    try:
        import xgboost as xgb
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.metrics import accuracy_score

        # reuse the XGBoost fit above if available, else refit
        try:
            bb_model = xgb_model
        except NameError:
            bb_model = xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.03,
                        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, n_jobs=-1,
                        random_state=SEED, eval_metric="logloss")
            bb_model.fit(Xtr.values, ytr)

        bb_test_pred = bb_model.predict(Xte2.values)   # the black box's OWN predictions

        surrogate = DecisionTreeClassifier(max_depth=3, random_state=SEED)
        surrogate.fit(Xte2.values, bb_test_pred)
        surrogate_pred = surrogate.predict(Xte2.values)
        fidelity = float(accuracy_score(bb_test_pred, surrogate_pred))
        label_acc = float(accuracy_score(yte, surrogate_pred))

        muse_imp = pd.Series(surrogate.feature_importances_, index=feats).sort_values(ascending=False)
        muse_faith = _faithfulness(muse_imp, "muse_brcg_surrogate")
        muse_faith["surrogate_fidelity_vs_blackbox"] = fidelity
        muse_faith["surrogate_accuracy_vs_true_labels"] = label_acc
        muse_faith["n_leaves"] = int(surrogate.get_n_leaves())
        muse_faith["top10"] = muse_imp.head(10).round(5).to_dict()
        out["muse_brcg"] = muse_faith
        print("MUSE/BRCG surrogate:", json.dumps(muse_faith, indent=2, default=str))

        fig, ax = plt.subplots(figsize=(7.5, 6))
        order = muse_imp.sort_values()
        colors = ["#B4740F" if f in SENT_FEATS else ("#C0362C" if f in GT_FEATS else "#2F4B94")
                  for f in order.index]
        ax.barh(order.index, order.values, color=colors)
        ax.set_xlabel("Surrogate tree feature_importances_")
        ax.set_title(f"MUSE/BRCG global rule surrogate (fidelity={fidelity:.3f})", fontsize=11)
        plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_muse_brcg.png"), bbox_inches="tight"); plt.close()
        print("MUSE/BRCG done")
    except Exception as e:
        import traceback; traceback.print_exc()
        print("MUSE/BRCG step failed:", e)
        out["muse_brcg"] = {"method": "muse_brcg_surrogate", "error": str(e)}

    with open(os.path.join(MET, "xai_concept_intrinsic.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(json.dumps(out, indent=2, default=str))
    data_vol.commit()
