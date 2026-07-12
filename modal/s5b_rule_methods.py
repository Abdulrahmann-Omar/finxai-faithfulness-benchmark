"""Stage 5b -- rule/example-based and Tier-2 vision-adapted XAI methods
(brief S14): Anchors, Counterfactual explanations, CEM (contrastive
pertinent-positive/pertinent-negative), and RISE (randomized input masking,
adapted from vision to tabular). None of alibi / dice-ml are installed and
the shared image in _common.py is intentionally NOT touched here (other
agents are editing it concurrently) -- every method below is a hand-rolled,
standalone implementation against the same XGBoost direction classifier used
elsewhere in stage 5 (PRICE_FEATS + SENT_FEATS + GT_FEATS, dir_next, 2023
held out).

  modal run s5b_rule_methods.py::run

Per brief S14, methods that only yield local rules/examples (Anchors,
counterfactuals, CEM) are scored on precision/coverage or validity/sparsity,
not against the ground-truth ranking. RISE is the one method here that
produces a genuine global per-feature importance vector (including
gt_noise/gt_signal/gt_redundant_copy), so it alone is scored with
_faithfulness.
"""
import modal
from _common import image, data_vol, DATA, SEED, GROUND_TRUTH

app = modal.App("finsent-s5b-rule-methods", image=image)

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


@app.function(volumes={DATA: data_vol}, cpu=8.0, memory=32768, timeout=60 * 40)
def run():
    import os, json, numpy as np, pandas as pd
    import xgboost as xgb
    from sklearn.preprocessing import StandardScaler
    plt = _mpl()
    os.makedirs(FIG, exist_ok=True); os.makedirs(MET, exist_ok=True)

    feats = PRICE_FEATS + SENT_FEATS + GT_FEATS
    n_feats = len(feats)
    df = pd.read_parquet(os.path.join(BUILD, "panel.parquet")).copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Date", "Symbol"]).reset_index(drop=True)
    tr = df[df["Date"].dt.year < 2023]
    te = df[df["Date"].dt.year == 2023]
    sc = StandardScaler().fit(tr[feats].values)
    Xtr = sc.transform(tr[feats].values)
    Xte = sc.transform(te[feats].values)
    ytr, yte = tr["dir_next"].values, te["dir_next"].values

    model = xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.03,
             subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, n_jobs=-1,
             random_state=SEED, eval_metric="logloss")
    model.fit(Xtr, ytr)
    print(f"model trained: {len(Xtr):,} train rows, {len(Xte):,} test rows, {n_feats} features")

    train_mean = Xtr.mean(axis=0)
    rng = np.random.RandomState(SEED)
    results = {}

    # ======================================================================
    # ANCHORS -- greedy rule search. Grow a set of feature-bin conditions
    # that maximizes LOCAL PRECISION (perturbation agreement with the
    # original instance's prediction), stopping at a target precision or
    # once coverage (fraction of training rows the rule would cover) drops
    # too low.
    # ======================================================================
    N_ANCHOR = 25
    N_BINS = 10
    TARGET_PRECISION = 0.95
    MIN_COVERAGE = 0.02
    MAX_RULE_LEN = 5
    N_PERTURB = 300

    edges = []
    for j in range(n_feats):
        qs = np.quantile(Xtr[:, j], np.linspace(0, 1, N_BINS + 1))
        qs[0] -= 1e-6; qs[-1] += 1e-6
        edges.append(qs)

    def bin_matrix(X):
        B = np.zeros(X.shape, dtype=np.int16)
        for j in range(n_feats):
            B[:, j] = np.searchsorted(edges[j], X[:, j], side="right") - 1
        return B

    Xtr_bins = bin_matrix(Xtr)

    anchor_idx = rng.choice(len(Xte), min(N_ANCHOR, len(Xte)), replace=False)
    anchor_rows = []
    for i in anchor_idx:
        x = Xte[i]
        xb = bin_matrix(x.reshape(1, -1))[0]
        orig_pred = int(model.predict(x.reshape(1, -1))[0])

        remaining = list(range(n_feats))
        chosen = []
        best_precision, best_coverage = 0.0, 1.0
        for _step in range(MAX_RULE_LEN):
            best_f, best_p, best_c = None, -1.0, None
            for f in remaining:
                trial = chosen + [f]
                bgidx = rng.choice(len(Xtr), N_PERTURB, replace=True)
                bg = Xtr[bgidx].copy()
                bg[:, trial] = x[trial]                       # fix anchor feats to x's value
                preds = model.predict(bg)
                precision = float((preds == orig_pred).mean())
                mask = np.ones(len(Xtr_bins), dtype=bool)
                for ff in trial:
                    mask &= (Xtr_bins[:, ff] == xb[ff])
                coverage = float(mask.mean())
                if precision > best_p:
                    best_p, best_f, best_c = precision, f, coverage
            chosen.append(best_f); remaining.remove(best_f)
            best_precision, best_coverage = best_p, best_c
            if best_precision >= TARGET_PRECISION or best_coverage < MIN_COVERAGE:
                break
        anchor_rows.append({
            "rule_len": len(chosen),
            "precision": best_precision,
            "coverage": best_coverage,
            "features": [feats[f] for f in chosen],
        })

    prec_arr = np.array([r["precision"] for r in anchor_rows])
    cov_arr = np.array([r["coverage"] for r in anchor_rows])
    results["anchors"] = {
        "n_instances": len(anchor_rows),
        "target_precision": TARGET_PRECISION,
        "min_coverage_stop": MIN_COVERAGE,
        "n_bins_per_feature": N_BINS,
        "n_perturbations_per_candidate": N_PERTURB,
        "mean_precision": float(prec_arr.mean()),
        "mean_coverage": float(cov_arr.mean()),
        "mean_rule_len": float(np.mean([r["rule_len"] for r in anchor_rows])),
        "pct_reaching_target_precision": float((prec_arr >= TARGET_PRECISION).mean()),
        "examples": anchor_rows[:5],
    }
    print(f"Anchors: mean precision={prec_arr.mean():.3f}  mean coverage={cov_arr.mean():.3f}  "
          f"mean rule length={np.mean([r['rule_len'] for r in anchor_rows]):.2f}")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(cov_arr, prec_arr, color="#2F4B94", s=45, alpha=0.85)
    ax.axhline(TARGET_PRECISION, color="#C0362C", ls="--", lw=1,
               label=f"target precision = {TARGET_PRECISION}")
    ax.set_xlabel("coverage (fraction of train rows satisfying the rule)")
    ax.set_ylabel("local precision (perturbation agreement)")
    ax.set_title(f"Anchors: rule precision vs. coverage (n={len(anchor_rows)} test instances)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_anchors.png"), bbox_inches="tight"); plt.close()

    # ======================================================================
    # COUNTERFACTUAL EXPLANATIONS -- randomized search (increasing number of
    # changed features) for the minimal-distance perturbation that flips the
    # predicted class. Report validity (fraction flipped within budget) and
    # sparsity (features changed).
    # ======================================================================
    N_CF = 30
    MAX_K = 6
    N_TRIALS = 200
    cf_grid = [np.quantile(Xtr[:, j], np.linspace(0, 1, N_BINS + 1)[1:-1]) for j in range(n_feats)]

    cf_idx = rng.choice(len(Xte), min(N_CF, len(Xte)), replace=False)
    cf_rows = []
    for i in cf_idx:
        x = Xte[i]
        orig_pred = int(model.predict(x.reshape(1, -1))[0])
        found, k_used = False, None
        for k in range(1, MAX_K + 1):
            cands = np.tile(x, (N_TRIALS, 1))
            for t in range(N_TRIALS):
                idxs = rng.choice(n_feats, k, replace=False)
                for ix in idxs:
                    cands[t, ix] = rng.choice(cf_grid[ix])
            preds = model.predict(cands)
            flips = np.where(preds != orig_pred)[0]
            if len(flips):
                dists = np.linalg.norm(cands[flips] - x, axis=1)
                best = flips[np.argmin(dists)]
                found, k_used = True, k
                break
        cf_rows.append({"validity": found, "sparsity": k_used})

    valid_cf = [r for r in cf_rows if r["validity"]]
    validity = float(len(valid_cf) / len(cf_rows))
    mean_sparsity = float(np.mean([r["sparsity"] for r in valid_cf])) if valid_cf else float("nan")
    results["counterfactual"] = {
        "n_instances": len(cf_rows),
        "max_features_changed_budget": MAX_K,
        "n_random_trials_per_k": N_TRIALS,
        "validity": validity,
        "mean_sparsity_features_changed": mean_sparsity,
    }
    print(f"Counterfactual: validity={validity:.3f}  mean sparsity={mean_sparsity:.2f} features")

    if valid_cf:
        fig, ax = plt.subplots(figsize=(6, 4))
        sp = [r["sparsity"] for r in valid_cf]
        ax.hist(sp, bins=range(1, MAX_K + 2), align="left", color="#2C7A55", edgecolor="white")
        ax.set_xlabel("number of features changed (sparsity)")
        ax.set_ylabel("count of test instances")
        ax.set_title(f"Counterfactual sparsity (validity={validity*100:.0f}% within budget)",
                     fontsize=10, fontweight="bold")
        plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_counterfactual.png"), bbox_inches="tight"); plt.close()

    # ======================================================================
    # CEM -- pertinent negative = the counterfactual above (minimal change
    # that flips the prediction). Pertinent positive: greedily impute
    # (to the training mean) every feature that CAN be removed without
    # flipping the prediction away from the original; the surviving "kept"
    # set is the pertinent-positive explanation.
    # ======================================================================
    N_CEM = 30
    cem_idx = rng.choice(len(Xte), min(N_CEM, len(Xte)), replace=False)
    pp_sizes = []
    for i in cem_idx:
        x = Xte[i]
        orig_pred = int(model.predict(x.reshape(1, -1))[0])
        kept = list(range(n_feats))
        imputed = x.copy()
        for _pass in range(n_feats):
            if not kept:
                break
            trials = np.tile(imputed, (len(kept), 1))
            for row_i, f in enumerate(kept):
                trials[row_i, f] = train_mean[f]
            preds = model.predict(trials)
            safe = [f for f, p in zip(kept, preds) if p == orig_pred]
            if not safe:
                break
            for f in safe:
                imputed[f] = train_mean[f]
            kept = [f for f in kept if f not in safe]
        pp_sizes.append(len(kept))

    results["cem"] = {
        "n_instances": len(pp_sizes),
        "n_total_features": n_feats,
        "pertinent_positive_mean_size": float(np.mean(pp_sizes)),
        "pertinent_positive_median_size": float(np.median(pp_sizes)),
        "pertinent_negative": {
            "note": "pertinent negative = minimal-change counterfactual, see the 'counterfactual' key",
            "validity": validity,
            "mean_sparsity_features_changed": mean_sparsity,
        },
    }
    print(f"CEM: mean pertinent-positive size={np.mean(pp_sizes):.2f} / {n_feats} features")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(pp_sizes, bins=range(0, n_feats + 2), align="left", color="#B4740F", edgecolor="white")
    ax.set_xlabel("pertinent-positive set size (features kept)")
    ax.set_ylabel("count of test instances")
    ax.set_title(f"CEM pertinent-positive sparsity (mean={np.mean(pp_sizes):.1f}/{n_feats})",
                 fontsize=10, fontweight="bold")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_cem.png"), bbox_inches="tight"); plt.close()

    # ======================================================================
    # RISE -- randomized input masking (vision -> tabular adaptation).
    # Many random binary masks over individual features; masked-out
    # features are imputed to the training mean. Each mask is weighted by
    # the resulting model output; per-feature importance is the difference
    # in mean output between masks that keep vs. drop that feature. This is
    # the one method here that yields a genuine global vector (including
    # gt_noise/gt_signal/gt_redundant_copy), so it is the one scored with
    # _faithfulness.
    # ======================================================================
    N_RISE_INSTANCES = 40
    N_MASKS = 700
    MASK_P = 0.5
    rise_idx = rng.choice(len(Xte), min(N_RISE_INSTANCES, len(Xte)), replace=False)
    per_instance_imp = np.zeros((len(rise_idx), n_feats))
    for row_i, i in enumerate(rise_idx):
        x = Xte[i]
        masks = (rng.random_sample((N_MASKS, n_feats)) < MASK_P).astype(np.float32)
        Xm = masks * x + (1 - masks) * train_mean
        outputs = model.predict_proba(Xm)[:, 1]
        imp = np.zeros(n_feats)
        for f in range(n_feats):
            on = masks[:, f] == 1
            off = ~on
            if on.any() and off.any():
                imp[f] = outputs[on].mean() - outputs[off].mean()
        per_instance_imp[row_i] = imp

    rise_imp = pd.Series(np.abs(per_instance_imp).mean(axis=0), index=feats).sort_values(ascending=False)
    sent_share = float(rise_imp[SENT_FEATS].sum() / rise_imp.sum())
    rise_faith = _faithfulness(rise_imp, "rise")
    results["rise"] = {
        "n_instances": len(rise_idx),
        "n_masks_per_instance": N_MASKS,
        "mask_keep_prob": MASK_P,
        "sentiment_attribution_share": sent_share,
        "top10": rise_imp.head(10).round(5).to_dict(),
        "faithfulness": rise_faith,
    }
    print(f"RISE: sentiment attribution share={sent_share:.4f}  faithfulness={rise_faith}")

    fig, ax = plt.subplots(figsize=(7.5, 6))
    order = rise_imp.sort_values()
    colors = ["#7A2C7A" if f in GT_FEATS else ("#B4740F" if f in SENT_FEATS else "#2F4B94")
              for f in order.index]
    ax.barh(order.index, order.values, color=colors)
    ax.set_xlabel("mean |RISE importance| (mask-on minus mask-off mean P(up))")
    ax.set_title(f"RISE per-feature importance (orange=sentiment, purple=ground-truth): "
                 f"{sent_share*100:.1f}% sentiment share", fontsize=10, fontweight="bold")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_rise.png"), bbox_inches="tight"); plt.close()

    # ======================================================================
    with open(os.path.join(MET, "xai_rule_methods.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    data_vol.commit()
