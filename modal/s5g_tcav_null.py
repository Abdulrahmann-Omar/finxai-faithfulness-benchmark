"""Stage 5g -- TCAV random-concept significance null (Kim et al. 2018).

  modal run s5g_tcav_null.py::run

s5c_concept_intrinsic.py reports a raw TCAV score for two real concepts
("high_attention", "positive_sentiment") on the trained GRU's last hidden
state: the fraction of a test sample whose directional derivative of the
scalar model output w.r.t. the CAV direction is positive. A raw score (e.g.
1.00) is not by itself evidence of a meaningful concept direction -- ANY
linear direction in a high-dimensional hidden space can produce a lopsided
TCAV score by chance. The standard fix (Kim et al. 2018, TCAV paper) is a
random-concept significance test: fit many CAVs from RANDOM relabelings of
the same activation pool and compare the real score against that null
distribution.

This script:
  1. Reloads the cached GRU (gru_dir_2023.pt) and its feature list
     (gru_ig_bundle.npz), rebuilds the exact same standardized test
     sequences s5c uses, and redefines the same two real concepts
     (high_attention: news_count > p75 vs < p50; positive_sentiment:
     fb_score > 0 vs < 0), all on TEST (2023) rows.
  2. For each real concept: fits the real CAV (LogisticRegression on the
     concept's activation pool) exactly like s5c, and reuses s5c's TCAV
     score formula (sens > 0).mean().
  3. Builds a null of M=100 random CAVs by permuting the labels of the SAME
     activation pool (same activations, random 0/1 relabeling with the same
     class sizes), fitting a LogisticRegression on each permutation, and
     scoring it with the SAME precomputed per-example gradients (the
     directional derivative of the model output w.r.t. the hidden state does
     not depend on which CAV is being tested, only on the model and the
     input -- so it is computed once per concept and reused for the real CAV
     and all M null CAVs, which is exact and far cheaper than 101 backward
     passes per example).
  4. Reports an empirical two-sided p-value, a z-score, and a normal-
     approximation p-value per concept.

Writes results/metrics/xai_tcav_significance.json. Does NOT touch s5c's
outputs (xai_concept_intrinsic.json, fig_tcav.png).
"""
import modal
from _common import image, data_vol, DATA, SEED, LOOKBACK

app = modal.App("finsent-s5g-tcav-null", image=image)

BUILD = f"{DATA}/build"
RES = f"{DATA}/results"
MET = f"{RES}/metrics"
ART = f"{RES}/artifacts"

M_NULL = 100


@app.function(volumes={DATA: data_vol}, cpu=4.0, memory=16384, timeout=30 * 60)
def run():
    import os, json, math
    import numpy as np
    import pandas as pd
    import torch, torch.nn as nn
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import norm

    os.makedirs(MET, exist_ok=True)

    df = pd.read_parquet(os.path.join(BUILD, "panel.parquet")).copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Symbol", "Date"]).reset_index(drop=True)

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
        directly (before the head) -- same wrapper as s5c, used both to
        build CAVs and to compute the directional derivative of the head
        output w.r.t. that hidden representation."""
        def __init__(self, gru_submodule):
            super().__init__()
            self.gru = gru_submodule
        def forward(self, x):
            o, _ = self.gru(x)
            return o[:, -1]

    gru_path = os.path.join(ART, "gru_dir_2023.pt")
    full_gru = GRUNet(len(gru_feats))
    full_gru.load_state_dict(torch.load(gru_path, map_location="cpu"))
    full_gru.eval()
    hidden_wrap = GRUHiddenWrap(full_gru.gru)
    hidden_wrap.eval()
    head = full_gru.head
    print(f"loaded cached GRU from {gru_path} with {len(gru_feats)} input features")

    # ---- build sequences exactly like s4_lstm.py / s5c ----
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

    sc = StandardScaler().fit(X_all[tr_mask].reshape(-1, X_all.shape[-1]))
    Xn = ((X_all.reshape(-1, X_all.shape[-1]) - sc.mean_) / np.sqrt(sc.var_ + 1e-9)
          ).reshape(X_all.shape).astype(np.float32)

    Xte = Xn[te_mask]
    news_te = news_all[te_mask]
    fb_te = fb_all[te_mask]

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

    results = []
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

        real_clf = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
        real_clf.fit(Hc, yc)
        cav_train_acc = float(real_clf.score(Hc, yc))
        real_dir = real_clf.coef_[0]
        real_dir = real_dir / (np.linalg.norm(real_dir) + 1e-12)

        # ---- directional derivative of the scalar output w.r.t. the hidden
        # state, on a fixed test sample -- computed ONCE per concept, then
        # reused (dotted with each CAV direction) for the real score and all
        # M null scores, since grad_h does not depend on the CAV. ----
        n_tcav = min(400, len(Xte))
        tcav_sample = rng.choice(len(Xte), n_tcav, replace=False)
        grad_h = np.zeros((n_tcav, Hc.shape[1]), dtype=np.float64)
        for j, i in enumerate(tcav_sample):
            x = torch.tensor(Xte[i:i + 1], requires_grad=False)
            h = hidden_wrap(x)
            h.retain_grad()
            out_scalar = head(h).squeeze()
            out_scalar.backward()
            grad_h[j] = h.grad[0].numpy()

        def tcav_score_for(direction):
            sens = grad_h @ direction
            return float((sens > 0).mean())

        real_score = tcav_score_for(real_dir)

        null_scores = np.zeros(M_NULL, dtype=np.float64)
        for m in range(M_NULL):
            yc_perm = rng.permutation(yc)
            # guard against a degenerate single-class permutation (won't
            # happen in practice with n_pos, n_neg >= 20, but keep it safe)
            if len(np.unique(yc_perm)) < 2:
                null_scores[m] = 0.5
                continue
            null_clf = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED + 1 + m)
            null_clf.fit(Hc, yc_perm)
            null_dir = null_clf.coef_[0]
            null_dir = null_dir / (np.linalg.norm(null_dir) + 1e-12)
            null_scores[m] = tcav_score_for(null_dir)

        random_mean = float(null_scores.mean())
        random_std = float(null_scores.std(ddof=1))
        z = float((real_score - random_mean) / random_std) if random_std > 0 else float("inf")
        n_extreme = int(np.sum(np.abs(null_scores - 0.5) >= np.abs(real_score - 0.5)))
        p_empirical = float((1 + n_extreme) / (M_NULL + 1))
        p_normal_approx = float(2 * norm.sf(abs(z))) if math.isfinite(z) else 0.0
        significant = bool(p_empirical < 0.05)

        entry = {
            "concept": cname,
            "n_pos_examples": int(n_pos),
            "n_neg_examples": int(n_neg),
            "cav_train_accuracy": cav_train_acc,
            "n_tcav_test_examples": int(n_tcav),
            "tcav_score": real_score,
            "random_mean": random_mean,
            "random_std": random_std,
            "z": z,
            "p_empirical": p_empirical,
            "p_normal_approx": p_normal_approx,
            "significant": significant,
        }
        results.append(entry)
        print(f"TCAV[{cname}]: real={real_score:.4f} null_mean={random_mean:.4f} "
              f"null_std={random_std:.4f} z={z:.3f} p_emp={p_empirical:.4f} "
              f"p_norm={p_normal_approx:.4g} significant={significant}")

    n_sig = sum(1 for r in results if r["significant"])
    n_tot = len(results)
    if n_tot == 0:
        verdict = "No real concepts had enough examples to test; no significance claim can be made."
    elif n_sig == n_tot:
        verdict = (f"All {n_tot} real concepts ({', '.join(r['concept'] for r in results)}) "
                   f"beat the {M_NULL}-random-CAV null at p<0.05 (empirical two-sided test), "
                   "so the TCAV directions are statistically distinguishable from random hidden-state directions.")
    elif n_sig == 0:
        verdict = (f"None of the {n_tot} real concepts beat the {M_NULL}-random-CAV null at p<0.05; "
                   "the raw TCAV scores are not statistically distinguishable from a random linear "
                   "direction in the GRU's hidden state, so TCAV significance is NOT established here "
                   "and the paper should carry this as a caveat rather than a claim.")
    else:
        sig_names = ", ".join(r["concept"] for r in results if r["significant"])
        nonsig_names = ", ".join(r["concept"] for r in results if not r["significant"])
        verdict = (f"{n_sig}/{n_tot} real concepts beat the {M_NULL}-random-CAV null at p<0.05 "
                   f"({sig_names} significant; {nonsig_names} not significant vs. the random-direction null).")

    out = {
        "method": "TCAV random-concept null (Kim et al. 2018)",
        "n_random_cavs": M_NULL,
        "concepts": results,
        "verdict": verdict,
    }

    out_path = os.path.join(MET, "xai_tcav_significance.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(json.dumps(out, indent=2, default=str))
    data_vol.commit()
    return out
