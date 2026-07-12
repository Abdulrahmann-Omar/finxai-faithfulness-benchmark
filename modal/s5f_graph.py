"""Stage 5f -- GNNExplainer, the graph-native XAI family (brief S14) for the
cross-stock correlation GNN (s4b_gnn.py).

  modal run s5f_graph.py::run

The saved artifact at results/artifacts/gnn_bundle.npz was trained on
PRICE_FEATS+SENT_FEATS only (22 features, no gt_* ground-truth columns), so
it cannot be scored for signal/noise faithfulness. Rather than fabricate
ground-truth ranks for features that were never in that model, this stage
RETRAINS a small GCNNet (same architecture, same build_snapshots/build_graph
pipeline as s4b_gnn.py, modest 8-epoch training on the 2023 test fold) on
PRICE_FEATS+SENT_FEATS+GT_FEATS, so the ground-truth faithfulness spine
(gt_signal vs gt_noise rank, gt_redundant_copy credit-splitting) can be
computed for real from GNNExplainer output, consistent with every other
s5* XAI stage.

Method: torch_geometric.explain.Explainer wrapping GNNExplainer(epochs=100),
node_mask_type='attributes' (per-node, per-feature mask), edge_mask_type=
'object' (per-edge mask), model_config mode='binary_classification' (matches
the dir_next / direction classification head), task_level='node',
return_type='raw' (the model returns raw logits, matching GCNNet.forward).
Run once per sampled test-fold date snapshot (the static graph structure is
shared across all snapshots -- only node features and the resulting masks
differ per date); aggregate node_mask across dates and nodes into one global
per-feature |importance| vector (scored with _faithfulness), and aggregate
edge_mask across dates into a mean per-(undirected)-edge importance, reported
as the top-10 highest-importance ticker pairs.
"""
import modal
from _common import image, data_vol, DATA, SEED, GROUND_TRUTH

app = modal.App("finsent-s5f-graph", image=image)

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

TEST_YEAR = 2023   # matches the saved gnn_bundle.npz / gnn_dir_2023.pt fold


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
    import os, json, numpy as np, pandas as pd, torch
    import torch.nn as nn
    from torch_geometric.nn import GCNConv
    from torch_geometric.explain import Explainer, GNNExplainer
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
    plt = _mpl()
    os.makedirs(MET, exist_ok=True); os.makedirs(FIG, exist_ok=True)

    torch.manual_seed(SEED); np.random.seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev)

    df = pd.read_parquet(os.path.join(BUILD, "panel.parquet")).copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Date", "Symbol"]).reset_index(drop=True)
    tickers = sorted(df["Symbol"].unique())
    n_nodes = len(tickers)
    tick_idx = {t: i for i, t in enumerate(tickers)}
    print(f"universe: {n_nodes} tickers, GT_FEATS={GT_FEATS}")

    # ---- exact same snapshot/graph pipeline as s4b_gnn.py, so node
    # ordering and graph structure match the saved artifact's convention ----
    def build_snapshots(feats):
        piv_counts = df.groupby("Date")["Symbol"].nunique()
        full_dates = piv_counts[piv_counts == n_nodes].index
        sub = df[df["Date"].isin(full_dates)].sort_values(["Date", "Symbol"])
        n_dates = len(full_dates)
        X = np.zeros((n_dates, n_nodes, len(feats)), dtype=np.float32)
        yr = np.zeros((n_dates, n_nodes), dtype=np.float32)
        yd = np.zeros((n_dates, n_nodes), dtype=np.float32)
        date_arr = pd.DatetimeIndex(sorted(full_dates))
        date_pos = {d: i for i, d in enumerate(date_arr)}
        for row in sub.itertuples(index=False):
            di = date_pos[row.Date]; ni = tick_idx[row.Symbol]
            X[di, ni] = [getattr(row, f) for f in feats]
            yr[di, ni] = row.ret_next
            yd[di, ni] = row.dir_next
        years = pd.DatetimeIndex(date_arr).year
        return X, yr, yd, years, date_arr

    def build_graph(train_mask, X_ref_ret):
        ret_tr = X_ref_ret[train_mask]
        corr = np.corrcoef(ret_tr.T)
        np.fill_diagonal(corr, 0.0)
        k = min(5, n_nodes - 1)
        src, dst = [], []
        for i in range(n_nodes):
            nbrs = np.argsort(-np.abs(corr[i]))[:k]
            for j in nbrs:
                src.append(i); dst.append(int(j))
                src.append(int(j)); dst.append(i)
        edge_index = torch.tensor(list({(a, b) for a, b in zip(src, dst)}), dtype=torch.long).t()
        return edge_index, corr

    class GCNNet(nn.Module):
        def __init__(self, nf, hidden=32):
            super().__init__()
            self.conv1 = GCNConv(nf, hidden)
            self.conv2 = GCNConv(hidden, hidden)
            self.act = nn.ReLU()
            self.drop = nn.Dropout(0.2)
            self.head = nn.Linear(hidden, 1)
        def forward(self, x, edge_index):
            x = self.act(self.conv1(x, edge_index))
            x = self.drop(x)
            x = self.act(self.conv2(x, edge_index))
            return self.head(x).squeeze(-1)

    # ---- retrain on PRICE_FEATS+SENT_FEATS+GT_FEATS, dir_next, 2023 fold ----
    feats = PRICE_FEATS + SENT_FEATS + GT_FEATS
    print(f"retraining GCNNet on {len(feats)} features (incl. ground truth) "
          f"for test_year={TEST_YEAR}, task=direction")
    X, yr, yd, years, date_arr = build_snapshots(feats)
    tr_mask = years < TEST_YEAR
    te_mask = years == TEST_YEAR
    assert te_mask.sum() > 0 and tr_mask.sum() >= 200, "insufficient snapshots for the chosen fold"

    edge_index, _corr = build_graph(tr_mask, yr)
    edge_index = edge_index.to(dev)
    n_edges = edge_index.shape[1]
    print(f"graph: {n_edges} directed edges over {n_nodes} nodes")

    sc = StandardScaler().fit(X[tr_mask].reshape(-1, X.shape[-1]))
    Xn = ((X.reshape(-1, X.shape[-1]) - sc.mean_) / np.sqrt(sc.var_ + 1e-9)
          ).reshape(X.shape).astype(np.float32)
    y = yd

    model = GCNNet(len(feats)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    lossf = nn.BCEWithLogitsLoss()

    tr_idx = np.where(tr_mask)[0]
    N_EPOCHS = 8   # modest -- this retrain is for XAI validation, not the model zoo
    for epoch in range(N_EPOCHS):
        model.train(); np.random.shuffle(tr_idx)
        ep_loss = 0.0
        for di in tr_idx:
            xt = torch.tensor(Xn[di]).to(dev)
            yt = torch.tensor(y[di]).to(dev)
            opt.zero_grad()
            out = model(xt, edge_index)
            loss = lossf(out, yt)
            loss.backward(); opt.step()
            ep_loss += float(loss.item())
        print(f"epoch {epoch+1}/{N_EPOCHS} mean_loss={ep_loss/len(tr_idx):.4f}")

    model.eval()
    te_idx = np.where(te_mask)[0]
    preds = np.zeros((len(te_idx), n_nodes), dtype=np.float32)
    with torch.no_grad():
        for k, di in enumerate(te_idx):
            xt = torch.tensor(Xn[di]).to(dev)
            preds[k] = model(xt, edge_index).cpu().numpy()
    yte = y[te_idx]
    phat = 1 / (1 + np.exp(-preds))
    pcls = (phat > 0.5).astype(int)
    test_acc = float(accuracy_score(yte.ravel(), pcls.ravel()))
    test_f1 = float(f1_score(yte.ravel(), pcls.ravel(), zero_division=0))
    test_mcc = float(matthews_corrcoef(yte.ravel(), pcls.ravel()))
    print(f"GT-retrain sanity check: test Acc={test_acc:.4f} F1={test_f1:.4f} MCC={test_mcc:.4f}")

    # ---- GNNExplainer ----
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=100),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(mode="binary_classification", task_level="node", return_type="raw"),
    )

    rng = np.random.RandomState(SEED)
    n_dates_sampled = int(min(25, len(te_idx)))
    sample_pos = rng.choice(len(te_idx), n_dates_sampled, replace=False)
    sample_di = te_idx[sample_pos]
    print(f"sampling {n_dates_sampled} test-fold date snapshots for GNNExplainer")

    node_feat_sum = np.zeros(len(feats), dtype=np.float64)
    n_node_obs = 0
    edge_sum = np.zeros(n_edges, dtype=np.float64)
    n_edge_obs = 0

    for count, di in enumerate(sample_di):
        xt = torch.tensor(Xn[di]).to(dev)
        explanation = explainer(xt, edge_index)
        nm = explanation.node_mask.detach().cpu().numpy()     # [n_nodes, n_feats]
        em = explanation.edge_mask.detach().cpu().numpy()     # [n_edges]
        node_feat_sum += np.abs(nm).sum(axis=0)
        n_node_obs += nm.shape[0]
        edge_sum += np.abs(em)
        n_edge_obs += 1
        if (count + 1) % 5 == 0 or count == 0:
            print(f"  explained {count+1}/{n_dates_sampled} snapshots "
                  f"(date={pd.Timestamp(date_arr[di]).date()})")

    feat_imp = pd.Series(node_feat_sum / n_node_obs, index=feats).sort_values(ascending=False)
    print("GNNExplainer global feature importance (top 10):")
    print(feat_imp.head(10).to_string())

    faith = _faithfulness(feat_imp, "gnnexplainer")
    faith["test_accuracy"] = test_acc
    faith["test_f1"] = test_f1
    faith["test_mcc"] = test_mcc
    faith["top10"] = feat_imp.head(10).round(6).to_dict()
    print("faithfulness:", json.dumps(faith, indent=2, default=str))

    # ---- edge aggregation: collapse the symmetric directed edges into
    # undirected ticker pairs, mean importance across the sampled dates ----
    edge_mean = edge_sum / n_edge_obs   # [n_edges], mean |importance| over sampled dates
    edge_index_np = edge_index.cpu().numpy()
    pair_scores = {}
    for e in range(n_edges):
        i, j = int(edge_index_np[0, e]), int(edge_index_np[1, e])
        key = tuple(sorted((i, j)))
        pair_scores.setdefault(key, []).append(edge_mean[e])
    pairs = [(k, float(np.mean(v))) for k, v in pair_scores.items()]
    pairs.sort(key=lambda kv: -kv[1])
    top_edges = [{"ticker_a": tickers[a], "ticker_b": tickers[b], "mean_importance": imp}
                 for (a, b), imp in pairs[:10]]
    print("top-10 edges by mean GNNExplainer importance:")
    for e in top_edges:
        print(f"  {e['ticker_a']:>6} -- {e['ticker_b']:<6} {e['mean_importance']:.5f}")

    results = {
        "feature_importance": faith,
        "top_edges": top_edges,
        "n_dates_sampled": n_dates_sampled,
        "task": "direction",
        "n_edges_undirected": len(pair_scores),
        "n_edges_directed": int(n_edges),
        "n_nodes": n_nodes,
        "features": feats,
        "retrained_with_gt_feats": True,
        "epochs_retrain": N_EPOCHS,
        "gnnexplainer_epochs": 100,
        "test_year": TEST_YEAR,
    }
    with open(os.path.join(MET, "xai_graph.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("saved", os.path.join(MET, "xai_graph.json"))

    # ---- figure: global feature-importance bar chart ----
    fig, ax = plt.subplots(figsize=(7.5, 6))
    order = feat_imp.sort_values()
    colors = ["#B4740F" if f in SENT_FEATS else ("#C0362C" if f in GT_FEATS else "#2F4B94")
              for f in order.index]
    ax.barh(order.index, order.values, color=colors)
    ax.set_xlabel("mean |GNNExplainer node-feature mask| "
                  f"(over {n_dates_sampled} dates x {n_nodes} nodes)")
    ax.set_title("GNNExplainer global feature importance (cross-stock GCN, direction head)",
                  fontsize=10, fontweight="bold")
    import matplotlib.patches as mp
    ax.legend(handles=[mp.Patch(color="#2F4B94", label="price/technical"),
                       mp.Patch(color="#B4740F", label="news sentiment"),
                       mp.Patch(color="#C0362C", label="ground truth (planted)")],
              loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig_gnnexplainer_features.png"), bbox_inches="tight")
    plt.close()
    print("saved fig_gnnexplainer_features.png")

    data_vol.commit()
