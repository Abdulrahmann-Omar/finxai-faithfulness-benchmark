"""Stage 4b — GNN over a cross-stock correlation graph (graph NN family,
brief S12 model #+1, the "+GNN" beyond the 15). Enables GNNExplainer, the
graph-native XAI cell in the applicability matrix (brief S14).

  modal run s4b_gnn.py::run

Design: one STATIC graph over the universe's tickers (nodes), edges from
return correlation on the TRAINING fold only (top-k neighbors by |corr|, to
avoid a future-information leak into graph structure). Each trading day is
one graph SNAPSHOT: node features = that day's price+sentiment features (same
feature blocks as every other model), node targets = that day's return/
direction for the corresponding stock. A 2-layer GCN does cross-sectional
message passing (no temporal lookback -- this model's contribution is the
GRAPH axis, not the sequence axis, which the GRU/LSTM/CNN/TCN/Transformer
already cover). Walk-forward: the graph and the GCN are both fit on the
training fold only, per test year.
"""
import modal
from _common import image, data_vol, DATA, SEED, TEST_YEARS

app = modal.App("finsent-s4b-gnn", image=image)

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


@app.function(volumes={DATA: data_vol}, gpu="a10g", timeout=60 * 90)
def run():
    import os, json, numpy as np, pandas as pd, torch
    import torch.nn as nn
    from torch_geometric.nn import GCNConv
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (r2_score, mean_squared_error, accuracy_score,
                                 f1_score, matthews_corrcoef)
    os.makedirs(MET, exist_ok=True); os.makedirs(ART, exist_ok=True); os.makedirs(FIG, exist_ok=True)
    torch.manual_seed(SEED); np.random.seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev)

    df = pd.read_parquet(os.path.join(BUILD, "panel.parquet")).copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Date", "Symbol"]).reset_index(drop=True)
    tickers = sorted(df["Symbol"].unique())
    n_nodes = len(tickers)
    tick_idx = {t: i for i, t in enumerate(tickers)}
    print(f"universe: {n_nodes} tickers")

    def build_snapshots(feats):
        """Only keep dates where EVERY ticker has a row (a complete graph
        snapshot); returns X [n_dates, n_nodes, n_feats], y_ret/y_dir
        [n_dates, n_nodes], dates array, aligned by tick_idx node order."""
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
        """Top-k (k=5) neighbor graph by |correlation| of TRAIN-fold returns
        only (never test-fold data -- avoids leaking future co-movement
        structure into the graph)."""
        ret_tr = X_ref_ret[train_mask]                       # [n_train_dates, n_nodes]
        corr = np.corrcoef(ret_tr.T)                          # [n_nodes, n_nodes]
        np.fill_diagonal(corr, 0.0)
        k = min(5, n_nodes - 1)
        src, dst = [], []
        for i in range(n_nodes):
            nbrs = np.argsort(-np.abs(corr[i]))[:k]
            for j in nbrs:
                src.append(i); dst.append(int(j))
                src.append(int(j)); dst.append(i)             # symmetric
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
            return self.head(x).squeeze(-1)                   # [n_nodes]

    def train_eval(feats, fs_name, test_year, task):
        X, yr, yd, years, date_arr = build_snapshots(feats)
        tr_mask = years < test_year
        te_mask = years == test_year
        if te_mask.sum() == 0 or tr_mask.sum() < 200:
            return None
        ret_for_graph = yr   # co-movement of next-day returns defines the graph
        edge_index, _ = build_graph(tr_mask, ret_for_graph)
        edge_index = edge_index.to(dev)

        sc = StandardScaler().fit(X[tr_mask].reshape(-1, X.shape[-1]))
        Xn = ((X.reshape(-1, X.shape[-1]) - sc.mean_) / np.sqrt(sc.var_ + 1e-9)
              ).reshape(X.shape).astype(np.float32)
        y = yr if task == "reg" else yd
        model = GCNNet(X.shape[-1]).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        lossf = nn.MSELoss() if task == "reg" else nn.BCEWithLogitsLoss()

        tr_idx = np.where(tr_mask)[0]
        for epoch in range(12):
            model.train(); np.random.shuffle(tr_idx)
            for di in tr_idx:
                xt = torch.tensor(Xn[di]).to(dev)
                yt = torch.tensor(y[di]).to(dev)
                opt.zero_grad()
                out = model(xt, edge_index)
                loss = lossf(out, yt)
                loss.backward(); opt.step()

        model.eval()
        te_idx = np.where(te_mask)[0]
        preds = np.zeros((len(te_idx), n_nodes), dtype=np.float32)
        with torch.no_grad():
            for k, di in enumerate(te_idx):
                xt = torch.tensor(Xn[di]).to(dev)
                preds[k] = model(xt, edge_index).cpu().numpy()
        yte = y[te_idx]

        if task == "reg":
            m = {"task": "return", "fold": int(test_year), "model": "GNN",
                 "features": fs_name, "R2": float(r2_score(yte.ravel(), preds.ravel())),
                 "RMSE": float(mean_squared_error(yte.ravel(), preds.ravel()) ** .5),
                 "DirAcc": float((np.sign(preds) == np.sign(yte)).mean())}
        else:
            phat = 1 / (1 + np.exp(-preds))
            pcls = (phat > 0.5).astype(int)
            m = {"task": "direction", "fold": int(test_year), "model": "GNN",
                 "features": fs_name, "Acc": float(accuracy_score(yte.ravel(), pcls.ravel())),
                 "F1": float(f1_score(yte.ravel(), pcls.ravel(), zero_division=0)),
                 "MCC": float(matthews_corrcoef(yte.ravel(), pcls.ravel()))}

        if test_year == 2023 and fs_name == "price+sentiment" and task == "clf":
            np.savez(os.path.join(ART, "gnn_bundle.npz"),
                     X_test=Xn[te_idx][:64], feat_names=np.array(feats),
                     tickers=np.array(tickers),
                     edge_index=edge_index.cpu().numpy())
            torch.save(model.state_dict(), os.path.join(ART, "gnn_dir_2023.pt"))
        return m

    rows = []
    for ty in TEST_YEARS:
        for fs_name, feats in [("price_only", PRICE_FEATS),
                               ("price+sentiment", PRICE_FEATS + SENT_FEATS)]:
            for task in ["reg", "clf"]:
                r = train_eval(feats, fs_name, ty, task)
                if r:
                    rows.append(r); print(r)
    pd.DataFrame(rows).to_csv(os.path.join(MET, "gnn_results.csv"), index=False)
    print("saved gnn_results.csv")
    data_vol.commit()
