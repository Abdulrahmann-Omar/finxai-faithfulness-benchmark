"""Stage 4c -- ground-truth-augmented artifacts for sequence/conv/attention
models (gap-fix round 2, "Gap 1"). The main model zoo (s4_lstm.py) is trained
WITHOUT the synthetic ground-truth features (gt_noise/gt_signal/gt_signal_nl/
gt_redundant_copy) on purpose, so the honest price/return/direction results
in the model-zoo table are never polluted by a synthetic leak. But that also
meant Integrated Gradients and the Grad-CAM family had no ground truth to be
scored against (N/A in the XAI method catalogue). This stage trains a SEPARATE,
XAI-purpose-only artifact per architecture (GRU, LSTM, CNN1d, TCN, Transformer)
on PRICE_FEATS+SENT_FEATS+GT_FEATS, fold=2023, direction task only -- mirroring
the pattern already used for the MLP (s5d_gradient_cam.py) and the GNN
(s5f_graph.py), so every sequence-model architecture now has a real
faithfulness-scoreable artifact.

  modal run s4c_gt_artifacts.py::run

Artifacts saved to /data/results/artifacts/ as <model>_gt_dir_2023.pt (state
dict) and <model>_gt_ig_bundle.npz (X_test, feat_names -- now including every
enabled GT_FEATS column).
"""
import modal
from _common import image, data_vol, DATA, SEED, LOOKBACK, GROUND_TRUTH

app = modal.App("finsent-s4c-gt-artifacts", image=image)

BUILD = f"{DATA}/build"
RES = f"{DATA}/results"
MET = f"{RES}/metrics"
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
FEATS = PRICE_FEATS + SENT_FEATS + GT_FEATS
TEST_YEAR = 2023


@app.function(volumes={DATA: data_vol}, gpu="a10g", timeout=60 * 60)
def run():
    import os, json, numpy as np, pandas as pd, torch, math
    import torch.nn as nn
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
    os.makedirs(ART, exist_ok=True); os.makedirs(MET, exist_ok=True)
    torch.manual_seed(SEED); np.random.seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev, "| features:", FEATS)

    df = pd.read_parquet(os.path.join(BUILD, "panel.parquet")).copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Symbol", "Date"]).reset_index(drop=True)

    def build_sequences():
        Xs, yd, yr_year = [], [], []
        for sym, g in df.groupby("Symbol"):
            g = g.sort_values("Date")
            F = g[FEATS].values.astype(np.float32)
            d = g["dir_next"].values.astype(np.float32)
            yr = g["Date"].dt.year.values
            for i in range(LOOKBACK - 1, len(g)):
                Xs.append(F[i - LOOKBACK + 1:i + 1])
                yd.append(d[i]); yr_year.append(yr[i])
        return np.stack(Xs), np.array(yd), np.array(yr_year)

    X, yd, yy = build_sequences()
    tr_mask = yy < TEST_YEAR
    te_mask = yy == TEST_YEAR
    print(f"sequences: {len(X)}, train={tr_mask.sum()}, test={te_mask.sum()}")
    sc = StandardScaler().fit(X[tr_mask].reshape(-1, X.shape[-1]))
    Xn = ((X.reshape(-1, X.shape[-1]) - sc.mean_) / np.sqrt(sc.var_ + 1e-9)
          ).reshape(X.shape).astype(np.float32)
    Xtr = torch.tensor(Xn[tr_mask]); ytr = torch.tensor(yd[tr_mask])
    Xte = torch.tensor(Xn[te_mask]); yte = yd[te_mask]
    nf = X.shape[-1]

    # ---- architectures (copied verbatim from s4_lstm.py / s4b_gnn.py) ----
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
            self.act = nn.ReLU(); self.pool = nn.AdaptiveAvgPool1d(1)
            self.drop = nn.Dropout(0.2); self.head = nn.Linear(channels, 1)
        def forward(self, x):
            x = x.transpose(1, 2)
            x = self.act(self.conv1(x)); x = self.act(self.conv2(x))
            x = self.drop(self.pool(x).squeeze(-1))
            return self.head(x).squeeze(-1)

    class TCNBlock(nn.Module):
        def __init__(self, in_ch, out_ch, kernel_size, dilation):
            super().__init__()
            self.pad = (kernel_size - 1) * dilation
            self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=self.pad, dilation=dilation)
            self.norm = nn.BatchNorm1d(out_ch); self.act = nn.ReLU()
        def forward(self, x):
            out = self.conv(x)
            out = out[:, :, :-self.pad] if self.pad > 0 else out
            return self.act(self.norm(out))

    class TCNNet(nn.Module):
        def __init__(self, nf, channels=32, levels=3, kernel_size=3):
            super().__init__()
            layers, in_ch = [], nf
            for i in range(levels):
                layers.append(TCNBlock(in_ch, channels, kernel_size, dilation=2 ** i)); in_ch = channels
            self.net = nn.Sequential(*layers)
            self.pool = nn.AdaptiveAvgPool1d(1); self.drop = nn.Dropout(0.2)
            self.head = nn.Linear(channels, 1)
        def forward(self, x):
            x = x.transpose(1, 2); x = self.net(x)
            x = self.drop(self.pool(x).squeeze(-1))
            return self.head(x).squeeze(-1)

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=64):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len).unsqueeze(1).float()
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term); pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe.unsqueeze(0))
        def forward(self, x): return x + self.pe[:, :x.size(1)]

    class TransformerNet(nn.Module):
        def __init__(self, nf, d_model=64, nhead=4, num_layers=2):
            super().__init__()
            self.proj = nn.Linear(nf, d_model); self.pos = PositionalEncoding(d_model)
            layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128,
                                               dropout=0.2, batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
            self.head = nn.Linear(d_model, 1)
        def forward(self, x):
            x = self.proj(x); x = self.pos(x); x = self.encoder(x)
            x = x[:, -1]; return self.head(x).squeeze(-1)

    archs = {
        "gru": GRUNet(nf), "lstm": LSTMNet(nf), "cnn1d": CNN1dNet(nf),
        "tcn": TCNNet(nf), "transformer": TransformerNet(nf),
    }
    lossf = nn.BCEWithLogitsLoss()
    results = {}
    for name, model in archs.items():
        model = model.to(dev)
        lr = 5e-4 if name == "transformer" else 1e-3
        epochs = 15 if name == "transformer" else 12
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        bs = 512
        idx = np.arange(len(Xtr))
        for epoch in range(epochs):
            model.train(); np.random.shuffle(idx)
            for s in range(0, len(idx), bs):
                b = idx[s:s + bs]
                opt.zero_grad()
                out = model(Xtr[b].to(dev))
                loss = lossf(out, ytr[b].to(dev))
                loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            preds = []
            for s in range(0, len(Xte), 1024):
                preds.append(model(Xte[s:s + 1024].to(dev)).cpu().numpy())
            pr = np.concatenate(preds)
        phat = 1 / (1 + np.exp(-pr))
        pcls = (phat > 0.5).astype(int)
        m = {"model": name, "task": "direction", "fold": TEST_YEAR,
             "features": "price+sentiment+gt", "n_features": nf,
             "Acc": float(accuracy_score(yte, pcls)),
             "F1": float(f1_score(yte, pcls, zero_division=0)),
             "MCC": float(matthews_corrcoef(yte, pcls))}
        results[name] = m
        print(m)
        np.savez(os.path.join(ART, f"{name}_gt_ig_bundle.npz"),
                 X_test=Xn[te_mask][:512], feat_names=np.array(FEATS))
        torch.save(model.state_dict(), os.path.join(ART, f"{name}_gt_dir_2023.pt"))
        print(f"saved {name}_gt_dir_2023.pt + {name}_gt_ig_bundle.npz")

    with open(os.path.join(MET, "gt_artifacts_summary.json"), "w") as f:
        json.dump({"features": FEATS, "n_features": nf, "test_year": TEST_YEAR,
                   "results": results}, f, indent=2)
    print("saved gt_artifacts_summary.json")
    data_vol.commit()
