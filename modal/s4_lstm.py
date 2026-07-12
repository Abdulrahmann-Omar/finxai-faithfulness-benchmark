"""Stage 4 — sequence-model baselines (GRU, LSTM, 1D-CNN, TCN, Transformer) on GPU.

  modal run s4_lstm.py::run              # GRU (recurrent)
  modal run s4_lstm.py::run_lstm         # LSTM (recurrent)
  modal run s4_lstm.py::run_cnn1d        # 1D-CNN over the lookback window (conv)
  modal run s4_lstm.py::run_tcn          # TCN: dilated causal conv stack (conv)
  modal run s4_lstm.py::run_transformer  # small self-attention encoder (attention)

Each trains for next-day return (regression) and direction (classification),
price-only vs price+sentiment, using the same expanding walk-forward folds and
strictly chronological sequence construction (a sequence for day t uses only
days <= t). Saves metrics and a held-out test batch (+ model weights) for
Integrated-Gradients attribution (and, model-class-specific: Grad-CAM family
for the CNN/TCN; attention-as-explanation/TCAV/SAE/ACDC for the Transformer).
"""
import modal
from _common import image, data_vol, DATA, SEED, LOOKBACK, TEST_YEARS

app = modal.App("finsent-s4-lstm", image=image)

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
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (r2_score, mean_squared_error, accuracy_score,
                                 f1_score, matthews_corrcoef)
    os.makedirs(MET, exist_ok=True); os.makedirs(ART, exist_ok=True); os.makedirs(FIG, exist_ok=True)
    torch.manual_seed(SEED); np.random.seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev)

    df = pd.read_parquet(os.path.join(BUILD, "panel.parquet")).copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Symbol", "Date"]).reset_index(drop=True)

    def build_sequences(feats):
        Xs, ys, yd, yr_year, syms = [], [], [], [], []
        for sym, g in df.groupby("Symbol"):
            g = g.sort_values("Date")
            F = g[feats].values.astype(np.float32)
            r = g["ret_next"].values.astype(np.float32)
            d = g["dir_next"].values.astype(np.float32)
            yr = g["Date"].dt.year.values
            for i in range(LOOKBACK - 1, len(g)):
                Xs.append(F[i - LOOKBACK + 1:i + 1])
                ys.append(r[i]); yd.append(d[i]); yr_year.append(yr[i]); syms.append(sym)
        return (np.stack(Xs), np.array(ys), np.array(yd),
                np.array(yr_year), np.array(syms))

    class GRUNet(nn.Module):
        def __init__(self, nf, hidden=64):
            super().__init__()
            self.gru = nn.GRU(nf, hidden, num_layers=2, batch_first=True, dropout=0.2)
            self.head = nn.Linear(hidden, 1)
        def forward(self, x):
            o, _ = self.gru(x)
            return self.head(o[:, -1]).squeeze(-1)

    def train_eval(feats, fs_name, test_year, task):
        X, yr, yd, yy, syms = build_sequences(feats)
        tr = yy < test_year
        te = yy == test_year
        if te.sum() == 0 or tr.sum() < 500:
            return None
        # standardize using training rows only (flatten over time)
        sc = StandardScaler().fit(X[tr].reshape(-1, X.shape[-1]))
        Xn = ((X.reshape(-1, X.shape[-1]) - sc.mean_) / np.sqrt(sc.var_ + 1e-9)
              ).reshape(X.shape).astype(np.float32)
        y = yr if task == "reg" else yd
        Xtr = torch.tensor(Xn[tr]); ytr = torch.tensor(y[tr].astype(np.float32))
        Xte = torch.tensor(Xn[te]); yte = y[te]
        model = GRUNet(X.shape[-1]).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        lossf = nn.MSELoss() if task == "reg" else nn.BCEWithLogitsLoss()
        bs = 512
        idx = np.arange(len(Xtr))
        for epoch in range(12):
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
        if task == "reg":
            m = {"task": "return", "fold": int(test_year), "model": "GRU",
                 "features": fs_name, "R2": float(r2_score(yte, pr)),
                 "RMSE": float(mean_squared_error(yte, pr) ** .5),
                 "DirAcc": float((np.sign(pr) == np.sign(yte)).mean())}
        else:
            phat = (1 / (1 + np.exp(-pr)))
            pcls = (phat > 0.5).astype(int)
            m = {"task": "direction", "fold": int(test_year), "model": "GRU",
                 "features": fs_name, "Acc": float(accuracy_score(yte, pcls)),
                 "F1": float(f1_score(yte, pcls, zero_division=0)),
                 "MCC": float(matthews_corrcoef(yte, pcls))}
        # save one artifact bundle (2023, price+sentiment, direction) for IG
        if test_year == 2023 and fs_name == "price+sentiment" and task == "clf":
            np.savez(os.path.join(ART, "gru_ig_bundle.npz"),
                     X_test=Xn[te][:512], feat_names=np.array(feats))
            torch.save(model.state_dict(), os.path.join(ART, "gru_dir_2023.pt"))
        return m

    rows = []
    for ty in TEST_YEARS:
        for fs_name, feats in [("price_only", PRICE_FEATS),
                               ("price+sentiment", PRICE_FEATS + SENT_FEATS)]:
            for task in ["reg", "clf"]:
                r = train_eval(feats, fs_name, ty, task)
                if r:
                    rows.append(r); print(r)
    pd.DataFrame(rows).to_csv(os.path.join(MET, "lstm_results.csv"), index=False)
    print("saved lstm_results.csv")
    data_vol.commit()


@app.function(volumes={DATA: data_vol}, gpu="a10g", timeout=60 * 90)
def run_cnn1d():
    """1D-CNN over the lookback window (conv NN family, brief S12 model #13).
    Same protocol as run() (GRU) so the two are directly comparable on the
    model-class axis (RQ3)."""
    import os, json, numpy as np, pandas as pd, torch
    import torch.nn as nn
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (r2_score, mean_squared_error, accuracy_score,
                                 f1_score, matthews_corrcoef)
    os.makedirs(MET, exist_ok=True); os.makedirs(ART, exist_ok=True); os.makedirs(FIG, exist_ok=True)
    torch.manual_seed(SEED); np.random.seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev)

    df = pd.read_parquet(os.path.join(BUILD, "panel.parquet")).copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Symbol", "Date"]).reset_index(drop=True)

    def build_sequences(feats):
        Xs, ys, yd, yr_year, syms = [], [], [], [], []
        for sym, g in df.groupby("Symbol"):
            g = g.sort_values("Date")
            F = g[feats].values.astype(np.float32)
            r = g["ret_next"].values.astype(np.float32)
            d = g["dir_next"].values.astype(np.float32)
            yr = g["Date"].dt.year.values
            for i in range(LOOKBACK - 1, len(g)):
                Xs.append(F[i - LOOKBACK + 1:i + 1])
                ys.append(r[i]); yd.append(d[i]); yr_year.append(yr[i]); syms.append(sym)
        return (np.stack(Xs), np.array(ys), np.array(yd),
                np.array(yr_year), np.array(syms))

    class CNN1dNet(nn.Module):
        def __init__(self, nf, channels=32):
            super().__init__()
            self.conv1 = nn.Conv1d(nf, channels, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
            self.act = nn.ReLU()
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.drop = nn.Dropout(0.2)
            self.head = nn.Linear(channels, 1)

        def forward(self, x):            # x: (B, T, F)
            x = x.transpose(1, 2)        # (B, F, T) -- conv over the time/lookback axis
            x = self.act(self.conv1(x))
            x = self.act(self.conv2(x))
            x = self.drop(self.pool(x).squeeze(-1))
            return self.head(x).squeeze(-1)

    def train_eval(feats, fs_name, test_year, task):
        X, yr, yd, yy, syms = build_sequences(feats)
        tr = yy < test_year
        te = yy == test_year
        if te.sum() == 0 or tr.sum() < 500:
            return None
        sc = StandardScaler().fit(X[tr].reshape(-1, X.shape[-1]))
        Xn = ((X.reshape(-1, X.shape[-1]) - sc.mean_) / np.sqrt(sc.var_ + 1e-9)
              ).reshape(X.shape).astype(np.float32)
        y = yr if task == "reg" else yd
        Xtr = torch.tensor(Xn[tr]); ytr = torch.tensor(y[tr].astype(np.float32))
        Xte = torch.tensor(Xn[te]); yte = y[te]
        model = CNN1dNet(X.shape[-1]).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        lossf = nn.MSELoss() if task == "reg" else nn.BCEWithLogitsLoss()
        bs = 512
        idx = np.arange(len(Xtr))
        for epoch in range(12):
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
        if task == "reg":
            m = {"task": "return", "fold": int(test_year), "model": "CNN1d",
                 "features": fs_name, "R2": float(r2_score(yte, pr)),
                 "RMSE": float(mean_squared_error(yte, pr) ** .5),
                 "DirAcc": float((np.sign(pr) == np.sign(yte)).mean())}
        else:
            phat = (1 / (1 + np.exp(-pr)))
            pcls = (phat > 0.5).astype(int)
            m = {"task": "direction", "fold": int(test_year), "model": "CNN1d",
                 "features": fs_name, "Acc": float(accuracy_score(yte, pcls)),
                 "F1": float(f1_score(yte, pcls, zero_division=0)),
                 "MCC": float(matthews_corrcoef(yte, pcls))}
        # save one artifact bundle (2023, price+sentiment, direction) for IG
        if test_year == 2023 and fs_name == "price+sentiment" and task == "clf":
            np.savez(os.path.join(ART, "cnn1d_ig_bundle.npz"),
                     X_test=Xn[te][:512], feat_names=np.array(feats))
            torch.save(model.state_dict(), os.path.join(ART, "cnn1d_dir_2023.pt"))
        return m

    rows = []
    for ty in TEST_YEARS:
        for fs_name, feats in [("price_only", PRICE_FEATS),
                               ("price+sentiment", PRICE_FEATS + SENT_FEATS)]:
            for task in ["reg", "clf"]:
                r = train_eval(feats, fs_name, ty, task)
                if r:
                    rows.append(r); print(r)
    pd.DataFrame(rows).to_csv(os.path.join(MET, "cnn1d_results.csv"), index=False)
    print("saved cnn1d_results.csv")
    data_vol.commit()


@app.function(volumes={DATA: data_vol}, gpu="a10g", timeout=60 * 90)
def run_lstm():
    """LSTM (recurrent NN family, brief S12 model #12) -- identical protocol to
    run() (GRU), swapping the recurrent cell, so the two recurrent variants are
    directly comparable on the model-class axis (RQ3)."""
    import os, json, numpy as np, pandas as pd, torch
    import torch.nn as nn
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (r2_score, mean_squared_error, accuracy_score,
                                 f1_score, matthews_corrcoef)
    os.makedirs(MET, exist_ok=True); os.makedirs(ART, exist_ok=True); os.makedirs(FIG, exist_ok=True)
    torch.manual_seed(SEED); np.random.seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev)

    df = pd.read_parquet(os.path.join(BUILD, "panel.parquet")).copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Symbol", "Date"]).reset_index(drop=True)

    def build_sequences(feats):
        Xs, ys, yd, yr_year, syms = [], [], [], [], []
        for sym, g in df.groupby("Symbol"):
            g = g.sort_values("Date")
            F = g[feats].values.astype(np.float32)
            r = g["ret_next"].values.astype(np.float32)
            d = g["dir_next"].values.astype(np.float32)
            yr = g["Date"].dt.year.values
            for i in range(LOOKBACK - 1, len(g)):
                Xs.append(F[i - LOOKBACK + 1:i + 1])
                ys.append(r[i]); yd.append(d[i]); yr_year.append(yr[i]); syms.append(sym)
        return (np.stack(Xs), np.array(ys), np.array(yd),
                np.array(yr_year), np.array(syms))

    class LSTMNet(nn.Module):
        def __init__(self, nf, hidden=64):
            super().__init__()
            self.lstm = nn.LSTM(nf, hidden, num_layers=2, batch_first=True, dropout=0.2)
            self.head = nn.Linear(hidden, 1)
        def forward(self, x):
            o, _ = self.lstm(x)
            return self.head(o[:, -1]).squeeze(-1)

    def train_eval(feats, fs_name, test_year, task):
        X, yr, yd, yy, syms = build_sequences(feats)
        tr = yy < test_year
        te = yy == test_year
        if te.sum() == 0 or tr.sum() < 500:
            return None
        sc = StandardScaler().fit(X[tr].reshape(-1, X.shape[-1]))
        Xn = ((X.reshape(-1, X.shape[-1]) - sc.mean_) / np.sqrt(sc.var_ + 1e-9)
              ).reshape(X.shape).astype(np.float32)
        y = yr if task == "reg" else yd
        Xtr = torch.tensor(Xn[tr]); ytr = torch.tensor(y[tr].astype(np.float32))
        Xte = torch.tensor(Xn[te]); yte = y[te]
        model = LSTMNet(X.shape[-1]).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        lossf = nn.MSELoss() if task == "reg" else nn.BCEWithLogitsLoss()
        bs = 512
        idx = np.arange(len(Xtr))
        for epoch in range(12):
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
        if task == "reg":
            m = {"task": "return", "fold": int(test_year), "model": "LSTM",
                 "features": fs_name, "R2": float(r2_score(yte, pr)),
                 "RMSE": float(mean_squared_error(yte, pr) ** .5),
                 "DirAcc": float((np.sign(pr) == np.sign(yte)).mean())}
        else:
            phat = (1 / (1 + np.exp(-pr)))
            pcls = (phat > 0.5).astype(int)
            m = {"task": "direction", "fold": int(test_year), "model": "LSTM",
                 "features": fs_name, "Acc": float(accuracy_score(yte, pcls)),
                 "F1": float(f1_score(yte, pcls, zero_division=0)),
                 "MCC": float(matthews_corrcoef(yte, pcls))}
        if test_year == 2023 and fs_name == "price+sentiment" and task == "clf":
            np.savez(os.path.join(ART, "lstm_ig_bundle.npz"),
                     X_test=Xn[te][:512], feat_names=np.array(feats))
            torch.save(model.state_dict(), os.path.join(ART, "lstm_dir_2023.pt"))
        return m

    rows = []
    for ty in TEST_YEARS:
        for fs_name, feats in [("price_only", PRICE_FEATS),
                               ("price+sentiment", PRICE_FEATS + SENT_FEATS)]:
            for task in ["reg", "clf"]:
                r = train_eval(feats, fs_name, ty, task)
                if r:
                    rows.append(r); print(r)
    pd.DataFrame(rows).to_csv(os.path.join(MET, "lstm_model_results.csv"), index=False)
    print("saved lstm_model_results.csv")
    data_vol.commit()


@app.function(volumes={DATA: data_vol}, gpu="a10g", timeout=60 * 90)
def run_tcn():
    """Temporal Convolutional Network (dilated causal conv, brief S12 model #14):
    a stack of causally-padded, increasingly-dilated 1D convs over the lookback
    window -- the modern conv-sequence alternative to the plain 1D-CNN, and
    architecturally distinct enough from CNN1d to be a separate model-class cell."""
    import os, json, numpy as np, pandas as pd, torch
    import torch.nn as nn
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (r2_score, mean_squared_error, accuracy_score,
                                 f1_score, matthews_corrcoef)
    os.makedirs(MET, exist_ok=True); os.makedirs(ART, exist_ok=True); os.makedirs(FIG, exist_ok=True)
    torch.manual_seed(SEED); np.random.seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev)

    df = pd.read_parquet(os.path.join(BUILD, "panel.parquet")).copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Symbol", "Date"]).reset_index(drop=True)

    def build_sequences(feats):
        Xs, ys, yd, yr_year, syms = [], [], [], [], []
        for sym, g in df.groupby("Symbol"):
            g = g.sort_values("Date")
            F = g[feats].values.astype(np.float32)
            r = g["ret_next"].values.astype(np.float32)
            d = g["dir_next"].values.astype(np.float32)
            yr = g["Date"].dt.year.values
            for i in range(LOOKBACK - 1, len(g)):
                Xs.append(F[i - LOOKBACK + 1:i + 1])
                ys.append(r[i]); yd.append(d[i]); yr_year.append(yr[i]); syms.append(sym)
        return (np.stack(Xs), np.array(ys), np.array(yd),
                np.array(yr_year), np.array(syms))

    class TCNBlock(nn.Module):
        def __init__(self, in_ch, out_ch, kernel_size, dilation):
            super().__init__()
            self.pad = (kernel_size - 1) * dilation   # causal: only look at the past
            self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=self.pad, dilation=dilation)
            self.norm = nn.BatchNorm1d(out_ch)
            self.act = nn.ReLU()
        def forward(self, x):
            out = self.conv(x)
            out = out[:, :, :-self.pad] if self.pad > 0 else out   # chomp trailing (future) leakage
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
        def forward(self, x):            # x: (B, T, F)
            x = x.transpose(1, 2)        # (B, F, T)
            x = self.net(x)
            x = self.drop(self.pool(x).squeeze(-1))
            return self.head(x).squeeze(-1)

    def train_eval(feats, fs_name, test_year, task):
        X, yr, yd, yy, syms = build_sequences(feats)
        tr = yy < test_year
        te = yy == test_year
        if te.sum() == 0 or tr.sum() < 500:
            return None
        sc = StandardScaler().fit(X[tr].reshape(-1, X.shape[-1]))
        Xn = ((X.reshape(-1, X.shape[-1]) - sc.mean_) / np.sqrt(sc.var_ + 1e-9)
              ).reshape(X.shape).astype(np.float32)
        y = yr if task == "reg" else yd
        Xtr = torch.tensor(Xn[tr]); ytr = torch.tensor(y[tr].astype(np.float32))
        Xte = torch.tensor(Xn[te]); yte = y[te]
        model = TCNNet(X.shape[-1]).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        lossf = nn.MSELoss() if task == "reg" else nn.BCEWithLogitsLoss()
        bs = 512
        idx = np.arange(len(Xtr))
        for epoch in range(12):
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
        if task == "reg":
            m = {"task": "return", "fold": int(test_year), "model": "TCN",
                 "features": fs_name, "R2": float(r2_score(yte, pr)),
                 "RMSE": float(mean_squared_error(yte, pr) ** .5),
                 "DirAcc": float((np.sign(pr) == np.sign(yte)).mean())}
        else:
            phat = (1 / (1 + np.exp(-pr)))
            pcls = (phat > 0.5).astype(int)
            m = {"task": "direction", "fold": int(test_year), "model": "TCN",
                 "features": fs_name, "Acc": float(accuracy_score(yte, pcls)),
                 "F1": float(f1_score(yte, pcls, zero_division=0)),
                 "MCC": float(matthews_corrcoef(yte, pcls))}
        if test_year == 2023 and fs_name == "price+sentiment" and task == "clf":
            np.savez(os.path.join(ART, "tcn_ig_bundle.npz"),
                     X_test=Xn[te][:512], feat_names=np.array(feats))
            torch.save(model.state_dict(), os.path.join(ART, "tcn_dir_2023.pt"))
        return m

    rows = []
    for ty in TEST_YEARS:
        for fs_name, feats in [("price_only", PRICE_FEATS),
                               ("price+sentiment", PRICE_FEATS + SENT_FEATS)]:
            for task in ["reg", "clf"]:
                r = train_eval(feats, fs_name, ty, task)
                if r:
                    rows.append(r); print(r)
    pd.DataFrame(rows).to_csv(os.path.join(MET, "tcn_results.csv"), index=False)
    print("saved tcn_results.csv")
    data_vol.commit()


@app.function(volumes={DATA: data_vol}, gpu="a10g", timeout=60 * 90)
def run_transformer():
    """Small Transformer encoder (attention NN family, brief S12 model #15) --
    the model that enables attention-as-explanation, TCAV, Sparse Autoencoders,
    and ACDC (brief S14; those XAI methods are built in a later stage against
    this model's saved artifact bundle, not here)."""
    import os, json, math, numpy as np, pandas as pd, torch
    import torch.nn as nn
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (r2_score, mean_squared_error, accuracy_score,
                                 f1_score, matthews_corrcoef)
    os.makedirs(MET, exist_ok=True); os.makedirs(ART, exist_ok=True); os.makedirs(FIG, exist_ok=True)
    torch.manual_seed(SEED); np.random.seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev)

    df = pd.read_parquet(os.path.join(BUILD, "panel.parquet")).copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Symbol", "Date"]).reset_index(drop=True)

    def build_sequences(feats):
        Xs, ys, yd, yr_year, syms = [], [], [], [], []
        for sym, g in df.groupby("Symbol"):
            g = g.sort_values("Date")
            F = g[feats].values.astype(np.float32)
            r = g["ret_next"].values.astype(np.float32)
            d = g["dir_next"].values.astype(np.float32)
            yr = g["Date"].dt.year.values
            for i in range(LOOKBACK - 1, len(g)):
                Xs.append(F[i - LOOKBACK + 1:i + 1])
                ys.append(r[i]); yd.append(d[i]); yr_year.append(yr[i]); syms.append(sym)
        return (np.stack(Xs), np.array(ys), np.array(yd),
                np.array(yr_year), np.array(syms))

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
        def forward(self, x):            # x: (B, T, F)
            x = self.proj(x)
            x = self.pos(x)
            x = self.encoder(x)
            x = x[:, -1]                 # last-token representation
            return self.head(x).squeeze(-1)

    def train_eval(feats, fs_name, test_year, task):
        X, yr, yd, yy, syms = build_sequences(feats)
        tr = yy < test_year
        te = yy == test_year
        if te.sum() == 0 or tr.sum() < 500:
            return None
        sc = StandardScaler().fit(X[tr].reshape(-1, X.shape[-1]))
        Xn = ((X.reshape(-1, X.shape[-1]) - sc.mean_) / np.sqrt(sc.var_ + 1e-9)
              ).reshape(X.shape).astype(np.float32)
        y = yr if task == "reg" else yd
        Xtr = torch.tensor(Xn[tr]); ytr = torch.tensor(y[tr].astype(np.float32))
        Xte = torch.tensor(Xn[te]); yte = y[te]
        model = TransformerNet(X.shape[-1]).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
        lossf = nn.MSELoss() if task == "reg" else nn.BCEWithLogitsLoss()
        bs = 512
        idx = np.arange(len(Xtr))
        for epoch in range(15):
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
        if task == "reg":
            m = {"task": "return", "fold": int(test_year), "model": "Transformer",
                 "features": fs_name, "R2": float(r2_score(yte, pr)),
                 "RMSE": float(mean_squared_error(yte, pr) ** .5),
                 "DirAcc": float((np.sign(pr) == np.sign(yte)).mean())}
        else:
            phat = (1 / (1 + np.exp(-pr)))
            pcls = (phat > 0.5).astype(int)
            m = {"task": "direction", "fold": int(test_year), "model": "Transformer",
                 "features": fs_name, "Acc": float(accuracy_score(yte, pcls)),
                 "F1": float(f1_score(yte, pcls, zero_division=0)),
                 "MCC": float(matthews_corrcoef(yte, pcls))}
        # save the artifact bundle every fold's model class needs for downstream
        # XAI (IG, attention-as-explanation, TCAV, SAE, ACDC all read this one)
        if test_year == 2023 and fs_name == "price+sentiment" and task == "clf":
            np.savez(os.path.join(ART, "transformer_ig_bundle.npz"),
                     X_test=Xn[te][:512], feat_names=np.array(feats))
            torch.save(model.state_dict(), os.path.join(ART, "transformer_dir_2023.pt"))
        return m

    rows = []
    for ty in TEST_YEARS:
        for fs_name, feats in [("price_only", PRICE_FEATS),
                               ("price+sentiment", PRICE_FEATS + SENT_FEATS)]:
            for task in ["reg", "clf"]:
                r = train_eval(feats, fs_name, ty, task)
                if r:
                    rows.append(r); print(r)
    pd.DataFrame(rows).to_csv(os.path.join(MET, "transformer_results.csv"), index=False)
    print("saved transformer_results.csv")
    data_vol.commit()
