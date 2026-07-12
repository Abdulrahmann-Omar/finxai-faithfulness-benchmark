"""Stage 6 - professional time-series EDA of the panel.

  modal run s6_eda.py::run

Establishes the diagnostics a senior time-series data scientist would check
before modelling, and that justify every downstream design choice:
  * stationarity: ADF on price LEVELS (unit root) vs RETURNS (stationary)
  * autocorrelation: ACF of levels (~1) vs returns (~0); Ljung-Box
  * volatility clustering (ARCH): Ljung-Box on squared returns
  * fat tails: excess kurtosis / skew of returns
  * cross-sectional co-movement of returns across the 57-stock universe
  * feature -> target correlations (why next-day prediction is hard)
  * a leakage screen: no backward-looking feature should correlate strongly
    with the next-day target
  * news/sentiment coverage over time and class balance
"""
import modal
from _common import image, data_vol, DATA, SEED

app = modal.App("finsent-s6-eda", image=image)
BUILD = f"{DATA}/build"
RES = f"{DATA}/results"
FIG = f"{RES}/figures"
MET = f"{RES}/metrics"

PRICE_FEATS = ["ret_lag1", "ret_lag2", "ret_lag3", "ret_lag5",
               "vol_roll5", "vol_roll20", "rsi14", "ma_gap20",
               "hl_range", "co_ret", "logvol", "vol_z20"]
SENT_FEATS = ["fb_score", "fb_pos", "fb_neg", "lm_score", "news_count",
              "has_news", "fb_roll3", "fb_roll5", "news_roll5", "fb_chg"]


def _mpl():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 150, "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25})
    return plt


def _acf(x, nlags=20):
    import numpy as np
    x = np.asarray(x, float); x = x - x.mean()
    n = len(x); denom = np.dot(x, x)
    return np.array([1.0] + [np.dot(x[k:], x[:-k]) / denom for k in range(1, nlags + 1)])


def _ljung_box(x, lags=10):
    import numpy as np
    from scipy import stats as st
    n = len(x); ac = _acf(x, lags)[1:]
    q = n * (n + 2) * np.sum([(ac[k] ** 2) / (n - k - 1) for k in range(lags)])
    return float(q), float(1 - st.chi2.cdf(q, lags))


def _adf_rho(x):
    """AR(1) coefficient rho as a simple unit-root indicator (rho~1 => unit root).
    Tries statsmodels adfuller; falls back to the OLS AR(1) slope."""
    import numpy as np
    try:
        from statsmodels.tsa.stattools import adfuller
        r = adfuller(np.asarray(x, float), autolag="AIC")
        return {"adf_stat": float(r[0]), "adf_p": float(r[1]), "method": "adfuller"}
    except Exception:
        x = np.asarray(x, float)
        x0, x1 = x[:-1], x[1:]
        rho = float(np.polyfit(x0, x1, 1)[0])
        return {"ar1_rho": rho, "method": "ar1_ols"}


@app.function(volumes={DATA: data_vol}, cpu=8.0, memory=32768, timeout=60 * 40)
def run():
    import os, json, numpy as np, pandas as pd
    from scipy import stats as st
    import seaborn as sns
    plt = _mpl()
    os.makedirs(FIG, exist_ok=True); os.makedirs(MET, exist_ok=True)

    df = pd.read_parquet(os.path.join(BUILD, "panel.parquet")).copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Symbol", "Date"]).reset_index(drop=True)
    rep = {}

    # ---- 1. structure ----
    rep["n_rows"] = int(len(df))
    rep["n_tickers"] = int(df.Symbol.nunique())
    rep["tickers"] = sorted(df.Symbol.unique())
    rep["date_range"] = [str(df.Date.min().date()), str(df.Date.max().date())]
    rep["direction_up_rate"] = float(df.dir_next.mean())

    # ---- 2. stationarity: levels vs returns (per ticker, averaged) ----
    lev_rho, ret_rho, lev_p, ret_p = [], [], [], []
    for s, g in df.groupby("Symbol"):
        a = _adf_rho(g["Close"].values)
        b = _adf_rho(g["ret"].dropna().values)
        if "adf_p" in a:
            lev_p.append(a["adf_p"]); ret_p.append(b["adf_p"])
        lev_rho.append(a.get("ar1_rho", np.nan)); ret_rho.append(b.get("ar1_rho", np.nan))
    rep["stationarity"] = {
        "levels_adf_p_median": float(np.median(lev_p)) if lev_p else None,
        "returns_adf_p_median": float(np.median(ret_p)) if ret_p else None,
        "levels_pct_nonstationary_p>0.05": float(np.mean(np.array(lev_p) > 0.05)) if lev_p else None,
        "returns_pct_stationary_p<0.05": float(np.mean(np.array(ret_p) < 0.05)) if ret_p else None,
    }

    # ---- 3. autocorrelation + Ljung-Box (levels, returns, squared returns) ----
    ex = df[df.Symbol == df.Symbol.iloc[0]]
    acf_lev = _acf(ex["Close"].values, 20)
    acf_ret = _acf(ex["ret"].dropna().values, 20)
    lb_ret, lb_ret_p = _ljung_box(df["ret"].dropna().values, 10)
    lb_sq, lb_sq_p = _ljung_box((df["ret"].dropna().values) ** 2, 10)
    rep["autocorrelation"] = {
        "acf_close_lag1": float(acf_lev[1]), "acf_return_lag1": float(acf_ret[1]),
        "ljungbox_returns_p": lb_ret_p, "ljungbox_squared_returns_p": lb_sq_p,
        "note": "levels highly autocorrelated; returns not; squared returns are (ARCH)",
    }

    # ---- 4. fat tails ----
    r = df["ret"].dropna().values
    rep["distribution"] = {
        "return_mean": float(r.mean()), "return_std": float(r.std()),
        "excess_kurtosis": float(st.kurtosis(r)), "skew": float(st.skew(r)),
        "jarque_bera_p": float(st.jarque_bera(r)[1]),
    }

    # ---- 5. cross-sectional correlation of returns ----
    wide = df.pivot_table(index="Date", columns="Symbol", values="ret")
    cc = wide.corr()
    iu = np.triu_indices_from(cc.values, k=1)
    rep["cross_sectional_return_corr"] = {
        "mean": float(np.nanmean(cc.values[iu])),
        "min": float(np.nanmin(cc.values[iu])), "max": float(np.nanmax(cc.values[iu])),
    }

    # ---- 6. feature -> next-day target correlations (why it's hard) + leakage screen ----
    feats = PRICE_FEATS + SENT_FEATS
    corr_tgt = df[feats + ["ret_next"]].corr()["ret_next"].drop("ret_next")
    rep["feature_target_corr"] = {
        "max_abs": float(corr_tgt.abs().max()),
        "max_abs_feature": str(corr_tgt.abs().idxmax()),
        "sentiment_max_abs": float(corr_tgt[SENT_FEATS].abs().max()),
        "leakage_screen_pass": bool(corr_tgt.abs().max() < 0.15),
    }

    with open(os.path.join(MET, "eda_report.json"), "w") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps(rep, indent=2))

    # ================= FIGURES =================
    sym = "AAPL" if "AAPL" in df.Symbol.values else df.Symbol.iloc[0]
    g = df[df.Symbol == sym]

    # (a) price level (nonstationary) vs return (stationary)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(g.Date, g.Close, color="#2F4B94"); ax[0].set_title(f"{sym} price level (non-stationary, I(1))")
    ax[1].plot(g.Date, g.ret, color="#B4740F", lw=0.6); ax[1].set_title(f"{sym} daily return (stationary, I(0))")
    for a in ax: a.tick_params(axis="x", rotation=30)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_eda_stationarity.png"), bbox_inches="tight")

    # (b) ACF levels vs returns
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    ax[0].bar(range(21), acf_lev, color="#2F4B94"); ax[0].set_title("ACF of price levels"); ax[0].set_ylim(-0.2, 1.05)
    ax[1].bar(range(21), acf_ret, color="#B4740F"); ax[1].set_title("ACF of returns"); ax[1].set_ylim(-0.2, 1.05)
    for a in ax: a.axhline(0, color="k", lw=.6); a.set_xlabel("lag")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_eda_acf.png"), bbox_inches="tight")

    # (c) return distribution vs normal
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.hist(r, bins=120, density=True, color="#566B86", alpha=.7)
    xs = np.linspace(r.min(), r.max(), 200)
    ax.plot(xs, st.norm.pdf(xs, r.mean(), r.std()), "r--", label="normal")
    ax.set_title(f"Return distribution: excess kurtosis={st.kurtosis(r):.1f} (fat tails)")
    ax.set_xlim(-0.1, 0.1); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_eda_return_dist.png"), bbox_inches="tight")

    # (d) volatility clustering / regimes
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for s2, g2 in df.groupby("Symbol"):
        ax.plot(g2.Date, g2.ret.rolling(20).std(), lw=.6, alpha=.5)
    ax.set_title("20-day rolling volatility (regimes: 2020 COVID, 2022 drawdown)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_eda_volatility.png"), bbox_inches="tight")

    # (e) cross-sectional return correlation heatmap (all n_tickers symbols, no per-cell
    # annotation -- with 57 symbols the matrix has 57*56/2=1596 off-diagonal cells, too
    # dense to annotate legibly; every tick is labelled so the true universe size is visible)
    n_sym = cc.shape[0]
    fig, ax = plt.subplots(figsize=(max(10, n_sym * 0.22), max(9, n_sym * 0.22)))
    sns.heatmap(cc, cmap="RdBu_r", center=0, vmin=-1, vmax=1, square=True,
                annot=False, cbar_kws={"shrink": .7}, xticklabels=True, yticklabels=True, ax=ax)
    ax.tick_params(axis="x", labelsize=7, rotation=90)
    ax.tick_params(axis="y", labelsize=7, rotation=0)
    ax.set_title(f"Cross-sectional return correlation across {n_sym} stocks "
                 f"(mean={rep['cross_sectional_return_corr']['mean']:.3f})")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_eda_cross_corr.png"), bbox_inches="tight")

    # (f) feature -> target correlation bars
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ct = corr_tgt.reindex(corr_tgt.abs().sort_values().index)
    colors = ["#B4740F" if f in SENT_FEATS else "#2F4B94" for f in ct.index]
    ax.barh(ct.index, ct.values, color=colors)
    ax.axvline(0, color="k", lw=.8)
    ax.set_title(f"Correlation of features with next-day return\n(max |r|={corr_tgt.abs().max():.3f}: near-unpredictable)")
    ax.set_xlim(-0.15, 0.15)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_eda_feat_target.png"), bbox_inches="tight")

    # (g) news coverage over time + sentiment dist
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    cov = df.groupby(df.Date.dt.to_period("Q"))["has_news"].mean()
    ax[0].bar([str(p) for p in cov.index], cov.values, color="#2C7A55")
    ax[0].set_title("News coverage per quarter"); ax[0].tick_params(axis="x", rotation=90, labelsize=6)
    ax[1].hist(df[df.has_news == 1]["fb_score"], bins=40, color="#B4740F", alpha=.75)
    ax[1].set_title("FinBERT sentiment distribution (news days)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_eda_news.png"), bbox_inches="tight")

    print("saved EDA figures: stationarity, acf, return_dist, volatility, cross_corr, feat_target, news")
    data_vol.commit()
