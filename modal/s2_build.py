"""Stage 2 — build the leakage-free multi-stock panel from FNSPID.

  modal run s2_build.py::build_news     # filter news, align to trading days  (CPU)
  modal run s2_build.py::finbert        # FinBERT sentiment on unique headlines (GPU)
  modal run s2_build.py::assemble       # lexicon + aggregate + features + targets (CPU)

Design decisions that fix the reviewed paper's leakage:
  * Time-aware news->trading-day alignment (after-close news -> next session).
  * All predictive features use ONLY information available by the close of day t.
  * Targets (t+1 return / direction / level) are produced by groupby-shift within
    each ticker; splits happen later and are strictly chronological.
"""
import modal
from _common import (
    image, data_vol, DATA, TICKERS, START_DATE, END_DATE, SEED, FINBERT_MODEL,
    MIN_NEWS_DAYS, RETURN_CLIP, GROUND_TRUTH, PRICE_SOURCE,
)

app = modal.App("finsent-s2-build", image=image)

RAW = f"{DATA}/raw"
BUILD = f"{DATA}/build"


# --------------------------------------------------------------------------
@app.function(volumes={DATA: data_vol}, cpu=32.0, memory=131072, timeout=60 * 90)
def build_news():
    """Distributed ingestion of the multi-GB news CSV with PySpark.

    Spark streams the 5.7 GB file with bounded memory (spilling to disk),
    filters to our ticker universe + date window, and writes a compact
    parquet. The fiddly ET-timezone / trading-session alignment then runs in
    pandas on the already-small filtered result.
    """
    import os, pandas as pd, numpy as np
    import pandas_market_calendars as mcal
    from pyspark.sql import SparkSession, functions as F

    os.makedirs(BUILD, exist_ok=True)
    os.makedirs("/tmp/spark", exist_ok=True)
    # The comprehensive 23 GB file spans 2018-2023 (All_external stops in 2020).
    npath = os.path.join(RAW, "nasdaq_exteral_data.csv")

    spark = (
        SparkSession.builder.appName("fnspid-ingest").master("local[*]")
        .config("spark.driver.memory", "90g")
        .config("spark.local.dir", "/tmp/spark")
        .config("spark.sql.shuffle.partitions", "64")
        .config("spark.driver.maxResultSize", "4g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    # multiLine=False lets Spark split the file across all cores (parallel read).
    # Our needed columns (Date, Article_title, Stock_symbol) precede the
    # newline-heavy Article column, so they parse correctly from each record's
    # primary line; trailing fragments from embedded newlines fail the ticker
    # filter and are dropped (never misattributed). ~10x faster on 23 GB.
    sdf = (
        spark.read.option("header", True).option("multiLine", False)
        .option("escape", '"').option("quote", '"')
        .option("mode", "PERMISSIVE").csv(npath)
    )
    cols = sdf.columns
    print("news columns:", cols)
    sym_col = next(c for c in cols if c.lower() in ("stock_symbol", "symbol", "ticker"))
    date_col = next(c for c in cols if "date" in c.lower())
    text_col = (next((c for c in cols if c.lower() in
                ("article_title", "title", "headline")), None) or cols[1])
    print(f"using symbol={sym_col!r} date={date_col!r} text={text_col!r}")

    filt = (
        sdf.select(
            F.col(date_col).alias("ts_raw"),
            F.col(sym_col).alias("Symbol"),
            F.col(text_col).alias("headline"),
        )
        .where(F.col("Symbol").isin(TICKERS))
        .where(F.col("headline").isNotNull())
        .where(F.length(F.trim(F.col("headline"))) >= 8)
    )
    total = sdf.count()
    kept = filt.count()
    print(f"Spark scanned {total:,} rows; kept {kept:,} for our {len(TICKERS)} tickers")

    per = (filt.groupBy("Symbol").count().orderBy(F.desc("count"))).toPandas()
    print("Spark per-ticker article counts:\n", per.to_string(index=False))

    news = filt.toPandas()
    spark.stop()
    print(f"collected {len(news):,} filtered rows to pandas for alignment")
    news["ts"] = pd.to_datetime(news["ts_raw"], errors="coerce", utc=True)
    news = news.dropna(subset=["ts", "headline"])
    news["headline"] = news["headline"].str.strip()
    news = news[news["headline"].str.len() >= 8]
    # ET local time for the 16:00 market-close cutoff
    ts_et = news["ts"].dt.tz_convert("America/New_York")
    news["cal_date"] = ts_et.dt.normalize().dt.tz_localize(None)
    news["after_close"] = (ts_et.dt.hour >= 16).astype(int)
    news = news[(news["cal_date"] >= START_DATE) & (news["cal_date"] <= END_DATE)]
    print(f"after date filter {START_DATE}..{END_DATE}: {len(news):,} rows")

    # trading sessions
    nyse = mcal.get_calendar("NYSE")
    sessions = pd.DatetimeIndex(
        nyse.valid_days(start_date="2017-06-01", end_date="2024-06-30")
    ).tz_localize(None).normalize()
    sess_arr = sessions.values

    def to_session(cal_date, after_close):
        # after-close news -> next session; else -> first session >= cal_date
        idx = np.searchsorted(sess_arr, np.datetime64(cal_date), side="left")
        if after_close:
            # move to strictly-next session at/after this date
            while idx < len(sess_arr) and sess_arr[idx] <= np.datetime64(cal_date):
                idx += 1
        if idx >= len(sess_arr):
            return pd.NaT
        return pd.Timestamp(sess_arr[idx])

    uniq = news[["cal_date", "after_close"]].drop_duplicates()
    uniq["trading_day"] = [to_session(d, a) for d, a in
                           zip(uniq["cal_date"], uniq["after_close"])]
    news = news.merge(uniq, on=["cal_date", "after_close"], how="left")
    news = news.dropna(subset=["trading_day"])
    news = news[["trading_day", "Symbol", "headline"]]

    news.to_parquet(os.path.join(BUILD, "news_aligned.parquet"), index=False)
    uheads = pd.DataFrame({"headline": sorted(news["headline"].unique())})
    uheads.to_parquet(os.path.join(BUILD, "unique_headlines.parquet"), index=False)
    print(f"saved news_aligned ({len(news):,}) and unique_headlines ({len(uheads):,})")
    per = news.groupby("Symbol").size().sort_values(ascending=False)
    print("aligned articles per ticker:\n", per.to_string())
    data_vol.commit()


# --------------------------------------------------------------------------
@app.function(volumes={DATA: data_vol}, gpu="a100", timeout=60 * 90)
def finbert():
    import os, time, pandas as pd, numpy as np, torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    uh = pd.read_parquet(os.path.join(BUILD, "unique_headlines.parquet"))
    heads = uh["headline"].tolist()
    print(f"scoring {len(heads):,} unique headlines with {FINBERT_MODEL}")

    tok = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    if device == "cuda":
        model.half()                       # fp16 for ~2x throughput on A100
    print("device:", device, "| labels:", model.config.id2label)
    # ProsusAI/finbert id2label: 0 positive, 1 negative, 2 neutral
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    pos_i = next(i for i, l in id2label.items() if "pos" in l)
    neg_i = next(i for i, l in id2label.items() if "neg" in l)
    neu_i = next(i for i, l in id2label.items() if "neu" in l)

    # sort by length for efficient padding, then restore order
    order = np.argsort([len(h) for h in heads])
    heads_sorted = [heads[i] for i in order]
    BS = 256
    probs_sorted = np.zeros((len(heads), 3), dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, len(heads_sorted), BS):
            batch = heads_sorted[s:s + BS]
            enc = tok(batch, return_tensors="pt", truncation=True,
                      padding=True, max_length=64).to(device)
            logits = model(**enc).logits
            p = torch.softmax(logits.float(), dim=-1).cpu().numpy()
            probs_sorted[s:s + BS] = p[:, [pos_i, neg_i, neu_i]]
            if (s // BS) % 200 == 0:
                el = time.time() - t0
                print(f"  {s:>8,}/{len(heads):,}  {el:6.0f}s", flush=True)
    probs = np.zeros((len(heads), 3), dtype=np.float32)
    probs[order] = probs_sorted            # restore original headline order
    out = pd.DataFrame({
        "headline": heads,
        "fb_pos": probs[:, 0], "fb_neg": probs[:, 1], "fb_neu": probs[:, 2],
    })
    out["fb_score"] = out["fb_pos"] - out["fb_neg"]          # signed sentiment in [-1,1]
    out.to_parquet(os.path.join(BUILD, "headline_finbert.parquet"), index=False)
    print(f"saved headline_finbert ({len(out):,}) in {time.time()-t0:.0f}s")
    data_vol.commit()


# --------------------------------------------------------------------------
@app.function(volumes={DATA: data_vol}, cpu=8.0, memory=32768, timeout=60 * 60)
def assemble():
    import os, zipfile, json, pandas as pd, numpy as np
    import pandas_market_calendars as mcal
    import pysentiment2 as ps

    os.makedirs(BUILD, exist_ok=True)

    # ---- 1. prices: FNSPID zip (reviewer profile) or cached yfinance (full profile) ----
    keep_cols = ["Date", "Symbol", "Open", "High", "Low", "Close", "Adj_close", "Volume"]
    if PRICE_SOURCE == "yfinance":
        ypath = os.path.join(RAW, "yfinance_prices.parquet")
        prices = pd.read_parquet(ypath)
        prices["Date"] = pd.to_datetime(prices["Date"], errors="coerce").dt.normalize()
        prices = prices.dropna(subset=["Date"])
        prices = prices[prices["Symbol"].isin(TICKERS)][keep_cols]
        print(f"loaded {ypath}: {len(prices):,} rows, {prices['Symbol'].nunique()} tickers")
    else:
        zpath = os.path.join(RAW, "full_history.zip")
        frames = []
        with zipfile.ZipFile(zpath) as z:
            names = z.namelist()
            for t in TICKERS:
                # EXACT basename match: 'HD' must not match 'SCHD.csv'/'THD.csv'
                cand = [n for n in names if os.path.basename(n) == f"{t}.csv"]
                if not cand:
                    print(f"  [warn] no price file for {t}")
                    continue
                with z.open(cand[0]) as fh:
                    d = pd.read_csv(fh)
                d.columns = [c.strip().lower() for c in d.columns]
                ren = {"date": "Date", "open": "Open", "high": "High", "low": "Low",
                       "close": "Close", "adj close": "Adj_close",
                       "adj_close": "Adj_close", "volume": "Volume"}
                d = d.rename(columns=ren)
                if "Adj_close" not in d:
                    d["Adj_close"] = d["Close"]
                d["Date"] = pd.to_datetime(d["Date"], errors="coerce").dt.normalize()
                d = d.dropna(subset=["Date"]).sort_values("Date")
                d["Symbol"] = t
                frames.append(d[keep_cols])
        prices = pd.concat(frames, ignore_index=True)
    prices = prices[(prices["Date"] >= START_DATE) & (prices["Date"] <= END_DATE)]
    print("per-ticker RAW price coverage:")
    for t, g in prices.groupby("Symbol"):
        print(f"  {t:6s} {len(g):5d} rows  {g.Date.min().date()} .. {g.Date.max().date()}")
    # restrict to real NYSE sessions
    nyse = mcal.get_calendar("NYSE")
    sess = pd.DatetimeIndex(nyse.valid_days(START_DATE, END_DATE)).tz_localize(None).normalize()
    prices = prices[prices["Date"].isin(set(sess))]
    print(f"prices: {len(prices):,} rows, {prices['Symbol'].nunique()} tickers")

    # ---- 2. sentiment per unique headline: FinBERT + Loughran-McDonald ----
    fb = pd.read_parquet(os.path.join(BUILD, "headline_finbert.parquet"))
    lm = ps.LM()
    def lm_score(text):
        toks = lm.tokenize(str(text))
        s = lm.get_score(toks)
        return s["Polarity"]              # (pos-neg)/(pos+neg+eps) in [-1,1]
    fb["lm_score"] = fb["headline"].map(lm_score)
    print("headline sentiment ready:", fb.shape,
          "| mean fb_score", round(fb.fb_score.mean(), 4),
          "| mean lm_score", round(fb.lm_score.mean(), 4))

    # ---- 3. attach sentiment to aligned news, aggregate per (Symbol, day) ----
    news = pd.read_parquet(os.path.join(BUILD, "news_aligned.parquet"))
    news = news.merge(fb, on="headline", how="left")
    agg = news.groupby(["Symbol", "trading_day"]).agg(
        fb_score=("fb_score", "mean"),
        fb_pos=("fb_pos", "mean"),
        fb_neg=("fb_neg", "mean"),
        lm_score=("lm_score", "mean"),
        news_count=("headline", "size"),
    ).reset_index().rename(columns={"trading_day": "Date"})
    print("daily sentiment rows:", len(agg))

    # ---- 4. merge onto price panel; neutral fill for no-news days ----
    df = prices.merge(agg, on=["Symbol", "Date"], how="left")
    df["has_news"] = df["news_count"].notna().astype(int)
    for c in ["fb_score", "fb_pos", "fb_neg", "lm_score"]:
        df[c] = df[c].fillna(0.0)
    df["news_count"] = df["news_count"].fillna(0.0)

    # ---- 5. per-ticker feature engineering (all backward-looking) + targets ----
    def _rsi(close, n=14):
        d = close.diff()
        up = d.clip(lower=0).rolling(n).mean()
        dn = (-d.clip(upper=0)).rolling(n).mean()
        rs = up / (dn + 1e-9)
        return 100 - 100 / (1 + rs)

    out = []
    for t, g in df.sort_values(["Symbol", "Date"]).groupby("Symbol"):
        g = g.copy()
        c = g["Adj_close"]                       # split/dividend-adjusted returns
        g["ret"] = np.log(c / c.shift(1))
        # FNSPID splices unadjusted/adjusted prices at ~2020-07-06, producing a
        # few impossible single-day returns (e.g. AMZN -294%). For these large
        # caps |ret|>0.5 is a data error, not a market move: mask it so the
        # contaminated rows are dropped rather than corrupting features/targets.
        n_bad = int((g["ret"].abs() > RETURN_CLIP).sum())
        if n_bad:
            print(f"    [{t}] masked {n_bad} implausible return(s) (|ret|>0.5)")
        g["ret"] = g["ret"].where(g["ret"].abs() <= RETURN_CLIP, np.nan)
        for k in (1, 2, 3, 5):
            g[f"ret_lag{k}"] = g["ret"].shift(k)
        g["ret_roll5"] = g["ret"].shift(1).rolling(5).mean()
        g["ret_roll10"] = g["ret"].shift(1).rolling(10).mean()
        g["vol_roll5"] = g["ret"].shift(1).rolling(5).std()
        g["vol_roll20"] = g["ret"].shift(1).rolling(20).std()
        g["rsi14"] = _rsi(c).shift(1)
        g["mom10"] = (c.shift(1) / c.shift(11) - 1)
        g["ma_gap5"] = (c.shift(1) / c.shift(1).rolling(5).mean() - 1)
        g["ma_gap20"] = (c.shift(1) / c.shift(1).rolling(20).mean() - 1)
        g["hl_range"] = (g["High"] - g["Low"]) / g["Close"]
        g["co_ret"] = (g["Close"] - g["Open"]) / g["Open"]
        g["logvol"] = np.log1p(g["Volume"])
        g["vol_z20"] = (g["logvol"] - g["logvol"].shift(1).rolling(20).mean()) / (
            g["logvol"].shift(1).rolling(20).std() + 1e-9)
        # sentiment (already known at close of t); add short history
        g["fb_roll3"] = g["fb_score"].shift(1).rolling(3).mean()
        g["fb_roll5"] = g["fb_score"].shift(1).rolling(5).mean()
        g["news_roll5"] = g["news_count"].shift(1).rolling(5).mean()
        g["fb_chg"] = g["fb_score"] - g["fb_score"].shift(1)
        # ----- targets (t+1) -----
        g["ret_next"] = g["ret"].shift(-1)
        g["dir_next"] = (g["ret_next"] > 0).astype(int)
        g["Close_next"] = g["Close"].shift(-1)
        for col in ["Open", "High", "Low", "Close", "Adj_close", "Volume"]:
            g[f"{col}_next"] = g[col].shift(-1)
        out.append(g)
    panel = pd.concat(out, ignore_index=True)

    # ----- P10: synthetic ground-truth features (the faithfulness spine) -----
    # These exist only to give a KNOWN answer for "did the XAI method recover
    # true importance?" (brief S8/S9/S14) -- they are not honest predictors and
    # must never appear in the main return/direction forecasting tables. Built
    # deterministically (SEED) so faithfulness scoring is reproducible.
    gt_cols = []
    gt_manifest = {}
    rng = np.random.RandomState(SEED)
    n = len(panel)

    if GROUND_TRUTH.get("noise_feature"):
        # (a) pure noise: true importance = 0. A faithful method must NOT
        # rank this above real (or the planted-signal) features.
        panel["gt_noise"] = rng.standard_normal(n)
        gt_cols.append("gt_noise")
        gt_manifest["gt_noise"] = {"true_importance": 0.0, "construction": "iid N(0,1)"}

    if GROUND_TRUTH.get("signal_feature"):
        # (b) planted signal: a controlled leak of ret_next with a KNOWN
        # target-correlation (signal_strength). Faithfulness = does the method
        # rank this above gt_noise (and above unrelated real features).
        rho = float(GROUND_TRUTH.get("signal_strength", 0.15))
        target = panel["ret_next"].values.astype(float)
        valid = ~np.isnan(target)
        t_mean, t_std = target[valid].mean(), target[valid].std()
        target_z = (target - t_mean) / (t_std + 1e-12)
        indep_noise = rng.standard_normal(n)
        panel["gt_signal"] = rho * target_z + np.sqrt(max(1.0 - rho ** 2, 0.0)) * indep_noise
        gt_cols.append("gt_signal")
        gt_manifest["gt_signal"] = {
            "true_importance": rho, "construction": "rho*zscore(ret_next) + sqrt(1-rho^2)*N(0,1)",
            "target_correlation": rho,
        }

    if GROUND_TRUTH.get("nonlinear_signal_feature"):
        # (b2) planted NONLINEAR/interaction signal: unlike gt_signal (a simple
        # additive linear leak), this encodes direction through a sign x
        # independent-magnitude INTERACTION -- rho_nl * sign(dir_next) *
        # |aux| + sqrt(1-rho_nl^2)*noise. A method that only detects simple
        # linear/additive correlation may understate this feature's true
        # importance relative to a method that captures interaction/nonlinear
        # structure (tree splits, kernel methods, flexible neural fits). This
        # directly answers the "our ground truth is a linear-only floor"
        # limitation: faithfulness is now also checked against a genuinely
        # non-additive known effect. The realized (empirical) correlation with
        # dir_next is measured and logged below rather than assumed, since the
        # sign*|aux| construction does not have a simple closed-form rho like
        # gt_signal's additive construction.
        rho_nl = float(GROUND_TRUTH.get("nonlinear_signal_strength", 0.15))
        dir_next = panel["dir_next"].values.astype(float)
        ret_next = panel["ret_next"].values.astype(float)
        valid_nl = ~np.isnan(ret_next)
        direction_sign = np.where(dir_next > 0.5, 1.0, -1.0)
        aux_magnitude = np.abs(rng.standard_normal(n))
        interaction_core = direction_sign * aux_magnitude
        core_mean, core_std = interaction_core[valid_nl].mean(), interaction_core[valid_nl].std()
        core_z = (interaction_core - core_mean) / (core_std + 1e-12)
        indep_noise2 = rng.standard_normal(n)
        panel["gt_signal_nl"] = rho_nl * core_z + np.sqrt(max(1.0 - rho_nl ** 2, 0.0)) * indep_noise2
        gt_cols.append("gt_signal_nl")
        realized_corr = float(np.corrcoef(panel["gt_signal_nl"].values[valid_nl],
                                          ret_next[valid_nl])[0, 1])
        realized_pointbiserial = float(np.corrcoef(panel["gt_signal_nl"].values[valid_nl],
                                                    dir_next[valid_nl])[0, 1])
        gt_manifest["gt_signal_nl"] = {
            "true_importance": "nonzero (interaction, not simple linear)",
            "construction": "rho_nl*zscore(sign(dir_next)*|aux_N(0,1)|) + sqrt(1-rho_nl^2)*N(0,1)",
            "nonlinear_signal_strength_param": rho_nl,
            "realized_pearson_corr_with_ret_next": realized_corr,
            "realized_pointbiserial_corr_with_dir_next": realized_pointbiserial,
        }

    if GROUND_TRUTH.get("weak_signal_feature"):
        # (b3) R3 revision (M5): a WEAK planted linear signal, same construction
        # as gt_signal but near the noise floor (default rho=0.05 vs 0.15) --
        # tests whether faithfulness methods still recover a true-but-faint
        # effect, not just an easy strong one ("floor, not ceiling").
        rho_w = float(GROUND_TRUTH.get("weak_signal_strength", 0.05))
        target_w = panel["ret_next"].values.astype(float)
        valid_w = ~np.isnan(target_w)
        tw_mean, tw_std = target_w[valid_w].mean(), target_w[valid_w].std()
        target_wz = (target_w - tw_mean) / (tw_std + 1e-12)
        indep_noise_w = rng.standard_normal(n)
        panel["gt_signal_weak"] = rho_w * target_wz + np.sqrt(max(1.0 - rho_w ** 2, 0.0)) * indep_noise_w
        gt_cols.append("gt_signal_weak")
        gt_manifest["gt_signal_weak"] = {
            "true_importance": rho_w, "construction": "rho*zscore(ret_next) + sqrt(1-rho^2)*N(0,1), weak variant",
            "target_correlation": rho_w,
        }

    if GROUND_TRUTH.get("redundant_copy"):
        # (c) exact duplicate of a real, already-informative feature: tests
        # whether a method over-credits the copy or (correctly) splits credit
        # between the two collinear columns instead of hiding/doubling it.
        redundant_source = "vol_z20"
        panel["gt_redundant_copy"] = panel[redundant_source].values
        gt_cols.append("gt_redundant_copy")
        gt_manifest["gt_redundant_copy"] = {
            "true_importance": "== " + redundant_source,
            "construction": f"exact copy of '{redundant_source}'", "source_feature": redundant_source,
        }

    if GROUND_TRUTH.get("partial_redundant_feature"):
        # (c2) R3 revision (M5): a PARTIALLY collinear feature (target
        # correlation with its source, default 0.8) rather than an exact
        # duplicate -- exact-copy credit-splitting (0/50/100 patterns) may not
        # generalize to the more realistic case of near-but-not-identical
        # features, which is the common real case (e.g. two related technical
        # indicators), not the edge case.
        partial_source = "vol_z20"
        rho_p = float(GROUND_TRUTH.get("partial_redundant_corr", 0.8))
        src = panel[partial_source].values.astype(float)
        valid_p = ~np.isnan(src)
        p_mean, p_std = src[valid_p].mean(), src[valid_p].std()
        src_z = (src - p_mean) / (p_std + 1e-12)
        indep_noise_p = rng.standard_normal(n)
        panel["gt_redundant_partial"] = rho_p * src_z + np.sqrt(max(1.0 - rho_p ** 2, 0.0)) * indep_noise_p
        gt_cols.append("gt_redundant_partial")
        realized_corr_p = float(np.corrcoef(panel["gt_redundant_partial"].values[valid_p],
                                             src[valid_p])[0, 1])
        gt_manifest["gt_redundant_partial"] = {
            "true_importance": f"partially collinear with {partial_source} (target rho={rho_p})",
            "construction": f"rho*zscore({partial_source}) + sqrt(1-rho^2)*N(0,1)",
            "source_feature": partial_source, "target_corr": rho_p,
            "realized_corr_with_source": realized_corr_p,
        }
    print(f"ground-truth features added: {gt_cols}")

    price_feats = ["Open", "High", "Low", "Close", "Adj_close", "Volume", "logvol",
                   "ret_lag1", "ret_lag2", "ret_lag3", "ret_lag5", "ret_roll5",
                   "ret_roll10", "vol_roll5", "vol_roll20", "rsi14", "mom10",
                   "ma_gap5", "ma_gap20", "hl_range", "co_ret", "vol_z20"]
    sent_feats = ["fb_score", "fb_pos", "fb_neg", "lm_score", "news_count",
                  "has_news", "fb_roll3", "fb_roll5", "news_roll5", "fb_chg"]

    panel = panel.dropna(subset=price_feats + ["ret_next"]).reset_index(drop=True)

    # keep only tickers with adequate news coverage so the sentiment ablation
    # is a fair test (documented selection: by data availability, not outcome)
    cov = panel.groupby("Symbol")["has_news"].sum()
    keep = cov[cov >= MIN_NEWS_DAYS].index.tolist()
    dropped = sorted(set(panel["Symbol"]) - set(keep))
    print(f"news-day coverage per ticker:\n{cov.sort_values(ascending=False).to_string()}")
    print(f"KEEP {len(keep)} tickers (>=150 news-days); DROP {dropped}")
    panel = panel[panel["Symbol"].isin(keep)].reset_index(drop=True)
    panel = panel.sort_values(["Date", "Symbol"]).reset_index(drop=True)

    panel.to_parquet(os.path.join(BUILD, "panel.parquet"), index=False)
    manifest = {
        "price_features": price_feats,
        "sentiment_features": sent_feats,
        "ground_truth_features": gt_cols,
        "ground_truth_construction": gt_manifest,
        "targets": {"return": "ret_next", "direction": "dir_next",
                    "level": "Close_next"},
        "n_rows": int(len(panel)), "n_tickers": int(panel["Symbol"].nunique()),
        "date_min": str(panel["Date"].min().date()),
        "date_max": str(panel["Date"].max().date()),
        "news_coverage": float((panel["has_news"] == 1).mean()),
    }
    with open(os.path.join(BUILD, "feature_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("PANEL:", json.dumps(manifest, indent=2))
    print(panel[["Date", "Symbol", "ret", "ret_next", "dir_next", "fb_score",
                 "news_count", "has_news"]].head(8).to_string())
    data_vol.commit()
