"""Stage 1 — download FNSPID to the Volume and inspect its structure.

Run:
    modal run s1_data.py::download      # ~6 GB, idempotent
    modal run s1_data.py::inspect
    modal run s1_data.py::inspect_universe   # full-profile 60-stock candidate scan
"""
import modal
from _common import image, data_vol, DATA, TICKERS, START_DATE, END_DATE, PRICE_SOURCE

app = modal.App(f"finsent-s1-data", image=image)

RAW = f"{DATA}/raw"

# ---------------------------------------------------------------------------
# Candidate universe for the `full` profile (brief P5/S11: 40-100 diverse US
# equities). 81 large/mid-cap, GICS-sector-diverse tickers, all with trading
# histories well before 2015 (so yfinance covers 2015-2024 fully); the final
# 60 are chosen from these by ACTUAL news-day coverage in the 23 GB
# nasdaq_exteral_data.csv (selection by data availability, not outcome --
# same principle as the reviewer profile's 12/6-ticker selection).
FULL_CANDIDATES = {
    "technology": ["AAPL", "MSFT", "NVDA", "INTC", "CSCO", "ORCL", "IBM", "ADBE",
                   "CRM", "TXN", "QCOM", "AMD"],
    "communication_services": ["GOOGL", "META", "DIS", "CMCSA", "VZ", "T"],
    "consumer_discretionary": ["AMZN", "HD", "MCD", "NKE", "SBUX", "TGT", "LOW", "BKNG"],
    "consumer_staples": ["WMT", "PG", "KO", "PEP", "COST", "CL", "MO"],
    "financials": ["JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "BLK", "SCHW", "USB"],
    "healthcare": ["JNJ", "PFE", "UNH", "MRK", "ABT", "TMO", "ABBV", "LLY", "BMY"],
    "industrials": ["BA", "CAT", "GE", "HON", "UPS", "MMM", "LMT", "DE"],
    "energy": ["XOM", "CVX", "COP", "SLB", "EOG", "PSX"],
    "materials": ["LIN", "APD", "ECL", "NEM", "DOW"],
    "utilities": ["NEE", "DUK", "SO", "D", "AEP"],
    "real_estate": ["AMT", "PLD", "CCI", "SPG", "EQIX"],
}


@app.function(volumes={DATA: data_vol}, timeout=60 * 60)
def download():
    import os
    from huggingface_hub import hf_hub_download

    os.makedirs(RAW, exist_ok=True)
    files = [
        "Stock_price/full_history.zip",   # 589 MB — per-ticker OHLCV
        "Stock_news/All_external.csv",     # 5.7 GB — news headlines + articles
    ]
    for f in files:
        target = os.path.join(RAW, os.path.basename(f))
        if os.path.exists(target) and os.path.getsize(target) > 1_000_000:
            print(f"[skip] {f} already present ({os.path.getsize(target)/1e6:.0f} MB)")
            continue
        print(f"[get ] {f} ...")
        p = hf_hub_download(
            repo_id="Zihan1004/FNSPID", filename=f, repo_type="dataset",
            local_dir=RAW,
        )
        # hf places it under RAW/<subdir>/..; move basename to RAW root
        if p != target:
            os.replace(p, target)
        print(f"[done] {target} ({os.path.getsize(target)/1e6:.0f} MB)")
    data_vol.commit()
    print("download complete")


@app.function(volumes={DATA: data_vol}, timeout=60 * 60)
def download_nasdaq():
    """The comprehensive 23 GB news file (spans 2018-2023, unlike All_external
    which stops in 2020). This is the file the original paper used."""
    import os
    from huggingface_hub import hf_hub_download
    os.makedirs(RAW, exist_ok=True)
    target = os.path.join(RAW, "nasdaq_exteral_data.csv")
    if os.path.exists(target) and os.path.getsize(target) > 1_000_000_000:
        print(f"[skip] already present ({os.path.getsize(target)/1e9:.1f} GB)")
        return
    p = hf_hub_download(repo_id="Zihan1004/FNSPID",
                        filename="Stock_news/nasdaq_exteral_data.csv",
                        repo_type="dataset", local_dir=RAW)
    if p != target:
        os.replace(p, target)
    print(f"[done] {target} ({os.path.getsize(target)/1e9:.1f} GB)")
    data_vol.commit()


@app.function(volumes={DATA: data_vol}, timeout=60 * 30)
def download_yfinance():
    """Cache adjusted OHLCV from yfinance for the `full` profile's universe
    (brief S11/S15.5: news still comes from FNSPID's nasdaq_exteral_data.csv,
    but the 2015-2024 price history needs a wider/longer source than FNSPID's
    price zip). Cached to the Volume + a data manifest so the panel is stable
    even if yfinance changes upstream. Idempotent; only runs when
    price_source: yfinance (the active profile in config.xai.yaml)."""
    import os, json, time
    import pandas as pd
    import yfinance as yf

    if PRICE_SOURCE != "yfinance":
        print(f"[skip] active profile's price_source={PRICE_SOURCE!r}, not 'yfinance'; nothing to do")
        return

    os.makedirs(RAW, exist_ok=True)
    target = os.path.join(RAW, "yfinance_prices.parquet")
    manifest_path = os.path.join(RAW, "yfinance_manifest.json")
    if os.path.exists(target):
        print(f"[skip] {target} already present")
        return

    frames, coverage, failed = [], {}, []
    for t in TICKERS:
        try:
            df = yf.download(t, start=START_DATE, end=END_DATE, auto_adjust=False,
                             progress=False, threads=False)
            if df.empty:
                print(f"  [warn] {t}: no data returned"); failed.append(t); continue
            df = df.reset_index()
            df.columns = [c if isinstance(c, str) else c[0] for c in df.columns]
            df = df.rename(columns={"Adj Close": "Adj_close"})
            if "Adj_close" not in df.columns:
                df["Adj_close"] = df["Close"]
            df["Symbol"] = t
            keep = ["Date", "Symbol", "Open", "High", "Low", "Close", "Adj_close", "Volume"]
            frames.append(df[keep])
            coverage[t] = {"rows": int(len(df)), "start": str(df["Date"].min().date()),
                           "end": str(df["Date"].max().date())}
            print(f"  {t:6s} {len(df):5d} rows  {df['Date'].min().date()} .. {df['Date'].max().date()}")
        except Exception as e:
            print(f"  [warn] {t} failed: {e}"); failed.append(t)
        time.sleep(0.3)   # be polite to the API

    prices = pd.concat(frames, ignore_index=True)
    prices.to_parquet(target, index=False)
    manifest = {
        "source": "yfinance", "yfinance_version": yf.__version__,
        "n_requested": len(TICKERS), "n_downloaded": len(coverage), "failed_tickers": failed,
        "start_date": START_DATE, "end_date": END_DATE, "per_ticker": coverage,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"saved {target} ({len(prices):,} rows, {len(coverage)}/{len(TICKERS)} tickers)")
    if failed:
        print(f"[warn] failed tickers (dropped from universe): {failed}")
    data_vol.commit()


@app.function(volumes={DATA: data_vol}, timeout=60 * 30)
def inspect():
    import os, zipfile
    import pandas as pd

    print("=" * 70)
    print("RAW contents:")
    for f in sorted(os.listdir(RAW)):
        print(f"  {os.path.getsize(os.path.join(RAW,f))/1e6:10.1f} MB  {f}")

    # --- price zip ---
    print("=" * 70)
    zpath = os.path.join(RAW, "full_history.zip")
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
        print(f"full_history.zip: {len(names)} entries; sample: {names[:5]}")
        # find our tickers
        present = [t for t in TICKERS if any(f"{t}.csv" in n for n in names)]
        print(f"our tickers present in price zip: {present}")
        # peek one
        cand = [n for n in names if n.endswith("AAPL.csv")]
        if cand:
            with z.open(cand[0]) as fh:
                df = pd.read_csv(fh)
            print(f"AAPL price columns: {list(df.columns)}")
            print(f"AAPL rows: {len(df)}; date span: {df.iloc[0,0]} .. {df.iloc[-1,0]}")
            print(df.head(3).to_string())

    # --- news csv header + small sample ---
    print("=" * 70)
    npath = os.path.join(RAW, "All_external.csv")
    head = pd.read_csv(npath, nrows=5)
    print(f"All_external.csv columns: {list(head.columns)}")
    print(head.to_string()[:2000])

    # count rows for our tickers (chunked scan, cheap columns only)
    print("=" * 70)
    print("scanning news for ticker coverage (this reads the file once)...")
    cols = list(head.columns)
    sym_col = next((c for c in cols if c.lower() in ("stock_symbol", "symbol", "ticker")), None)
    date_col = next((c for c in cols if "date" in c.lower()), None)
    print(f"detected symbol column = {sym_col!r}, date column = {date_col!r}")
    if sym_col:
        from collections import Counter
        counts = Counter()
        tset = set(TICKERS)
        n = 0
        for chunk in pd.read_csv(npath, usecols=[sym_col], chunksize=1_000_000):
            n += len(chunk)
            vc = chunk[chunk[sym_col].isin(tset)][sym_col].value_counts()
            for k, v in vc.items():
                counts[k] += int(v)
        print(f"total news rows: {n:,}")
        print("news articles per ticker (our universe), {}..{}:".format(START_DATE, END_DATE))
        for t in TICKERS:
            print(f"  {t:6s} {counts.get(t,0):>10,}")
        print(f"TOTAL our-universe articles (all dates): {sum(counts.values()):,}")


@app.function(volumes={DATA: data_vol}, cpu=16.0, memory=65536, timeout=60 * 40)
def inspect_universe():
    """Score FULL_CANDIDATES by real news-DAY coverage in the 23 GB
    nasdaq_exteral_data.csv (the file s2_build.py::build_news actually uses --
    unlike inspect() above, which checks the smaller/older All_external.csv).
    Prints a per-sector, per-ticker table and a proposed final-60 selection
    (top tickers by news-day coverage, capped per sector for diversity) so the
    result can be reviewed before being hardcoded into config.xai.yaml."""
    import os, json
    import pandas as pd
    from collections import Counter

    npath = os.path.join(RAW, "nasdaq_exteral_data.csv")
    all_candidates = [t for syms in FULL_CANDIDATES.values() for t in syms]
    sector_of = {t: s for s, syms in FULL_CANDIDATES.items() for t in syms}
    print(f"scanning {npath} for {len(all_candidates)} candidate tickers "
          f"across {len(FULL_CANDIDATES)} sectors...")

    head = pd.read_csv(npath, nrows=5)
    cols = list(head.columns)
    sym_col = next(c for c in cols if c.lower() in ("stock_symbol", "symbol", "ticker"))
    date_col = next(c for c in cols if "date" in c.lower())
    print(f"symbol column = {sym_col!r}, date column = {date_col!r}")

    tset = set(all_candidates)
    day_sets = {t: set() for t in all_candidates}   # distinct calendar days with >=1 article
    article_counts = Counter()
    n_scanned = 0
    for chunk in pd.read_csv(npath, usecols=[sym_col, date_col], chunksize=2_000_000):
        n_scanned += len(chunk)
        sub = chunk[chunk[sym_col].isin(tset)]
        article_counts.update(sub[sym_col].value_counts().to_dict())
        d = pd.to_datetime(sub[date_col], errors="coerce", utc=True).dt.date
        for t, dt in zip(sub[sym_col].values, d.values):
            if pd.notna(dt):
                day_sets[t].add(dt)
    print(f"scanned {n_scanned:,} total news rows")

    rows = []
    for t in all_candidates:
        rows.append({"ticker": t, "sector": sector_of[t],
                     "news_days": len(day_sets[t]), "articles": int(article_counts.get(t, 0))})
    report = pd.DataFrame(rows).sort_values("news_days", ascending=False).reset_index(drop=True)
    print(report.to_string(index=False))

    # proposed final-60: FIRST enforce the same MIN_NEWS_DAYS floor assemble()
    # applies downstream (else a chosen ticker would just get silently dropped
    # later, shrinking the universe below the target); then within qualifying
    # candidates, take a base share per sector (diversity), then top up from
    # the overall coverage ranking to reach TARGET.
    TARGET = 60
    MIN_NEWS_DAYS = 150
    disqualified = report[report.news_days < MIN_NEWS_DAYS]
    qualifying = report[report.news_days >= MIN_NEWS_DAYS].copy()
    print(f"\ndisqualified (<{MIN_NEWS_DAYS} news-days): "
          f"{list(zip(disqualified.ticker, disqualified.news_days))}")

    n_sectors = qualifying["sector"].nunique()
    base_per_sector = 3
    chosen = []
    for s in sorted(qualifying["sector"].unique()):
        sec_ranked = qualifying[qualifying.sector == s].sort_values("news_days", ascending=False)
        chosen += sec_ranked.head(base_per_sector)["ticker"].tolist()
    remaining = TARGET - len(chosen)
    if remaining > 0:
        leftover = qualifying[~qualifying.ticker.isin(chosen)].sort_values("news_days", ascending=False)
        chosen += leftover.head(remaining)["ticker"].tolist()
    chosen = chosen[:TARGET]
    min_days = int(report[report.ticker.isin(chosen)]["news_days"].min())
    print(f"\nFINAL {len(chosen)}-TICKER UNIVERSE (>= {MIN_NEWS_DAYS}-day floor, "
          f"{base_per_sector}/sector base + top-up by overall coverage rank):")
    print(json.dumps(chosen, indent=2))
    print(f"minimum news-day coverage among chosen: {min_days}")

    manifest = {
        "target": TARGET, "min_news_days_floor": MIN_NEWS_DAYS,
        "base_per_sector": base_per_sector, "n_candidates": len(all_candidates),
        "n_qualifying": int(len(qualifying)), "n_disqualified": int(len(disqualified)),
        "disqualified": disqualified[["ticker", "sector", "news_days"]].to_dict("records"),
        "chosen": chosen, "min_news_days_among_chosen": min_days,
        "full_report": report.to_dict("records"),
    }
    os.makedirs(f"{DATA}/results/metrics", exist_ok=True)
    with open(f"{DATA}/results/metrics/universe_selection.json", "w") as f:
        json.dump(manifest, f, indent=2)
    data_vol.commit()
    print("saved universe_selection.json")
    return chosen
