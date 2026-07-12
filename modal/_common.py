"""Shared Modal config for the FinSentImpact rebuild.

One persistent Volume holds raw FNSPID data, the built panel, metrics and figures.
Every stage imports from here so the image and data universe stay consistent.
"""
import os as _os
import modal

APP_PREFIX = "finsent"
_REPO_ROOT = _os.path.join(_os.path.dirname(__file__), "..")

# ---- Compute image -------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "openjdk-17-jre-headless")   # JRE for PySpark
    .pip_install(
        "pyspark==3.5.1",                             # distributed CSV ETL
        "pandas==2.2.2",
        "numpy==1.26.4",
        "scikit-learn==1.5.2",
        "xgboost==2.1.1",
        "lightgbm==4.5.0",
        "shap==0.46.0",
        "lime==0.2.0.1",
        "PyALE==1.2.0",
        "statsmodels==0.14.2",
        "transformers==4.44.2",
        "torch==2.4.1",
        "captum==0.7.0",
        "huggingface_hub==0.25.1",
        "hf_transfer==0.1.8",
        "matplotlib==3.9.2",
        "seaborn==0.13.2",
        "tqdm==4.66.5",
        "pyarrow==17.0.0",
        "pandas_market_calendars==4.4.1",
        "pysentiment2==0.1.1",   # bundled Loughran-McDonald financial lexicon
        "interpret==0.6.1",      # EBM / GA2M (glass-box model + intrinsic explanations)
        "yfinance==0.2.44",      # full-profile price source (60-stock universe, 2015-2024)
        "catboost==1.2.7",       # ordered-boosting tree ensemble (model zoo #6)
        "torch_geometric==2.6.1",  # GNN (cross-stock graph) + GNNExplainer
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "JAVA_HOME": "/usr/lib/jvm/java-17-openjdk-amd64",
    })
    .add_local_python_source("_common")   # make this module importable remotely
    .add_local_file(                      # config-as-code must be readable in the container too
        _os.path.join(_REPO_ROOT, "config.xai.yaml"), remote_path="/config.xai.yaml",
    )
)

# ---- Persistent storage --------------------------------------------------
data_vol = modal.Volume.from_name("finsent-data", create_if_missing=True)
DATA = "/data"

# ---- Study design (config-as-code) ---------------------------------------
# Single source of truth: ../config.xai.yaml. `profile` selects `reviewer`
# (small/fast, proves the pipeline) or `full` (the paper: 15 models + GNN,
# ~26 XAI methods, 60 stocks). Shared keys apply to both; profile-specific
# keys in profiles.<profile> override/extend them. See config.xai.yaml for
# the full spec (ground-truth synthetic features, model zoo, XAI methods).
import yaml as _yaml

# locally this resolves under the repo root; remotely _common.py lives at
# /root/_common.py and the mount above places the file at /config.xai.yaml,
# which "/root/../config.xai.yaml" resolves to as well -- same path shape.
_CFG_PATH = _os.path.join(_os.path.dirname(__file__), "..", "config.xai.yaml")
with open(_CFG_PATH) as _f:
    _RAW = _yaml.safe_load(_f)

PROFILE = _RAW["profile"]
_PROF = _RAW["profiles"][PROFILE]


def _cfg(key, default=None):
    """Profile-specific value if present, else shared, else default."""
    if key in _PROF:
        return _PROF[key]
    if key in _RAW:
        return _RAW[key]
    return default


# shared across profiles
SEED = _RAW["seed"]
FINBERT_MODEL = _RAW["finbert_model"]
RETURN_CLIP = _RAW["return_clip"]
EMBARGO_DAYS = _RAW["embargo_days"]
LOOKBACK = _RAW["lookback"]
HORIZONS = _RAW["horizons"]
VOL_ESTIMATOR = _RAW["vol_estimator"]
GROUND_TRUTH = _RAW["ground_truth"]           # noise/signal/redundant-copy spec (P10)
CLUSTER_UNIT = _RAW["cluster_unit"]           # "symbol" -- never cluster by stock-day
MDES_REPORT = _RAW["mdes_report"]
MIN_NEWS_DAYS = _cfg("min_news_days", 150)

# profile-specific
TICKERS = _PROF.get("tickers")                # explicit list (reviewer) or None (full -> yfinance universe)
START_DATE = _PROF["start_date"]
END_DATE = _PROF["end_date"]
TEST_YEARS = _PROF["test_years"]
PRICE_SOURCE = _PROF["price_source"]           # "fnspid" | "yfinance"
UNIVERSE_SIZE = _PROF.get("universe_size")
MODELS = _PROF["models"]
XAI_METHODS = _PROF["xai_methods"]
N_BOOTSTRAP = _PROF["n_bootstrap"]
FINBERT_BATCH = _PROF["finbert_batch"]
