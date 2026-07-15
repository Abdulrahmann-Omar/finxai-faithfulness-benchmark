"""Stage 8 -- execute the notebook suite on Modal, not locally.

  modal run s8_notebooks.py::run_all

Runs every notebook in ../notebooks/ inside its own Modal container (same base
image as the rest of the pipeline plus jupyter/nbconvert), each with the Volume
mounted so `Path('../results')` resolves to the real, committed
results/panel.parquet + results/metrics/* + results/artifacts/* -- the exact
files the paper and make_tables.py use, not a local copy that can drift stale.
Runs all notebooks concurrently (one container each, no shared-CPU contention),
pulls each executed .ipynb back, and overwrites the local file in place.
"""
import os
import modal
from _common import data_vol, DATA

app = modal.App("finsent-s8-notebooks")

_REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
NB_DIR = os.path.join(_REPO_ROOT, "notebooks")

# Built independently from _common.image (rather than chaining onto it) so
# add_local_dir can come after pip_install without Modal's "local files must
# be added last" restriction -- _common.image already ends with its own
# add_local_* calls, and Modal disallows further build steps after those
# unless every preceding add_local_* also opts into copy=True.
nb_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "openjdk-17-jre-headless")
    .pip_install(
        "pyspark==3.5.1", "pandas==2.2.2", "numpy==1.26.4", "scikit-learn==1.5.2",
        "xgboost==2.1.1", "lightgbm==4.5.0", "shap==0.46.0", "lime==0.2.0.1",
        "PyALE==1.2.0", "statsmodels==0.14.2", "transformers==4.44.2", "torch==2.4.1",
        "captum==0.7.0", "huggingface_hub==0.25.1", "hf_transfer==0.1.8",
        "matplotlib==3.9.2", "seaborn==0.13.2", "tqdm==4.66.5", "pyarrow==17.0.0",
        "pandas_market_calendars==4.4.1", "pysentiment2==0.1.1", "interpret==0.6.1",
        "yfinance==0.2.44", "catboost==1.2.7", "torch_geometric==2.6.1",
        "jupyter==1.0.0", "nbconvert==7.16.4", "nbformat==5.10.4",
        "ipykernel==6.29.5", "notebook==7.2.2",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "JAVA_HOME": "/usr/lib/jvm/java-17-openjdk-amd64",
    })
    .add_local_python_source("_common")
    .add_local_file(
        os.path.join(_REPO_ROOT, "config.xai.yaml"), remote_path="/config.xai.yaml",
    )
    .add_local_dir(NB_DIR, remote_path="/root/notebooks_src")
    # paper/tables/*.csv is read by notebook 34 for a cross-check against the
    # committed paper output; paper/ is a local-only directory, never on the
    # Volume, so it must be mounted separately from the results/ tree above.
    .add_local_dir(os.path.join(_REPO_ROOT, "paper", "tables"), remote_path="/root/paper/tables")
)

NOTEBOOK_NAMES = [
    "00_data_and_panel.ipynb", "01_advanced_eda.ipynb", "02_baselines.ipynb",
    "10_linear.ipynb", "11_decision_tree.ipynb", "12_random_forest.ipynb",
    "13_xgboost.ipynb", "14_lightgbm.ipynb",
    "15_catboost.ipynb", "16_svm.ipynb", "17_knn.ipynb", "18_ebm.ipynb",
    "19_mlp.ipynb", "20_gru.ipynb", "21_lstm.ipynb", "22_cnn1d.ipynb",
    "23_tcn.ipynb", "24_transformer.ipynb", "25_gnn.ipynb",
    "30_faithfulness_benchmark.ipynb", "31_agreement_and_modelclass.ipynb",
    "32_mechanistic_deep_dive.ipynb", "33_actionability_recourse.ipynb",
    "34_paper_figures.ipynb",
    # R4 revision: reviewer-requested null baselines and robustness checks.
    "26_tcav_null.ipynb", "27_mechanistic_null.ipynb",
    "28_faithfulness_robustness.ipynb",
]


@app.function(volumes={DATA: data_vol}, image=nb_image, cpu=4.0, memory=16384,
               timeout=40 * 60, max_containers=12)
def run_one(name: str):
    import os, shutil, subprocess, json, time

    def fail(msg):
        return {"name": name, "ok": False, "returncode": -1, "n_code_cells": 0,
                "n_with_output": 0, "n_errors": 0, "log_tail": msg, "nb_bytes": b""}

    # notebooks reference Path('../results') relative to their own directory,
    # matching the LOCAL layout the Makefile's `results:` target creates by
    # pulling two different Volume locations into one local results/ folder:
    #   modal volume get build/panel.parquet          -> results/panel.parquet
    #   modal volume get build/feature_manifest.json   -> results/feature_manifest.json
    #   modal volume get results/{figures,metrics}     -> results/{figures,metrics}
    # On the Volume itself there is no single results/panel.parquet -- it
    # only exists at build/panel.parquet. Recreate the merged local layout
    # here with symlinks rather than a single directory-level symlink.
    try:
        os.makedirs("/root/results", exist_ok=True)
        for sub in ("artifacts", "figures", "metrics"):
            link = f"/root/results/{sub}"
            if not os.path.islink(link):
                os.symlink(f"{DATA}/results/{sub}", link)
        for fname in ("panel.parquet", "feature_manifest.json"):
            link = f"/root/results/{fname}"
            if not os.path.islink(link):
                os.symlink(f"{DATA}/build/{fname}", link)

        target = "/root/results/panel.parquet"
        for _ in range(20):
            if os.path.exists(target):
                break
            time.sleep(1)
        else:
            listing = f"os.listdir({DATA}/build) = {os.listdir(f'{DATA}/build') if os.path.isdir(f'{DATA}/build') else 'MISSING'}"
            return fail(f"panel.parquet not visible after 20s wait. {listing}")
    except Exception as e:
        return fail(f"volume/symlink setup raised: {type(e).__name__}: {e}")

    work_dir = "/root/notebooks"
    os.makedirs(work_dir, exist_ok=True)
    src = os.path.join("/root/notebooks_src", name)
    dst = os.path.join(work_dir, name)
    shutil.copy(src, dst)

    try:
        proc = subprocess.run(
            ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
             "--ExecutePreprocessor.timeout=1800", name],
            cwd=work_dir, capture_output=True, text=True,
        )
    except Exception as e:
        return fail(f"nbconvert subprocess raised: {type(e).__name__}: {e}")

    log_tail = (proc.stdout[-3000:] + "\n" + proc.stderr[-3000:])
    ok = proc.returncode == 0

    with open(dst, "rb") as f:
        nb_bytes = f.read()

    # verify every code cell has real output before declaring success; `ok`
    # (the subprocess exit code) is authoritative -- a notebook that crashed
    # mid-run can still leave a prior (e.g. stale, locally-executed) version
    # of the file on disk untouched, which would otherwise look deceptively
    # complete by cell-output-count alone.
    nb = json.loads(nb_bytes)
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    n_with_output = sum(1 for c in code_cells if c.get("outputs"))
    n_errors = sum(1 for c in code_cells for o in c.get("outputs", []) if o.get("output_type") == "error")

    return {
        "name": name, "ok": ok, "returncode": proc.returncode,
        "n_code_cells": len(code_cells), "n_with_output": n_with_output,
        "n_errors": n_errors, "log_tail": log_tail, "nb_bytes": nb_bytes,
    }


@app.local_entrypoint()
def run_all(names: str = ""):
    """modal run s8_notebooks.py::run_all [--names "16_svm.ipynb,17_knn.ipynb"]
    Omit --names to run the full suite; pass a comma-separated subset to retry
    only specific notebooks (e.g. after fixing a bug found in one)."""
    import os
    targets = [n.strip() for n in names.split(",") if n.strip()] or NOTEBOOK_NAMES
    results = list(run_one.map(targets))
    print(f"\n{'='*70}\nEXECUTION SUMMARY ({len(results)} notebooks)\n{'='*70}")
    all_good = True
    for r in sorted(results, key=lambda x: x["name"]):
        status = "OK" if (r["ok"] and r["n_with_output"] == r["n_code_cells"]
                           and r["n_code_cells"] > 0 and r["n_errors"] == 0) else "FAILED"
        if status == "FAILED":
            all_good = False
        print(f"  {r['name']:40s} {status:8s} "
              f"cells_with_output={r['n_with_output']}/{r['n_code_cells']} errors={r['n_errors']}")
        if status == "FAILED":
            print(f"    --- log tail ---\n{r['log_tail']}\n")
        # only overwrite the local file if we actually got bytes back (a
        # pre-execution setup failure returns nb_bytes=b"" -- leave the local
        # file untouched rather than clobbering it with nothing)
        if r["nb_bytes"]:
            out_path = os.path.join(NB_DIR, r["name"])
            with open(out_path, "wb") as f:
                f.write(r["nb_bytes"])
    print(f"\n{'ALL NOTEBOOKS EXECUTED CLEANLY' if all_good else 'SOME NOTEBOOKS FAILED, see logs above'}")
