"""Stage 5d -- gradient/propagation and CAM-family XAI methods (brief's
gradient-and-propagation family + the vision-native CAM family), completing
the XAI method roster started in s5_xai.py (SHAP/LIME/permutation/ALE/IG).

  modal run s5d_gradient_cam.py::run

Two sub-problems, handled separately because the two model classes involved
support genuinely different method families:

  1. LRP / Guided Backprop / DeconvNet (captum.attr.LRP, GuidedBackprop,
     Deconvolution). captum's LRP only has propagation rules defined for
     feedforward layer types (Linear, Conv*, pooling, ReLU, ...) -- NOT for
     recurrent layers such as nn.GRU. We test that directly against the
     saved GRUNet artifact first (expect failure), then fall back to a small
     feedforward MLP trained specifically for this cell on
     PRICE_FEATS+SENT_FEATS+GT_FEATS (dir_next, train<2023/test=2023). This
     also lets LRP/GuidedBackprop/Deconvolution see the ground-truth columns,
     which the saved GRU/CNN1d artifacts do not have.

  2. Grad-CAM / Grad-CAM++ / Score-CAM. This is explicitly a vision-native,
     spatial-map family (per the brief: "only valid on the CNN1d variant").
     Applied ONLY to the saved CNN1d artifact, targeting its conv2 layer.
     Grad-CAM uses captum.attr.LayerGradCam directly. Grad-CAM++ and
     Score-CAM are not in captum and are hand-rolled here, adapted to the
     1D/time-axis case (channels-over-time instead of channels-over-pixels).
     These three give a per-timestep (length-LOOKBACK) importance curve, not
     a per-feature vector, so as a supplement we also run Integrated
     Gradients on the CNN1d model directly (same recipe as the GRU IG cell
     in s5_xai.py) to get a per-INPUT-feature global vector that CAN be
     scored against the P10 ground truth.

Writes /data/results/metrics/xai_gradient_cam.json and a few example figures.
"""
import modal
from _common import image, data_vol, DATA, SEED, LOOKBACK, GROUND_TRUTH

app = modal.App("finsent-s5d-gradient-cam", image=image)

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


def _faithfulness(imp, method_name):
    out = {"method": method_name}
    ranked = imp.sort_values(ascending=False)
    rank_of = {f: int(ranked.index.get_loc(f)) + 1 for f in imp.index}
    if "gt_signal" in imp.index and "gt_noise" in imp.index:
        out["signal_rank"] = rank_of["gt_signal"]; out["noise_rank"] = rank_of["gt_noise"]
        out["n_features"] = len(imp)
        out["ranks_signal_above_noise"] = bool(rank_of["gt_signal"] < rank_of["gt_noise"])
    if "gt_signal_nl" in imp.index and "gt_noise" in imp.index:
        out["signal_nl_rank"] = rank_of["gt_signal_nl"]
        out["ranks_nonlinear_signal_above_noise"] = bool(rank_of["gt_signal_nl"] < rank_of["gt_noise"])
    if "gt_redundant_copy" in imp.index and "vol_z20" in imp.index:
        a, b = float(imp["gt_redundant_copy"]), float(imp["vol_z20"])
        out["redundant_copy_importance"] = a; out["source_feature_importance"] = b
        out["redundant_copy_credit_share"] = float(a / (a + b)) if (a + b) > 0 else float("nan")
    return out


def _mpl():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 150, "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False})
    return plt


@app.function(volumes={DATA: data_vol}, cpu=4.0, memory=16384, timeout=30 * 60)
def run():
    import os, json, numpy as np, pandas as pd, torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    plt = _mpl()
    os.makedirs(FIG, exist_ok=True); os.makedirs(MET, exist_ok=True)
    torch.manual_seed(SEED); np.random.seed(SEED)
    rng = np.random.RandomState(SEED)

    results = {}

    # =====================================================================
    # 0. GRUNet class (must match s4_lstm.py exactly to reload weights) and
    #    a first-pass compatibility test: does captum.attr.LRP work on it?
    # =====================================================================
    class GRUNet(nn.Module):
        def __init__(self, nf, hidden=64):
            super().__init__()
            self.gru = nn.GRU(nf, hidden, num_layers=2, batch_first=True, dropout=0.2)
            self.head = nn.Linear(hidden, 1)
        def forward(self, x):
            o, _ = self.gru(x); return self.head(o[:, -1]).squeeze(-1)

    from captum.attr import LRP, GuidedBackprop, Deconvolution, IntegratedGradients, LayerGradCam

    gru_lrp_attempt = {"attempted": True}
    try:
        bundle = np.load(os.path.join(ART, "gru_ig_bundle.npz"), allow_pickle=True)
        Xt_gru = torch.tensor(bundle["X_test"][:4]).float()
        gru = GRUNet(Xt_gru.shape[-1])
        gru.load_state_dict(torch.load(os.path.join(ART, "gru_dir_2023.pt"), map_location="cpu"))
        gru.eval()
        lrp_gru = LRP(gru)
        _ = lrp_gru.attribute(Xt_gru, target=None)
        gru_lrp_attempt["success"] = True
        gru_lrp_attempt["note"] = "unexpected: captum.attr.LRP ran on GRUNet without error"
        print("LRP on GRUNet: succeeded (unexpected)")
    except Exception as e:
        gru_lrp_attempt["success"] = False
        gru_lrp_attempt["error"] = f"{type(e).__name__}: {e}"
        print("LRP on GRUNet failed as expected:", gru_lrp_attempt["error"])

    applicability_note_gradient = (
        "captum.attr.LRP only has propagation rules defined for feedforward layer "
        "types (Linear, Conv*, pooling, ReLU, ...); it has no rule for nn.GRU, so it "
        "was tested directly against the saved GRUNet artifact (gru_dir_2023.pt) and "
        f"{'succeeded' if gru_lrp_attempt.get('success') else 'failed'} "
        f"({gru_lrp_attempt.get('error', 'n/a')}). Per the brief, LRP/GuidedBackprop/"
        "Deconvolution were instead applied to a small feedforward MLP trained "
        "specifically for this cell on PRICE_FEATS+SENT_FEATS+GT_FEATS -- this "
        "MLP-friendly architecture is exactly what these three propagation-based "
        "methods are designed for, and it lets the ground-truth columns (gt_signal/"
        "gt_noise/gt_redundant_copy) be included, which the saved GRU/CNN1d "
        "artifacts do not have. GuidedBackprop and Deconvolution were not separately "
        "re-tested on GRUNet: both work by hooking nn.ReLU modules, and GRUNet "
        "contains none (its nonlinearities are internal to the fused GRU cell), so "
        "even a non-erroring run would be architecturally meaningless (equivalent to "
        "plain gradients) rather than a real guided-backprop/deconv attribution -- "
        "the same MLP fallback is used for both, for a like-for-like comparison "
        "with LRP."
    )

    # =====================================================================
    # 1. Feedforward MLP trained specifically for LRP/GuidedBackprop/Deconv
    # =====================================================================
    feats = PRICE_FEATS + SENT_FEATS + GT_FEATS
    df = pd.read_parquet(os.path.join(BUILD, "panel.parquet")).copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Date", "Symbol"]).reset_index(drop=True)
    tr = df[df["Date"].dt.year < 2023]
    te = df[df["Date"].dt.year == 2023]
    sc = StandardScaler().fit(tr[feats].values)
    Xtr = sc.transform(tr[feats].values).astype(np.float32)
    Xte = sc.transform(te[feats].values).astype(np.float32)
    ytr = tr["dir_next"].values.astype(np.float32)
    yte = te["dir_next"].values.astype(np.float32)

    mlp = nn.Sequential(nn.Linear(len(feats), 64), nn.ReLU(),
                         nn.Linear(64, 32), nn.ReLU(),
                         nn.Linear(32, 1))
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-5)
    lossf = nn.BCEWithLogitsLoss()
    Xtr_t = torch.tensor(Xtr); ytr_t = torch.tensor(ytr).unsqueeze(-1)
    idx = np.arange(len(Xtr_t)); bs = 512
    mlp.train()
    for epoch in range(15):
        rng.shuffle(idx)
        for s in range(0, len(idx), bs):
            b = idx[s:s + bs]
            opt.zero_grad()
            out = mlp(Xtr_t[b])
            loss = lossf(out, ytr_t[b])
            loss.backward(); opt.step()
    mlp.eval()
    with torch.no_grad():
        pte = torch.sigmoid(mlp(torch.tensor(Xte))).numpy().ravel()
    mlp_acc = float(accuracy_score(yte, (pte > 0.5).astype(int)))
    print(f"MLP (for LRP/GuidedBackprop/Deconv) test accuracy: {mlp_acc:.4f}")

    n_explain = min(500, len(Xte))
    sample_i = rng.choice(len(Xte), n_explain, replace=False)
    Xexp = torch.tensor(Xte[sample_i]); Xexp.requires_grad_(True)

    def _mlp_method(attr_obj, name):
        try:
            attr = attr_obj.attribute(Xexp, target=None)
            imp = pd.Series(attr.abs().mean(dim=0).detach().numpy(), index=feats).sort_values(ascending=False)
            out = {
                "applicability_note": applicability_note_gradient,
                "gru_lrp_attempt": gru_lrp_attempt,
                "fallback_model": "feedforward MLP (Linear64-ReLU-Linear32-ReLU-Linear1) trained "
                                   "specifically for this cell on PRICE_FEATS+SENT_FEATS+GT_FEATS "
                                   "(dir_next, train<2023/test=2023, StandardScaler fit on train)",
                "mlp_test_accuracy": mlp_acc,
                "n_samples_explained": int(n_explain),
                "top_features": {k: float(v) for k, v in imp.head(10).items()},
            }
            out.update(_faithfulness(imp, name))
            print(f"{name} done; top feature = {imp.index[0]}")
            return out, imp
        except Exception as e:
            print(f"{name} step failed:", e)
            return {"applicability_note": applicability_note_gradient, "error": f"{type(e).__name__}: {e}"}, None

    results["lrp"], lrp_imp = _mlp_method(LRP(mlp), "lrp")
    results["guided_backprop"], gb_imp = _mlp_method(GuidedBackprop(mlp), "guided_backprop")
    results["deconvnet"], dc_imp = _mlp_method(Deconvolution(mlp), "deconvnet")

    # figure: LRP + Guided Backprop + Deconvolution feature bars (MLP)
    try:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
        for ax, (name, imp) in zip(axes, [("LRP", lrp_imp), ("Guided Backprop", gb_imp), ("Deconvolution", dc_imp)]):
            if imp is None:
                ax.set_title(f"{name} (failed)"); continue
            order = imp.sort_values()
            colors = ["#B4740F" if f in SENT_FEATS else ("#7A2E8E" if f.startswith("gt_") else "#2F4B94")
                      for f in order.index]
            ax.barh(order.index, order.values, color=colors)
            ax.set_title(name, fontsize=11)
            ax.set_xlabel("mean |attribution|")
        fig.suptitle("MLP propagation-based attributions (purple = ground-truth columns)", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG, "fig_lrp_guided_deconv_mlp.png"), bbox_inches="tight")
        plt.close()
        print("saved fig_lrp_guided_deconv_mlp.png")
    except Exception as e:
        print("figure step (mlp propagation bars) failed:", e)

    # =====================================================================
    # 2. CNN1dNet class (must match s4_lstm.py exactly) + Grad-CAM family
    # =====================================================================
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
            x = x.transpose(1, 2)        # (B, F, T)
            x = self.act(self.conv1(x))
            x = self.act(self.conv2(x))
            x = self.drop(self.pool(x).squeeze(-1))
            return self.head(x).squeeze(-1)

    cnn_applicability_note = (
        "Grad-CAM / Grad-CAM++ / Score-CAM are a vision-native, spatial-attribution "
        "family (per the brief, 'only valid on the CNN1d variant'); they are applied "
        "ONLY to the CNN1d artifact, targeting its conv2 layer, and are NOT applied "
        "to the GRU or the MLP used for LRP/GuidedBackprop/Deconvolution above. "
        "conv1 and conv2 both use kernel_size=3, padding=1, stride=1, so the time "
        f"axis stays at length {LOOKBACK} through both conv layers -- the CAM at "
        "conv2 is already the same length as the lookback window and needs no "
        "upsampling. Gap-fix round 2: this now loads the GROUND-TRUTH-AUGMENTED "
        "CNN1d artifact (cnn1d_gt_dir_2023.pt, s4c_gt_artifacts.py -- "
        "PRICE_FEATS+SENT_FEATS+GT_FEATS, trained separately from the honest "
        "model-zoo results), so the three CAM methods' primary output (a "
        "per-timestep curve, not a per-feature vector) still cannot itself carry a "
        "signal/noise rank, but the supplementary per-feature vector via Integrated "
        "Gradients on this same (now GT-augmented) CNN1d model (see "
        "'feature_axis_via_integrated_gradients' below) now gets a real faithfulness "
        "score, and is the feature-level faithfulness proxy reported for this "
        "whole CAM family."
    )

    cnn_bundle = np.load(os.path.join(ART, "cnn1d_gt_ig_bundle.npz"), allow_pickle=True)
    cnn_feats = list(cnn_bundle["feat_names"])
    cnn1d = CNN1dNet(len(cnn_feats))
    cnn1d.load_state_dict(torch.load(os.path.join(ART, "cnn1d_gt_dir_2023.pt"), map_location="cpu"))
    cnn1d.eval()

    n_cam = min(64, len(cnn_bundle["X_test"]))
    cam_i = rng.choice(len(cnn_bundle["X_test"]), n_cam, replace=False)
    Xcam = torch.tensor(cnn_bundle["X_test"][cam_i]).float()

    # ---- Grad-CAM (captum.attr.LayerGradCam on conv2) ----
    try:
        gradcam = LayerGradCam(cnn1d, cnn1d.conv2)
        cam_attr = gradcam.attribute(Xcam, target=None, relu_attributions=True)   # (B, 1, T)
        gradcam_curve = cam_attr.squeeze(1).mean(dim=0).detach().numpy()          # (T,)
        gradcam_ok = True
    except Exception as e:
        gradcam_curve = None
        print("Grad-CAM step failed:", e)
        gradcam_ok = False

    # ---- Grad-CAM++ (hand-rolled, 1D/time-axis; grad-power approximation) ----
    def gradcam_pp(model, layer, x):
        """Standard practical Grad-CAM++ approximation (as used by common public
        implementations, e.g. jacobgil/pytorch-grad-cam): alpha weights are built
        from powers of the first-order gradient (grad^2, grad^3) rather than true
        closed-form second/third-order autograd. The exact Chattopadhay et al.
        alpha-weighting derivation assumes a softmax multi-class score; our CNN1d
        outputs a single scalar logit (binary direction, no softmax), so this
        power-of-gradient approximation is the documented, honest adaptation used
        here, applied over the time axis instead of 2D pixels."""
        acts = {}
        def fwd_hook(m, i, o): acts["v"] = o
        h = layer.register_forward_hook(fwd_hook)
        xg = x.clone().requires_grad_(True)
        out = model(xg)                      # (B,)
        grads = torch.autograd.grad(out.sum(), acts["v"], retain_graph=False)[0]   # (B,C,T)
        h.remove()
        A = acts["v"].detach()
        g1, g2, g3 = grads, grads ** 2, grads ** 3
        sum_A_g3 = (A * g3).sum(dim=2, keepdim=True)          # (B,C,1)
        denom = 2 * g2 + sum_A_g3
        denom = torch.where(denom != 0, denom, torch.ones_like(denom))
        alpha = g2 / denom                                     # (B,C,T)
        weights = (alpha * F.relu(g1)).sum(dim=2)              # (B,C)
        cam = F.relu((weights.unsqueeze(-1) * A).sum(dim=1))   # (B,T)
        return cam.mean(dim=0).detach().numpy()

    try:
        gradcam_pp_curve = gradcam_pp(cnn1d, cnn1d.conv2, Xcam)
        gradcam_pp_ok = True
    except Exception as e:
        gradcam_pp_curve = None
        print("Grad-CAM++ step failed:", e)
        gradcam_pp_ok = False

    # ---- Score-CAM (hand-rolled, 1D/time-axis) ----
    def score_cam(model, layer, x):
        """Mask the INPUT sequence with each (already-full-length, no upsampling
        needed here) channel activation map from conv2, run the masked input
        through the model, and weight that channel's map by the resulting output
        score (softmax-normalized across channels), summed."""
        acts = {}
        def fwd_hook(m, i, o): acts["v"] = o.detach()
        h = layer.register_forward_hook(fwd_hook)
        with torch.no_grad():
            model(x)
        h.remove()
        A = acts["v"]                      # (B, C, T)
        B, C, T = A.shape
        Amin = A.amin(dim=2, keepdim=True); Amax = A.amax(dim=2, keepdim=True)
        Anorm = (A - Amin) / (Amax - Amin + 1e-9)     # (B, C, T), in [0, 1]
        scores = torch.zeros(B, C)
        with torch.no_grad():
            for c in range(C):
                mask = Anorm[:, c, :].unsqueeze(-1)    # (B, T, 1) -- broadcasts over feature dim
                x_masked = x * mask                    # (B, T, F)
                scores[:, c] = model(x_masked)
        weights = torch.softmax(scores, dim=1)          # (B, C)
        cam = F.relu((weights.unsqueeze(-1) * Anorm).sum(dim=1))   # (B, T)
        return cam.mean(dim=0).numpy()

    try:
        score_cam_curve = score_cam(cnn1d, cnn1d.conv2, Xcam)
        score_cam_ok = True
    except Exception as e:
        score_cam_curve = None
        print("Score-CAM step failed:", e)
        score_cam_ok = False

    # ---- supplementary per-feature vector: Integrated Gradients on CNN1d ----
    try:
        ig = IntegratedGradients(cnn1d)
        ig_attr = ig.attribute(Xcam, baselines=torch.zeros_like(Xcam), n_steps=32)
        ig_imp = pd.Series(ig_attr.abs().mean(dim=(0, 1)).detach().numpy(), index=cnn_feats).sort_values(ascending=False)
        ig_feature_axis = {
            "computed_as": "Integrated Gradients (captum) on the full CNN1d model, mean(|attr|) "
                            "over batch and time axes -> one score per INPUT feature. This is the "
                            "'supplementary Integrated-Gradients-style feature axis' the brief "
                            "allows when the pure spatial CAM does not naturally give per-feature "
                            "scores (Grad-CAM/Grad-CAM++/Score-CAM only yield a per-timestep curve).",
            "n_samples": int(n_cam),
            "top_features": {k: float(v) for k, v in ig_imp.head(10).items()},
        }
        ig_feature_axis.update(_faithfulness(ig_imp, "cnn1d_ig_feature_axis"))
        ig_feature_ok = True
    except Exception as e:
        ig_feature_axis = {"error": f"{type(e).__name__}: {e}"}
        ig_imp = None
        ig_feature_ok = False
        print("CNN1d feature-axis IG step failed:", e)

    results["gradcam"] = {
        "applicability_note": cnn_applicability_note,
        "layer": "conv2",
        "n_samples": int(n_cam),
        "per_timestep_importance": (gradcam_curve.tolist() if gradcam_ok else None),
        "success": gradcam_ok,
        "feature_axis_via_integrated_gradients": ig_feature_axis,
    }
    results["gradcam_pp"] = {
        "applicability_note": cnn_applicability_note,
        "layer": "conv2",
        "n_samples": int(n_cam),
        "per_timestep_importance": (gradcam_pp_curve.tolist() if gradcam_pp_ok else None),
        "success": gradcam_pp_ok,
        "note": "hand-rolled (not in captum); grad-power alpha approximation, see code comment",
        "feature_axis_via_integrated_gradients": ig_feature_axis,
    }
    results["score_cam"] = {
        "applicability_note": cnn_applicability_note,
        "layer": "conv2",
        "n_samples": int(n_cam),
        "per_timestep_importance": (score_cam_curve.tolist() if score_cam_ok else None),
        "success": score_cam_ok,
        "note": "hand-rolled (not in captum); channel activation maps used to mask the input "
                "sequence, output score softmax-normalized across channels",
        "feature_axis_via_integrated_gradients": ig_feature_axis,
    }

    # ---- figures: CAM-family per-timestep curves + feature-axis IG bar ----
    try:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        days = np.arange(1, LOOKBACK + 1)
        if gradcam_ok:
            ax.plot(days, gradcam_curve / (np.abs(gradcam_curve).max() + 1e-9), label="Grad-CAM", marker="o")
        if gradcam_pp_ok:
            ax.plot(days, gradcam_pp_curve / (np.abs(gradcam_pp_curve).max() + 1e-9), label="Grad-CAM++", marker="s")
        if score_cam_ok:
            ax.plot(days, score_cam_curve / (np.abs(score_cam_curve).max() + 1e-9), label="Score-CAM", marker="^")
        ax.set_xlabel(f"lookback day (1 = oldest, {LOOKBACK} = most recent / prediction day)")
        ax.set_ylabel("normalized importance")
        ax.set_title("CNN1d Grad-CAM family -- per-timestep attribution", fontsize=11)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(FIG, "fig_gradcam_family_timestep.png"), bbox_inches="tight")
        plt.close()
        print("saved fig_gradcam_family_timestep.png")
    except Exception as e:
        print("figure step (cam timestep curves) failed:", e)

    try:
        if ig_imp is not None:
            plt.figure(figsize=(7.5, 6))
            order = ig_imp.sort_values()
            colors = ["#B4740F" if f in SENT_FEATS else "#2F4B94" for f in order.index]
            plt.barh(order.index, order.values, color=colors)
            plt.xlabel("mean |Integrated Gradients attribution|")
            plt.title("CNN1d supplementary feature-axis attribution (Integrated Gradients)", fontsize=11)
            plt.tight_layout()
            plt.savefig(os.path.join(FIG, "fig_cnn1d_feature_axis_ig.png"), bbox_inches="tight")
            plt.close()
            print("saved fig_cnn1d_feature_axis_ig.png")
    except Exception as e:
        print("figure step (cnn1d feature-axis ig bar) failed:", e)

    with open(os.path.join(MET, "xai_gradient_cam.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2)[:4000])
    data_vol.commit()
