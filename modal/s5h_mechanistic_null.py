"""Stage 5h -- null baselines for the SAE latent interpretability count and the
ACDC-inspired ablation z>2 sweep from s5e_mechanistic.py (S14), plus an exact
recurrence probability for the "layer1_full" top-ablation-component claim in
xai_mechanistic_sensitivity.json.

Motivation (R4 review, item A2, HIGHEST PRIORITY): s5e reports 253/256 SAE
latents "interpretable" at |corr| > 0.3, where each latent's correlation is the
MAX over 22 named input features -- a best-of-22 selection that inflates the
count via multiple comparisons, with no null baseline reported. Similarly the
ablation sweep declares a component "notable" at a fixed z > 2 threshold with
no null, and the multi-seed sensitivity run (xai_mechanistic_sensitivity.json)
reports "layer1_full" as the top ablation component in most seed/fold runs
with no probability computed for how likely that recurrence is by chance.

This script does NOT modify xai_mechanistic.json or
xai_mechanistic_sensitivity.json (read-only). It reuses the exact same cached
Transformer (transformer_dir_2023.pt) and test bundle
(transformer_ig_bundle.npz) as s5e's run(), retrains the identical SAE (same
architecture, hyperparameters, SEED) to reproduce the same latent
activations, and adds:

  1. SAE feature-shuffle null: independently permute each of the 22 feature
     columns S=20 times, recompute the max-over-features-per-latent count at
     tau in {0.2,0.3,0.4,0.5,0.6}, and compare observed vs null
     (null_mean / null_p95 / null_max).
  2. Ablation null: independently permute each of the 22 input feature columns
     (across examples, same style as (1)) S=20 times, rerun the 10-component
     ablation sweep on the structurally-broken inputs, and count how many
     components cross z > 2 by chance alone.
  3. Recurrence exact probability for the modal top ablation component
     (layer1_full) using math.comb (exact binomial tail): both for that one
     specific component (p_binom_one) and, Bonferroni-style, for "any of the
     C candidate components recurring that often" (p_union).

  modal run s5h_mechanistic_null.py::run

Writes /data/results/metrics/xai_sae_null.json and
/data/results/metrics/xai_ablation_null.json.
"""
import modal
from _common import image, data_vol, DATA, SEED

app = modal.App("finsent-s5h-mech-null", image=image)

RES = f"{DATA}/results"
MET = f"{RES}/metrics"
ART = f"{RES}/artifacts"


@app.function(volumes={DATA: data_vol}, gpu="a10g", timeout=60 * 30)
def run():
    import os, json, math, numpy as np, torch
    import torch.nn as nn
    os.makedirs(MET, exist_ok=True)
    torch.manual_seed(SEED); np.random.seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev)

    # ---------------------------------------------------------------
    # load transformer + held-out test bundle (identical to s5e_mechanistic.py::run)
    # ---------------------------------------------------------------
    bundle = np.load(os.path.join(ART, "transformer_ig_bundle.npz"), allow_pickle=True)
    X_test = bundle["X_test"]                       # [N, 20, 22] standardized
    feat_names = [str(f) for f in bundle["feat_names"]]
    N, T, NF = X_test.shape
    print(f"loaded bundle: N={N} T={T} NF={NF}")

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
        def forward(self, x):
            x = self.proj(x)
            x = self.pos(x)
            x = self.encoder(x)
            x = x[:, -1]
            return self.head(x).squeeze(-1)

    D_MODEL, NHEAD, NLAYERS = 64, 4, 2
    model = TransformerNet(NF, d_model=D_MODEL, nhead=NHEAD, num_layers=NLAYERS).to(dev)
    state = torch.load(os.path.join(ART, "transformer_dir_2023.pt"), map_location=dev)
    model.load_state_dict(state)
    model.eval()

    Xt = torch.tensor(X_test).float().to(dev)

    # =================================================================
    # reproduce s5e's SAE exactly (same architecture, same hyperparameters,
    # same SEED, same training loop) to obtain the same latent activations
    # =================================================================
    class EncoderWrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            x = self.m.proj(x)
            x = self.m.pos(x)
            x = self.m.encoder(x)
            return x[:, -1]

    enc_wrap = EncoderWrapper(model).to(dev).eval()
    with torch.no_grad():
        H = enc_wrap(Xt).cpu().numpy()
    H_mean = H.mean(0, keepdims=True)
    H_std = H.std(0, keepdims=True) + 1e-6
    Hn = (H - H_mean) / H_std
    Hn_t = torch.tensor(Hn).float().to(dev)

    D_IN = H.shape[1]
    D_HID = 4 * D_IN

    class SAE(nn.Module):
        def __init__(self, d_in, d_hidden):
            super().__init__()
            self.enc = nn.Linear(d_in, d_hidden)
            self.dec = nn.Linear(d_hidden, d_in)
            self.relu = nn.ReLU()
        def forward(self, x):
            h = self.relu(self.enc(x))
            xhat = self.dec(h)
            return xhat, h

    sae = SAE(D_IN, D_HID).to(dev)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
    L1_COEF = 1e-3
    N_EPOCHS = 400
    sae.train()
    for epoch in range(N_EPOCHS):
        opt.zero_grad()
        xhat, h = sae(Hn_t)
        loss = nn.functional.mse_loss(xhat, Hn_t) + L1_COEF * h.abs().mean()
        loss.backward(); opt.step()
    sae.eval()
    with torch.no_grad():
        _, h_act = sae(Hn_t)
    H_act = h_act.cpu().numpy()      # [N, 256] SAE latent activations
    print("H_act shape:", H_act.shape)

    feat_vals = X_test[:, -1, :]     # [N, 22], row-aligned with H_act by construction

    def safe_corr(a, b):
        if np.std(a) < 1e-10 or np.std(b) < 1e-10:
            return 0.0
        c = np.corrcoef(a, b)[0, 1]
        return float(c) if np.isfinite(c) else 0.0

    def best_abs_corr_per_latent(H_act_arr, feats):
        """[D_HID] vector: for each latent, max |corr| over the NF feature columns."""
        out = np.zeros(H_act_arr.shape[1])
        for j in range(H_act_arr.shape[1]):
            act = H_act_arr[:, j]
            if act.std() < 1e-6:
                out[j] = 0.0
                continue
            corrs = [safe_corr(act, feats[:, k]) for k in range(feats.shape[1])]
            out[j] = float(np.max(np.abs(corrs)))
        return out

    # ---------------------------------------------------------------
    # observed count (reproduces s5e's 253-254/256-style number on this fold)
    # ---------------------------------------------------------------
    observed_best = best_abs_corr_per_latent(H_act, feat_vals)
    CORR_THRESHOLD = 0.3
    observed_interpretable = int((observed_best > CORR_THRESHOLD).sum())
    print(f"observed interpretable at tau={CORR_THRESHOLD}: {observed_interpretable}/{D_HID}")

    TAU_SWEEP = [0.2, 0.3, 0.4, 0.5, 0.6]
    S = 20

    # null count matrix: [S, len(TAU_SWEEP)] -- fixed latent activations H_act,
    # each shuffle independently permutes every one of the 22 feature columns
    null_counts = np.zeros((S, len(TAU_SWEEP)))
    for s in range(S):
        rs = np.random.RandomState(SEED + 1 + s)
        feats_shuf = feat_vals.copy()
        for k in range(NF):
            perm = rs.permutation(N)
            feats_shuf[:, k] = feat_vals[perm, k]
        best_shuf = best_abs_corr_per_latent(H_act, feats_shuf)
        for ti, tau in enumerate(TAU_SWEEP):
            null_counts[s, ti] = int((best_shuf > tau).sum())
        print(f"[sae-null] shuffle {s+1}/{S}: counts@tau=",
              {tau: int(null_counts[s, ti]) for ti, tau in enumerate(TAU_SWEEP)})

    threshold_sweep = []
    for ti, tau in enumerate(TAU_SWEEP):
        obs = int((observed_best > tau).sum())
        nm = float(null_counts[:, ti].mean())
        np95 = float(np.percentile(null_counts[:, ti], 95))
        threshold_sweep.append({"tau": tau, "observed": obs, "null_mean": nm, "null_p95": np95})

    idx03 = TAU_SWEEP.index(0.3)
    tau03 = threshold_sweep[idx03]
    excess_at_03 = tau03["observed"] - tau03["null_mean"]

    smallest_tau_exceeds = None
    for row in sorted(threshold_sweep, key=lambda r: r["tau"]):
        if row["observed"] > row["null_p95"]:
            smallest_tau_exceeds = row["tau"]
            break

    null_mean_03 = float(null_counts[:, idx03].mean())
    null_p95_03 = float(np.percentile(null_counts[:, idx03], 95))
    null_max_03 = int(null_counts[:, idx03].max())

    if smallest_tau_exceeds is None:
        sae_verdict = (
            f"At NO threshold in {TAU_SWEEP} does the observed interpretable-latent count clearly "
            f"exceed the 95th percentile of a feature-shuffle null; at tau=0.3 the observed count "
            f"({tau03['observed']}) is essentially indistinguishable from chance (null_mean="
            f"{null_mean_03:.1f}, null_p95={null_p95_03:.1f} out of {D_HID}). The headline 253/256 "
            f"figure is almost entirely a multiple-comparisons artifact of taking the max |correlation| "
            f"over 22 features per latent: shuffling the features destroys any real feature-latent "
            f"relationship yet still produces a similarly large 'interpretable' count by chance alone, "
            f"because with 22 independent draws per latent the max easily exceeds 0.3 even under pure "
            f"noise. The paper should NOT report 253/256 unguarded; it should report the excess over "
            f"the null ({excess_at_03:.1f} latents at tau=0.3) or restrict the claim to a threshold "
            f"where the count clearly separates from the null, and state plainly that most nominally "
            f"'interpretable' latents at tau=0.3 are not distinguishable from a best-of-22 chance artifact."
        )
    else:
        frac_real = excess_at_03 / tau03["observed"] if tau03["observed"] > 0 else 0.0
        if frac_real >= 0.95:
            sae_verdict = (
                f"The multiplicity concern is NOT confirmed: at tau={smallest_tau_exceeds} the observed "
                f"interpretable-latent count already clearly exceeds the feature-shuffle null's 95th "
                f"percentile, and at tau=0.3 the feature-shuffle null is essentially zero (null_mean="
                f"{null_mean_03:.2f}, null_p95={null_p95_03:.2f}, null_max={null_max_03} out of {D_HID}) "
                f"against an observed count of {tau03['observed']}. Destroying the real feature-latent "
                f"relationship (independently permuting each of the 22 feature columns) collapses the "
                f"'interpretable' count to essentially 0, so the max-over-22-features selection is NOT "
                f"manufacturing the 253/256 figure by chance -- almost all of it ({excess_at_03:.0f}/"
                f"{tau03['observed']} = {100*frac_real:.1f}%) is excess over the null and reflects a real, "
                f"reproducible correlation between SAE latents and named input features, not a "
                f"multiple-comparisons artifact. The paper can keep reporting {tau03['observed']}/{D_HID} "
                f"but SHOULD add the null baseline (null_mean={null_mean_03:.2f} at tau=0.3, essentially "
                f"0/{D_HID} under a feature-shuffle null) as the guard that was previously missing, rather "
                f"than reporting 253/256 unguarded."
            )
        else:
            sae_verdict = (
                f"At tau={smallest_tau_exceeds} the observed interpretable-latent count first clearly "
                f"exceeds the feature-shuffle null's 95th percentile; at tau=0.3 the observed count "
                f"({tau03['observed']}) exceeds null_mean ({null_mean_03:.1f}) by {excess_at_03:.1f} "
                f"latents ({100*frac_real:.1f}% of the observed count), so part of the raw 253/256 is "
                f"still inflated by the max-over-22-features selection (null_mean is {null_mean_03:.1f}/"
                f"{D_HID} at tau=0.3 under pure chance). The paper should replace the bare 253/256 with "
                f"the excess-over-chance figure ({excess_at_03:.1f} latents at tau=0.3) or report the "
                f"count only at tau>= {smallest_tau_exceeds}, where the signal clearly separates from "
                f"the null."
            )
    print(sae_verdict)

    sae_null_results = {
        "correlation_threshold": CORR_THRESHOLD,
        "observed_interpretable": observed_interpretable,
        "n_latents": D_HID,
        "n_shuffles": S,
        "null_interpretable_mean": null_mean_03,
        "null_interpretable_p95": null_p95_03,
        "null_interpretable_max": null_max_03,
        "excess_over_chance_at_0p3": float(excess_at_03),
        "threshold_sweep": threshold_sweep,
        "smallest_tau_observed_exceeds_null_p95": smallest_tau_exceeds,
        "verdict": sae_verdict,
    }
    with open(os.path.join(MET, "xai_sae_null.json"), "w") as f:
        json.dump(sae_null_results, f, indent=2)
    print("saved xai_sae_null.json")

    # =================================================================
    # PART 2 -- ablation z>2 null + exact recurrence probability
    # =================================================================
    def zero_ablate_head(mdl, layer_idx, head_idx):
        layer = mdl.encoder.layers[layer_idx]
        w = layer.self_attn.out_proj.weight
        head_dim = D_MODEL // NHEAD
        s, e = head_idx * head_dim, (head_idx + 1) * head_dim
        orig = w.data[:, s:e].clone()
        w.data[:, s:e] = 0.0
        return orig, s, e

    def restore_head(mdl, layer_idx, orig, s, e):
        mdl.encoder.layers[layer_idx].self_attn.out_proj.weight.data[:, s:e] = orig

    def bypass_layer(mdl, layer_idx):
        layer = mdl.encoder.layers[layer_idx]
        orig_forward = layer.forward
        def identity_forward(src, *a, **kw):
            return src
        layer.forward = identity_forward
        return orig_forward

    def restore_layer(mdl, layer_idx, orig_forward):
        mdl.encoder.layers[layer_idx].forward = orig_forward

    def ablation_sweep(mdl, X_in):
        """Identical exhaustive head+layer ablation sweep as s5e; returns
        (comp_names, effects) -- mean |delta logit| per component vs baseline."""
        with torch.no_grad():
            baseline_out = mdl(X_in).cpu().numpy()
        comp_names, effects = [], []
        with torch.no_grad():
            for li in range(NLAYERS):
                for hi in range(NHEAD):
                    orig, s, e = zero_ablate_head(mdl, li, hi)
                    out = mdl(X_in).cpu().numpy()
                    restore_head(mdl, li, orig, s, e)
                    effects.append(float(np.mean(np.abs(out - baseline_out))))
                    comp_names.append(f"layer{li}_head{hi}")
            for li in range(NLAYERS):
                orig_fwd = bypass_layer(mdl, li)
                out = mdl(X_in).cpu().numpy()
                restore_layer(mdl, li, orig_fwd)
                effects.append(float(np.mean(np.abs(out - baseline_out))))
                comp_names.append(f"layer{li}_full")
        with torch.no_grad():
            restored_out = mdl(X_in).cpu().numpy()
        assert np.allclose(restored_out, baseline_out, atol=1e-5), "model not fully restored after sweep"
        return comp_names, np.array(effects)

    NOTABLE_Z = 2.0
    C_COMPONENTS = NLAYERS * NHEAD + NLAYERS   # 8 heads (2 layers x 4 heads) + 2 full layers = 10
    assert C_COMPONENTS == 10

    # sanity: reproduce s5e's observed sweep on this exact cached model/fold
    comp_names_obs, effects_obs = ablation_sweep(model, Xt)
    mean_obs, std_obs = float(effects_obs.mean()), float(effects_obs.std())
    z_obs = (effects_obs - mean_obs) / (std_obs + 1e-12)
    n_notable_obs = int((z_obs > NOTABLE_Z).sum())
    print(f"observed (reproduced) ablation sweep: n_notable={n_notable_obs}, "
          f"top={comp_names_obs[int(np.argmax(z_obs))]}, max_z={z_obs.max():.3f}")

    S_ABL = 20
    null_notable_counts = np.zeros(S_ABL)
    for s in range(S_ABL):
        rs = np.random.RandomState(SEED + 500 + s)
        X_shuf = X_test.copy()
        for k in range(NF):
            perm = rs.permutation(N)
            X_shuf[:, :, k] = X_test[perm, :, k]
        X_shuf_t = torch.tensor(X_shuf).float().to(dev)
        _, effects_shuf = ablation_sweep(model, X_shuf_t)
        mean_s, std_s = float(effects_shuf.mean()), float(effects_shuf.std())
        z_s = (effects_shuf - mean_s) / (std_s + 1e-12)
        null_notable_counts[s] = int((z_s > NOTABLE_Z).sum())
        print(f"[ablation-null] shuffle {s+1}/{S_ABL}: n_notable={int(null_notable_counts[s])}")

    null_notable_mean = float(null_notable_counts.mean())
    null_notable_p95 = float(np.percentile(null_notable_counts, 95))

    # ---------------------------------------------------------------
    # exact recurrence probability for the modal top-ablation-component claim
    # (reads xai_mechanistic_sensitivity.json -- NOT modified, only read)
    # ---------------------------------------------------------------
    sens_path = os.path.join(MET, "xai_mechanistic_sensitivity.json")
    with open(sens_path) as f:
        sens = json.load(f)
    top_components = sens["top_component_per_run"]
    R = len(top_components)
    MODAL_COMPONENT = max(set(top_components), key=top_components.count)
    k = top_components.count(MODAL_COMPONENT)
    C = C_COMPONENTS
    p1 = 1.0 / C
    p_binom_one = float(sum(math.comb(R, i) * (p1 ** i) * ((1 - p1) ** (R - i)) for i in range(k, R + 1)))
    p_union = float(min(1.0, C * p_binom_one))
    print(f"recurrence: R={R} C={C} k={k} modal_component={MODAL_COMPONENT} "
          f"p_binom_one={p_binom_one:.6g} p_union={p_union:.6g}")

    if p_union < 0.05:
        recur_verdict = (
            f"The recurrence of '{MODAL_COMPONENT}' as the top ablation component in {k}/{R} "
            f"independent seed/fold runs beats chance: under the null that each run's top component "
            f"is uniform over the C={C} candidates, P(a specific component recurs >= {k}/{R} times) = "
            f"{p_binom_one:.2e}, and even after a union-bound correction over all {C} candidate "
            f"components (P(ANY component recurs that often) <= {p_union:.2e}) this remains far below "
            f"0.05. The z>{NOTABLE_Z} threshold itself is not special on its own -- the shuffle null "
            f"shows {null_notable_mean:.2f} components cross z>{NOTABLE_Z} by chance on average "
            f"(null_p95={null_notable_p95:.2f}) -- but the SAME component winning in {k}/{R} runs is "
            f"not a fluke of that arbitrary threshold."
        )
    else:
        recur_verdict = (
            f"The recurrence of '{MODAL_COMPONENT}' as the top ablation component in {k}/{R} "
            f"independent seed/fold runs does NOT clearly beat chance: p_binom_one={p_binom_one:.3g}, "
            f"p_union={p_union:.3g} (>= 0.05 after correcting for {C} candidate components). Combined "
            f"with the null-shuffle finding that {null_notable_mean:.2f} components cross z>"
            f"{NOTABLE_Z} by chance alone on average (null_p95={null_notable_p95:.2f}), the "
            f"'{MODAL_COMPONENT}' claim should be reported as suggestive, not established."
        )
    print(recur_verdict)

    ablation_null_results = {
        "notable_z_threshold": NOTABLE_Z,
        "n_shuffles": S_ABL,
        "null_notable_count_mean": null_notable_mean,
        "null_notable_count_p95": null_notable_p95,
        "observed_n_notable_reproduced": n_notable_obs,
        "recurrence": {
            "R": R, "C": C, "k": k, "modal_component": MODAL_COMPONENT,
            "p_binom_one": p_binom_one, "p_union": p_union,
        },
        "verdict": recur_verdict,
    }
    with open(os.path.join(MET, "xai_ablation_null.json"), "w") as f:
        json.dump(ablation_null_results, f, indent=2)
    print("saved xai_ablation_null.json")

    data_vol.commit()
