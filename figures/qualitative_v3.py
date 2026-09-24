"""Stage 6: qualitative evidence.
  (a) Fig C: RF per-pixel probability vs masks_v2 ground truth on held-out TEST chips
      (the 'does it work spatially' gut-check; no U-Net needed, RF is per-pixel).
  (b) Kenya + Egypt national suitability maps (data-starved: proof-of-behaviour, NOT ROC).
Uses the deployed R model by default.
"""
import os, sys, traceback
import numpy as np, pandas as pd, rasterio

WS = os.path.expanduser("~/Solar_Workspace"); os.chdir(WS)
sys.path.insert(0, os.path.join(WS, "Solar-Siting"))
import results_util_v3 as R
import joblib

PCH = "Data/aef_solar_chips/positives/chips"
PMK = "Data/aef_solar_chips/positives/masks_v2"
MODEL = "Solar-Siting/artifacts/paper_v3/models/rf30_v3_R.joblib"
FIGDIR = "Solar-Siting/artifacts/paper_v3/figures/qualitative"
N_CHIPS = 8


def norm(a):
    lo, hi = np.nanpercentile(a, [2, 98])
    return np.clip((a - lo) / (hi - lo + 1e-9), 0, 1)


def fig_c():
    if os.path.exists(os.path.join(WS, f"{FIGDIR}/figC_chip_prob_vs_mask.png")):
        print("skip Fig C (exists — resume)"); return
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    rf = joblib.load(MODEL)
    sp = pd.read_csv("Solar-Siting/artifacts/splits/emb_search_split_v2.csv")
    test = sp[sp.split == "test"]
    picks = []
    for chip in test.chip:
        mp = os.path.join(PMK, chip)
        if not os.path.exists(mp):
            continue
        with rasterio.open(mp) as s:
            m = s.read(1)
        cov = m.mean()
        if 0.02 < cov < 0.6:              # visible but not whole-frame solar
            picks.append(chip)
        if len(picks) >= N_CHIPS:
            break
    if not picks:
        print("no suitable test chips for Fig C"); return
    fig, axes = plt.subplots(len(picks), 3, figsize=(9, 3 * len(picks)))
    if len(picks) == 1:
        axes = axes[None, :]
    for r, chip in enumerate(picks):
        with rasterio.open(os.path.join(PCH, chip)) as s:
            a = s.read().astype(np.float32)           # (64,256,256)
        with rasterio.open(os.path.join(PMK, chip)) as s:
            m = s.read(1)
        H, W = a.shape[1:]
        X = a.reshape(64, -1).T
        ok = np.isfinite(X).all(1)
        p = np.full(len(X), np.nan)
        p[ok] = rf.predict_proba(X[ok])[:, 1]
        pm = p.reshape(H, W)
        rgb = np.dstack([norm(a[0]), norm(a[1]), norm(a[2])])
        axes[r, 0].imshow(rgb); axes[r, 0].set_ylabel(chip, fontsize=7)
        axes[r, 1].imshow(pm, cmap="magma", vmin=0, vmax=1)
        axes[r, 2].imshow(m, cmap="Greens", vmin=0, vmax=1)
        for c in range(3):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
    for c, t in enumerate(["AEF embedding (dims 0-2)", "RF P(solar)", "masks_v2 ground truth"]):
        axes[0, c].set_title(t, fontsize=10)
    os.makedirs(os.path.join(WS, FIGDIR), exist_ok=True)
    out = f"{FIGDIR}/figC_chip_prob_vs_mask.png"
    fig.savefig(os.path.join(WS, out), dpi=120, bbox_inches="tight"); plt.close(fig)
    R.append_result("qualitative", "rf", "R", "test-chips", "figC", 1, figure_path=out,
                    notes=f"{len(picks)} chips")
    print(f"saved {out}", flush=True)


def lowdata_maps():
    from run_country_rf import run_country
    for lsib, iso3 in [("Kenya", "KEN"), ("Egypt", "EGY")]:
        prefix = f"Solar-Siting/artifacts/paper_v3/scores/country/{lsib.lower()}_R_2019_qualitative"
        if os.path.exists(os.path.join(WS, prefix + "_scores.npz")):
            print(f"skip qualitative {lsib} (exists — resume)"); continue
        try:
            m = run_country(lsib, year=2019, min_install=2021, model_path=MODEL,
                            out_prefix=prefix, max_sites=3000)
            R.append_result("qualitative", "rf", "R", iso3, "roc", m["ROC"], n_pos=m["n_sites"],
                            figure_path=prefix + "_fig.png", aef_year=2019,
                            notes="LOW-DATA qualitative — ROC not reliable")
            print(f"### qualitative map {lsib}: n_sites={m['n_sites']} (ROC {m['ROC']:.2f}, noisy)", flush=True)
        except Exception:
            traceback.print_exc(); R.log_stage(f"qualitative/{iso3}", prefix, "fail")


if __name__ == "__main__":
    fig_c()
    lowdata_maps()
