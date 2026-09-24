"""Is Solargis long-term PVOUT (= World Bank study's core metric) informative for our model?
It IS the World_PVOUT_LTAy GSA covariate we already hold. Quantify:
  - corr(RF suitability, PVOUT) at real sites
  - PVOUT at real sites vs random land (does resource discriminate? -> AUC)
across the 13 validation countries (reusing saved site/rand coords)."""
import os, glob
import numpy as np
os.environ["PROJ_DATA"] = os.path.expanduser("~/Solar_Workspace/envs/env_solar/share/proj")
WS = os.path.expanduser("~/Solar_Workspace/Solar-Siting"); os.chdir(WS)
import covariate_append_manual as cam
from covariate_append_manual import ensure_gsa_tif, sample_raster
cam.CD = os.path.join(WS, "covariate_data")
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

pvout_tif = ensure_gsa_tif("World_PVOUT_GISdata_LTAy")
print("PVOUT layer:", os.path.basename(pvout_tif))

site_s_all, site_pv_all, rand_pv_all, ctry_all = [], [], [], []
rows = []
for f in sorted(glob.glob("artifacts/paper_v3/scores/country/*_R_2019_scores.npz")):
    c = os.path.basename(f).replace("_R_2019_scores.npz", "")
    z = np.load(f)
    ss, sxy, rxy = z["site_s"], z["site_xy"], z["rand_xy"]
    m = min(len(ss), len(sxy)); ss, sxy = ss[:m], sxy[:m]
    s_pv = sample_raster(pvout_tif, sxy[:, 0], sxy[:, 1])
    r_pv = sample_raster(pvout_tif, rxy[:, 0], rxy[:, 1])
    ok = np.isfinite(s_pv); rok = np.isfinite(r_pv)
    if ok.sum() < 20:
        continue
    rho = spearmanr(ss[ok], s_pv[ok]).correlation
    # does PVOUT alone separate sites from random land?
    y = np.r_[np.ones(ok.sum()), np.zeros(rok.sum())]
    auc_pv = roc_auc_score(y, np.r_[s_pv[ok], r_pv[rok]])
    rows.append((c, int(ok.sum()), round(np.median(s_pv[ok]), 0), round(np.median(r_pv[rok]), 0),
                 round(rho, 3), round(auc_pv, 3)))
    site_s_all.append(ss[ok]); site_pv_all.append(s_pv[ok]); rand_pv_all.append(r_pv[rok]); ctry_all += [c]*ok.sum()

print(f"\n{'country':14s}{'n':>6}{'sitePV':>8}{'randPV':>8}{'rho(RF,PV)':>12}{'AUC_PVonly':>12}")
for r in rows:
    print(f"{r[0]:14s}{r[1]:6d}{r[2]:8.0f}{r[3]:8.0f}{r[4]:12.3f}{r[5]:12.3f}")

ss = np.concatenate(site_s_all); spv = np.concatenate(site_pv_all); rpv = np.concatenate(rand_pv_all)
print("\nPOOLED corr(RF suitability, PVOUT) :", round(spearmanr(ss, spv).correlation, 3), "n=", len(ss))
y = np.r_[np.ones(len(spv)), np.zeros(len(rpv))]
print("POOLED PVOUT-only site-vs-land AUC :", round(roc_auc_score(y, np.r_[spv, rpv]), 3),
      "  (vs RF's ~0.87)")
print("median PVOUT  sites=%.0f  random-land=%.0f  (kWh/kWp/yr)" % (np.median(spv), np.median(rpv)))
