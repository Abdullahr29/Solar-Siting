# Solar-Siting Paper — Results & Methodology Compendium

**Purpose.** A complete, self-contained record of every result and analysis produced since the
`masks_v2` retrain (the paper_v3 campaign), organised in the intended paper narrative. Each item has:
(i) a brief methodology, (ii) the numbers/tables, (iii) the figure(s), (iv) the finding. Use this to
draft the 4pp paper + appendix. **Nothing here should be forgotten** — everything is either main-text
or appendix. Sources: `PUBLISHING.md`, `artifacts/paper_v3/` (CSVs, figures, `results_master.csv`),
and memory (`paper-v3-campaign`, `tzsam-validation`, `tz-sam-methodology`, `loco-outcome`,
`mask-contamination`, `negative-set-remine`, `mcda-v3-benchmark`, `model2-*`, `yardstick-not-neutral`).

**One-line thesis.** *How much solar-siting signal is in a geospatial foundation-model embedding
(AlphaEarth), how best to extract it, how country-dependent it is — and how, regardless, it beats
published MCDA/GIS siting built on hand-picked data layers; released as a global suitability layer.*

**The engine (applies to everything below).** For each solar site we take the **AlphaEarth (AEF)
64-band per-pixel embedding of the surrounding area, sampled 2 years BEFORE installation** (to capture
the pre-existing land, not the panels), and learn to separate it from land that did not become solar.
The deployed model outputs **P(becomes NEW solar in ~2 yr)**. Reference model = **`rf30_v3_R`** (Random
Forest, 30 trees, `min_samples_leaf=1500`, class-balanced), trained on `masks_v2` positives + 13,440
decontaminated negatives.

---

## ✅ COMPLETENESS CHECKLIST (every result, where it goes)

| # | Result / analysis | Artifact | Paper? |
|---|---|---|---|
| 1 | Dataset curation: `masks_v2` contamination fix | memory `mask-contamination`; `masks_v2/` | Methods |
| 2 | Negatives: 13,440 decontaminated via TZ-SAM, audited | `negative_final_audit_*.csv` | Methods |
| 3 | TZ-SAM as independent negative-miner + validator (principled asymmetry) | `tz-sam-dataset`/`-methodology` | Methods |
| 4 | 8-method extraction bake-off (Model-1 → linear → non-linear) | `results_master.csv` (bakeoff); `figures/bakeoff/bakeoff_roc.png` | **Fig (D)** + table |
| 5 | Data-quality ablation (old vs masks_v2+decontam) | `results_master.csv` (data_ablation); `figures/data_ablation_old_vs_new.png` | Appendix |
| 6 | Per-country forward validation (2021+, dedup), 13 countries | `scores/country/country_summary.csv` | **Fig B** + table |
| 7 | LOCO spatial holdout (9 countries) | `results_master.csv` (loco); `figures/loco/` | **Fig B** / appendix |
| 8 | Temporal holdout (≤2021→2022+, leak-proof) | `results_master.csv` (temporal); `figures/temporal/` | **Fig B** (headline honest number) |
| 9 | TZ-SAM external validation (13 countries, independent) | `tzsam/tzsam_validation.csv` | **Fig B** + table |
| 10 | Bootstrap 95% CIs (inventory/tzsam/combined) | `tzsam/bootstrap_ci.csv` | **Fig B** error bars |
| 11 | Cross-country comparability (global-pooled ROC) | (computed) 0.868 / global 0.915 | Result sentence |
| 12 | Weak-country analysis + normalisation demo | `figures/weak_country_v3/` | **Fig B** inset / appendix |
| 13 | Covariate "embedding novelty" (13-country, +population) | `figA_combined/covariate_novelty/novelty_v3.csv` | **Fig B** panel |
| 14 | 64-dim permutation importance | `results_master.csv` (importance); `figures/emb_importance_R.png` | Appendix |
| 15 | Site-level GIS profiling (siting drivers) | `figA_combined/site_profile/site_profile_v3.csv` | Mechanism / appendix |
| 16 | RF vs published MCDA, 13 countries, paired CIs | `figA_combined/figA_bootstrap.csv` | **Fig A** (headline) |
| 17 | Expert-weights vs learned-importance | `figures/fig_e_expert_vs_learned.png` | **Fig A** panel |
| 18 | Qualitative chip: RF prob vs masks_v2 | `figures/qualitative/figC_chip_prob_vs_mask.png` | **Fig C** |
| 19 | Capacity-weighted validation | `figA_combined/capacity_weighted.csv` | Result sentence |
| 20 | Capacity coverage + per-size-bin ROC | `figures/capacity_coverage.png` | **Fig** (capacity) |
| 21 | Global suitable-land fraction + global-pooled ROC | `figA_combined/global_suitable.npz` | **Climate** headline |
| 22 | Grid-constrained realistic potential | `figA_combined/grid_constrained.npz` | **Climate** |
| 23 | Global suitability raster (dataset release) | `data_products/`, 100m preview | **Fig E** / release |
| 24 | Resolution ablation (native 10 m optimal) | memory `resolution-ablation` | Methods note / appendix |
| — | **TO GENERATE:** RF-vs-MCDA spatial map on one country (agreement/disagreement) | — | **Fig A** panel (see §6) |

---

## 1. Dataset curation

### 1a. Positives — the `masks_v2` rebuild (the correctness fix)
**Method.** Positives come from a global PV-facility inventory (140,945 sites). The original masks were
rasterised from `base_train` (2019–2024 installs only) which (a) **left ~76k operating 2017–2018 farms
unmasked** (labelled background — "operating solar as not-solar"; 32.5% of scenes, Mode A) and
(b) **dropped cap-filtered *future* solar** (22.7% of scenes, Mode C — will-become-solar penalised as
background). **Fix (locked 2026-08-05): binary rebuild `masks_v2` — pixel = 1 iff covered by a solar
footprint whose `install_year > chip embedding_year`, else 0**, drawn from the FULL inventory. This
recovers all future solar, keeps existing solar as background (an operating farm is *not* a candidate
for NEW solar → correctly teaches "don't re-nominate built land"; its embedding still enters as
positive *context*). Deployed probability = **P(new solar in ~2 yr)**, not abstract suitability.
**Finding:** masks_v2 measurably improves the model (see §3b ablation: +0.027 pixel ROC).

### 1b. Negatives — 13,440 decontaminated true-negatives
**Method.** Globally random land ≥5 km from any solar site of any year (presumed non-suitable),
**decontaminated against TZ-SAM** (108,340 utility sites) + the inventory (audit vs 841,556 solar
sites: 0 chips contain solar; dropped 4 boundary breaches → final min-dist 4,988 m). New 80/20
scene-level split. **Caveat (report honestly):** the ≥5 km buffer × country-size geometry structurally
under-samples dense-solar countries — so per-pixel ROC is scored against a biased negative set and is
**not** the neutral referee (§3, §4 use country/forward validation as the honest metric).

### 1c. TZ-SAM — independent negative-miner AND validator (principled asymmetry)
**Method + reviewer defence.** TZ-SAM (TransitionZero Solar Asset Mapper) is an independent,
Sentinel-2-derived, utility-scale (>500 kW) global solar dataset (UNet detection + 400k manual
validations, ~1% FP; Eq.1 capacity model). We use it for **negatives (locations only → guarantees true
negatives)** and **validation (independent forward sites)** but **NOT positives** — positives need
reliable install *dates* (AEF sampled 2 yr pre-install) and matched footprint granularity, which
TZ-SAM's coarse date brackets + cluster geometry lack. *Framed as a strength: "we used an independent
solar dataset to guarantee negatives and to externally validate," not an omission.* Full methodology in
memory `tz-sam-methodology`.

---

## 2. Model bake-off — how to extract siting signal from the embedding

**Method.** All 8 extraction methods trained/evaluated on the SAME `masks_v2` 80/20 balanced pixel pool
(6.19M test-pos / 30.17M test-neg px), per-pixel ROC + PR. Ladder from naive → linear → non-linear.

| Method | test-pixel ROC | PR-AUC | note |
|---|---|---|---|
| **Model-1** mean-embedding cosine similarity | **0.702** | 0.318 | the "naive embedding" baseline |
| QDA | 0.854 | 0.521 | |
| Logistic probe | 0.866 | 0.551 | linear |
| MaxEnt (L1 logistic+quadratic, classic SDM) | 0.874 | 0.577 | **beats the standard species-distribution baseline** |
| Random Fourier Features (512) | 0.891 | 0.628 | |
| **RF (deployed)** | **0.898** | 0.641 | full-pool RF-R |
| RF (subsample, fairness ref) | 0.914 | 0.696 | |
| HistGB | 0.919 | 0.713 | |
| MLP | **0.924** | 0.728 | best on pixels |

**Findings.** (1) Naive mean-embedding similarity is weak (0.70) — motivates a *discriminative* model.
(2) Non-linear >> linear (RF/HistGB/MLP ~0.90–0.92 vs logistic 0.87). (3) **RF deployed** (not the
marginally-better MLP/HistGB) because it injects natively into Earth Engine (`rf_to_strings`) for the
global product; MLP's +0.026 pixel edge does not survive as the real-world referee (see `yardstick`).
(4) **MaxEnt — the standard SDM/GIS baseline — is beaten by RF** on the same features.
**Figure:** `figures/bakeoff/bakeoff_roc.png`. → **Fig D** (REO: the representation-probing core; CCAI: a "why naive fails" inset).

### 2b. Data-quality ablation (does masks_v2 + decontam-neg actually help?)
Old `rf30_final` vs new `rf30_v3_R` on the SAME masks_v2 pixel pool: **ROC 0.871 → 0.898 (+0.027),
PR 0.539 → 0.641 (+0.102).** Clear improvement. `figures/data_ablation_old_vs_new.png`. → Appendix.

### 2c. Resolution ablation (justifies native 10 m)
**AEF native 10 m is optimal — coarsening the embedding only degrades performance (and is slower for
point queries); we were always at native.** Justifies scoring at native 10 m throughout, and is the
reason the coarse 100 m global asset is a wrong-CRS *placeholder*, not a validated product (a trustworthy
coarse product needs re-validation on pooled embeddings). Memory `resolution-ablation`. → Methods note / appendix.

---

## 3. Validation of the RF (leakage-aware ladder)

Three tiers, each removing a leakage path. R vs D (de-confounded) is a wash (±0.01) → **R shipped**
("de-confounding doesn't change forward performance = a robustness result, not a defect").

### 3a. Per-country forward (2021+ installs, training-site dedup) — `country_summary.csv`
The map year is AEF-2019; validation sites installed ≥2021 (the +2 yr lead) did not exist in the
imagery. 13 countries, R scheme:

| | GRC | DEU | CHN | ESP | POL | JPN | IND | USA | ZAF | COL | PHL | MYS | CHL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ROC (R) | .895 | .850 | .876 | .821 | .793 | .897 | .827 | .883 | .847 | .957 | .871 | .793 | .934 |
| n_pos | 221 | 465 | 2978 | 397 | 190 | 1450 | 464 | 2993 | 28 | 23 | 22 | 32 | 81 |

### 3b. LOCO — leave-one-country-out spatial holdout (`loco`)
Model never sees the target country in training. R: DEU .790, ESP .826, POL .730, CHN .756,
**IND .662 (floor)**, BRA .821, ZAF .753, CHL .869, PHL .833. **Mean ~0.78.**
Figures: `figures/loco/`. → **Fig B** or appendix.

### 3c. Temporal holdout — train ≤2021 imagery, validate 2022+ (LEAK-PROOF, the honest headline)
CHN .831, USA .897, IND .858, ESP .845, TUR .796, BRA .874; **pooled global-2024 .829. Mean ~0.85.**
Figures: `figures/temporal/`. → **Fig B** (this is the number to lead validation with — no spatial or
temporal leakage).

**Finding (ties §3b+§3c together):** the leakage/optimism premium (in-sample − held-out) **tracks
country DIFFICULTY**: ~0.01 for easy countries, ~0.04–0.05 (spatial) / larger for hard, diverse ones
(India in-sample .827 → LOCO .662). Report the forward numbers honestly; the drop is expected and
interpretable, not a defect.

---

## 4. RF performance — country-dependence, external validation, visuals

### 4a. TZ-SAM external validation (independent, 1.4–6.5× more positives) — `tzsam_validation.csv`
Same model, same AEF-2019, TZ-SAM new sites (constructed_after≥2020, ≥1 km from training). ROCs track
the inventory (median |Δ|≈0.015; TZ slightly lower = honest, noisier detector):

| | GRC | DEU | CHN | ESP | POL | JPN | IND | USA | ZAF | COL | PHL | MYS | CHL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| inventory | .895 | .850 | .876 | .821 | .793 | .897 | .827 | .883 | .847 | .957 | .871 | .793 | .934 |
| TZ-SAM | .883 | .812 | .874 | .840 | .751 | .904 | .814 | .881 | .828 | .926 | .887 | .887 | .949 |
| **combined** | .886 | .837 | .872 | .847 | .768 | .902 | .830 | .884 | .857 | .937 | .871 | .891 | .942 |

**Findings.** Two independent datasets agree → not overfit to the supervisor's inventory. **Malaysia
rehabilitated** (0.79 on n=32 was small-n noise → 0.89); **Poland confirmed genuinely weak** (0.75 on
n=1042, robust). Combined-set n_pos: GRC 876, CHN 14062, IND 4386, USA 7564, ZAF 168, COL 116, PHL 73.

### 4b. Bootstrap 95% CIs (`bootstrap_ci.csv`)
TZ-SAM roughly HALVES the fragile countries' intervals: **S.Africa 0.167→0.070 wide, Philippines
0.133→0.065**; well-powered countries always tight (China ±0.006, USA ±0.007). → **Fig B** error bars.
Directly defuses the "only ~350 positives" scrutiny.

### 4c. Cross-country comparability (a strong, cheap validation)
**Global-pooled ROC (all 53k sites vs 104k land, one test) = 0.868 ≈ per-country mean 0.871.** Raw
RF-R scores ARE comparable across countries — a real site anywhere outranks random land anywhere ~87%.
On the truly-global sample (§7): **0.915**. Percentile-normalisation doesn't change it → it's a
per-country *readability* tool only. **Report as a headline: "a single pooled test across 13 countries
/ 3 continents, ROC 0.87 (0.92 globally)."**

### 4d. Weak-country analysis — `figures/weak_country_v3/`
Combined-ROC ranking: **Poland weakest (0.768)**, India 0.830, Germany 0.837; strongest Chile 0.942,
Colombia 0.937. **ROC-vs-data-volume Spearman −0.30** (China 27k pos → 0.872 vs Chile 544 → 0.942) ⇒
**weakness tracks DIFFICULTY, not data-starvation.** Normalisation demo on India (dimmest): raw land
squished at median 0.30 while sites sit at median 92nd percentile; percentile-normalising brightens the
map, **AUC identical 0.827** (rank-preserving → readability up, skill unchanged). This is the honest
framing for the Raw-release / percentile-in-discussion decision.
Figures: `india_normalization_heatmap.png` (physically spot-on — NW Rajasthan/Gujarat + S Deccan light
up, sites land in bright zones), `india_normalization_demo.png`. → few lines + **Fig B** inset.

### 4e. Covariate "embedding novelty" (13-country, WITH population) — `novelty_v3.csv`
Regress the RF score on GIS covariates over random land → residual = variance the embedding captures
that GIS layers cannot. Residual **0.12–0.33** (Malaysia 0.33, S.Africa 0.31, India 0.31 highest;
Colombia 0.12, Greece 0.15, China 0.16 lowest); R²_rf ≫ R²_lin (nonlinear). Population is a real
feature (China #2 driver, Malaysia #3). *Consistent with the original 7-country run (0.15–0.34).* This
is the "the embedding knows more than hand-picked layers" panel. → **Fig B** panel.

### 4f. 64-dim permutation importance
Diffuse (top-1 importance ~0.02; top dims A51/A40/A39) — no single embedding dimension dominates; the
signal is distributed. `figures/emb_importance_R.png`. → Appendix.

### 4g. Qualitative spatial check — `figures/qualitative/figC_chip_prob_vs_mask.png`
RF per-pixel probability heatmap vs `masks_v2` ground truth on held-out/forward chips (no U-Net needed —
RF is per-pixel). The "does it work spatially" gut-check. → **Fig C** (Abdullah must-have).

---

## 5. Capacity findings (TZ-SAM `capacity_mw` + footprints)

**Key physics (settles what's learnable).** TZ-SAM capacity = footprint area × modelled density
(Eq.1: `C = A·I·η·GCR·ILR`); measured density **median 55, area-weighted aggregate 39.4 MW/km²**,
CV 0.17, varies 2× by country / 1.7× by size (log-log capacity~area slope 0.919, R² 0.991). So capacity
≈ area × a physical constant → a "capacity-density surface" would just be suitability×40 (a relabeling,
**not** a contribution — rejected).

### 5a. Capacity-weighted validation — `capacity_weighted.csv`
Weighting each site by installed MW: global cap-weighted ROC **0.76** (vs 0.86 unweighted);
score~log(capacity) Spearman **−0.13** (mostly negative; India +0.16 exception). **Honest limitation:**
the model is tuned to the prevailing small-to-mid deployments; the largest farms sit on atypical remote
land and score lower.

### 5b. Coverage curve + per-size-bin ROC — `figures/capacity_coverage.png`
**Coverage (positive, decision-relevant):** the model's top **20% of land holds 55% of the world's
installed GW** (top 10% → 38%, 3.8×; top 30% → 67%). **Per-size-bin ROC (the limitation, quantified):**
monotonic **0.881 (<1 MW) → 0.876 → 0.848 → 0.828 → 0.724 (100+ MW, n=561, tight CI)**. One coherent
narrative: *a location model tuned to the dominant small-mid deployment mode; captures the majority of
installed capacity in its top-ranked land; discrimination declines for the largest, atypically-sited
farms.* → **Capacity figure** (good paper piece).

### 5c. Realized-generation validation (Song et al. 2026 dataset) — `figures/generation_analysis.png`
Independent per-facility **modelled annual generation** (plane-of-array, irradiance-driven; Song, Yin,
Muller et al., *Nature Sustainability* 2026, `s41893-026-01836-5`), 2019–2023, joined by coordinates to
8,090 of our validation sites across 13 countries (`artifacts/paper_v3/generation/`). **Data handling
sanity-checked against the paper's own headline**: we reproduce global 2023 aerosol loss **5.52 %
(paper ~5.8 %), 110.9 TWh (paper 111 TWh)**, China **7.17 % (paper 7.7 %)** — confirms correct column
interpretation. Median realized yield **142.6 kWh m⁻² yr⁻¹** (of array; CV 13 %).

Two findings (`gen_join.py`, `gen_figure.py`):
- **Suitability is orthogonal to energy yield (the interesting one).** Spearman ρ(score, yield) = **−0.06**
  pooled, **+0.016** within-country (climate-adjusted); ρ(score, GSA-GHI) = **−0.067**. Yield genuinely
  tracks resource (ρ(GHI, yield) = +0.42), so the near-zero score↔yield relation means **the model
  encodes land/infrastructure developability, not the irradiance field** — it is *not* covertly a sunny-
  latitude map, and contributes information complementary to resource layers and to the GHI-weighted MCDA
  baselines. **Directly reinforces §6a** (resource is a weak discriminator; RF beats resource-weighted MCDA).
- **Generation-weighted validation holds.** Weighting real sites by measured annual generation leaves
  discrimination essentially unchanged (site-vs-random AUC **0.871 gen-weighted vs 0.875 unweighted**) —
  the model prioritizes genuinely productive solar, not marginal installs. **Generation concentration:**
  the top **10 % most-suitable land hosts 56 %** of realized generation, top **20 % → 78 %** (vs 1:1 for
  random land). Per-country tracks our ROC ranking (USA 63/80 %, China 55/80 %, India 72/80 %, Japan
  57/79 %; weak in high-baseline Poland 21/28 % and small-n South Africa). Table
  `gen_coverage_by_country.csv`.
- **Energy calibration (supporting number, not a headline).** Observed land yield ≈ **71 GWh km⁻² yr⁻¹**
  (median array yield × GCR 0.5) — an empirical land→energy constant. *Caveat, consistent with our earlier
  stance:* land is not the binding constraint, so we do **not** multiply the raw suitable-area fraction into
  a giant potential; the constant is available for bounded, grid-constrained statements only.

---

## 6. Improvement over SOTA — RF vs published MCDA

### 6a. Feature analysis / mechanism — why RF beats MCDA
**Site profiling** (`site_profile_v3.csv`): real solar siting is **infrastructure/terrain-driven** —
flat slope (universal), near roads, near grid, high population density — while **solar resource
(GHI/PVOUT/GTI) is a weak, inconsistent discriminator** (China/USA/S.Africa sites have *lower* GHI than
random land; only Colombia has a strong GHI signal — and Colombia is where MCDA comes closest). This is
the MECHANISM for RF>MCDA.
**PVOUT — the one covariate to hone in on (authoritative-product disparity)** (`pvout_check.py`): the
World Bank / Solargis *Global PV Power Potential* study (2020) — the field's canonical potential product —
defines practical potential as **PVOUT × exclusion masks**, i.e. it is *anchored on* long-term PVOUT.
That layer is the SAME Solargis LTAy PVOUT we already carry as a covariate (`World_PVOUT_LTAy`). Directly:
**corr(RF suitability, PVOUT) = −0.07** (n=9,344; matches GHI −0.07 and generation-yield +0.02), and
**PVOUT alone separates real sites from random land at AUC 0.49 — chance** (median PVOUT identical, 4≈4
kWh kWp⁻¹ d⁻¹), vs **our RF 0.87**. Per-country it is often *inverted*: China 0.32, USA 0.43, S.Africa
0.37 (<0.5 → real solar on LOWER-PVOUT land than national average; highest-PVOUT deserts lack grid/demand);
Colombia 0.86 the lone exception. → the single cleanest "current products vs ours" contrast: the metric the
authoritative potential map is built on has **no power to predict actual deployment**; our learned map is
orthogonal to it and captures the developability axis instead. (WB masks archived on JASMIN
`Data/external/gsa_pv_potential/masks/` for an optional exclusion-overlap follow-up; not needed for now.)
**Expert-weights vs learned-importance** (`fig_e_expert_vs_learned.png`): empirical per-criterion
discrimination is **even (0.09–0.14 across all criteria)**; the published expert vectors are **lopsided**
(Richards puts 0.585 on road = 5× the empirical 0.121; Chen puts 0.192 on GHI = 2× empirical 0.092).
Neither captures the balance a learned model does. → **Fig A** panel.

### 6b. RF vs MCDA, 13 countries, paired bootstrap — `figA_bootstrap.csv` (THE headline)
Published weights (Richards 2025 AHP + Chen 2024) × aggregators (WLC + TOPSIS) × land-cover both ways,
country-relative min-max normalisation, local-UTM distances, full criteria incl. population, scored on
the SAME combined PV_Facility+TZ-SAM positives vs land. **RF beats every configuration in ALL 13
countries; every combined-set gap is statistically significant (paired bootstrap, lower CI > 0):**

| Country | RF | best MCDA | gap [95% CI] |
|---|---|---|---|
| China | 0.872 | 0.615 | **+0.257** [.248,.265] |
| Chile | 0.942 | 0.725 | +0.217 [.198,.236] |
| Greece | 0.886 | 0.683 | +0.203 [.187,.219] |
| USA | 0.884 | 0.741 | +0.143 |
| Spain | 0.847 | 0.720 | +0.127 |
| Japan | 0.902 | 0.779 | +0.122 |
| India | 0.830 | 0.712 | +0.118 |
| Malaysia | 0.891 | 0.788 | +0.102 |
| Poland | 0.768 | 0.671 | +0.097 |
| Germany | 0.837 | 0.740 | +0.097 |
| Philippines | 0.871 | 0.831 | +0.039 |
| Colombia | 0.937 | 0.905 | +0.031 |
| South Africa | 0.857 | 0.828 | +0.029 |

**Does MCDA fail where RF is good? YES — the vivid case is China:** the Chen/equal weight vectors score
**< 0.5 (anti-rank)** in China while RF is 0.876 — the hand-tuned expert weights actively *mis-rank*
Chinese sites (over-weighting irradiance, which anti-correlates there). On inventory-only, the S.Africa
& Philippines gaps were NOT significant (small-n) — pooling the independent TZ-SAM positives is what
makes them significant. → **Fig A** (headline).

### 6c. Spatial RF-vs-MCDA maps — DONE (`figures/{china,colombia}_rf_vs_mcda_raster.png`)
A matched **agreement/disagreement pair**, both accurate **1 km LOCAL-raster MCDA** built to match the
benchmark rules exactly (published weights [Richards for China, Chen incl. population for Colombia],
country-relative 2–98 pct min-max, **equal-area EPSG:6933 road/grid distances**, **water/wetland +
WDPA protected exclusion**). RF panel = the 100 m placeholder asset pulled at 1 km (tiled getDownloadURL,
no re-classify). Sites = our PV_Facility (all yrs) + non-duplicate TZ-SAM new, deduped. Latitude-corrected
aspect. Caveat: slope/aspect from 1 km GSA elevation (minor weight) vs the 30 m DEM.
- **China (disagreement, gap +0.26):** disagreement panel systematically **blue west / red east** — MCDA
  over-rates the empty western deserts and under-rates the eastern belt where solar + RF + the 56,555 real
  sites concentrate. MCDA misplaces suitability to where the sun is, not where solar goes.
- **Colombia (agreement, gap +0.03):** RF, Chen-MCDA and sites all agree on the Andean belt; disagreement
  is only local terrain noise — because siting there is genuinely GHI-driven, the expert prior holds.
Scripts: `mcda_raster.py` (build, run via `mcda_raster.sbatch`), `render_raster.py` (instant re-render from
`{slug}_raster_data.npz` cache; arg2 = #sites shown). Note: run via SLURM, not nohup/ssh (launch fragility).

---

## 7. Climate-relevant metrics + global dataset

### 7a. Global suitable-land fraction + global-pooled ROC — `global_suitable.npz`
Point-sampling (client-side uniform-sphere land points + reduceRegions — `sample()` over the global
region times out in restricted mode). 40,784 global land points, 10,000 global TZ-SAM sites, threshold
= 10th pct of real-site scores (0.436). **Global suitable-land fraction = 23.6%**; **global-pooled ROC
= 0.915** (strongest single validation — global land includes obviously-unsuitable terrain that scores
low, cleanly separated). 32% averaged over the 13 (solar-heavy, biased high) vs 23.6% true-global.

### 7b. Capacity potential — the honest bracket
23.6% × ~130M km² land × 39.4 MW/km² ≈ **~1,200 TW** vs **1.33 TW installed = ~900×**. So the raw number
is effectively unbounded ⇒ **land suitability is NOT the binding constraint (≈3 orders of magnitude);
grid, economics and policy are.**

### 7c. Grid-constrained realistic potential — `grid_constrained.npz`
Literature cutoffs: **slope ≤5°** (NREL/IRENA), **grid ≤10 km**, **WDPA excluded**, **land cover =
shrub/grass/bare** (cropland reported separately; road omitted — no global roads dataset, grid is the
dominant infra constraint). **Constrained realistic fraction = 5.6% of global land (10.9% WITH
cropland — record BOTH).** Per-gate REAL-SITE pass rates validate the cutoffs: RF-suitable 90%,
grid ≤10 km **98.7%**, non-protected 96.5% — but shrub/grass/bare only 45% (**real solar heavily uses
CROPLAND** → 10.9% is the empirically honest number; excluding it is a food-security scenario).
**Number sequence: ~900× (all suitable) → ~217× (grid/terrain/protected-constrained, 100% coverage) →
~4–11× (applying a standard 1–5% land-use-availability factor).** *This is where the "~4×" instinct
legitimately lands — report as a bracket, not a point.*

### 7d. Global suitability raster — the dataset release
Released global per-pixel solar-suitability layer, model RF-R on 2025 AEF. 100 m preview
`figures/global_100m_preview.png` (⚠️ EPSG:4326 placeholder, not for quantitative use). 10 m native-UTM
global export is quota-gated (GEE noncommercial); scoped/coarse is the near-term option. CEDA DOI +
GEE asset planned. → **Fig E** (CCAI dataset teaser).

---

## Figure-packaging suggestions (4pp → ~4–5 tight multi-panels; overflow → appendix)

- **Fig 1 — METHODOLOGY (the engine, Abdullah's ask).** A single visual of the pipeline: *a solar site
  → step back 2 years → extract the 64-band AlphaEarth embedding of the surrounding area → learn to
  discriminate it from non-solar land → per-pixel P(new solar).* Show a real chip: RGB/site outline →
  the 64-band embedding (e.g. a few channels or a PCA-to-RGB of the embedding) → the RF → probability
  heatmap. This is the core engine behind every result; it earns a schematic. *(Diagram, to design.)*
- **Fig A — RF beats SOTA MCDA.** Panel (i) per-country RF-vs-best-MCDA gap with paired CIs (all 13);
  panel (ii) expert-weights-vs-empirical-importance (`fig_e`); panel (iii) [to generate] the China
  spatial agreement/disagreement map (§6c). *Headline.*
- **Fig B — country-dependence & validation.** Panel (i) per-country ROC with TZ-SAM + bootstrap CIs
  (inventory/TZ-SAM/combined); panel (ii) covariate "embedding novelty" bars; panel (iii) weak-country /
  difficulty (ROC vs data-volume) + the India normalisation inset. Temporal & LOCO as an inset or
  appendix table.
- **Fig C — qualitative spatial** (`figC_chip_prob_vs_mask.png`): RF probability vs masks_v2.
- **Fig D — extraction bake-off** (`bakeoff_roc.png`): Model-1 naive (0.70) → linear → RF/MLP (~0.90+).
  (REO lead figure; CCAI inset.)
- **Fig E — global suitability raster + climate** (CCAI): the released map + the suitable-fraction /
  land-not-the-constraint result + the capacity coverage curve.
- **Capacity coverage** (`capacity_coverage.png`): can live in Fig E or its own small figure.

**Appendix candidates:** data-ablation, full LOCO/temporal tables, 64-dim importance, site-profiling
tables, per-country MCDA configs, grid-constrained gate table, methodology/cutoff documentation.

## Open items before submission
- Generate the **methodology schematic (Fig 1)** and the **China RF-vs-MCDA spatial map (§6c)**.
- Venue/format: CCAI (Aug 29) vs REO-2 (Sep 2); 4pp, double-blind, non-archival (see `PUBLISHING.md`).
- Decide cropland in/out for the headline realistic fraction (record both: 5.6% / 10.9%).
- Global raster release (CEDA DOI / GEE asset) — release track, not paper-blocking.
