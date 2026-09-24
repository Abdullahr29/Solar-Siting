# MCDA baseline — source verification sheet

Purpose: the GIS-MCDA baseline (`mcda_benchmark.py`) is the paper's SOTA comparison ("RF beats
hand-tuned MCDA"). Every weight, criterion, normalisation choice, and land-cover score must trace to
a **verified published source**. Goal is NOT to compute our own AHP weights — it is to compare against
values already published in the literature.

Populated 2026-08-19 from 5 candidate papers Abdullah supplied (`~/Downloads/mcda_papers/`).

---

## 0. Paper triage — which of the 5 carry usable siting weights

| Paper | Usable for solar SITING weights? | Why |
|-------|----------------------------------|-----|
| **Richards et al. 2025** (Environ. Sci. Pollut. Res. 32:2007–2035, doi 10.1007/s11356-024-35669-6) | ✅ **PRIMARY** | Full AHP-MCDM utility-scale solar siting (Jamaica), CR-validated. Gives weight vector, reclass breakpoints, exclusion buffers, literature-frequency. |
| **Chen & Jong 2024** (IEEE Access 12:143458, doi 10.1109/ACCESS.2024.3461948) | ✅ **SECONDARY** | Systematic review of 47 SES criteria ranked by publication frequency (2020–2023) + a concrete weight vector (Table 6). |
| Matulaitis et al. 2016 (Solar Energy Mat. & Solar Cells 156:122) | ❌ | Ranks PV *support-policy schemes* (FiT/net-metering) via ELECTRE III with **equal weights**; criteria are NPV/IRR/PBT/CO₂/CFS. Not siting. |
| Kozlov & Sałabun 2021 (Procedia CS 192:4913) | ❌ | Ranks solar *panels* (products) by electrical specs (V, A, W, price, efficiency, area). Manual weights all 0.2. Not siting. |
| Mathebula & Mbuli 2025 (Energies 18:3478) | ❌ | TOPSIS-in-power-systems review; criteria are power-engineering indices (voltage deviation, congestion, losses…). Not siting. |

---

## 1. Richards et al. 2025 — PRIMARY source (AHP-MCDM, CR < 0.1)

### 1a. Weight vector — Table 5, p. 2016 ("Suitability model with weights according to category")

**Category (Level-2) weights** — note how strongly ECONOMIC dominates:
| Category | Weight |
|----------|--------|
| Economic | **52.53 %** |
| Technical | 30.88 % |
| Social & safety | 9.47 % |
| Environmental | 7.12 % |

**Sub-criterion (Level-3) GLOBAL weights** (category × local; they reconcile to the category totals):
| Criterion | Global weight | Category |
|-----------|--------------|----------|
| Distance to roads & highways | **0.384** | Economic |
| Distance to transmission lines | 0.099 | Economic |
| Slope | 0.043 | Economic |
| Distance to sensitive sites | 0.036 | Environmental |
| Distance to protected areas | 0.036 | Environmental |
| Distance to land use [residential] | 0.079 | Social |
| Elevation | 0.016 | Social |
| Solar radiation (parent) | 0.014 | Technical |
| — GHI | 0.025 | Technical |
| — PVOUT | 0.018 | Technical |
| — GTI | 0.012 | Technical |
| — OPTA | (n/a) | Technical |
| Aspect (north @ 20°) | 0.054 | Technical |
| Air temperature | 0.060 | Technical |

> **Headline finding for us:** in a CR-validated AHP, **GHI is weighted only ~0.025 (2.5 %)** — because
> within-region irradiance is fairly uniform — while **road proximity dominates at 0.384**. This is the
> OPPOSITE of our current benchmark (GHI 0.35, road 0.10). The economic/technical sub-weights don't fully
> reconcile to 30.88 % as printed (economic/environmental/social blocks do) — minor table inconsistency,
> flagged; the relative ordering is unambiguous.

### 1b. Reclassification breakpoints — Table 6, p. 2017 (most / moderately / marginally / not suitable)

| Criterion | Most (3) | Moderately (2) | Marginally (1) | Not (0) |
|-----------|----------|----------------|----------------|---------|
| Dist. to roads (m) | 100–4000 | 4000–6000 | 6000–8000 | 8000–10000 |
| Dist. to transmission (m) | <4000 | 4000–6000 | 6000–8000 | 8000–10000 |
| Slope (°) | 0–5.5 | 5.5–7.5 | 7.5–10.5 | 10.5–21.91 |
| Dist. to sensitive sites (m) | 10000–6000 | 6000–4000 | 4000–2000 | 2000–1000 |
| Dist. to protected areas (m) | 10000–6000 | 6000–4000 | 4000–2000 | 2000–1000 |
| Land-use proximity (m) | 1000–4000 | 4000–6000 | 6000–8000 | 8000–10000 |
| Elevation (m) | 30–300 | 300–500 | 500–800 | 800–2126 |
| GHI | 21.25–18.50 | 18.00–17 | 17.06–15.06 | <15.06 |
| PVOUT | 4.85–3.50 | 3.50–3.00 | 3.00–2.85 | <2.85 |
| GTI | 6.31–4.00 | 4.00–3.00 | 3.00–1.92 | 1.92–0 |
| OPTA | 20.00–16.00 | 16.00–15.00 | 15.00–14.00 | <14 |
| Aspect | North/NE | North-west | East/West | SE/SW/S |
| Air temp (°C) | 4–22.7 | 22.7–25.7 | 25.7–27.8 | >27.8 |

> A published, expert-derived reclassification scheme — an alternative to our country-relative 2–98 pct
> min-max. Note the roads scheme EXCLUDES <100 m (too close) and rewards 100–4000 m.

### 1c. Constraint exclusion buffers — Table 4, p. 2015

| Excluded feature | Buffer (m) |
|------------------|-----------|
| Buildings | 1000 |
| Roads | 500 |
| Railways | 500 |
| Waterways | 200 |
| Water bodies | 200 |
| Land uses | 1000 |
| Special sites | 1000 |
| Protected areas | 1000 |

### 1d. Literature feature-frequency — Table 2, p. 2012 (# of prior studies using each; "raw count")

Solar radiation/GHI 10 · Road access 10 · Slope 10 · Grid connection 9 · Air temperature 5 ·
Minimum suitable area 4 · Equivalent sun hours 3 · Elevation 2 · Soil 2 · Constraint area 2 ·
Land cover 1 · Diffuse irradiation 1 · PVOUT 0. (GHI, roads, slope, grid = near-universal.)

---

## 2. Chen & Jong 2024 — SECONDARY source (systematic review, 47 criteria)

### 2a. Criteria ranked by # of high-impact publications 2020–2023 — Table 3 / Fig 16 / pp. 143471–473

**Top criteria by citation frequency** (a literature-derived importance ranking):
| Criterion | # pubs | | Criterion | # pubs |
|-----------|--------|-|-----------|--------|
| Solar radiation / GHI | **21** | | Land cover | 14 |
| Proximity to power networks (grid) | **19** | | Aspect | 10 |
| Proximity to roads/railways | **19** | | Dist. to water bodies | 10 |
| Proximity to urban/residential | 17 | | Dist. to protected areas | 10 |
| Slope | 17 | | Elevation | 9 |
| Air temperature | 15 | | Construction/maint. costs | 7 |
| Population density | 15 | | Relative humidity | 9 |

> Confirms our 5 criteria (GHI, slope, grid, road, land-cover) are all top-tier, AND that population
> density, air temperature, aspect, distance-to-protected are commonly-used criteria we currently omit.

### 2b. Weight vector — Table 6, p. 143475 (fuzzy median weights, from their ref [52]; Sarawak)

| Criterion | Weight | Attribute |
|-----------|--------|-----------|
| GHI (kWh/m²/yr) | 12.73 | Benefit |
| Distance from protected areas | 11.53 | Benefit |
| Temperature (°C) | 11.26 | Cost |
| Proximity to power networks (grid, km) | 10.99 | Cost |
| Elevation (m) | 10.55 | Benefit |
| Slope (°) | 10.43 | Cost |
| Population density (ppl/km²) | 10.22 | Benefit |
| Proximity to residential areas (km) | 9.66 | Cost |

> A roughly **FLAT** weighting (all ~10–13 %) — a useful contrast to Richards' economic-dominant vector.
> (Weights sum to ~87, not 100 — they are medians of triangular fuzzy weights; use the relative spread.)

---

## 3. Our CURRENT implementation (`mcda_benchmark.py`) — for comparison

- **Criteria (5):** GHI (benefit), slope (cost), grid_dist (cost), road_dist (cost), land-cover class score.
- **Weight sets:** balanced `GHI .35 / slope .20 / grid .15 / road .10 / land .20`; irradiance `.45/.20/.15/.10/.10`; infrastructure `.30/.15/.25/.15/.15`.
- **Normalisation:** min–max, 2–98 pct of each country's random land.
- **Exclusions:** WDPA, WorldCover water(80)/wetland(90).
- **Land-cover lookup:** tree .3 / shrub .8 / grass .9 / crop .6 / built .1 / bare 1.0 / snow .2 / water 0 / wetland 0 / mangrove 0 / moss .5.

## 4. Gap analysis — our weights vs the literature

1. **We over-weight irradiance.** GHI = our top weight (.35–.45) vs Richards 0.025 and Chen 12.7 % (of a flat
   spread). Rationale in both papers: within-country GHI varies little, so it discriminates weakly — exactly
   the pattern our own covariate analysis found (irradiance weak in single-country runs).
2. **We under-weight infrastructure.** Road+grid = our .25 vs Richards 0.48 (roads alone 0.384) — the single
   biggest driver in the CR-validated AHP.
3. **Criteria-set mismatch:** neither source uses a **land-cover-suitability weighted criterion** — both handle
   land via *exclusion buffers* (Richards Table 4) or a residential-proximity criterion. Our weighted land-cover
   score is a design choice not mirrored in these sources.
4. **Criteria we omit that the literature weights:** aspect, elevation, air temperature, population density,
   distance-to-protected/sensitive.

## 5. Proposed finalised weight band for the benchmark (literature-faithful)

Map published vectors onto our 5 criteria [GHI, slope, grid_dist, road_dist, land], renormalised to sum 1:

| Vector | Source | GHI | slope | grid | road | land | notes |
|--------|--------|-----|-------|------|------|------|-------|
| **infrastructure-heavy** | Richards Table 5 (road .384 / grid(trans) .099 / slope .043 / GHI .025) | 0.045 | 0.078 | 0.180 | 0.697 | — | land via exclusion only |
| **balanced** | Chen Table 6 (GHI 12.73 / slope 10.43 / grid 10.99 / residential→road 9.66) | 0.29 | 0.24 | 0.25 | 0.22 | — | flat |
| **equal** | Richards Scenario-4 / Matulaitis (risk-averse baseline) | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 | |

> Claim becomes: **RF beats MCDA across the published weight spectrum** (economic-heavy → balanced → equal),
> not against our own invented weights. Much stronger and reviewer-proof.

## 6. Open decisions (need Abdullah)

- [ ] **Land-cover:** keep it as a weighted criterion (our design), or move to exclusion-only (literature-standard,
      Richards Table 4) and report both for robustness? (Recommend: report both.)
- [ ] **Reclassification:** keep our country-relative 2–98 pct min-max, or adopt Richards' fixed expert breakpoints
      (Table 6)? (Recommend: keep min-max as primary — country-relative is fairer across our 4 countries — and cite
      Richards' scheme as an appendix robustness check.)
- [ ] **Criteria set:** add air-temperature and/or aspect (both in our covariate stack already) to better match the
      literature, or keep the lean 5-criteria MCDA? (Recommend: keep lean 5 for the headline; note omissions.)
- [ ] Lock the vectors in Section 5, then re-run `mcda_benchmark.py` against the v2/R model (single GEE pass) → Fig A.
