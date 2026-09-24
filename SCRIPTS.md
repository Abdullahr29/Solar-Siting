# SCRIPTS.md — Solar-Siting code map

Auto-generated map of every script after the 2026-09-23 reorganisation. Scripts are grouped by function into subfolders; the 11 shared-library modules stay at the Solar-Siting root.

**Running scripts:** heavy jobs go through `jobs/*.sbatch` (each sets `PYTHONPATH` already). For an interactive run, first `source env.sh` (sets `PYTHONPATH` to the Solar-Siting root) so the shared-lib imports resolve, e.g. `source env.sh && python validate/final8_ladder.py --aggregate`.

**Key data & models** (all under `artifacts/`, preserved):
- `paper_v3/pools/` — training pools: `px_v3_*` (cap120) and `px_master_train.npz` (93 M-pos master).
- `paper_v3/pools_t3/` — `cap120_t3_train.npz`, validation embeddings (`val_emb/`), tree-count + LOCO outputs.
- `paper_v3/pools_t3/final8/` — **final 8-model head-to-head** (`final8_master.csv`, `models/`). Primary model = `models/cap120_ne15_t3.joblib`; see repo `IMPROVEMENTS.md` FINAL VERDICT + memory.
- `stats/px_temporal_le2021.npz` — temporal-holdout training pool.
- `artifacts/models/` — deployed models (e.g. `rf30_v3_R.joblib`).
- Canonical vectors at root: `global_pv_facility_inventory.gpkg`, `global_solar_ml_pipeline.gpkg`.

Companion docs: `PAPER_COMPENDIUM.md` (every result → output file) and repo-root `IMPROVEMENTS.md` (the model-improvement tracks + final verdict).

## `Solar-Siting/ (root)`
**Shared library modules** — imported by scripts across every folder (there are zero leaf-to-leaf imports; every local `import` targets one of these). Kept at the Solar-Siting root so `import <name>` resolves; most are also runnable as scripts.

| file | purpose |
|---|---|
| `run_country_rf.py` | Country-scale solar-suitability run with the GEE-injected random forest. |
| `results_util_v3.py` | Single source of truth for the paper_v3 campaign: append one row per (experiment, model, |
| `country_runs_v3.py` | Stage 4: country forward validation (installs >= 2021), leakage-deduped, for BOTH schemes. |
| `gee_tree_fix.py` | Fix for geemap.ml serialization of BEST-FIRST-grown sklearn trees (max_leaf_nodes). |
| `train_rf_v3.py` | Train the GEE-deployable RF on a v3 pool — with optional de-confounding (Model D) and |
| `covariate_append_manual.py` | Append manually-transferred covariates to a sampled point table, keyed on (lon,lat). |
| `covariate_sample.py` | Covariate-analysis SAMPLING pass (GEE-native half). |
| `covariate_aspect.py` | Aspect (slope-face direction) probe for the covariate analysis. |
| `covariate_analysis.py` | Covariate-analysis ANALYSIS pass (offline, on the sampled point table). |
| `mcda_benchmark_v3.py` | MCDA benchmark v3 — literature-sourced weights, WLC + TOPSIS, full criteria set, v2/R model. |
| `mcda_map_render.py` | Render the high-res China RF-vs-MCDA figure from china_grid.npz (local, no GEE). |

## `pipeline`
Training-pool construction from the AEF chips: per-pixel extraction, chip geometry, census, and the 8-method extraction bake-off.

| file | purpose |
|---|---|
| `aef_dtype.py` | — |
| `asset_crs.py` | — |
| `bakeoff_v3.py` | 8-method extraction bake-off on masks_v2, all on a COMMON balanced subsample, all scored |
| `chip_geo_v3.py` | Chip -> (iso3, cluster, year, centroid lat/lon) table for the spatial-diversity sampler |
| `extract_pixels_master.py` | Master pixel-pool extraction for the training-data investigation (IMPROVEMENTS.md Track 1, S1). |
| `extract_pixels_v3.py` | Extract the per-pixel training pool from AEF chips — v3 (masks_v2 + current negatives). |
| `pixel_census_v3.py` | S0 census for the training-data investigation (IMPROVEMENTS.md, Track 1). |

## `train`
Model training — the final model, the 9 MB-budget capacity×data sweep, the tree-count sweeps, and the candidate trainer.

| file | purpose |
|---|---|
| `build_final_model.py` | Build + validate the candidate FINAL model: cap120 . ne15 . mf32 . t-3. |
| `rf_budget_sweep.py` | Joint capacity x data-size sweep for the BEST GEE-deployable (<=9 MB) RF, per data size. |
| `train_candidates.py` | S2 step 1: train + SQUEEZE candidate RF models at chosen per-country budgets T, from the |
| `treecount_sweep.py` | Probe n_estimators BELOW 30 at the fixed ~9MB leaf budget (the one untested lever). |
| `treecount_temporal_train.py` | Temporal-holdout training: train ne15/20/30 (mf32, 9MB budget) on the canonical temporal pool |

## `validate`
The validation ladder — 13-country forward ROC, LOCO (spatial holdout), temporal holdout, the final 8-model head-to-head (final8_ladder), bootstrap CIs, TZ-SAM external validation, capacity-weighted.

| file | purpose |
|---|---|
| `bootstrap_ci.py` | Bootstrap 95% CIs for country ROC-AUC — inventory, TZ-SAM, and combined positive sets. |
| `capacity_weighted.py` | Capacity-weighted validation: does RF-R discriminate solar CAPACITY, not just site count? |
| `country_candidates.py` | S2 step 2: country forward-validation (the honest referee) for the T-sweep candidates. |
| `final8_ladder.py` | FINAL head-to-head: the full validation ladder on the 6 GEE-free candidates, one consistent harness. |
| `loco_country.py` | Reusable leave-one-country-out (LOCO) driver: extract -> train -> score -> plots -> CSV. |
| `loco_debates.py` | LOCO (leave-one-country-out) for the three open debates, scored LOCALLY on cached val embeddings. |
| `loco_temporal_v3.py` | Stage 5: LOCO (spatial holdout) + temporal (strict 2024 forecast), for BOTH schemes. |
| `rf_budget_country.py` | Stage 2 of the budget sweep: country forward-ROC for the BEST feasible RF per data size. |
| `temporal_holdout.py` | Temporal holdout (task 1b): honest future-forecast vs leaky in-sample. |
| `treecount_country.py` | Full country validation for ne15/20/25/30 (cap120, mf32) on cached AEF-2019 val embeddings. |
| `treecount_temporal_score.py` | Temporal-holdout scoring: sample AEF-2022 at the SAME 2024-install validation points used by |
| `validate_tzsam.py` | External forward-validation of the RF-R model against TZ-SAM Q1-2026 solar sites. |
| `verify_baseline.py` | Reproduction check: run the incumbent rf30_v3_R through the SAME driver/path used for the |

## `t3`
The t-3 (AEF install-3) lag pool build + scoring (minimal-GEE point-sample of install-3 embeddings, then local train/score).

| file | purpose |
|---|---|
| `t3_coverage_check.py` | Decide whether a 6M . t-3 pool can be built WITHOUT new GEE. |
| `t3_deeplookback_redo.py` | Redo the deep-lookback probe (STEP6 of step56.py) with the NEW model ne30_mf32. |
| `t3_sample_pool.py` | Build a t-3 (AEF install-3) version of the cap120 (px_v3) positive pool — minimal GEE. |
| `t3_train_score.py` | Train t-3 models on cap120_t3, score them + the existing t-2 cap120 models LOCALLY on the |
| `t3_val_embeddings.py` | Sample AEF(2019) 64-d embeddings ONCE at every country's validation points (site + rand), |

## `gee`
Earth Engine deploy/ops — global 10 m / 100 m exports, EECU cost probe, task audit/cancel. NOTE: gee_tree_fix.py (the max_leaf_nodes serialization fix) is a shared lib and lives at the root.

| file | purpose |
|---|---|
| `check_gee.py` | Audit live Earth Engine tasks for project ee-abdullahr-solar. |
| `deploy_global_2025.py` | Submit the master 100m global suitability export on 2025 AEF using the released R model. |
| `eecu_audit.py` | — |
| `export_10m.py` | Global 10m solar-suitability export (2025 AEF, released R model) -> Google Drive. |
| `final8_eecu_probe.py` | Empirically estimate the GEE EECU cost of building the 6M.t-3 pool. |
| `stop_gee.py` | Cancel any non-terminal Earth Engine operations for ee-abdullahr-solar. |

## `figures`
Paper figure / panel generators and high-res map renders.

| file | purpose |
|---|---|
| `appendix_figs.py` | Appendix figures: (1) 13-country forward ROC curves; (2) leakage-controlled ablations |
| `build_china_fig.py` | China 1x3 results figure (wide-and-short, PVOUT dropped from the narrative): |
| `build_colombia_fig.py` | Colombia 1x3 RF-vs-MCDA figure (appendix) -- a mirror of the China figure (Fig 3): |
| `build_coverage_fig.py` | Capacity + generation coverage figure (results). |
| `build_fig1_tiles.py` | Generate the Fig-1 tile assets for one site (China POS_CHN_12282_2018_1). |
| `build_fig2a_candidates.py` | Fig 2(a) candidate generator. |
| `build_fig2a_japan.py` | Paper Fig 2(a): 2x2 chip panel for the Japan site POS_JPN_15723_2017_3. |
| `build_fig_appendix_3x5.py` | Appendix figure: 3x5 grid of additional site-scale examples. |
| `build_fig_rf.py` | RF validation figure (paper). |
| `build_importance_fig.py` | Appendix covariate permutation-importance heatmap (redone on the paper_v3 RF), from |
| `build_india_norm.py` | Appendix weak-country figure (magma scheme): percentile-normalisation on the dimmest country |
| `build_leakage_fig.py` | Summarise the -2 vs -3 AEF-lag leakage test in one wide 2-panel figure + a table. |
| `build_suitability_globe_texture.py` | Export a GLOBAL equirectangular texture of the 100m suitability asset (magma, |
| `build_us_covariate_3x3.py` | US 3x3 covariate figure (appendix, covariate-novelty section). |
| `capacity_coverage.py` | Capacity coverage/gains curve (a) + per-size-bin ROC (b) — the capacity contribution. |
| `china_grid_sample.py` | Dense regular-grid sampling over China for a high-res, aligned RF-vs-MCDA map. |
| `china_mcda_map.py` | RF-vs-MCDA spatial agreement/disagreement map for China (Fig A panel). |
| `china_mcda_render.py` | Render the high-res China RF-vs-MCDA figure from china_grid.npz (local, no GEE). |
| `covariate_importance_figure.py` | Consolidated feature-importance figure across all 7 countries. |
| `covariate_maps.py` | Side-by-side spatial maps: our RF suitability raster vs the physical covariates. |
| `fig1_methodology.py` | Fig 1 methodology-schematic DRAFTS from a real AEF chip. |
| `fig_e_weights.py` | Fig E — published expert MCDA weights vs the EMPIRICAL importance of each criterion. |
| `gen_figure.py` | Confirm suitability-vs-resource orthogonality against GSA GHI directly, build the |
| `mcda_raster.py` | High-res LOCAL raster MCDA (1 km) + RF-vs-MCDA figure — accurate: equal-area distances, tiled 1 km RF, GPW pop. |
| `pick_chip.py` | Find big, clearly-visible utility solar farms (IND/ESP/CHN/USA/CHL) with good RF contrast; |
| `qualitative_v3.py` | Stage 6: qualitative evidence. |
| `render_india_clean.py` | Clean paper version of the India percentile-normalisation weak-country figure (magma throughout). |
| `render_india_norm.py` | India normalisation demo in the run_country_rf HEATMAP style (getThumbURL thumbnails, not rasters). |
| `render_raster.py` | Re-render the RF-vs-MCDA raster figure from cached arrays ({slug}_raster_data.npz) — instant, no recompute. |
| `viz_100m.py` | Render a global preview of the 100m suitability master asset (2025 AEF, RF-R). |

## `analysis`
Paper analyses — covariate 'embedding-novelty', capacity coverage/weighting, realized-generation, MCDA benchmark bootstrap, weak-country, global-suitable & grid-constrained potential.

| file | purpose |
|---|---|
| `bootstrap_figA.py` | Paired bootstrap for RF vs MCDA (Fig A) from saved per-point scores. |
| `build_roads_pbf.py` | Extract major-road network from a country .osm.pbf to <slug>_roads.gpkg (pyrosm), whole-country. |
| `count_pvids.py` | Count INDIVIDUAL PV facilities (PV_IDs) vs clustered scenes actually used as positives. |
| `covariate_corrected.py` | Emit the CORRECTED covariate table for the re-run of the covariate analysis + MCDA. |
| `covariate_novelty_v3.py` | Covariate-novelty analysis, uniform across all 13 countries, from the Fig-A per-point npz. |
| `covariate_recompute_v3.py` | Stage 8: recompute the covariate analysis with the NEW RF (embedding-novelty residuals + |
| `covariate_summary.py` | Roll up every country's covariate-analysis JSON into one tidy CSV for a morning read. |
| `density_check.py` | TZ-SAM capacity-density verification + exploration. |
| `embedding_importance_v3.py` | Stage 7: which of the 64 AlphaEarth dimensions drive the RF (permutation importance on the |
| `gen_eda.py` | EDA on PV facility generation data (Song et al. 2026, Nature Sust.). |
| `gen_join.py` | Join per-facility generation (Song 2026) to our saved per-site RF validation scores, |
| `global_suitable.py` | Global suitable-land fraction via POINT-SAMPLING (no raster, restricted-mode-safe). |
| `grid_constrained.py` | Grid/terrain/land-cover-constrained REALISTIC solar potential (global point-sampling). |
| `grid_sample.py` | Dense regular-grid sampling over a country for a high-res, aligned RF-vs-MCDA map. |
| `inspect_pvids.py` | Investigate: how many INDIVIDUAL PV facilities (PV_IDs) sit inside the sites/scenes we actually used? |
| `mcda_bootstrap_v3.py` | Combined-set RF-vs-MCDA benchmark with per-point scores saved for paired bootstrap. |
| `pvout_check.py` | Is Solargis long-term PVOUT (= World Bank study's core metric) informative for our model? |
| `site_profiling_v3.py` | Site-level covariate profiling from the Fig-A per-point npz — enlarged PV_Facility+TZ-SAM sites. |
| `tz_sam_overlap.py` | Quick overlap analysis: TZ-SAM Q1-2026 solar assets vs our PV facility inventory. |
| `weak_country_v3.py` | Weak-country wrap-up on the COMBINED (PV_Facility + TZ-SAM) data + a percentile-normalisation demo. |

## `negatives`
Negative-set re-mine / repair / contamination-audit pipeline.

| file | purpose |
|---|---|
| `audit_negatives_final.py` | Contamination audit of the FINAL on-disk negative set (13,444 chips). |
| `negative_remine_generate.py` | Generate NEW negative sites for the re-mine (4,242 across 22 deficit countries). |
| `negative_remine_manifest.py` | Lock the final negative-set target at 13,450 and emit the per-country DOWNLOAD |
| `negative_repair_eu_check.py` | After the 13,200 plan (waterfill to 12,500 + top the 8 flagged countries up to |
| `negative_repair_step1.py` | Step 1 of the negative-set repair: |
| `negative_repair_step2.py` | Step 2: (a) permanently delete the quarantined contaminated negatives (shared GWS, |
| `negative_repair_step3.py` | Step 3: execute the trim (delete excess pos-light negatives, densest-first to keep |
| `negative_tzsam_contamination.py` | Contamination check: do our downloaded NEGATIVE chips overlap solar sites that |
| `regen_neg_split.py` | Regenerate the negative split table from the CURRENT on-disk negatives (13,440). |

## `jobs`
SLURM `.sbatch` submitters + shell runners. Their python paths point into the subfolders and each exports `PYTHONPATH=<Solar-Siting>` so the shared-lib imports resolve.

| file | purpose |
|---|---|
| `build_final.sbatch` | SLURM/shell job |
| `china_grid.sbatch` | SLURM/shell job |
| `country_candidates.sbatch` | SLURM/shell job |
| `extract_master.sbatch` | SLURM/shell job |
| `figA_seq.sbatch` | SLURM/shell job |
| `final8_6M.sbatch` | SLURM/shell job |
| `final8_cap120.sbatch` | SLURM/shell job |
| `global_suitable.sbatch` | SLURM/shell job |
| `grid_constrained.sbatch` | SLURM/shell job |
| `grid_sample.sbatch` | SLURM/shell job |
| `loco_debates.sbatch` | SLURM/shell job |
| `mcda_bootstrap.sbatch` | SLURM/shell job |
| `mcda_raster.sbatch` | SLURM/shell job |
| `mcda_v3.sbatch` | SLURM/shell job |
| `postproc_figA.sbatch` | SLURM/shell job |
| `relay_10m.sh` | SLURM/shell job |
| `rf_budget_country.sbatch` | SLURM/shell job |
| `rf_budget_sweep.sbatch` | SLURM/shell job |
| `rf_sweep_stage1.sbatch` | SLURM/shell job |
| `rf_sweep_stage2.sbatch` | SLURM/shell job |
| `roads_ogr.sbatch` | SLURM/shell job |
| `roads_pbf.sbatch` | SLURM/shell job |
| `run_all_v3.sh` | SLURM/shell job |
| `run_mcda_v3.sh` | SLURM/shell job |
| `submit_sweep.sh` | SLURM/shell job |
| `t3_cap120.sbatch` | SLURM/shell job |
| `t3_train_score.sbatch` | SLURM/shell job |
| `train_candidates.sbatch` | SLURM/shell job |
| `treecount.sbatch` | SLURM/shell job |
| `treecount_temporal.sbatch` | SLURM/shell job |
| `tsweep_v3.sbatch` | SLURM/shell job |
| `tzsam.sbatch` | SLURM/shell job |

## `notebooks`
Original exploratory Jupyter notebooks (historical; the pipeline is now the scripts).

| file | purpose |
|---|---|
| `AlphaEarth_Solar_Model_Training.ipynb` | notebook |
| `DataHandling.ipynb` | notebook |
| `Discriminative_Suitability.ipynb` | notebook |
| `Embedding_Similarity_Search.ipynb` | notebook |
| `Initial_Analysis.ipynb` | notebook |
| `Solar Embeddings.ipynb` | notebook |

## `archive`
Superseded / interim / one-off diagnostic scripts, kept for provenance. NOT part of the current pipeline — e.g. the pre-v3 extract/train, the earlier rf_sweep (replaced by train/rf_budget_sweep), resolution & data ablations, the speckle probes, and the GEE-bug diagnosis scripts (`_diag_*`, `_pool_*`).

| file | purpose |
|---|---|
| `_chile_now.py` | — |
| `_diag_fix_test.py` | Validate gee_tree_fix: (1) reorder preserves predictions exactly; (2) all 7 winners now |
| `_diag_gee_roundtrip.py` | Round-trip each pool's BEST model through GEE to find which tree-strings actually PARSE. |
| `_diag_tokens.py` | Find the malformed tree-string line. Compare a WORKING incumbent vs a FAILING winner. |
| `_diag_treestring.py` | Diagnose the GEE 'expected 8, got 3' tree-string parse error. |
| `_local_vs_gee.py` | Confirm local sklearn predict_proba == GEE server-side classify score. |
| `_model_types.py` | Architecture (model-type) analysis on stage-1 pixel-val ROC. |
| `_pool_percountry.py` | Per-country BEST-model ROC across all 7 data pools (t-2 country scores, cached npz, no GEE). |
| `_pool_relationship.py` | Rigorous data-pool comparison for the BEST (ne30_mf32) architecture. |
| `_pool_robustness.py` | Is cap120's data-pool win robust to country composition, or a big-country artifact? |
| `_recon_table.py` | Reconstruct the full country ROC table directly from saved npz score arrays |
| `_sweep_verdict.py` | — |
| `_t3_layout.py` | Gate the t-3 build: (1) AEF-2017 floor = fraction of positives whose install-3 < 2017 (unusable), |
| `_t3_pilot.py` | PILOT the t-3 chain on a few chips (minimal GEE), no 40GB master load: |
| `broad_render.py` | Broad regional suitability heatmap (ne15 vs ne30) via GEE server-side thumbnail — to eyeball |
| `census_s0.sbatch` | — |
| `data_ablation_v3.py` | Data-quality ablation: does masks_v2 + the decontaminated/rebalanced negatives improve the |
| `extract_pbf_roads.sh` | — |
| `extract_pixels.py` | Extract the per-pixel training pool from AEF chips, with ablation filters. |
| `gee_tree_fix.py` | Fix for geemap.ml serialization of BEST-FIRST-grown sklearn trees (max_leaf_nodes). |
| `inv_year_counts.py` | Per-country install-year counts from the inventory — to choose validation / LOCO / temporal |
| `loco_grc.sh` | — |
| `loco_greece_allyears.py` | Resume Greece LOCO: only the all-years diagnostic (post-2021 already done + plotted). |
| `loco_greece_score.py` | Greece LOCO validation + organised plots. |
| `lotus_probe.sh` | — |
| `mcda_benchmark.py` | GIS-MCDA benchmark: does our AlphaEarth RF rank real installs better than a standard |
| `paper_v3.sbatch` | — |
| `patch_run_country.py` | Add an optional `exclude_pv_ids` kwarg to run_country() so validation sites that were seen |
| `plot_resolution_ablation.py` | Closing figure for the Greece sampling-resolution ablation (embedding models). |
| `probe.sbatch` | — |
| `pull_osm_7.sh` | — |
| `ratio_sweep_quick.py` | Quick train-ratio sensitivity check (IMPROVEMENTS.md Track 1). |
| `resolution_ablation.py` | Sampling-resolution ablation for the embedding-model deployment. |
| `rf_bench.py` | — |
| `rf_sweep.py` | Stage 1 of the RF hyperparameter sweep — the GEE-DEPLOYABLE (<=9 MB) frontier. |
| `rf_sweep_aggregate.py` | Stage 1 aggregator. Reads all sweep/config_*.json, writes a ranked table, and emits the |
| `rf_sweep_peek.py` | — |
| `rf_sweep_stage2.py` | Stage 2 of the RF sweep — the HONEST metric. Takes the Stage-1 shortlist (top GEE-feasible |
| `rf_sweep_stage2_aggregate.py` | Stage 2 aggregator / decision. Recomputes country-forward ROC from every sweep scores npz |
| `rf_sweep_stage2_peek.py` | Interim Stage-2 read: per-config country-forward ROC from whatever sweep npz exist so far, |
| `run_mcda_pass.sh` | — |
| `run_overnight.sh` | — |
| `run_pbf_roads_pass.sh` | — |
| `speckle_render.py` | Render ne15/ne20/ne30 suitability heatmaps for a few AEF-2019 patches, side by side, |
| `speckle_scale.py` | Does the ne15-vs-ne30 speckle 'cease to be an issue at scale'? Sample a fine grid, score |
| `speckle_test.py` | Speckle proof-or-disproof: does ne15 (fewer trees) produce noisier heatmaps than ne30? |
| `train_rf.py` | Train the GEE-deployable random forest on a pixel pool and score it on held-out pixels. |
| `tsweep_v3.py` | Per-country pixel-budget T-sweep with spatial diversity (IMPROVEMENTS.md Track 1, S1). |
| `tz_area.py` | — |
| `tz_new.py` | — |
| `weak_country.py` | Weak-country analysis (LOCAL, no GEE). |
