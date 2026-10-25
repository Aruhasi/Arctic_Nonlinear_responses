# Code supporting “Atlantic heat transport drives the breakdown of the Arctic warming pattern scaling during CO₂ decline”
**Authors:** Hasi Aru and Chao Li  
**Code scope:** Main Figures 1–5 and associated Extended Data figure visualizations

## Overview

This repository contains Python scripts and Jupyter notebooks used to produce the principal figures and supporting diagnostics for the manuscript **“Atlantic heat transport drives the breakdown of the Arctic warming pattern scaling during CO₂ decline.”** The analysis examines whether spatial relationships between Arctic surface air temperature (SAT) and global-mean surface air temperature (GSAT) established during increasing atmospheric CO₂ remain applicable during subsequent CO₂ decline.

The study uses four CMIP6 initial-condition large ensembles (LEs) under SSP1-1.9 and SSP1-2.6. It compares Arctic warming pattern scaling (AWPS), Arctic SAT–GSAT relationships and matched-GSAT climate states across distinct forcing phases. Additional diagnostics investigate Arctic Ocean thermal persistence, heat-budget components, lateral ocean heat transport (OHT), ocean volume transport (OVT), and the spatial association between upper-ocean heat-content change and SAT pattern-scaling error.

**Scope of reproducibility:** The supplied scripts primarily read **previously processed NetCDF data** and produce manuscript figures. Some scripts also calculate figure-level regressions, ensemble statistics, significance overlays, or diagnostic CSV files. The complete raw CMIP6 download, regridding, pattern-scaling estimation, ocean transport integration, and Figure 5 input-generation workflows are **not contained in this figure-script collection**. Reproducing the figures therefore requires the corresponding intermediate data products and, for full end-to-end reproducibility, the upstream processing code.

## Models, experiments and conventions

| Item | Specification |
| --- | --- |
| Climate models | CanESM5, MIROC6, MIROC-ES2L, MPI-ESM1.2-LR |
| Nominal ensemble sizes | 50, 50, 10 and 50 members, respectively; individual scripts may use fewer available members |
| Scenarios | SSP1-1.9 (`ssp119`) and SSP1-2.6 (`ssp126`), with historical simulations as appropriate |
| Historical analysis | 1980–2014 |
| CO₂-increasing phase | 2015–2049 |
| CO₂-decreasing phase | 2065–2099 |
| Temperature reference | 1850–1900 historical mean |
| Arctic latitude criterion | ≥66.5° N for the principal spatial AWPS analysis |
| Multimodel averaging | First average available members within each model, then average the four model means with equal weight |
| Spatial comparison | SAT fields remapped to a common 1° × 1° grid upstream; depth-dependent ocean analyses use model-specific vertical coordinates where appropriate |

The model identifiers in the intermediate file names are not always the published CMIP6 names: `MIROC` refers to MIROC6, and `MPI-ESM-LR` or `MPI_ESM1-2-LR` refers to MPI-ESM1.2-LR. These aliases are retained to match the file names expected by the original scripts.

The phase labels refer to **atmospheric CO₂ concentration trajectories** and should not be interpreted as isolating CO₂ from all other time-varying external forcings in the SSP experiments.

## Script inventory and figure correspondence

The table gives each file's role. Some output file stems use legacy numbering that differs from the Extended Data figure number or script name; use the manuscript figure captions to determine the final panel order.

| Script or notebook | Manuscript material | Contents |
| --- | --- | --- |
| `Fig1.py` | Main Fig. 1 | Multimodel-mean Arctic AWPS coefficient maps for historical, CO₂-increasing and CO₂-decreasing periods under both SSPs. |
| `Fig2.py` | Main Fig. 2 | Multimodel AWPS differences: increasing minus historical, decreasing minus historical, and decreasing minus increasing; hatching where the paired-member contrast passes BH-FDR in at least three of four LEs. |
| `Fig3_FigS5.py` | Main Fig. 3 and companion matched-GSAT diagnostics | Arctic SAT–GSAT relationships, 80% highest-density-region (HDR) contours, matched-GSAT conditional SAT densities, September sea-ice-area (SIA)–GSAT relationships, member-cluster bootstrap intervals and diagnostic CSV summaries; creates a separate figure for each SSP. |
| `Fig4.py` | Main Fig. 4 | MPI-ESM1.2-LR Arctic Ocean heat-budget time series and depth-resolved profiles of total gateway OHT, OVT and ocean potential-temperature anomalies. |
| `Fig5.py` | Main Fig. 5 | Multimodel SAT scaling-error maps and the spatial regression coefficient linking SAT scaling error with standardized phase-to-phase 0–700 m Arctic OHC change. |
| `FIG_S1_CMIP6_HIST_tas_anomalies_timeseries_CO2.ipynb` | Extended Data Fig. 1 and related time-series diagnostics | Atmospheric CO₂, Arctic SAT, GSAT and Arctic-to-global warming-ratio time series; includes exploratory and intermediate diagnostic cells. |
| `FIG_S2_patternScaling_maps_with_AgreementMask.ipynb` | Extended Data Fig. 2 | Individual-LE Arctic AWPS patterns for historical and SSP phase comparisons. |
| `FIG_S3_S4_diff_pattern_scaling_with_AgreementMask.ipynb` | Extended Data Figs. 3–4 | Individual-LE AWPS differences for each scenario, with model-specific BH-FDR significance hatching. |
| `FIG_S6_Arctic_regional_SAT_SIA_vs_CO2_concentration.py` | Supporting SAT/SIA–CO₂ relationship figure (script label `S6`) | Arctic SAT and September SIA versus atmospheric CO₂ concentration, separated by model, scenario and phase. The hard-coded output stem currently says `Extended_Data_FIG5`; check numbering before archiving figures. |
| `FIG_S7_ARO_temp_evolution_hovmoller_LEs.py` | Extended Data Fig. 7 | Ensemble-mean Arctic Ocean potential-temperature anomaly Hovmöller diagrams for all four LEs, both scenarios and three phases, displayed to 2,000 m. |
| `FIG_S8_Arctic_ocean_schematic_diagram.py` | Arctic gateway/bathymetry schematic (script label `S8`) | Map of Arctic bathymetry and the four gateway sections: Barents Sea Opening (BSO), Fram Strait, Bering Strait and Davis Strait. It does **not** itself calculate gateway OHT contribution percentages. |
| `FIG_S9_OHT_gateways_decompose_MPI_ESM.ipynb` | Extended Data Fig. 9 | MPI-ESM1.2-LR depth profiles of OHT, OVT and potential-temperature anomalies at each of the four gateways. |

The available `Fig3_FigS5.py` code writes scenario-specific `FIG3_*` files rather than an explicitly named `FigS5` file; assign the main and companion panels according to the final manuscript layout. Likewise, Extended Data figure numbering should be checked against the assembled submission files rather than inferred solely from historical script or output names.

## Analytical methods represented in the code

### Arctic warming pattern scaling

For each model, ensemble member, grid cell and analysis period, the manuscript defines a linear relationship between local SAT anomaly and the corresponding **ensemble-mean** GSAT anomaly:

$$
\Delta T_{i,j,m}(t) = \alpha_{i,j,m} + \beta_{i,j,m}\,\Delta T_{\mathrm{GSAT}}^{\mathrm{ens}}(t) + \epsilon_{i,j,m}(t).
$$

Here, $\beta$ is the AWPS coefficient, with units °C °C⁻¹. The spatial maps in `Fig1.py` and `Fig2.py` load coefficients and differences calculated by upstream routines; they **do not recompute** the grid-cell regressions from the original monthly `tas` data.

Contrasts are computed for the same ensemble members across periods. The manuscript's statistical method applies a two-sided paired-member test at each Arctic grid cell, followed by the Benjamini–Hochberg false-discovery-rate (BH-FDR) correction at $q\leq0.05$. `Fig2.py` displays hatching only where **at least three of the four individual models** meet this criterion for the relevant contrast; `FIG_S3_S4_diff_pattern_scaling_with_AgreementMask.ipynb` shows the individual-model significance masks. Required significance fields are read from the upstream `*_LE_stats.nc` files (`significant_fdr`). These hatches express **statistical support**, not merely intermodel sign agreement.

### Matched-GSAT and regional responses

`Fig3_FigS5.py` examines changes in Arctic SAT–GSAT and September SIA–GSAT relationships and estimates the SAT difference between CO₂-decreasing and CO₂-increasing phases within overlapping GSAT ranges. The included script defaults to model- and scenario-specific matched-GSAT bands (`MATCH_SCOPE = "per_model"`), with an initially preferred width of 0.20 K and adaptive selection when necessary. It computes pooled ordinary-least-squares diagnostic fits, 80% HDR contours for SAT–GSAT pairs, and **10,000 member-cluster bootstrap replicates** (seed `42`) for uncertainty intervals, using paired-member resampling for the matched-phase difference when sufficient common members are available.

**Implementation distinction:** The provided `Fig3_FigS5.py` loader reads *member-specific* GSAT anomaly files and joins member–year pairs. The manuscript's spatial AWPS method, by contrast, specifies *ensemble-mean* GSAT as its predictor. These are different predictors; the manuscript's description of the **Figure 3 diagnostic regressions** should be checked against the final analysis code before release. Figure 3 pooled regression diagnostics are separate from the grid-cell regressions that define AWPS.

### Arctic Ocean heat budget and gateways

`Fig4.py` visualizes the MPI-ESM1.2-LR Arctic Ocean budget for the domain enclosed by the **BSO, Fram, Bering and Davis Strait** gateways. The budget contains the ocean heat-storage tendency, lateral OHT convergence, net surface heat flux into the liquid ocean and the latent-heat-equivalent sea-ice mass tendency. Positive fluxes signify heat gain by the Arctic Ocean. Anomalies are measured relative to 1850–1900; the available figure script reads upstream heat-budget diagnostics rather than deriving each flux from raw ocean-grid velocities, temperature and sea-ice fields.

The vertical gateway profiles use the native MPI-ESM1.2-LR ocean-grid sections in the upstream processing. In the manuscript, gateway OHT is evaluated relative to a reference potential temperature of 0 °C, with sensitivity to a pre-industrial Arctic mean reference temperature examined separately. The provided figure scripts **do not include** this reference-temperature sensitivity computation or a complete gateway heat-budget closure calculation.

### SAT scaling error and upper-ocean heat-content association

For each member, the CO₂-increasing-phase AWPS relationship is transferred without refitting to GSAT during 2065–2099:

$$
\widehat T_m^{\uparrow}(\mathbf{x},t)=\alpha_m^{\uparrow}(\mathbf{x})+\beta_m^{\uparrow}(\mathbf{x})G^{\mathrm{ens}}(t),\qquad E_m^{\mathrm{SAT}}(\mathbf{x},t)=T_m^{\mathrm{actual}}(\mathbf{x},t)-\widehat T_m^{\uparrow}(\mathbf{x},t).
$$

A positive SAT scaling error means that the simulated Arctic is warmer than predicted using the increasing-phase scaling relationship. In the **current Methods definition**, the heat-content predictor is the member's *direct phase-to-phase change* in Arctic-integrated upper-700-m OHC:

$$
\Delta H_m=\overline H_{m,2065\text{–}2099}-\overline H_{m,2015\text{–}2049},\qquad Z_{\Delta H,m}=\frac{\Delta H_m-\langle\Delta H\rangle}{s_{\Delta H}}.
$$

The spatial association is estimated within each model and scenario by

$$
\overline E_m^{\mathrm{SAT}}(\mathbf{x})=a(\mathbf{x})+\gamma_{\Delta H}(\mathbf{x})Z_{\Delta H,m}+\xi_m(\mathbf{x}).
$$

The coefficient $\gamma_{\Delta H}$ has units °C per one standard deviation of $\Delta H$; four model-specific coefficient maps are averaged equally. `Fig5.py` reads **precomputed** SAT error, standardized OHC–SAT slope and model-agreement fields from `Fig5_direct_OHC_change_{scenario}.nc`. The script also annotates an Arctic spatial pattern correlation read from the input dataset attributes. It does not independently reconstruct the SAT error or estimate $\gamma_{\Delta H}$ from raw member time series.

For Fig. 5, the supplied plotting routine hatches cells **without** the configured 75% agreement threshold (fewer than three agreeing models), due to the selection of the zero-valued part of the agreement mask in `add_hatching`. This convention is **different** from the significant-cell hatching in Fig. 2.

## Software requirements

These files are Python 3 scripts or Jupyter notebooks and use the following packages across the collection:

- Core numerical/data analysis: `numpy`, `pandas`, `xarray`, `scipy`, `dask` (for chunked/multifile xarray input).
- Visualization and mapping: `matplotlib`, `cartopy`, `seaborn`, `palettable`, `cycler`.
- Notebook execution: `jupyterlab` or `notebook`, `ipykernel`.
- GeoTIFF bathymetry map (`FIG_S8_*`): `rioxarray`, `rasterio`, `pyproj`.
- NetCDF access: an appropriate xarray backend, such as `netCDF4` or `h5netcdf`.

A starting installation command is:

```bash
conda create -n arctic-awps -c conda-forge python=3.11 \
  numpy pandas xarray scipy dask netcdf4 h5netcdf matplotlib \
  cartopy seaborn palettable cycler jupyterlab ipykernel \
  rioxarray rasterio pyproj
conda activate arctic-awps
```

This is a **suggested environment**, not a tested or version-locked specification of the authors' original HPC environment. Package versions should be recorded in a separate environment file when depositing a frozen archive. Cartopy may require local Natural Earth map data or access to fetch them.

## Input data and directory configuration

The scripts retain absolute paths from the original MPI-M/Levante analysis workspace, with a project root of the form:

```text
/root/
└── Atlantic_capacitor_on_Arctic/
    ├── docs/
    │   ├── data/
    │   │   ├── Spatial_pattern/
    │   │   ├── Regional_diagnosis/
    │   │   ├── Heat_budget/
    │   │   └── Amon_volume_transport/
    │   └── figs/
    └── scripts/
        ├── SpatialPattern/
        ├── Timeseries/
        ├── Heat_budget/
        ├── Nonlinear_diagnosis/
        └── OHT_constant_temp/
```

Some additional inputs reside in other workspace directories (for example, `Data_storage`, `Land_surf_temp`, and `Observational_constraints`). **Before running** a script, update its path variables (`ROOT`, `PROJECT_ROOT`, `DATA_DIR`, `DIR_*`, `FIG_DIR`, `FIG_OUT`, etc.) to point to locally available datasets. Merely placing the 12 source files into one directory will not satisfy the data dependencies.

Important intermediate inputs include:

| Inputs | Main consumers |
| --- | --- |
| `*_Beta_patterns_1980-2014.nc`, `*_Beta_patterns_2015-2049.nc`, `*_Beta_patterns_2065-2099.nc` | Fig. 1, Extended Data Fig. 2 |
| `Beta_pattern_mean_*_differences_Among_LEs.nc` and model-specific difference files | Fig. 2, Extended Data Figs. 3–4 |
| `*_LE_stats.nc` including `significant_fdr`; `GR15_lsm_regrid_1deg.nc` | Fig. 2, Extended Data Figs. 3–4 |
| Regional Arctic SAT, September SIA, member GSAT anomaly NetCDF files; atmospheric CO₂ scenario files | Fig. 3, supporting SAT/SIA–CO₂ figure, Extended Data Fig. 1 |
| Arctic heat-budget components, gateway OHT/OVT depth profiles, sea-ice mass budget and ocean temperature-profile NetCDF files | Fig. 4 and Extended Data Fig. 9 |
| `Fig5_direct_OHC_change_ssp119.nc`, `Fig5_direct_OHC_change_ssp126.nc` | Fig. 5 |
| `*_Arctic_thetao_hov_vs_1850-1900.nc` | Extended Data Fig. 7 |
| `ibcao_v5_1_200m.tif` (external IBCAO bathymetry raster) | Arctic gateway/bathymetry schematic |

NetCDF variable names and coordinates must agree with the code: examples include `tas`, `significant_fdr`, `ArcticOcean66.5N`, `SIA_anom`, `OHT`, `OVT`, `thetao_hovmoller_anom`, `sat_error_mmm` and `ohc_change_standardized_slope_mmm`. Individual scripts enforce different year, ensemble-member, depth-coordinate and longitude conventions.

## Generating figures

Once the matching processed datasets are accessible and paths have been updated, standalone scripts can be run from the command line:

```bash
python Fig1.py
python Fig2.py
python Fig3_FigS5.py
python Fig4.py
python Fig5.py
python FIG_S6_Arctic_regional_SAT_SIA_vs_CO2_concentration.py
python FIG_S7_ARO_temp_evolution_hovmoller_LEs.py
python FIG_S8_Arctic_ocean_schematic_diagram.py
```

The notebook files are:

```text
FIG_S1_CMIP6_HIST_tas_anomalies_timeseries_CO2.ipynb
FIG_S2_patternScaling_maps_with_AgreementMask.ipynb
FIG_S3_S4_diff_pattern_scaling_with_AgreementMask.ipynb
FIG_S9_OHT_gateways_decompose_MPI_ESM.ipynb
```

**Notebook validation:** The supplied `FIG_S2_patternScaling_maps_with_AgreementMask.ipynb` contains a trailing comma in a JSON `source` array and is not valid strict notebook JSON as delivered. Remove that extraneous comma or resave a repaired notebook before opening it with Jupyter. The other three uploaded notebooks passed JSON parsing during preparation of this README. Execution of the full analysis was **not tested**, because the intermediate data products are not included in the supplied collection.

### Representative output names

| Script | Output filename(s) in the supplied code |
| --- | --- |
| `Fig1.py` | `Fig1_pattern_scaling_SSP119_SSP126.pdf` / `.png` |
| `Fig2.py` | `Fig2_pattern_scaling_differences.pdf` / `.png` |
| `Fig3_FigS5.py` | `FIG3_three_row_ssp119_jointHDR_matchedGSAT.pdf` and `FIG3_three_row_ssp126_jointHDR_matchedGSAT.pdf` (plus PNG equivalents); four `FIG3_*` diagnostic CSV files |
| `Fig4.py` | `Fig4_merged_heat_budget_profiles_fulldepth.pdf` / `.png` |
| `Fig5.py` | `Fig5.pdf` / `.png` |
| `FIG_S2_*.ipynb` | `Extended_Data_Fig2_fingerprint_LEs_polar(Hist&MT&LT).pdf` / `.png` |
| `FIG_S3_S4_*.ipynb` | `Fig_S3_4LE_ssp119_pattern_scaling_with_FDRSignificance.pdf` and corresponding `ssp126` output |
| `FIG_S6_*.py` | `Extended_Data_FIG5_Arctic_SAT_SIA_vs_CO2_linear.pdf` / `.png` (legacy stem) |
| `FIG_S7_*.py` | `4LEs_Arctic_thetao_hovmoller_ssp119_ssp126_3periods_2000m.pdf` / `.png` |
| `FIG_S8_*.py` | `Arctic_gateway_schematic_depth_shading.pdf` / `.png` |
| `FIG_S9_*.ipynb` | `Fig5_S9_profiles_all_gateways_4x3_Annual_TotOHT.pdf` / `.png` |

Some figures are saved into a `docs/figs/` subdirectory and others into the working directory or `./figs`, as specified within each script. The notebooks may save intermediate products in addition to the final files shown above.

## Data availability, provenance and limitations

The underlying climate simulations are from the CMIP6 model archives and the four initial-condition large ensembles identified above. The prepared figure inputs are derived from those simulations using additional preprocessing, regridding, regional averaging, heat-budget and regression workflows. The bathymetry schematic uses an IBCAO v5.1 GeoTIFF input. **No raw CMIP6 files, preprocessed NetCDF inputs, IBCAO raster, or DOI-backed archived data package were supplied with this code collection.** A final public repository or data-deposition record should cite the specific datasets and archive accessions actually released with the submission.

This repository should therefore be understood as **figure-generation and figure-level diagnostic code**, not a standalone installer or a complete end-to-end climate-analysis pipeline. It has not been executed in a clean environment as part of preparation of this README. The figure script outputs are not a substitute for the definitions, quality controls, ensemble-member selections, sign conventions and uncertainty procedures documented in the paper's Methods.

## Citation and contact

When using these scripts, please cite the accompanying manuscript:

> Aru, H. and Li, C. **Atlantic heat transport drives the breakdown of the Arctic warming pattern scaling during CO₂ decline.** Manuscript prepared for *Nature Geoscience* (2026).

For methodological questions, contact the corresponding author using the address listed in the submitted manuscript. Add a permanent DOI and repository license to the deposited code record when assigned; neither is specified in the supplied script collection.
