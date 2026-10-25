# Code for Arctic warming pattern scaling under CO₂ decline

This repository contains the Python scripts and Jupyter notebooks used to generate the main and Extended Data figures for the manuscript:

**Aru, H. and Li, C.** *Atlantic heat transport drives the breakdown of the Arctic warming pattern scaling during CO₂ decline.* Manuscript prepared for submission to *Nature Geoscience*.

## Study overview

The study examines whether Arctic warming pattern scaling (AWPS) relationships established during increasing atmospheric CO₂ remain valid after CO₂ begins to decline. It uses four CMIP6 initial-condition large ensembles—**CanESM5, MIROC6, MIROC-ES2L and MPI-ESM1.2-LR**—under **SSP1-1.9** and **SSP1-2.6**.

Three periods are analysed: historical (**1980–2014**), CO₂-increasing (**2015–2049**) and CO₂-decreasing (**2065–2099**). Temperature anomalies are referenced to **1850–1900**. The analysis covers Arctic SAT–GSAT relationships, matched-GSAT comparisons, Arctic Ocean heat storage and transport, and SAT pattern-scaling errors.

## Figure scripts

| File | Description |
| --- | --- |
| `Fig1.py` | Arctic warming pattern-scaling maps. |
| `Fig2.py` | Changes in pattern-scaling coefficients between periods, with statistical-significance hatching. |
| `Fig3_FigS5.py` | Arctic SAT–GSAT and sea-ice-area relationships, including matched-GSAT comparisons. |
| `Fig4.py` | Arctic Ocean heat budget and vertical profiles of heat transport, volume transport and temperature. |
| `Fig5.py` | SAT scaling errors and their association with upper-ocean heat-content changes. |
| `FIG_S1_CMIP6_HIST_tas_anomalies_timeseries_CO2.ipynb` | Temperature and atmospheric CO₂ time series. |
| `FIG_S2_patternScaling_maps_with_AgreementMask.ipynb` | Individual-model pattern-scaling maps. |
| `FIG_S3_S4_diff_pattern_scaling_with_AgreementMask.ipynb` | Individual-model pattern-scaling differences. |
| `FIG_S6_Arctic_regional_SAT_SIA_vs_CO2_concentration.py` | Arctic SAT and sea-ice area versus atmospheric CO₂. |
| `FIG_S7_ARO_temp_evolution_hovmoller_LEs.py` | Arctic Ocean temperature evolution with depth. |
| `FIG_S8_Arctic_ocean_schematic_diagram.py` | Arctic Ocean gateway and bathymetry map. |
| `FIG_S9_OHT_gateways_decompose_MPI_ESM.ipynb` | Gateway-specific ocean heat and volume transport profiles. |

## Requirements and use

The scripts use **Python 3** with packages including `numpy`, `pandas`, `xarray`, `scipy`, `matplotlib`, `cartopy`, `seaborn` and `jupyter`. The bathymetry script additionally uses `rioxarray` and `pyproj`.

Install the required packages, update the input/output paths in each script to match your local directories, and run, for example:

```bash
python Fig1.py
```

Open `.ipynb` files using Jupyter Notebook or JupyterLab.

**Data requirement:** Most scripts use preprocessed NetCDF files, rather than raw CMIP6 output. These input files and upstream preprocessing workflows are not included in this script collection. Figure reproduction therefore requires access to the corresponding processed datasets.

For detailed definitions and methods, please refer to the manuscript.
For inquiries, contact Hasi Aru at josie.aruhasi@mpimet.mpg.de.