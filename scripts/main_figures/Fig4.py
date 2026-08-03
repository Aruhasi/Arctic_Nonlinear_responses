#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %%
import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.path as mpath

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.geoaxes import GeoAxes
# %%
# ============================================================
# ---------------------- USER SETTINGS ------------------------
# ============================================================
MODEL = "MPI-ESM1-2-LR"

DIR_HEAT_BUDGET = "/work/mh0033/m301036/Atlantic_heat_capacitor_Arctic/Atlantic_capacitor_on_Arctic/scripts/Heat_budget/heat_diagnosis/Atlantic_mask_monthly_output"
DIR_OHT_TS = "/work/mh0033/m301036/Atlantic_heat_capacitor_Arctic/Atlantic_capacitor_on_Arctic/docs/data/Heat_budget/OHT/output"
DIR_OVT = "/work/mh0033/m301036/Atlantic_heat_capacitor_Arctic/Atlantic_capacitor_on_Arctic/docs/data/Amon_volume_transport/output"
DIR_THETAO = "/work/mh0033/m301036/Land_surf_temp/Observational_constraints/OHC_OHT_cal/ARO_temp_profile/output"

MASK_FILE = "/work/mh0033/m301036/Atlantic_heat_capacitor_Arctic/Atlantic_capacitor_on_Arctic/scripts/Heat_budget/ARO_mask/gateway_masks/ARO_manual_minus_Atlantic_on_atmos_grid.nc"

FIG_OUT = "/work/mh0033/m301036/Atlantic_heat_capacitor_Arctic/Atlantic_capacitor_on_Arctic/docs/figs/Main_Figs"
os.makedirs(FIG_OUT, exist_ok=True)

MEMBERS = [f"r{i}i1p1f1" for i in range(1, 51)]

SCENARIOS = ["historical", "ssp119", "ssp126"]
SCENARIOS_PERIODS = {
    "historical": ["1980-2014"],
    "ssp119": ["2015-2049", "2065-2099"],
    "ssp126": ["2015-2049", "2065-2099"],
}
PERIOD_ORDER = ["1980-2014", "2015-2049", "2065-2099"]

SCENARIO_COLORS = {
    "historical": "black",
    "ssp119": "#407BD0",
    "ssp126": "#A32A31",
}

PERIOD_COLORS = {
    "1980-2014": "black",
    "2015-2049": "#407BD0",
    "2065-2099": "#A32A31",
}

PROFILE_LINESTYLES = {
    "historical": "-",
    "ssp119": ":",
    "ssp126": "-.",
}

OHT_GATEWAY_NAMES = {
    "BeringStrait": "Bering Strait",
    "DavisStrait": "Davis Strait",
    "FramStrait": "Fram Strait",
    "BSO": "BSO",
    "total": "Total",
}

OVT_GATEWAY_NAMES = {
    "bering_strait": "Bering Strait",
    "davis_strait": "Davis Strait",
    "fram_strait": "Fram Strait",
    "barents_opening": "BSO",
    "total": "Total",
}

OHT_GATEWAYS = ['BeringStrait',  'DavisStrait', 'FramStrait', 'BSO', 'total']
OVT_GATEWAYS = ['bering_strait',  'davis_strait', 'fram_strait', 'barents_opening', 'total']

GATEWAY_FOR_PANEL_B = "total"
GATEWAY_FOR_PANEL_C = "total"
# %%
# ============================================================
# ---------------------- HELPERS ------------------------------
# ============================================================
def find_existing_file(patterns):
    for pat in patterns:
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    return None

def ensure_time_sorted_unique(ds):
    if "time" not in ds.coords:
        return ds
    time_idx = pd.Index(ds["time"].values)
    if time_idx.duplicated().any():
        keep = ~time_idx.duplicated(keep="first")
        ds = ds.isel(time=keep)
    return ds.sortby("time")

def annualize_da(da):
    if "time" not in da.dims:
        return da
    return da.resample(time="YS").mean("time")

def choose_var(ds, preferred_names=None):
    if preferred_names is None:
        preferred_names = []
    for v in preferred_names:
        if v in ds.data_vars:
            return ds[v]
    numeric_vars = [v for v in ds.data_vars if np.issubdtype(ds[v].dtype, np.number)]
    if not numeric_vars:
        raise ValueError("No numeric variable found.")
    return ds[numeric_vars[0]]

def get_depth_coord(da):
    for cand in ["depth", "depth_2", "lev", "olevel", "z", "olev"]:
        if cand in da.coords or cand in da.dims:
            return cand
    raise ValueError(f"No depth coordinate found in {da.dims} / {list(da.coords)}")

def reduce_extra_dims(da, keep_dims=("member", "time")):
    extra = [d for d in da.dims if d not in keep_dims]
    if extra:
        da = da.mean(dim=extra)
    return da

def clean_for_concat(da):
    # Keep only dimension coordinates, remove aux/scalar coords that often break concat
    drop_names = [c for c in da.coords if c not in da.dims]
    if drop_names:
        da = da.drop_vars(drop_names, errors="ignore")
    return da

def add_panel_label(ax, label):
    ax.text(
        -0.05, 1.1, label,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=18, fontweight="bold"
    )

def transform_depth_axis(ax, upper_limit=700, lower_limit=4500):
    """
    Apply custom depth axis transformation to zoom in on upper ocean (0-700m)
    and compress deep ocean (700-4500m) for Arctic basin.
    
    Uses unequal tick spacing to achieve the visual effect.
    """
    # Define depth levels with unequal spacing
    # Dense spacing for 0-700m, sparse for deeper (Arctic max depth ~4500m)
    depth_ticks = [0, 100, 200, 300, 400, 500, 600, 700, 1000, 1500, 2000, 3000, 4000, 4500]
    
    # Create positions that compress deeper depths
    # 0-700m uses positions 0-700 (linear)
    # 700-4500m compressed into 700-1400 (700 units for 3800m)
    positions = []
    for depth in depth_ticks:
        if depth <= 700:
            positions.append(depth)
        else:
            # Compress 700-4500m into 700-1400 range
            compressed = 700 + (depth - 700) / 3800 * 700
            positions.append(compressed)
    
    ax.set_ylim(max(positions), 0)
    ax.set_yticks(positions)
    ax.set_yticklabels([str(int(d)) for d in depth_ticks])
    
    # Add horizontal line to mark the transition
    ax.axhline(y=700, color='lightgray', linestyle=':', linewidth=0.8, alpha=0.5, zorder=0)
    
    return positions, depth_ticks

def map_depth_to_compressed(depth_values, upper_limit=700):
    """
    Map actual depth values to compressed y-axis positions.
    Arctic basin max depth ~4500m, so compress 700-4500m range.
    """
    depth_array = np.asarray(depth_values)
    compressed = np.where(
        depth_array <= upper_limit,
        depth_array,
        upper_limit + (depth_array - upper_limit) / 3800 * 700
    )
    return compressed
# %%
# ============================================================
# ------------- LOAD PANEL-A HEAT-BUDGET TERMS ---------------
# ============================================================
def load_heat_budget_component(var_name):
    out = {}

    for scen in SCENARIOS:
        member_das = []

        for mem in MEMBERS:
            base_dir = f"{DIR_HEAT_BUDGET}/{scen}/{mem}"
            patterns = [
                f"{base_dir}/{var_name}_area_anom.nocrossterms.{MODEL}_{scen}_{mem}.nc",
                f"{base_dir}/{var_name}_area_anom.{MODEL}_{scen}_{mem}.nc",
                f"{base_dir}/{var_name}_*.{MODEL}_{scen}_{mem}.nc",
                f"{base_dir}/{var_name}*.nc",
            ]
            fn = find_existing_file(patterns)
            if fn is None:
                continue

            try:
                ds = xr.open_dataset(fn, decode_times=True)
                ds = ensure_time_sorted_unique(ds)
                da = choose_var(ds, [var_name]).squeeze(drop=True)

                for d in ["lat", "lon"]:
                    if d in da.dims and da.sizes[d] == 1:
                        da = da.squeeze(d, drop=True)

                for d in ["lev", "depth", "depth_2", "bnds"]:
                    if d in da.dims:
                        da = da.mean(d)

                da = reduce_extra_dims(da, keep_dims=("time",))
                da = annualize_da(da)
                da = clean_for_concat(da)
                da = da.expand_dims(member=[mem])
                member_das.append(da)

            except Exception as e:
                print(f"[warn] failed to read {fn}: {e}")

        if member_das:
            out[scen] = xr.concat(member_das, dim="member", coords="minimal", compat="override", join="override")
            print(f"[load] {var_name} {scen}: {out[scen].sizes['member']} members")
        else:
            print(f"[warn] no files for {var_name} {scen}")

    return out

# def load_ohs_timeseries():
#     out = {}
#     dir_ohs = "/work/mh0033/m301036/Atlantic_heat_capacitor_Arctic/Atlantic_capacitor_on_Arctic/docs/data/Heat_budget/Heat_content_Arctic_full_depth/MPI-ESM1-2-LR"

#     for scen in SCENARIOS:
#         member_das = []

#         for mem in MEMBERS:
#             fn = f"{dir_ohs}/{scen}/{MODEL}_{scen}_{mem}_OHS_Arctic_TW_anom_timeseries.nc"
#             if not os.path.exists(fn):
#                 continue
#             try:
#                 ds = xr.open_dataset(fn, decode_times=True)
#                 da = choose_var(ds, ["OHS_Arctic_TW_anom", "OHS"]).squeeze(drop=True)
#                 da = annualize_da(da)
#                 da = reduce_extra_dims(da, keep_dims=("time",))
#                 da = clean_for_concat(da)
#                 da = da.expand_dims(member=[mem])
#                 member_das.append(da)
#             except Exception as e:
#                 print(f"[warn] failed OHS file {fn}: {e}")

#         if member_das:
#             out[scen] = xr.concat(member_das, dim="member", coords="minimal", compat="override", join="override")
#             print(f"[load] OHS {scen}: {out[scen].sizes['member']} members")

#     return out
def load_ohs_timeseries():
    out = {}
    # dir_ohs = "/work/mh0033/m301036/Atlantic_heat_capacitor_Arctic/Atlantic_capacitor_on_Arctic/docs/data/Heat_budget/Heat_content_Arctic_full_depth/MPI-ESM1-2-LR"
    dir_ohs = DIR_HEAT_BUDGET

    for scen in SCENARIOS:
        member_das = []

        for mem in MEMBERS:
            fn = f"{dir_ohs}/{scen}/{mem}/opottemptend_area_anom.nocrossterms.MPI-ESM1-2-LR_{scen}_{mem}.nc"
            if not os.path.exists(fn):
                continue
            try:
                ds = xr.open_dataset(fn, decode_times=True)
                da = choose_var(ds, ["opottemptend"]).squeeze(drop=True)
                da = annualize_da(da)
                da = reduce_extra_dims(da, keep_dims=("time",))
                da = clean_for_concat(da)
                da = da.expand_dims(member=[mem])
                member_das.append(da)
            except Exception as e:
                print(f"[warn] failed OHS file {fn}: {e}")

        if member_das:
            out[scen] = xr.concat(member_das, dim="member", coords="minimal", compat="override", join="override")
            print(f"[load] OHS {scen}: {out[scen].sizes['member']} members")

    return out

def load_oht_timeseries_total():
    out = {}
    model_file = "MPI_ESM1-2-LR"  # Files use MPI_ESM1-2-LR (only first hyphen replaced)

    file_map = {
        "historical": {
            "1980-2014": (f"{DIR_OHT_TS}/{model_file}_OHT_total_anom_1980-2014.nc", (1980, 2014)),
        },
        "ssp119": {
            "2015-2049": (f"{DIR_OHT_TS}/{model_file}_OHT_total_anom_ssp119_2015-2049.nc", (2015, 2049)),
            "2065-2099": (f"{DIR_OHT_TS}/{model_file}_OHT_total_anom_ssp119_2065-2099.nc", (2065, 2099)),
        },
        "ssp126": {
            "2015-2049": (f"{DIR_OHT_TS}/{model_file}_OHT_total_anom_ssp126_2015-2049.nc", (2015, 2049)),
            "2065-2099": (f"{DIR_OHT_TS}/{model_file}_OHT_total_anom_ssp126_2065-2099.nc", (2065, 2099)),
        },
    }

    for scen in SCENARIOS:
        das = []
        for period, (fn, expected_years) in file_map.get(scen, {}).items():
            if not os.path.exists(fn):
                print(f"[warn] missing OHT total file: {fn}")
                continue
            try:
                ds = xr.open_dataset(fn, decode_times=True)
                da = choose_var(ds, ["OHT"]).squeeze(drop=True)

                for d in ["depth_2", "depth", "lev"]:
                    if d in da.dims:
                        da = da.sum(d)

                for d in ["lat", "lon"]:
                    if d in da.dims and da.sizes[d] == 1:
                        da = da.squeeze(d, drop=True)

                da = annualize_da(da)
                da = reduce_extra_dims(da, keep_dims=("member", "time"))
                
                # FIX TIME COORDINATES if they're wrong (SSP126 2065-2099 has wrong times!)
                if 'time' in da.dims:
                    actual_start_year = pd.to_datetime(da.time.values[0]).year
                    expected_start_year = expected_years[0]
                    
                    if actual_start_year != expected_start_year:
                        print(f"  [warn] {scen} {period}: TIME MISMATCH! File has {actual_start_year}, expected {expected_start_year}")
                        print(f"    Correcting time coordinates to {expected_years[0]}-{expected_years[1]}...")
                        
                        # Create correct time coordinates
                        n_years = len(da.time)
                        new_times = pd.date_range(
                            start=f"{expected_years[0]}-01-01",
                            periods=n_years,
                            freq='YS'
                        )
                        da = da.assign_coords(time=new_times)
                
                da = clean_for_concat(da)
                das.append(da)
            except Exception as e:
                print(f"[warn] failed OHT total file {fn}: {e}")

        if das:
            out[scen] = xr.concat(das, dim="time", coords="minimal", compat="override", join="override").sortby("time")
            print(f"[load] OHT total {scen}: {out[scen].sizes}")

    return out

def load_qice_timeseries():
    """Load Arctic sea-ice heat flux anomaly (Qice_total_TW_anom) for all scenarios.
    One file per scenario already contains all 50 members.
    """
    dir_root = "/work/mh0033/m301036/Atlantic_heat_capacitor_Arctic/Atlantic_capacitor_on_Arctic/docs/data/Regional_diagnosis/seaice_heat_budget"
    out = {}
    for scen in SCENARIOS:
        file_path = (
            f"{dir_root}/"
            f"MPI-ESM1-2-LR_{scen}_Arctic66p5N_sidmass_seaice_heat_budget_annual.nc"
        )
        if not os.path.exists(file_path):
            print(f"[warn] Qice file not found: {file_path}")
            continue
        try:
            ds = xr.open_dataset(file_path, decode_times=True)
            da = ds["Qice_total_TW_anom"]   # (member, time); annual means
            out[scen] = da
            print(f"[load] Qice {scen}: members={da.sizes['member']}, time={da.sizes['time']}")
        except Exception as e:
            print(f"[warn] failed to load Qice {scen}: {e}")
    return out
# %%
# ============================================================
# ------------------- LOAD PROFILE DATA -----------------------
# ============================================================
def load_oht_profiles_with_members(gateway="total"):
    """Load OHT profiles with all members for proper ensemble statistics."""
    dir_in = "/work/mh0033/m301036/Atlantic_heat_capacitor_Arctic/Atlantic_capacitor_on_Arctic/docs/data/Heat_budget/OHT/output"
    model_file = "MPI_ESM1-2-LR"  # Files use MPI_ESM1-2-LR (only first hyphen replaced)
    
    scenarios_periods = {
        "historical": ["historical"],  # alias; real period name will be "1980-2014"
        "ssp119":     ["2015-2049", "2065-2099"],
        "ssp126":     ["2015-2049", "2065-2099"],
    }
    
    ds_gateway_OHT_mem = {}
    ds_gateway_OHT_mem[gateway] = {}  # nested dict
    
    for scen, periods in scenarios_periods.items():
        ds_gateway_OHT_mem[gateway][scen] = []  # list for each scenario
        
        for period in periods:
            if scen == "historical":
                period_str = "1980-2014"
                file_path = f"{dir_in}/{model_file}_OHT_{gateway}_anom_{period_str}.nc"
            else:
                period_str = period
                file_path = f"{dir_in}/{model_file}_OHT_{gateway}_anom_{scen}_{period_str}.nc"
            
            if not os.path.exists(file_path):
                print(f"[warn] OHT file not found: {file_path}")
                continue
            
            try:
                ds = xr.open_dataset(file_path)
                # load OHT, add "period" coordinate
                da = ds['OHT'].load().expand_dims(dim={"period": [period_str]})
                ds_gateway_OHT_mem[gateway][scen].append(da)
            except Exception as e:
                print(f"[warn] failed to load OHT file {file_path}: {e}")
    
    # calculate the ensemble mean for each scenario
    ds_gateway_OHT_mean = {}
    ds_gateway_OHT_mean[gateway] = {}
    for scen in scenarios_periods.keys():
        if len(ds_gateway_OHT_mem[gateway][scen]) == 0:
            continue
        ds_gateway_OHT_mean[gateway][scen] = xr.concat(
            ds_gateway_OHT_mem[gateway][scen], dim="member"
        ).mean(dim="member")
    
    return ds_gateway_OHT_mem, ds_gateway_OHT_mean

def load_ovt_profiles_with_members(gateway="total"):
    """Load OVT profiles with all members for proper ensemble statistics."""
    dir_in = "/work/mh0033/m301036/Atlantic_heat_capacitor_Arctic/Atlantic_capacitor_on_Arctic/docs/data/Amon_volume_transport/output"
    model_file = "MPI_ESM1-2-LR"  # Files use MPI_ESM1-2-LR (only first hyphen replaced)
    
    scenarios_periods = {
        "historical": ["1980-2014"],
        "ssp119":     ["2015-2049", "2065-2099"],
        "ssp126":     ["2015-2049", "2065-2099"],
    }
    
    ds_gateway_OVT_mem = {}
    ds_gateway_OVT_mem[gateway] = {}  # nested dict
    
    for scen, periods in scenarios_periods.items():
        ds_gateway_OVT_mem[gateway][scen] = []  # list for each scenario
        
        for period in periods:
            if scen == "historical":
                period_str = "1980-2014"
                file_path = f"{dir_in}/{model_file}_OVT_{gateway}_anom_{period_str}.nc"
            else:
                period_str = period
                file_path = f"{dir_in}/{model_file}_OVT_{gateway}_anom_{scen}_{period_str}.nc"
            
            if not os.path.exists(file_path):
                print(f"[warn] OVT file not found: {file_path}")
                continue
            
            try:
                ds = xr.open_dataset(file_path)
                # load OVT, add "period" coordinate
                da = ds['OVT'].load().expand_dims(dim={"period": [period_str]})
                ds_gateway_OVT_mem[gateway][scen].append(da)
            except Exception as e:
                print(f"[warn] failed to load OVT file {file_path}: {e}")
    
    # calculate the ensemble mean for each scenario
    ds_gateway_OVT_mean = {}
    ds_gateway_OVT_mean[gateway] = {}
    for scen in scenarios_periods.keys():
        if len(ds_gateway_OVT_mem[gateway][scen]) == 0:
            continue
        ds_gateway_OVT_mean[gateway][scen] = xr.concat(
            ds_gateway_OVT_mem[gateway][scen], dim="member"
        ).mean(dim="member")
    
    return ds_gateway_OVT_mem, ds_gateway_OVT_mean

def load_thetao_profiles():
    out = {"historical": {}, "ssp119": {}, "ssp126": {}}

    period_alias = {
        "1980-2014": ["1980-2014", "1980–2014"],
        "2015-2049": ["2015-2049", "2015–2049"],
        "2065-2099": ["2065-2099", "2065–2099"],
    }

    for scen in ["ssp119", "ssp126"]:
        per_profiles = {p: [] for p in PERIOD_ORDER}
        per_depth = {p: None for p in PERIOD_ORDER}

        for mem in MEMBERS:
            fn = f"{DIR_THETAO}/{MODEL}_{scen}_{mem}_Arctic_thetao_profiles_vs_1850-1900.nc"
            if not os.path.exists(fn):
                continue

            try:
                ds = xr.open_dataset(fn)
                if "thetao_profile_anom" not in ds:
                    continue

                period_vals = ds["period"].values.astype(str)

                for period in PERIOD_ORDER:
                    match = next((p for p in period_alias[period] if p in period_vals), None)
                    if match is None:
                        continue

                    prof = ds["thetao_profile_anom"].sel(period=match).squeeze()
                    depth_name = get_depth_coord(prof)
                    depth = prof[depth_name].values

                    if per_depth[period] is None:
                        per_depth[period] = depth

                    per_profiles[period].append(np.asarray(prof.values))
            except Exception as e:
                print(f"[warn] failed thetao file {fn}: {e}")

        for period in PERIOD_ORDER:
            if per_profiles[period]:
                arr = np.asarray(per_profiles[period])
                out[scen][period] = {
                    "mean": np.nanmean(arr, axis=0),
                    "std": np.nanstd(arr, axis=0),
                    "depth": per_depth[period],
                }

    if "1980-2014" in out["ssp119"] and "1980-2014" in out["ssp126"]:
        mean_hist = 0.5 * (out["ssp119"]["1980-2014"]["mean"] + out["ssp126"]["1980-2014"]["mean"])
        std_hist = 0.5 * np.sqrt(out["ssp119"]["1980-2014"]["std"]**2 + out["ssp126"]["1980-2014"]["std"]**2)
        depth_hist = out["ssp119"]["1980-2014"]["depth"]
        out["historical"]["1980-2014"] = {"mean": mean_hist, "std": std_hist, "depth": depth_hist}

    return out
# %%
# ============================================================
# -------------------- PLOTTING HELPERS ----------------------
# ============================================================
def plot_panel_a(ax, oht_total, ohs, nshf, qflux=None):
    """
    Panel a: Arctic Ocean heat-budget component time series.

    Colour = component
    Line style = scenario
    """

    # ------------------------------------------------------------
    # Component colours
    # ------------------------------------------------------------
    component_colors = {
        "OHT":   "#A32A31", #"#AA0A2E",
        "OHS":   "#F4683F",
        "Q$_{surf}$":   "#407BD0", # "#575AA7"
        "Q$_{ice}$": "#6B92C5",
    }

    # ------------------------------------------------------------
    # Scenario line styles
    # ------------------------------------------------------------
    scenario_linestyles = {
        "historical": "-",
        "ssp119": "--",
        "ssp126": "-",
    }

    scenario_labels = {
        "historical": "Historical",
        "ssp119": "SSP1-1.9",
        "ssp126": "SSP1-2.6",
    }

    components = {
        "OHT":   oht_total,
        "OHS":   ohs,
        "Q$_{surf}$":   nshf,
        "Q$_{ice}$": qflux if qflux is not None else {},
    }

    y_values = []

    for comp_name, comp_dict in components.items():
        if not comp_dict:
            continue

        comp_color = component_colors[comp_name]

        for scen in SCENARIOS:
            if scen not in comp_dict:
                continue

            da = comp_dict[scen]

            if "member" in da.dims:
                mean_da = da.mean("member")
                std_da = da.std("member")
            else:
                mean_da = da
                std_da = None

            # cftime-safe fractional year
            time_vals = mean_da["time"].values
            try:
                t = pd.to_datetime(time_vals)
                x = t.year + (t.month - 0.5) / 12.0
            except (TypeError, ValueError):
                x = np.array([
                    tt.year + (tt.month - 0.5) / 12.0
                    for tt in time_vals
                ])

            y = np.asarray(mean_da).squeeze()

            # Hide the gap between mid-term and long-term future periods
            y_masked = y.copy()
            mask_hide = (x > 2049) & (x < 2065)
            y_masked[mask_hide] = np.nan

            ax.plot(
                x,
                y_masked,
                color=comp_color,
                linestyle=scenario_linestyles[scen],
                linewidth=3.5,
                alpha=0.95,
            )

            if std_da is not None:
                ystd = np.asarray(std_da).squeeze()
                ystd_masked = ystd.copy()
                ystd_masked[mask_hide] = np.nan

                ax.fill_between(
                    x,
                    y_masked - ystd_masked,
                    y_masked + ystd_masked,
                    color=comp_color,
                    alpha=0.13,
                    linewidth=0,
                )

                valid_mask = ~np.isnan(y_masked) & (x >= 1980) & (x <= 2100)
                if valid_mask.any():
                    valid_vals = y_masked[valid_mask]
                    valid_std = ystd_masked[valid_mask]
                    y_values.extend([
                        valid_vals.max(),
                        valid_vals.min(),
                        (valid_vals + valid_std).max(),
                        (valid_vals - valid_std).min(),
                    ])
            else:
                valid_mask = ~np.isnan(y_masked) & (x >= 1980) & (x <= 2100)
                if valid_mask.any():
                    y_values.extend([
                        y_masked[valid_mask].max(),
                        y_masked[valid_mask].min(),
                    ])

    # ------------------------------------------------------------
    # Axis formatting
    # ------------------------------------------------------------
    ax.axhline(0, color="gray", linestyle="--", linewidth=1.0)
    ax.set_xlim(1980, 2100)
    # make the axis tick label font size larger
    ax.tick_params(axis='both', which='major', labelsize=12)

    if y_values:
        y_max = max(abs(np.nanmin(y_values)), abs(np.nanmax(y_values)))
        ax.set_ylim(-y_max, y_max)

    ax.set_ylabel("Heat-budget anomaly (TW)", fontsize=12)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_title("Arctic Ocean Heat Budget Components", fontsize=14)

    # ------------------------------------------------------------
    # Legends
    # ------------------------------------------------------------
    component_handles = [
        Line2D(
            [0], [0],
            color=component_colors[name],
            lw=2.8,
            linestyle="-",
            label=name,
        )
        for name in components.keys()
        if components[name]
    ]

    scenario_handles = [
        Line2D(
            [0], [0],
            color="black",
            lw=2.5,
            linestyle=scenario_linestyles[s],
            label=scenario_labels[s],
        )
        for s in ["historical", "ssp119", "ssp126"]
    ]

    leg1 = ax.legend(
        handles=component_handles,
        title="Component",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
        fontsize=9,
    )
    ax.add_artist(leg1)

    ax.legend(
        handles=scenario_handles,
        title="Scenario",
        loc="upper left",
        bbox_to_anchor=(1.02, 0.65),
        frameon=False,
        fontsize=9,
    )

    # ------------------------------------------------------------
    # Phase annotations
    # ------------------------------------------------------------
    ylim = ax.get_ylim()
    y_pos = ylim[0] + 0.08 * (ylim[1] - ylim[0])

    for year in [2015, 2049, 2065]:
        ax.axvline(
            year,
            color="gray",
            linestyle=":",
            linewidth=1.5,
            alpha=0.6,
            zorder=1,
        )

    ax.text(
        1997.5, y_pos, "Present-day",
        ha="center", va="bottom",
        fontsize=10, style="italic", color="black",
    )
    ax.text(
        2032.5, y_pos, "CO$_2$ increase",
        ha="center", va="bottom",
        fontsize=10, style="italic", color="black",
    )
    ax.text(
        2082.5, y_pos, "CO$_2$ decreasing",
        ha="center", va="bottom",
        fontsize=10, style="italic", color="black",
    )
# def plot_panel_a(ax, oht_total, ohs, nshf, qflux=None):
    # Compute OHT + SHF + Qflux: residual should approximate OHS if budget closes.
    # Uses the intersection of available time coordinates across all three terms.
    # oht_plus_shf_qflux = {}
    # for scen in SCENARIOS:
    #     terms = {}
    #     if scen in oht_total:
    #         terms["oht"] = oht_total[scen]
    #     if scen in nshf:
    #         terms["shf"] = nshf[scen]
    #     if qflux is not None and scen in qflux:
    #         terms["qflux"] = qflux[scen]

    #     if "oht" not in terms:
    #         continue

    #     # Align on common times across all available terms
    #     common_times = terms["oht"].time.values
    #     for v in list(terms.values())[1:]:
    #         common_times = np.intersect1d(common_times, v.time.values)

    #     if len(common_times) > 0:
    #         result = terms["oht"].sel(time=common_times)
    #         if "shf" in terms:
    #             result = result + terms["shf"].sel(time=common_times)
    #         if "qflux" in terms:
    #             result = result + terms["qflux"].sel(time=common_times)
    #         oht_plus_shf_qflux[scen] = result

    # components = {
    #     "OHT":             (oht_total,             "-"),
    #     "OHS":             (ohs,                   (0, (5, 1))),
    #     "SHF":             (nshf,                  (0, (1, 1))),
    #     "Qflux":           (qflux if qflux else {}, (0, (3, 1, 1, 1))),
    #     "OHT+SHF+Qflux":  (oht_plus_shf_qflux,    (0, (5, 1, 1, 1, 1, 1))),
    # }

    # y_values = []
    # for comp_name, (comp_dict, linestyle) in components.items():
    #     for scen in SCENARIOS:
    #         if scen not in comp_dict:
    #             continue

    #         da = comp_dict[scen]
    #         if "member" in da.dims:
    #             mean_da = da.mean("member")
    #             std_da = da.std("member")
    #         else:
    #             mean_da = da
    #             std_da = None

    #         # cftime-safe fractional year (handles both datetime64 and cftime objects)
    #         time_vals = mean_da["time"].values
    #         try:
    #             t = pd.to_datetime(time_vals)
    #             x = t.year + (t.month - 0.5) / 12.0
    #         except (TypeError, ValueError):
    #             x = np.array([tt.year + (tt.month - 0.5) / 12.0 for tt in time_vals])
    #         y = np.asarray(mean_da).squeeze()

    #         # Insert NaN values for 2050-2064 to create visible gaps
    #         # This prevents matplotlib from drawing lines between 2049-2065
    #         y_masked = y.copy()
    #         mask_hide = (x > 2049) & (x < 2065)  # Hide 2050-2064
    #         y_masked[mask_hide] = np.nan
            
    #         # Plot the full array - NaN creates gaps, xlim controls display range
    #         ax.plot(
    #             x, y_masked,
    #             color=SCENARIO_COLORS[scen],
    #             linestyle=linestyle,
    #             linewidth=3.5,
    #             alpha=0.95
    #         )

    #         if std_da is not None:
    #             ystd = np.asarray(std_da).squeeze()
    #             ystd_masked = ystd.copy()
    #             ystd_masked[mask_hide] = np.nan
                
    #             ax.fill_between(
    #                 x, y_masked - ystd_masked, y_masked + ystd_masked,
    #                 color=SCENARIO_COLORS[scen],
    #                 alpha=0.15, linewidth=0
    #             )
    #             # Calculate y_values only from non-NaN, non-hidden data
    #             valid_mask = ~np.isnan(y_masked) & (x >= 1980) & (x <= 2100)
    #             if valid_mask.any():
    #                 valid_vals = y_masked[valid_mask]
    #                 valid_std = ystd_masked[valid_mask]
    #                 y_values.extend([valid_vals.max(), valid_vals.min(), 
    #                                (valid_vals + valid_std).max(), (valid_vals - valid_std).min()])
    #         else:
    #             # Calculate y_values only from non-NaN data
    #             valid_mask = ~np.isnan(y_masked) & (x >= 1980) & (x <= 2100)
    #             if valid_mask.any():
    #                 y_values.extend([y_masked[valid_mask].max(), y_masked[valid_mask].min()])

    # ax.axhline(0, color="gray", linestyle="--", linewidth=1.0)
    # ax.set_xlim(1980, 2100)
    
    # # Set symmetric y-axis range
    # if y_values:
    #     y_max = max(abs(min(y_values)), abs(max(y_values)))
    #     ax.set_ylim(-y_max, y_max)
    
    # ax.set_ylabel("Heat budget term (TW)")
    # ax.set_xlabel("Year")
    # ax.set_title("Arctic Ocean Heat Budget Components", fontsize=14)

    # scenario_handles = [
    #     Line2D([0], [0], color=SCENARIO_COLORS[s], lw=2.0, linestyle="-", label=s)
    #     for s in ["historical", "ssp119", "ssp126"]
    # ]
    # component_handles = [
    #     Line2D([0], [0], color="black", lw=2.0, linestyle=ls, label=name)
    #     for name, (_, ls) in components.items()
    # ]

    # # Place legends outside the plot area, stacked vertically, without frames
    # leg1 = ax.legend(
    #     handles=scenario_handles, title="Scenario",
    #     loc="upper left", bbox_to_anchor=(1.02, 1.0),
    #     frameon=False, fontsize=9
    # )
    # ax.add_artist(leg1)
    # ax.legend(
    #     handles=component_handles, title="Component",
    #     loc="upper left", bbox_to_anchor=(1.02, 0.65),
    #     frameon=False, fontsize=9
    # )
    
    # # Add phase annotations above x-axis
    # ylim = ax.get_ylim()
    # y_pos = ylim[0] + 0.08 * (ylim[1] - ylim[0])  # Position just above bottom
    
    # # Phase boundaries
    # phase_transitions = [2015, 2049]
    
    # # Add vertical lines to mark phase transitions
    # for year in phase_transitions:
    #     ax.axvline(year, color='gray', linestyle=':', linewidth=1.5, alpha=0.6, zorder=1)
    
    # # Add phase labels
    # ax.text(1997.5, y_pos, 'Present-day', ha='center', va='bottom', 
    #         fontsize=10, fontweight='normal', style='italic', color='black')
    # ax.text(2032.5, y_pos, 'CO$_2$ increase', ha='center', va='bottom',
    #         fontsize=10, fontweight='normal', style='italic', color='black')
    # ax.text(2082.5, y_pos, 'CO$_2$ decreasing', ha='center', va='bottom',
    #         fontsize=10, fontweight='normal', style='italic', color='black')

def plot_oht_profile_panel(ax, ds_gateway_OHT_mem, ds_gateway_OHT_mean, gateway='total'):
    """Plot OHT profile for total gateway with proper styling."""
    period_color = {
        '1980-2014': 'black',
        '2015-2049': "#407BD0",
        '2065-2099': "#A32A31",
    }
    
    # Plot 1980-2014 (historical)
    hist_mean = ds_gateway_OHT_mean[gateway]['historical'].mean(dim='time').squeeze()
    hist_std = xr.concat(ds_gateway_OHT_mem[gateway]['historical'], dim='member').mean(dim='time').std(dim='member').squeeze()
    
    depth_coord = 'depth_2' if 'depth_2' in hist_mean.coords else get_depth_coord(hist_mean)
    depth_vals = hist_mean[depth_coord].values
    depth_compressed = map_depth_to_compressed(depth_vals)
    
    ax.fill_betweenx(depth_compressed, 
                     (hist_mean - hist_std).values, 
                     (hist_mean + hist_std).values, 
                     color='black', alpha=0.2)
    ax.plot(hist_mean.values, depth_compressed, 
            color='black', linewidth=2, label='1980-2014')
    
    # Plot SSP119 periods
    for idx, period in enumerate(['2015-2049', '2065-2099']):
        ssp119_mean = ds_gateway_OHT_mean[gateway]['ssp119'].sel(period=period).mean(dim='time').squeeze()
        ssp119_std = ds_gateway_OHT_mem[gateway]['ssp119'][idx].mean(dim='time').std(dim='member').squeeze()
        
        depth_coord = 'depth_2' if 'depth_2' in ssp119_mean.coords else get_depth_coord(ssp119_mean)
        depth_vals = ssp119_mean[depth_coord].values
        depth_compressed = map_depth_to_compressed(depth_vals)
        
        ax.fill_betweenx(depth_compressed, 
                         (ssp119_mean - ssp119_std).values, 
                         (ssp119_mean + ssp119_std).values, 
                         color=period_color[period], alpha=0.2)
        ax.plot(ssp119_mean.values, depth_compressed, 
                color=period_color[period], linewidth=2, linestyle=':', 
                label=f'{period} (SSP119)')
    
    # Plot SSP126 periods
    for idx, period in enumerate(['2015-2049', '2065-2099']):
        ssp126_mean = ds_gateway_OHT_mean[gateway]['ssp126'].sel(period=period).mean(dim='time').squeeze()
        # Handle 'month' dimension if present
        if 'month' in ssp126_mean.dims:
            ssp126_mean = ssp126_mean.mean(dim='month')
        ssp126_std = ds_gateway_OHT_mem[gateway]['ssp126'][idx].mean(dim='time').std(dim='member').squeeze()
        if 'month' in ssp126_std.dims:
            ssp126_std = ssp126_std.mean(dim='month')
        
        depth_coord = 'depth_2' if 'depth_2' in ssp126_mean.coords else get_depth_coord(ssp126_mean)
        depth_vals = ssp126_mean[depth_coord].values
        depth_compressed = map_depth_to_compressed(depth_vals)
        
        ax.fill_betweenx(depth_compressed, 
                         (ssp126_mean - ssp126_std).values, 
                         (ssp126_mean + ssp126_std).values, 
                         color=period_color[period], alpha=0.15)
        ax.plot(ssp126_mean.values, depth_compressed, 
                color=period_color[period], linewidth=2, linestyle='-.', 
                label=f'{period} (SSP126)')
    
    # Apply custom depth axis transformation
    transform_depth_axis(ax)
    
    ax.set_xlabel('OHT (TW)', fontsize=16)
    ax.set_ylabel('Depth (m)', fontsize=16)
    ax.grid(False)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1.5)
    ax.text(-0.1, 1.1, 'b', transform=ax.transAxes, fontsize=18, fontweight='bold', va='top', ha='right')

def plot_ovt_profile_panel(ax, ds_gateway_OVT_mem, ds_gateway_OVT_mean, gateway='total'):
    """Plot OVT profile for total gateway with proper styling."""
    period_color = {
        '1980-2014': 'black',
        '2015-2049': "#407BD0",
        '2065-2099': "#A32A31",
    }
    
    # Plot 1980-2014 (historical)
    hist_mean = ds_gateway_OVT_mean[gateway]['historical'].mean(dim='time').squeeze()
    hist_std = xr.concat(ds_gateway_OVT_mem[gateway]['historical'], dim='member').mean(dim='time').std(dim='member').squeeze()
    
    depth_coord = 'lev' if 'lev' in hist_mean.coords else get_depth_coord(hist_mean)
    depth_vals = hist_mean[depth_coord].values
    depth_compressed = map_depth_to_compressed(depth_vals)
    
    ax.fill_betweenx(depth_compressed, 
                     (hist_mean - hist_std).values, 
                     (hist_mean + hist_std).values, 
                     color='black', alpha=0.2)
    ax.plot(hist_mean.values, depth_compressed, 
            color='black', linewidth=2, label='1980-2014')
    
    # Plot SSP119 periods
    for idx, period in enumerate(['2015-2049', '2065-2099']):
        ssp119_mean = ds_gateway_OVT_mean[gateway]['ssp119'].sel(period=period).mean(dim='time').squeeze()
        ssp119_std = ds_gateway_OVT_mem[gateway]['ssp119'][idx].mean(dim='time').std(dim='member').squeeze()
        
        depth_coord = 'lev' if 'lev' in ssp119_mean.coords else get_depth_coord(ssp119_mean)
        depth_vals = ssp119_mean[depth_coord].values
        depth_compressed = map_depth_to_compressed(depth_vals)
        
        ax.fill_betweenx(depth_compressed, 
                         (ssp119_mean - ssp119_std).values, 
                         (ssp119_mean + ssp119_std).values, 
                         color=period_color[period], alpha=0.2)
        ax.plot(ssp119_mean.values, depth_compressed, 
                color=period_color[period], linewidth=2, linestyle=':', 
                label=f'{period} (SSP119)')
    
    # Plot SSP126 periods
    for idx, period in enumerate(['2015-2049', '2065-2099']):
        ssp126_mean = ds_gateway_OVT_mean[gateway]['ssp126'].sel(period=period).mean(dim='time').squeeze()
        ssp126_std = ds_gateway_OVT_mem[gateway]['ssp126'][idx].mean(dim='time').std(dim='member').squeeze()
        
        depth_coord = 'lev' if 'lev' in ssp126_mean.coords else get_depth_coord(ssp126_mean)
        depth_vals = ssp126_mean[depth_coord].values
        depth_compressed = map_depth_to_compressed(depth_vals)
        
        ax.fill_betweenx(depth_compressed, 
                         (ssp126_mean - ssp126_std).values, 
                         (ssp126_mean + ssp126_std).values, 
                         color=period_color[period], alpha=0.15)
        ax.plot(ssp126_mean.values, depth_compressed, 
                color=period_color[period], linewidth=2, linestyle='-.', 
                label=f'{period} (SSP126)')
    
    # Apply custom depth axis transformation
    transform_depth_axis(ax)
    
    ax.set_xlabel('OVT (Sv)', fontsize=16)
    ax.set_ylabel('Depth (m)', fontsize=16)
    ax.grid(False)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1.5)
    ax.text(-0.1, 1.1, 'c', transform=ax.transAxes, fontsize=18, fontweight='bold', va='top', ha='right')

def plot_thetao_profile_panel(ax, profile_dict):
    """Plot temperature profile with updated styling."""
    period_color = {
        '1980-2014': 'black',
        '2015-2049': "#407BD0",
        '2065-2099': "#A32A31",
    }
    
    for scen in SCENARIOS:
        for period in SCENARIOS_PERIODS[scen]:
            if scen not in profile_dict or period not in profile_dict[scen]:
                continue

            prof = profile_dict[scen][period]
            x = np.asarray(prof["mean"])
            s = np.asarray(prof["std"])
            z = np.asarray(prof["depth"])
            z_compressed = map_depth_to_compressed(z)

            color = period_color.get(period, PERIOD_COLORS[period])
            linestyle = PROFILE_LINESTYLES[scen]

            ax.plot(x, z_compressed, color=color, linestyle=linestyle, linewidth=2.0)
            ax.fill_betweenx(z_compressed, x - s, x + s, color=color, alpha=0.16)

    # Apply custom depth axis transformation
    transform_depth_axis(ax)
    
    ax.axvline(0, color="gray", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Temperature anomaly (°C)", fontsize=16)
    ax.set_ylabel("Depth (m)", fontsize=16)
    ax.grid(False)
    ax.text(-0.1, 1.1, 'd', transform=ax.transAxes, fontsize=18, fontweight='bold', va='top', ha='right')
    
    # Move legends outside the panel, similar to panel a
    period_handles = [
        Line2D([0], [0], color='black', lw=2, label='1980-2014'),
        Line2D([0], [0], color="#407BD0", lw=2, label='2015-2049'),
        Line2D([0], [0], color="#A32A31", lw=2, label='2065-2099'),
    ]
    scenario_handles = [
        Line2D([0], [0], color='black', lw=2, linestyle='-', label='historical'),
        Line2D([0], [0], color='black', lw=2, linestyle=':', label='ssp119'),
        Line2D([0], [0], color='black', lw=2, linestyle='-.', label='ssp126'),
    ]
    
    # Place legends outside, stacked vertically, without frames (matching panel a style)
    leg1 = ax.legend(
        handles=period_handles, title="Period",
        loc="upper left", bbox_to_anchor=(1.02, 1.0),
        frameon=False, fontsize=8
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=scenario_handles, title="Scenario",
        loc="upper left", bbox_to_anchor=(1.02, 0.65),
        frameon=False, fontsize=8
    )
# %%
# ============================================================
# -------------------------- MAIN -----------------------------
# ============================================================
def main():
    print("Loading panel-a time series...")
    hfds = load_heat_budget_component("hfds")
    ohs = load_ohs_timeseries()
    oht_total = load_oht_timeseries_total()
    qice = load_qice_timeseries()

    print("\nLoading panel-b/c/d profiles...")
    ds_gateway_OHT_mem, ds_gateway_OHT_mean = load_oht_profiles_with_members(gateway=GATEWAY_FOR_PANEL_B)
    ds_gateway_OVT_mem, ds_gateway_OVT_mean = load_ovt_profiles_with_members(gateway=GATEWAY_FOR_PANEL_C)
    thetao_profiles = load_thetao_profiles()

    print("\nCreating figure...")
    fig = plt.figure(figsize=(16, 10.5))

    # Outer grid: 2 rows with independent spacing per row
    gs_outer = gridspec.GridSpec(
        2, 1,
        height_ratios=[1.2, 1.0],
        hspace=0.3,
        figure=fig,
    )

    # Top row: panel a centered (~60% width); large wspace to leave room for legend
    gs_top = gridspec.GridSpecFromSubplotSpec(
        1, 3,
        subplot_spec=gs_outer[0],
        width_ratios=[1, 3, 1],
        wspace=0.1,
    )
    ax_a = fig.add_subplot(gs_top[0, 1])

    # Bottom row: b/c/d use full width with tighter spacing
    gs_bot = gridspec.GridSpecFromSubplotSpec(
        1, 3,
        subplot_spec=gs_outer[1],
        wspace=0.40,
    )
    ax_b = fig.add_subplot(gs_bot[0, 0])
    ax_c = fig.add_subplot(gs_bot[0, 1])
    ax_d = fig.add_subplot(gs_bot[0, 2])

    plot_panel_a(ax_a, oht_total=oht_total, ohs=ohs, nshf=hfds, qflux=qice)
    # add_gateway_inset_mask(ax_a)
    add_panel_label(ax_a, "a")

    plot_oht_profile_panel(ax_b, ds_gateway_OHT_mem, ds_gateway_OHT_mean, gateway=GATEWAY_FOR_PANEL_B)
    plot_ovt_profile_panel(ax_c, ds_gateway_OVT_mem, ds_gateway_OVT_mean, gateway=GATEWAY_FOR_PANEL_C)
    plot_thetao_profile_panel(ax_d, thetao_profiles)

    fig.subplots_adjust(right=0.95)  # Make room for external legends

    out_png = f"{FIG_OUT}/Fig4_merged_heat_budget_profiles_fulldepth.png"
    out_pdf = f"{FIG_OUT}/Fig4_merged_heat_budget_profiles_fulldepth.pdf"

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    print(f"\nSaved:\n  {out_png}\n  {out_pdf}")

    plt.show()
# %%
if __name__ == "__main__":
    main()
# %%
