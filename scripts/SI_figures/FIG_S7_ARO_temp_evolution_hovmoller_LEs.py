# %%
import os
from time import time
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %%
# define a helper function to standardize the vertical dimension to 'lev'
def standardize_vertical_dim(ds, varname="thetao_hovmoller_anom"):
    """
    Make sure the Hovmoller variable has only one vertical dimension: lev.
    Handles files with depth, lev, or both.
    """

    da = ds[varname]

    # Case 1: both depth and lev are dimensions -> usually duplicate vertical axes
    if "depth" in da.dims and "lev" in da.dims:
        # If depth has length 1, remove it
        if da.sizes["depth"] == 1:
            ds[varname] = da.isel(depth=0, drop=True)
        # If lev has length 1, remove it
        elif da.sizes["lev"] == 1:
            ds[varname] = da.isel(lev=0, drop=True).rename({"depth": "lev"})
        else:
            raise ValueError(
                f"{varname} has both depth and lev as real dimensions: {da.dims}. "
                "Please inspect this file manually."
            )

    # Case 2: vertical dimension is depth only
    elif "depth" in da.dims and "lev" not in da.dims:
        ds = ds.rename({"depth": "lev"})

    # Remove non-dimension depth coordinate if still present
    if "depth" in ds.coords and "depth" not in ds.dims:
        ds = ds.drop_vars("depth")

    return ds
# %%
dir_in = "/work/mh0033/m301036/Atlantic_heat_capacitor_Arctic/Atlantic_capacitor_on_Arctic/scripts/Nonlinear_diagnosis/ARO_temp_profile/output/"
dir_interp_in = "/work/mh0033/m301036/Atlantic_heat_capacitor_Arctic/Atlantic_capacitor_on_Arctic/scripts/Nonlinear_diagnosis/ARO_temp_profile/MIROC_revise_sigmaInterp/output_sigmaInterp/"
# Define members per model with correct variant labels
MEMBERS_BY_MODEL = {
    "MIROC6": [f"r{i}i1p1f1" for i in range(1, 51)],  # 50 members
    "MIROC-ES2L": [f"r{i}i1p1f2" for i in range(1, 11)],  # 10 members
    "MPI-ESM1-2-LR": [f"r{i}i1p1f1" for i in range(1, 51)],  # 50 members
    "CanESM5": [f"r{i}i1p1f1" for i in range(1, 26)] + [f"r{i}i1p2f1" for i in range(1, 26)],  # 50 members
}

scenarios = ["ssp119", "ssp126"]
LENS_name = ["MIROC6", "MIROC-ES2L", "MPI-ESM1-2-LR", "CanESM5"]

ds_all_by_model_scenario = {}

for model in LENS_name:
    ds_all_by_model_scenario[model] = {}
    members = MEMBERS_BY_MODEL[model]  # Use model-specific member list

    for scenario in scenarios:
        ds_list = []

        # Define scenario directory based on model-specific structure
        if model == "MPI-ESM1-2-LR":
            scenario_dir = dir_in
        elif model == "MIROC-ES2L":
            # MIROC-ES2L files are directly in scenario folder (no hovmoller subdirectory)
            scenario_dir = f"{dir_interp_in}{model}/{scenario}/"
        elif model == "MIROC6":
            # MIROC6 files are in hovmoller subdirectory
            scenario_dir = f"{dir_in}{model}/{scenario}/hovmoller/"
        else:
            # CanESM5 and MIROC6 have hovmoller subdirectory
            scenario_dir = f"{dir_in}{model}/{scenario}/hovmoller/"

        for member in members:
            if model == "MPI-ESM1-2-LR":
                in_file = (
                    f"{dir_in}{model}_{scenario}_{member}"
                    f"_Arctic_thetao_hov_vs_1850-1900.nc"
                )
            else:
                in_file = (
                    f"{scenario_dir}{model}_{scenario}_{member}"
                    f"_Arctic_thetao_hov_vs_1850-1900.nc"
                )

            if not os.path.exists(in_file):
                print(f"Missing file: {in_file}")
                continue

            ds = xr.open_dataset(in_file)

            ds = standardize_vertical_dim(ds, varname="thetao_hovmoller_anom")

            ds = ds.expand_dims(member=[member])
            ds = ds.assign_coords(model=model, scenario=scenario)
            ds_list.append(ds)

        if not ds_list:
            print(f"No files found for {model} {scenario}")
            continue

        ds_all = xr.concat(ds_list, dim="member", coords='minimal')
        ds_all_by_model_scenario[model][scenario] = ds_all

        os.makedirs(scenario_dir, exist_ok=True)
        out_file = (
            f"{scenario_dir}{model}_{scenario}"
            f"_all_members_Arctic_thetao_hov_vs_1850-1900.nc"
        )
        # ds_all.to_netcdf(out_file)
# %%
# Calculate annual mean temp anomaly for ALL models and scenarios
ds_annual_by_model_scenario = {}

for model in LENS_name:
    ds_annual_by_model_scenario[model] = {}
    
    for scenario in scenarios:
        if scenario not in ds_all_by_model_scenario[model]:
            print(f"No data for {model} {scenario}")
            continue
            
        ds = ds_all_by_model_scenario[model][scenario]
        ds = standardize_vertical_dim(ds, varname="thetao_hovmoller_anom")

        ds_annual = (
            ds["thetao_hovmoller_anom"]
            .resample(time="YS")
            .mean()
            .to_dataset(name="thetao_hovmoller_anom")
        )

        ds_annual = standardize_vertical_dim(ds_annual, varname="thetao_hovmoller_anom")
        
        # Drop 'depth' coordinate if it exists (standardize on 'lev')
        if 'depth' in ds_annual.coords:
            ds_annual = ds_annual.drop_vars('depth')
        
        ds_annual_by_model_scenario[model][scenario] = ds_annual

        # Save annual mean - use correct directory structure
        if model == "MPI-ESM1-2-LR":
            scenario_dir = dir_in
        elif model == "MIROC-ES2L":
            scenario_dir = f"{dir_interp_in}{model}/{scenario}/"
        elif model == "MIROC6":
            scenario_dir = f"{dir_in}{model}/{scenario}/hovmoller/"
        else:
            scenario_dir = f"{dir_in}{model}/{scenario}/hovmoller/"
        
        os.makedirs(scenario_dir, exist_ok=True)
        out_file = f"{scenario_dir}{model}_{scenario}_all_members_Arctic_thetao_hov_annual_vs_1850-1900.nc"
        # ds_annual.to_netcdf(out_file)
        print(f"Saved: {out_file}")
# %%
# For backward compatibility: keep MPI-ESM accessible as before
ds_all_by_scenario = ds_all_by_model_scenario["MPI-ESM1-2-LR"]
ds_annual_by_scenario = ds_annual_by_model_scenario["MPI-ESM1-2-LR"]
# %%
# Calculate ensemble mean and std for ALL models and scenarios
print("\n=== Computing ensemble statistics for all models ===")

for model in LENS_name:
    if model not in ds_annual_by_model_scenario:
        print(f"Skipping {model} - no data loaded")
        continue
    
    for scenario in scenarios:
        if scenario not in ds_annual_by_model_scenario[model]:
            print(f"Skipping {model} {scenario} - no data")
            continue
        
        ds = ds_annual_by_model_scenario[model][scenario]
   
        # Ensemble mean (across all members)
        ds_ens_mean = ds.mean(dim="member")
        
        # Ensemble spread (std across members)
        ds_ens_std = ds.std(dim="member")
        
        # Get member count
        n_members = ds.sizes["member"]
        
        print(f"\n{model} {scenario.upper()}:")
        print(f"  Members: {n_members}")
        print(f"  Annual mean shape: {ds['thetao_hovmoller_anom'].shape}")
        print(f"  Time range: {ds.time.dt.year.min().values} - {ds.time.dt.year.max().values}")
        print(f"  Ensemble mean anomaly range: {float(ds_ens_mean['thetao_hovmoller_anom'].min()):.3f} to {float(ds_ens_mean['thetao_hovmoller_anom'].max()):.3f} °C")
        
        # Save ensemble statistics to correct directory
        if model == "MPI-ESM1-2-LR":
            scenario_dir = dir_in
        elif model == "MIROC-ES2L":
            scenario_dir = f"{dir_in}{model}/{scenario}/"
        else:
            scenario_dir = f"{dir_in}{model}/{scenario}/hovmoller/"
        
        os.makedirs(scenario_dir, exist_ok=True)
        
        out_ens_mean = f"{scenario_dir}{model}_{scenario}_ensemble_mean_Arctic_thetao_annual_vs_1850-1900.nc"
        out_ens_std = f"{scenario_dir}{model}_{scenario}_ensemble_std_Arctic_thetao_annual_vs_1850-1900.nc"
        
        # ds_ens_mean.to_netcdf(out_ens_mean)
        # ds_ens_std.to_netcdf(out_ens_std)
        
        print(f"  Saved: {os.path.basename(out_ens_mean)}")
        print(f"  Saved: {os.path.basename(out_ens_std)}")

print("\n=== Ensemble statistics computation complete ===\n")
# %%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
# Plot 4 models ensemble mean temperature evolution for both scenarios
# Upper 2000 m depth T_anom, 3 time periods
for scenario in scenarios:
    print(f"\n{'='*70}")
    print(f"{scenario.upper()}")
    print(f"{'='*70}")
    for model in ["CanESM5", "MIROC6", "MIROC-ES2L", "MPI-ESM1-2-LR"]:
        if model in ds_annual_by_model_scenario and scenario in ds_annual_by_model_scenario[model]:
            ds = ds_annual_by_model_scenario[model][scenario]
            years = sorted([int(y) for y in ds.time.dt.year.values])
            print(f"{model}: {len(years)} years ({years[0]}-{years[-1]})")
        else:
            print(f"{model}: NO DATA")
# %%
time_periods = {
    "1980–2014": slice("1980", "2014"),
    "2015–2049": slice("2015", "2049"),
    "2065–2099": slice("2065", "2099"),
}
zname = "lev"
tname = "time"

# asymmetric diverging scale with 0 = white
# vmin, vmax = -0.5, 1.0
# levels = np.linspace(vmin, vmax, 21)
# norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
vmin, vmax = -0.5, 1.0
# levels = np.linspace(vmin, vmax, 25)
levels = np.linspace(-0.5, 1.0, 21)
norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
cmap = plt.get_cmap("RdBu_r")  # more perceptually uniform for positive anomalies

# Model display order and labels
models_ordered = ["CanESM5", "MIROC6", "MIROC-ES2L", "MPI-ESM1-2-LR"]
scenario_labels = {"ssp119": "SSP1-1.9", "ssp126": "SSP1-2.6"}

# Diagnose the maximum temperature anomaly in the upper 2000 m
# for each model, each scenario, and each time period
print("\n=== Maximum temperature anomaly diagnostics (upper 2000 m) ===")
for model in models_ordered:
    if model not in ds_annual_by_model_scenario:
        continue
    for scenario in scenarios:
        if scenario not in ds_annual_by_model_scenario[model]:
            continue
        ds = ds_annual_by_model_scenario[model][scenario]
        # Ensemble mean over members
        da_ens_mean = ds['thetao_hovmoller_anom'].mean(dim='member')
        # Restrict to upper 2000 m
        da_ens_mean = da_ens_mean.sel(lev=slice(None, 2000))
        print(f"\n  {model} | {scenario_labels.get(scenario, scenario.upper())}:")
        for period_name, time_slice in time_periods.items():
            da_period = da_ens_mean.sel(time=time_slice)
            if da_period.sizes['time'] == 0:
                print(f"    {period_name}: no data")
                continue
            max_anom = float(da_period.max())
            # Stack to 1D to find the joint (time, lev) location of the global max
            da_flat = da_period.stack(pts=('time', 'lev'))
            loc = da_flat.pts.isel(pts=int(da_flat.argmax(dim='pts')))
            max_year = int(loc.time.dt.year)
            max_depth = int(loc.lev)
            print(f"    {period_name}: max = {max_anom:.3f} °C  |  year = {max_year}  |  depth = {max_depth} m")
print("\n=== Diagnostics complete ===\n")
# Skip the multi-model (4LEs/MMLE) mean; plot each model's own evolution below.
# %%
"""
=================================================================================
4 Large Ensembles Hovmoller Comparison Plots
=================================================================================
Plot 4 models × 3 time periods Hovmoller diagrams for comprehensive comparison
across CanESM5, MIROC6, MIROC-ES2L, and MPI-ESM1-2-LR
"""
# Display settings follow Figure 2's typography and tick styling.
TITLE_FONTSIZE = 18
SCENARIO_FONTSIZE = 18
MODEL_FONTSIZE = 16
AXIS_LABEL_FONTSIZE = 14
TICK_FONTSIZE = 13
TICK_LENGTH = 7
TICK_WIDTH = 1.4
COLORBAR_LABEL_FONTSIZE = 16
SHOW_BOTTOM_YEAR_LABELS = True

def plot_4LEs_hovmoller_3periods(scenario=None, save_fig=True):
    """Plot both scenarios as stacked model blocks with one shared colorbar.

    Pass a scenario name to retain the optional single-scenario view.
    Labels show nominal ensemble sizes, except CanESM5 SSP1-2.6, which
    shows the actual loaded count. The plotted data selection is unchanged.
    """
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter

    plot_scenarios = list(scenarios) if scenario is None else [scenario]
    models_ordered = ["CanESM5", "MIROC6", "MIROC-ES2L", "MPI-ESM1-2-LR"]
    model_labels = {
        "CanESM5": "CanESM5",
        "MIROC6": "MIROC6",
        "MIROC-ES2L": "MIROC-ES2L",
        "MPI-ESM1-2-LR": "MPI-ESM1.2-LR",
    }
    scenario_labels = {"ssp119": "SSP1-1.9", "ssp126": "SSP1-2.6"}
    period_titles = [
        "Historical\n1980–2014",
        "CO$_2$ increasing\n2015–2049",
        "CO$_2$ decreasing\n2065–2099",
    ]

    models_with_data = []
    for model in models_ordered:
        if all(s in ds_annual_by_model_scenario.get(model, {}) for s in plot_scenarios):
            models_with_data.append(model)
        else:
            print(f"Warning: Skipping {model}; missing data for a requested scenario")
    if not models_with_data:
        print("ERROR: No models with data for all requested scenarios")
        return None

    n_models = len(models_with_data)
    nrows = n_models * len(plot_scenarios)
    ncols = len(time_periods)
    fig = plt.figure(figsize=(16, 2.4 * nrows + 2.0))
    outer = fig.add_gridspec(
        len(plot_scenarios), 1, left=0.19, right=0.98,
        top=0.94, bottom=0.11, hspace=0.035,
    )
    axes = np.empty((nrows, ncols), dtype=object)

    for block, scenario_name in enumerate(plot_scenarios):
        grid = outer[block].subgridspec(n_models, ncols, hspace=0.10, wspace=0.08)
        model_label_texts = []
        for model_row, model in enumerate(models_with_data):
            row = block * n_models + model_row
            ds = ds_annual_by_model_scenario[model][scenario_name]
            n_members = (ds.sizes["member"]
                         if model == "CanESM5" and scenario_name == "ssp126"
                         else len(MEMBERS_BY_MODEL[model]))
            T_anom = ds["thetao_hovmoller_anom"].mean(dim="member").squeeze(drop=True)
            if "depth" in T_anom.dims and "lev" not in T_anom.dims:
                T_anom = T_anom.rename({"depth": "lev"})
            if "depth" in T_anom.coords and "depth" not in T_anom.dims:
                T_anom = T_anom.drop_vars("depth")

            for column, (_, time_slice) in enumerate(time_periods.items()):
                ax = fig.add_subplot(grid[model_row, column])
                axes[row, column] = ax
                hov = T_anom.sel(time=time_slice).transpose("lev", "time")
                time_years = hov["time"].dt.year + (hov["time"].dt.dayofyear - 1) / 365.25
                cf = ax.contourf(
                    time_years, hov["lev"], hov,
                    levels=levels, cmap=cmap, norm=norm, extend="both",
                )
                ax.set_xlim(time_years.values[0], time_years.values[-1])
                ax.set_ylim(2000, 0)
                ax.xaxis.set_major_locator(MultipleLocator(5))
                ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
                ax.yaxis.set_major_locator(MultipleLocator(500))
                ax.minorticks_off()
                ax.tick_params(
                    axis="both", which="major", direction="out",
                    length=TICK_LENGTH, width=TICK_WIDTH, labelsize=TICK_FONTSIZE,
                    top=False, right=False, color="0.2",
                )
                bottom_labels = SHOW_BOTTOM_YEAR_LABELS and row == nrows - 1
                ax.tick_params(axis="x", labelbottom=bottom_labels)
                ax.tick_params(axis="y", labelleft=(column == 0))
                if bottom_labels:
                    ax.set_xlabel("Year", fontsize=AXIS_LABEL_FONTSIZE)
                if column == 0:
                    # One letter identifies each model row across all three periods.
                    ax.text(
                        -0.2, 1.02, chr(ord("a") + row), transform=ax.transAxes,
                        fontsize=24, fontweight="bold", ha="center", va="top",
                        clip_on=False,
                    )
                    ax.set_ylabel("Depth (m)", fontsize=AXIS_LABEL_FONTSIZE)
                    model_text = ax.text(
                        -0.29, 0.5, f"{model_labels[model]}\n({n_members})",
                        transform=ax.transAxes, rotation=90,
                        ha="center", va="center", fontsize=MODEL_FONTSIZE,
                    )
                    model_label_texts.append(model_text)
                if row == 0:
                    ax.set_title(period_titles[column], fontsize=TITLE_FONTSIZE,
                                 fontweight="bold", pad=12)

        # Center the scenario label beside all model rows in its block.
        top = axes[block * n_models, 0].get_position().y1
        bottom = axes[(block + 1) * n_models - 1, 0].get_position().y0
        scenario_text = fig.text(
            0.0, (top + bottom) / 2, scenario_labels[scenario_name],
            rotation=90, ha="center", va="center",
            fontsize=SCENARIO_FONTSIZE, fontweight="bold",
        )
        # Keep the scenario beside the model labels, separated by six points.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        to_figure = fig.transFigure.inverted()
        model_left = min(text.get_window_extent(renderer).transformed(to_figure).x0
                         for text in model_label_texts)
        scenario_width = scenario_text.get_window_extent(renderer).transformed(to_figure).width
        gap = (6.0 / 72.0) / fig.get_figwidth()
        scenario_text.set_x(model_left - gap - scenario_width / 2)

    # All panels retain the same normalization and share one bottom colorbar.
    cbar_ax = fig.add_axes([0.25, 0.055, 0.65, 0.012])
    cbar = fig.colorbar(cf, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Arctic Ocean temperature anomaly wrt 1850–1900 (°C)",
                   fontsize=COLORBAR_LABEL_FONTSIZE)
    tick_locs = np.arange(vmin, vmax + 0.001, 0.25)
    cbar.set_ticks(tick_locs)
    cbar.set_ticklabels([f"{value:g}" for value in tick_locs])
    cbar.ax.tick_params(labelsize=TICK_FONTSIZE, length=TICK_LENGTH,
                        width=TICK_WIDTH, color="0.2")
    cbar.outline.set_linewidth(1.2)
    cbar.ax.minorticks_off()

    if save_fig:
        output_dir = (
            "/work/mh0033/m301036/Atlantic_heat_capacitor_Arctic/"
            "Atlantic_capacitor_on_Arctic/docs/figs/SI_Figs"
        )
        os.makedirs(output_dir, exist_ok=True)
        scenario_tag = "_".join(plot_scenarios)
        stem = f"4LEs_Arctic_thetao_hovmoller_{scenario_tag}_3periods_2000m"
        for extension in ("png", "pdf"):
            out_file = os.path.join(output_dir, f"{stem}.{extension}")
            fig.savefig(out_file, dpi=300, bbox_inches="tight")
            print(f"Saved: {out_file}")

    plt.show()
    return fig
# %%
# Upper four model rows: SSP1-1.9; lower four: SSP1-2.6.
plot_4LEs_hovmoller_3periods(save_fig=True)
print("\n=== Combined 4LEs Hovmoller plot completed! ===")
# %%
