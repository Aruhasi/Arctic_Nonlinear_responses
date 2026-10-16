# %%
import xarray as xr
import numpy as np
import os
# %%
output_dir = "/work/mh0033/m301036/Atlantic_heat_capacitor_Arctic/Atlantic_capacitor_on_Arctic/docs/data/Spatial_pattern/pattern_diff_4LEs/"
# Create output directory if not exists
os.makedirs(output_dir, exist_ok=True)
LENS_name = ["MIROC", "MIROC-ES2L", "MPI-ESM-LR", "CanESM5"]
scenarios = ["ssp119", "ssp126"]
# %%
# input ssp119 difference data
Beta_LT_diff_ds= xr.open_mfdataset(output_dir + 'Beta_pattern_mean_long_vs_Present_differences_Among_LEs.nc')
Beta_MT_diff_ds= xr.open_mfdataset(output_dir + 'Beta_pattern_mean_mid_vs_Present_differences_Among_LEs.nc')
Beta_LT_vs_MT_ds= xr.open_mfdataset(output_dir + 'Beta_pattern_mean_long_vs_mid_differences_Among_LEs.nc')
# %%
time_frames = ["MT_vs_Present", "LT_vs_Present","LT_vs_MT"]
ds_map = [Beta_MT_diff_ds, Beta_LT_diff_ds, Beta_LT_vs_MT_ds]

per_tf = []
for tf_name, ds in zip(time_frames, ds_map):
    # collect DataArrays for the scenarios in the desired order
    arrs = [ds[scen] for scen in scenarios]            # list of DataArray (lat,lon,...)
    # concat them into a single DataArray with a new 'scenario' dim
    stacked = xr.concat(arrs, dim="scenario")
    stacked = stacked.assign_coords({"scenario": np.array(scenarios, dtype=object)})
    # add a time_frame dimension so we can concat across time_frames later
    stacked = stacked.expand_dims({"time_frame": [tf_name]})
    per_tf.append(stacked)

# final combined DataArray: dims ('time_frame','scenario', 'lat','lon', ...)
combined = xr.concat(per_tf, dim="time_frame")
combined.name = "beta_diff" 
# %%
# FDR significance for the member-wise contrast plotted in each column.
# A multi-model cell is hatched when at least 3 of the 4 large ensembles
# independently pass BH-FDR q <= 0.05 for that contrast.
uncertainty_dir = (
    "/work/mh0033/m301036/Atlantic_heat_capacitor_Arctic/"
    "Atlantic_capacitor_on_Arctic/scripts/SpatialPattern/"
    "data_35yr_segments/output/uncertainty_statistics_vs_HIST/"
)
FDR_REQUIRED_MODELS = 3
FDR_CONTRAST_BY_TIMEFRAME = {
    "MT_vs_Present": "mid_minus_hist",
    "LT_vs_Present": "late_minus_hist",
    "LT_vs_MT": "late_minus_mid",
}

def load_fdr_significant(time_frame, scenario):
    """Return cells FDR-significant in at least 3 of the 4 LEs."""
    contrast = FDR_CONTRAST_BY_TIMEFRAME[time_frame]
    masks = []
    for model in LENS_name:
        stats_path = (
            f"{uncertainty_dir}{model}_{scenario}_{contrast}_LE_stats.nc"
        )
        if not os.path.exists(stats_path):
            raise FileNotFoundError(
                f"Missing FDR statistics for {model}, {scenario}, "
                f"{time_frame}: {stats_path}"
            )
        with xr.open_dataset(stats_path) as ds:
            if "significant_fdr" not in ds:
                raise KeyError(f"'significant_fdr' missing from {stats_path}")
            mask = ds["significant_fdr"].load().fillna(0).astype(bool)
        masks.append(mask.expand_dims(model=[model]))

    stacked = xr.concat(masks, dim="model", join="exact")
    significant = stacked.sum("model") >= FDR_REQUIRED_MODELS
    significant.name = "significant_fdr_3_of_4_models"
    significant.attrs.update(
        contrast=contrast,
        criterion=(
            "At least 3 of 4 large ensembles have BH-FDR q <= 0.05 "
            "for the member-wise contrast"
        ),
    )
    return significant

fdr_by_time_frame = []
for time_frame in time_frames:
    by_scenario = xr.concat(
        [load_fdr_significant(time_frame, scenario) for scenario in scenarios],
        dim=xr.IndexVariable("scenario", scenarios),
        join="exact",
    )
    fdr_by_time_frame.append(
        by_scenario.expand_dims(time_frame=[time_frame])
    )

fdr_significant = xr.concat(
    fdr_by_time_frame,
    dim="time_frame",
    join="exact",
).assign_coords(time_frame=time_frames)
# %%
# %%
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.colors import BoundaryNorm
import cartopy.util as cutil
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# RGB from your image (0–255)
colors_255 = [
    (32,  54,  79),   # 1
    (49,  100, 108),  # 2
    (78,  146, 128),  # 3
    (150, 184, 155),  # 4
    (220, 223, 210),  # 5
    (236, 217, 207),  # 6
    (212, 156, 135),  # 7
    (184, 98,  101),  # 8
    (139, 52,  94),   # 9
    (80,  24,  78),   # 10
]
# --- your 10 colors (we already have colors_255) ---
colors = [(r/255, g/255, b/255) for r, g, b in colors_255]

# 10-step discrete cmap (if you ever need discrete)
aru_cmap_10 = ListedColormap(colors, name="aru_pastel_10")

# Smooth diverging version, interpolated between the 10 colors
aru_cmap_div = LinearSegmentedColormap.from_list(
    "aru_pastel_div", colors, N=256
)
# %%
import cartopy.crs as ccrs
from matplotlib import gridspec
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
import matplotlib.path as mpath
def circular_boundary(ax, radius=0.5):
    theta = np.linspace(0, 2 * np.pi, 200)
    center = [0.5, 0.5]
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T * radius + center
    return mpath.Path(verts)

def to_0360(da):
    if "lon" in da.coords and (da.lon < 0).any():
        da = da.assign_coords(lon=(da.lon % 360)).sortby("lon")
    return da

def prepare_lsm_on_grid(ref2d):
    lsm = xr.open_dataset("/work/mh0033/m301036/Data_storage/Data/GR15_lsm_regrid_1deg.nc")["var1"]

    rename_dict = {}
    if "latitude" in lsm.dims:
        rename_dict["latitude"] = "lat"
    if "longitude" in lsm.dims:
        rename_dict["longitude"] = "lon"
    if rename_dict:
        lsm = lsm.rename(rename_dict)

    if ref2d.lon.min() >= 0:
        lsm = to_0360(lsm)

    lsm = lsm.interp_like(ref2d)
    ocean_mask_2d = (lsm == 0).squeeze(drop=True)
    return ocean_mask_2d

def _draw_hatch(ax, lon2d, lat2d, min_lat, hatch_mask, ocean_mask, hatch_pattern, hatch_color, zorder):
    hm = np.asarray(hatch_mask)
    if isinstance(hm, np.ma.MaskedArray):
        hm = hm.filled(False)
    hm = np.squeeze(hm).astype(bool)

    if hm.shape != lat2d.shape:
        if hm.T.shape == lat2d.shape:
            hm = hm.T
        else:
            raise ValueError(f"hatch mask shape {hm.shape} does not match map shape {lat2d.shape}")

    if ocean_mask is not None:
        om = np.asarray(ocean_mask)
        if isinstance(om, np.ma.MaskedArray):
            om = om.filled(False)
        om = np.squeeze(om).astype(bool)

        if om.shape != lat2d.shape:
            if om.T.shape == lat2d.shape:
                om = om.T
            else:
                raise ValueError(f"ocean mask shape {om.shape} does not match map shape {lat2d.shape}")
    else:
        om = np.ones_like(hm, dtype=bool)

    hatch_field = np.where((lat2d >= min_lat) & hm & om, 1.0, np.nan)

    n0 = len(ax.collections)
    ax.contourf(
        lon2d, lat2d, hatch_field,
        levels=[0.5, 1.5],
        hatches=[hatch_pattern],
        colors="none",
        transform=ccrs.PlateCarree(),
        zorder=zorder
    )

    new_cols = ax.collections[n0:]
    for coll in new_cols:
        coll.set_facecolor("none")
        coll.set_edgecolor(hatch_color)
        coll.set_linewidth(0.0)   # removes polygon borders, keeps hatch strokes

def plot_data_polar(
    data, lats, lons, ax, levels, cmap, extend="both", title="",
    show_xticks=True, show_yticks=True, label_longitude=False,
    min_lat=66.5, hatch_mask=None, hatch_pattern="///",
    hatch_color="magenta", ocean_mask=None
):
    lon2d, lat2d = np.meshgrid(lons, lats)
    masked_data = np.where(lat2d >= min_lat, data, np.nan)

    cf = ax.contourf(
        lon2d, lat2d, masked_data,
        levels=levels, cmap=cmap, extend=extend,
        transform=ccrs.PlateCarree(), zorder=1
    )

    ax.set_extent([-180, 180, min_lat, 90], crs=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.45, zorder=4)
    ax.set_boundary(circular_boundary(ax), transform=ax.transAxes)

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(), draw_labels=False,
        linewidth=0.35, linestyle="--", color="gray", alpha=0.5
    )
    if show_xticks:
        gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 60))
    if show_yticks:
        gl.ylocator = mticker.FixedLocator(np.arange(min_lat, 91, 15))

    if label_longitude:
        for lon in np.arange(0, 360, 60):
            ax.text(
                lon, min_lat + 2.0, f"{lon}°",
                transform=ccrs.PlateCarree(),
                ha="center", va="center", fontsize=8, color="gray"
            )

    if hatch_mask is not None:
        _draw_hatch(
            ax, lon2d, lat2d, min_lat, hatch_mask, ocean_mask,
            hatch_pattern, hatch_color, zorder=3
        )

    ax.set_title(title, fontsize=10, pad=3)
    return cf
# %%
import seaborn as sns
sns.set_style("white")
sns.set_context("notebook", font_scale=1.0)

# Two scenarios by three matching member-wise contrasts.
n_rows, n_cols = 2, 3
fig, axes = plt.subplots(
    n_rows, n_cols,
    figsize=(15, 10),
    subplot_kw={"projection": ccrs.NorthPolarStereo()},
    constrained_layout=False,
)
intervals = np.arange(-2.5, 2.75, 0.25)
fig.subplots_adjust(
    left=0.10, right=0.90, top=0.94, bottom=0.20,
    wspace=0.06, hspace=0.08,
)

tf_order = ["MT_vs_Present", "LT_vs_Present", "LT_vs_MT"]
# Shorter, cleaner titles
col_titles = [
    r"$\mathbf{CO_{2}}$$\uparrow$ $-$ Hist.",
    r"$\mathbf{CO_{2}}$$\downarrow$ $-$ Hist.",
    r"$\mathbf{CO_{2}}$$\downarrow$ $-$ $\mathbf{CO_{2}}$$\uparrow$",
]
panel_labels = ["a", "b", "c", "d", "e", "f"]
scenario_labels = ["SSP1-1.9", "SSP1-2.6"]

reference = to_0360(
    combined.sel(time_frame=tf_order[0], scenario=scenarios[0]).squeeze()
).transpose("lat", "lon")
ocean_mask_2d = prepare_lsm_on_grid(reference).transpose("lat", "lon")

contour_objects = []
for row, scenario in enumerate(scenarios):
    for col, time_frame in enumerate(tf_order):
        ax = axes[row, col]
        print(f"Plotting panel {scenario} - {time_frame}")

        beta = to_0360(
            combined.sel(time_frame=time_frame, scenario=scenario).squeeze()
        ).transpose("lat", "lon")
        fdr_mask = to_0360(
            fdr_significant.sel(
                time_frame=time_frame, scenario=scenario
            ).squeeze()
        ).transpose("lat", "lon")

        try:
            beta, fdr_mask, ocean_mask = xr.align(
                beta, fdr_mask, ocean_mask_2d, join="exact"
            )
        except ValueError as exc:
            raise ValueError(
                f"Grid mismatch for {scenario}, {time_frame}; the plotted "
                "field, FDR mask, and ocean mask must share lat/lon coordinates."
            ) from exc

        fdr_mask = fdr_mask.fillna(False).astype(bool).where(ocean_mask, False)
        print(
            "  3-of-4-model FDR-significant ocean cells: "
            f"{int(fdr_mask.sum())}"
        )

        data_cyclic, lon_cyclic = cutil.add_cyclic_point(
            beta.values, coord=beta.lon.values, axis=-1
        )
        fdr_cyclic, _ = cutil.add_cyclic_point(
            fdr_mask.values.astype(bool), coord=fdr_mask.lon.values, axis=-1
        )
        ocean_cyclic, _ = cutil.add_cyclic_point(
            ocean_mask.values.astype(bool), coord=ocean_mask.lon.values, axis=-1
        )

        contour_obj = plot_data_polar(
            data_cyclic,
            beta.lat.values,
            lon_cyclic,
            levels=intervals,
            extend="both",
            cmap=aru_cmap_div,
            ax=ax,
            min_lat=66.5,
            hatch_mask=fdr_cyclic,
            hatch_pattern="//",
            hatch_color="Azure",
            ocean_mask=ocean_cyclic,
        )
        contour_objects.append(contour_obj)
        ax.add_feature(
            cfeature.LAND, facecolor="lightgrey", edgecolor="none", zorder=3.5
        )

        if row == 0:
            ax.set_title(col_titles[col], fontsize=18, pad=10, fontweight="bold")
        if col == 0:
            ax.text(
                -0.18, 0.5, scenario_labels[row], transform=ax.transAxes,
                rotation=90, fontsize=18, fontweight="bold",
                ha="center", va="center",
            )
        ax.text(
            -0.05, 1.02, panel_labels[row * n_cols + col],
            transform=ax.transAxes, fontsize=24, fontweight="bold",
            ha="center", va="center",
        )
# -------------------------------------------------------------
# Colorbar
# -------------------------------------------------------------
cbar_ax = fig.add_axes([0.20, 0.165, 0.60, 0.018])
cbar = fig.colorbar(
    contour_objects[0],
    cax=cbar_ax,
    orientation="horizontal",
    extend="both",
)

tick_locs = np.arange(-2.5, 2.6, 0.5)
cbar.set_ticks(tick_locs)
cbar.ax.tick_params(labelsize=13, length=7, width=1.4, color="0.2")
cbar.ax.set_xticklabels([f"{x:.1f}" for x in tick_locs], fontsize=13)
cbar.ax.spines["outline"].set_linewidth(1.2)
cbar.ax.minorticks_off()
cbar.set_label(r"Warming pattern scaling difference (°C/°C)", fontsize=16)

# -------------------------------------------------------------
# Optional significance explanation
# -------------------------------------------------------------
# fig.text(
#     0.50,
#     0.08,
#     r"Hatching: member-wise contrast is BH-FDR significant "
#     r"in at least 3 of 4 large ensembles (q ≤ 0.05; cells ≥66.5°N)",
#     ha="center", va="center", fontsize=14,
# )

dir_fig_out = (
    "/work/mh0033/m301036/Atlantic_heat_capacitor_Arctic/"
    "Atlantic_capacitor_on_Arctic/docs/figs/Spatial_pattern"
)
os.makedirs(dir_fig_out, exist_ok=True)
plt.savefig(
    f"{dir_fig_out}/Fig2_pattern_scaling_differences.png",
    dpi=300, bbox_inches="tight",
)
plt.savefig(
    f"{dir_fig_out}/Fig2_pattern_scaling_differences.pdf",
    format="pdf", dpi=300, bbox_inches="tight",
)
plt.show()
# %%
