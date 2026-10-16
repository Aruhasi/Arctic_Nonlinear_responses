# %%
import xarray as xr
import numpy as np
import logging
import glob
import os
import src.SAT_function_Obs_EC as data_process
# from src.plot_func import plot_data_polar
# import importlib
# # importlib.reload(src.plot_func)
# %%
output_dir = "/work/mh0033/m301036/Atlantic_heat_capacitor_Arctic/Atlantic_capacitor_on_Arctic/docs/data/Spatial_pattern"
# %%
# Define model names and scenarios
LENS_name = ["MIROC", "MIROC-ES2L", "MPI-ESM-LR", "CanESM5"]
base_dir = "/work/mh0033/m301036/Atlantic_heat_capacitor_Arctic/Atlantic_capacitor_on_Arctic/docs/data/Spatial_pattern/output/"

scenarios = ["ssp119", "ssp126"]
# %%
beta_ds_Future_longTerm = {}
# Loop through scenarios and models 
# SSP119: loaded from dir_ssp119_longterm (2051-2085 period)
# SSP126: loaded from base_dir/mid_term (2065-2099 period)
for model in LENS_name:
    file_pattern = f"{base_dir}/long_term/{model}_Beta_patterns_2065-2099.nc"
    # file_ssp119 = f"{dir_ssp119_longterm}{model}_Beta_patterns_2051-2085.nc"
    # Check if the file exists
    if not os.path.exists(file_pattern):
            print(f"[Warning] File not found for {model}. Skipping.")
            continue

    try:
        # Open dataset with memory-efficient settings
        ds = xr.open_mfdataset(
                file_pattern,
                combine='by_coords',
                parallel=True,
                chunks={"member": 1, "scenario": 1, "lat": 90, "lon": 180}
            )

        # Normalize longitude if negative values exist
        if (ds['lon'] < 0).any():
            ds = ds.assign_coords(lon=((ds['lon'] + 360) % 360)).sortby('lon')

            # Save the dataset into the dictionary
        beta_ds_Future_longTerm[model] = ds

    except Exception as e:
        print(f"[Error] Failed to load dataset for {model}: {e}")
        continue
# %%
beta_ds_Future_longTerm
# %%
beta_ds_Hist = {}
# Loop through scenarios and models
for model in LENS_name:
    file_pattern = f"{base_dir}/present_day/{model}_Beta_patterns_1980-2014.nc"

    # Check if the file exists
    if not os.path.exists(file_pattern):
        print(f"[Warning] File not found for {model}. Skipping.")
        continue

    try:
        # Open dataset with memory-efficient settings
        ds = xr.open_mfdataset(
            file_pattern,
            combine='by_coords',
            parallel=True,
            chunks={"member": 1, "scenario": 1, "lat": 90, "lon": 180}
        )

        # Normalize longitude if negative values exist
        if (ds['lon'] < 0).any():
            ds = ds.assign_coords(lon=((ds['lon'] + 360) % 360)).sortby('lon')

        # Save the dataset into the dictionary
        beta_ds_Hist[model] = ds

    except Exception as e:
        print(f"[Error] Failed to load dataset for {model}: {e}")
        continue
# %%
beta_ds_Hist
# %%
beta_ds_Future_midTerm = {}
for model in LENS_name:
    file_pattern = f"{base_dir}/mid_term/{model}_Beta_patterns_2015-2049.nc"

    # Check if the file exists
    if not os.path.exists(file_pattern):
        print(f"[Warning] File not found for {model}. Skipping.")
        continue
    try:
        ds = xr.open_mfdataset(
                file_pattern,
                combine='by_coords',
                parallel=True,
                chunks={"member": 1, "scenario": 1, "lat": 90, "lon": 180}
            )
        if (ds['lon'] < 0).any():
            ds = ds.assign_coords(lon=((ds['lon'] + 360) % 360)).sortby('lon')

        beta_ds_Future_midTerm[model] = ds
    except Exception as e:
        print(f"[Error] Failed to load dataset for {model}: {e}")
        continue
# %%
beta_ds_Future_midTerm
# %%
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.colors import BoundaryNorm
import cartopy.util as cutil
import seaborn as sns
import matplotlib.colors as mcolors
import palettable
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
# # Normalize to 0–1 for Matplotlib
# colors = [(r/255, g/255, b/255) for r, g, b in colors_255]

# # Create the colormap
# aru_cmap = ListedColormap(colors, name="aru_pastel")

# Quick test plot
# plt.figure(figsize=(6, 1))
# plt.imshow([range(len(colors))], aspect="auto", cmap=aru_cmap_div)
# plt.gca().set_axis_off()
# plt.tight_layout()
# plt.show()
# %%
# # define an asymmetric colormap
# from matplotlib.colors import LinearSegmentedColormap, Normalize
# from matplotlib.colors import BoundaryNorm

# %%
# calculate the ensemble mean beta patterns for each scenario and time frame
beta_list = []
for scen in scenarios:
    scen_models = []  # Collect all models for this scenario
    for model in LENS_name:
        # present-day beta model mean 
        beta_hist = beta_ds_Hist[model].sel(scenario=scen)['tas']
        beta_hist_mean = beta_hist.mean(dim='member')

        # mid-term beta mean
        beta_mid = beta_ds_Future_midTerm[model].sel(scenario=scen)['tas']
        beta_mid_mean = beta_mid.mean(dim='member')
        
        # long-term beta mean
        beta_long = beta_ds_Future_longTerm[model].sel(scenario=scen)['tas']
        beta_long_mean = beta_long.mean(dim='member')

        # stack with time_frame coordinate
        beta_model = xr.concat([
            beta_hist_mean,
            beta_mid_mean,
            beta_long_mean
        ], dim=xr.DataArray(['Hist', 'MidTerm', 'LongTerm'], 
                            dims='time_frame', name='time_frame'))
        
        # add model coordinate
        beta_model = beta_model.expand_dims({'model': [model]})
        scen_models.append(beta_model)
    
    # Combine all models for this scenario and add scenario coordinate
    scen_combined = xr.concat(scen_models, dim='model')
    scen_combined = scen_combined.expand_dims({'scenario': [scen]})
    beta_list.append(scen_combined)
# %%
# transform list to a single xarray DataArray
combined = xr.concat(beta_list, dim='scenario')
# %%
# calculate multi-model mean for each scenario and time frame
combined = combined.mean(dim='model')
# %%
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import matplotlib.colors as mcolors
import palettable
from matplotlib import gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
import matplotlib.path as mpath
import numpy as np
# %%
data_fig_out = "/work/mh0033/m301036/Atlantic_heat_capacitor_Arctic/Atlantic_capacitor_on_Arctic/docs/figs/Spatial_pattern"
if not os.path.exists(data_fig_out):
    os.makedirs(data_fig_out)
# %%
# import seaborn as sns
# sns.set_style("white")
# sns.set_context("notebook", font_scale=1.0)

# Typography matches Figure 2; keep display sizes together for easy adjustment.
TITLE_FONTSIZE = 18
ROW_LABEL_FONTSIZE = 18
PANEL_LABEL_FONTSIZE = 26
COLORBAR_TICK_FONTSIZE = 16
COLORBAR_LABEL_FONTSIZE = 18
COLORBAR_TICK_LENGTH = 7
COLORBAR_TICK_WIDTH = 1.4

def circular_boundary(ax, radius=0.5):
    """
    Creates a circular boundary for polar projection.
    """
    theta = np.linspace(0, 2 * np.pi, 100)
    center = [0.5, 0.5]
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T * radius + center
    return mpath.Path(verts)
# Define colormaps:
cmap_beta = aru_cmap_div  # For Beta patterns

def plot_data_polar(data, lats, lons, ax, levels, norm, cmap, extend='both', title='',
                    show_xticks=True, show_yticks=True, label_longitude=False,
                    min_lat=66.5, hatch_mask=None, hatch_pattern='///', hatch_color='cyan'):
    """
    data: 2D ndarray (lat, lon)
    hatch_mask: 2D boolean ndarray (lat, lon), True where hatching should be drawn
    """
    lon2d, lat2d = np.meshgrid(lons, lats)
    masked_data = np.where(lat2d >= min_lat, data, np.nan)

    cf = ax.contourf(
        lon2d, lat2d, masked_data,
        levels=levels, norm=norm, cmap=cmap, extend=extend,
        transform=ccrs.PlateCarree(), zorder=1
    )

    ax.set_extent([-180, 180, min_lat, 90], crs=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.6, zorder=4)
    ax.set_boundary(circular_boundary(ax), transform=ax.transAxes)

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(), draw_labels=False,
        linewidth=0.4, linestyle='--', color='gray', alpha=0.3
    )
    # Fixed locators prevent Cartopy from adding dense automatic gridlines.
    gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 60))
    gl.ylocator = mticker.FixedLocator(np.arange(min_lat, 91, 15))
    gl.xlines = show_xticks
    gl.ylines = show_yticks

    if label_longitude:
        for lon in np.arange(0, 360, 60):
            ax.text(
                lon, min_lat + 2.0, f"{lon}°",
                transform=ccrs.PlateCarree(),
                ha='center', va='center', fontsize=8, color='gray'
            )

    if hatch_mask is not None:
        hm = np.asarray(hatch_mask)
        hm = np.squeeze(hm).astype(bool)

        if hm.shape != lat2d.shape:
            if hm.T.shape == lat2d.shape:
                hm = hm.T
            else:
                raise ValueError(f"hatch_mask shape {hm.shape} does not match data shape {lat2d.shape}")

        hatch_field = np.where((lat2d >= min_lat) & hm, 1.0, np.nan)

        n0 = len(ax.collections)
        ax.contourf(
            lon2d, lat2d, hatch_field,
            levels=[0.5, 1.5],
            hatches=[hatch_pattern],
            colors='none',
            transform=ccrs.PlateCarree(),
            zorder=3
        )

        # style only the newly added hatch collections
        for coll in ax.collections[n0:]:
            coll.set_facecolor('none')
            coll.set_edgecolor(hatch_color)
            coll.set_linewidth(0.0)   # removes polygon outlines

    ax.set_title(title, loc='center', fontsize=TITLE_FONTSIZE, pad=10)
    return cf
# %%
"""
Plot the land mask, Arctic ocean only projection pattern
"""
# 1) Load the land/sea mask and harmonize dims/coords
lsm = xr.open_dataset("/work/mh0033/m301036/Data_storage/Data/GR15_lsm_regrid_1deg.nc")["var1"]  # 1=land, 0=ocean
# (a) Make sure dims are named 'lat' and 'lon'
if "latitude" in lsm.dims or "longitude" in lsm.dims:
    lsm = lsm.rename({"latitude": "lat", "longitude": "lon"})
# (b) Match longitude convention (0–360 vs −180–180)
def to_0360(da):
    if (da.lon < 0).any():
        da = da.assign_coords(lon=(da.lon % 360)).sortby("lon")
    return da
def to_m180180(da):
    # only needed if your data uses −180..180 (then convert lsm the same way)
    lon = xr.where(da.lon > 180, da.lon - 360, da.lon)
    return da.assign_coords(lon=lon).sortby("lon")
# Pick one conversion so lsm matches 'combined':
if (getattr(combined, "lon", None) is not None) and (combined.lon.min() >= 0):
    lsm = to_0360(lsm)
else:
    # lsm = to_m180180(lsm)
    pass

# (c) Put lsm on the SAME grid as a representative slice of combined
# ref = combined.isel(time_frame=0, scenario=0, drop=True)  # 2-D (lat,lon)
# # If grids are already identical, this is a no-op; otherwise it will interpolate
# lsm_on_data_grid = lsm.interp_like(ref)

# # 2) Build a 2-D boolean mask and broadcast it to 'combined'
# # If you want OCEAN ONLY, mask where lsm==0 is True; for LAND ONLY use (lsm==1)
# ocean_mask_2d = (lsm_on_data_grid == 0)

# # Broadcast the 2-D mask across all extra dims in 'combined'
# ocean_mask = ocean_mask_2d.broadcast_like(ref)  # 2-D still, just ensures same shape
# # Apply to full 4-D array by letting xarray broadcast over (time_frame, scenario, ...)
# combined_mask = combined.where(ocean_mask_2d)
# %%
# DISPLAY Horizontal: 3 scenarios x 3 time periods
# plot the ensemble mean beta patterns for each scenario and time frame in polar projection
# --- Colormap ---
# =========================
#  Colormap / norm for beta
# =========================
def_levs = np.arange(-5.0, 5.5, 0.5)
vmin, vmax = def_levs[0], def_levs[-1]
aru_norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)  # central beige at 0

# import seaborn as sns
# sns.set_style("white")
# sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

# Define labels
row_labels  = ["SSP1-1.9", "SSP1-2.6"]
col_labels  = [
    "Historical\n1980-2014",
    "CO$_2$ increasing\n2015-2049",
    "CO$_2$ decreasing\n2065-2099"
]

# Build grid with one axes spanning both rows in the first column
# gs = plt.GridSpec(
#     nrows=2, ncols=3, figure=plt.figure(figsize=(21, 12)),
#     width_ratios=[1.0, 1.0, 1.0], wspace=0.15, hspace=0.08,
#     left=0.06, right=0.96, top=0.88, bottom=0.17
# )
# fig = gs.figure

# # Axes:
# ax_hist = fig.add_subplot(gs[:, 0], projection=ccrs.NorthPolarStereo())  # span both rows
# ax_mt_top = fig.add_subplot(gs[0, 1], projection=ccrs.NorthPolarStereo())
# ax_mt_bot = fig.add_subplot(gs[1, 1], projection=ccrs.NorthPolarStereo())
# ax_lt_top = fig.add_subplot(gs[0, 2], projection=ccrs.NorthPolarStereo())
# ax_lt_bot = fig.add_subplot(gs[1, 2], projection=ccrs.NorthPolarStereo())
# ============================================================
# Five equal-sized polar panels with compact horizontal spacing
# ============================================================
fig = plt.figure(figsize=(18, 10))

# All five panels have EXACTLY the same width and height
panel_w = 0.245
panel_h = 0.38

# Horizontal positions
x_hist = 0.085
x_mid  = 0.360
x_long = 0.645

# Vertical positions of the two scenario rows
y_top = 0.525
y_bot = 0.135

# Historical panel centered vertically between the two rows
y_hist = 0.330

ax_hist = fig.add_axes(
    [x_hist, y_hist, panel_w, panel_h],
    projection=ccrs.NorthPolarStereo()
)

ax_mt_top = fig.add_axes(
    [x_mid, y_top, panel_w, panel_h],
    projection=ccrs.NorthPolarStereo()
)

ax_mt_bot = fig.add_axes(
    [x_mid, y_bot, panel_w, panel_h],
    projection=ccrs.NorthPolarStereo()
)

ax_lt_top = fig.add_axes(
    [x_long, y_top, panel_w, panel_h],
    projection=ccrs.NorthPolarStereo()
)

ax_lt_bot = fig.add_axes(
    [x_long, y_bot, panel_w, panel_h],
    projection=ccrs.NorthPolarStereo()
)
# Helper to plot one panel
def _plot_panel(beta_da, ax, title=None, show_yticks=False):
    # ensure dims (..., lat, lon)
    if beta_da.dims[-1] != "lon":
        beta_da = beta_da.transpose(..., "lat", "lon")

    data_cyclic, lon_cyclic = cutil.add_cyclic_point(
        beta_da.values, coord=beta_da["lon"].values, axis=-1
    )
    lats = beta_da["lat"].values

    cf = plot_data_polar(
        data_cyclic, lats, lon_cyclic,
        ax=ax,
        levels=def_levs,
        cmap=cmap_beta,
        norm=aru_norm,
        extend='both',
        show_xticks=True,
        show_yticks=show_yticks,
        min_lat=66.5
    )
    ax.add_feature(cfeature.LAND, facecolor='lightgrey', edgecolor='none', zorder=3)
    # if title:
    #     ax.set_title(title, fontsize=14, pad=8)
    return cf

# HIST (ssp126, Hist) — single panel spanning both rows
beta_hist = combined.sel(scenario="ssp126", time_frame="Hist").squeeze()
cf = _plot_panel(beta_hist, ax_hist, title="HIST (1980–2014)", show_yticks=True)

# Right columns: MidTerm and LongTerm for SSP1-1.9 (top) and SSP1-2.6 (bottom)
# MidTerm
beta_mt_119 = combined.sel(scenario="ssp119", time_frame="MidTerm").squeeze()
_ = _plot_panel(beta_mt_119, ax_mt_top, title="(2015–2049)", show_yticks=True)

beta_mt_126 = combined.sel(scenario="ssp126", time_frame="MidTerm").squeeze()
_ = _plot_panel(beta_mt_126, ax_mt_bot, title="(2015–2049)", show_yticks=True)

# LongTerm
beta_lt_119 = combined.sel(scenario="ssp119", time_frame="LongTerm").squeeze()
_ = _plot_panel(beta_lt_119, ax_lt_top, title="(2065-2099)", show_yticks=False)

beta_lt_126 = combined.sel(scenario="ssp126", time_frame="LongTerm").squeeze()
_ = _plot_panel(beta_lt_126, ax_lt_bot, title="(2065-2099)", show_yticks=False)

# Row labels on the left side of the right block (anchor at column 1 top/bottom)
for i, label in enumerate(row_labels):
    anchor_ax = ax_mt_top if i == 0 else ax_mt_bot
    anchor_ax.text(
        -0.18, 0.5, label,
        transform=anchor_ax.transAxes,
        fontsize=ROW_LABEL_FONTSIZE, fontweight='bold',
        ha='center', va='center', rotation=90
    )

# Column headers above each column
ax_hist.text(0.5, 1.06, col_labels[0], transform=ax_hist.transAxes,
             fontsize=TITLE_FONTSIZE, fontweight='bold', ha='center', va='bottom')
ax_mt_top.text(0.5, 1.06, col_labels[1], transform=ax_mt_top.transAxes,
               fontsize=TITLE_FONTSIZE, fontweight='bold', ha='center', va='bottom')
ax_lt_top.text(0.5, 1.06, col_labels[2], transform=ax_lt_top.transAxes,
               fontsize=TITLE_FONTSIZE, fontweight='bold', ha='center', va='bottom')
# add panel labels a–e
panel_labels = list("abcde")
axes_list = [ax_hist, ax_mt_top, ax_mt_bot, ax_lt_top, ax_lt_bot]
for ax, lab in zip(axes_list, panel_labels):
    ax.text(
        -0.02, 1.02, lab,
        transform=ax.transAxes,
        fontsize=PANEL_LABEL_FONTSIZE, fontweight='bold',
        ha='left', va='top'
    )
# Shared horizontal colorbar
cbar_ax = fig.add_axes([0.20, 0.08, 0.60, 0.018])
cbar = fig.colorbar(cf, cax=cbar_ax, orientation='horizontal')
cbar.set_label('Warming pattern scaling (°C/°C)', fontsize=COLORBAR_LABEL_FONTSIZE)
tick_locs = np.arange(-5, 6, 1)
cbar.set_ticks(tick_locs)
cbar.ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE, length=COLORBAR_TICK_LENGTH,
                    width=COLORBAR_TICK_WIDTH, color='0.2')
cbar.ax.set_xticklabels([f"{x:.1f}" for x in tick_locs], fontsize=COLORBAR_TICK_FONTSIZE)
cbar.outline.set_linewidth(1.2)
cbar.ax.minorticks_off()

plt.savefig(
    f"{data_fig_out}/Fig1_pattern_scaling_SSP119_SSP126.pdf",
    dpi=300, bbox_inches='tight'
)
plt.savefig(
    f"{data_fig_out}/Fig1_pattern_scaling_SSP119_SSP126.png",
    dpi=300, bbox_inches='tight'
)
plt.show()
# %%
