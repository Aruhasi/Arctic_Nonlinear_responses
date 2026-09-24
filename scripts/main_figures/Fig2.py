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
# input the agreement masks for the differences
agreement_masks_dir = "significance_masks/"
agreement_masks_ds = xr.open_mfdataset(output_dir + agreement_masks_dir + 'Beta_pattern_mean_long_vs_present_differences_Among_LEs_agreementMasks.nc')
agreement_masks_ds_vsMT = xr.open_mfdataset(output_dir + agreement_masks_dir + 'Beta_pattern_mean_mid_vs_present_differences_Among_LEs_agreementMasks.nc')  
agreement_masks_ds_LTvsMT = xr.open_mfdataset(output_dir + agreement_masks_dir + 'Beta_pattern_mean_long_vs_mid_differences_Among_LEs_agreementMasks.nc')
# %%
agreement_masks = []

for time_frame, ds in zip(["LT_vs_Present", "MT_vs_Present", "LT_vs_MT"], [agreement_masks_ds, agreement_masks_ds_vsMT, agreement_masks_ds_LTvsMT]):
    # collect DataArrays for the scenarios in the desired order
    arrs = [ds[scen] for scen in scenarios]            # list of DataArray (lat,lon,...)
    # concat them into a single DataArray with a new 'scenario' dim
    stacked = xr.concat(arrs, dim="scenario")
    stacked = stacked.assign_coords({"scenario": np.array(scenarios, dtype=object)})
    # add a time_frame dimension so we can concat across time_frames later
    stacked = stacked.expand_dims({"time_frame": [time_frame]})

    agreement_masks.append(stacked)

# final combined DataArray: dims ('time_frame','scenario', 'lat','lon', ...)
agreement_masks = xr.concat(agreement_masks, dim="time_frame")
agreement_masks.name = "agreement_mask"   # optional    
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
    ocean_mask_2d = (lsm == 0)
    return ocean_mask_2d

def plot_data_polar(
    data, lats, lons, ax, levels, cmap, extend="both", title="",
    show_xticks=True, show_yticks=True, label_longitude=False,
    min_lat=66.5, hatch_mask=None, hatch_pattern="//",
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
            zorder=3
        )

        new_cols = ax.collections[n0:]
        for coll in new_cols:
            coll.set_facecolor("none")
            coll.set_edgecolor(hatch_color)
            coll.set_linewidth(0.0)   # removes polygon borders, keeps hatch strokes

    ax.set_title(title, fontsize=10, pad=3)
    return cf
# %%
import seaborn as sns
sns.set_style("white")
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

# Create a figure with 2x3 grid (scenarios × time comparisons)
n_rows, n_cols = 2, 3
fig, axes = plt.subplots(
    n_rows, n_cols,
    figsize=(20, 15),  # Increased size for better display
    subplot_kw={'projection': ccrs.NorthPolarStereo()},
    constrained_layout=False  # Disable to avoid clipping issues
)
intervals = np.arange(-2.5, 2.75, 0.25)
# Manually adjust spacing
fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.15, wspace=0.15, hspace=0.15)

# --- Rows represent scenarios, Columns represent time comparisons ---
# Columns: MT_vs_Present, LT_vs_Present, LT_vs_MT
tf_order = ["MT_vs_Present", "LT_vs_Present", "LT_vs_MT"]

# Column titles with arrows
col_titles = ["CO₂↑", "CO₂↓", "CO₂↓ minus CO₂↑"]

# Panel letters
titles_left = ['a', 'b', 'c', 'd', 'e', 'f']
# Scenario labels for rows
scenario_labels = ["SSP1-1.9", "SSP1-2.6"]

# Time frame descriptions (for reference)
time_descriptions = ["(2015-2049)\nvs.\n(1980-2014)",
                     "(2065-2099)\nvs.\n(1980-2014)",
                     "(2065-2099)\nvs.\n(2015-2049)"]

# Store contour objects for colorbar
contour_objects = []

for row in range(n_rows):
    for col in range(n_cols):
        idx = row * n_cols + col
        ax = axes[row, col]
        
        scenario = scenarios[row]
        tf = tf_order[col]
        
        print(f"Plotting panel {scenario} - {tf} at row {row}, column {col}")

        # Row label (scenario) on the left side
        if col == 0:
            ax.text(
                -0.18, 0.5, scenario_labels[row],
                transform=ax.transAxes,
                rotation=90,
                fontsize=24, fontweight="bold",
                ha="center", va="center"
            )
        # Column title (time comparison) on the top
        if row == 0:
            ax.text(
                0.5, 1.1, col_titles[col],
                transform=ax.transAxes,
                fontsize=24, fontweight="bold",
                ha="center", va="bottom"
            )
       
        # select DataArray from combined_mask
        beta = combined.sel(time_frame=tf, scenario=scenario).squeeze()

        # make sure the data is (lat, lon) with lon last:
        if beta.dims[-1] != "lon":
            beta = beta.transpose(..., "lat", "lon") if "lat" in beta.dims and "lon" in beta.dims else beta

        # 4) NOW add the cyclic point safely (lon length must match data.shape[-1])
        data_cyclic, lon_cyclic = cutil.add_cyclic_point(
            beta.values, coord=beta["lon"].values, axis=-1
        )

        # agreement mask should go through the same steps
        amask = agreement_masks.sel(time_frame=tf, scenario=scenario).squeeze()

        if amask.dims[-1] != "lon":
            amask = amask.transpose(..., "lat", "lon")

        mask_cyclic, lon_mask_cyclic = cutil.add_cyclic_point(
            amask.values.astype(bool), coord=amask["lon"].values, axis=-1
        )

        contour_obj = plot_data_polar(
            data_cyclic, beta["lat"].values, lon_cyclic,
            levels=intervals, extend="both", cmap=aru_cmap_div, title="",
            ax=ax, label_longitude=True, min_lat=66.5,
            hatch_mask=~mask_cyclic, hatch_pattern="//"
        )
        
        # Store the contour object
        contour_objects.append(contour_obj)
        
        # Add grey land shading on top of the data
        ax.add_feature(cfeature.LAND, facecolor='lightgrey', edgecolor='none', zorder=3)
        
        # Add longitude labels around the plot
        for lon in [0, 60, 120, 180, 240, 300]:
            ax.text(lon, 65.0, f"{lon}°",  # Adjusted latitude closer to min_lat=66.5
                    transform=ccrs.PlateCarree(),
                    ha='center', va='top', fontsize=14, color='black',
                    bbox=dict(boxstyle='round,pad=0.03', facecolor='none', 
                             edgecolor='none', alpha=0.7))

# add colorbar below the plot (use first contour object, all have same levels)
cbar_ax = fig.add_axes([0.2, 0.1, 0.6, 0.02])
cbar = fig.colorbar(
    contour_objects[0],  # Use first contour object
    cax=cbar_ax,
    orientation="horizontal",
    extend="both"
)
cbar.ax.tick_params(labelsize=20)
# set the interval for the labels on the colorbar
cbar.set_ticks(intervals[::2])  # Show every other tick to avoid crowding
cbar.set_label("Warming pattern scaling difference"+" ("+r'$\beta$'+":°C/°C)", fontsize=22)
# add the bold a, b, c, d, e, f labels to the top left of each panel
for idx, ax in enumerate(axes.flat):
    ax.text(
        -0.05, 1.05, titles_left[idx],
        transform=ax.transAxes,
        fontsize=28, fontweight="bold",
        ha="center", va="bottom")
    
dir_fig_out = "/work/mh0033/m301036/Atlantic_heat_capacitor_Arctic/Atlantic_capacitor_on_Arctic/docs/figs/Spatial_pattern"

plt.savefig(f"{dir_fig_out}/Fig2_pattern_scaling_differences.png", dpi=300, bbox_inches='tight')
plt.savefig(f"{dir_fig_out}/Fig2_pattern_scaling_differences.pdf", format='pdf', dpi=300, bbox_inches='tight')

plt.show()
# %%
