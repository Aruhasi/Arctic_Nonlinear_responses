#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %%
import os
import numpy as np
import xarray as xr
import rioxarray  # For reading GeoTiff files
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
# ============================================================
# User settings
# ============================================================
# %%
# IBCAO Arctic bathymetry (200m resolution, GeoTiff format)
BATHY_FILE = "/work/mh0033/m301036/Atlantic_heat_capacitor_Arctic/Atlantic_capacitor_on_Arctic/docs/data/bathymetry/ibcao_v5_1_200m.tif"

OUTDIR = "./figs"
os.makedirs(OUTDIR, exist_ok=True)

OUT_PNG = f"{OUTDIR}/Arctic_gateway_schematic_depth_shading.png"
OUT_PDF = f"{OUTDIR}/Arctic_gateway_schematic_depth_shading.pdf"

MIN_LAT = 50.0  # Data latitude cutoff (lower than plot extent for complete coverage)
PLOT_MIN_LAT = 65.0  # Actual plot extent minimum latitude

# ============================================================
# Load bathymetry
# ============================================================
# Read IBCAO GeoTiff with rioxarray
bathy = rioxarray.open_rasterio(BATHY_FILE, masked=True).squeeze()

# IBCAO is in Polar Stereographic (EPSG:3996), need to extract lat/lon
# Get coordinates from the raster
x_coords = bathy.x.values
y_coords = bathy.y.values

# Create meshgrid of x, y coordinates
x2d, y2d = np.meshgrid(x_coords, y_coords)

# Transform from Polar Stereographic to lat/lon
import pyproj
transformer = pyproj.Transformer.from_crs("EPSG:3996", "EPSG:4326", always_xy=True)
lon2d, lat2d = transformer.transform(x2d, y2d)

# Get depth values (IBCAO: negative for ocean, positive for land)
depth = bathy.values
# Flip sign for ocean depth (make ocean positive)
depth = -depth
# Set land (negative or zero depths) to NaN
depth = np.where(depth > 0, depth, np.nan)

# Apply Arctic filter
arctic_mask = lat2d >= MIN_LAT
depth = np.where(arctic_mask, depth, np.nan)

# Coarsen for faster plotting (every 5th point for 200m -> ~1km resolution)
# Using finer resolution to ensure complete circular coverage
coarsen = 5
lon2d = lon2d[::coarsen, ::coarsen]
lat2d = lat2d[::coarsen, ::coarsen]
depth = depth[::coarsen, ::coarsen]
# ============================================================
# Plot settings
# ============================================================
proj = ccrs.NorthPolarStereo()
pc = ccrs.PlateCarree()
depth_levels = [0, 50, 100, 200, 500, 1000, 1500, 2000, 3000, 4000, 4500]
cmap = LinearSegmentedColormap.from_list(
    "arctic_depth",
    [
        "#f2fbff",  # very shallow
        "#cdeff5",
        "#9bd6e5",
        "#6bb6d6",
        "#3f8ec4",
        "#2762aa",
        "#1d3f8f",
        "#102a63",
    ],
    N=256
)

norm = BoundaryNorm(depth_levels, cmap.N, extend="neither")

fig = plt.figure(figsize=(8.5, 8.5))
ax = plt.axes(projection=proj)

ax.set_extent([-180, 180, PLOT_MIN_LAT, 90], crs=pc)

# Depth shading (rasterized to reduce PDF file size)
im = ax.pcolormesh(
    lon2d, lat2d, depth,
    transform=pc,
    cmap=cmap,
    norm=norm,
    shading="auto",
    zorder=1,
    rasterized=True
)

# Circular boundary (set after plotting to ensure proper clipping)
theta = np.linspace(0, 2 * np.pi, 300)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)

# Land and coastlines (land rasterized to reduce file size)
ax.add_feature(cfeature.LAND, facecolor="#b8b8b8", edgecolor="none", zorder=3, rasterized=True)
ax.coastlines(linewidth=0.5, color="0.35", zorder=4)

# Gridlines with longitude labels (overlay on top of land)
gl = ax.gridlines(
    crs=pc,
    draw_labels=True,
    linewidth=0.4,
    color="gray",
    alpha=0.35,
    linestyle="--",
    zorder=5,
    x_inline=False,
    y_inline=False,
    xlabel_style={'size': 10, 'color': 'black'},
    ylabel_style={'size': 10, 'color': 'black'}
)
gl.top_labels = False
gl.left_labels = False
gl.right_labels = False

# ============================================================
# Gateways
# ============================================================
gateways = {
    "Bering\nStrait": {
        "lon": [-169.5, -166.0],
        "lat": [65.8, 65.8],
        "text": (-168, 62.0),
        "color": "#84056B",
    },
    "Davis\nStrait": {
        "lon": [-63.0, -52.0],
        "lat": [66.0, 68.5],
        "text": (-67, 65.5),
        "color": "#84056B",
    },
    "Fram\nStrait": {
        "lon": [-18.0, 8.0],
        "lat": [79.5, 79.5],
        "text": (0, 77.2),
        "color": "#84056B",
    },
    "Barents Sea\nOpening": {
        "lon": [20.0, 20.0],
        "lat": [69.91, 77.04],
        "text": (8.0, 70.8),
        "color": "#84056B",
    },
}
# gateways = {
#     "Bering\nStrait": {
#         "lon": [-169.5, -166.0],
#         "lat": [65.8, 65.8],
#         "text": (-168.9, 62.8),
#         "color": "#7B3294",   # purple
#     },
#     "Davis\nStrait": {
#         "lon": [-61.5, -53.4],
#         "lat": [66.00, 66.31],
#         "text": (-57.5, 63.5),
#         "color": "#7B3294",   # purple
#     },
#     "Fram\nStrait": {
#         "lon": [-18.0, 8.0],
#         "lat": [78.5, 79.5],
#         "text": (0.4, 77.4),
#         "color": "#7B3294",   # purple
#     },
#     "Barents Sea\nOpening": {
#         "lon": [20.8, 20.8],
#         "lat": [69.91, 77.04],
#         "text": (22.0, 67.4),
#         "color": "#7B3294",   # purple
#     },
# }
for name, g in gateways.items():
    ax.plot(
        g["lon"], g["lat"],
        transform=pc,
        color=g["color"],
        linewidth=2.5,
        solid_capstyle="round",
        zorder=10
    )

    ax.text(
        g["text"][0], g["text"][1], name,
        transform=pc,
        fontsize=11,
        fontweight="bold",
        ha="center",
        va="center",
        color=g["color"],
        zorder=11
    )
# ============================================================
# Main Arctic labels
# ============================================================
# labels = {
#     "Amerasian\nBasin": (-150, 82),
#     "Eurasian\nBasin": (60, 84),
#     "Beaufort\nSea": (-145, 74),
#     "Chukchi\nSea": (-170, 70),
#     "East Siberian\nSea": (160, 73),
#     "Laptev\nSea": (125, 76),
#     "Kara\nSea": (75, 75),
#     "Barents\nSea": (40, 74),
#     "Greenland\nSea": (-5, 75),
#     "Baffin\nBay": (-62, 73),
# }

labels = {
    "Amerasian\nBasin": (-145, 82),
    "Eurasian\nBasin": (55, 84),
    "Beaufort\nSea": (-145, 74),
    "Chukchi\nSea": (-170, 70),
    "East Siberian\nSea": (160, 73),
    "Laptev\nSea": (125, 76),
    "Kara\nSea": (75, 75),
    "Barents\nSea": (40, 74),
    # "Greenland\nSea": (-5, 75),
    "Baffin\nBay": (-62, 73),
}
for txt, (lo, la) in labels.items():
    ax.text(
        lo, la, txt,
        transform=pc,
        fontsize=10,
        ha="center",
        va="center",
        color="black",
        zorder=9
    )
# ============================================================
# Schematic arrows (REMOVED as requested)
# ============================================================
# def add_arrow(lons, lats, color, lw=3.0):
#     ax.plot(
#         lons, lats,
#         transform=pc,
#         color=color,
#         linewidth=lw,
#         zorder=7,
#         solid_capstyle="round"
#     )
#     ax.annotate(
#         "",
#         xy=(lons[-1], lats[-1]),
#         xytext=(lons[-2], lats[-2]),
#         xycoords=pc._as_mpl_transform(ax),
#         textcoords=pc._as_mpl_transform(ax),
#         arrowprops=dict(
#             arrowstyle="-|>",
#             color=color,
#             lw=lw,
#             mutation_scale=18
#         ),
#         zorder=8
#     )

# # Atlantic inflow
# add_arrow([-20, 0, 8], [60, 72, 78], color="red", lw=3.2)
# add_arrow([20, 35, 55], [68, 72, 78], color="#f0b000", lw=3.2)

# # Arctic outflow
# add_arrow([-5, -20, -40], [80, 75, 68], color="#153f6f", lw=3.2)
# add_arrow([-160, -150, -130], [72, 76, 78], color="#153f6f", lw=3.2)

# ============================================================
# Colorbar
# ============================================================
cbar = plt.colorbar(
    im,
    ax=ax,
    orientation="horizontal",
    pad=0.04,
    shrink=0.82,
    extend="neither"
    )
cbar.set_label("Ocean depth (m)", fontsize=12)
cbar.set_ticks(depth_levels)

# ax.set_title(
#     "Arctic Ocean gateways and bathymetric setting",
#     fontsize=15,
#     fontweight="bold",
#     pad=14
# )

# Save without bbox_inches='tight' to preserve circular boundary
fig.savefig(OUT_PNG, dpi=300)
fig.savefig(OUT_PDF, dpi=300)

plt.show()
# %%