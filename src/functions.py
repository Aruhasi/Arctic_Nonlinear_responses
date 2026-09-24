# -*- coding: utf-8 -*-
"""

This module contains functions for plotting and visualizing data related to
AR6 regions using Cartopy and Matplotlib.

"""
# Imports
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
# ====================================================================
def shift_lon_data_xr(da):
    """
    Shift xarray DataArray or Dataset longitudes from 0–360 to -180–180 cleanly.
    """
    if np.any(da.lon > 180):
        da = da.roll(lon=da.sizes['lon'] // 2, roll_coords=True)
        da = da.assign_coords(lon=(((da.lon + 180) % 360) - 180))
        da = da.sortby('lon')
    return da
# ====================================================================
def plot_ar6_regions(data, projection=ccrs.Robinson(), text_color="#67000d", fontsize=8):
    """
    Plots AR6 regions with a specified projection.

    Parameters:
    - data: The dataset containing AR6 regions.
    - projection: Cartopy projection (default: Robinson).
    - text_color: Color of the text annotations.
    - fontsize: Font size of the text annotations.

    Returns:
    - Matplotlib figure and axis.
    """
    fig, ax = plt.subplots(subplot_kw=dict(projection=projection))

    text_kws = dict(color=text_color, fontsize=fontsize, bbox=dict(pad=0.2, color="w"))

    ax = data.plot(
        ax=ax,
        add_ocean=False,
        line_kws=dict(linewidth=1),
        coastline_kws=dict(color="0.5", linewidth=0.5),
        text_kws=text_kws,
    )

    ax.coastlines(color="0.5", lw=0.5)

    return fig, ax
# ====================================================================
from matplotlib.path import Path

def create_region_mask_wrapper(ds, lat_vals, lon_vals, region_type=None, lat_name="lat", lon_name="lon"):
    """
    Unified region masking function.
    
    Parameters:
    - ds: xr.DataArray (with lat/lon)
    - lat_vals: tuple or list of latitudes
    - lon_vals: tuple or list of longitudes
    - region_type: "box", "polygon", or "points"
    - lat_name, lon_name: coordinate names
    
    Returns:
    - mask: xr.DataArray (bool) of shape (lat, lon)
    """

    lat = ds[lat_name]
    lon = ds[lon_name]

    # Fix longitudes to be consistent
    if lon.max() > 180:
        lon_vals = np.array(lon_vals) % 360

    lon2d, lat2d = np.meshgrid(lon, lat)

    if region_type == "box":
        lat_min, lat_max = lat_vals
        lon_min, lon_max = lon_vals
        # Convert to NumPy for broadcasting
        lat_mask = ((lat >= lat_min) & (lat <= lat_max)).values
        lon_mask = ((lon >= lon_min) & (lon <= lon_max)).values
        mask = xr.DataArray(
            lat_mask[:, np.newaxis] & lon_mask[np.newaxis, :],
            coords={lat_name: lat, lon_name: lon},
            dims=[lat_name, lon_name],
        )

    elif region_type == "polygon":
        poly_path = Path(list(zip(lon_vals, lat_vals)))
        points = np.vstack((lon2d.flatten(), lat2d.flatten())).T
        mask_flat = poly_path.contains_points(points)
        mask = xr.DataArray(
            mask_flat.reshape(lon2d.shape),
            coords={lat_name: lat, lon_name: lon},
            dims=[lat_name, lon_name],
        )

    elif region_type == "points":
        mask = np.zeros((len(lat), len(lon)), dtype=bool)
        for la, lo in zip(lat_vals, lon_vals):
            lat_idx = np.abs(lat - la).argmin().item()
            lon_idx = np.abs(lon - lo).argmin().item()
            mask[lat_idx, lon_idx] = True
        mask = xr.DataArray(mask, coords={lat_name: lat, lon_name: lon}, dims=[lat_name, lon_name])

    else:
        raise ValueError(f"Unknown region_type: {region_type}")

    return mask
# ==================================================================
# def create_box_mask(ds, lat_bounds, lon_bounds, lat_name="lat", lon_name="lon"):
#     """
#     Create a rectangular region mask for an xarray Dataset/DataArray.
#     """
#     lat_mask = (ds[lat_name] >= lat_bounds[0]) & (ds[lat_name] <= lat_bounds[1])
#     lon_mask = (ds[lon_name] >= lon_bounds[0]) & (ds[lon_name] <= lon_bounds[1])
#     print(type(lat_mask), type(lat_mask.values))

#     # Convert to NumPy arrays before broadcasting
#     mask = lat_mask.values[:, np.newaxis] & lon_mask.values[np.newaxis, :]

#     return xr.DataArray(mask.astype(int), coords={lat_name: ds[lat_name], lon_name: ds[lon_name]}, dims=[lat_name, lon_name])
# # ===================================================================
# # Polygon mask creation function
# from matplotlib.path import Path
# def get_polygon_mask(lon2d, lat2d, poly_lon, poly_lat):
#     points = np.vstack((lon2d.flatten(), lat2d.flatten())).T
#     path = Path(list(zip(poly_lon, poly_lat)))
#     mask_flat = path.contains_points(points)
#     return mask_flat.reshape(lon2d.shape)
# # ==================================================================
# def create_parallelogram_mask(ds, poly_lon, poly_lat, lat_name="lat", lon_name="lon"):
#     """
#     Create a mask from a parallelogram polygon over an xarray DataArray.
#     """
#     # Make 2D lon/lat meshgrid
#     lon = ds[lon_name]
#     lat = ds[lat_name]
#     lon2d, lat2d = np.meshgrid(lon, lat)

#     # Flatten meshgrid and combine
#     points = np.vstack((lon2d.flatten(), lat2d.flatten())).T
#     polygon = Path(list(zip(poly_lon, poly_lat)))

#     # Create boolean mask and reshape to 2D
#     mask_flat = polygon.contains_points(points)
#     mask = mask_flat.reshape(lon2d.shape)

#     # Return as xarray DataArray
#     return xr.DataArray(mask, coords={lat_name: lat, lon_name: lon}, dims=[lat_name, lon_name])
# ===================================================================
def calc_region_temp(ds, mask):
    """
    Calculates the regional temperature time series using a mask.

    Parameters:
    - ds: xarray.Dataset
        The dataset containing temperature variables.
    - mask: xarray.DataArray
        Boolean mask where True represents the region.

    Returns:
    - xarray.DataArray
        Regional temperature time series.
    """
    masked_ds = ds.where(mask)  # Apply the mask
    weighted_temp = (masked_ds * np.cos(np.deg2rad(ds.lat))).sum(dim=["lat", "lon"])
    area_weight = np.cos(np.deg2rad(ds.lat)).sum(dim=["lat", "lon"])
    
    return weighted_temp / area_weight