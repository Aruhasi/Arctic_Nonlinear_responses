#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thursday Dec 18, 2025
__author__ = "Dr. Josie Aruhasi"
This script contains the functions used in the NHLSAT analysis.
The functions are:
    - latlon2d_for_da(da)
    - interpolate_to_ocean_balltree(source_da, target_lat2d, target_lon2d)

"""
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Import modules
import os
import sys

import xarray as xr
import numpy as np
import pandas as pd
import numpy.ma as ma
import scipy.stats as stats
import numpy as np
import xarray as xr
from pathlib import Path
import pandas as pd

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def latlon2d_for_da(da):
    """
    Return 2D lat/lon arrays (numpy) for a given DataArray (da).
    Works for 1D lat/lon or already-2D lat/lon.
    """
    coords = list(da.coords)
    lat_name = next((c for c in coords if "lat" in c.lower()), None)
    lon_name = next((c for c in coords if "lon" in c.lower()), None)
    if lat_name is None or lon_name is None:
        raise KeyError(f"No lat/lon coords found in {coords}")

    lat = da.coords[lat_name]
    lon = da.coords[lon_name]

    if lat.ndim == 1 and lon.ndim == 1:
        lon2d, lat2d = np.meshgrid(lon.values, lat.values)
    elif lat.ndim == 2 and lon.ndim == 2:
        lat2d = lat.values
        lon2d = lon.values
    else:
        raise ValueError("Unsupported lat/lon shapes")

    # Normalize longitudes to 0–360 to match MPIOM grid style
    lon2d = lon2d % 360.0
    return lat2d, lon2d
# %%
from sklearn.neighbors import BallTree

def interpolate_to_ocean_balltree(source_da, target_lat2d, target_lon2d, mask_ds):
    """
    Nearest-neighbor interpolation of a 2D source DataArray to target lat-lon grid
    using BallTree + haversine distance.

    source_da: xr.DataArray with dims (lat, lon) or (y, x)
    target_lat2d, target_lon2d: 2D numpy arrays (or DataArray.values)
    """
    # Make sure source is 2D: collapse time if needed
    if "time" in source_da.dims:
        source_da = source_da.mean("time")

    # 2D source lat/lon
    src_lat2d, src_lon2d = latlon2d_for_da(source_da)

    # Flatten source
    src_vals = source_da.values
    src_vals_flat = src_vals.ravel()
    src_lat_flat = src_lat2d.ravel()
    src_lon_flat = src_lon2d.ravel()

    # Only use finite points
    valid = np.isfinite(src_vals_flat)
    src_vals_flat = src_vals_flat[valid]
    src_lat_flat = src_lat_flat[valid]
    src_lon_flat = src_lon_flat[valid]

    # Build BallTree in radians
    src_pts_rad = np.vstack(
        (np.deg2rad(src_lat_flat), np.deg2rad(src_lon_flat))
    ).T  # (N, 2)
    tree = BallTree(src_pts_rad, metric="haversine")

    # Prepare target points
    tgt_lat_flat = target_lat2d.ravel()
    tgt_lon_flat = (target_lon2d.ravel() % 360.0)
    tgt_pts_rad = np.vstack(
        (np.deg2rad(tgt_lat_flat), np.deg2rad(tgt_lon_flat))
    ).T

    dist, idx = tree.query(tgt_pts_rad, k=1)
    interp_vals_flat = src_vals_flat[idx[:, 0]]
    interp_vals = interp_vals_flat.reshape(target_lat2d.shape)

    # Return DataArray on (y_c, x_c)
    da_interp = xr.DataArray(
        interp_vals,
        coords={
            "y_c": mask_ds["lat_c"].coords["y_c"],
            "x_c": mask_ds["lat_c"].coords["x_c"],
            "lat_c": (("y_c", "x_c"), target_lat2d),
            "lon_c": (("y_c", "x_c"), target_lon2d),
        },
        dims=("y_c", "x_c"),
        
        name=source_da.name,
    )
    return da_interp
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# transform the ocean mask to the target grid (atmos grid)
from scipy.spatial import cKDTree

def interpolate_mask_to_grid(source_da, target_lat2d, target_lon2d):
    """
    Nearest-neighbor interpolation of a source mask DataArray to a target lat-lon grid.

    Parameters
    ----------
    source_da : xr.DataArray
        Source mask with 2D lat/lon coordinates.
    target_lat2d, target_lon2d : xr.DataArray
        Target 2D latitude and longitude arrays.

    Returns
    -------
    xr.DataArray
        Interpolated mask on the target grid.
    """
    # Extract source lat/lon and mask values
    source_lat, source_lon = latlon2d_for_da(source_da)
    source_points = np.column_stack((source_lon.values.ravel(), source_lat.values.ravel()))
    source_values = source_da.values.ravel()

    # Build KDTree for nearest-neighbor search
    tree = cKDTree(source_points)

    # Prepare target points
    target_points = np.column_stack((target_lon2d.values.ravel(), target_lat2d.values.ravel()))

    # Query nearest neighbors
    dist, idx = tree.query(target_points)

    # Get interpolated values
    interpolated_values = source_values[idx].reshape(target_lat2d.shape)

    return xr.DataArray(interpolated_values, coords=target_lat2d.coords, dims=target_lat2d.dims)