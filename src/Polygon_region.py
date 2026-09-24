import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import warnings
warnings.filterwarnings("ignore")
# In[2]:
import numpy as np
#from numba import jit
#@jit(nopython=True, nogil=True)
def is_left(xp, yp, x0, y0, x1, y1):
    """
    Check whether point (xp,yp) is left of line segment ((x0,y0) to (x1,y1))
    returns:  >0 if left of line, 0 if on line, <0 if right of line
    """

    return (x1-x0) * (yp-y0) - (xp-x0) * (y1-y0)

#@jit(nopython=True, nogil=True)
def distance(x1, y1, x2, y2):
    """
    Calculate Euclidean distance.
    """
    return ((x1-x2)**2 + (y1-y2)**2)**0.5

#@jit(nopython=True, nogil=True)
def point_is_on_line(x, y, x1, y1, x2, y2):
    """
    Check whether point it exactly on line
    """

    d1 = distance(x,  y,  x1, y1)
    d2 = distance(x,  y,  x2, y2)
    d3 = distance(x1, y1, x2, y2)

    eps = 1e-12
    return np.abs((d1+d2)-d3) < eps

#@jit(nopython=True, nogil=True)
def is_inside(xp, yp, x_set, y_set, size):
    """
    Given location (xp,yp) and set of line segments (x_set, y_set), determine
    whether (xp,yp) is inside (or on) polygon.
    """

    # First simple check on bounds
    if (xp < x_set.min() or xp > x_set.max() or yp < y_set.min() or yp > y_set.max()):
        return False

    wn = 0
    for i in range(size-1):

        # Second check: see if point exactly on line segment:
        if point_is_on_line(xp, yp, x_set[i], y_set[i], x_set[i+1], y_set[i+1]):
            return False

        #if (is_left(xp, yp, x_set[i], y_set[i], x_set[i+1], y_set[i+1]) == 0):
        #    return False

        # Calculate winding number
        if (y_set[i] <= yp):
            if (y_set[i+1] > yp):
                if (is_left(xp, yp, x_set[i], y_set[i], x_set[i+1], y_set[i+1]) > 0):
                    wn += 1
        else:
            if (y_set[i+1] <= yp):
                if (is_left(xp, yp, x_set[i], y_set[i], x_set[i+1], y_set[i+1]) < 0):
                    wn -= 1

    if (wn == 0):
        return False
    else:
        return True

#@jit(nopython=True, nogil=True)
def get_mask(mask, x, y, poly_x, poly_y):
    """
    Generate mask for grid points inside polygon
    """

    for j in range(y.size):
        for i in range(x.size):
            if is_inside(x[i], y[j], poly_x, poly_y, poly_x.size):
                mask[j,i] = True

    return mask


if __name__ == '__main__':
    import matplotlib.pyplot as pl
    pl.close('all')

    # Dummy data.
    lats = np.linspace(-25, 1, 14)
    lons = np.linspace(-160, -80, 26)
    print(lats)
    print(lons)
    
    data = np.arange(lats.size*lons.size).reshape((lats.size, lons.size))
    
    # Bounding box.
    poly_x = np.array([-110, -160, -80, -80, -110])  # Longitude values
    poly_y = np.array([-25, 0, 0, -25, -25])         # Latitude values
    
    # Generate mask for calculating statistics.
    mask = np.zeros_like(data, dtype=bool)
    get_mask(mask, lons, lats, poly_x, poly_y)
    
    # Calculate statistics.
    max_val = data[mask].max()
    
    # Plot data and mask.
    pl.figure(figsize=(10,4))
    pl.subplot(121)
    pl.title('data')
    pl.pcolormesh(lons, lats, data)
    pl.plot(poly_x, poly_y)
    pl.colorbar()
    
    pl.subplot(122)
    pl.title('averaging mask, max_value={}'.format(max_val))
    pl.pcolormesh(lons, lats, mask)
    pl.plot(poly_x, poly_y)
    pl.colorbar()
    
    pl.tight_layout()
# %%
# save the mask
