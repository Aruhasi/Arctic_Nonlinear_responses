# ====================================================================
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
import matplotlib.colors as colors
import matplotlib.ticker as mticker
import cartopy.mpl.ticker as cticker
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import matplotlib.path as mpath
import seaborn as sns
from matplotlib.colors import ListedColormap, BoundaryNorm

plt.rcParams['figure.figsize'] = (8, 10)
plt.rcParams['font.family'] = 'sans-serif'
# plt.rcParams["font.family"] = "Times New Roman"
plt.rc('font', family='serif')
plt.rcParams['font.size'] = 16
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['ytick.direction'] = 'out'
plt.rcParams['ytick.minor.visible'] = True
plt.rcParams['ytick.major.right'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.bottom'] = True
plt.rcParams['savefig.transparent'] = True
plt.rcParams['savefig.dpi'] = 300

# Function to plot 2D data with Cartopy
def plot_data(data, lats, lons, levels=None, extend=None, cmap=None, norm=None, title="", ax=None, show_xticks=False, show_yticks=False):
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={'projection': ccrs.Robinson()})
    
    ax.coastlines(resolution='110m')
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                      colors='gray', alpha=0.25, linestyle='--', linewidth=0.25)
    
    gl.top_labels = False
    gl.right_labels = False
    gl.bottom_labels = show_xticks
    gl.left_labels = show_yticks
    gl.xformatter = cticker.LongitudeFormatter()
    gl.yformatter = cticker.LatitudeFormatter()
    gl.xlabel_style = {'size': 12}
    gl.ylabel_style = {'size': 12}
    
    if show_xticks:
        gl.bottom_labels = True
    if show_yticks:
        gl.left_labels = True
    
    cf = ax.contourf(lons, lats, data, levels=levels, extend=extend, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
    ax.set_title(title, loc='center', fontsize=22, pad=5.0)
    return cf
# ====================================================================
def plot_data_with_agreement(data, lats, lons, levels=None, extend=None, cmap=None, norm=None, 
                             title="", ax=None, show_xticks=False, show_yticks=False,
                             agreement_mask=None, hatch_pattern='///'):
    if ax is None:
        ax = plt.subplot(projection=ccrs.Robinson(180))

    ax.coastlines(resolution='110m', linewidth=0.5)
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                      colors='gray', alpha=0.25, linestyle='--', linewidth=0.25)
    gl.top_labels = False
    gl.right_labels = False
    gl.bottom_labels = show_xticks
    gl.left_labels = show_yticks
    gl.xformatter = cticker.LongitudeFormatter()
    gl.yformatter = cticker.LatitudeFormatter()
    gl.xlabel_style = {'size': 16}
    gl.ylabel_style = {'size': 16}

    # plot main data first (zorder low)
    cf = ax.contourf(lons, lats, data, levels=levels, extend=extend,
                     cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), zorder=1)

    # then overlay agreement mask as hatch (zorder higher so hatch is visible)
    if agreement_mask is not None:
        # create masked array where agreement_mask == 1 is kept, else masked
        masked = np.ma.masked_where(agreement_mask == 0, agreement_mask)
        # single contour interval for hatch (0.5 - 1.5 ensures values==1 get hatched)
        ax.contourf(lons, lats, masked, levels=[0.5, 1.5],
                    colors='none', hatches=[hatch_pattern], transform=ccrs.PlateCarree(),
                    zorder=2)

    ax.set_title(title, loc='center', fontsize=20, pad=5.0)
    return cf
# ====================================================================
def plot_data_with_significance(data, lats, lons, p_values, GMST_p_values=None, levels=None, extend=None, cmap=None, title="", ax=None, show_xticks=False, show_yticks=False):
    """
    Parameters:
    - data: 2D numpy array.
    - lats, lons: 1D arrays of latitudes and longitudes.
    - p_values: 2D array with p-values for each grid point.
    - GMST_p_values: 2D array with GMST p-values for each grid point.
    - title: Title for the plot.
    - ax: Existing axis to plot on. If None, a new axis will be created.
    - show_xticks, show_yticks: Boolean flags to show x and y axis ticks.
    
    Returns:
    - contour_obj: The contour object from the plot.
    """
    # Create a new figure/axis if none is provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 15), subplot_kw={'projection': ccrs.Robinson()})
        ax.set_global()

    # Determine significance mask (where p-values are less than 0.05)
    insignificance_mask = p_values >= 0.05
    # Plotting
    # contour_obj = ax.pcolormesh(lons, lats, data,  cmap='RdBu_r',vmin=-5.0, vmax=5.0, transform=ccrs.PlateCarree(central_longitude=180), shading='auto')
    contour_obj = ax.contourf(lons, lats, data, levels=levels, extend=extend, cmap=cmap, transform=ccrs.PlateCarree(central_longitude=0))

    # Plot significance masks with different hatches
    ax.contourf(lons, lats, insignificance_mask, levels=[0.0, 0.05, 1.5],hatches=[None,'///'], colors='none', transform=ccrs.PlateCarree())

    ax.coastlines(resolution='110m')
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                      colors='gray', alpha=0.5, linestyle='--', linewidth=0.5)

    # Disable labels on the top and right of the plot
    gl.top_labels = False
    gl.right_labels = False

    # Enable labels on the bottom and left of the plot
    gl.bottom_labels = show_xticks
    gl.left_labels = show_yticks
    gl.xformatter = cticker.LongitudeFormatter()
    gl.yformatter = cticker.LatitudeFormatter()
    gl.xlabel_style = {'size': 18}
    gl.ylabel_style = {'size': 18}
    
    if show_xticks:
        gl.bottom_labels = True
    if show_yticks:
        gl.left_labels = True

    ax.set_title(title, loc='left', fontsize=22, pad=5.0)

    return contour_obj
# ====================================================================
# PlateCarree projection
def plot_data_PlateCarree(data, lats, lons, levels=None, extend=None, cmap=None, norm=None, title="",
              ax=None, show_xticks=False, show_yticks=False):
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})  

    # Add coastlines and gridlines
    ax.coastlines(resolution='110m')
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                      colors='gray', alpha=0.25, linestyle='--', linewidth=0.25)
    
    gl.top_labels = False
    gl.right_labels = False
    gl.bottom_labels = show_xticks
    gl.left_labels = show_yticks
    gl.xformatter = cticker.LongitudeFormatter()
    gl.yformatter = cticker.LatitudeFormatter()
    gl.xlabel_style = {'size': 18}
    gl.ylabel_style = {'size': 18}

    # Contour plot
    cf = ax.contourf(lons, lats, data, levels=levels, extend=extend,
                     cmap=cmap, norm=norm, transform=ccrs.PlateCarree())

    ax.set_title(title, loc='center', fontsize=22, pad=5.0)
    return cf
# ===================================================================
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker

def draw_longitude_labels(ax, ticks=np.arange(60, 301, 60)):
    for lon in ticks:
        label = f"{lon}°E" if lon <= 180 else f"{360 - lon}°W"
        ax.text(
            lon, 5, label, ha='center', va='bottom',
            transform=ccrs.PlateCarree(), fontsize=10
        )

def plot_NH_panel(data, lats, lons, ax, title="", levels=None, cmap=None, extend="both",
                  norm=None, show_xticks=False, show_yticks=False,
                  show_longitude_labels=True, add_features=True):
    """
    Plots North Pacific zoom on Lambert Conformal projection.
    Zooms from 60°E to 60°W (centered on 180°), 0–90°N.
    """
    # Setup projection (should be already applied to ax externally)
    ax.set_global()  # Optional reset

    # Add map features
    if add_features:
        ax.coastlines(resolution='110m')
        # ax.add_feature(cfeature.BORDERS, linewidth=0.4)
        # ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)

    # North Pacific extent
    ax.set_extent([60, 300, 0, 90], crs=ccrs.PlateCarree())

    # Gridlines (optional, no labels here)
    gl = ax.gridlines(draw_labels=False, linestyle='--', linewidth=0.25, alpha=0.3)
    gl.xlocator = plt.MultipleLocator(60)
    gl.ylocator = plt.MultipleLocator(30)

    # Plot data
    cf = ax.contourf(lons, lats, data, levels=levels, cmap=cmap,
                     extend=extend, transform=ccrs.PlateCarree(), norm=norm)

    # Longitude labels (custom ring-like)
    if show_longitude_labels:
        draw_longitude_labels(ax)

    # Title
    ax.set_title(title, fontsize=18, pad=6)

    return cf

# ====================================================================
import numpy as np
from matplotlib.path import Path
import cartopy.crs as ccrs

def add_circular_boundary(ax, radius=0.5, n_points=200):
    """
    Clip the given GeoAxes to a circle in Axes-fraction coordinates.
    """
    # Generate circle vertices in axes‐fraction space
    theta = np.linspace(0, 2*np.pi, n_points)
    verts = np.vstack([
        0.5 + radius * np.sin(theta),
        0.5 + radius * np.cos(theta)
    ]).T
    circle_path = Path(verts)  # <-- raw Path, not a Patch

    # Apply as the boundary for the GeoAxes
    ax.set_boundary(circle_path, transform=ax.transAxes)
# ====================================================================
# def NH_plot_data(data, lats, lons, levels=None, extend=None, cmap=None, norm=None, title="", ax=None):
#     if ax is None:
#         fig, ax = plt.subplots(subplot_kw={'projection': ccrs.NorthPolarStereo()})
    
#     # 1) create axes if needed
#     if ax is None:
#         fig, ax = plt.subplots(
#             subplot_kw={'projection': ccrs.NorthPolarStereo()}
#         )

#     # 2) plot your data & coastlines
#     ax.coastlines(resolution='110m', linewidth=0.5)
#     cf = ax.contourf(
#         lons, lats, data,
#         levels=levels, extend=extend,
#         cmap=cmap, norm=norm,
#         transform=ccrs.PlateCarree()
#     )

#     # 3) title
#     ax.set_title(title, fontsize=14, pad=5)

#     # 4) circular boundary clip
#     add_circular_boundary(ax)

#     # 5) gridlines (customize as you like)
#     gl = ax.gridlines(
#         crs=ccrs.PlateCarree(), draw_labels=False,
#         linewidth=1, color='c', linestyle='--'
#     )
#     gl.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
#     gl.ylocator = mticker.FixedLocator([30, 60, 90])

#     return cf
# ====================================================================
def NH_plot_data(data, lats, lons, levels=None, extend=None, cmap=None, norm=None, title="", ax=None, use_pcolormesh=False):
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={'projection':ccrs.AzimuthalEquidistant(central_longitude=180.0, central_latitude=90.0)})
    # set the region of interest
    ax.set_extent([-180, 180, 30, 90], ccrs.PlateCarree())
    # Plot filled contours
    # ax.coastlines(resolution='110m')
    ax.coastlines(resolution='110m', linewidth=0.5)
    
    # Try contourf first, fallback to pcolormesh if it fails (curvilinear grids)
    try:
        if not use_pcolormesh:
            cf = ax.contourf(lons, lats, data, levels=levels, extend=extend, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
        else:
            raise ValueError("Force pcolormesh")
    except (ValueError, RuntimeError) as e:
        # Fallback to pcolormesh for curvilinear/irregular grids
        import warnings
        warnings.warn(f"contourf failed ({str(e)[:60]}), using pcolormesh instead")
        if levels is not None:
            vmin, vmax = levels[0], levels[-1]
        else:
            vmin, vmax = None, None
        cf = ax.pcolormesh(lons, lats, data, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, 
                          transform=ccrs.PlateCarree(), shading='auto')
    
    ax.set_title(title, loc='center', fontsize=14, pad=5.0)
    gl=ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,linewidth=1, color='c', alpha=1, linestyle='--')
    gl.xlocator = mticker.FixedLocator([-90,0,90,180])
    gl.xlines = False
    gl.ylocator = mticker.FixedLocator([0, 30, 65])
    # gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
    #                   color='gray', alpha=0.35, linestyle='--')

    # # Disable labels on the top and right of the plot
    # gl.top_labels = False
    # gl.right_labels = False

    # # Enable labels on the bottom and left of the plot
    # gl.bottom_labels = show_xticks
    # gl.left_labels = show_yticks
    # gl.xformatter = cticker.LongitudeFormatter()
    # gl.yformatter = cticker.LatitudeFormatter()
    # gl.xlabel_style = {'size': 18}
    # gl.ylabel_style = {'size': 18}
    
    return cf
# ====================================================================
import cartopy.crs as ccrs
import cartopy.mpl.ticker as mticker
import numpy as np
def plot_SH_panel(
    plot_cells, MODELS, SCEN_COLS,
    levels, cmap, pkw,
    field_label="SIC anomaly [% points]",
    suptitle="",
    out_fig=None,
    boundary_lat=0,
    prefer_imshow=True,
    figsize_scale=3.9,
    dpi=300
):
    """
    Plot a SH polar panel (nrows x ncols) using Lambert Azimuthal Equal Area projection.
    plot_cells: dict with keys (model, col_index) and values (field, lats, lons, ...)
    MODELS: list of row labels
    SCEN_COLS: list of column tuples, last element is title
    levels, cmap, pkw: colormap and colorbar settings
    field_label: colorbar label
    suptitle: figure title
    out_fig: if given, save to this path (png/pdf)
    boundary_lat: southernmost latitude to plot (default 0)
    prefer_imshow: use imshow for plotting (recommended)
    figsize_scale: size multiplier per panel
    dpi: output dpi
    """
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs

    nrows, ncols = len(MODELS), len(SCEN_COLS)
    fig_w, fig_h = figsize_scale * ncols, figsize_scale * nrows
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h),
        subplot_kw=dict(projection=ccrs.LambertAzimuthalEqualArea(
            central_longitude=180, central_latitude=-90))
    )

    mappable = None
    for i, model in enumerate(MODELS):
        for j in range(ncols):
            ax = axes[i, j]
            item = plot_cells.get((model, j), None)
            if item is None:
                ax.text(0.5, 0.5, "missing", transform=ax.transAxes, ha="center", va="center")
                continue
            field, lats, lons, _ = item
            mappable = SH_plot_data(
                field, lats=lats, lons=lons,
                levels=levels, cmap=cmap, extend=pkw.get("extend", "both"),
                title="", ax=ax, boundary_lat=boundary_lat, prefer_imshow=prefer_imshow
            )
        # Row label
        bb = axes[i, 0].get_position()
        fig.text(bb.x0 - 0.1, 0.5 * (bb.y0 + bb.y1), model,
                 rotation=90, va="center", ha="right", fontsize=18)
    # Column headers
    for j, (_, _, _, title) in enumerate(SCEN_COLS):
        axes[0, j].set_title(title, fontsize=18, pad=6)
    # Colorbar
    if mappable is not None:
        cax = fig.add_axes([0.15, 0.03, 0.7, 0.018])
        cb = fig.colorbar(mappable, cax=cax, orientation="horizontal")
        cb.set_label(field_label)
    # Title and layout
    if suptitle:
        fig.suptitle(suptitle, y=0.995, fontsize=18)
    plt.subplots_adjust(left=0.02, right=0.985, bottom=0.07, top=0.95, wspace=0.06, hspace=0.06)
    # Save if requested
    if out_fig:
        out_pdf = out_fig.replace(".png", ".pdf")
        plt.savefig(out_fig, dpi=dpi, bbox_inches="tight")
        plt.savefig(out_pdf, dpi=dpi, bbox_inches="tight")
        print(f"[OK] Figure saved → {out_fig}")
    return fig, axes

def SH_plot_data(data, lats, lons, levels, extend, cmap, norm, title, ax):
    if np.all(np.isnan(data)):
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        return None
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={'projection': ccrs.LambertAzimuthalEqualArea(central_longitude=180, central_latitude=-90)})
    ax.set_extent([-180, 180, -90, -30], ccrs.PlateCarree())
    ax.coastlines(resolution='110m', linewidth=0.5)
    cf = ax.contourf(lons, lats, data, levels=levels, extend=extend, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
    ax.set_title(title, loc='center', fontsize=14, pad=5.0)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1, color='c', alpha=1, linestyle='--')
    gl.xlocator = mticker.FixedLocator([-90, 0, 90, 180])
    gl.xlines = False
    gl.ylocator = mticker.FixedLocator([-65, -30, 0])
    return cf
# ====================================================================
def plot_data_Orthographic(data, lats, lons, levels=None, extend=None, cmap=None, title="", ax=None, show_xticks=True, show_yticks=False):
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={'projection': ccrs.Orthographic(central_longitude=-60, central_latitude=40)})

    # Plot filled contours
    cf = ax.contourf(
        lons, lats, data,
        levels=levels,
        cmap=cmap,
        extend=extend,
        transform=ccrs.PlateCarree()
    )

    # Coastlines and gridlines only — no manual ticks
    ax.coastlines()
    gl = ax.gridlines(draw_labels=False, linestyle='--', color='gray', linewidth=0.5, alpha=0.7)
    
    # Title
    ax.set_title(title, fontsize=16, pad=10)
    return cf
# ====================================================================
# def add_circular_boundary(ax, radius=0.5, n_points=200):
#     """
#     Clip the given GeoAxes to a circle in Axes-fraction coordinates.
#     """
#     theta = np.linspace(0, 2*np.pi, n_points)
#     verts = np.vstack([
#         0.5 + radius * np.sin(theta),
#         0.5 + radius * np.cos(theta)
#     ]).T
#     circle_path = Path(verts)
#     ax.set_boundary(circle_path, transform=ax.transAxes)


# def plot_sia_north_pole(sia, lons, lats, ax=None,
#                         boundary_lat=66.5, cmap='Blues',
#                         vmin=None, vmax=None, **pcol_kwargs):
#     """
#     Plot a 2D SIA field on a North-Polar Stereographic projection,
#     clipped to latitude ≥ boundary_lat.
    
#     Parameters
#     ----------
#     sia : 2D array-like (j×i)
#         Sea-ice area (or fraction) field.
#     lons, lats : 2D array-like (j×i)
#         The longitude and latitude coordinates for sia.
#     ax : GeoAxes (optional)
#         If None, a new GeoAxes with NorthPolarStereo() is created.
#     boundary_lat : float
#         Latitude (°N) of the circular outline (default 66.5).
#     cmap, vmin, vmax : passed to pcolormesh.
#     **pcol_kwargs : additional keyword args for pcolormesh.
    
#     Returns
#     -------
#     mesh : QuadMesh
#         The result of ax.pcolormesh(...).
#     """
#     # 1) Set up the polar stereographic axes
#     if ax is None:
#         fig = plt.gcf()
#         ax = fig.add_subplot(1,1,1, projection=ccrs.NorthPolarStereo())
    
#     # 2) Clip to a circle that corresponds roughly to boundary_lat
#     #    radius in axes coords = sin(90° - boundary_lat)
#     radius = np.sin(np.deg2rad(90 - boundary_lat))
#     add_circular_boundary(ax, radius=radius)
    
#     # 3) Limit the view extents to boundary_lat–90°N
#     ax.set_extent([-180, 180, boundary_lat, 90], ccrs.PlateCarree())
    
#     # 4) Plot the data
#     mesh = ax.pcolormesh(
#         lons, lats, sia,
#         transform=ccrs.PlateCarree(),
#         cmap=cmap, vmin=vmin, vmax=vmax,
#         **pcol_kwargs
#     )
    
#     # 5) Draw the 66.5° N outline explicitly
#     theta = np.linspace(0, 2*np.pi, 361)
#     circle_lons = np.degrees(theta) - 180  # from -180 to +180 for PlateCarree
#     circle_lats = np.full_like(theta, boundary_lat)
#     ax.plot(
#         circle_lons, circle_lats,
#         transform=ccrs.PlateCarree(),
#         color='k', linewidth=1
#     )
    
#     # 6) Add coastlines (optional—you can style further)
#     ax.coastlines()
    
#     return mesh
import numpy as np
import cartopy.crs as ccrs
import matplotlib.path as mpath
import matplotlib.pyplot as plt

def add_circular_boundary(ax, radius=0.5, n_points=200):
    theta = np.linspace(0, 2*np.pi, n_points)
    verts  = np.vstack([0.5 + radius*np.sin(theta), 0.5 + radius*np.cos(theta)]).T
    ax.set_boundary(mpath.Path(verts), transform=ax.transAxes)

def plot_sic_north_pole(sic2d, lons, lats, ax=None,
                        boundary_lat=66.5, cmap='RdBu_r',
                        vmin=None, vmax=None, **pcol_kwargs):
    """
    Plot a 2D *SIC anomaly* field (percentage points) on a North-Polar projection.
    Accepts 1D or 2D lons/lats. Returns the QuadMesh.
    """
    if ax is None:
        ax = plt.gca(projection=ccrs.NorthPolarStereo())
    elif not hasattr(ax, 'projection'):
        raise TypeError("ax must be a Cartopy GeoAxes (create subplot with a projection).")

    # circular clip
    add_circular_boundary(ax, radius=0.5)
    ax.set_extent([-180, 180, boundary_lat, 90], crs=ccrs.PlateCarree())

    # make 2D lon/lat if 1D
    if lons.ndim == 1 and lats.ndim == 1:
        LON, LAT = np.meshgrid(lons, lats)
    else:
        LON, LAT = lons, lats

    m = ax.pcolormesh(
        LON, LAT, sic2d,
        transform=ccrs.PlateCarree(),
        cmap=cmap, vmin=vmin, vmax=vmax,
        shading="auto", **pcol_kwargs
    )

    # outline of 66.5°N
    theta = np.linspace(-180, 180, 361)
    ax.plot(theta, np.full_like(theta, boundary_lat),
            transform=ccrs.PlateCarree(), color='k', lw=0.8)
    ax.coastlines(resolution="110m", linewidth=0.5)
    return m

# ======================================
"""
Plotting functions for AMOC depth vs. latitude
"""
def plot_amoc_depth_vs_latitude(ds, var='msftmz', title="AMOC Streamfunction (Sv)", cmap='RdBu_r', ax=None, vmin=None, vmax=None, levels=None):
    """
    Plots AMOC streamfunction (Sv) as a contour plot over depth and latitude.

    Parameters:
    - ds: xarray Dataset containing AMOC streamfunction with 'lev' (depth) and 'lat'
    - var: variable name in ds (default: 'msftmz')
    - title: title for the plot
    - cmap: colormap
    - ax: matplotlib axis object. If None, creates a new figure.
    - vmin, vmax: min/max for color scale
    - levels: contour levels (list or int)

    Returns:
    - ax: axis with the contour plot
    """
    # Convert to Sv
    amoc = ds[var].isel(basin=2).sel(year=2014) / 1e9  # time-averaged or snapshot
    lat = amoc['lat']
    depth = amoc['lev']

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Use user-supplied levels, or default to 11
    if levels is None:
        levels = 11

    cf = ax.contourf(lat, depth, amoc.transpose('lev', 'lat'), levels=levels, cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(cf, ax=ax, label='AMOC Streamfunction (Sv)')

    # Plot aesthetics
    ax.set_title(title)
    ax.set_xlabel('Latitude (°N)')
    ax.set_ylabel('Depth (m)')
    ax.invert_yaxis()  # so surface is at top

    return ax
# %%
"""
Plot the ocean heat content (OHC), SSH
"""
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
# copy from the Figure 2 script;
def circular_boundary(ax, radius=0.5):
    """
    Creates a circular boundary for polar projection.
    """
    theta = np.linspace(0, 2 * np.pi, 100)
    center = [0.5, 0.5]
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T * radius + center
    return mpath.Path(verts)

def plot_data_polar_with_mask(data, lats, lons, ax, levels, cmap, extend='both', title ='',
                    show_xticks=True, show_yticks=True, label_longitude=True,
                    min_lat=66.5, hatch_mask=None, hatch_pattern='///'):
    """
    Plot polar map; optionally overlay hatching where hatch_mask==True.
    hatch_mask may be a numpy array, masked array or xarray DataArray.
    """
    lon2d, lat2d = np.meshgrid(lons, lats)
    masked_data = np.where(lat2d >= min_lat, data, np.nan)

    # main filled contours (lower zorder)
    cf = ax.contourf(lon2d, lat2d, masked_data,
                     levels=levels, cmap=cmap, extend=extend,
                     transform=ccrs.PlateCarree(), zorder=1)

    ax.set_extent([-180, 180, min_lat, 90], crs=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.6)
    ax.set_boundary(circular_boundary(ax), transform=ax.transAxes)

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                      linewidth=0.5, linestyle='--', color='gray', alpha=0.6)
    if show_xticks:
        gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 60))
    if show_yticks:
        gl.ylocator = mticker.FixedLocator(np.arange(min_lat, 91, 15))

    if label_longitude:
        for lon in np.arange(0, 360, 60):
            ax.text(lon, min_lat + 31, f"{lon}°",
                    transform=ccrs.PlateCarree(),
                    ha='center', va='center', fontsize=16, color='gray')

    # --- optional hatching overlay where hatch_mask==True ---
    if hatch_mask is not None:
        hm = np.asarray(hatch_mask)
        if isinstance(hm, np.ma.MaskedArray):
            hm = hm.filled(0)
        if hm.ndim > 2:
            hm = np.squeeze(hm)

        if hm.shape != lat2d.shape:
            if hm.T.shape == lat2d.shape:
                hm = hm.T
            else:
                hm = np.broadcast_to(hm, lat2d.shape)

        hatch_field = np.where(((lat2d >= min_lat) & (hm.astype(bool))), 1, np.nan)

        # --- IMPORTANT: grab newly-added Axes collections to style hatches ---
        n0 = len(ax.collections)

        ax.contourf(
            lon2d, lat2d, hatch_field,
            levels=[0.5, 1.5],
            hatches=[hatch_pattern],
            colors="none",  # hatch only
            transform=ccrs.PlateCarree(),
            zorder=2
        )

        new_cols = ax.collections[n0:]  # these are the contourf collections we just added
        for coll in new_cols:
            coll.set_edgecolor("cyan")   # <- change to "magenta" if you want
            coll.set_linewidth(0.6)
            coll.set_facecolor("none")
            coll.set_linewidth(0.0)      # <<< removes the outlines

    ax.set_title(title, loc='center', fontsize=20, pad=5.0)
    return cf
# %%
def plot_data_polar(data, lats, lons, ax, levels, cmap, norm, extend='both', title ='',
                    show_xticks=True, show_yticks=True, label_longitude=True,
                    min_lat=60.5, hatch_mask=None, hatch_pattern='///'):

    # handle 1D vs 2D lat/lon
    if lats.ndim == 1 and lons.ndim == 1:
        lon2d, lat2d = np.meshgrid(lons, lats)
    elif lats.ndim == 2 and lons.ndim == 2:
        lat2d = lats
        lon2d = lons
    else:
        raise ValueError("lats and lons must both be 1D or both be 2D arrays")

    # *** NEW: wrap longitudes to [-180, 180] ***
    if np.nanmax(lon2d) > 180:
        lon2d = ((lon2d + 180) % 360) - 180

    # Ensure data shape
    data = np.asarray(data)  # in case it's a DataArray
    if data.shape != lat2d.shape:
        if data.T.shape == lat2d.shape:
            data = data.T
        else:
            raise ValueError(...)

    masked_data = np.where(lat2d >= min_lat, data, np.nan)

    # Use pcolormesh to avoid Shapely MultiPolygon issues from contourf on polar projection
    # Only create BoundaryNorm if norm is not provided (allows TwoSlopeNorm, etc.)
    if norm is None:
        from matplotlib.colors import BoundaryNorm
        norm = BoundaryNorm(levels, ncolors=plt.get_cmap(cmap).N, clip=False)
    
    cf = ax.pcolormesh(
        lon2d, lat2d, masked_data,
        cmap=cmap, norm=norm,
        transform=ccrs.PlateCarree(), shading='auto', zorder=1
    )

    ax.set_extent([-180, 180, min_lat, 90], crs=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.6)
    # Use existing helper to apply a circular boundary; radius based on min_lat
    add_circular_boundary(ax, radius=np.sin(np.deg2rad(90 - min_lat)))
    ax.set_title(title, loc='center', fontsize=20, pad=5.0)
    return cf
# ====================================================================
# plot global map with Robinson projection
def plot_data_robinson(data, lats, lons, levels=None, extend=None, cmap=None, norm=None, title="",
              ax=None, show_xticks=False, show_yticks=False):
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=0.0)})
    # Add coastlines and gridlines
    ax.coastlines(resolution='110m')
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                      colors='gray', alpha=0.25, linestyle='--', linewidth=0.25)
    
    gl.top_labels = False
    gl.right_labels = False
    gl.bottom_labels = show_xticks
    gl.left_labels = show_yticks
    gl.xformatter = cticker.LongitudeFormatter()
    gl.yformatter = cticker.LatitudeFormatter()
    gl.xlabel_style = {'size': 18}
    gl.ylabel_style = {'size': 18}

    # handle 1D vs 2D lat/lon
    if lats.ndim == 1 and lons.ndim == 1:
        lon2d, lat2d = np.meshgrid(lons, lats)
    elif lats.ndim == 2 and lons.ndim == 2:
        lat2d = lats
        lon2d = lons
    else:
        raise ValueError("lats and lons must both be 1D or both be 2D arrays")

    # *** NEW: wrap longitudes to [-180, 180] ***
    if np.nanmax(lon2d) > 180:
        lon2d = ((lon2d + 180) % 360) - 180

    # Ensure data shape
    data = np.asarray(data)  # in case it's a DataArray
    if data.shape != lat2d.shape:
        if data.T.shape == lat2d.shape:
            data = data.T
        else:
            raise ValueError("Data shape does not match lat/lon shape")
    # Contour plot
    cf = ax.contourf(lon2d, lat2d, data, levels=levels, extend=extend,
                     cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
    

    ax.set_title(title, loc='center', fontsize=22, pad=5.0)
    return cf