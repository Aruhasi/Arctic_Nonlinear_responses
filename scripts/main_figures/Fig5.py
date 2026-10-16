#!/usr/bin/env python3
# %% 
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature

ROOT_DEFAULT=Path('/work/mh0033/m301036/Atlantic_heat_capacitor_Arctic/Atlantic_capacitor_on_Arctic')
SCENARIOS=('ssp119','ssp126')
SCENARIO_LABELS={'ssp119':'SSP1-1.9','ssp126':'SSP1-2.6'}

# Match the model order used in the other main-text LE figures.
# Diverging palette used by manuscript Figs. 1-2.
COLORS_255=((32,54,79),(49,100,108),(78,146,128),(150,184,155),
            (220,223,210),(236,217,207),(212,156,135),(184,98,101),
            (139,52,94),(80,24,78))
ARU_CMAP_DIV=LinearSegmentedColormap.from_list(
    'aru_pastel_div',[(r/255,g/255,b/255) for r,g,b in COLORS_255],N=256)

HATCH_PATTERN='//'          # sparse diagonal sign-agreement hatch
HATCH_COLOR='azure'
AGREE_FRAC=0.75             # at least 3 of 4 LEs

PANEL_FS=16
TITLE_FS=11
ROW_FS=11
TICK_FS=8.5
CBAR_LABEL_FS=9.2
CBAR_TICK_FS=8.0

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=ROOT_DEFAULT)
    p.add_argument('--output-name',default='Fig5_ocean_memory_AWPS_bridge_polished')
    return p.parse_args()

def circular_boundary(ax):
    th=np.linspace(0,2*np.pi,240)
    verts=np.column_stack([np.sin(th),np.cos(th)])*0.5+np.array([0.5,0.5])
    ax.set_boundary(mpath.Path(verts),transform=ax.transAxes)

def cyclic(da):
    lon=da.lon.values
    if len(lon)<2:
        return da
    step=float(lon[1]-lon[0])
    first=da.isel(lon=0).assign_coords(lon=float(lon[-1]+step))
    return xr.concat([da,first],dim='lon')

def sign_agreement_fraction(model_field,mmm):
    """Fraction of valid models sharing the sign of the plotted MMM field."""
    model_field,mmm=xr.align(model_field,mmm,join='inner')
    valid=np.isfinite(model_field)
    n=valid.sum('model')
    pos=xr.where(n>0,(model_field>0).sum('model')/n,np.nan)
    neg=xr.where(n>0,(model_field<0).sum('model')/n,np.nan)
    return xr.where(mmm>=0,pos,neg)

def add_hatching(ax,mask):
    hm=cyclic(mask.astype(float))
    hatch=ax.contourf(
        hm.lon,hm.lat,hm,levels=[0.5,1.5],colors='none',
        hatches=[HATCH_PATTERN],transform=ccrs.PlateCarree(),zorder=3)
    # Matplotlib >=3.8 exposes setters on ContourSet; older releases expose
    # the individual hatch artists through the collections attribute.
    if hasattr(hatch,'set_edgecolor'):
        hatch.set_edgecolor(HATCH_COLOR)
        hatch.set_linewidth(0.0)
    else:
        collections=getattr(hatch,'collections',())
        if not collections:
            raise RuntimeError(
                'Unsupported Matplotlib ContourSet API: cannot style hatching')
        for artist in collections:
            artist.set_edgecolor(HATCH_COLOR)
            artist.set_linewidth(0.0)

def plot_map(ax,da,levels,hmask=None):
    da=cyclic(da)
    norm=TwoSlopeNorm(vmin=float(levels[0]),vcenter=0.,vmax=float(levels[-1]))
    cf=ax.contourf(
        da.lon,da.lat,da,levels=levels,cmap=ARU_CMAP_DIV,norm=norm,
        extend='both',transform=ccrs.PlateCarree(),zorder=1)
    if hmask is not None:
        add_hatching(ax,hmask)
    ax.add_feature(cfeature.LAND,facecolor='#E6E6E6',edgecolor='none',zorder=4)
    ax.coastlines(linewidth=.60,color='#202020',zorder=7)
    ax.set_extent([-180,180,66.5,90],crs=ccrs.PlateCarree())
    circular_boundary(ax)
    return cf

def panel_letter(ax,label,x=-.10,y=1.035):
    ax.text(x,y,label,transform=ax.transAxes,fontsize=PANEL_FS,
            fontweight='bold',ha='left',va='bottom')

def plot_main_figure(spatial,figdir,output_name,elev,slev):
    fig=plt.figure(figsize=(7.4,7.0))
    gs=fig.add_gridspec(
        2,2,left=.07,right=.995,bottom=.17,top=.94,
        wspace=-.10,hspace=.035)
    letters=iter('abcd')
    cf_error=cf_sens=None
    last_error_ax=last_sens_ax=None

    for row,scenario in enumerate(SCENARIOS):
        ds=spatial[scenario]
        error_ax=fig.add_subplot(
            gs[row,0],projection=ccrs.NorthPolarStereo(central_longitude=0))
        error_hatch=(ds.sat_error_sign_agreement_fraction>=AGREE_FRAC)
        cf_error=plot_map(
            error_ax,ds.sat_transfer_error_mmm,elev,error_hatch)

        sens_ax=fig.add_subplot(
            gs[row,1],projection=ccrs.NorthPolarStereo(central_longitude=0))
        sens_agreement=sign_agreement_fraction(
            ds.model_ohc_sat_error_sensitivity,
            ds.ohc_sat_error_sensitivity_mmm)
        cf_sens=plot_map(
            sens_ax,ds.ohc_sat_error_sensitivity_mmm,slev,
            sens_agreement>=AGREE_FRAC)

        pattern_r=float(ds.attrs.get(
            'weighted_spatial_pattern_corr_error_vs_sensitivity',np.nan))
        if np.isfinite(pattern_r):
            sens_ax.text(
                .96,.96,rf'$r_{{\rm spatial}}={pattern_r:.2f}$',
                transform=sens_ax.transAxes,ha='right',va='top',
                fontsize=8.5,
                bbox=dict(facecolor='white',edgecolor='none',
                          alpha=.72,pad=1.5),zorder=10)

        letter_y=1.035 if row==0 else 1.01
        panel_letter(error_ax,next(letters),y=letter_y)
        panel_letter(sens_ax,next(letters),y=letter_y)
        error_ax.text(
            -.15,.50,SCENARIO_LABELS[scenario],
            transform=error_ax.transAxes,rotation=90,
            fontsize=ROW_FS,fontweight='bold',ha='center',va='center')

        if row==0:
            error_ax.set_title(
                'SAT scaling error during CO$_2$ decline\n'
                +r'$E^{\mathrm{SAT}}=T_{\mathrm{actual},\downarrow}'
                 r'-T_{\mathrm{AWPS},\uparrow}$',
                fontsize=TITLE_FS,pad=7)
            sens_ax.set_title(
                'OHC-SAT scaling-error coupling\n'
                +r'$\gamma_T=\partial E^{\mathrm{SAT}}/'
                 r'\partial E^{\mathrm{OHC}}$',
                fontsize=TITLE_FS,pad=7)

        last_error_ax=error_ax
        last_sens_ax=sens_ax

    pos1=last_error_ax.get_position()
    pos2=last_sens_ax.get_position()
    cax1=fig.add_axes(
        [pos1.x0-.05*pos1.width,.095,1.10*pos1.width,.018])
    cb1=fig.colorbar(cf_error,cax=cax1,orientation='horizontal')
    cb1.set_label(
        r'SAT scaling error, $E^{\mathrm{SAT}}$ ($^\circ$C)',
        fontsize=CBAR_LABEL_FS,labelpad=4)
    cb1.set_ticks(np.linspace(-1.5,1.5,7))
    cb1.ax.tick_params(labelsize=CBAR_TICK_FS,length=3)

    cax2=fig.add_axes(
        [pos2.x0-.05*pos2.width,.095,1.10*pos2.width,.018])
    cb2=fig.colorbar(cf_sens,cax=cax2,orientation='horizontal')
    cb2.set_label(
        r'OHC-SAT scaling-error coupling, $\gamma_T$ ($^\circ$C ZJ$^{-1}$)',
        fontsize=CBAR_LABEL_FS,labelpad=4)
    cb2.set_ticks(np.linspace(-.2,.2,5))
    cb2.ax.tick_params(labelsize=CBAR_TICK_FS,length=3)

    out=figdir/f'{output_name}.pdf'
    fig.savefig(out,dpi=600,bbox_inches='tight',facecolor='white')
    fig.savefig(
        out.with_suffix('.png'),dpi=300,bbox_inches='tight',
        facecolor='white')
    plt.close(fig)
    print(f'Saved {out}')
    return out
# %%
def main():
    args=parse_args()
    root=args.root.expanduser().resolve()
    datadir=root/'docs/data/Spatial_pattern/Fig5_redesign'
    figdir=root/'docs/figs/Spatial_pattern'
    figdir.mkdir(parents=True,exist_ok=True)

    spatial={}
    for scenario in SCENARIOS:
        path=datadir/f'Fig5_spatial_bridge_{scenario}.nc'
        if not path.is_file():
            raise FileNotFoundError(f'Run step 2 first: {path}')
        spatial[scenario]=xr.open_dataset(path).load()
    elev=np.linspace(-1.5,1.5,13)
    slev=np.linspace(-.2,.2,17)
    plot_main_figure(
        spatial,figdir,args.output_name,elev,slev)
# %%
if __name__=='__main__':
    main()
# %%
