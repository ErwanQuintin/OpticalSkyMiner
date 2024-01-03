import subprocess
import shlex
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('TkAgg')
from LoadMasterSources import *
from astropy.coordinates import SkyCoord
import astropy.units as u
import os
from matplotlib import rc
import cmasher as cmr
plt.rcParams.update({'font.size': 15})
rc('text', usetex=True)
rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})



dic_master_sources = load_master_sources(only_galaxies=True)

tab_to_plot=[]
for ms in dic_master_sources.values():
    if len(ms.optical_sources.keys())>2:
        tab_to_plot.append(ms)
tab_to_plot[10].plot_lightcurve()

def compare_GALEX_to_OM_UVOT():
    """This function will compare the GALEX NUV flux to any UV flux from OM or UVOT, allowing to check for the
    consistency between these bands"""

    dic_OM_GALEX={"UVW2":[],"UVM2":[],"UVW1":[]} #OM fluxes that match GALEX
    dic_GALEX_OM={"UVW2":[],"UVM2":[],"UVW1":[]} #GALEX fluxes that match OM
    dic_UVOT_GALEX={"UVW2":[],"UVM2":[],"UVW1":[]} #UVOT fluxes that match GALEX
    dic_GALEX_UVOT={"UVW2":[],"UVM2":[],"UVW1":[]} #GALEX fluxes that match UVOT

    for ms in tqdm(dic_master_sources.values()):
        if 'GALEX' in ms.optical_sources.keys():
            galex_nuv_flux = ms.optical_sources['GALEX'].band_flux[0][1]

            if 'OM' in ms.optical_sources.keys():
                for detection in ms.optical_sources['OM'].band_flux:
                    for ind, band in enumerate(["UVW2","UVM2","UVW1"]):
                        if not np.isnan(detection[ind]):
                            dic_OM_GALEX[band].append(detection[ind])
                            dic_GALEX_OM[band].append(galex_nuv_flux)
            if 'UVOT' in ms.optical_sources.keys():
                for detection in ms.optical_sources['UVOT'].band_flux:
                    for ind, band in enumerate(["UVW2","UVM2","UVW1"]):
                        if not np.isnan(detection[ind]):
                            dic_UVOT_GALEX[band].append(detection[ind])
                            dic_GALEX_UVOT[band].append(galex_nuv_flux)

    fig, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2,3)
    for band, ax in zip(["UVW2","UVM2","UVW1"],(ax1,ax2,ax3)):
        ax.hexbin(dic_OM_GALEX[band],dic_GALEX_OM[band], bins='log',xscale='log',yscale='log',mincnt=1,cmap="cmr.ocean")
        ax.set_xlabel(f"OM {band} flux")
        ax.set_ylabel('GALEX NUV flux')
        ax.set_title(band)
    for band, ax in zip(["UVW2","UVM2","UVW1"],(ax4,ax5,ax6)):
        ax.hexbin(dic_UVOT_GALEX[band],dic_GALEX_UVOT[band], bins='log',xscale='log',yscale='log',mincnt=1,cmap="cmr.ocean")
        ax.set_xlabel(f"UVOT {band} flux")
        ax.set_ylabel('GALEX NUV flux')
        ax.set_title(band)
    for ax in (ax1,ax2,ax3,ax4,ax5,ax6):
        ax.plot([1e-18,1e-13],[1e-18,1e-13], c='k',lw=4)
        ax.plot([1e-18,1e-13],[1e-18,1e-13], c='w',lw=1)
        ax.plot([1e-18,1e-13],[3e-18,3e-13], c='k',lw=4)
        ax.plot([1e-18,1e-13],[3e-18,3e-13], c='w',lw=1,ls='--')
        ax.plot([1e-18,1e-13],[0.33e-18,0.33e-13], c='k',lw=4)
        ax.plot([1e-18,1e-13],[0.33e-18,0.33e-13], c='w',lw=1,ls='--')
    plt.show()


def band_GLADE_luminosities():
    """This function plots the histograms of luminosities of each band, using the GLADE+ distance to convert from
    flux to luminosity"""
    tab_luminosities=[[] for band in range(6)]
    for ms in tqdm(dic_master_sources.values()):
        if not np.isnan(ms.glade_distance):
            for ind_band in range(6):
                if not np.isnan(ms.optical_max_lower[ind_band]):
                    tab_luminosities[ind_band].append(ms.optical_max_lower[ind_band]*
                                                      ms.flux_lum_conv_factor*band_width['OM'][ind_band])
    fig, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2,3)
    for band_ind, ax in enumerate((ax1,ax2,ax3,ax4,ax5,ax6)):
        ax.hist(tab_luminosities[band_ind], bins=np.geomspace(1e30,1e50,100))
        ax.loglog()
    plt.show()

