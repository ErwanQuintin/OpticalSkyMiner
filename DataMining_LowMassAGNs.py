import subprocess
import shlex
import matplotlib
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



master_sources = load_master_sources('Master_source_LowMass.fits')

def lum_vs_AGNmass():
    tab_xlum= []
    tab_bh_mass=[]
    for ms in tqdm(master_sources.values()):
        if not np.isnan(ms.sources_fluxes).all() and np.nanmax(ms.sources_fluxes)>0:
            tab_bh_mass.append(ms.glade_bh_mass)
            tab_xlum.append(np.nanmax(ms.sources_fluxes)*ms.flux_lum_conv_factor)

    plt.rcParams.update({'font.size': 15})
    fig = plt.hexbin(tab_bh_mass, tab_xlum, bins="log", xscale="log", yscale="log", mincnt=1, cmap="cmr.ocean", gridsize=25)
    plt.plot([1e3, 1e10], [1.26e38 * 1e3, 1.26e38 * 1e10], c="r", ls="--", lw=5)
    plt.plot([1e3, 1e10], [1.26e38 * 1e3 / 10, 1.26e38 * 1e10 / 10], c="r", ls="--", lw=4)
    plt.plot([1e3, 1e10], [1.26e38 * 1e3 / 100, 1.26e38 * 1e10 / 100], c="r", ls="--", lw=3)
    plt.xlabel(r"Estimated BH mass (M$_{\odot}$)")
    plt.ylabel(r"X-ray luminosity (erg.s$^{-1}$)")
    cbar = plt.colorbar(fig)
    cbar.set_label("Number of AGNs")
    plt.loglog()
    plt.show()
#lum_vs_AGNmass()

def select_variable_AGNs():
    variable_agns=[]
    variability_table=[]
    for ms in tqdm(master_sources.values()):
        variability_table.append(ms.var_ratio)
        if ms.var_ratio>5:
            variable_agns.append(ms)
    plt.figure()
    plt.hist(variability_table, np.geomspace(1,1e3,100))
    plt.loglog()
    plt.xlabel("Variability amplitude")
    return variable_agns

def generate_table_XMM_obsids():
    tab_obsid=[]
    tab_srcname=[]
    for ms in tqdm(master_sources.values()):
        if 'XMM' in ms.sources.keys():
            for obsid in ms.sources['XMM'].obsids:
                tab_obsid.append(str(obsid).zfill(10))
                tab_srcname.append(ms.sources["XMM"].name)
    table = Table((tab_srcname, tab_obsid),names=('IAUNAME','ObsID'))
    table.write(os.path.join(path_to_catalogs,'XMM_identifiers_LowMassAGNs.fits'))
generate_table_XMM_obsids()