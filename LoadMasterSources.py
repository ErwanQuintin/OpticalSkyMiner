import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
import astropy.units as u
from tqdm import tqdm
from astropy.constants import c
from astropy.coordinates import SkyCoord
import webbrowser
from itertools import combinations
from matplotlib import rc
import os
import cmasher as cmr
from astropy.coordinates import Longitude, Latitude, Angle
from astroquery.hips2fits import hips2fits
plt.rcParams.update({'font.size': 20})
rc('text', usetex=True)

cmap_to_use="cmr.ocean"

src_names={}
optical_catalogs = ["OM","UVOT","GALEX"]
optical_colors_cat={}
colors = cmr.take_cmap_colors(cmap_to_use,len(optical_catalogs), cmap_range=(0,0.7))
for ind, opt_cat in enumerate(optical_catalogs):
    src_names[opt_cat]=f"{opt_cat}_IAUNAME"
    optical_colors_cat[opt_cat]=colors[ind]
optical_formats={'OM':'o','UVOT':'s', 'GALEX':'x'}
colors = cmr.take_cmap_colors(cmap_to_use,6, cmap_range=(0,0.7))
optical_colors={}
for ind, band in enumerate(["UVW2","UVM2","UVW1","U","B","V"]):
    optical_colors[band] = colors[ind]



# Defining all the catalog-related column names
time_names={"OM":"MJD_START",
            "UVOT":"DATE_MIN"}

band_flux_names = {"OM":["UVW2_AB_FLUX","UVM2_AB_FLUX","UVW1_AB_FLUX","U_AB_FLUX","B_AB_FLUX","V_AB_FLUX"],
                  "UVOT":["UVW2_FLUX","UVM2_FLUX","UVW1_FLUX","U_FLUX","B_FLUX","V_FLUX"],
                   "GALEX":["flux_FUV_perAngstrom","flux_NUV_perAngstrom"]}
band_fluxerr_names = {"OM":["UVW2_AB_FLUX_ERR","UVM2_AB_FLUX_ERR","UVW1_AB_FLUX_ERR","U_AB_FLUX_ERR","B_AB_FLUX_ERR","V_AB_FLUX_ERR"],
                     "UVOT":["UVW2_FLUX_ERR","UVM2_FLUX_ERR","UVW1_FLUX_ERR","U_FLUX_ERR","B_FLUX_ERR","V_FLUX_ERR"],
                     "GALEX":["fluxerr_FUV_perAngstrom","fluxerr_NUV_perAngstrom"]}
band_center = {"OM":[2120,2310,2910,3440,4500,5430],
               "UVOT":[2120,2310,2910,3440,4500,5430],
               "GALEX":[1565,2300]}
band_width = {"OM":[500,480,830,840,1050,700],
               "UVOT":[500,480,830,840,1050,700],
               "GALEX":[440,1120]}

optical_effective_wavelengths=[2120,2310,2910,3440,4500,5430] #In Angstroms, used to convert from erg/s/cm2/angstroms to erg/s/cm2

path_to_catalogs = os.path.join(os.getcwd(),"OpticalCatalogs")

def click_action(ra, dec, om_obsids, uvot_obsids):
    url_esaskyDSS = "http://sky.esa.int/?target="+str(np.round(ra,4))+" "+str(np.round(dec,4))+"&hips=DSS2+color&fov=0.1&cooframe=J2000&sci=true&lang=en"
    url_esaskyXMM = "http://sky.esa.int/?target=" + str(np.round(ra, 4)) + " " + str(
        np.round(dec, 4)) + "&hips=XMM-Newton+EPIC+color&fov=0.1&cooframe=J2000&sci=true&lang=en"
    url_esaskyChandra = "http://sky.esa.int/?target=" + str(np.round(ra, 4)) + " " + str(
        np.round(dec, 4)) + "&hips=Chandra+RGB&fov=0.1&cooframe=J2000&sci=true&lang=en"
    url_simbad = "http://simbad.u-strasbg.fr/simbad/sim-coo?Coord="+str(ra)+"+"+str(dec)+"&Radius=1&Radius.unit=arcmin&submit=submit+query"
    url_xmm = f"http://xmm-catalog.irap.omp.eu/sources?f={ra}%20{dec}"
    webbrowser.get('firefox').open(url_simbad)
    webbrowser.get('firefox').open(url_esaskyDSS, new=0)
    webbrowser.get('firefox').open(url_esaskyXMM, new=0)
    webbrowser.get('firefox').open(url_esaskyChandra, new=0)
    webbrowser.get('firefox').open(url_xmm, new=0)
    for om_obsid in om_obsids:
        url_om = f"http://nxsa.esac.esa.int/nxsa-web/#obsid={om_obsid}"
        webbrowser.get('firefox').open(url_om, new=0)
    # for uvot_obsid in uvot_obsids:
    #     url_uvot = f"http://nxsa.esac.esa.int/nxsa-web/#obsid={om_obsid}"
    #     webbrowser.get('firefox').open(url_uvot, new=0)



class OpticalSource:
    """
    An OpticalSource object corresponds to a source from one of the optical/UV catalogs. It has several attributes:
    - catalog: the corresponding catalog name, in the same naming convention as the catalog Table defined at the top
    - iau_name: the name of the source, considered as a unique identifier
    - band_flux and band_fluxerr: the fluxes and errors in the 6 bands, given in erg/s/cm2/A. For GALEX, there are only
    two bands.
    - timesteps: a table containing the MJD dates of detections
    In the end, each Source will be associated to a unique MasterSource, each MasterSource having Source objects from
    several distinct catalogs
    """
    def __init__(self, catalog, iau_name, timesteps, band_flux, band_fluxerr, obsids=[]):
        """
        Initialisation function, used to build the OpticalSource object.
        """
        self.catalog = catalog
        self.name = iau_name
        self.master_source = []
        self.band_flux = band_flux
        self.band_fluxerr=band_fluxerr
        self.timesteps=[float(elt) for elt in timesteps]
        self.obsids = obsids

class MasterSource:
    """
    A MasterSource corresponds to a single physical source, built on the association of multiple archival catalogs.
    A MasterSource has several attributes:
    - source: A dictionary which gives access to the underlying catalogs sources, which are Source objects in our framework.
    The keys of this dictionary are the names of the corresponding catalogs.
    - optical_var: correspond to the variability of each of the six energy bands

    A MasterSource only has one method, plot_lightcurve(), which produces a multi-panel plot of all relevant information
    """
    def __init__(self, id, ra, dec, poserr, tab_optical_sources):
        """
        Initialisation function, used to build a MasterSource object. We also compile the multi-instrument properties at
        this stage (variability,...)
        :param id: Identifier of the MasterSource, used to access it in a dictionary with ms.id as a key, and ms as value
        :param ra: RA of the MasterSource computed as weighted average of the constituting Source objects
        :param dec: Dec of the MasterSource computed as weighted average of the constituting Source objects
        :param poserr: 1 sigma Position Error of the MasterSource computed as weighted average of the constituting Source objects
        :param tab_optical_sources: Table containing the OpticalSource objects
        """

        self.id = id
        self.ra = float(ra)
        self.dec = float(dec)
        self.pos_err = float(poserr)

        self.optical_sources={}
        self.optical_obsids={"OM":[],"UVOT":[]}
        self.optical_min_upper = [1,1,1,1,1,1]
        self.optical_var = [1, 1, 1, 1, 1, 1]
        self.optical_max_lower = [0,0,0,0,0,0]
        for opt_source in tab_optical_sources:
            self.optical_sources[opt_source.catalog]=opt_source
            self.optical_obsids[opt_source.catalog]=opt_source.obsids
            opt_source.master_source = self
            if len(opt_source.band_flux)>0:
                if opt_source.catalog!='GALEX':
                    lower_fluxes = opt_source.band_flux-opt_source.band_fluxerr
                    upper_fluxes = opt_source.band_flux+opt_source.band_fluxerr
                else: #For GALEX, we consider NUV as UVM2, the rest is ignored. This might need to change in the future
                    upper_fluxes = np.array([[np.nan,opt_source.band_flux[0][1]+opt_source.band_fluxerr[0][1],np.nan,np.nan,np.nan,np.nan]])
                    lower_fluxes = np.array([[np.nan,opt_source.band_flux[0][1]-opt_source.band_fluxerr[0][1],np.nan,np.nan,np.nan,np.nan]])
                self.optical_min_upper = np.nanmin([self.optical_min_upper, np.nanmin(upper_fluxes, axis=0)], axis=0)
                self.optical_max_lower = np.nanmax([self.optical_max_lower, np.nanmax(lower_fluxes, axis=0)], axis=0)
                self.optical_var = np.array(self.optical_max_lower) / np.array(self.optical_min_upper)

        self.glade_distance=np.nan
        self.glade_stellar_mass = np.nan
        self.flux_lum_conv_factor = np.nan

    def plot_lightcurve(self, with_image=True):
        """
        Produces a multi-panel plot with most of the useful multi-instrument information about the source.
         From left to right and top to bottom:
        1. Long term multi-instrument optical/UV lightcurves
        2. Multi-instrument optical/UV spectra, used to assess a spectral change in the OpticalSources
        3. DSS Image (optical)
        4. GALEX Image (UV)
        :return: Nothing
        """
        plt.rcParams.update({'font.size': 15})
        if with_image:
            fig, [[ax1, ax2], [ax3,ax4]] = plt.subplots(2,2, figsize=(10,10))
        else:
            fig, [ax1, ax2] = plt.subplots(1,2, figsize=(10,10))

        plt.suptitle(f'More details', picker=True, bbox=dict(facecolor=(180 / 256., 204 / 256., 252 / 256.)))
        fig.canvas.mpl_connect('pick_event', lambda event: click_action(self.ra, self.dec,
                                                                        self.optical_obsids['OM'],
                                                                        self.optical_obsids['UVOT']))

        optical_band_observed={"UVW2":False,"UVM2":False,"UVW1":False,"U":False,"B":False,"V":False}
        for cat in optical_catalogs:
            if cat in self.optical_sources.keys():
                opt_source = self.optical_sources[cat]
                lightcurves = np.transpose(opt_source.band_flux)
                ligthcurve_errors = np.transpose(opt_source.band_fluxerr)
                for lightcurve, lightcurve_err, band in zip(lightcurves, ligthcurve_errors, ["UVW2","UVM2","UVW1","U","B","V"]):
                    if not np.isnan(lightcurve).all():
                        optical_band_observed[band]=True
                    ax1.errorbar(Time(opt_source.timesteps, format='mjd').decimalyear, lightcurve, yerr=lightcurve_err,
                                 fmt=optical_formats[cat], markeredgecolor='gray', c=optical_colors[band])

                for det, det_err in zip(opt_source.band_flux, opt_source.band_fluxerr):
                    good_indices=np.where(np.array(det)>0)
                    ax2.plot(np.array(band_center[cat])[good_indices],
                             np.array(det)[good_indices], c=optical_colors_cat[cat],
                             lw=3)
                    ax2.errorbar(band_center[cat], np.array(det),
                                 yerr=np.array(det_err),
                                 xerr=np.array(band_width[cat])/2,
                                 fmt="", c=optical_colors_cat[cat], alpha=0.4)
                    ax2.scatter(band_center[cat], np.array(det) , facecolor=optical_colors_cat[cat], marker="o",
                                edgecolor='gray', zorder=1)
                ax2.errorbar([], [], [], [], fmt='o', markeredgecolor='gray',
                             c=optical_colors_cat[cat], label=opt_source.name)
        for band in ["UVW2","UVM2","UVW1","U","B","V"]:
            if optical_band_observed[band]:
                ax1.errorbar([], [], fmt="o", c=optical_colors[band], label=band, markeredgecolor='gray')

        ax1.legend()
        ax1.set_xlabel("Time")
        ax1.set_ylabel(r"Flux ($erg~s^{-1}~cm^{-2}$~\AA$^{-1}$)")
        ax1.set_yscale("log")

        ax2.legend()
        ax2.set_xlabel(r"Wavelength (\AA)")
        ax2.set_ylabel(r"Flux ($erg~s^{-1}~cm^{-2}$~\AA$^{-1}$)")
        ax2.set_yscale("log")

        if with_image:
            try:
                size = 1500
                fov = 2 * u.arcmin

                result = hips2fits.query(
                    hips='CDS/P/DSS2/color',
                    width=size,
                    height=size,
                    ra=Longitude(self.ra * u.deg),
                    dec=Latitude(self.dec * u.deg),
                    fov=Angle(fov),
                    projection="AIT",
                    get_query_payload=False,
                    format="jpg",
                    min_cut=0.5,
                    max_cut=99.5
                )
                im = ax3.imshow(result)
                positions_x = [0.1 * size, 0.35 * size]
                positions_y = [0.15 * size, 0.15 * size]
                text_position_x = 0.225 * size
                text_position_y = 0.1 * size
                ax3.plot(positions_x, positions_y, c="w",lw=3)
                ax3.scatter(positions_x, positions_y, c="w", marker="o", s=20)
                scaletext = 30
                ax3.text(text_position_x, text_position_y, f'{scaletext}"', c="w", fontsize=20,
                         horizontalalignment='center')
                ax3.axis("off")
                c1 = plt.Circle((size // 2, size // 2), size * 3 * self.pos_err / (fov.to(u.arcsec).value), color='r',
                                fill=False)
                ax3.add_patch(c1)

                result = hips2fits.query(
                    hips='CDS/P/GALEXGR6/AIS/color',
                    width=size,
                    height=size,
                    ra=Longitude(self.ra * u.deg),
                    dec=Latitude(self.dec * u.deg),
                    fov=Angle(fov),
                    projection="AIT",
                    get_query_payload=False,
                    format="jpg",
                    min_cut=0.5,
                    max_cut=99.5
                )
                im = ax4.imshow(result)
                positions_x = [0.1 * size, 0.35 * size]
                positions_y = [0.15 * size, 0.15 * size]
                text_position_x = 0.225 * size
                text_position_y = 0.1 * size
                ax4.plot(positions_x, positions_y, c="w", lw=3)
                ax4.scatter(positions_x, positions_y, c="w", marker="o", s=20)
                scaletext = 30
                ax4.text(text_position_x, text_position_y, f'{scaletext}"', c="w", fontsize=20,
                         horizontalalignment='center')
                ax4.axis("off")
                c1 = plt.Circle((size // 2, size // 2), size * 3 * self.pos_err / (fov.to(u.arcsec).value), color='r',
                                fill=False)
                ax4.add_patch(c1)

            except:
                ax3.text(0.5,0.5, "Issues connecting to CDS server")
                ax3.axis("off")
        plt.tight_layout()
        plt.show()


def load_optical_source(cat):
    print(f"Loading {cat}...")
    raw_data = fits.open(os.path.join(path_to_catalogs,f"{cat}.fits"), memmap=True)
    sources_raw = raw_data[1].data
    sources_raw = Table(sources_raw)
    sources_raw = sources_raw[np.argsort(sources_raw[src_names[cat]])]

    indices_for_source = [i for i in range(1, len(sources_raw)) if (sources_raw[src_names[cat]][i] != sources_raw[src_names[cat]][i - 1])]

    #We divide up the catalog in sub-samples corresponding to each source
    if cat!='GALEX':
        timesteps = np.split(np.array(sources_raw[time_names[cat]]), indices_for_source)
        obsids = np.split(np.array(sources_raw['OBSID']), indices_for_source)
    else:
        timesteps = np.split(np.array([54101 for line in sources_raw]), indices_for_source) #Time is set at January 1st 2007 for all GALEX data
        obsids = np.split(np.array([0 for line in sources_raw]), indices_for_source) #OBSID is set to 0
    names = np.split(np.array(sources_raw[src_names[cat]]), indices_for_source)

    band_fluxes = []
    band_flux_errors=[]
    for band_flux_name, band_fluxerr_name in zip(band_flux_names[cat], band_fluxerr_names[cat]):
        band_fluxes.append(sources_raw[band_flux_name])#*2*halfband_width)
        band_flux_errors.append(sources_raw[band_fluxerr_name])#*halfband_width)

    band_fluxes = np.transpose(np.array(band_fluxes))
    band_flux_errors = np.transpose(np.array(band_flux_errors))
    band_fluxes = np.split(band_fluxes, indices_for_source)
    band_flux_errors = np.split(band_flux_errors, indices_for_source)

    dic_sources = {}

    #This loops on all sources, to build the Source objects
    pbar=tqdm(total=len(band_fluxes))
    for (index, time, name, band_flux, band_fluxerr, obsid) in (
            zip(range(len(band_fluxes)), timesteps, names, band_fluxes, band_flux_errors, obsids)):
        source = OpticalSource(cat, name[0].strip(), time, band_flux, band_fluxerr, obsid)
        dic_sources[name[0].strip()] = source
        pbar.update(1)
    pbar.close()
    return dic_sources

def load_master_sources(only_galaxies=False):
    tab_optical_sources = {}
    for opt_cat in optical_catalogs:
        tab_optical_sources[opt_cat] = load_optical_source(opt_cat)

    print(f"Loading Master Sources...")
    raw_data = fits.open(os.path.join(path_to_catalogs,"OpticalMasterSources.fits"), memmap=True)
    sources_raw = raw_data[1].data
    sources_raw = Table(sources_raw)
    if only_galaxies:
        sources_raw=sources_raw[sources_raw['GLADE_IAUNAME']>0]
    dic_master_sources = {}
    pbar=tqdm(total=len(sources_raw))
    for ind,line in enumerate(sources_raw):
        tab_optical_sources_for_this_ms = []
        for cat in optical_catalogs:
            if line[cat+'_IAUNAME']!='':
                name=line[cat+'_IAUNAME'].strip()
                if name in tab_optical_sources[cat].keys():
                    tab_optical_sources_for_this_ms.append(tab_optical_sources[cat][name])
        ms = MasterSource(ind, line["RA_OM_UVOT"], line["DEC_OM_UVOT"], line["OM_UVOT_PosErr"], tab_optical_sources_for_this_ms)
        if line["GLADE_IAUNAME"]>0:
            (ms.glade_distance,ms.glade_stellar_mass) = line["d_L"], line['stellar_mass']
            ms.flux_lum_conv_factor = 4*np.pi*(ms.glade_distance*3.086E+24)**2
        dic_master_sources[ind] = ms
        pbar.update(1)
    pbar.close()
    print("Master sources loaded!")
    return dic_master_sources

if __name__=="__main__":
    dic_master_sources=load_master_sources(only_galaxies=True)


