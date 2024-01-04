# OpticalSkyMiner
A light version of the STONKS package adapted for optical/UV data, meant to allow to do data mining in the optical/UV multi-instrument catalog, and study the selected sources and their properties.

Requirements:
- STILTS (https://www.star.bris.ac.uk/~mbt/stilts/)
- Standard Python packages

"LoadMasterSources.py" will load the relevant multi-instrument sources in custom Python objects. This gives access to flux and band photometry.
"DataMining_OpticalUV_TDEs.py" will use the loaded multi-instrument sources and allow you to study their properties

To function, the catalog data file needs to be unzipped in the same file as the Python scripts.
