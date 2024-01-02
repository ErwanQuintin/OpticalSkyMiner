# XraySkyMiner
A light version of the STONKS package, meant to allow to do data mining in the X-ray multi-instrument catalog, and study the selected sources and their properties

Requirements:
- STILTS (https://www.star.bris.ac.uk/~mbt/stilts/)
- Standard Python packages

"LoadMasterSources.py" will load the relevant multi-instrument sources in custom Python objects. This gives access to flux and band photometry.
"DataMining.py" will use the loaded multi-instrument sources and allow you to study their properties

To function, the catalog data needs to be downloaded from the following [link](https://drive.google.com/file/d/15rhDMWqbiylh0lS5gZ9jS7VzjHe29vBc/view?usp=drive_link)https://drive.google.com/file/d/15rhDMWqbiylh0lS5gZ9jS7VzjHe29vBc/view?usp=drive_link and unzipped in the same file as the Python scripts.
