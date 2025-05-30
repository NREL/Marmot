<div align="center">
  <img src=https://upload.wikimedia.org/wikipedia/commons/3/3b/Marmot-edit1.jpg width="450"><br>
</div>

# Marmot: Energy Analysis and Visualization

[![Documentation](https://img.shields.io/badge/docs-ready-blue.svg)](https://nrel.github.io/Marmot/index.html)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6388281.svg)](https://doi.org/10.5281/zenodo.6388281)
[![License](https://img.shields.io/pypi/l/pandas.svg)](https://github.com/NREL/Marmot/blob/main/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


## What is it?
Marmot is a data formatting and visualization tool for production cost and capacity expansion modelling results. It provides an efficient way to analysis data by combing temporally disaggregated results and allowing the aggregation of different device types and modelling regions.

Marmot currently supports analysis of [**PLEXOS**](https://www.energyexemplar.com/plexos) production costs modelling and [**ReEDS**](https://www.nrel.gov/analysis/reeds/) capacity expansion results.

## Main Features

- Formats modelling results data to a standard format and saves them to a hdf5 file.
- Combines temporally disaggregated results.
- Handles every timestep whether that be 5 minute or yearly interval data.
- Provides approximately 120 pre-built plots which offer vast user customization through an easy to use configuration file and various mapping and input data csvs. 

## Installation
For detailed Installation Instruction see the docs at: https://nrel.github.io/Marmot/get-started/install.html

- To install the latest version 
``` 
git clone --recurse-submodules https://github.com/NREL/Marmot.git
```
(Make sure to include `--recurse-submodules` else h5plexos will not be included correctly)

- Marmot includes a [conda environment for linux users](marmot-linux.yml) and [requirements.txt](requirements.txt) file to ensure all dependencies are available. Users are advised to begin by trying the requirements.txt as current best practice.


## Documentation
The official documentation is hosted on github-pages: https://nrel.github.io/Marmot

## License
[BSD 3](LICENSE)

## Background
Work on ``Marmot`` started at the National Renewable Energy Laboratory [NREL](https://www.nrel.gov/about/) (A national laboratory of the U.S. Department of Energy) in 2019 and
has been under active development since then.

## Tips and tricks

- Some modifications to the structure of the PLEXOS model before running can facilitate analysis in Marmot, such as placing generators in meaningful categories for later mapping in Marmot and making sure any desired properties are enabled in the report.
- When running the Marmot formatter, it is often convenient to process all properties at once by setting all values in the plexos_properties.csv file to TRUE. Marmot will skip properties that are not available in a PLEXOS solution.
- **View formatted contents:** In order to check the contents of an existing processed HDF5 file, type the following in a Python terminal or workspace:

  ```python
  import pandas as pd
  temp=pd.HDFStore("path to formatted hdf5 file")
  temp.keys()
  temp.close()
  ```
