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

<<<<<<< HEAD
- To use Marmot as a Python Module that can be imported, it first needs to be made visible to the Python package directory. This can be done two ways, either by pip installing Marmot (preferred method) or by adding the Marmot directory folder to the system path.
- To pip install Marmot:
  1. Open a cmd window that is setup with Python and change directory to your desired install location.
  2. Type the following `pip3 install --user -e git+https://github.com/NREL/Marmot.git#egg=marmot` This will install Marmot from the current master branch, however this can be changed to a specific commit or tagged release if desired by adding **@comit_id** after Marmot.git and before the **#** symbol.
For example, the following can be used to install Marmot release v0.7.0 `git+https://github.com/NREL/Marmot.git@v0.7.0#egg=marmot`  
  3. If no error messages appeared Marmot has been installed correctly. To import the formatter or plotter, use the following import commands:
=======
## Installation
For detailed Installation Instruction see the docs at: https://nrel.github.io/Marmot/get-started/install.html
>>>>>>> e113d4197d71da9c1b06b4d868a91bb2c76e7270

- To install the latest version 
``` 
git clone --recurse-submodules https://github.com/NREL/Marmot.git --branch v0.9.0
```
(Make sure to include `--recurse-submodules` else h5plexos will not be included correctly)

- Marmot includes a [conda environment](marmot-env10.yml) and [requirements.txt](requirements.txt) file to ensure all dependencies are available.


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
