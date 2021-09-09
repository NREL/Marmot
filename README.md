# Marmot
Marmot is a data formatting and visualization tool for PLEXOS production cost modelling results. It provides an efficient way to view PLEXOS results, while also creating publication ready figures and data tables.

![Yellow-bellied marmot](https://upload.wikimedia.org/wikipedia/commons/3/3b/Marmot-edit1.jpg)

Click the following to quickly navigate to the main sections of the ReadME:
- [Main Python Scripts](https://github.com/NREL/Marmot#main-python-scripts)
- [Initial Setup](https://github.com/NREL/Marmot#initial-setup)
- [Marmot Formatter](https://github.com/NREL/Marmot#marmot-formatter)
- [Marmot Plotter](https://github.com/NREL/Marmot#marmot-plotter)
- [Mapping Files](https://github.com/NREL/Marmot#mapping-files)
- [Additional Configuration Settings ](https://github.com/NREL/Marmot#Additional-Configuration-Settings)
- [Tips and tricks](https://github.com/NREL/Marmot#Tips-and-tricks)


## Main Python Scripts
Marmot is written in Python 3 and has two main programs:

* [**Marmot Formatter**](https://github.com/NREL/Marmot/blob/master/marmot/marmot_h5_formatter.py): Formatting Data using marmot_h5_formatter.py

* [**Marmot Plotter**](https://github.com/NREL/Marmot/blob/master/marmot/marmot_plot_main.py): Plotting Figures using marmot_plot_main.py


A high-level explanation of what these files do and suggested settings to change are described in this readme. Code specifics are described in more detail in the code docstrings.

## Initial Setup

- Marmot requires Python 3 and the following prerequisites to run.
  - hdf5>=1.10.4 *(Install with Conda or Download from HDF website)*
  - numpy
  - pandas
  - PyYAML
  - h5py==2.10.0
  - matplotlib>=3.1.0
  - [h5plexos](https://github.com/NREL/h5plexos)>=0.6 *(See more details [below](https://github.com/NREL/Marmot#marmot_h5_formatter))*
  - tables


- These prerequisites can be installed manually with conda or pip, but it is recommended to install all requirements with the provided conda environment or requirements.txt file.
Each are explained [below](https://github.com/NREL/Marmot#conda-environment):

#### Conda Environment
- Setting up a new conda environment is the recommended route for running Marmot. Users will need certain user admin/rights to setup conda environments so this may not be possible on all systems.
To ensure you are using all the required python modules, create a new conda environment using the provided [environment yml file](https://github.com/NREL/Marmot/blob/master/marmot-env.yml). If you are unsure how to do this, follow [these steps](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).
Once the environment has been created it can be activated by typing `conda activate marmot-env `

#### requirements txt file
- A [requirements.txt](https://github.com/NREL/Marmot/blob/master/requirements.txt) file is also included with the repo, this can be used in place of the conda environment file. The txt file contains all the python modules that are required by Marmot to run. To install from the file run the following from any cmd window that is setup with Python (e.g Git Bash, Anaconda Prompt) `pip install -r requirements.txt`. If installing on a machine with restricted user rights adding `--user` to the command may be required.

After all required prerequisites are installed, you are ready to install and run Marmot. There are **Two ways** to run Marmot, **Running Directly (A)** or **Importing as a Module (B)**. Most users will probably want to run Marmot directly, however some users may want to import Marmot into their own code. Both are explained below.

### A: Downloading, installing, and running Marmot Directly

*The following command will clone the most recent commit to the master branch of Marmot which may not be production ready, to get the most recent stable release see the [Releases](https://github.com/NREL/Marmot/releases) section of the repo.*

- First `git clone --recurse-submodules https://git@github.com:NREL/Marmot.git` to any location you like, make sure to include `--recurse-submodules` else h5plexos will not be included correctly.

- The Marmot formatter imports h5plexos. To avoid import and version errors, h5plexos is included as a submodule in Marmot. If you already cloned the project and forgot `--recurse-submodules`, you can combine the git submodule init and git submodule update steps by running `git submodule update --init`.

- Follow the **Marmot Formatter** and **Marmot Plotter** steps below to run Marmot.

### B: Importing and running Marmot as a Module

- To use Marmot as a Python Module that can be imported, it first needs to be made visible to the Python package directory. This can be done two ways, either by pip installing Marmot (preferred method) or by adding the Marmot directory folder to the system path.
- To pip install Marmot:
  1. Open a cmd window that is setup with Python and change directory to your desired install location.
  2. Type the following `pip3 install --user -e git+https://github.com:NREL/Marmot.git#egg=marmot` This will install Marmot from the current master branch, however this can be changed to a specific commit or tagged release if desired by adding **@comit_id** after Marmot.git and before the **#** symbol.
For example, the following can be used to install Marmot release v0.7.0 `git+https://github.com:NREL/Marmot.git@v0.7.0#egg=marmot`  
  3. If no error messages appeared Marmot has been installed correctly. To import the formatter or plotter, use the following import commands:

 ```python
 from marmot.marmot_h5_formatter import MarmotFormat
 ```
 ```python
from marmot.marmot_plot_main import MarmotPlot
```
- When importing Marmot directly, the **Marmot_user_defined_inputs.csv** described below is not used. However, several other input files are still required. For more details see internal code docstrings within the '__init__' methods.


## Marmot Formatter
The **marmot_h5_formatter** reads in PLEXOS hdf5 files created with the h5plexos library (the repo for which can be found [here](https://github.com/NREL/h5plexos)) and processes the output results to ready them for plotting. Once the outputs have been processed, they are saved to an intermediary hdf5 file which can then be read into the Marmot plotting code. From the h5plexos ReadMe: "This package provides a Python interface for reading HDF5 files with H5PLEXOS v0.5 and v0.6 formatting. To create v0.5 files, use a version of this package in the 0.5 series. To create v0.6 files, use H5PLEXOS.jl."

Before you use the **marmot_h5_formatter** you will need to adjust and set the input settings in the **Marmot_user_defined_inputs.csv** and select PLEXOS properties to process in the **plexos_properties.csv**. You may also want to edit the Mapping Files described [here](https://github.com/NREL/Marmot#mapping-files). These files can be edited once you clone the repo.

### 1. Adjusting User Defined Inputs CSV

Required and Optional Settings to adjust in the **Marmot_user_defined_inputs.csv** to run the formatter include:

- `PLEXOS_Solutions_folder` **Required** The directory that contains all h5plexos files that have come from PLEXOS. This directory should include a sub-folder for each scenario in the `Scenarios`, each of which holds the individual h5plexos solution files. Here's an example of how that would look:

  ![marmot folder structure](https://user-images.githubusercontent.com/43964549/132605149-fd088a10-8c4a-49f1-b8b7-d3d31e3f5a30.png)

   Multiple h5plexos files within a single scenario sub-folder will be combined to form a single timeseries, with any overlapping periods trimmed.

- `Marmot_Solutions_folder` **Optional** This is the base directory for saving outputs. When **marmot_h5_formatter** is run it will create a `Processed_HDF5 folder` here, and will save all the processed results with the extension "*_formatted.h5*". Including a folder path here allows the user to save outputs in a different location from inputs if desired. If left blank all data and plots will be saved in `PLEXOS_Solutions_folder`.

- `Scenario_process_list` **Required** This is the list of scenarios to process. The h5plexos hdf5 results files should be saved in folders with these names. The list must contain at least one entry. Using the above example, this list would be "*Scenario_name_1, Scenario_name_2*"

- `VoLL` **Required** Value of lost load for calculating the Cost of Unserved Energy, default is 10,000 $/MWh

- `Region_Mapping.csv_name` **Optional** The name of the Region_Mapping.csv described in more detail in [Mapping Files](https://github.com/NREL/Marmot#mapping-files) below.

- `gen_names.csv_name` **Required** The name of the gen_names.csv described in more detail in [Mapping Files](https://github.com/NREL/Marmot#mapping-files) below.

### 2. Selecting Properties to Process
The **plexos_properties.csv** file determines which PLEXOS properties to pull from the h5plexos results. Under the *"collect_data"* column, adjust the property to be TRUE or FALSE to set whether that particular property will be processed. If a property you would like to process is not in this list, add it as a new line with the same format.

### 3. Running the Formatter
To run the Marmot Formatter open a terminal that is setup with Python, go to the Marmot folder containg the marmot_h5_formatter.py file (see example image, **cd C:\Users\DLEVIE\Documents\Marmot\marmot**), and run the following command:
`python marmot_h5_formatter.py`

![Run Formatter](https://user-images.githubusercontent.com/43964549/132605182-1d2f3d48-355e-4877-80b6-ea5ec398faa5.png)
  

## Marmot Plotter

**marmot_plot_main.py** is the main plotting script within Marmot. It reads in data that has been processed by the Marmot Formatter, creates plots, and then save the figures and data. These figures can be saved to any format specified by the user including both raster and vector formats. An associated .csv data file will also be created with each figure allowing users to view the plot data in a tabular format.

The main plotting script works by calling on individual plotting modules. These can be viewed within the repo [plottingmodules](https://github.com/NREL/Marmot/tree/master/marmot/plottingmodules) folder and have descriptive names such as **total_generation.py**, **generation_stack.py**, **curtaiment.py** etc.

As with the Marmot Formatter, users will need to adjust the input settings in the **Marmot_user_defined_inputs.csv** and set which plots to create in **Marmot_plot_select.csv**.

### 1. Adjusting User Defined Inputs CSV

Required and Optional Settings to adjust in the **Marmot_user_defined_inputs.csv** to run the plotting code include:

- `PLEXOS_Solutions_folder` **Required** Same as described [above](https://github.com/NREL/Marmot#marmot-formatter)

- `Marmot_Solutions_folder` **Optional** Same as described [above](https://github.com/NREL/Marmot#marmot-formatter)

- `Scenarios` **Required** This is the name of the scenario(s) to plot. The order of the scenarios will determine the order of the scenarios in the plot. Resulting outputs will be saved in the "*Figures_Output*" folder contained with the `Marmot_Solutions_folder`. "*Figures_Output*" is created automatically when **marmot_plot_main.py** is run. If you are making difference plots, the second scenario in this list will be subtracted from the first.

- `AGG_BY` **Required** A string that determines the region type by which to aggregate when creating plots. The default options are *”regions”* and *“zones”*. Other options can be added based on how the user sets up **Region_mapping.csv**, described [below](https://github.com/NREL/Marmot#mapping-files)

- `zone_region_sublist` **Optional** List of *"regions/zones”* to plot if results are not required for all regions. The list of *"regions/zones”* should be contained within the `AGG_BY` aggregation. This is an optional field and can be left empty if not needed.

- `Facet_ylabels` & `Facet_xlabels` **Optional** If you wish to create a Facet plot, these labels will be applied to the axis. The number of entries given to each label will determine the dimensions of the Facet plot.  For example, if you have 6 scenarios, your facet grid dimensions in x,y coordinates may be [2,3], [3,2] or [1,6] etc. This is an optional field and can be left empty if not needed. Facet plots can still be created without labels if desired. The facet layout will default to max 3 plots wide, with no maximum to the length.

- `Tick_labels` **Optional** List of custom tick labels, which allows adjustment of scenario names on all bar plots. Does not apply to difference plots or plots from the `capacity_factor` module.

### 2. Selecting which Figures to Plot
**Marmot_plot_select.csv** determines which figures to plot. Adjust the *"Plot Graph"* column (column *B*) to be TRUE or FALSE to decide whether to plot the corresponding. The *"Figure Output Name"* column (column *A*) controls the output name of the plots and data. The user can name these output files anything they like. Column *C* allows the user to adjust certain properties within the plot (examples given). Columns *D* and *E* adjust the range of days to plot either side of the specified property in *D*. Column *F* adjusts the time zone to plot on the figure. Column *G* and *H* allow adjustment of the date range of the data to be plotted; if left blank data from the entire date range is plotted. The rightmost columns (columns *I* and *J*) should not be edited as these inform Marmot which plotting module and method to use to create the corresponding figure. Advanced users writing new plots for Marmot can add new lines to this .csv, as long as they ensure columns *I* and *J* match the appropriate new plotting module and method, respectively.


### 3. Running the Plotter
To run the Marmot Plotter open a terminal that is setup with Python, go to the marmot folder containg the marmot_plot_main.py file (see example image **cd C:\Users\DLEVIE\Documents\Marmot\marmot**), and run the following command:
`python marmot_plot_main.py`

![Run Plotter](https://user-images.githubusercontent.com/43964549/132605191-b1c5aa81-6f1d-42a5-ac7a-c0ac47b745d9.png)

## Mapping Files
Marmot gives the user the ability to map extra regions to your data, rename generators, adjust generator technology colors, and group different technologies together using a set of csv files. Adjusting these values to your specific PLEXOS database is not required for Marmot to run but is recommended for the best results.
Examples of these files can be found within in the repo in the [mapping_folder](https://github.com/NREL/Marmot/tree/master/mapping_folder).

These csv files are:

- **gen_names.csv** This file allows you to change the name of the PLEXOS generator technology categories to be consistent. For example, change all the possible gas cc generator names to just be called "Gas-CC". The csv file has two columns *"Original"*, which contains the name of all the PLEXOS generator categories and *"New"*, which is the new name you want to give the categories. The *"Original"* column needs to include every generation category present in your PLEXOS database. The Marmot Formatter will warn the user if they are missing a category.

- **ordered_gen** Ordered list of generators which determines how they appear in a stack plot; generator names should equal those in the gen_names.csv *"New"* column. 

  ***New in v0.8.0*** - A reduced list of generators can be included to limit the technologies that are plotted e.g (only plot Gas or Hydro). Marmot will print a warning message if there are generators missing from the ordered list that are in the gen_names.csv *"New"* column. This message can be ignored if you are you are down-selecting technology types on purpose. 

- **Region_mapping.csv** This file allows you to group PLEXOS regions together to create aggregated regions. The first column in the file should always be called *"region"* and should contain the names of all the regions in your PLEXOS database. The names given to all other columns is up to you. In the example given in the repo, we aggregated regions from NREL's recent [North American Renewable Integration Study](https://www.nrel.gov/analysis/naris.html) to the country and interconnect level. The use of Region_mapping is entirely optional; if not needed its entry in the **Marmot_user_defined_inputs.csv** can be left blank.

- **colour_dictionary.csv** This allows the user to adjust the color used to plot generation technology types e.g Gas-CC, Wind, PV etc. The names in the generator column should equal those in the gen_names.csv *"New"* column.

 - **pv_gen_cat.csv** , **re_gen_cat.csv** , **vre_gen_cat.csv** & **thermal_gen_cat.csv** - These files are used for grouping specified generator categories for certain actions like calculating curtailment (Marmot's *generator_Curtailment* property) and selecting specific technologies to plot in certain figures. The generators included in each file should have the same names as the "New" name column in the gen_names.csv. Below is an example of technologies included in the **vre_gen_cat.csv** (variable renewable energy). These technologies are used to determine the *generator_Curtailment* property values.

  | VRE_Gen_Categories |
  | ------------------ |
  | Wind               |
  | PV                 |
  | dPV                |
  | Offshore Wind      |
  

## Additional Configuration Settings

**The adjustment of the following settings are all optional** and are setup with default values. Users who would like to further customize how Marmot works can change these values in a standard text editor.

These additional configuration settings live within the [Marmot/marmot/config](https://github.com/NREL/Marmot/tree/master/marmot/config) folder of the repo. A **config.yml** file will be created the first time the **marmot_h5_formatter** or **marmot_plot_main** is run. After the yml file has been created users will be able to change any settings they like within the file. Once created, this file is never overwritten by Marmot. To revert to default settings, delete the config.yml file and Marmot will create it again at runtime.
The config.yml file is not to be confused with the marmot_logging_config.yml file which is used to set the logging defaults used by Marmot and also exists in the same folder.

The **config.yml** settings and their defaults are as follows:

- **font_settings:**
  - xtick_size: 12
  - ytick_size: 12
  - axes_label_size: 16
  - legend_size: 12
  - title_size: 16
  - font_family: serif

  *Settings to adjust font sizes, family, and tick size within figures*

- **text_position:**
  - title_height: 40

  *Adjust the position of text relative to the edge of the plot (matplotlib points)*

- **figure_size:**
  - xdimension: 6
  - ydimension: 4

  *Adjust the x and y-axes dimensions of the output figure*  

- **axes_options:**
  - x_axes_maxticks: 8
  - x_axes_minticks: 4
  - y_axes_decimalpt: 1

  *Allows adjustment of the minimum and maximum tick marks on datetime x-axes and the number of decimal points to include on the y-axes*

- **axes_label_options:**
  - rotate_x_labels: true
  - rotate_at_num_labels: 7
  - rotation_angle: 45

  *Controls whether x-axes labels are rotated from their default horizonal (0 degrees), and at what number of labels to begin rotating to the specified rotation_angle. By default, labels will begin rotating at 7 labels to an angle of 45 degrees from 0 degrees*

- **plot_data:**
  - curtailment_property: Curtailment
  - include_total_pumped_load_line: false
  - include_timeseries_pumped_load_line: true

  *Controls certain plot data settings. `curtailment_property` source of Curtailment data. The code defaults to Marmot's calculated Curtailment property. `include_total_pumped_load_line` specifies whether to include the line representing pumped load in total generation bar plots. `include_timeseries_pumped_load_line` specifies whether to include the line representing pumped load in timeseries generation plots*

- **figure_file_format:** svg

  *Adjust the plot image format. The default is **svg**, a vector-based image. This field accepts any format that is compatible with matplotlib*  

- **shift_leapday:** false

  *Handles auto shifting of leap day, if required by your model. The default is false*  

- **skip_existing_properties:** true

  *Toggles whether existing properties are skipped or overwritten if they already contained in a previous processed_h5 file, the default is to skip*

- **auto_convert_units:** true

  *If True automatically converts Energy and Capacity units so that no number exceeds 1000. All base units are in MW, and units can be converted to GW, TW and kW*

- **plot_title_as_region:** true

  *If True a the region/zone name will be added as a title to the figure*

- **user_defined_inputs_file:** Marmot_user_defined_inputs.csv

  *Change the default Marmot_user_defined_inputs file, file must be created first*  

- **plot_select_file:** Marmot_plot_select.csv

  *Change the default Marmot_plot_select.csv file, file must be created first*  

- **plexos_properties_file:** plexos_properties.csv

  *Change the default plexos_properties_file.csv file, file must be created first*  

- **category_file_names:**
  - pv_gen_cat: pv_gen_cat.csv
  - re_gen_cat: re_gen_cat.csv
  - thermal_gen_cat: thermal_gen_cat.csv
  - vre_gen_cat: vre_gen_cat.csv

  *Change the default category files that live within the Mapping Folder, files must be created first*  

- **color_dictionary_file:** colour_dictionary.csv

  *Change the default color dictionary file that lives within the Mapping Folder, file must be created first*  

- **ordered_gen_file:** ordered_gen.csv

  *Change the default ordered_gen file that lives within the Mapping Folder, file must be created first*  

## Tips and tricks

- Some modifications to the structure of the PLEXOS model before running can facilitate analysis in Marmot, such as placing generators in meaningful categories for later mapping in Marmot and making sure any desired properties are enabled in the report.
- In most cases Marmot assumes standard units for output properties; check the plexos_properties.csv file to verify that units are as you expect. For some properties (such as CO2 emissions), you can set the input units for the PLEXOS results. Marmot will query and autoconvert these units as necessary.   
- When running the Marmot formatter, it is often convenient to process all properties at once by setting all values in the plexos_properties.csv file to TRUE. Marmot will skip properties that are not available in a PLEXOS solution.
- **View formatted contents:** In order to check the contents of an existing processed HDF5 file, type the following in a Python terminal or workspace:

  ```python
  import pandas as pd
  temp=pd.HDFStore("path to formatted hdf5 file")
  temp.keys()
  temp.close()
  ```
