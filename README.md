# Marmot
Marmot is a data formatting and visualization tool for PLEXOS production cost modelling results. It provides an efficient way to view PLEXOS results, while also creating publication ready figures and data tables. 

![Yellow-bellied marmot](https://upload.wikimedia.org/wikipedia/commons/3/3b/Marmot-edit1.jpg)

Click the following to quickly navigate to the main sections of the ReadME:
- [Main Python Scripts](https://github.nrel.gov/PCM/Marmot#main-python-scripts)
- [Initial Setup](https://github.nrel.gov/PCM/Marmot#initial-setup)
- [Marmot Formatter](https://github.nrel.gov/PCM/Marmot#marmot-formatter)
- [Marmot Plotter](https://github.nrel.gov/PCM/Marmot#marmot-plotter)
- [Mapping Files](https://github.nrel.gov/PCM/Marmot#mapping-files)
- [Additional Configuration Settings ](https://github.nrel.gov/PCM/Marmot#Additional-Configuration-Settings)


## Main Python Scripts
Marmot is written in Python 3 and has two main programs:

* [**Marmot Formatter**](https://github.nrel.gov/PCM/Marmot/blob/master/marmot/marmot_h5_formatter.py): Formatting Data using marmot_h5_formatter.py

* [**Marmot Plotter**](https://github.nrel.gov/PCM/Marmot/blob/master/marmot/marmot_plot_main.py): Plotting Figures using marmot_plot_main.py


A high-level explanation of what these files do and suggested settings to change are described in this readme. Code specifics are described in more detail in the code docstrings. 

## Initial Setup

- Marmot requires Python 3 and the following prerequisites to run.
  - hdf5>=1.10.4 *(Install with Conda or Download from HDF website)*
  - numpy
  - pandas
  - PyYAML
  - h5py==2.10.0
  - matplotlib>=3.1.0
  - [h5plexos](https://github.com/NREL/h5plexos)>=0.6 *(See more details [below](https://github.nrel.gov/PCM/Marmot#marmot_h5_formatter))*
  - tables


- These prerequisites can be installed manually with conda or pip, but it is recommended to install all requirements with the provided conda environment or requirements.txt file.
Each are explained [below](https://github.nrel.gov/PCM/Marmot#conda-environment):

#### Conda Environment
- Setting up a new conda environment is the recommended route for running Marmot. Users will need certain user admin/rights to setup conda environments so this may not be possible on all systems.
To ensure you are using all the required python modules, create a new conda environment using the provided [environment yml file](https://github.nrel.gov/PCM/Marmot/blob/master/marmot-env.yml). If you are unsure how to do this, follow [these steps](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).
Once the environment has been created it can be activated by typing `conda activate marmot-env `

#### requirements txt file
- A [requirements.txt](https://github.nrel.gov/PCM/Marmot/blob/master/requirements.txt) file is also included with the repo, this can be used in place of the conda environment file. The txt file contains all the python modules that are required by Marmot to run. To install from the file run the following from any cmd window that is setup with Python (e.g Git Bash, Anaconda Prompt) `pip install -r requirements.txt`. If installing on a machine with restricted user rights adding `--user` to the command may be required. 

After all required prerequisites are installed, you are ready to install and run Marmot. There are two ways to run Marmot, directly or as a module. Most users will probably want to run Marmot directly, however some users may want to import Marmot into their own code. Both are explained below.

### Downloading, installing, and running Marmot Directly 

*The following command will clone the most recent commit to the master branch of Marmot which may not be production ready, to get the most recent stable release see the [Releases](https://github.nrel.gov/PCM/Marmot/releases) section of the repo.*

- First `git clone --recurse-submodules https://github.nrel.gov/PCM/Marmot.git` to any location you like, make sure to include `--recurse-submodules` else h5plexos will not be included correctly.

- The Marmot formatter imports h5plexos. To avoid import and version errors, h5plexos is included as a submodule in Marmot. If you already cloned the project and forgot `--recurse-submodules`, you can combine the git submodule init and git submodule update steps by running `git submodule update --init`.

- Follow the **Marmot Formatter** and **Marmot Plotter** steps below to run Marmot.

### Importing and running Marmot as a Module

- To use Marmot as a Python Module that can be imported, it first needs to be made visible to the Python package directory. This can be done two ways, pip installing Marmot (preferred method) or adding the Marmot directory folder to the system path. 
- To pip install Marmot:
  1. Open a cmd window that is setup with Python and change directory to your desired install location.
  2. Type the following `pip3 install --user -e git+https://github.nrel.gov/PCM/Marmot.git#egg=marmot` This will install Marmot from the current master branch, however this can be changed to a specific commit or tagged release if desired by adding **@comit_id** after Marmot.git and before the **#** symbol. 
For example, the following can be used to install Marmot release v0.7.0 `git+https://github.nrel.gov/PCM/Marmot.git@v0.7.0#egg=marmot`  
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

Before you use the **marmot_h5_formatter** you will need to adjust and set the input settings in the **Marmot_user_defined_inputs.csv** and set which PLEXOS properties to process in the **plexos_properties.csv**. You may also want to edit the Mapping Files described [here](https://github.nrel.gov/PCM/Marmot#mapping-files). These files are located in the repo and are available to be edited once you clone the repo. 

Required and Optional Settings to adjust in the **Marmot_user_defined_inputs.csv** to run the plotting code include:

- `PLEXOS_Solutions_folder` **Required** The folder that contains all h5plexos files that have come from PLEXOS. This folder should contain the Scenario_process_list sub-folders in which the h5plexos files are contained. Here's an example of how that would look:
  
  ![marmot folder structure](https://github.nrel.gov/storage/user/1084/files/bf9d1670-1254-11eb-8e62-c06455591fb2)

- `Marmot_Solutions_folder` **Optional** This is the base directory to create folders and save outputs in. When **marmot_h5_formatter** is run it will create a `Processed_HDF5 folder` here, in which all the proceessed results will be saved with the extension "*_formatted.h5*". This folder can have the same address as PLEXOS_Solutions_folder, having an alternative address allows the user to save outputs in a different location from inputs if desired. if left blank all data and plots will be saved in PLEXOS_Solutions_folder.

- `Scenario_process_list` **Required** This is the list of scenarios to process. The h5plexos hdf5 results files should be saved in folders with these names. The list must contain at least one entry. Using the above example this would be "*Scenario_name_1, Scenario_name_2*"

- `VoLL` **Required** Value of lost load for calculating the Cost of Unserved Energy, default is 10,000 $/MWh

- `Region_Mapping.csv_name` **Optional** The name of the Region_Mapping.csv described in more detail in [Mapping Files](https://github.nrel.gov/PCM/Marmot#mapping-files) below.

- `gen_names.csv_name` **Required** The name of the gen_names.csv described in more detail in [Mapping Files](https://github.nrel.gov/PCM/Marmot#mapping-files) below.

Finally adjust the PLEXOS properties to process in the **plexos_properties.csv**. This csv file determines which PLEXOS properties to pull from the h5plexos results and process. Under the *"collect_data"* column adjust the property to be TRUE or FALSE set whether that particular property will be processed. If a property you would like to process is not in this list, add it as a new line with the same format. 

## Marmot Plotter

**marmot_plot_main.py** is the main plotting script within Marmot. It reads in data that has been processed by the Marmot Formatter, creates plots, and then save the figures and data. These figures can be saved to any format specified by the user including both raster and vector formats. An associated csv data file will also be created with each figure allowing users to view the plot data in a tabular format. 

The main plotting script works by calling on individual plotting modules. These can be viewed within the repo [plottingmodules](https://github.nrel.gov/PCM/Marmot/tree/master/marmot/plottingmodules) folder and have descriptive names such as **total_generation.py**, **generation_stack.py**, **curtaiment.py** etc. 

As with the Marmot Formatter, users will need to adjust the input settings in the **Marmot_user_defined_inputs.csv** and set which plots to create in **Marmot_plot_select.csv**. 
**Marmot_plot_select.csv** is a csv file which determines which figures to plot. This file is in the repo. Under the *"Plot Graph"* (column *D*) adjust the property to be TRUE or FALSE to decide whether to plot the figure. Column *C* allows the user to adjust certain properties within the plot (examples given). Columns *D* and *E* adjust the range of days to plot either side of the specified property in *D*. Column *F* adjusts the time zone to plot on the figure. Column *G* and *H* allow adjustment of the date range of the data to be plotted, if left blank data from the entire date range is plotted. The user is free to change the values within the *"Figure Output Name"* column (column *A*) to whatever they like as this will be the name given to the output files. The rightmost columns (columns *I* and *J*) should not be edited as these inform Marmot which plotting module and method to use to create the corresponding figure. 

Required and Optional Settings to adjust in the **Marmot_user_defined_inputs.csv** to run the plotting code include:

- `PLEXOS_Solutions_folder` **Required** Same as described [above](https://github.nrel.gov/PCM/Marmot#plexos_h5_results_formatter)

- `Marmot_Solutions_folder` **Optional** Same as described [above](https://github.nrel.gov/PCM/Marmot#plexos_h5_results_formatter)

- `Scenarios` **Required** This is the name of the scenario(s) to plot. The order of the scenarios will determine the order of the scenarios in the plot. Resulting outputs will be saved in the "*Figures_Output*" folder contained with the `Marmot_Solutions_folder`. "*Figures_Output*" is created automatically when **marmot_plot_main.py** is run

- `Scenario_Diff_plot` **Optional** This is a list which can contain max two entries. This list is used to create plots using the difference of the values between two scenarios. The second scenario in the list is subtracted from the first. If you are not creating difference plots this list can remain empty.

- `AGG_BY` **Required** A string which tells Marmot which region type to aggregate by when creating plots. The default options are *”regions”* and *“zones”*. Other options can be added based on how the user sets up **Region_mapping.csv** described [above](https://github.nrel.gov/PCM/Marmot#mapping-files)

- `zone_region_sublist` **Optional** List of *"regions/zones”* to plot if results are not required for all regions. The list of *"regions/zones”* should be contained within the `AGG_BY` aggregation. This is an optional field and can be left empty if not needed.

- `Facet_ylabels` & `Facet_xlabels` **Optional** If you wish to create a Facet plot, these labels will be applied to the axis. The number of entries given to each label will determine the dimensions of the Facet plot.  For example, if you have 6 scenarios your Facet Grid dimensions in x,y coordinates may be [2,3], [3,2] or [1,6] etc. This is an optional field and can be left empty if not needed. Facet plots can still be created without labels if desired, the facet layout will default to max 3 plots wide, with no maximum to the length.

- `Tick_labels` **Optional** List of custom tick labels, allows adjustment of scenario names on certain plots. Not setup for every plot. 

## Mapping Files
Marmot gives the user the ability to map in extra regions to your data, rename generators, adjust generator technology colors using a set of csv files and group different technologies together. Adjusting these values to your specific PLEXOS database is not required for Marmot to run but recommended for best results.
Examples of these files can be found within in the repo in the [mapping_folder](https://github.nrel.gov/PCM/Marmot/tree/master/mapping_folder).
These csv files are:

- **gen_names.csv** This file allows you to change the name of the PLEXOS generator technology categories to be consistent. For example, change all the possible gas cc generator names to just be called "Gas-CC". The csv file has two columns *"Original"*, which contains the name of all the PLEXOS generator categories and *"New"*, which is the new name you want to give the categories. The *"Original"* column needs to include every generation category present in your PLEXOS database. The Marmot Formatter will warn the user if they are missing a category. 

- **ordered_gen** Ordered list of generators which determines how they appear in a stack plot; generator names should equal those in the gen_names.csv *"New"* column

- **Region_mapping.csv** This file allows you to group PLEXOS regions together to create aggregated regions. The first column in the file should always be called *"region"* and should contain the name of all the regions in your PLEXOS database. The names given to all other columns is up to you. In the example given in the repo, we aggregated NARIS regions to the country and interconnect level. The use of Region_mapping is entirely optional, if not needed its entry in the **Marmot_user_defined_inputs.csv** can be left blank.

- **colour_dictionary.csv** This allows the user to adjust the color used to plot generation technology types e.g Gas-CC, Wind, PV etc. The names in the generator column should equal those in the gen_names.csv *"New"* column.

 - **pv_gen_cat.csv** , **re_gen_cat.csv** , **vre_gen_cat.csv** & **thermal_gen_cat.csv** - These files are used for grouping specified generator categories for certain actions like calculating curtailment (Marmot's *generator_Curtailment* property) and selecting specific technologies to plot in certain figures. The generators included in each file should have the same names as the "New" name column in the gen_names.csv. Below is an example of technologies included in the **vre_gen_cat.csv** (variable renewable). These technologies are used to determine the *generator_Curtailment* property values. 

  | VRE_Gen_Categories |
  | ------------------ |
  | Wind               | 
  | PV                 | 
  | dPV                | 
  | Offshore Wind      | 

### View formatted contents
In order to check the contents of an existing processed HDF5 folder, use the following:

```python 
temp=pd.HDFStore("path to formatted hdf5 file")
temp.keys()
temp.close()
```

## Additional Configuration Settings 

**The adjustment of the following settings are all optional** and are setup with default values. Users who would like to further customize how Marmot works will be able to easily change these values. 

These additional configuration settings live within the [Marmot/marmot/config](https://github.nrel.gov/PCM/Marmot/tree/master/marmot/config) folder of the repo. A **config.yml** file will be created the first time the **marmot_h5_formatter** or **marmot_plot_main** are run. After the yml file has been created users will be able to change any settings they like within the file, this file is never overwritten by Marmot once created. To revert to default settings, delete the config.yml file and Marmot will create it again once run. 
The config.yml file is not to be confused with the marmot_logging_config.yml file which is used to set the logging defaults used by Marmot and also exhists in the same folder.

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
  
  *Controls certain plot data settings, curtailment_property allows the property used to represent Curtailment in figures to be changed (default Marmot's calculated Curtailment property). include_total_pumped_load_line specifies whether to include the line representing pumped load in total generation bar plots. include_timeseries_pumped_load_line specifies whether to include the line representing pumped load in timeseries generation plots*

- **figure_file_format:** svg

  *Adjust the format figures are saved in, the default is **svg** a vector-based image. This field accepts any format that is compatible with matplotlib*  
  
- **shift_leapday:** false

  *Handles auto shifting of leap day if model contains it, default is false*  

- **skip_existing_properties:** true

  *Toggles whether existing properties are skipped or overwritten if they already contained in the processed_h5 file, the default is to skip*
  
- **auto_convert_units:** true

  *If True automatically converts Energy and Capacity units so that no number exceeds 1000, all base units are in MW, units can be converted to GW, TW and kW*
  
- **plot_title_as_region:** true

  *If True a title will be added to the figure which will be the region/zone name*
  
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







