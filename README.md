# Marmot
Marmot is a set of python scripts to process h5plexos PLEXOS results plot the outputs 
![Yellow-bellied marmot](https://upload.wikimedia.org/wikipedia/commons/3/3b/Marmot-edit1.jpg)

## Main Python Scripts
Marmot consists of two main .py files:
* [**PLEXOS_H5_results_formatter.py**](https://github.nrel.gov/PCM/Marmot#marmot_plot_main)
* [**Marmot_plot_main.py**](https://github.nrel.gov/PCM/Marmot#marmot_plot_main)

A high level explanation of what these files do and suggested settings to change are described in this readme. Code specifics are decribed in more detail in the code comments. 

## PLEXOS_H5_results_formatter
The **PLEXOS_H5_results_formatter** reads in PLEXOS hdf5 files created with the h5plexos library (the repo for which can be found [here](https://github.com/NREL/h5plexos)) and processes the output results to ready them for plotting. Once the outputs have been processed they are saved to a intermediary hdf5 file which can then be read into the Marmot plotting code.

Before you use the **PLEXOS_H5_results_formatter** you will need to adjust the input settings and mapping files which are all defined at the top of the code under *Import Python *User Defined Names, Directories and Settings*, these include:

- `Marmot_DIR` This is the directory of the cloned Marmot repo and loaction of the PLEXOS_H5_results_formatter.py file. Update this when you clone the repo

- `Plexos_Properties` This is a csv file which determines which plexos properties to pull from the h5plexos results and process. This file is in the repo. Under the *"collect_data"* column adjust the property to be TRUE or FALSE to change if the data is processed. If your property is not here, add it as a new line with the same format. See the table at the end of this README file to see which properties are necessary for which Marmot plots.

- `Scenario_list` This is the list of scenarios to process. The h5plexos hdf5 results files should be saved in folders with these names. The list must contain at least one entry. 

- `HDF5_folder_in` The folder that contains all h5plexos files that have come from PLEXOS. This folder should contain the Scenario_list sub-folders in which the h5plexos files are contained. Here's an example of how that would look:
  - HDF5_folder_in
    - Scenario_name_1
      - results_m1.h5
      - results_m2.h5
      - .....
    - Scenario_name_2
      - results.h5

- `Solutions_folder` This is the base directory to create folders and save outputs in (Default is Marmot_DIR but you can change to wherever you like). When **PLEXOS_H5_results_formatter** is run it will create a *PLEXOS_Scenarios* folder here, in which all the proceessed scenarios and their results will be saved. 

- `Mapping_folder` This is the folder which contains mapping csv files to rename generators to whatever the user desires and also organise regions and reserves. Each of these files will be explained in more detail in the **Mapping Files** section bellow. 

- `overlap` This is the number of hours overlapped between two adjacent models, default is 0

- `VoLL` Value of lost load for calculating the Cost of Unserved Energy, default is 10,000 $/MWh


  ### Mapping Files
Marmot uses csv files to map in extra regions to your data and rename generators and reserve properties. Examples of these files can be found within in the repo in the [mapping_folder](https://github.nrel.gov/PCM/Marmot/tree/master/mapping_folder). The examples are setup to work with the NARIS PLEXOS databses so make sure to adjust these csv files if you are not running with that underlying database. These csv files are:

- **gen_names.csv** This file allows you to change the name of the PLEXOS generator technology categories to be consistant. For example change all the possible gas cc generator names to just be called "Gas-CC". The csv file has two columns *"Original"*, which contains the name of all the PLEXOS generator categories and *"New"*, which is the new name you want to give the categories. 

- **Region_mapping.csv** This file allows you to group PLEXOS regions together to create aggregated regions. The first column in the file should always be called *"region"* and should contain the name of all the regions in your PLEXOS databse. The names given to all other columns is up to you. In the example given in the repo, we aggreagted NARIS regions to the zonal, country and interconnect level. Currently Marmot is setup to have PLEXOS regions to be the smallest of all areas, we will work to give the option to have this as zones in coming updates.

- **reserve_region_type.csv** This file allows you to adjust the Reserve Region names and reserve types. Check the file for an example.

## Marmot_plot_main

**Marmot_plot_main.py** is the main plotting script within Marmot which calls on supporting files to read in data, create the plot, and then return the plot and data to **Marmot_plot_main.py**. The supporting files can be viewed within the repo and have descriptive names such as **total_generation.py**, **generation_stack.py**, **curtaiment.py** etc. 

Most users will only need to make adjustments to **Marmot_plot_main.py**

As with the processing script users will need to adjust the input settings and mapping files which are all defined at the top of the code under *Import Python *User Defined Names, Directories and Settings*. 

- `Marmot_DIR` Same as described [above](https://github.nrel.gov/PCM/Marmot#plexos_h5_results_formatter)

- `Marmot_plot_select` This is a csv file which determines which figures to plot. This file is in the repo. Under the *"Plot Graph"* column adjust the property to be TRUE or FALSE to decide whether to plot the figure. Column *D* allows the user to adjust certain properties within the plot (examples given). Columns *E* and *F* adjust the range of days to plot either side of the specified property in *D*. Column *G* adjusts the timezone to plot on the figure. The list of figures to plot is currently limited by what code has been written for.  

- `Scenario_name` This is the name of the scenario to plot and is also the folder where plots are saved. This folder will be located in the `Solutions_folder` decribed [above](https://github.nrel.gov/PCM/Marmot#plexos_h5_results_formatter)

- `Solutions_folder` Same as described [above](https://github.nrel.gov/PCM/Marmot#plexos_h5_results_formatter)

- `Multi_Scenario` This is a list of scenarios to plot on the same figure, allowing comparisons to be made between them. The order of the scenarios will determine the order of the scenarios in the plot. This list should have at least one entry and be equal to `Scenario_name`

- `Scenario_Diff` This is a list which can contain max two entries. This list is used to create plots using the difference of the values between two scenarios. The second scenario in the list is subtracted from the first. If you are not creating difference plots this list can remain empty.

- `Mapping_folder` Same as described [above](https://github.nrel.gov/PCM/Marmot#plexos_h5_results_formatter)

- `AGG_BY` A string which tells Marmot which region type to aggregate by when creating plots. The options you have here will be based on how you setup the column names in **Region_mapping.csv** described [above](https://github.nrel.gov/PCM/Marmot#plexos_h5_results_formatter)

- `ylabels` & `xlabels` - If you wish to create a Facet plot, these labels will be applied to the axis. The amount of entries given to each label will determine the dimensions of the Facet plot. This should be equal to the number of scenarios you are plotting. For example if you have 6 sceanrios your Facet Grid dimensions may be 23, 32 or 16. For the Facet plottting to work these lists must at least contain empty strings e.g ("") 

 ### Other Settings (Optional Changes)
 
 The following are found within **Marmot_plot_main.py** and are optional to change.
 
 - `ordered_gen` - Ordered list of generators which determines how they appear in a stack plot.
 
 - `pv_gen_cat` , `re_gen_cat` & `vre_gen_cat`- Generators which belong to specified category, used for certain figures.
 
 - `PLEXOS_color_dict` - Dictionary which holds the colors to apply to each generator type within a plot.
 
 - `color_list` - List of colors to apply to non generator specific line plot and bar plot figures

##Required PLEXOS properties

To use the following table, find the rows that correspond to your desired plots. Make sure to include the corresponding properties in your PLEXOS solutions files. In addition, turn these properties to *TRUE* in **Plexos_Properties.csv**. `interval` corresponds to time series data with observations at every timestemp, whereas `year/summary` properties report a single value for every object.

|                          |Generator -- Generation|Generator -- Available Capacity|Generator -- Installed Capacity|Generator -- Pump Load|Generator -- Total Generation Cost|Generator -- Pool Revenue|Generator -- Reserves Revenue|Generator -- Hours at Minimum|Reserve generators -- Provision|Region -- Load|Region -- Unserved Energy|Region -- Cost Unserved Energy|
|--------------------------------|-----------------------|-------------------------------|-------------------------------|----------------------|----------------------------------|-------------------------|-----------------------------|-----------------------------|-------------------------------|--------------|-------------------------|------------------------------|
|Average Output When Committed   |interval               |                               |year/summary                   |                      |                                  |                         |                             |                             |                               |              |                         |                              |
|Capacity Factor                 |interval               |                               |year/summary                   |                      |                                  |                         |                             |                             |                               |              |                         |                              |
|Capacity Started                |interval               |                               |year/summary                   |                      |                                  |                         |                             |                             |                               |              |                         |                              |
|Curtailment Duration Curve      |interval               |interval                       |                               |                      |                                  |                         |                             |                             |                               |              |                         |                              |
|Curtailment vs. Penetration     |interval               |interval                       |                               |                      |year/summary                      |                         |                             |                             |                               |              |                         |                              |
|Generation Stack                |interval               |                               |                               |interval              |                                  |                         |                             |                             |                               |interval      |interval                 |                              |
|Generation Stack Facet Grid     |interval               |                               |                               |interval              |                                  |                         |                             |                             |                               |interval      |interval                 |                              |
|Generation Timeseries Difference|interval               |                               |                               |                      |                                  |                         |                             |                             |                               |              |                         |                              |
|Generation Unstacked            |interval               |interval                       |                               |interval              |                                  |                         |                             |                             |                               |interval      |interval                 |                              |
|Generation Unstacked Facet Grid |interval               |interval                       |                               |interval              |                                  |                         |                             |                             |                               |interval      |interval                 |                              |
|Production Cost                 |                       |                               |year/summary                   |                      |year/summary                      |interval                 |interval                     |                             |                               |              |                         |                              |
|Reserve Timeseries              |                       |                               |                               |                      |                                  |                         |                             |                             |interval                       |              |                         |                              |
|Reserve Timeseries Facet Grid   |                       |                               |                               |                      |                                  |                         |                             |                             |interval                       |              |                         |                              |
|Time at Minimum Generation      |interval               |                               |                               |                      |                                  |                         |                             |year/summary                 |                               |              |                         |                              |
|Total Generation                |                       |interval                       |                               |interval              |                                  |                         |                             |                             |                               |interval      |                         |                              |
|Total Generation Facet Grid     |                       |interval                       |                               |interval              |                                  |                         |                             |                             |                               |interval      |                         |                              |
|Total Installed Capacity        |                       |                               |year/summary                   |                      |                                  |                         |                             |                             |                               |              |                         |                              |
|Total System Cost               |                       |                               |                               |                      |year/summary                      |                         |                             |                             |                               |              |                         |interval                      |
|Total Unserved Energy           |                       |                               |                               |                      |                                  |                         |                             |                             |                               |              |interval                 |                              |
|Unserved Energy Timeseries      |                       |                               |                               |                      |                                  |                         |                             |                             |                               |              |interval                 |                              |



