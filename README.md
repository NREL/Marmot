# Marmot
Marmot is a set of python scripts to process h5plexos PLEXOS results plot the outputs 
![Yellow-bellied marmot](https://upload.wikimedia.org/wikipedia/commons/3/3b/Marmot-edit1.jpg)

## Main Python Scripts
Marmot consists of two main .py files:
* [**PLEXOS_H5_results_formatter.py**](https://github.nrel.gov/PCM/Marmot#plexos_h5_results_formatter)
* [**Marmot_plot_main.py**](https://github.nrel.gov/PCM/Marmot#marmot_plot_main)

A high-level explanation of what these files do and suggested settings to change are described in this readme. Code specifics are decribed in more detail in the code comments. 

## PLEXOS_H5_results_formatter
The **PLEXOS_H5_results_formatter** reads in PLEXOS hdf5 files created with the h5plexos library (the repo for which can be found [here](https://github.com/NREL/h5plexos)) and processes the output results to ready them for plotting. Once the outputs have been processed, they are saved to an intermediary hdf5 file which can then be read into the Marmot plotting code.

Before you use the **PLEXOS_H5_results_formatter** you will need to adjust and set the input settings in the **Marmot_user_defined_inputs.csv** and set which PLEXOS properties to process in the **plexos_properties.csv**. You may also want to edit the Mapping Files described [here](https://github.nrel.gov/PCM/Marmot#mapping-files). These files are located in the repo and are available to be edited once you clone the repo. 

Settings to adjust in the **Marmot_user_defined_inputs.csv** required to run the formatter include:

- `PLEXOS_Solutions_folder` The folder that contains all h5plexos files that have come from PLEXOS. This folder should contain the Scenario_list sub-folders in which the h5plexos files are contained. Here's an example of how that would look:
  - PLEXOS_Solutions_folder
    - Scenario_name_1
      - results_m1.h5
      - results_m2.h5
      - .....
    - Scenario_name_2
      - results.h5

- `Marmot_Solutions_folder` This is the base directory to create folders and save outputs in. When **PLEXOS_H5_results_formatter** is run it will create a folder for each sceanrio here, in which all the proceessed results and figures will be saved. This folder can have the same address as PLEXOS_Solutions_folder, having an alternative address allows the user to save outputs in a different location from inputs if desired. 

- `Scenario_process_list` This is the list of scenarios to process. The h5plexos hdf5 results files should be saved in folders with these names. The list must contain at least one entry. 

- `overlap` This is the number of hours overlapped between two adjacent models, default is 0

- `VoLL` Value of lost load for calculating the Cost of Unserved Energy, default is 10,000 $/MWh

- `Region_Mapping.csv_name` The name of the Region_Mapping.csv described in more detail in [Mapping Files](https://github.nrel.gov/PCM/Marmot#mapping-files) bellow.

- `gen_names.csv_name` The name of the gen_names.csv described in more detail in [Mapping Files](https://github.nrel.gov/PCM/Marmot#mapping-files) bellow.

- `reserve_region_type.csv_name` the name of the reserve_region_type.csv described in more detail in [Mapping Files](https://github.nrel.gov/PCM/Marmot#mapping-files) bellow.

Finally adjust the PLEXOS properties to process in the **plexos_properties.csv**. This csv file determines which PLEXOS properties to pull from the h5plexos results and process. Under the *"collect_data"* column adjust the property to be TRUE or FALSE to change if the data is processed. If your property is not here, add it as a new line with the same format. See the table at the end of this README file to see which properties are necessary for which Marmot plots.

  ### Mapping Files
Marmot gives the user the ability to map in extra regions to your data, rename generators and reserve properties, adjust generator technology colors using a set of csv files. Adjusting these values to your specific PLEXOS database is not required for Marmot to run but recommended for best results.
Examples of these files can be found within in the repo in the [mapping_folder](https://github.nrel.gov/PCM/Marmot/tree/master/mapping_folder). The examples are setup to work with the NARIS PLEXOS databases so make sure to adjust these csv files if you are not running with that underlying database. The settings in these mapping files are optional but give the user more control over plotting and aggregation of data. These csv files are:

- **gen_names.csv** This file allows you to change the name of the PLEXOS generator technology categories to be consistent. For example, change all the possible gas cc generator names to just be called "Gas-CC". The csv file has two columns *"Original"*, which contains the name of all the PLEXOS generator categories and *"New"*, which is the new name you want to give the categories. 

- **ordered_gen** Ordered list of generators which determines how they appear in a stack plot; generator names should equal those in the gen_names.csv *"New"* column

- **Region_mapping.csv** This file allows you to group PLEXOS regions together to create aggregated regions. The first column in the file should always be called *"region"* and should contain the name of all the regions in your PLEXOS database. The names given to all other columns is up to you. In the example given in the repo, we aggregated NARIS regions to the country and interconnect level.   

- **reserve_region_type.csv** This file allows you to adjust the Reserve Region names and reserve types. Check the file for an example. 

- **colour_dictionary.csv** This allows the user to adjust the color used to plot generation technology types e.g Gas-CC, Wind, PV etc. The names in the generator column should equal those in the gen_names.csv *"New"* column. The current colors are the default SEAC colors recommended for use in publications.

 - **pv_gen_cat.csv** , **re_gen_cat.csv** , **vre_gen_cat.csv** & **thermal_gen_cat.csv** - Generators which belong to specified category, used for certain figures and determining which generators to include for curtailment calculations.

### View formatted contents
In order to check the contents of an existing processed HDF5 folder, use the following (adjusting scenario as desired):
temp=pd.HDFStore(hdf_out_folder+"/"+Multi_Scenario[0]+"_formatted.h5")
temp.keys()
temp.close()

## Marmot_plot_main

**Marmot_plot_main.py** is the main plotting script within Marmot which calls on supporting files to read in data, create the plot, and then return the plot and data to **Marmot_plot_main.py**. The supporting modules can be viewed within the repo [plottingmodules](https://github.nrel.gov/PCM/Marmot/tree/master/plottingmodules) folder and have descriptive names such as **total_generation.py**, **generation_stack.py**, **curtaiment.py** etc. 

As with the processing script users will need to adjust the input settings in the **Marmot_user_defined_inputs.csv** and set which plots to create in **Marmot_plot_select.csv**. 
**Marmot_plot_select.csv** is a csv file which determines which figures to plot. This file is in the repo. Under the *"Plot Graph"* column adjust the property to be TRUE or FALSE to decide whether to plot the figure. Column *D* allows the user to adjust certain properties within the plot (examples given). Columns *E* and *F* adjust the range of days to plot either side of the specified property in *D*. Column *G* adjusts the time zone to plot on the figure. The list of figures to plot is currently limited by what code has been written for.  

Settings to adjust in the **Marmot_user_defined_inputs.csv** required to run the plotting code include:

- `PLEXOS_Solutions_folder` Same as described [above](https://github.nrel.gov/PCM/Marmot#plexos_h5_results_formatter)

- `Marmot_Solutions_folder` Same as described [above](https://github.nrel.gov/PCM/Marmot#plexos_h5_results_formatter)

- `Main_scenario_plot` This is the name of the scenario to plot and is also the folder where plots are saved. This folder will be located in the `Marmot_Solutions_folder` decribed [above](https://github.nrel.gov/PCM/Marmot#plexos_h5_results_formatter)

- `Multi_scenario_plot` This is a list of scenarios to plot on the same figure, allowing comparisons to be made between them. The order of the scenarios will determine the order of the scenarios in the plot. This list should have at least one entry and be equal to `Main_scenario_plot`

- `Scenario_Diff_plot` This is a list which can contain max two entries. This list is used to create plots using the difference of the values between two scenarios. The second scenario in the list is subtracted from the first. If you are not creating difference plots this list can remain empty.

- `AGG_BY` A string which tells Marmot which region type to aggregate by when creating plots. The default options are *”regions”* and *“zones”*. Other options can be added based on how the user sets up **Region_mapping.csv** described [above](https://github.nrel.gov/PCM/Marmot#mapping-files)

- `zone_region_sublist` List of *"regions/zones”* to plot if results are not required for all regions. The list of *"regions/zones”* should be contained within the `AGG_BY` aggregation. This is an optional field and can be left empty if not needed.

- `Facet_ylabels` & `Facet_xlabels` If you wish to create a Facet plot, these labels will be applied to the axis. The amount of entries given to each label will determine the dimensions of the Facet plot. This should be equal to the number of scenarios you are plotting. For example, if you have 6 scenarios your Facet Grid dimensions may be [2,3], [3,2] or [1,6] etc. This is an optional field and can be left empty if not needed.

- `Figure_Format` The format to save images in. If left blank Marmot will output all figures as .png. We recommend that the user uses a .svg figure format which is a vector-based image format and compatible with all Microsoft Office products. Vector images uses mathematical paths to determine how images are drawn compared to raster images (jpg,png etc.) which use pixels. The main advantages of vector images are reduced size and that they do not decrease in quality when image display size is adjusted. For more information see the following article https://www.geeksforgeeks.org/vector-vs-raster-graphics/


##Required PLEXOS properties

To use the following table, find the columns that correspond to your desired plots. The corresponding rows contain the properties required in your PLEXOS solutions files. Make sure to these properties to *TRUE* in **Plexos_Properties.csv**. `interval` corresponds to time series data with observations at every timestemp, whereas `year/summary` properties report a single value for every object.

|                           		|Average Output When Committed|Capacity Factor|Capacity Started|Curtailment Duration Curve|Curtailment vs. Penetration|Generation Stack|Generation Stack Facet Grid|Generation Timeseries Difference|Generation Unstacked|Generation Unstacked Facet Grid|Production Cost|Reserve Timeseries|Reserve Timeseries Facet Grid|Time at Minimum Generation|Total Generation|Total Generation Facet Grid|Total Installed Capacity|Total System Cost|Total Unserved Energy|Unserved Energy Timeseries| Utilization Factor Fleet | Utilization Factor Generators |Line Utilization Annual | Line Utilization Hourly | Region Price | Region Price Timeseries | Constraint Violation
|----------------------------------|-----------------------------|---------------|----------------|--------------------------|---------------------------|----------------|---------------------------|--------------------------------|--------------------|-------------------------------|---------------|------------------|-----------------------------|--------------------------|----------------|---------------------------|------------------------|-----------------|---------------------|--------------------------|--------------------------|--------------------------|--------------------------|--------------------------|--------------------------|--------------------------|--------------------------|
|Generator -- Generation 			|interval                     |interval       |interval        |interval                  |interval                   |interval        |interval                   |interval                        |interval            |interval                       |               |                  |                             |interval                  |                |                           |                        |                 |                     |                          |interval					|interval					|							|						|							|							|							|
|Generator -- Available Capacity   |                             |               |                |interval                  |interval                   |                |                           |                                |interval            |interval                       |               |                  |                             |                          |interval        |interval                   |                        |                 |                     |                          |interval					|interval					|							|						|							|							|							|
|Generator -- Installed Capacity   |year/summary                 |year/summary   |year/summary    |                          |                           |                |                           |                                |                    |                               |year/summary   |                  |                             |                          |                |                           |year/summary            |                 |                     |                          |							|							|							|						|							|							|							|
|Generator -- Pump Load            |                             |               |                |                          |                           |interval        |interval                   |                                |interval            |interval                       |               |                  |                             |                          |interval        |interval                   |                        |                 |                     |                          |							|							|							|						|							|							|							|
|Generator -- Total Generation Cost|                             |               |                |                          |year/summary               |                |                           |                                |                    |                               |year/summary   |                  |                             |                          |                |                           |                        |year/summary     |                     |                          |							|							|							|						|							|							|							|
|Generator -- Pool Revenue         |                             |               |                |                          |                           |                |                           |                                |                    |                               |interval       |                  |                             |                          |                |                           |                        |                 |                     |                          |							|							|							|						|							|							|							|
|Generator -- Reserves Revenue     |                             |               |                |                          |                           |                |                           |                                |                    |                               |interval       |                  |                             |                          |                |                           |                        |                 |                     |                          |							|							|							|						|							|							|							|
|Generator -- Hours at Minimum     |                             |               |                |                          |                           |                |                           |                                |                    |                               |               |                  |                             |year/summary              |                |                           |                        |                 |                     |                          |							|							|							|						|							|							|							|
|Reserve generators -- Provision   |                             |               |                |                          |                           |                |                           |                                |                    |                               |               |interval          |interval                     |                          |                |                           |                        |                 |                     |                          |							|							|							|						|							|							|							|
|Region -- Load                    |                             |               |                |                          |                           |interval        |interval                   |                                |interval            |interval                       |               |                  |                             |                          |interval        |interval                   |                        |                 |                     |                          |							|							|							|						|							|							|							|
|Region -- Unserved Energy         |                             |               |                |                          |                           |interval        |interval                   |                                |interval            |interval                       |               |                  |                             |                          |                |                           |                        |                 |interval             |interval                  |							|							|							|						|							|							|							|
|Region -- Cost Unserved Energy    |                             |               |                |                          |                           |                |                           |                                |                    |                               |               |                  |                             |                          |                |                           |                        |interval         |                     |                          |							|							|							|						|							|							|							|
|Region -- Price					|                             |               |                |                          |                           |                |                           |                                |                    |                               |               |                  |                             |                          |                |                           |                        |     		    |                     |                          |							|							|							|						|interval						|							|							|
|Line --	Flow					|                             |               |                |                          |                           |                |                           |                                |                    |                               |               |                  |                             |                          |                |                           |                        |       			 |                     |                          |							|							|	interval				|interval				|							|							|							|
