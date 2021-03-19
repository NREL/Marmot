# Marmot
Marmot is a set of python scripts to process h5plexos PLEXOS results plot the outputs 
![Yellow-bellied marmot](https://upload.wikimedia.org/wikipedia/commons/3/3b/Marmot-edit1.jpg)

## Main Python Scripts
Marmot consists of two main .py files:
* [**marmot_h5_formatter.py**](https://github.nrel.gov/PCM/Marmot/blob/ReEDS2_integration/marmot/marmot_h5_formatter.py)
* [**marmot_plot_main.py**](https://github.nrel.gov/PCM/Marmot/blob/ReEDS2_integration/marmot/marmot_plot_main.py)

A high-level explanation of what these files do and suggested settings to change are described in this readme. Code specifics are decribed in more detail in the code comments. 

## Initial Setup
* First `git clone git@github.nrel.gov:PCM/Marmot.git` to any location you like.
* To ensure you are using all the required python modules, create a new conda environment using the provided [environment yml file](https://github.nrel.gov/PCM/Marmot/blob/master/marmot-env.yml). If you are unsure how to do this, follow [these steps](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).
* Follow the **marmot_h5_formatter** and **marmot_plot_main** steps below to run Marmot.

## marmot_h5_formatter
The **marmot_h5_formatter** reads in PLEXOS hdf5 files created with the h5plexos library (the repo for which can be found [here](https://github.com/NREL/h5plexos)) and processes the output results to ready them for plotting. Once the outputs have been processed, they are saved to an intermediary hdf5 file which can then be read into the Marmot plotting code. From the h5plexos ReadMe: "This package provides a Python interface for reading HDF5 files with H5PLEXOS v0.5 and v0.6 formatting. To create v0.5 files, use a version of this package in the 0.5 series. To create v0.6 files, use H5PLEXOS.jl."

The Marmot formatter imports h5plexos. To avoid errors, use h5plexos v0.5 or later. You may run into issues if you created the intermediate hdf5 file with h5plexos in Julia, but the h5plexos installation called by **marmot_h5_formatter** is v0.4 or earlier.

Before you use the **marmot_h5_formatter** you will need to adjust and set the input settings in the **Marmot_user_defined_inputs.csv** and set which PLEXOS properties to process in the **plexos_properties.csv**. You may also want to edit the Mapping Files described [here](https://github.nrel.gov/PCM/Marmot#mapping-files). These files are located in the repo and are available to be edited once you clone the repo. 

Settings to adjust in the **Marmot_user_defined_inputs.csv** required to run the formatter include:

- `PLEXOS_Solutions_folder` The folder that contains all h5plexos files that have come from PLEXOS. This folder should contain the Scenario_list sub-folders in which the h5plexos files are contained. Here's an example of how that would look:
  
  ![marmot folder structure](https://github.nrel.gov/storage/user/1084/files/bf9d1670-1254-11eb-8e62-c06455591fb2)

- `Marmot_Solutions_folder` This is the base directory to create folders and save outputs in. When **PLEXOS_H5_results_formatter** is run it will create a folder for each sceanrio here, in which all the proceessed results and figures will be saved. This folder can have the same address as PLEXOS_Solutions_folder, having an alternative address allows the user to save outputs in a different location from inputs if desired. 

- `Scenario_process_list` This is the list of scenarios to process. The h5plexos hdf5 results files should be saved in folders with these names. The list must contain at least one entry. Using the above example this would be "*Scenario_name_1, Scenario_name_2*"

- `VoLL` Value of lost load for calculating the Cost of Unserved Energy, default is 10,000 $/MWh

- `Region_Mapping.csv_name` The name of the Region_Mapping.csv described in more detail in [Mapping Files](https://github.nrel.gov/PCM/Marmot#mapping-files) bellow.

- `gen_names.csv_name` The name of the gen_names.csv described in more detail in [Mapping Files](https://github.nrel.gov/PCM/Marmot#mapping-files) bellow.

Finally adjust the PLEXOS properties to process in the **plexos_properties.csv**. This csv file determines which PLEXOS properties to pull from the h5plexos results and process. Under the *"collect_data"* column adjust the property to be TRUE or FALSE to change if the data is processed. If your property is not here, add it as a new line with the same format. See the table at the end of this README file to see which properties are necessary for which Marmot plots.

  ### Mapping Files
Marmot gives the user the ability to map in extra regions to your data, rename generators and reserve properties, adjust generator technology colors using a set of csv files. Adjusting these values to your specific PLEXOS database is not required for Marmot to run but recommended for best results.
Examples of these files can be found within in the repo in the [mapping_folder](https://github.nrel.gov/PCM/Marmot/tree/master/mapping_folder). The examples are setup to work with the NARIS PLEXOS databases so make sure to adjust these csv files if you are not running with that underlying database. The settings in these mapping files are optional but give the user more control over plotting and aggregation of data. These csv files are:

- **gen_names.csv** This file allows you to change the name of the PLEXOS generator technology categories to be consistent. For example, change all the possible gas cc generator names to just be called "Gas-CC". The csv file has two columns *"Original"*, which contains the name of all the PLEXOS generator categories and *"New"*, which is the new name you want to give the categories. 

- **ordered_gen** Ordered list of generators which determines how they appear in a stack plot; generator names should equal those in the gen_names.csv *"New"* column

- **Region_mapping.csv** This file allows you to group PLEXOS regions together to create aggregated regions. The first column in the file should always be called *"region"* and should contain the name of all the regions in your PLEXOS database. The names given to all other columns is up to you. In the example given in the repo, we aggregated NARIS regions to the country and interconnect level.   

- **colour_dictionary.csv** This allows the user to adjust the color used to plot generation technology types e.g Gas-CC, Wind, PV etc. The names in the generator column should equal those in the gen_names.csv *"New"* column. The current colors are the default SEAC colors recommended for use in publications.

 - **pv_gen_cat.csv** , **re_gen_cat.csv** , **vre_gen_cat.csv** & **thermal_gen_cat.csv** - Generators which belong to specified category, used for certain figures and determining which generators to include for curtailment calculations.

### View formatted contents
In order to check the contents of an existing processed HDF5 folder, use the following (adjusting scenario as desired):
temp=pd.HDFStore(hdf_out_folder+"/"+Multi_Scenario[0]+"_formatted.h5")
temp.keys()
temp.close()

## marmot_plot_main

**marmot_plot_main.py** is the main plotting script within Marmot which calls on supporting files to read in data, create the plot, and then return the plot and data to **marmot_plot_main.py**. The supporting modules can be viewed within the repo [plottingmodules](https://github.nrel.gov/PCM/Marmot/tree/master/plottingmodules) folder and have descriptive names such as **total_generation.py**, **generation_stack.py**, **curtaiment.py** etc. 

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
