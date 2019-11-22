# Marmot
Marmot is a set of python scripts to process h5plexos PLEXOS results to ready them for plotting and then plot the outputs 
![Yellow-bellied marmot](https://upload.wikimedia.org/wikipedia/commons/3/3b/Marmot-edit1.jpg)

## Main Python Scripts
Marmot consists of two .py files:
* **PLEXOS_H5_results_formatter.py**
* **Marmot_results_plotting.py**

## PLEXOS_H5_results_formatter
The **PLEXOS_H5_results_formatter** reads in PLEXOS hdf5 files created with the h5plexos library (the repo for which can be found [here](https://github.com/NREL/h5plexos)) and processes the output results to ready them for plotting. Once the outputs have been processed they are saved to a itermediatry hdf5 file which can then be read into **Marmot_results_plotting.py**.

Before you use the **PLEXOS_H5_results_formatter** you will need to set adjust the input settings and mapping files which are all defined at the top of the code under *User Defined Names, Directories and Settings*, these include:

- `Plexos_Properties` This is a csv file which determiens which plexos properties to pull from the h5plexos results and process, this file is in the repo. Under the *"collect_data"* column adjust the property to be TRUE or FALSE to change if the data is processed. If your property is not here, add it as a new line with the same format. 

- `Scenario_name` This is the name of the scenario being run and is a way to differentiate between multiple h5plexos results you may want to process. The PLEXOS hdf5 results files should be saved in a folder with this name.

- `HDF5_folder_in` This is the folder where you store all your PLEXOS hdf5 results. This folder should contain the Scenario_name sub-folders. Here's an example of how that would look:
  - HDF5_folder_in
    - Scenario_name_1
      - results_m1.h5
      - results_m2.h5
      - .....
    - Scenario_name_2
      - results.h5

- `Run_folder` This is the base directory where all processed scenarios, intermediary h5files and figures will live. When **PLEXOS_H5_results_formatter** is run it will create a *PLEXOS_Scenarios* folder here, in which all the proceessed scenarios and their results will be saved. 

- `Mapping_folder` This is the folder which contains mapping csv files to rename generators to whatever the user desires and also organise regions and reserves. Each of these files will be explained in more detail in the **Mapping Files** section bellow. 

- `overlap` This is the number of hours overlapped between two adjacent models, default is 0

- `VoLL` Value of lost load for calculating the Cost of Unserved Energy, default is 10,000 $/MWh

- `HDF5_output` Name of hdf5 file which holds the itermediatry hdf5 outputs, default name is "PLEXOS_outputs_formatted.h5"

  ### Mapping Files
Marmot uses csv files to map in extra regions to your data and rename generators and reserve properties. Examples of these files can be found within in the repo in the [mapping_folder](https://github.nrel.gov/PCM/Marmot/tree/master/mapping_folder). The examples are setup to work with the NARIS PLEXOS databses so make sure to adjust these csv files if you are not running with that underlying database. These csv files are:

- **gen_names.csv** This file allows you to change the name of the PLEXOS generator technology categories to be consistant. For example change all the possible gas cc generator names to just be called "Gas-CC". The csv file has two columns *"Original"*, which contains the name of all the PLEXOS generator categories and *"New"*, which is the new name you want to give the categories. 

- **Region_mapping.csv** This file allows you to group PLEXOS regions together to create aggregated regions. The first column in the file should always be called *"region"* and should contain the name of all the regions in your PLEXOS databse. The names given to all other columns is up to you. In the example given in the repo, we aggreagted NARIS regions to the zonal, country and interconnect level. Currently Marmot is setup to have PLEXOS regions to be the smallest of all areas, we will work to give the option to have this as zones in coming updates.

- **reserve_region_type.csv** This file allows you to adjust the Reserve Region names and reserve types. Check the file for an example.

## Marmot_results_plotting
I will write this section in a while, email daniel.levie@nrel.gov if you need help
