.. raw:: html

   <script>
      var arr = document.getElementsByClassName('reference internal');
      for(var i = 0; i < arr.length; i++) {
      arr[i].innerHTML = arr[i].innerHTML.replace(/\./g, '.<wbr/>');
      }
   </script>

======================================
Marmot_user_defined_inputs: csv file
======================================

Sets what data will be processed and plotted:

- **For formatting:** 
   Controls the type and location of data to process.
- **For Plotting:** 
   Sets the scenarios to plot and compare, the specific aggregation to plot,
   and any figure axes labels.

The names of the mapping files to use are also set here. By default, these are looked 
for in the *mapping folder*. A full path can also be used in their place. 

Input Definitions
-----------------

Columns
~~~~~~~~~
The Marmot_user_defined_inputs.csv has 4 columns: 

- **Input:** 
   Name of input being adjusted
- **Input_type:** 
   Python data type of input 
- **Importance:** 
   Whether something is *Required* or *Optional*
- **User_defined_value:** 
   The user value for the specific input

Rows 
~~~~~
The following are the rows by Input

- **Simulation Model (str, required):** 
   Name of model being processed, current options are PLEXOS, ReEDS and EGRET

   .. versionadded:: 0.9.0

- **PLEXOS_data_blocks (list, optional):** 
   PLEXOS results type. e.g 'ST', 'MT', 'LT' or 'PASA', can pass a list of values (comma separated) 
   or a single value. If left blank will default to ST. Only used when processing PLEXOS results.

   .. versionadded:: 0.9.0
   
      .. note::
         If processing multiple result sets of different types, be sure to set **formatter_settings: append_plexos_block_name**
         to **true** in the :ref:`config: yml file` before formatting data. If this value is not set to true, results will be 
         saved to the same "*_formatted.h5*" file which may result in overwritten data.
   
- **Model_Solutions_folder (path, required):** 
   The directory that contains all the **Simulation Model** results. 
   This directory should include a sub-folder for each scenario in the **Scenario_process_list**, 
   each of which holds the individual scenario results files. For PLEXOS this would be h5plexos files and for ReEDS 
   this would be the output results folder containing gdx files. 
- **Marmot_Solutions_folder (path, optional):** 
   This is the base directory for saving outputs. 
   When *marmot_h5_formatter.py* is run it will create a `Processed_HDF5 folder` here, and will save all the processed results with the extension 
   "*_formatted.h5*". Including a folder path here allows the user to save outputs in a different location from 
   inputs if desired. If left blank all data and plots will be saved in **Model_Solutions_folder**.
- **Scenario_process_list:** (path, required) 
   This is the list of scenarios to process. The simulation results files should be saved in folders with these names. 
   The list must contain at least one entry. 
- **process_subset_years (list, optional):**
   For ReEDS models specifies a subset of years to process to the "*_formatted.h5*" file. If left blank all years will be processed.
   
   .. versionadded:: 0.10.0

- **Region_Mapping.csv_name (str, optional):** 
   The name of the Region_Mapping.csv. This file allows custom geographic aggregations to be 
   created. It is described in more detail in :ref:`Region Mapping: csv file`.
- **gen_names.csv_name (str, required):** 
   The name of the gen_names.csv. Used to rename existing model generator technology types.
   It is described in more detail in :ref:`gen_names: csv file`.
- **ordered_gen_categories_file (str, required):**
   The name of the ordered_gen_categories_file.csv. Used to specify the order
   generator technologies appear in a figure and allows grouping of technologies into
   pre-defined and custom categories. It is described in more detail in :ref:`ordered_gen_categories: csv file`.
- **emit_names.csv_name (str, required):**
   The name of the emit_names.csv. Allows renaming of existing emission types. 
   It is described in more detail in :ref:`emit_names: csv file`.
- **color_dictionary_file (str, required):**
   The name of the color_dictionary_file.csv. Sets the color for each generator technology.
   It is described in more detail in :ref:`colour_dictionary: csv file`.
- **Scenarios (list, required):** 
   This is the name(s) of the scenario(s) to plot. The order of the scenarios will determine the order of the 
   scenarios in the plot. The plotter will look for these scenarios in the `Processed_HDF5 folder` of the 
   **Marmot_Solutions_folder** described above. The "*_formatted.h5*" suffix should not be included in the scenario name. 
- **Scenario_Diff_plot (list, optional):**
   Two value list of scenarios which is used to compare one scenario directly against another. The first scenario will be 
   subtracted from the second. This setting is currently only used by the `gen_diff` plotting method in the generation_stack.py module.
- **AGG_BY (str, required):** 
   A string that determines the region type by which to aggregate when creating plots. Values are `case sensitive`.
   The default options are *”regions”* and *“zones”*. Other options can be added based on how the user sets 
   up the **Region_mapping.csv**, described in :ref:`Region Mapping: csv file`.
- **zone_region_sublist (list, optional):** 
   List of *"regions/zones”* to plot if results are not required for all regions. 
   The list of *"regions/zones”* should be contained within the **AGG_BY** aggregation. 
   This is an optional field and can be left empty if not needed.
- **TECH_SUBSET (str, optional):** 
   Tech category to subset by to allow plotting of only certain technologies, 
   e.g., thermal, vre, storage etc. The value given here should correspond to one of the columns 
   in the :ref:`ordered_gen_categories: csv file`. Values are `case sensitive`.
   If left blank all available technologies will be plotted.
- **Facet_ylabels Facet_xlabels (list, optional):** 
   Labels that will be applied to the axes of facet plots. 
   The number of entries given to each label will determine the dimensions of the Facet plot.  
   For example, if you have 6 scenarios, your facet grid dimensions in x,y coordinates may be [2,3], [3,2] or [1,6] etc. 
   This is an optional field and can be left empty if not needed. Facet plots can still be created without labels if desired. 
   The facet layout will default to max 3 plots wide, with no maximum to the length if not Facet labels are supplied. 
- **Tick_labels (list, optional):** 
   List of custom tick labels, which allows adjustment of scenario names on all bar plots. 
   Does not apply to difference plots or plots from the `capacity_factor.py` module.

.. admonition:: See Also

   What’s a Facet Plot? Examples and usage guide.

Input Example
--------------

.. csv-table:: `Marmot_user_defined_inputs <https://github.com/NREL/Marmot/blob/ReEDS_formatter/marmot/input_files/Marmot_user_defined_inputs.csv>`_
   :file: ../../tables/user_defined_inputs_example.csv
   :widths: 35, 15, 15,  35
   :header-rows: 1
