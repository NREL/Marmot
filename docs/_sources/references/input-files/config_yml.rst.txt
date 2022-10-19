.. raw:: html

   <script>
      var arr = document.getElementsByClassName('reference internal');
      for(var i = 0; i < arr.length; i++) {
      arr[i].innerHTML = arr[i].innerHTML.replace(/\./g, '.<wbr/>');
      }
   </script>

===============================
config: yml file
===============================


The config.yml allows adjustment of many internal settings for Marmot.
The adjustment of these settings are all optional and are setup with default values. 
Users who would like to further customize how Marmot works can change these values in a standard text editor.

Configurations include:

- Figure fonts
- Figure size
- Figure axes options
- Figure axes label options
- Figure Legend control
- Formatter control settings
- Assignment of primary input files 

Unlike the other input files, the config.yml is located in the root Marmot directory.
A config.yml file will be created the first time the `marmot_h5_formatter.py` or `marmot_plot_main.py` is run. 
After the yml file has been created users will be able to change any settings they like within the file. 
Once created, this file is never overwritten by Marmot. 
To revert to default settings, delete the config.yml file and Marmot will create it again at runtime. 


Default Values and Descriptions
--------------------------------

- **font_settings:**
   Settings to adjust font sizes, family, and tick size within figures.

   - xtick_size: 12
   - ytick_size: 12
   - axes_label_size: 16
   - legend_size: 12
   - title_size: 16
   - font_family: serif

- **text_position:**
   Adjust the position of text relative to the edge of the plot (matplotlib points).

   - title_height: 40

- **figure_size:**
   Adjust the x and y-axes dimensions of the output figure.
   
   - xdimension: 6
   - ydimension: 4

- **axes_options:**
   Allows adjustment of the minimum and maximum tick marks on datetime x-axes and 
   the number of decimal points to include on the y-axes. Also controls visibility of
   figure spines and legend settings and position.

   - x_axes_maxticks: 8
   - x_axes_minticks: 4
   - y_axes_decimalpt: 1
   - major_x_tick_length: 5
   - major_y_tick_length: 5
   - hide_top_spine: true
   - hide_right_spine: true
   - hide_bottom_spine: false
   - hide_left_spine: false
   - legend_position: center right
   - legend_columns: 1
   - show_legend_frame: true

   .. versionadded:: 0.9.0
      legend control settings

- **axes_label_options:**
   Controls whether x-axes labels are rotated from their default horizontal (0 degrees),
   and at what number of labels to begin rotating to the specified rotation_angle.
   By default, labels will begin rotating at 7 labels to an angle of 45 degrees from 
   0 degrees.

   - rotate_x_labels: true
   - rotate_at_num_labels: 7
   - rotation_angle: 45

- **plot_data:**
   Controls certain plot data settings. 
   `curtailment_property` source of Curtailment data. The code defaults to 
   Marmot's calculated Curtailment property.
   `plot_title_as_region` If True a the region/zone name will be added as a title to the 
   figure
   `include_barplot_load_storage_charging_line` specifies whether to include the line 
   representing pumped load in total generation bar plots. 
   `include_timeseries_load_storage_charging_line` specifies whether to include the line 
   representing pumped load in timeseries generation plots 
   `*_net_imports` settings controls whether net imports should be included in the figures.

   - curtailment_property: Curtailment
   - plot_title_as_region: true
   - include_barplot_load_lines: true
   - include_stackplot_load_lines: true
   - include_barplot_load_storage_charging_line: true
   - include_timeseries_load_storage_charging_line: true
   - include_barplot_net_imports: true
   - include_stackplot_net_imports: true

- **load_legend_names:**
   Sets the legened name of load and demand lines.

   - load: 'Demand + Storage Charging'
   - demand: Demand

   .. versionadded:: 0.10.0

- **formatter_settings:**
   Formatter specific settings, VOLL value, 
   `skip_existing_properties` Toggles whether existing properties are skipped or 
   overwritten if they already contained in a previous processed_h5 file, the default is 
   to skip.
   `append_plexos_block_name` Toggles whether to append PLEXOS block name to formatted 
   results e.g ST, MT, LT, PASA. Defaults to False.
   `exclude_pumping_from_reeds_storage_gen` toggles whether to exclude pumping 
   (negative gen) from ReEDS storage generation. Defaults to True.

   - VoLL: 10000
   - skip_existing_properties: true
   - append_plexos_block_name: false
   - exclude_pumping_from_reeds_storage_gen: true

   .. versionadded:: 0.10.0
      exclude_pumping_from_reeds_storage_gen setting

- **multithreading_workers:** 1
   Sets multithread workers when reading data, Defaults to 1.

- **figure_file_format:** svg
   Adjust the plot image format. The default is **svg**, a vector-based image.
   This field accepts any format that is compatible with matplotlib.

- **shift_leapday:** false
   Handles auto shifting of leap day, if required by your model. The default is false.

- **read_csv_properties:** false
   If True the Marmot plotter will attempt to read the required plot property from a 
   csv file if it cannot be found in the formatted h5 file.
   Format of data must adhere to the standard 
   Marmot formats for each data class, e.g generator, line etc.

   Filename should be of the following pattern:
      - {scenario}_{plx_prop_name}.csv
   

   An example of a line_Net_Import:
      - Base DA_line_Net_Import.csv
   

   These csv files should be saved in the *csv_properties* folder which will be 
   created in the Marmot_Solutions_folder.

   .. versionadded:: 0.11.0

- **auto_convert_units:** true
   If True automatically converts Energy and Capacity units so that no number 
   exceeds 1000. All base units are in MW, and units can be converted to GW, TW and kW.

- **user_defined_inputs_file:** Marmot_user_defined_inputs.csv
   Change the default Marmot_user_defined_inputs file, file must be created first.

- **plot_select_file:** Marmot_plot_select.csv
   Change the default Marmot_plot_select.csv file, file must be created first.

- **plexos_properties_file:** plexos_properties.csv
   Change the default plexos_properties_file.csv file, file must be created first.

- **reeds_properties_file:** reeds_properties.csv
   Change the default reeds_properties_file.csv file, file must be created first.
