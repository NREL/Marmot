.. raw:: html

   <script>
      var arr = document.getElementsByClassName('reference internal');
      for(var i = 0; i < arr.length; i++) {
      arr[i].innerHTML = arr[i].innerHTML.replace(/\./g, '.<wbr/>');
      }
   </script>

================================
Marmot_plot_select: csv file 
================================

Select which plots to create. 
Set values to **TRUE** to create the specified plot. 

Allows some customization on how the plot will appear, including:

- Adding property annotations 
- Set max y-axis value 
- Set start and end dates
- Specify timescale for timeseries plots 
- Adjust scenario Grouping

Since individual plots can be modified in several ways, copying of rows, and adjusting the 
**Figure Output Name** and other settings is permitted and encouraged.
The following describes the layout of the input sheet. 

Input Definitions
-----------------
Columns
~~~~~~~~~
The Marmot_plot_select.csv has 14 columns: 

- **Figure Output Name (required):**
   The name given to the figure when saved as output. 
   This name is user editable and can be adjusted as liked. Figures will be saved in the 
   `Figures_Output` folder where the **Marmot_Solutions_folder** has been set in the 
   :ref:`Marmot_user_defined_inputs: csv file`.
- **Plot Graph (required):**
   Set to TRUE to plot the figure or FALSE to skip.
- **Plot Property (optional):**
   Controls the addition of annotations or selection of certain data elements such as 
   individual generator or line names. The following are the list of available annotations that can be applied
   to the specified plots:

      - **gen_stack:**
         - `Peak Demand`
         - `Min Net Load`
         - `Peak RE`
         - `Peak Unserved Energy`
         - `Peak Curtailment`
      - **gen_unstack:**
         - `Min Net Load`
         - `Peak Demand`
      - **reserve_gen_timeseries:**
         - `Peak Reserve Provision`

- **Y-Axis Max (optional):**
   The maximum y-axis value. Currently only used in the price.py plots
- **Day Before (optional):**
    Sets the number of days to include before the specified annotation in **Plot Property**. 
    If timestamps data intervals are larger than hours the interval range will be used in place of Days. 
    (e.g., if data interval is years, this is the equivalent of `Year before`)
- **Day After (optional):**
   Sets the number of days to include after the specified annotation in **Plot Property**
   If timestamps data intervals are larger than hours, the interval range will be used in place of Days. 
   (e.g., if data interval is years, this is the equivalent of `Year before`)
- **Timezone (optional):**
   Optional time zone to apply to the figures x-axes
- **Start date (optional):**
   The start date to plot data from. Accepts any of the following formats:

      - M/D/Y
      - M/Y
      - Y
      - M/D/Y HH:MM:SS

   It will also accept dates formatted as D/M/Y and Y/M/D. Python will make a best guess at the 
   inputted format. If left blank the entire range of data will be plotted. 
- **End date (optional):**
   The end date to plot data to. Accepts any of the following formats:

      - M/D/Y
      - M/Y
      - Y
      - M/D/Y HH:MM:SS

   It will also accept dates formatted as D/M/Y and Y/M/D. Python will make a best guess at the 
   inputted format. If left blank the entire range of data will be plotted.
- **Timeseries Plot Resolution (optional):**
   The resolution to plot timeseries figures at. Current available options are 
   `Interval` and `Annual`. If the value is left blank and the plot creates a timeseries, Marmot will
   default to plotting interval data.

   .. versionadded:: 0.10.0

- **Group by Scenario or Year-Scenario (optional):**
   Specifies whether to group data by `Scenario` or `Year-Scenario`. This setting will only
   effect data that has been grouped into bar plots or diurnal line plots and will have no 
   effect on timeseries plots.
   If grouping by `Year-Scenario` the year will be identified 
   from the input data timestamp and appended to the scenario name defined in 
   :ref:`Marmot_user_defined_inputs: csv file`. This is useful when 
   plotting data which covers multiple years such as ReEDS. If left blank Marmot will default to
   grouping data by `Scenario`.

   .. versionadded:: 0.10.0

- **Custom Data File (optional):**
   Path to a custom data file allowing the insertion of custom columns that will be
   appended to the final data frame before plotting
   The custom data file must be a csv.
   Default position of new columns is at the end of the existing data frame.
   Custom data files should have the same format as the final output data csv's, 
   the usual format is scenarios in index, generator techs as column names. 
   However, this format may not hold true for all plots in the future. 
   Therefore, using this feature may involve some trial and error and should be considered experimental.
   Custom Data Files can be added into the following plots:

      - production_cost: prod_cost
      - production_cost: sys_cost
      - production_cost: detailed_gen_cost
      - production_cost : sys_cost_type
      - emissions : total_emissions_by_type

   .. versionadded:: 0.9.0

- **Marmot Module (required):**
   The name of the module containing the plotting methods. Plotting modules group plots by type, 
   e.g `total_generation`, `prices`, `curtailment` etc. 
   This is a required value and should not be modified. 
   Copying the data to a new line is permitted with the inclusion of the corresponding method.
- **Method (required):**
   The name of the method containing the specific plot logic. These methods are contained within 
   the corresponding module.
   This is a required value and should not be modified.
   Copying the data to a new line is permitted with the inclusion of the corresponding module.

Input Example
--------------

For an example of the file see it on the GitHub repo:
`Marmot_plot_select <https://github.com/NREL/Marmot/blob/ReEDS_formatter/marmot/input_files/Marmot_plot_select.csv>`_


