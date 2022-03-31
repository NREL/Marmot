.. raw:: html

   <script>
      var arr = document.getElementsByClassName('reference internal');
      for(var i = 0; i < arr.length; i++) {
      arr[i].innerHTML = arr[i].innerHTML.replace(/\./g, '.<wbr/>');
      }
   </script>

===============================
reeds_properties: csv file
===============================


The reeds_properties.csv file determines which ReEDS properties to process with Marmot.

Set values to **TRUE** to process the specified property.

If a property you would like to process is not in this file, add it as a new row with the same format.

Input Definitions
-----------------
Columns
~~~~~~~~~
The reeds_properties.csv has 6 columns: 

- **data_type:**
   Specifies the interval of data to process. `interval` will always process the smallest interval data available. 
   For ReEDS this is hourly data. Since ReEDS does not natively export full 8760 hourly data, Marmot will
   create this using the ReEDS timesteps and an internal ReEDS mapping file. 
   The `year` data_type will format the data at the standard ReEDS year interval.
- **group:** 
   The data class group, used by Marmot to group property types.
   Valus should all be lower-case strings.
- **data_set:** 
   The property associated with the group, e.g, `generator gen_out`. These names are equivalent to the 
   ReEDS properties located in the results gdx file. 
- **collect_data:**
   Set to TRUE to process the property or FALSE to skip.
- **description:**
   Describes the property.
- **Marmot Property:**
   The name Marmot will assign the property once processed. The values assigned here do not effect the internal code 
   and are just provided for user information.


Input Example
--------------

.. csv-table:: `reeds_properties <https://github.com/NREL/Marmot/blob/ReEDS_formatter/marmot/input_files/reeds_properties.csv>`_
   :file: ../../tables/reeds_properties_example.csv
   :widths: 10, 15, 15, 10, 30, 20
   :header-rows: 1




