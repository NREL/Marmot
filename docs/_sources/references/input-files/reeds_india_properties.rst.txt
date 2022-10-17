.. raw:: html

   <script>
      var arr = document.getElementsByClassName('reference internal');
      for(var i = 0; i < arr.length; i++) {
      arr[i].innerHTML = arr[i].innerHTML.replace(/\./g, '.<wbr/>');
      }
   </script>

=================================
reeds_india_properties: csv file
=================================


The reeds_india_properties.csv file determines which ReEDS India properties to process with Marmot.

Set values to **TRUE** to process the specified property.

If a property you would like to process is not in this file, add it as a new row with the same format.

Input Definitions
-----------------
Columns
~~~~~~~~~
The reeds_india_properties.csv has 6 columns: 

- **data_type:**
   Specifies the interval of data to process. `interval` will always process the smallest interval data available. 
   For ReEDS India this is hourly data. Since ReEDS India does not natively export full 8760 hourly data, Marmot will
   create this using the ReEDS India timesteps and an internal ReEDS mapping file. 
   The `year` data_type will format the data at the standard ReEDS year interval.
- **group:** 
   The data class group, used by Marmot to group property types.
   Valus should all be lower-case strings.
- **data_set:** 
   The property associated with the group, e.g, `generator GEN`. These names are equivalent to the 
   ReEDS India properties located in the results gdx file. 
- **collect_data:**
   Set to TRUE to process the property or FALSE to skip.
- **description:**
   Describes the property.
- **Marmot Property:**
   The name Marmot will assign the property once processed. The values assigned here do not effect the internal code 
   and are just provided for user information.


Input Example
--------------

For an example of the file see it on the GitHub repo:
`reeds_india_properties <https://github.com/NREL/Marmot/blob/main/input_files/reeds_india_properties.csv>`_




