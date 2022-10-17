.. raw:: html

   <script>
      var arr = document.getElementsByClassName('reference internal');
      for(var i = 0; i < arr.length; i++) {
      arr[i].innerHTML = arr[i].innerHTML.replace(/\./g, '.<wbr/>');
      }
   </script>

=================================
siip_properties: csv file
=================================


The siip_properties.csv file determines which SIIP properties to process with Marmot.

Set values to **TRUE** to process the specified property.

If a property you would like to process is not in this file, add it as a new row with the same format.

Input Definitions
-----------------
Columns
~~~~~~~~~
The siip_properties.csv has 6 columns: 

- **data_type:**
   Specifies the interval of data to process. `interval` will always process the smallest interval data available. 
   Not currently fully utilized by the SIIP formatter so changing this value has no effect on the formatted data.
- **group:** 
   The data class group, used by Marmot to group property types.
   Valus should all be lower-case strings.
- **data_set:** 
   The property associated with the group, e.g, `generator generation_actual`. These names are equivalent to the 
   SIIP csv's located in the results folder. 
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
`siip_properties <https://github.com/NREL/Marmot/blob/main/input_files/siip_properties.csv>`_




