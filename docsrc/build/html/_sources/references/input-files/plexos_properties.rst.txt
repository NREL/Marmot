.. raw:: html

   <script>
      var arr = document.getElementsByClassName('reference internal');
      for(var i = 0; i < arr.length; i++) {
      arr[i].innerHTML = arr[i].innerHTML.replace(/\./g, '.<wbr/>');
      }
   </script>

===============================
plexos_properties: csv file
===============================


The plexos_properties.csv file determines which PLEXOS properties to process with Marmot from the h5plexos results.

Set values to **TRUE** to process the specified property.

If a property you would like to process is not in this file, add it as a new row with the same format.

Input Definitions
-----------------
Columns
~~~~~~~~~
The plexos_properties.csv has 5 columns: 

- **data_type:**
   Specifies the interval of data to process. `interval` will always process the smallest interval data available 
   in the PLEXOS results set, e.g, 5 minute or hourly. Options available here depend on how the PLEXOS report setting 
   was defined. Common options include:

      - hour
      - month
      - year

- **group:** 
   The PLEXOS data class group, also used by Marmot to group property types.
   Valus should all be lower-case strings.
- **data_set:** 
   The property associated with the group, e.g, `generator Generation`. These names are equivalent to the 
   PLEXOS properties reported in the output file. 
- **collect_data:**
   Set to TRUE to process the property or FALSE to skip.
- **description:**
   Describes the property. 


Input Example
--------------

.. csv-table:: `plexos_properties <https://github.com/NREL/Marmot/blob/main/marmot/input_files/plexos_properties.csv>`_
   :file: ../../tables/plexos_properties_example.csv
   :widths: 10, 20, 20, 10, 40
   :header-rows: 1




