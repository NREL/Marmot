.. raw:: html

   <script>
      var arr = document.getElementsByClassName('reference internal');
      for(var i = 0; i < arr.length; i++) {
      arr[i].innerHTML = arr[i].innerHTML.replace(/\./g, '.<wbr/>');
      }
   </script>

============================
colour_dictionary: csv file
============================

This file is used to assign a color to generation technology types 
e.g, Gas-CC, Wind, PV etc. The names in the **Generator** column should equal those in 
the :ref:`gen_names: csv file`. **New** column.

Input Definitions
-----------------

Columns
~~~~~~~~~
The colour_dictionary.csv has 2 columns: 

- **Generator:** 
   The names of the generators. 
   Generators that are contained in this column and **not** present in your database will be ignored.

- **Colour:**
   The color identifier, this can either be a string name, e.g, `red` or a 
   hex color, e.g, `#820000`

Input Example
--------------

.. csv-table:: `colour_dictionary <https://github.com/NREL/Marmot/blob/main/input_files/mapping_folder/colour_dictionary.csv>`_
   :file: ../../../tables/color_example.csv
   :widths: 50,50
   :header-rows: 1

