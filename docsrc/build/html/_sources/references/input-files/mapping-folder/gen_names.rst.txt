.. raw:: html

   <script>
      var arr = document.getElementsByClassName('reference internal');
      for(var i = 0; i < arr.length; i++) {
      arr[i].innerHTML = arr[i].innerHTML.replace(/\./g, '.<wbr/>');
      }
   </script>

======================
gen_names: csv file
======================


This file is used to rename existing generator technologies. 
For example, change all the possible "gas cc" generator names to "Gas-CC". 

Generator renaming takes place during the Marmot plotting process. 

Input Definitions
-----------------

Columns
~~~~~~~~~
The gen_names.csv has 2 columns: 

- **Original:** 
   The existing generator technology names. 
   This column should contain the name of every generator technology present in
   your simulation model. The Marmot Plotter will warn the 
   user if they are missing a category and rename any unidentified technologies 'Other'.
   Generators that are contained in this column and **not** present in your database will be ignored.

- **New:**
   The new names to assign to the existing generator technologies.

Input Example
--------------

.. csv-table:: `Gen names <https://github.com/NREL/Marmot/blob/main/input_files/mapping_folder/gen_names.csv>`_
   :file: ../../../tables/gen_names_example.csv
   :widths: 50,50
   :header-rows: 1



