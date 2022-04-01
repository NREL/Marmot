.. raw:: html

   <script>
      var arr = document.getElementsByClassName('reference internal');
      for(var i = 0; i < arr.length; i++) {
      arr[i].innerHTML = arr[i].innerHTML.replace(/\./g, '.<wbr/>');
      }
   </script>

======================
emit_names: csv file
======================

This file is used to change the name of the model emissions categories to be 
consistent across different models, it works in the same manner as the :ref:`gen_names: csv file`.

Emission category renaming takes place during the Marmot formatting process.

Input Definitions
-----------------

Columns
~~~~~~~~~
The gen_names.csv has 2 columns: 

- **Original:** 
   The existing emission category names. 
   This column should contain the name of every emission categorypresent in
   your simulation model. The Marmot Formatter will warn the 
   user if they are missing a category.
   Emission categories that are contained in this column and **not** present in your database will be ignored.

- **New:**
   The new names to assign to the existing emission categories.

Input Example
--------------

.. csv-table:: `emit_names <https://github.com/NREL/Marmot/blob/main/marmot/mapping_folder/emit_names.csv>`_
   :file: ../../../tables/emit_names_example.csv
   :widths: 50,50
   :header-rows: 1



