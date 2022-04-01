.. raw:: html

   <script>
      var arr = document.getElementsByClassName('reference internal');
      for(var i = 0; i < arr.length; i++) {
      arr[i].innerHTML = arr[i].innerHTML.replace(/\./g, '.<wbr/>');
      }
   </script>

==================================
ordered_gen_categories: csv file
==================================

Ordered list of generators which determines how they appear in a stack plot.
Generator names should equal those in the :ref:`gen_names: csv file` **New** column.

Additional columns in the file allow plotting of only certain 
technologies when using the **TECH_SUBSET** entry in the 
:ref:`Marmot_user_defined_inputs: csv file`.
Users can change which technologies belong to the different technology categories 
by setting the values to *TRUE*. 

As well as being available to use by the **TECH_SUBSET** entry, 
the *vre* category is also used to determine which technologies 
should be included in Marmot's internal Curtailment property. 

The *re* and *pv* categories are also used by certain plots. Users are free to 
add their own technology categories by adding new columns to the file and 
setting which generators, they would like to include to *TRUE*.

.. versionchanged:: 0.9.0
   The additional columns added to the file now replace the previous **_gen_cat.csv* files.


Input Definitions
-----------------

Columns
~~~~~~~~~
The gen_names.csv has 5 required columns, 
new columns can be added to create new tech grouping: 

- **Ordered_Gen:**
   Names of the ordered generator technologies.
   The first row will be placed at the bottom of a generator stackplot or barplot.
   Generators that are contained in this column and **not** present in your database will be ignored.

- **vre:**
   Variable renewable generator technology categories. Used by Marmot to 
   calculate curtailment.
- **re:**
   Renewable generator technology categories.
- **pv:**
   PV	generator technology categories.
- **thermal:**
   Thermal generator technology categories.

Input Example
--------------

.. csv-table:: `ordered_gen_categories <https://github.com/NREL/Marmot/blob/main/marmot/input_files/mapping_folder/ordered_gen_categories.csv>`_
   :file: ../../../tables/ordered_gen_names_example.csv
   :widths: 40,15,15,15,15
   :header-rows: 1






