.. raw:: html

   <script>
      var arr = document.getElementsByClassName('reference internal');
      for(var i = 0; i < arr.length; i++) {
      arr[i].innerHTML = arr[i].innerHTML.replace(/\./g, '.<wbr/>');
      }
   </script>

=========================
Region Mapping: csv file
=========================

This file allows custom geographic aggregations to be created by grouping regions together.
The first column in the file should always be called *"region"* and should contain the names 
of the regions in your model database. 

Each subsequent column added defines a new type of region grouping. The names given to each of these
columns are user definable. 
The **AGG_BY** parameter in the :ref:`Marmot_user_defined_inputs: csv file` 
can then be used to plot this grouping.
In the example provided setting **AGG_BY** to 'Country' would group the data to this aggregation 
and create plots for *USA* and *Canada* 

The custom aggregations are merged into the data during formatting.

Input Example
--------------

.. csv-table:: `Regions Mapping <https://github.com/NREL/Marmot/blob/main/input_files/mapping_folder/Regions_Zones_Mapping_Full.csv>`_
   :file: ../../../tables/region_mapping_example.csv
   :widths: 25, 20, 10, 20, 25
   :header-rows: 1




