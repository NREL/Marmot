.. raw:: html

   <script>
      var arr = document.getElementsByClassName('reference internal');
      for(var i = 0; i < arr.length; i++) {
      arr[i].innerHTML = arr[i].innerHTML.replace(/\./g, '.<wbr/>');
      }
   </script>
   
Mapping Folder
================

This section describes the Mapping Folder which contains csv files that allow the re-naming of generator 
and emission categories, setting the stack order that generators appear in plots, specifying new region aggregations,
and assigning colors to the different generator categories.

.. toctree::
   :maxdepth: 1

   gen_names
   ordered_gen_categories
   emit_names
   colour_dictionary
   region_mapping

