.. raw:: html

   <script>
      var arr = document.getElementsByClassName('reference internal');
      for(var i = 0; i < arr.length; i++) {
      arr[i].innerHTML = arr[i].innerHTML.replace(/\./g, '.<wbr/>');
      }
   </script>



marmot.formatters.formatreeds.ProcessReEDS
==========================================

.. currentmodule:: marmot.formatters.formatreeds

.. autoclass:: ProcessReEDS
   :members:
   :show-inheritance:

   
   
   .. rubric:: Methods

   .. autosummary::
   
        ~ProcessReEDS.df_process_generator
        ~ProcessReEDS.df_process_line
        ~ProcessReEDS.df_process_reserves_generators
        ~ProcessReEDS.get_processed_data
        ~ProcessReEDS.merge_timeseries_block_data
        ~ProcessReEDS.output_metadata
   
   

   
   
   .. rubric:: Attributes

   .. autosummary::
   
      ~ProcessReEDS.EXTRA_MARMOT_PROPERTIES
      ~ProcessReEDS.PROPERTY_MAPPING
      ~ProcessReEDS.UNITS_CONVERSION
      ~ProcessReEDS.data_collection
      ~ProcessReEDS.get_input_data_paths
      ~ProcessReEDS.input_folder
      ~ProcessReEDS.property_units
      ~ProcessReEDS.wind_resource_to_pca
   
   