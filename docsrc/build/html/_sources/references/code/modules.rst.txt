.. raw:: html

   <script>
      var arr = document.getElementsByClassName('reference internal');
      for(var i = 0; i < arr.length; i++) {
      arr[i].innerHTML = arr[i].innerHTML.replace(/\./g, '.<wbr/>');
      }
   </script>

Code Reference
===============


Initialization Code 
---------------------

.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   :recursive:

   marmot.marmot_h5_formatter
   marmot.marmot_plot_main

.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   :recursive:

   marmot.meta_data

Config Package
---------------------

.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   :recursive:

   marmot.config.mconfig

Plotting Modules Package 
-------------------------

.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   :recursive:

   marmot.plottingmodules.capacity_factor
   marmot.plottingmodules.capacity_out
   marmot.plottingmodules.curtailment
   marmot.plottingmodules.emissions
   marmot.plottingmodules.generation_stack
   marmot.plottingmodules.generation_unstack
   marmot.plottingmodules.hydro
   marmot.plottingmodules.prices
   marmot.plottingmodules.production_cost
   marmot.plottingmodules.ramping
   marmot.plottingmodules.reserves
   marmot.plottingmodules.sensitivities
   marmot.plottingmodules.storage
   marmot.plottingmodules.thermal_cap_reserve
   marmot.plottingmodules.total_generation
   marmot.plottingmodules.total_installed_capacity
   marmot.plottingmodules.transmission
   marmot.plottingmodules.unserved_energy
   marmot.plottingmodules.utilization_factor

Plot Utility Package 
-------------------------

.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   :recursive:

   marmot.plottingmodules.plotutils.plot_data_helper
   marmot.plottingmodules.plotutils.plot_exceptions
   marmot.plottingmodules.plotutils.plot_library



