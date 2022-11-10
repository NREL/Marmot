.. raw:: html

   <script>
      var arr = document.getElementsByClassName('reference internal');
      for(var i = 0; i < arr.length; i++) {
      arr[i].innerHTML = arr[i].innerHTML.replace(/\./g, '.<wbr/>');
      }
   </script>

=================
Plot Utilities
=================

Classes to assist in creating plots. Used extensively by the Plot classes.

.. currentmodule:: marmot.plottingmodules.plotutils.plot_data_helper
.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   PlotDataStoreAndProcessor

.. currentmodule:: marmot.plottingmodules.plotutils.plot_library
.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   SetupSubplot
   PlotLibrary

.. currentmodule:: marmot.plottingmodules.plotutils
.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   :recursive:

   timeseries_modifiers

.. currentmodule:: marmot.plottingmodules.plotutils
.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   :recursive:

   styles

.. versionadded:: 0.11.0

.. currentmodule:: marmot.plottingmodules.plotutils
.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   :recursive:

   plot_exceptions

