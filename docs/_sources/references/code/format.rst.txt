.. raw:: html

   <script>
      var arr = document.getElementsByClassName('reference internal');
      for(var i = 0; i < arr.length; i++) {
      arr[i].innerHTML = arr[i].innerHTML.replace(/\./g, '.<wbr/>');
      }
   </script>

=================
Formatters
=================

Model specific formatting classes to process input data.
Also includes the ExtraProperties class to create extra properties 
required by Marmots plotter.

.. currentmodule:: marmot.formatters.formatbase
.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   Process

.. currentmodule:: marmot.formatters.formatextra
.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   ExtraProperties
   ExtraPLEXOSProperties
   ExtraReEDSProperties
   ExtraReEDSIndiaProperties
   ExtraSIIProperties

.. versionadded:: 0.10.0
   
PLEXOS
~~~~~~~~~~~~~~~~~~

.. currentmodule:: marmot.formatters.formatplexos
.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   ProcessPLEXOS

ReEDS
~~~~~~~~~~~~~~~~~~
.. versionadded:: 0.10.0

.. currentmodule:: marmot.formatters.formatreeds
.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   ProcessReEDS
   ReEDSPropertyColumns

ReEDS India
~~~~~~~~~~~~~~~~~~
.. versionadded:: 0.11.0

.. currentmodule:: marmot.formatters.formatreeds_india
.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   ProcessReEDSIndia
   ReEDSIndiaPropertyColumns

SIIP
~~~~~~~~~~~~~~~~~~
.. versionadded:: 0.11.0

.. currentmodule:: marmot.formatters.formatsiip
.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   ProcessSIIP

EGRET
~~~~~~~~~~~~~~~~~~
.. currentmodule:: marmot.formatters.formategret
.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   ProcessEGRET




