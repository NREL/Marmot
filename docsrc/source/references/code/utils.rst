.. raw:: html

   <script>
      var arr = document.getElementsByClassName('reference internal');
      for(var i = 0; i < arr.length; i++) {
      arr[i].innerHTML = arr[i].innerHTML.replace(/\./g, '.<wbr/>');
      }
   </script>

=================
Utilities
=================

Utility classes and functions needed by Marmot.
Includes things such as logger setup and config file parsers.

.. currentmodule:: marmot.utils.loggersetup
.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   SetupLogger

.. currentmodule:: marmot.utils
.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   :recursive:

   definitions

.. versionadded:: 0.10.0

.. currentmodule:: marmot.utils
.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   :recursive:

   dataio

.. versionadded:: 0.11.0


.. currentmodule:: marmot.utils
.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   :recursive:

   mconfig
