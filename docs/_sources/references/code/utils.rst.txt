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

Output Logs
~~~~~~~~~~~~~~

.. currentmodule:: marmot.utils.loggersetup
.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   SetupLogger

Definitions and Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: marmot.utils
.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   :recursive:

   definitions

.. currentmodule:: marmot.utils
.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   :recursive:

   mconfig

Data Input/Output
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: marmot.utils
.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   :recursive:

   dataio


Error and Exception Classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: marmot.utils.error_handler
.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   MissingH5PLEXOSDataError
   PropertyNotFound
   ConfigFileReadError
   ReEDSColumnLengthError
   ReEDSYearTypeConvertError

.. versionadded:: 0.11.0


