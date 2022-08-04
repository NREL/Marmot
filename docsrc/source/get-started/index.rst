
.. raw:: html

   <script>
      var arr = document.getElementsByClassName('reference internal');
      for(var i = 0; i < arr.length; i++) {
      arr[i].innerHTML = arr[i].innerHTML.replace(/\./g, '.<wbr/>');
      }
   </script>

===============
Getting Started
===============

Installation
------------

.. panels::
    :card: + install-card
    :column: col-12 p-3

   
    Quick Install from Repo
    ^^^^^^^^^^^^^^^^^^^^^^^

    The following command will clone the latest Marmot version.

    ++++

    .. code-block:: bash

        git clone --recurse-submodules https://github.com/NREL/Marmot.git --branch v0.11.0

    ---
    :column: col-12 p-3

    In-depth installation instructions?
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    Need more help in installation or setting up an environment? 
    Check the indepth installation page.

    .. link-button:: ./install.html
        :type: url
        :text: Learn more
        :classes: btn-secondary stretched-link


Getting Started Tutorials
-------------------------

.. panels::
    :card: + install-card
    :column: col-lg-6 col-md-6 col-sm-12 col-xs-12 p-3

    Format data
    ^^^^^^^^^^^^^
 
    An introductory tutorial on formatting data with Marmot.

    .. link-button:: ./format-tutorial.html
      :type: url
      :text: Learn more
      :classes: btn-secondary stretched-link

    ---

    Plotting data
    ^^^^^^^^^^^^^^^^^^

    An introductory tutorial on creating plots with Marmot.
   
    .. link-button:: ./plot-tutorial.html
      :type: url
      :text: Learn more
      :classes: btn-secondary stretched-link


.. toctree::
   :maxdepth: 1
   :hidden:

   install
   format-tutorial
   plot-tutorial

