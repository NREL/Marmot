.. Marmot documentation master file, created by
   sphinx-quickstart on Tue Oct 12 13:36:33 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. raw:: html



    <script>
      var arr = document.getElementsByClassName('reference internal');
      for(var i = 0; i < arr.length; i++) {
      arr[i].innerHTML = arr[i].innerHTML.replace(/\./g, '.<wbr/>');
      }
    </script>



Marmot - an energy modelling results visualization tool 
========================================================

Marmot is a data formatting and visualization tool for production cost and capacity expansion 
modelling results. 
It provides an efficient way to analysis data by combing temporally disaggregated results 
and allowing the aggregation of different device types and modelling regions.

Marmot currently supports analysis of `PLEXOS <https://www.energyexemplar.com/plexos>`_ and 
`SIIP <https://www.nrel.gov/analysis/siip.html>`_
production costs modelling results, and `ReEDS <https://www.nrel.gov/analysis/reeds/>`_ capacity expansion results.

Marmot has approximately 120 pre-built plots which allow significant customization, allowing a
user to create publication figures and data-tables fast.

.. figure:: /images/3_Stacked_Gen_Facet_Grid_Peak_Demand.svg
    :align: center

    Stacked Generation Plot

If you are a new Marmot user, check out the **Getting Started** section of the documentation for 
installation instructions and introductory tutorials. For more in depth explanations of various 
operations and details on Marmot, read our **How-to Guides**. If you would like to see some 
example plot outputs, you can view them in the **Plotting Examples Gallery**.


.. panels::
    :card: + intro-card text-center
    :column: col-lg-6 col-md-6 col-sm-6 col-xs-12 d-flex

    ---

    Getting started
    ^^^^^^^^^^^^^^^

    Install the software, introductory tutorials.

    +++

    .. link-button:: ./get-started/index.html
            :type: url
            :text: To the getting started guides
            :classes: btn-block btn-secondary stretched-link

    ---

    How-to Guides
    ^^^^^^^^^^^^^^^

    Guides on using Marmot.

    +++

    .. link-button:: ./how-to/index.html
            :type: url
            :text: To the how-to guides
            :classes: btn-block btn-secondary stretched-link

    ---

    Plotting Examples Gallery
    ^^^^^^^^^^^^^^^^^^^^^^^^^^

    Marmot output figures examples

    +++

    .. link-button:: ./plot-gallery/index.html
            :type: url
            :text: To the plotting gallery
            :classes: btn-block btn-secondary stretched-link

    ---

    Input File References
    ^^^^^^^^^^^^^^^^

    Guide to input files.

    +++

    .. link-button:: ./references/input-files/files.html
            :type: url
            :text: To the input file reference
            :classes: btn-block btn-secondary stretched-link

    ---


    Code References
    ^^^^^^^^^^^^^^^^

    Guide to code api.

    +++

    .. link-button:: ./references/code/modules.html
            :type: url
            :text: To the code reference api
            :classes: btn-block btn-secondary stretched-link





.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :hidden:

   get-started/index
   how-to/index
   plot-gallery/index
   references/input-files/files
   references/code/modules
   

Indices and tables
~~~~~~~~~~~~~~~~~~

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
