.. Marmot documentation master file, created by
   sphinx-quickstart on Tue Oct 12 13:36:33 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. raw:: html

    <style>
        .row {clear: both}
        h2  {border-bottom: 1px solid gray;}

        .column img {border: 1px solid gray;}

        @media only screen and (min-width: 1000px),
               only screen and (min-width: 500px) and (max-width: 768px){

            .column {
                padding-left: 5px;
                padding-right: 5px;
                float: left;
            }

            .column3  {
                width: 33.3%;
            }

            .column2  {
                width: 50%;
            }
        }
    </style>

    <script>
      var arr = document.getElementsByClassName('reference internal');
      for(var i = 0; i < arr.length; i++) {
      arr[i].innerHTML = arr[i].innerHTML.replace(/\./g, '.<wbr/>');
      }
    </script>



Marmot - an energy modelling results visualization tool 
========================================================

Marmot is a data formatting and visualization tool for PLEXOS production cost modelling results.
It provides an efficient way to view PLEXOS results by combing temporally disaggregated results 
and allowing the aggregation of different device types and modelling regions.

Marmot has approximately 100 pre-built plots which allow significant customization, allowing a
user to create publication figures and data-tables fast.

.. figure:: /images/3_Stacked_Gen_Facet_Grid_Peak_Demand.svg
    :align: center

    Stacked Generation Plot

.. figure:: /images/3_Total_Installed_Capacity_and_Generation_Facet.svg
    :align: center

    Total Installed Capacity and Total Generation Plot

If you are a new Marmot user, check out the **Get Started** section of the documentation for 
installation instructions and introductory tutorials. For more in depth explanations of various 
operations and details on Marmot, read our **How-to Guides**. If you would like to see some 
example plot outputs, you can view them in the **Plotting Examples Gallery**.

Contents
------------

.. rst-class:: clearfix row

.. rst-class:: column column2

:doc:`Get started <get-started/index>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install the software, introductory tutorials.

.. rst-class:: column column2

:doc:`How-to Guides <how-to/index>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Guides on using Marmot.

.. rst-class:: clearfix row

.. rst-class:: column column2

:doc:`Plotting Examples Gallery <plot-gallery/index>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Marmot output figures examples

.. rst-class:: column column2

:doc:`Input File References <references/input-files/files>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Guide to input files.


.. rst-class:: clearfix row

:doc:`Code References <references/code/modules>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Guide to key Python classes and functions.

.. rst-class:: column column2



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
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
