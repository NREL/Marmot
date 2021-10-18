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


Marmot - an energy modelling results visualization tool 
========================================================

Marmot is a data formatting and visualization tool for PLEXOS production cost modelling results.
It provides and efficient way to view PLEXOS results by combing temporally disaggregated results 
and allowing teh aggregation of different device types and modelling region.

Marmot has approximately 100 pre-built plots which allow significant customization to allow a
user to create publication figures and data-tables fast.

If you are a new Marmot user, check out the **Get Started** section of the documentation for 
installation instructions and introductory tutorials. For more in depth explanations of various 
operations and details on Marmot, read our **How-to Guides**. If you would like to see some 
example plot outputs, they can be viewed in the **Plotting Examples Gallery**.

Contents
------------

.. rst-class:: clearfix row

.. rst-class:: column column2

:doc:`Get started <get-started/index>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install the software, introductory tutorials

.. rst-class:: column column2

:doc:`How-to Guides <how-to/index>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Guides on using Marmot

.. rst-class:: clearfix row

.. rst-class:: column column2

:doc:`Plotting Examples Gallery <plot-gallery/index>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Marmot output figures examples

.. rst-class:: column column2

:doc:`References <references/index>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Guide to input files and key python classes and functions


.. rst-class:: clearfix row


.. toctree::
   :maxdepth: 1
   :caption: Contents:
   :hidden:

   get-started/index
   how-to/index
   plot-gallery/index
   references/index
   

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
