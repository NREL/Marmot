
Tutorial: Formatting PLEXOS modelling results 
===============================================

This is an introductory tutorial on formatting PLEXOS modelling results with Marmot.
By the end of this tutorial you will have completed the following:

   1. Set up your input data folders.
   2. Set up the user input files required for formatting.
   3. Run the formatter and created Marmot formatted h5 files.

**Before you do anything else**, make sure you have installed Marmot and its prerequisites correctly 
by following the :ref:`Install the Marmot software` instructions.

Your input data and how to organize it 
-----------------------------------------

For this tutorial we will use example data located in the 
`Marmot/example-data <https://github.com/NREL/Marmot/tree/sphinx-docs/example-data>`_ folder.
This folder contains two sets of example data, 5 minute and hourly. 
We will work with the hourly data. 

Copy the **hourly** folder to a directory of your choosing.
Once you open the hourly data folder you will see the following sub-folders:

.. image:: ../images/hourly-data-folder.png

**Base DA** and **High VG DA** contain data that are grouped together under the same scenario.
The name of the folder is the scenario name and is an important identifier in Marmot.
Depending on how a user has setup their PLEXOS model, this folder should contain 1 or more files that are 
temporally disaggregated.

.. image:: ../images/h5-files.png

These scenario folders contain two files each, broken into monthly data sets. 
The files are h5 files and are what the Marmot formatter expects as an input. 
See `h5plexos <https://github.com/NREL/h5plexos>`_ for an explanation on how these files are created. 

Setting up the input files
----------------------------

To let Marmot know where our data is stored and which properties we would like to process,
we need to setup our input files. 





