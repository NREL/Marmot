.. raw:: html

    <style>
        h2  {border-bottom: 1px solid gray;}
    </style>

   <script>
      var arr = document.getElementsByClassName('reference internal');
      for(var i = 0; i < arr.length; i++) {
      arr[i].innerHTML = arr[i].innerHTML.replace(/\./g, '.<wbr/>');
      }
   </script>

========================
Running the Marmot code
========================

This is a short introduction to running Marmot. 
This guide will demonstrate how to run Marmot directly as an application, 
or import as a module into your own code. 

Runing Marmot as an application
---------------------------------

The most common way to use Marmot is by launching the code from the command 
line and treating it as an application.
Marmot uses a seperate launcher file for the formatter and plotting applications.

The files are located in the `Marmot/bin <https://github.com/NREL/Marmot/tree/main/bin>`_ directory.

.. note::
   To get any useful output, you will need to setup the :ref:`Marmot_user_defined_inputs: csv file` 
   first before starting Marmot. 
   If you are new to Marmot be sure to check out the :ref:`Getting-Started Tutorials<Tutorial: Formatting PLEXOS modelling results>`.

Start Marmot formatter
~~~~~~~~~~~~~~~~~~~~~~~
To start the Marmot formatter run the following in any terminal setup with python

1. First change directory to ``Marmot\bin``::

      cd E:\Marmot\bin

2. If needed activate your python environment::

      conda activate marmot-env10

3. Run the following to launch the program::

      python run_marmot_formatter.py


Start Marmot plotter
~~~~~~~~~~~~~~~~~~~~~
Similiarly to start the Marmot plotter run the following

1. Change directory to ``Marmot\bin``::

      cd E:\Marmot\bin

2. If needed activate your python environment::

      conda activate marmot-env10

3. Run the following to launch the program::

      python run_marmot_plotter.py


Importing Marmot as a module 
-----------------------------

Importing Marmot as a module involves interacting with the internal API code. 
The main access points being:

- :class:`marmot.marmot_h5_formatter.MarmotFormat` for the formatter.
- :class:`marmot.marmot_plot_main.MarmotPlot` for the plotter. 

To use Marmot as a python module that can be imported, it first needs to be made visible to the python package directory. 
This can be done two ways, either by pip installing Marmot (preferred method) or by adding the Marmot directory folder to the system path.

pip install Marmot
~~~~~~~~~~~~~~~~~~~~~

Open a cmd window that is setup with python and change directory to your desired install location.
Type the following::
   
   pip3 install git+https://github.com/NREL/Marmot.git#egg=marmot 
   
This will install Marmot from the current master branch, however this can be changed to a specific commit or 
tagged release if desired by adding ``@comit_id`` after ``Marmot.git`` and before the # symbol. 
For example, the following can be used to install Marmot release v0.11.0::

   git+https://github.com/NREL/Marmot.git@v0.11.0#egg=marmot 
   
If no error messages appeared, Marmot has been installed correctly. 

.. warning::
   It is recommended to install Marmot into a separate environment than the one used for running the 
   code as an application. This will prevent the python interpreter from choosing the incorrect Marmot instance.

Import the main classes
~~~~~~~~~~~~~~~~~~~~~~~~

To import the formatter or plotter, use the following import commands in a python session

.. code-block:: python

   from marmot.marmot_h5_formatter import MarmotFormat

   from marmot.marmot_plot_main import MarmotPlot

When importing Marmot directly, the :ref:`Marmot_user_defined_inputs: csv file` is not used. 
However, several other input files are still required which are passed in through the main class init methods in
:class:`marmot.marmot_h5_formatter.MarmotFormat` and :class:`marmot.marmot_plot_main.MarmotPlot`. 
