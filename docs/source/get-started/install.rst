.. raw:: html

    <style>
        h2  {border-bottom: 1px solid gray;}
    </style>


Install the Marmot software
=============================

- Marmot requires Python 3.6 and the following prerequisites to run:
  
  * hdf5>=1.10.4 (Install with conda or download from `HDF group website <https://www.hdfgroup.org/downloads/hdf5>`_)
  * numpy
  * pandas==1.0.5
  * PyYAML
  * h5py==2.10.0
  * matplotlib>=3.1.0
  * `h5plexos <https://github.com/NREL/h5plexos>`_ >=0.6
  * tables

Install system packages
-------------------------

Install the latest versions of `Anaconda <https://www.anaconda.com/products/individual>`_ and `Git <https://git-scm.com/>`_

Clone the Marmot repository
-----------------------------

Clone the most recent version of the ``Marmot`` repository::

   git clone --recurse-submodules git@github.com:NREL/Marmot.git --branch v0.8.0

You will need to have set up a public key using ``ssh-keygen`` and `added your public key to your GitHub account
<https://github.com/settings/ssh/new>`_ for this to work. Or, you can use HTTPS instead::

    git clone --recurse-submodules https://github.com/NREL/Marmot.git --branch v0.8.0

Set up a conda environment
---------------------------

Setting up a new conda environment is the recommended route for running Marmot.
To ensure you are using all the required Python modules, create a new conda 
environment using the provided `marmot-env yml file <https://github.com/NREL/Marmot/blob/main/marmot-env.yml>`_, 
located in the Marmot repository.

- To create and activate a new conda environment open a terminal and follow these steps:

   1. Create the environment from the ``marmot-env.yml`` file, if you are not in the Marmot directory, use the full file path to the file::

         conda env create -f marmot-env.yml

   2. Activate the new environment::
   
         conda activate marmot-env

   3. The required modules should now be ready to use, to verify the environment was setup correctly type::

         conda list


