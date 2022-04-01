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

Install the Marmot software
=============================

- Marmot requires Python 3.6 and the following minimum prerequisites to run:
  
  * hdf5=1.10.4 (Install with conda or download from `HDF group website <https://www.hdfgroup.org/downloads/hdf5>`_)
  * numpy
  * pandas>=1.0.5
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

   git clone --recurse-submodules git@github.com:NREL/Marmot.git --branch v0.10.0

You will need to have set up a public key using ``ssh-keygen`` and `added your public key to your GitHub account
<https://github.com/settings/ssh/new>`_ for this to work. Or, you can use HTTPS instead::

    git clone --recurse-submodules https://github.com/NREL/Marmot.git --branch v0.10.0

.. note::
   The Marmot PLEXOS formatter imports h5plexos. To avoid import and version errors, 
   h5plexos is included as a submodule in Marmot. If you already cloned the project and forgot ``--recurse-submodules``, 
   you can combine the git submodule init and git submodule update steps by running ``git submodule update --init``

Set up a conda environment
---------------------------

Setting up a new conda environment is the recommended route for running Marmot.
To ensure you are using all the required Python modules, create a new conda 
environment using the provided system specific `marmot-env yml file <https://github.com/NREL/Marmot/blob/main/marmot-env10.yml>`_, 
located in the Marmot repository.

.. note::
  The following example uses the provided **Windows** conda environment.

- To create and activate a new conda environment open a terminal and follow these steps:

   1. Create the environment from the ``marmot-env.yml`` file, if you are not in the Marmot directory, use the full file path to the file::

         conda env create -f marmot-env10.yml

   2. Activate the new environment::
   
         conda activate marmot-env10

   3. The required modules should now be ready to use, to verify the environment was setup correctly type::

         conda list

Alternative dependecies installation
---------------------------------------

If you prefer not to use Annaconda, dependecies can also be installed using the provided 
`requirements.txt <https://github.com/NREL/Marmot/blob/main/requirements.txt>`_ file. 
The text file contains all the python modules that are required by Marmot to run. 

To install from the file run the following from any terminal window that is setup with Python 
(e.g Git Bash, Anaconda Prompt)::

   pip install -r requirements.txt. 
   
If installing on a machine with restricted user rights adding ``--user`` to the command may be required.
