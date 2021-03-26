# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 13:32:13 2021

@author: Daniel Levie
"""

from setuptools import setup

setup(name='marmot',
      version='0.5.0',
      author='Daniel Levie',
      author_email='daniel.levie@nrel.gov',
      description='A Python package to process and plot PLEXOS outputs',
      url='https://github.nrel.gov/PCM/Marmot',
      packages=['marmot',
                'marmot.config',
                'marmot.plottingmodules'],
      install_requires=['numpy',
                        'pandas',
                        'PyYAML',
                        'h5py>=2.10.0',
                        'tables',
                        'matplotlib>=3.1.0'
                        'h5plexos @ git+https://github.com/NREL/h5plexos.git#egg=h5plexos']
      )