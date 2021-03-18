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
                'marmot.plottingmoduls'],
      install_requires=['h5py',
                        'pandas',
                        'yaml',
                        'h5plexos @ git+https://github.com/NREL/h5plexos.git#egg=h5plexos']
      )