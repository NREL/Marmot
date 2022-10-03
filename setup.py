# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 13:32:13 2021

@author: Daniel Levie
"""

from setuptools import setup

setup(
    name="marmot",
    version_config=True,
    setup_requires=["setuptools-git-versioning"],
    author="Daniel Levie",
    author_email="daniel.levie@nrel.gov",
    description="A Python package to process and plot PLEXOS outputs",
    url="https://github.nrel.gov/PCM/Marmot",
    packages=[
        "marmot",
        "marmot.formatters",
        "marmot.metamanagers",
        "marmot.utils",
        "marmot.plottingmodules",
        "marmot.plottingmodules.plotutils",
    ],
    install_requires=[
        "h5py==2.10.0",
        "numpy",
        "pandas>=1.0.5",
        "tables",
        "PyYAML",
        "matplotlib>=3.1.0",
        "h5plexos @ git+https://github.com/NREL/h5plexos.git#egg=h5plexos",
    ],
)
