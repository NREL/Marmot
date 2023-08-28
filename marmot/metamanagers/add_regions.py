# -*- coding: utf-8 -*-
"""Add additional mapping categories to already formatted properties.

@author: Marty Schwarz, May 26th 2023
"""
import logging
import sys
from pathlib import Path

import h5py
import pandas as pd
import os

scen = 'Cap100'
dir = '/Users/mschwarz/BVRE'
prop = 'generator_Installed_Capacity'
mapping = pd.read_csv('/Users/mschwarz/Marmot_local/Marmot/input_files/mapping_folder/Region_mapping_ReEDS.csv')
cat = 'BA'

#def add_region_cat(scen,prop,cat):
mapper = mapping[['region',cat]]

for scen in ['Cap100', 'Cap100_MinCF1', 'Cap100_MinCF6']:
    print(scen)
    for prop in ['generator_Installed_Capacity','generator_Generation']:
        df = pd.read_hdf(os.path.join(dir,'Processed_HDF5_folder',f"{scen}_formatted.h5"),prop)
        df = df.reset_index()
        df = df.merge(mapper,on='region')
        index_cols = df.columns.drop('values').to_list()
        df.set_index(index_cols,inplace = True)
        df.to_hdf(os.path.join(dir,'Processed_HDF5_folder',f"{scen}_formatted.h5"),prop)
