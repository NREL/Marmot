# -*- coding: utf-8 -*- 
"""Main formatting source code to format modelling results for plotting.

This code was orginally written to process PLEXOS HDF5 outputs to get them ready for plotting,
but has since been expanded to allow class additions to process results from any energy 
simulation model. 
Once the data is processed it is outputted as an intermediary HDF5 file format so that
it can be read into the marmot_plot_main.py file

@author: Daniel Levie
"""
# =======================================================================================
# Import Python Libraries
# =======================================================================================

import os
import sys
import pathlib

FILE_DIR = pathlib.Path(__file__).parent.absolute() # Location of this module
if __name__ == '__main__':  # Add Marmot directory to sys path if running from __main__
    if os.path.dirname(os.path.dirname(__file__)) not in sys.path:
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        os.chdir(pathlib.Path(__file__).parent.absolute().parent.absolute())
import time
import re
import logging
import logging.config
import pandas as pd
import h5py
import yaml
from typing import Union
import json

try:
    from marmot.meta_data import MetaData
except ModuleNotFoundError:
    print("Attempted import of Marmot as a module from a Git directory. "
          "Import of Marmot will not function in this way. "
          "To import Marmot as a module use the preferred method of pip "
          "installing Marmot, or add the Marmot directory to the system path, "
          "see ReadME for details.\nSystem will now exit")
    sys.exit()
import marmot.config.mconfig as mconfig

# Import as Submodule
try:
    from h5plexos.query import PLEXOSSolution
except ModuleNotFoundError:
    from marmot.h5plexos.h5plexos.query import PLEXOSSolution


# A bug in pandas requires this to be included,
# otherwise df.to_string truncates long strings. Fix available in Pandas 1.0
# but leaving here in case user version not up to date
pd.set_option("display.max_colwidth", 1000)

# Conversion units dict, key values is a tuple of new unit name and 
# conversion multiplier
UNITS_CONVERSION = {
                    'kW': ('MW', 1e-3),
                    'MW': ('MW', 1),
                    'GW': ('MW', 1e3),
                    'TW': ('MW', 1e6),
                    'kWh': ('MWh', 1e-3),
                    'MWh': ('MWh', 1),
                    'GWh': ('MWh', 1e3),
                    'TWh': ('MWh', 1e6),
                    'lb': ('kg', 0.453592),
                    'ton': ('kg', 907.18474),
                    'kg': ('kg', 1),
                    'tonne': ('kg', 1000),
                    '$': ('$', 1),
                    '$000': ('$', 1000),
                    'h': ('h', 1),
                    'MMBTU': ('MMBTU', 1),
                    'GBTU': ('MMBTU', 1000),
                    'GJ"': ('MMBTU', 0.947817),
                    'TJ': ('MMBTU', 947.817120),
                    '$/MW': ('$/MW', 1),
                    'lb/MWh' : ('kg/MWh', 0.453592),
                    'Kg/MWh': ('Kg/MWh', 1)
                    }


class SetupLogger():
    """Sets up the python logger.

    This class handles the following.

    1. Configures logger from marmot_logging_config.yml file.
    2. Handles rollover of log file on each instantiation.
    3. Sets log_directory.
    4. Append optional suffix to the end of the log file name

    Optional suffix is useful when running multiple processes in parallel to 
    allow logging to separate files.
    """

    def __init__(self, log_directory: str = 'logs', 
                 log_suffix: str = None):
        """
        Args:
            log_directory (str, optional): log directory to save logs. 
                Defaults to 'logs'.
            log_suffix (str, optional): Optional suffix to add to end of log file. 
                Defaults to None.
        """
        if log_suffix is None:
            self.log_suffix = ''
        else:
             self.log_suffix = f'_{log_suffix}'

        current_dir = os.getcwd()
        os.chdir(FILE_DIR)

        try:
            os.makedirs(log_directory)
        except FileExistsError:
            # log directory already exists
            pass

        with open('config/marmot_logging_config.yml', 'rt') as f:
            conf = yaml.safe_load(f.read())
            conf['handlers']['warning_handler']['filename'] = \
                (conf['handlers']['warning_handler']['filename']
                .format(log_directory, 'formatter', self.log_suffix))
            conf['handlers']['info_handler']['filename'] = \
                (conf['handlers']['info_handler']['filename']
                .format(log_directory, 'formatter', self.log_suffix))

            logging.config.dictConfig(conf)

        self.logger = logging.getLogger('marmot_format')
        # Creates a new log file for next run
        self.logger.handlers[1].doRollover()
        self.logger.handlers[2].doRollover()

        os.chdir(current_dir)


class Process():
    """Base class for processing simulation model data.
    """

    def __init__(self, input_folder: str, Region_Mapping: pd.DataFrame, 
                 emit_names: pd.DataFrame, logger: logging.Logger, **kwargs):
        """
        Args:
            input_folder (str): Folder containing model input files.
            Region_Mapping (pd.DataFrame): DataFrame to map custom 
                regions/zones to create custom aggregations.
            emit_names (pd.DataFrame): DataFrame with 2 columns to rename 
                emission names.
            logger (logging.Logger): logger object from SetupLogger.
        """
        self.input_folder = input_folder
        self.Region_Mapping = Region_Mapping
        self.emit_names = emit_names
        self.logger = logger

        if not self.emit_names.empty:
            self.emit_names_dict = (self.emit_names[['Original', 'New']]
                                        .set_index("Original").to_dict()["New"])

    def get_input_files(self) -> list:
        """Gets a list of input files within the scenario folders
        """
        startdir = os.getcwd()
        os.chdir(self.input_folder)

        files = []
        for names in os.listdir():
            files.append(names)  

        # List of all files in input folder in alpha numeric order
        files_list = sorted(files, key=lambda x:int(re.sub('\D', '', x)))

        os.chdir(startdir)

        return files_list

    def output_metadata(self, files_list: list, output_file_path: str) -> None:
        pass

    def get_processed_data(self, prop_class: str, property: str, 
                           timescale: str, model_filename: str) -> pd.DataFrame:
        pass

    def report_prop_error(self, property: str, 
                          prop_class: str) -> pd.DataFrame:
        """Outputs a warning message when the get_processed_data method
        cannot find the specified property in the simulation model solution files.

        Args:
            property (str): property e.g Max Capacity, Generation etc.
            prop_class (str): property class e.g Region, Generator, Zone etc.

        Returns:
            pd.DataFrame: Empty DataFrame.
        """
        self.logger.warning(f'CAN NOT FIND "{prop_class} {property}". ' 
                            f'"{property}" DOES NOT EXIST')
        self.logger.info('SKIPPING PROPERTY\n')
        df = pd.DataFrame()
        return df


class ProcessPLEXOS(Process):
    """Process PLEXOS class specific data from a h5plexos database.
    """
    def __init__(self, input_folder: str, Region_Mapping: pd.DataFrame, 
                *args, plexos_block: str ='ST', **kwargs):
        """
        Args:
            input_folder (str): Folder containing h5plexos h5 files.
            Region_Mapping (pd.DataFrame): DataFrame to map custom 
                regions/zones to create custom aggregations.
            plexos_block (str, optional): PLEXOS results type. Defaults to 'ST'.
        """
        self.plexos_block = plexos_block
        self.hdf5_collection = {}
        self.metadata = MetaData(input_folder, read_from_formatted_h5=False, 
                                 Region_Mapping=Region_Mapping)
        # Instantiation of Process Base class
        super().__init__(input_folder, Region_Mapping, *args, **kwargs) 
        
    def get_input_files(self) -> list:
        """Gets a list of h5plexos input files within the scenario folders

        Returns:
            list: list of h5plexos input filenames to process
        """
        startdir = os.getcwd()
        os.chdir(self.input_folder) 
        
        files = []
        for names in os.listdir():
            if names.endswith(".h5"):
                files.append(names)  # Creates a list of only the hdf5 files

        # List of all hf files in hdf5 folder in alpha numeric order
        files_list = sorted(files, key=lambda x:int(re.sub('\D', '', x)))
        os.chdir(startdir)

        # Read in all HDF5 files into dictionary
        self.logger.info("Loading all HDF5 files to prepare for processing")
        regions = set()
        for file in files_list:                
            self.hdf5_collection[file] = PLEXOSSolution(os.path.join(self.input_folder, 
                                                                     file))
            if not self.Region_Mapping.empty:
                regions.update(list(self.metadata.regions(file)['region']))

        if not self.Region_Mapping.empty:
            if regions.issubset(self.Region_Mapping['region']) is False:
                missing_regions = list(regions - set(self.Region_Mapping['region']))
                self.logger.warning("The Following PLEXOS REGIONS are missing from "
                                    "the 'region' column of your mapping file: "
                                    f"{missing_regions}\n")
        return files_list

    def output_metadata(self, files_list: list, output_file_path: str) -> None:
        """Transfers metadata from original PLEXOS solutions file to processed HDF5 file.  
        
        For each partition in a given scenario, the metadata from that partition 
        is copied over and saved in the processed output file.

        Args:
            files_list (list): List of all h5 files in hdf5 folder in alpha numeric order
            output_file_path (str): Location of formatted output h5 file 
        """
        for partition in files_list:
            f = h5py.File(os.path.join(self.input_folder, partition),'r')
            meta_keys = [key for key in f['metadata'].keys()]

            group_dict = {}
            for key in meta_keys:
                sub_dict = {}
                subkeys = [key for key in f['metadata'][key].keys()]
                for sub in subkeys:
                    dset = f['metadata'][key][sub]
                    sub_dict[sub] = dset
                group_dict[key] = sub_dict

            with h5py.File(output_file_path,"a") as g:
                # check if metadata group already exists
                existing_groups = [key for key in g.keys()]
                if 'metadata' not in existing_groups:
                    grp = g.create_group('metadata')
                else:
                    grp = g['metadata']

                partition_group = grp.create_group(partition)
                for key in list(group_dict.keys()):
                    subgrp = partition_group.create_group(key)
                    s_dict = group_dict[key]
                    for key2 in list(s_dict.keys()):
                        dset = s_dict[key2]
                        subgrp.create_dataset(name=key2, data=dset)
            f.close()

    def get_processed_data(self, plexos_class: str, plexos_prop: str, 
                  timescale: str, model_filename: str) -> pd.DataFrame:
        """Handles the pulling of data from the h5plexos hdf5
        file and then passes the data to one of the formating functions

        Args:
            plexos_class (str): PLEXOS class e.g Region, Generator, Zone etc
            plexos_prop (str): PLEXOS property e.g Max Capacity, Generation etc.
            timescale (str): Data timescale, e.g Hourly, Monthly, 5 minute etc.
            model_filename (str): name of model to process.

        Returns:
            pd.DataFrame: Formatted results dataframe.
        """
        db = self.hdf5_collection.get(model_filename)
        try:
            if "_" in plexos_class:
                df = db.query_relation_property(plexos_class, 
                                                plexos_prop, 
                                                timescale=timescale,
                                                phase=self.plexos_block)
                object_class = plexos_class
            else:
                df = db.query_object_property(plexos_class, 
                                              plexos_prop, 
                                              timescale=timescale,
                                              phase=self.plexos_block)
                if ((0,6,0) <= db.version and db.version < (0,7,0)):
                    object_class = f"{plexos_class}s"
                else:
                    object_class = plexos_class

        except (ValueError, KeyError):
            df = self.report_prop_error(plexos_prop, plexos_class)
            return df
        
        if self.plexos_block != 'ST':
            df = self.merge_timeseries_block_data(db, df)

        # handles h5plexos naming discrepency 
        if ((0,6,0) <= db.version and db.version < (0,7,0)):
            # Get original units from h5plexos file 
            df_units = (db.h5file[f'/data/{self.plexos_block}/{timescale}'
                                  f'/{object_class}/{plexos_prop}'].attrs['units']
                                                                   .decode('UTF-8'))
        else:
            df_units = (db.h5file[f'/data/{self.plexos_block}/{timescale}'
                                  f'/{object_class}/{plexos_prop}'].attrs['unit'])
        # find unit conversion values
        converted_units = UNITS_CONVERSION.get(df_units, (df_units, 1))

        # Get desired method
        process_att = getattr(self, f'df_process_{plexos_class}')
        # Process attribute and return to df
        df = process_att(df, model_filename)
        
        # Convert units and add unit column to index 
        df = df*converted_units[1]
        units_index = pd.Index([converted_units[0]] *len(df), name='units')
        df.set_index(units_index, append=True, inplace=True)

        if plexos_class == 'region' and \
           plexos_prop == "Unserved Energy" and \
           int(df.sum(axis=0)) > 0:
            self.logger.warning(f"Scenario contains Unserved Energy: "
                                f"{int(df.sum(axis=0))} MW\n")
        return df

    def merge_timeseries_block_data(self, db: PLEXOSSolution, 
                                    df: pd.DataFrame) -> pd.DataFrame:
        """Merge timeseries and block data found in LT, MT and PASA results

        Args:
            db (PLEXOSSolution): PLEXOSSolution instance for specific h5plexos file.
            df (pd.DataFrame): h5plexos dataframe 

        Returns:
            pd.DataFrame: df with merged in timeseries data 
        """

        block_mapping = db.blocks[self.plexos_block]
        block_mapping.index.rename('timestamp', inplace=True)
        df = df.reset_index(level='block')

        merged_data = df.reset_index().merge(block_mapping.reset_index(), on='block')
        merged_data.drop('block', axis=1, inplace=True)
        index_cols = list(merged_data.columns)
        index_cols.remove(0)
        merged_data.set_index(index_cols, inplace=True)
        return merged_data

    def df_process_generator(self, df: pd.DataFrame, 
                             model_filename: str) -> pd.DataFrame:
        """Format PLEXOS Generator Class data.

        Args:
            df (pd.DataFrame): h5plexos dataframe to process
            model_filename (str): name of h5plexos h5 file being processed 

        Returns:
            pd.DataFrame: Processed output, single value column with multiindex.
        """
        df = df.droplevel(level=["band", "property"])
        df.index.rename(['tech', 'gen_name'], 
                        level=['category', 'name'], 
                        inplace=True)

        region_gen_cat_meta = self.metadata.region_generator_category(model_filename)
        zone_gen_cat_meta = self.metadata.zone_generator_category(model_filename)
        timeseries_len = len(df.index.get_level_values('timestamp').unique())

        if region_gen_cat_meta.empty is False:
            region_gen_idx = pd.CategoricalIndex(region_gen_cat_meta
                                                 .index.get_level_values(0))

            region_gen_idx = region_gen_idx.repeat(timeseries_len)

            idx_region = pd.MultiIndex(levels=df.index.levels 
                                       + [region_gen_idx.categories],
                                       codes=df.index.codes 
                                       + [region_gen_idx.codes],
                                       names=df.index.names 
                                       + region_gen_idx.names)
        else:
            idx_region = df.index

        if zone_gen_cat_meta.empty is False:
            zone_gen_idx = pd.CategoricalIndex(zone_gen_cat_meta
                                               .index.get_level_values(0))

            zone_gen_idx = zone_gen_idx.repeat(timeseries_len)

            idx_zone = pd.MultiIndex(levels=idx_region.levels 
                                     + [zone_gen_idx.categories],
                                     codes=idx_region.codes 
                                     + [zone_gen_idx.codes],
                                     names=idx_region.names 
                                     + zone_gen_idx.names)
        else:
            idx_zone = idx_region

        if not self.Region_Mapping.empty:
            region_gen_mapping_idx = pd.MultiIndex.from_frame(region_gen_cat_meta
                                                  .merge(self.Region_Mapping,
                                                         how="left",
                                                         on='region')
                                                  .sort_values(by=['tech', 'gen_name'])
                                                  .drop(['region', 'tech', 'gen_name'], 
                                                        axis=1))

            region_gen_mapping_idx = region_gen_mapping_idx.repeat(timeseries_len)

            idx_map = pd.MultiIndex(levels=idx_zone.levels 
                                    + region_gen_mapping_idx.levels,
                                    codes=idx_zone.codes 
                                    + region_gen_mapping_idx.codes,
                                    names=idx_zone.names 
                                    + region_gen_mapping_idx.names)
        else:
            idx_map = idx_zone

        df = pd.DataFrame(data=df.values.reshape(-1), index=idx_map)
        # Gets names of all columns in df and places in list
        df_col = list(df.index.names)
        # move timestamp to start of df
        df_col.insert(0, df_col.pop(df_col.index("timestamp")))  
        df = df.reorder_levels(df_col, axis=0)
        df[0] = pd.to_numeric(df[0], downcast='float')

        return df

    def df_process_region(self, df: pd.DataFrame, 
                          model_filename: str) -> pd.DataFrame:
        """Format PLEXOS Region Class data.

        Args:
            df (pd.DataFrame): h5plexos dataframe to process
            model_filename (str): name of h5plexos h5 file being processed 

        Returns:
            pd.DataFrame: Processed output, single value column with multiindex.
        """
        df = df.droplevel(level=["band", "property", "category"])
        df.index.rename('region', level='name', inplace=True)

        timeseries_len = len(df.index.get_level_values('timestamp').unique())

        # checks if Region_Mapping contains data to merge, skips if empty
        if not self.Region_Mapping.empty:  
            mapping_idx = pd.MultiIndex.from_frame(self.metadata
                                       .regions(model_filename)
                                       .merge(self.Region_Mapping,
                                              how="left",
                                              on='region')
                                       .drop(['region', 'category'], axis=1))

            mapping_idx = mapping_idx.repeat(timeseries_len)

            idx = pd.MultiIndex(levels=df.index.levels 
                                + mapping_idx.levels,
                                codes=df.index.codes 
                                + mapping_idx.codes,
                                names=df.index.names 
                                + mapping_idx.names)
        else:
            idx = df.index

        df = pd.DataFrame(data=df.values.reshape(-1), index=idx)
        df_col = list(df.index.names)
        df_col.insert(0, df_col.pop(df_col.index("timestamp")))
        df = df.reorder_levels(df_col, axis=0)
        df[0] = pd.to_numeric(df[0], downcast='float')
        return df

    def df_process_zone(self, df: pd.DataFrame, 
                        model_filename: str) -> pd.DataFrame:
        """Format PLEXOS Zone Class data.

        Args:
            df (pd.DataFrame): h5plexos dataframe to process
            model_filename (str): name of h5plexos h5 file being processed 

        Returns:
            pd.DataFrame: Processed output, single value column with multiindex.
        """
        df = df.droplevel(level=["band", "property", "category"])
        df.index.rename('zone', level='name', inplace=True)
        df = pd.DataFrame(data=df.values.reshape(-1), index=df.index)
        df_col = list(df.index.names)  #
        df_col.insert(0, df_col.pop(df_col.index("timestamp")))
        df = df.reorder_levels(df_col, axis=0)
        df[0] = pd.to_numeric(df[0], downcast='float')
        return df

    def df_process_line(self, df: pd.DataFrame, 
                        model_filename: str) -> pd.DataFrame:
        """Format PLEXOS Line Class data.

        Args:
            df (pd.DataFrame): h5plexos dataframe to process
            model_filename (str): name of h5plexos h5 file being processed 

        Returns:
            pd.DataFrame: Processed output, single value column with multiindex.
        """
        df = df.droplevel(level=["band", "property", "category"])
        df.index.rename('line_name', level='name', inplace=True)
        df = pd.DataFrame(data=df.values.reshape(-1), index=df.index)
        df_col = list(df.index.names)  
        df_col.insert(0, df_col.pop(df_col.index("timestamp"))) 
        df = df.reorder_levels(df_col, axis=0)
        df[0] = pd.to_numeric(df[0], downcast='float')
        return df

    def df_process_interface(self, df: pd.DataFrame, 
                             model_filename: str) -> pd.DataFrame:
        """Format PLEXOS PLEXOS Interface Class data.

        Args:
            df (pd.DataFrame): h5plexos dataframe to process
            model_filename (str): name of h5plexos h5 file being processed 

        Returns:
            pd.DataFrame: Processed output, single value column with multiindex.
        """
        df = df.droplevel(level=["band", "property"])
        df.index.rename(['interface_name', 'interface_category'], 
                            level=['name', 'category'], inplace=True)
        df = pd.DataFrame(data=df.values.reshape(-1), index=df.index)
        df_col = list(df.index.names)  
        df_col.insert(0, df_col.pop(df_col.index("timestamp"))) 
        df = df.reorder_levels(df_col, axis=0)
        df[0] = pd.to_numeric(df[0], downcast='float')
        return df

    def df_process_reserve(self, df: pd.DataFrame, 
                           model_filename: str) -> pd.DataFrame:
        """Format PLEXOS Reserve Class data.

        Args:
            df (pd.DataFrame): h5plexos dataframe to process
            model_filename (str): name of h5plexos h5 file being processed 

        Returns:
            pd.DataFrame: Processed output, single value column with multiindex.
        """
        df = df.droplevel(level=["band", "property"])
        df.index.rename(['parent', 'Type'], level=['name', 'category'], 
                        inplace=True)
        df = df.reset_index()  # unzip the levels in index
        if self.metadata.reserves_regions(model_filename).empty is False:
            # Merges in regions where reserves are located
            df = df.merge(self.metadata.reserves_regions(model_filename), 
                            how='left', on='parent')

        if self.metadata.reserves_zones(model_filename).empty is False:
            # Merges in zones where reserves are located
            df = df.merge(self.metadata.reserves_zones(model_filename), 
                            how='left', on='parent')  
        df_col = list(df.columns)  
        df_col.remove(0)
        # move timestamp to start of df
        df_col.insert(0, df_col.pop(df_col.index("timestamp")))  
        df.set_index(df_col, inplace=True)
        df[0] = pd.to_numeric(df[0], downcast='float')
        return df

    def df_process_reserves_generators(self, df: pd.DataFrame, 
                                       model_filename: str) -> pd.DataFrame:
        """Format PLEXOS Reserve_Generators Relational Class data.

        Args:
            df (pd.DataFrame): h5plexos dataframe to process
            model_filename (str): name of h5plexos h5 file being processed 

        Returns:
            pd.DataFrame: Processed output, single value column with multiindex.
        """
        df = df.droplevel(level=["band", "property"])
        df.index.rename(['gen_name'], level=['child'], inplace=True)
        df = df.reset_index()  # unzip the levels in index
        df = df.merge(self.metadata.generator_category(model_filename), 
                        how='left', on='gen_name')

        # merging in generator region/zones first prevents double 
        # counting in cases where multiple model regions are within a reserve region
        if self.metadata.region_generators(model_filename).empty is False:
            df = df.merge(self.metadata.region_generators(model_filename), 
                            how='left', on='gen_name')
        if self.metadata.zone_generators(model_filename).empty is False:
            df = df.merge(self.metadata.zone_generators(model_filename), 
                            how='left', on='gen_name')

        # now merge in reserve regions/zones
        if self.metadata.reserves_regions(model_filename).empty is False:
            # Merges in regions where reserves are located
            df = df.merge(self.metadata.reserves_regions(model_filename), 
                            how='left', on=['parent', 'region'])  
        if self.metadata.reserves_zones(model_filename).empty is False:
            # Merges in zones where reserves are located
            df = df.merge(self.metadata.reserves_zones(model_filename), 
                            how='left', on=['parent', 'zone'])  

        df_col = list(df.columns) 
        df_col.remove(0)
        df_col.insert(0, df_col.pop(df_col.index("timestamp")))
        df.set_index(df_col, inplace=True)
        df[0] = pd.to_numeric(df[0], downcast='float')
        return df

    def df_process_fuel(self, df: pd.DataFrame, 
                        model_filename: str) -> pd.DataFrame:
        """Format PLEXOS Fuel Class data.

        Args:
            df (pd.DataFrame): h5plexos dataframe to process
            model_filename (str): name of h5plexos h5 file being processed 

        Returns:
            pd.DataFrame: Processed output, single value column with multiindex.
        """
        df = df.droplevel(level=["band", "property", "category"])
        df.index.rename('fuel_type', level='name', inplace=True)
        df = pd.DataFrame(data=df.values.reshape(-1), index=df.index)
        df_col = list(df.index.names)
        df_col.insert(0, df_col.pop(df_col.index("timestamp")))
        df = df.reorder_levels(df_col, axis=0)
        df[0] = pd.to_numeric(df[0], downcast='float')
        return df

    def df_process_constraint(self, df: pd.DataFrame, 
                              model_filename: str) -> pd.DataFrame:
        """Format PLEXOS Constraint Class data.

        Args:
            df (pd.DataFrame): h5plexos dataframe to process
            model_filename (str): name of h5plexos h5 file being processed 

        Returns:
            pd.DataFrame: Processed output, single value column with multiindex.
        """
        df = df.droplevel(level=["band", "property"])
        df.index.rename(['constraint_category', 'constraint'], 
                        level=['category', 'name'], inplace=True)
        df = pd.DataFrame(data=df.values.reshape(-1), index=df.index)
        df_col = list(df.index.names)
        df_col.insert(0, df_col.pop(df_col.index("timestamp")))
        df = df.reorder_levels(df_col, axis=0)
        df[0] = pd.to_numeric(df[0], downcast='float')
        return df

    def df_process_emission(self, df: pd.DataFrame, 
                            model_filename: str) -> pd.DataFrame:
        """Format PLEXOS Emission Class data.

        Args:
            df (pd.DataFrame): h5plexos dataframe to process
            model_filename (str): name of h5plexos h5 file being processed 

        Returns:
            pd.DataFrame: Processed output, single value column with multiindex.
        """
        df = df.droplevel(level=["band", "property"])
        df.index.rename('emission_type', level='name', inplace=True)
        df = pd.DataFrame(data=df.values.reshape(-1), index=df.index)
        df_col = list(df.index.names)
        df_col.insert(0, df_col.pop(df_col.index("timestamp")))
        df = df.reorder_levels(df_col, axis=0)
        df[0] = pd.to_numeric(df[0], downcast='float')
        return df

    def df_process_emissions_generators(self, df: pd.DataFrame, 
                                        model_filename: str) -> pd.DataFrame:
        """Format PLEXOS Emissions_Generators Relational Class data.

        Args:
            df (pd.DataFrame): h5plexos dataframe to process
            model_filename (str): name of h5plexos h5 file being processed 

        Returns:
            pd.DataFrame: Processed output, single value column with multiindex.
        """
        df = df.droplevel(level=["band", "property"])
        df.index.rename(['gen_name'], level=['child'], inplace=True)
        df.index.rename(['pollutant'], level=['parent'], inplace=True)

        df = df.reset_index()  # unzip the levels in index
        # merge in tech information
        df = df.merge(self.metadata.generator_category(model_filename), 
                        how='left', on='gen_name') 
        # merge in region and zone information
        if self.metadata.region_generator_category(model_filename).empty is False:
            # merge in region information
            df = df.merge(self.metadata
                              .region_generator_category(model_filename)
                              .reset_index(), 
                          how='left', 
                          on=['gen_name', 'tech'])

        if self.metadata.zone_generator_category(model_filename).empty is False:
            # Merges in zones where reserves are located
            df = df.merge(self.metadata
                              .zone_generator_category(model_filename)
                              .reset_index(), 
                          how='left',
                          on=['gen_name', 'tech'])

        if not self.Region_Mapping.empty:
            df = df.merge(self.Region_Mapping, how="left", on="region")

        if not self.emit_names.empty:
            # reclassify emissions as specified by user in mapping
            df['pollutant'] = pd.Categorical(df['pollutant'].map(lambda x: 
                                                                 self.emit_names_dict
                                                                     .get(x, x)))

        # remove categoricals (otherwise h5 save will fail)
        df = df.astype({'tech': 'object', 'pollutant': 'object'})

        # Checks if all emissions categories have been identified and matched. 
        # If not, lists categories that need a match
        if not self.emit_names.empty:
            if self.emit_names_dict != {} and \
            (set(df['pollutant'].unique())
                                .issubset(self.emit_names["New"].unique())) is False:
                missing_emit_cat = list((set(df['pollutant'].unique())) 
                                        - (set(self.emit_names["New"].unique())))
                self.logger.warning("The following emission objects do not have a "
                                    f"correct category mapping: {missing_emit_cat}\n")

        df_col = list(df.columns)
        df_col.remove(0)
        df_col.insert(0, df_col.pop(df_col.index("timestamp")))
        df.set_index(df_col, inplace=True)
        # downcast values to save on memory
        df[0] = pd.to_numeric(df[0].values, downcast='float')
        # convert to range index (otherwise h5 save will fail)
        df.columns = pd.RangeIndex(0, 1, step=1)
        return df

    def df_process_storage(self, df: pd.DataFrame, 
                           model_filename: str) -> pd.DataFrame:
        """Format PLEXOS Storage Class data.

        Args:
            df (pd.DataFrame): h5plexos dataframe to process
            model_filename (str): name of h5plexos h5 file being processed 

        Returns:
            pd.DataFrame: Processed output, single value column with multiindex.
        """
        df = df.droplevel(level=["band", "property", "category"])
        df = df.reset_index()  # unzip the levels in index
        df = df.merge(self.metadata.generator_storage(model_filename), 
                        how='left', on='name')
        if self.metadata.region_generators(model_filename).empty is False:
            # Merges in regions where generators are located
            df = df.merge(self.metadata.region_generators(model_filename),
                          how='left', on='gen_name')  
        if self.metadata.zone_generators(model_filename).empty is False:
            # Merges in zones where generators are located
            df = df.merge(self.metadata.zone_generators(model_filename), 
                            how='left', on='gen_name')  
        # checks if Region_Maping contains data to merge, skips if empty (Default)
        if not self.Region_Mapping.empty:
            # Merges in all Region Mappings
            df = df.merge(self.Region_Mapping, how='left', on='region')  
        df.rename(columns={'name': 'storage_resource'}, inplace=True)
        df_col = list(df.columns)
        df_col.remove(0)
        df_col.insert(0, df_col.pop(df_col.index("timestamp")))
        df.set_index(df_col, inplace=True)
        df[0] = pd.to_numeric(df[0], downcast='float')
        return df

    def df_process_region_regions(self, df: pd.DataFrame, 
                                  model_filename: str) -> pd.DataFrame:
        """Format PLEXOS Region_Regions Relational Class data.

        Args:
            df (pd.DataFrame): h5plexos dataframe to process
            model_filename (str): name of h5plexos h5 file being processed 

        Returns:
            pd.DataFrame: Processed output, single value column with multiindex.
        """
        df = df.droplevel(level=["band", "property"])
        df = pd.DataFrame(data=df.values.reshape(-1), index=df.index)
        df_col = list(df.index.names)
        df_col.insert(0, df_col.pop(df_col.index("timestamp")))
        df = df.reorder_levels(df_col, axis=0)
        df[0] = pd.to_numeric(df[0], downcast='float')
        return df

    def df_process_node(self, df: pd.DataFrame, 
                        model_filename: str) -> pd.DataFrame:
        """Format PLEXOS Node Class data.

        Args:
            df (pd.DataFrame): h5plexos dataframe to process
            model_filename (str): name of h5plexos h5 file being processed 

        Returns:
            pd.DataFrame: Processed output, single value column with multiindex.
        """
        df = df.droplevel(level=["band", "property", "category"])
        df.index.rename('node', level='name', inplace=True)
        df.sort_index(level=['node'], inplace=True)

        node_region_meta = self.metadata.node_region(model_filename)
        node_zone_meta = self.metadata.node_zone(model_filename)
        timeseries_len = len(df.index.get_level_values('timestamp').unique())

        if node_region_meta.empty is False:
            node_region_idx = pd.CategoricalIndex(node_region_meta
                                                  .index.get_level_values(0))

            node_region_idx = node_region_idx.repeat(timeseries_len)

            idx_region = pd.MultiIndex(levels=df.index.levels
                                       + [node_region_idx.categories],
                                       codes=df.index.codes 
                                       + [node_region_idx.codes],
                                       names=df.index.names 
                                       + node_region_idx.names)
        else:
            idx_region = df.index

        if node_zone_meta.empty is False:
            node_zone_idx = pd.CategoricalIndex(node_zone_meta
                                                .index.get_level_values(0))

            node_zone_idx = node_zone_idx.repeat(timeseries_len)

            idx_zone = pd.MultiIndex(levels=idx_region.levels 
                                     + [node_zone_idx.categories],
                                     codes=idx_region.codes 
                                     + [node_zone_idx.codes],
                                     names=idx_region.names 
                                     + node_zone_idx.names)
        else:
            idx_zone = idx_region

        if not self.Region_Mapping.empty:
            region_mapping_idx = pd.MultiIndex.from_frame(node_region_meta
                                              .merge(self.Region_Mapping,
                                                     how="left",
                                                     on='region')
                                              .drop(['region', 'node'], axis=1))
                                
            region_mapping_idx = region_mapping_idx.repeat(timeseries_len)

            idx_map = pd.MultiIndex(levels=idx_zone.levels 
                                    + region_mapping_idx.levels,
                                    codes=idx_zone.codes 
                                    + region_mapping_idx.codes,
                                    names=idx_zone.names 
                                    + region_mapping_idx.names)
        else:
            idx_map = idx_zone

        df = pd.DataFrame(data=df.values.reshape(-1), index=idx_map)
        df_col = list(df.index.names)
        df_col.insert(0, df_col.pop(df_col.index("timestamp")))
        df = df.reorder_levels(df_col, axis=0)
        df[0] = pd.to_numeric(df[0], downcast='float')
        return df

    def df_process_abatement(self, df: pd.DataFrame, 
                             model_filename: str) -> pd.DataFrame:
        """Format PLEXOS Abatement Class data.

        Args:
            df (pd.DataFrame): h5plexos dataframe to process
            model_filename (str): name of h5plexos h5 file being processed 

        Returns:
            pd.DataFrame: Processed output, single value column with multiindex.
        """
        df = df.droplevel(level=["band", "property"])
        df.index.rename('abatement_name', level='name', inplace=True)
        df = pd.DataFrame(data=df.values.reshape(-1), index=df.index)
        df_col = list(df.index.names)
        df_col.insert(0, df_col.pop(df_col.index("timestamp")))
        df = df.reorder_levels(df_col, axis=0)
        df[0] = pd.to_numeric(df[0], downcast='float')
        return df

######################################################################################################

class ProcessEGRET(Process):
    """Process Egret specific data from a json file.
    """
    def __init__(self,
                 input_folder: str,
                 Region_Mapping: pd.DataFrame, 
                *args,
                 plexos_block: str ='',
                 **kwargs):
        """
        Args:
            input_folder (str): Folder containing EGRET json files.
            Region_Mapping (pd.DataFrame): DataFrame to map custom 
                regions/zones to create custom aggregations.
            plexos_block (str, optional): PLEXOS results type. Defaults to 'ST'.
        """
        self.plexos_block = plexos_block
        self.hdf5_collection = {}
        self.metadata = MetaData(input_folder, read_from_formatted_h5=False, Region_Mapping=Region_Mapping)
        # Instantiation of Process Base class
        super().__init__(input_folder, Region_Mapping, *args, **kwargs) 

    # I think the default method works for Egret
    # def get_input_files(self) -> list:
    #     """Gets a list of Egret input files within the scenario folders

    #     Returns:
    #         list: list of Egret input filenames to process
    #     """
    #     return files_list

    # I think the default method works for Egret
    # def output_metadata(self, files_list: list, output_file_path: str) -> None:
    #     """

    #     Args:
    #         files_list (list): List of all h5 files in hdf5 folder in alpha numeric order
    #         output_file_path (str): Location of formatted output h5 file 
    #     """

    def get_processed_data(self, plexos_class: str, plexos_prop: str, 
                  timescale: str, model_filename: str) -> pd.DataFrame:
        """Handles the pulling of data from the egret json
        file and then passes the data to one of the formating functions

        Args:
            plexos_class (str): PLEXOS class e.g Region, Generator, Zone etc
            plexos_prop (str): PLEXOS property e.g Max Capacity, Generation etc.
            timescale (str): Data timescale, e.g Hourly, Monthly, 5 minute etc.
            model_filename (str): name of model to process.

        Returns:
            pd.DataFrame: Formatted results dataframe.
        """
        # db = self.hdf5_collection.get(model_filename)
        # try:
        #     if "_" in plexos_class:
        #         df = db.query_relation_property(plexos_class, 
        #                                         plexos_prop, 
        #                                         timescale=timescale,
        #                                         phase=self.plexos_block)
        #         object_class = plexos_class
        #     else:
        #         df = db.query_object_property(plexos_class, 
        #                                       plexos_prop, 
        #                                       timescale=timescale,
        #                                       phase=self.plexos_block)
        #         if ((0,6,0) <= db.version and db.version < (0,7,0)):
        #             object_class = f"{plexos_class}s"
        #         else:
        #             object_class = plexos_class

        # except (ValueError, KeyError):
        #     df = self.report_prop_error(plexos_prop, plexos_class)
        #     return df
        
        # if self.plexos_block != 'ST':
        #     df = self.merge_timeseries_block_data(db, df)

        # # handles h5plexos naming discrepency 
        # if ((0,6,0) <= db.version and db.version < (0,7,0)):
        #     # Get original units from h5plexos file 
        #     df_units = (db.h5file[f'/data/{self.plexos_block}/{timescale}'
        #                           f'/{object_class}/{plexos_prop}'].attrs['units']
        #                                                            .decode('UTF-8'))
        # else:
        #     df_units = (db.h5file[f'/data/{self.plexos_block}/{timescale}'
        #                           f'/{object_class}/{plexos_prop}'].attrs['unit'])
        
        # # find unit conversion values
        # converted_units = UNITS_CONVERSION.get(df_units, (df_units, 1))

        # Read json file
        f = open(model_filename, 'r')
        data = json.load(f)
        f.close()
        
        # Get desired method
        process_att = getattr(self, f'df_process_{plexos_class}')
        # Process attribute and return to df
        df = process_att(data, plexos_prop)
        
        # # Convert units and add unit column to index 
        # df = df*converted_units[1]
        # units_index = pd.Index([converted_units[0]] *len(df), name='units')
        # df.set_index(units_index, append=True, inplace=True)

        # if plexos_class == 'region' and \
        #    plexos_prop == "Unserved Energy" and \
        #    int(df.sum(axis=0)) > 0:
        #     self.logger.warning(f"Scenario contains Unserved Energy: "
        #                         f"{int(df.sum(axis=0))} MW\n")
        return df


    # def merge_timeseries_block_data(self, db: PLEXOSSolution, 
    #                                 df: pd.DataFrame) -> pd.DataFrame:
    #     """Merge timeseries and block data found in LT, MT and PASA results

    #     Args:
    #         db (PLEXOSSolution): PLEXOSSolution instance for specific h5plexos file.
    #         df (pd.DataFrame): h5plexos dataframe 

    #     Returns:
    #         pd.DataFrame: df with merged in timeseries data 
    #     """

    #     block_mapping = db.blocks[self.plexos_block]
    #     block_mapping.index.rename('timestamp', inplace=True)
    #     df = df.reset_index(level='block')

    #     merged_data = df.reset_index().merge(block_mapping.reset_index(), on='block')
    #     merged_data.drop('block', axis=1, inplace=True)
    #     index_cols = list(merged_data.columns)
    #     index_cols.remove(0)
    #     merged_data.set_index(index_cols, inplace=True)
    #     return merged_data

    
    def df_process_generator(self, data, egret_property) -> pd.DataFrame:
        """Format PLEXOS Generator Class data.

        Args:
            data (dictionary): Egret json data read into a nested dictionary
            egret_property (string): Egret property name; key of json file

        Returns:
            pd.DataFrame: Processed output, single value column with multiindex.
        """
        # Initialize dataframe column lists
        timestamp = []
        tech = []
        gen_name = []
        region = []
        zone = []
        superzone = []
        Midwest_Agg = []
        Usual = []
        Country = []
        Interconnection = []
        CountryInterconnect = []
        Summary = []
        values = []

        # Loop through generators 
        for generator in data['elements'][egret_property].keys():
            
            # Get generator time series values
            vals = data['elements'][egret_property][generator]['pg']['values']

            # Get zone and area labels if they exists
            if 'zone' in data['elements'][egret_property][generator].keys():
                zone_val = data['elements'][egret_property][generator]['zone']
            else:
                zone_val = '0'
            if 'area' in data['elements'][egret_property][generator].keys():
                area_val = data['elements'][egret_property][generator]['area']
            else:
                area_val = '0'

            timestamp += [(datetime.datetime(2020,1,1) + datetime.timedelta(hours=i)).strftime("%Y-%m-%d %H:%M:%S") for i in range(len(vals))]
            tech += ['NaN']*len(vals)
            gen_name += [generator]*len(vals)
            region += [area_val]*len(vals)
            zone += [zone_val]*len(vals)
            superzone += ['NaN']*len(vals)
            Midwest_Agg += ['NaN']*len(vals)
            Usual += ['NaN']*len(vals)
            Country += ['NaN']*len(vals)
            Interconnection += ['NaN']*len(vals)
            CountryInterconnect += ['NaN']*len(vals)
            Summary += ['NaN']*len(vals)
            values += vals

        # Put data into pandas dataframe
        gen_df = pd.DataFrame({'timestamp':timestamp,
                             'tech':tech,
                             'gen_name':gen_name,
                             'region':region,
                             'zone':zone,
                             'superzone':superzone,
                             'Midwest_Agg':Midwest_Agg,
                             'Usual':Usual,
                             'Country':Country,
                             'Interconnection':Interconnection,
                             'CountryInterconnect':CountryInterconnect,
                             'Summary':Summary,
                             '0':values})

        # Set dataframe column indices
        gen_df = gen_df.set_index(['timestamp',
                        'tech',
                        'gen_name',
                        'region',
                        'zone',
                        'superzone',
                        'Midwest_Agg',
                        'Usual',
                        'Country',
                        'Interconnection',
                        'CountryInterconnect',
                        'Summary'])

        return gen_df


    def df_process_region(self, data, egret_property) -> pd.DataFrame:
        """Format EGRET region data. I think currently this will only work for load.

        Args:
            data (dictionary): EGRET json data read into a nested dictionary
            egret_property (string): Egret property name; key of json file

        Returns:
            pd.DataFrame: Processed output, single value column with multiindex.
        """
        # If there is no region information, demand is the only key
        if 'demand' in data['elements'][egret_property].keys():
            values = data['elements'][egret_property]['demand']['p_load']['values']
            timestamp = [i for i in range(len(values))]
            region = [0]*len(values)
            superzone = ['NaN']*len(values)
            Midwest_Agg = ['NaN']*len(values)
            Usual = ['NaN']*len(values)
            Country = ['NaN']*len(values)
            Interconnection = ['NaN']*len(values)
            CountryInterconnect = ['NaN']*len(values)
            Summary = ['NaN']*len(values)

        # If there is region information, aggregate by region
        else:
            values_dict = {}
            for key in data['elements'][egret_property].keys():
                area = data['elements'][egret_property][key]['area']
                # Creat new key 
                if area not in values_dict.keys():
                    values_dict[area] = np.array(data['elements'][egret_property][key]['p_load']['values'])
                # Add values if key already exists
                else:
                    values_dict[area] += np.array(data['elements'][egret_property][key]['p_load']['values'])

            # Create actual columns        
            values = []
            timestamp = []
            region = [] # this is area in Egret
            superzone = []
            Midwest_Agg = []
            Usual = []
            Country = []
            Interconnection = []
            CountryInterconnect = []
            Summary = []
            for key in values_dict.keys():
                values += list(values_dict[key])
                timestamp += [(datetime.datetime(2020,1,1) + datetime.timedelta(hours=i)).strftime("%Y-%m-%d %H:%M:%S") for i in range(len(values_dict[key]))]
                region += [key]*len(values_dict[key])
                superzone += ['NaN']*len(values_dict[key])
                Midwest_Agg += ['NaN']*len(values_dict[key])
                Usual += ['NaN']*len(values_dict[key])
                Country += ['NaN']*len(values_dict[key])
                Interconnection += ['NaN']*len(values_dict[key])
                CountryInterconnect += ['NaN']*len(values_dict[key])
                Summary += ['NaN']*len(values_dict[key])

        # Create dataframe
        region_df = pd.DataFrame({'timestamp':timestamp,
                                  'region':region,
                                  'superzone':superzone,
                                  'Midwest_Agg':Midwest_Agg,
                                  'Usual':Usual,
                                  'Country':Country,
                                  'Interconnection':Interconnection,
                                  'CountryInterconnect':CountryInterconnect,
                                  'Summary':Summary,
                                  '0':values})

        # Set index
        region_df = region_df.set_index(['timestamp',
                                       'region',
                                       'superzone',
                                       'Midwest_Agg',
                                       'Usual',
                                       'Country',
                                       'Interconnection',
                                       'CountryInterconnect',
                                       'Summary'])

        return region_df
    

    def df_process_zone(self, data, egret_property) -> pd.DataFrame:
        """Format EGRET zone data. I think currently this will only work for load.

        Args:
            data (dictionary): Egret json data read into a nested dictionary
            egret_property (string): Egret property name; key of json file

        Returns:
            pd.DataFrame: Processed output, single value column with multiindex.
        """
        # If there is no zone information, demand is the only key
        if 'demand' in data['elements'][egret_property].keys():
            values = data['elements'][egret_property]['demand']['p_load']['values']
            timestamp = [i for i in range(len(values))]
            zone = [0]*len(values)

        # If there is zone information, aggregate by zone
        else:
            values_dict = {}
            for key in data['elements'][egret_property].keys():
                zone = data['elements'][egret_property][key]['zone']
                # Creat new key 
                if zone not in values_dict.keys():
                    values_dict[zone] = np.array(data['elements'][egret_property][key]['p_load']['values'])
                # Add values if key already exists
                else:
                    values_dict[zone] += np.array(data['elements'][egret_property][key]['p_load']['values'])

            # Create actual columns        
            values = []
            timestamp = []
            zone = []
            for key in values_dict.keys():
                values += list(values_dict[key])
                timestamp += [(datetime.datetime(2020,1,1) + datetime.timedelta(hours=i)).strftime("%Y-%m-%d %H:%M:%S") for i in range(len(values_dict[key]))]
                zone += [key]*len(values_dict[key])

        # Create dataframe
        zone_df = pd.DataFrame({'timestamp':timestamp,
                                'zone':zone,
                                '0':values})
        # Set index
        zone_df = zone_df.set_index(['timestamp','zone'])
        return zone_df


    def df_process_line(self, data) -> pd.DataFrame:
        """Format PLEXOS Line Class data.

        Args:
            data (dictionary): Egret json data read into a nested dictionary

        Returns:
            pd.DataFrame: Processed output, single value column with multiindex.
        """
        df = pd.DataFrame()
        return df

    def df_process_interface(self, data) -> pd.DataFrame:
        """Format PLEXOS PLEXOS Interface Class data.

        Args:
            data (dictionary): Egret json data read into a nested dictionary

        Returns:
            pd.DataFrame: Processed output, single value column with multiindex.
        """
        df = pd.DataFrame()
        return df

    def df_process_reserve(self, data) -> pd.DataFrame:
        """Format PLEXOS Reserve Class data.

        Args:
            data (dictionary): Egret json data read into a nested dictionary

        Returns:
            pd.DataFrame: Processed output, single value column with multiindex.
        """
        df = pd.DataFrame()
        return df

    def df_process_reserves_generators(self, data) -> pd.DataFrame:
        """Format PLEXOS Reserve_Generators Relational Class data.

        Args:
            data (dictionary): Egret json data read into a nested dictionary

        Returns:
            pd.DataFrame: Processed output, single value column with multiindex.
        """
        df = pd.DataFrame()
        return df

    def df_process_fuel(self, data) -> pd.DataFrame:
        """Format PLEXOS Fuel Class data.

        Args:
            data (dictionary): Egret json data read into a nested dictionary

        Returns:
            pd.DataFrame: Processed output, single value column with multiindex.
        """
        df = pd.DataFrame()
        return df

    def df_process_constraint(self, data) -> pd.DataFrame:
        """Format PLEXOS Constraint Class data.

        Args:
            data (dictionary): Egret json data read into a nested dictionary

        Returns:
            pd.DataFrame: Processed output, single value column with multiindex.
        """
        df = pd.DataFrame()
        return df

    def df_process_emission(self, data) -> pd.DataFrame:
        """Format PLEXOS Emission Class data.

        Args:
            data (dictionary): Egret json data read into a nested dictionary

        Returns:
            pd.DataFrame: Processed output, single value column with multiindex.
        """
        df = pd.DataFrame()
        return df

    def df_process_emissions_generators(self, data) -> pd.DataFrame:
        """Format PLEXOS Emissions_Generators Relational Class data.

        Args:
            data (dictionary): Egret json data read into a nested dictionary

        Returns:
            pd.DataFrame: Processed output, single value column with multiindex.
        """
        df = pd.DataFrame()
        return df

    def df_process_storage(self, data) -> pd.DataFrame:
        """Format PLEXOS Storage Class data.

        Args:
            data (dictionary): Egret json data read into a nested dictionary

        Returns:
            pd.DataFrame: Processed output, single value column with multiindex.
        """
        df = pd.DataFrame()
        return df

    def df_process_region_regions(self, data) -> pd.DataFrame:
        """Format PLEXOS Region_Regions Relational Class data.

        Args:
           data (dictionary): Egret json data read into a nested dictionary

        Returns:
            pd.DataFrame: Processed output, single value column with multiindex.
        """
        df = pd.DataFrame()
        return df

    def df_process_node(self, data) -> pd.DataFrame:
        """Format PLEXOS Node Class data.

        Args:
           data (dictionary): Egret json data read into a nested dictionary

        Returns:
            pd.DataFrame: Processed output, single value column with multiindex.
        """
        df = pd.DataFrame()
        return df

    def df_process_abatement(self, data) -> pd.DataFrame:
        """Format PLEXOS Abatement Class data.

        Args:
            data (dictionary): Egret json data read into a nested dictionary 

        Returns:
            pd.DataFrame: Processed output, single value column with multiindex.
        """
        df = pd.DataFrame()
        return df

######################################################################################################


    

class MarmotFormat(SetupLogger):
    """Main module class to be instantiated to run the formatter.

    MarmotFormat handles the passing on information to the various
    Process classes and handles the saving of formatted results.
    Once the outputs have been processed, they are saved to an intermediary hdf5 file
    which can then be read into the Marmot plotting code
    """

    def __init__(self, Scenario_name: str, 
                 Model_Solutions_folder: str, 
                 Plexos_Properties: Union[str, pd.DataFrame],
                 Marmot_Solutions_folder: str = None,
                 mapping_folder: str = 'mapping_folder',
                 Region_Mapping: Union[str, pd.DataFrame] = pd.DataFrame(),
                 emit_names: Union[str, pd.DataFrame] = pd.DataFrame(),
                 VoLL: int = 10000,
                 **kwargs):
        """
        Args:
            Scenario_name (str): Name of scenario to process.
            Model_Solutions_folder (str): Folder containing model simulation 
                results subfolders and their files.
            Plexos_Properties (Union[str, pd.DataFrame]): PLEXOS properties 
                to process, must follow format seen in Marmot directory.
            Marmot_Solutions_folder (str, optional): Folder to save Marmot 
                solution files.
                Defaults to None.
            mapping_folder (str, optional): The location of the Marmot 
                mapping folder.
                Defaults to 'mapping_folder'.
            Region_Mapping (Union[str, pd.DataFrame], optional): Mapping file 
                to map custom regions/zones to create custom aggregations.
                Aggregations are created by grouping PLEXOS regions.
                Defaults to pd.DataFrame().
            emit_names (Union[str, pd.DataFrame], optional): Mapping file 
                to rename emissions types. 
                Defaults to pd.DataFrame().
            VoLL (int, optional): Value of lost load, used to calculate 
                cost of unserved energy. 
                Defaults to 10000.
        """
        super().__init__(**kwargs) # Instantiation of SetupLogger

        self.Scenario_name = Scenario_name
        self.Model_Solutions_folder = Model_Solutions_folder
        self.Marmot_Solutions_folder = Marmot_Solutions_folder
        self.mapping_folder = mapping_folder
        self.VoLL = VoLL

        if self.Marmot_Solutions_folder is None:
            self.Marmot_Solutions_folder = self.Model_Solutions_folder

        if isinstance(Plexos_Properties, str):
            try:
                self.Plexos_Properties = pd.read_csv(Plexos_Properties)
            except FileNotFoundError:
                self.logger.warning("Could not find specified "
                                    "Plexos_Properties file; check file name. "
                                    "This is required to run Marmot, "
                                    "system will now exit")
                sys.exit()
        elif isinstance(Plexos_Properties, pd.DataFrame):
            self.Plexos_Properties = Plexos_Properties

        if isinstance(Region_Mapping, str):
            try:
                self.Region_Mapping = pd.read_csv(Region_Mapping)
                if not self.Region_Mapping.empty:
                    self.Region_Mapping = self.Region_Mapping.astype(str)
            except FileNotFoundError:
                self.logger.warning("Could not find specified "
                                    "Region Mapping file; "
                                    "check file name\n")
                self.Region_Mapping = pd.DataFrame()
        elif isinstance(Region_Mapping, pd.DataFrame):
            self.Region_Mapping = Region_Mapping
            if not self.Region_Mapping.empty:
                self.Region_Mapping = self.Region_Mapping.astype('string')
        try:
            # delete category columns if exists
            self.Region_Mapping = self.Region_Mapping.drop(["category"], axis=1)  
        except KeyError:
            pass

        if isinstance(emit_names, str):
            try:
                self.emit_names = pd.read_csv(emit_names)
                if not self.emit_names.empty:
                    self.emit_names.rename(columns=
                                           {self.emit_names.columns[0]: 'Original',
                                           self.emit_names.columns[1]: 'New'},
                                           inplace=True)
            except FileNotFoundError:
                self.logger.warning("Could not find specified emissions "
                                    "mapping file; check file name\n")
                self.emit_names = pd.DataFrame()
        elif isinstance(emit_names, pd.DataFrame):
            self.emit_names = emit_names
            if not self.emit_names.empty:
                self.emit_names.rename(columns={self.emit_names.columns[0]: 'Original',
                                                self.emit_names.columns[1]: 'New'},
                                       inplace=True)

    @staticmethod
    def _save_to_h5(df: pd.DataFrame, file_name: str, key: str, 
                    mode: str = "a", complevel: int = 9, 
                    complib: str ='blosc:zlib', **kwargs) -> None:
        """Saves data to formatted hdf5 file

        Args:
            df (pd.DataFrame): Dataframe to save 
            file_name (str): name of hdf5 file
            key (str): formatted property identifier, 
                e.g generator_Generation
            mode (str, optional): file access mode. 
                Defaults to "a".
            complevel (int, optional): compression level. 
                Defaults to 9.
            complib (str, optional): compression library. 
                Defaults to 'blosc:zlib'.
        """
        df.to_hdf(file_name, key=key, mode=mode,
                    complevel=complevel,
                    complib=complib,
                    **kwargs)

    def run_formatter(self, sim_model='PLEXOS', plexos_block='ST', 
                      append_block_name=False) -> None:
        """Main method to call to begin formatting simulation model results

        Args:
            sim_model (str, optional): Name of simulation model to 
                process data for.
                Defaults to 'PLEXOS'.
            plexos_block (str, optional): PLEXOS results type. 
                Defaults to 'ST'.
            append_block_name (bool, optional): Append block type to 
                scenario name. 
                Defaults to False.
        """
        if append_block_name:
            scen_name = f"{self.Scenario_name} {plexos_block}"
        else:
            scen_name = self.Scenario_name
        
        self.logger.info(f"#### Processing {scen_name} PLEXOS Results ####")

        hdf5_output_name = f"{scen_name}_formatted.h5"
        input_folder = os.path.join(self.Model_Solutions_folder, 
                                      str(self.Scenario_name))
        output_folder = os.path.join(self.Marmot_Solutions_folder, 
                                      'Processed_HDF5_folder')
        try:
            os.makedirs(output_folder)
        except FileExistsError:
            # directory already exists
            pass

        output_file_path = os.path.join(output_folder, hdf5_output_name)
        
        process_sim_model = globals()[f"Process{sim_model}"](input_folder, 
                                                      self.Region_Mapping,
                                                      self.emit_names,
                                                      self.logger,
                                                      plexos_block=plexos_block)
        files_list = process_sim_model.get_input_files()

        # ===============================================================================
        # Process the Outputs
        # ===============================================================================

        # Creates Initial HDF5 file for outputting formated data
        Processed_Data_Out = pd.DataFrame()
        if os.path.isfile(output_file_path) is True:
            self.logger.info(f"'{output_file_path}' already exists: New "
                             "variables will be added\n")
            # Skip properties that already exist in *formatted.h5 file.
            with h5py.File(output_file_path, 'r') as f:
                existing_keys = [key for key in f.keys()]
            # The processed HDF5 output file already exists. If metadata is already in
            # this file, leave as is. Otherwise, append it to the file.
            if 'metadata' not in existing_keys:
                self.logger.info('Adding metadata to processed HDF5 file.')
                process_sim_model.output_metadata(files_list, output_file_path)

            if not mconfig.parser('skip_existing_properties'):
                existing_keys = []

        # The processed HDF5 file does not exist. 
        # Create the file and add metadata to it.
        else:
            existing_keys = []
            # Create empty hdf5 file 
            f = h5py.File(output_file_path, "w")
            f.close()
            process_sim_model.output_metadata(files_list, output_file_path)

        process_properties = (self.Plexos_Properties
                                  .loc[self.Plexos_Properties["collect_data"] == True])
        
        start = time.time()
        # Main loop to process each output and pass data to functions
        for index, row in process_properties.iterrows():
            Processed_Data_Out = pd.DataFrame()
            data_chunks = []

            self.logger.info(f'Processing {row["group"]} {row["data_set"]}')
            prop_underscore = row["data_set"].replace(' ', '_')
            key_path = row["group"] + "_" + prop_underscore
            
            if key_path not in existing_keys:
                for model in files_list:
                    self.logger.info(f"      {model}")
                    processed_data = process_sim_model.get_processed_data(row["group"], 
                                                                  row["data_set"], 
                                                                  row["data_type"], 
                                                                  model)
                    if processed_data.empty is True:
                        break
                    
                    # Check if data is for year interval and of type capacity
                    if (row["data_type"] == "year") & (
                            (row["data_set"] == "Installed Capacity")
                            | (row["data_set"] == "Export Limit")
                            | (row["data_set"] == "Import Limit")
                            ):
                        data_chunks.append(processed_data)
                        self.logger.info(f"{row['data_set']} Year property reported "
                                         "from only the first partition")
                        break
                    else:
                        data_chunks.append(processed_data)

                if data_chunks:
                    Processed_Data_Out = pd.concat(data_chunks, copy=False)

                if Processed_Data_Out.empty is False:
                    if (row["data_type"] == "year"):
                        self.logger.info("Please Note: Year properties can not "
                                         "be checked for duplicates.\n"
                                         "Overlaping data cannot be removed from "
                                         "'Year' grouped data.\nThis will effect "
                                         "Year data that differs between partitions "
                                         "such as cost results.\nIt will not effect "
                                         "Year data that is equal in all partitions "
                                         "such as Installed Capacity or "
                                         "Line Limit results")
                    else:
                        oldsize = Processed_Data_Out.size
                        # Remove duplicates; keep first entry
                        Processed_Data_Out = (Processed_Data_Out.loc
                                              [~Processed_Data_Out
                                              .index.duplicated(keep='first')])

                        if (oldsize - Processed_Data_Out.size) > 0:
                            self.logger.info("Drop duplicates removed "
                                             f"{oldsize-Processed_Data_Out.size} rows")

                    row["data_set"] = row["data_set"].replace(' ', '_')
                    
                    save_attempt=1
                    while save_attempt<=3:
                        try:
                            self.logger.info("Saving data to h5 file...")
                            MarmotFormat._save_to_h5(Processed_Data_Out,
                                                     output_file_path, 
                                                     key=(f'{row["group"]}_'
                                                          f'{row["data_set"]}'))

                            self.logger.info("Data saved to h5 file successfully\n")
                            save_attempt=4
                        except:
                            self.logger.warning("h5 File is probably in use, "
                                                "waiting to attempt to save again")
                            time.sleep(60)
                            save_attempt+=1
                else:
                    continue
            else:
                self.logger.info(f"{key_path} already exists in output .h5 file.")
                self.logger.info("PROPERTY ALREADY PROCESSED\n")
                continue

        # ===============================================================================
        # Calculate Extra Outputs
        # ===============================================================================
        if "generator_Curtailment" not in \
            h5py.File(output_file_path, 'r') or not \
            mconfig.parser('skip_existing_properties'):
            try:
                self.logger.info("Processing generator Curtailment")
                try:
                    Avail_Gen_Out = pd.read_hdf(output_file_path,
                                                'generator_Available_Capacity')
                    Total_Gen_Out = pd.read_hdf(output_file_path,
                                                'generator_Generation')
                    if Total_Gen_Out.empty is True:
                        self.logger.warning("generator_Available_Capacity & "
                                            "generator_Generation are required "
                                            "for Curtailment calculation")
                except KeyError:
                    self.logger.warning("generator_Available_Capacity & "
                                        "generator_Generation are required "
                                        "for Curtailment calculation")

                Curtailment_Out = Avail_Gen_Out - Total_Gen_Out

                Upward_Available_Capacity = Curtailment_Out

                MarmotFormat._save_to_h5(Curtailment_Out,
                                    output_file_path, 
                                    key="generator_Curtailment")

                MarmotFormat._save_to_h5(Upward_Available_Capacity,
                                    output_file_path, 
                                    key="generator_Upward_Available_Capacity")

                self.logger.info("Data saved to h5 file successfully\n")
                # Clear Some Memory
                del Total_Gen_Out
                del Avail_Gen_Out
                del Curtailment_Out
            except Exception:
                self.logger.warning("NOTE!! Curtailment not calculated, "
                                    "processing skipped\n")

        if "region_Cost_Unserved_Energy" not in \
            h5py.File(output_file_path, 'r') or not \
            mconfig.parser('skip_existing_properties'):
            try:
                self.logger.info("Calculating Cost Unserved Energy: Regions")
                Cost_Unserved_Energy = pd.read_hdf(output_file_path,
                                                   'region_Unserved_Energy')
                                                   
                Cost_Unserved_Energy = Cost_Unserved_Energy * self.VoLL

                MarmotFormat._save_to_h5(Cost_Unserved_Energy,
                                    output_file_path, 
                                    key="region_Cost_Unserved_Energy")
            except KeyError:
                self.logger.warning("NOTE!! Regional Unserved Energy not available "
                                    "to process, processing skipped\n")
                pass

        if "zone_Cost_Unserved_Energy" not in \
            h5py.File(output_file_path, 'r') or not \
            mconfig.parser('skip_existing_properties'):
            try:
                self.logger.info("Calculating Cost Unserved Energy: Zones")
                Cost_Unserved_Energy = pd.read_hdf(output_file_path,
                                                   'zone_Unserved_Energy')
                Cost_Unserved_Energy = Cost_Unserved_Energy * self.VoLL

                MarmotFormat._save_to_h5(Cost_Unserved_Energy,
                                    output_file_path, 
                                    key="zone_Cost_Unserved_Energy")
            except KeyError:
                self.logger.warning("NOTE!! Zonal Unserved Energy not available to "
                                    "process, processing skipped\n")
                pass

        end = time.time()
        elapsed = end - start
        self.logger.info('Main loop took %s minutes', round(elapsed/60, 2))
        self.logger.info(f'Formatting COMPLETED for {scen_name}')


def main():
    """Run the formatting code and format desired properties based on user input files."""

    # ===================================================================================
    # Input Properties
    # ===================================================================================

    # Changes working directory to location of this python file
    os.chdir(FILE_DIR)

    Marmot_user_defined_inputs = pd.read_csv(mconfig.parser("user_defined_inputs_file"),
                                             usecols=['Input', 'User_defined_value'],
                                             index_col='Input',
                                             skipinitialspace=True)

    simulation_model = (Marmot_user_defined_inputs.loc['Simulation_model']
                                                        .to_string(index=False).strip())

    if pd.isna(Marmot_user_defined_inputs.loc['PLEXOS_data_blocks',
                                              'User_defined_value']):
        plexos_data_blocks = ['ST']
    else:
        plexos_data_blocks = (pd.Series(Marmot_user_defined_inputs.loc['PLEXOS_data_blocks']
                                                                  .squeeze().split(","))
                                                                  .str.strip().tolist())

    # File which determiens which plexos properties to pull from the h5plexos results and 
    # process, this file is in the repo
    Plexos_Properties = pd.read_csv(mconfig.parser('plexos_properties_file'))
    
    # Name of the Scenario(s) being run, must have the same name(s) as the folder 
    # holding the runs HDF5 file
    Scenario_List = (pd.Series(Marmot_user_defined_inputs.loc['Scenario_process_list']
                                                         .squeeze().split(","))
                                                         .str.strip().tolist())
    # The folder that contains all the simulation model outputs - the files should 
    # be contained in another folder with the Scenario_name
    Model_Solutions_folder = (Marmot_user_defined_inputs.loc['Model_Solutions_folder']
                                                         .to_string(index=False).strip())

    # Folder to save your processed solutions
    if pd.isna(Marmot_user_defined_inputs.loc['Marmot_Solutions_folder',
                                              'User_defined_value']):
        Marmot_Solutions_folder = None
    else:
        Marmot_Solutions_folder = (Marmot_user_defined_inputs.loc
                                                             ['Marmot_Solutions_folder']
                                                             .to_string(index=False)
                                                             .strip())

    # This folder contains all the csv required for mapping and selecting outputs 
    # to process. Examples of these mapping files are within the Marmot repo, you 
    # may need to alter these to fit your needs
    Mapping_folder = 'mapping_folder'

    if pd.isna(Marmot_user_defined_inputs.loc['Region_Mapping.csv_name', 
                                              'User_defined_value']) is True:
        Region_Mapping = pd.DataFrame()
    else:
        Region_Mapping = (pd.read_csv(os.path.join(Mapping_folder, 
                                                   Marmot_user_defined_inputs
                                                   .loc['Region_Mapping.csv_name']
                                                   .to_string(index=False).strip())))

    # Value of Lost Load for calculating cost of unserved energy
    VoLL = pd.to_numeric(Marmot_user_defined_inputs.loc['VoLL'].to_string(index=False))

    # ===================================================================================
    # Standard Naming of Emissions types (optional)
    # ===================================================================================

    emit_names = os.path.join(Mapping_folder, Marmot_user_defined_inputs
                              .loc['emit_names.csv_name']
                              .to_string(index=False).strip())

    # ===================================================================================
    # Loop through scenarios in list
    # ===================================================================================

    for Scenario_name in Scenario_List:
        
        initiate = MarmotFormat(Scenario_name, Model_Solutions_folder, 
                                Plexos_Properties,
                                Marmot_Solutions_folder=Marmot_Solutions_folder,
                                mapping_folder=Mapping_folder,
                                Region_Mapping=Region_Mapping,
                                emit_names=emit_names,
                                VoLL=VoLL)

        if simulation_model=='PLEXOS':
            for block in plexos_data_blocks:
                initiate.run_formatter(plexos_block=block, 
                                       append_block_name=mconfig.parser('append_plexos_block_name'))
        else:
            initiate.run_formatter(sim_model=simulation_model)


if __name__ == '__main__':
    main()

#===============================================================================
# Code that can be used to test PLEXOS_H5_results_formatter
#===============================================================================
# test = test.xs("p60",level='region')
# test = test.xs("gas-ct",level='tech')
# test = test.reset_index(['timestamp','node'])
# test = test.groupby(["timestamp", "node"], as_index=False).sum()
# test = test.pivot(index='timestamp', columns='node', values=0)

# test = test[['600003_PR IS31G_20','600005_MNTCE31G_22']]
# test = test.reset_index()

# test.index.get_level_values('region') = (test.index.get_level_values('region')
#                                                       .astype("category"))

# test['timestamp'] = test['timestamp'].astype("category")

# test.index = (test.index.set_levels(test.index.levels[-1].
#                                           astype('category'), level=-1))

# test.memory_usage(deep=True)
# test[0] = pd.to_numeric(test[0], downcast='float')

# test.memory_usage(deep=False)

# Stacked_Gen_read = Stacked_Gen_read.reset_index() 
# Stacked_Gen_read.rename(columns={'name':'zone'}, inplace=True)
#         Stacked_Gen_read = Stacked_Gen_read.drop(["band", 
#                                               "property", "category"],axis=1)

    #storage = db.storage("Generation")
    #storage = df_process_storage(storage, overlap)

# df_old = df
# t =df.loc[~df.index.duplicated()]
# df_old.equals(df)
