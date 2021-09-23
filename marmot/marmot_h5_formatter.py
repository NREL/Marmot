# -*- coding: utf-8 -*-
"""
First Created on Wed May 22 14:29:48 2019

This code was written to process PLEXOS HDF5 outputs to get them ready for plotting.
Once the data is processed it is outputed as an intermediary HDF5 file format so that
it can be read into the marmot_plot_main.py file


@author: Daniel Levie
"""
# ===============================================================================
# Import Python Libraries
# ===============================================================================

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

try:
    from marmot.meta_data import MetaData
except ModuleNotFoundError:
    print("Attempted import of Marmot as a module from a Git directory. ", end='')
    print("Import of Marmot will not function in this way. ", end='')
    print("To import Marmot as a module use the preferred method of pip installing Marmot, ", end='')
    print("or add the Marmot directory to the system path, see ReadME for details.\n")
    print("System will now exit")
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

# Conversion units dict, key values is a tuple of new unit name and conversion multiplier
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

    Allows an optional suffix to be included which will be appended to the
    end of the log file name, this is useful when running multiple
    processes in parallel to allow logging to seperate files.

    Allows log_directory to be changed from default

    SetupLogger is a subclass of all other module classes
    """

    def __init__(self, log_directory='logs', log_suffix=None):
        """Setuplogger __init__ method.

        Formats log filename,
        configures logger from marmot_logging_config.yml file,
        handles rollover of log file on each instantiation.

        Allows log_directory to be changed from default

        Parameters
        ----------
        log_directory : string, optional
            log directory to save logs, The default is 'logs'
        log_suffix : string, optional
            optional suffix to add to end of log file. The default is None.

        Returns
        -------
        None.

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


class Process(SetupLogger):
    """Conatins methods for processing h5plexos query data.

    All methods are PLEXOS Class specific
    e.g generator, region, zone, line etc.
    """

    def __init__(self, df, metadata, model, Region_Mapping, emit_names, logger):
        """Process __init__ method.

        Parameters
        ----------
        df : pd.DataFrame
            Unprocessed h5plexos dataframe containing
            class and property specifc data.
        metadata : meta_data.MetaData (class instantiation)
            Instantiation of MetaData for specific h5plexos file.
        Region_Mapping : pd.DataFrame
            DataFrame to map custom regions/zones to create custom aggregations.
        emit_names : pd.DataFrame
            DataFrame with 2 columns to rename emmission names.
        logger : logger object.
            logger object from SetupLOgger.

        Returns
        -------
        None.
        """

        # certain methods require information from metadata.  metadata is now
        # passed in as an instance of MetaData class for the appropriate model
        self.df = df
        self.metadata = metadata
        self.model = model
        self.Region_Mapping = Region_Mapping
        self.emit_names = emit_names
        self.logger = logger

        if not self.emit_names.empty:
            self.emit_names_dict = self.emit_names[['Original', 'New']].set_index("Original").to_dict()["New"]

    def df_process_generator(self) -> pd.DataFrame:
        """Format data which comes form the PLEXOS Generator Class.

        Returns
        -------
        df : pd.DataFrame
            Processed Output, single value column with multiindex.

        """
        df = self.df.droplevel(level=["band", "property"])
        df.index.rename(['tech', 'gen_name'], level=['category', 'name'], inplace=True)

        if self.metadata.region_generator_category(self.model).empty is False:
            region_gen_idx = pd.CategoricalIndex(self.metadata.region_generator_category(self.model).index.get_level_values(0))
            region_gen_idx = region_gen_idx.repeat(len(df.index.get_level_values('timestamp').unique()))

            idx_region = pd.MultiIndex(levels=df.index.levels + [region_gen_idx.categories],
                                       codes=df.index.codes + [region_gen_idx.codes],
                                       names=df.index.names + region_gen_idx.names)
        else:
            idx_region = df.index

        if self.metadata.zone_generator_category(self.model).empty is False:
            zone_gen_idx = pd.CategoricalIndex(self.metadata.zone_generator_category(self.model).index.get_level_values(0))
            zone_gen_idx = zone_gen_idx.repeat(len(df.index.get_level_values('timestamp').unique()))

            idx_zone = pd.MultiIndex(levels=idx_region.levels + [zone_gen_idx.categories],
                                     codes=idx_region.codes + [zone_gen_idx.codes],
                                     names=idx_region.names + zone_gen_idx.names)
        else:
            idx_zone = idx_region

        if not self.Region_Mapping.empty:
            region_gen_mapping_idx = pd.MultiIndex.from_frame(self.metadata.region_generator_category(self.model)
                                                              .merge(self.Region_Mapping,
                                                                     how="left",
                                                                     on='region')
                                                              .sort_values(by=['tech', 'gen_name'])
                                                              .drop(['region', 'tech', 'gen_name'], axis=1)
                                                              )
            region_gen_mapping_idx = region_gen_mapping_idx.repeat(len(df.index.get_level_values('timestamp').unique()))

            idx_map = pd.MultiIndex(levels=idx_zone.levels + region_gen_mapping_idx.levels,
                                    codes=idx_zone.codes + region_gen_mapping_idx.codes,
                                    names=idx_zone.names + region_gen_mapping_idx.names)
        else:
            idx_map = idx_zone

        df = pd.DataFrame(data=df.values.reshape(-1), index=idx_map)
        df_col = list(df.index.names)  # Gets names of all columns in df and places in list
        df_col.insert(0, df_col.pop(df_col.index("timestamp")))  # move timestamp to start of df
        df = df.reorder_levels(df_col, axis=0)
        df[0] = pd.to_numeric(df[0], downcast='float')

        return df

    def df_process_region(self) -> pd.DataFrame:
        """Format data which comes from the PLEXOS Region Class.

        Returns
        -------
        df : pd.DataFrame
            Processed Output, single value column with multiindex.

        """
        df = self.df.droplevel(level=["band", "property", "category"])
        df.index.rename('region', level='name', inplace=True)
        if not self.Region_Mapping.empty:  # checks if Region_Mapping contains data to merge, skips if empty
            mapping_idx = pd.MultiIndex.from_frame(self.metadata.regions(self.model)
                                                   .merge(self.Region_Mapping,
                                                          how="left",
                                                          on='region')
                                                   .drop(['region', 'category'], axis=1)
                                                   )
            mapping_idx = mapping_idx.repeat(len(df.index.get_level_values('timestamp').unique()))

            idx = pd.MultiIndex(levels=df.index.levels + mapping_idx.levels,
                                codes=df.index.codes + mapping_idx.codes,
                                names=df.index.names + mapping_idx.names)
        else:
            idx = df.index
        df = pd.DataFrame(data=df.values.reshape(-1), index=idx)
        df_col = list(df.index.names)  # Gets names of all columns in df and places in list
        df_col.insert(0, df_col.pop(df_col.index("timestamp")))  # Move timestamp to start of df
        df = df.reorder_levels(df_col, axis=0)
        df[0] = pd.to_numeric(df[0], downcast='float')
        return df

    def df_process_zone(self) -> pd.DataFrame:
        """
        Method for formating data which comes from the PLEXOS Zone Class

        Returns
        -------
        df : pd.DataFrame
            Processed Output, single value column with multiindex.

        """
        df = self.df.droplevel(level=["band", "property", "category"])
        df.index.rename('zone', level='name', inplace=True)
        df = pd.DataFrame(data=df.values.reshape(-1), index=df.index)
        df_col = list(df.index.names)  # Gets names of all columns in df and places in list
        df_col.insert(0, df_col.pop(df_col.index("timestamp")))  # move timestamp to start of df
        df = df.reorder_levels(df_col, axis=0)
        df[0] = pd.to_numeric(df[0], downcast='float')
        return df

    def df_process_line(self) -> pd.DataFrame:
        """
        Method for formatting data which comes form the PLEXOS Line Class

        Returns
        -------
        df : pd.DataFrame
            Processed Output, single value column with multiindex.

        """
        df = self.df.droplevel(level=["band", "property", "category"])
        df.index.rename('line_name', level='name', inplace=True)
        df = pd.DataFrame(data=df.values.reshape(-1), index=df.index)
        df_col = list(df.index.names)  # Gets names of all columns in df and places in list
        df_col.insert(0, df_col.pop(df_col.index("timestamp")))  # move timestamp to start of df
        df = df.reorder_levels(df_col, axis=0)
        df[0] = pd.to_numeric(df[0], downcast='float')
        return df

    def df_process_interface(self) -> pd.DataFrame:
        """
        Method for formatting data which comes form the PLEXOS Interface Class

        Returns
        -------
        df : pd.DataFrame
            Processed Output, single value column with multiindex.

        """
        df = self.df.droplevel(level=["band", "property"])
        df.index.rename(['interface_name', 'interface_category'], level=['name', 'category'], inplace=True)
        df = pd.DataFrame(data=df.values.reshape(-1), index=df.index)
        df_col = list(df.index.names)  # Gets names of all columns in df and places in list
        df_col.insert(0, df_col.pop(df_col.index("timestamp")))  # move timestamp to start of df
        df = df.reorder_levels(df_col, axis=0)
        df[0] = pd.to_numeric(df[0], downcast='float')
        return df

    def df_process_reserve(self) -> pd.DataFrame:
        """
        Method for formatting data which comes form the PLEXOS Reserve Class

        Returns
        -------
        df : pd.DataFrame
            Processed Output, single value column with multiindex.

        """
        df = self.df.droplevel(level=["band", "property"])
        df.index.rename(['parent', 'Type'], level=['name', 'category'], inplace=True)
        df = df.reset_index()  # unzip the levels in index
        if self.metadata.reserves_regions(self.model).empty is False:
            df = df.merge(self.metadata.reserves_regions(self.model), how='left', on='parent')  # Merges in regions where reserves are located
        if self.metadata.reserves_zones(self.model).empty is False:
            df = df.merge(self.metadata.reserves_zones(self.model), how='left', on='parent')  # Merges in zones where reserves are located
        df_col = list(df.columns)  # Gets names of all columns in df and places in list
        df_col.remove(0)
        df_col.insert(0, df_col.pop(df_col.index("timestamp")))  # move timestamp to start of df
        df.set_index(df_col, inplace=True)
        df[0] = pd.to_numeric(df[0], downcast='float')
        return df

    def df_process_reserves_generators(self) -> pd.DataFrame:
        """
        Method for formatting data which comes form the PLEXOS Reserve_Generators Relational Class

        Returns
        -------
        df : pd.DataFrame
            Processed Output, single value column with multiindex.

        """
        df = self.df.droplevel(level=["band", "property"])
        df.index.rename(['gen_name'], level=['child'], inplace=True)
        df = df.reset_index()  # unzip the levels in index
        df = df.merge(self.metadata.generator_category(self.model), how='left', on='gen_name')

        # merging in generator region/zones first prevents double counting in cases where multiple model regions are within a reserve region
        if self.metadata.region_generators(self.model).empty is False:
            df = df.merge(self.metadata.region_generators(self.model), how='left', on='gen_name')
        if self.metadata.zone_generators(self.model).empty is False:
            df = df.merge(self.metadata.zone_generators(self.model), how='left', on='gen_name')

        # now merge in reserve regions/zones
        if self.metadata.reserves_regions(self.model).empty is False:
            df = df.merge(self.metadata.reserves_regions(self.model), how='left', on=['parent', 'region'])  # Merges in regions where reserves are located
        if self.metadata.reserves_zones(self.model).empty is False:
            df = df.merge(self.metadata.reserves_zones(self.model), how='left', on=['parent', 'zone'])  # Merges in zones where reserves are located

        df_col = list(df.columns)  # Gets names of all columns in df and places in list
        df_col.remove(0)
        df_col.insert(0, df_col.pop(df_col.index("timestamp")))  # move timestamp to start of df
        df.set_index(df_col, inplace=True)
        df[0] = pd.to_numeric(df[0], downcast='float')
        return df

    def df_process_fuel(self) -> pd.DataFrame:
        """
        Methodfor formatting data which comes form the PLEXOS Fuel Class

        Returns
        -------
        df : pd.DataFrame
            Processed Output, single value column with multiindex.

        """
        df = self.df.droplevel(level=["band", "property", "category"])
        df.index.rename('fuel_type', level='name', inplace=True)
        df = pd.DataFrame(data=df.values.reshape(-1), index=df.index)
        df_col = list(df.index.names)  # Gets names of all columns in df and places in list
        df_col.insert(0, df_col.pop(df_col.index("timestamp")))  # move timestamp to start of df
        df = df.reorder_levels(df_col, axis=0)
        df[0] = pd.to_numeric(df[0], downcast='float')
        return df

    def df_process_constraint(self) -> pd.DataFrame:
        """
        Method for formatting data which comes form the PLEXOS Constraint Class

        Returns
        -------
        df : pd.DataFrame
            Processed Output, single value column with multiindex.

        """
        df = self.df.droplevel(level=["band", "property"])
        df.index.rename(['constraint_category', 'constraint'], level=['category', 'name'], inplace=True)
        df = pd.DataFrame(data=df.values.reshape(-1), index=df.index)
        df_col = list(df.index.names)  # Gets names of all columns in df and places in list
        df_col.insert(0, df_col.pop(df_col.index("timestamp")))  # move timestamp to start of df
        df = df.reorder_levels(df_col, axis=0)
        df[0] = pd.to_numeric(df[0], downcast='float')
        return df

    def df_process_emission(self) -> pd.DataFrame:
        """
        Method for formatting data which comes form the PLEXOS Emission Class

        Returns
        -------
        df : pd.DataFrame
            Processed Output, single value column with multiindex.

        """
        df = self.df.droplevel(level=["band", "property"])
        df.index.rename('emission_type', level='name', inplace=True)
        df = pd.DataFrame(data=df.values.reshape(-1), index=df.index)
        df_col = list(df.index.names)  # Gets names of all columns in df and places in list
        df_col.insert(0, df_col.pop(df_col.index("timestamp")))  # move timestamp to start of df
        df = df.reorder_levels(df_col, axis=0)
        df[0] = pd.to_numeric(df[0], downcast='float')
        return df

    def df_process_emissions_generators(self) -> pd.DataFrame:
        """
        Method for formatting data which comes from the PLEXOS Emissions_Generators Relational Class

        Returns
        -------
        df : pd.DataFrame
            Processed Output, single value column with multiindex.

        """
        df = self.df.droplevel(level=["band", "property"])
        df.index.rename(['gen_name'], level=['child'], inplace=True)
        df.index.rename(['pollutant'], level=['parent'], inplace=True)

        df = df.reset_index()  # unzip the levels in index
        df = df.merge(self.metadata.generator_category(self.model), how='left', on='gen_name') # merge in tech information

        # merge in region and zone information
        if self.metadata.region_generator_category(self.model).empty is False:
            # merge in region information
            df = df.merge(self.metadata.region_generator_category(self.model).reset_index(), how='left', on=['gen_name', 'tech'])
        if self.metadata.zone_generator_category(self.model).empty is False:
            df = df.merge(self.metadata.zone_generator_category(self.model).reset_index(), how='left', on=['gen_name', 'tech'])  # Merges in zones where reserves are located

        if not self.Region_Mapping.empty:
            df = df.merge(self.Region_Mapping, how="left", on="region")

        if not self.emit_names.empty:
            # reclassify emissions as specified by user in mapping
            df['pollutant'] = pd.Categorical(df['pollutant'].map(lambda x: self.emit_names_dict.get(x, x)))

        # remove categoricals (otherwise h5 save will fail)
        df = df.astype({'tech': 'object', 'pollutant': 'object'})

        # Checks if all emissions categorieses have been identified and matched. If not, lists categories that need a match
        if not self.emit_names.empty:
            if self.emit_names_dict != {} and (set(df['pollutant'].unique()).issubset(self.emit_names["New"].unique())) is False:
                missing_emit_cat = list((set(df['pollutant'].unique())) - (set(self.emit_names["New"].unique())))
                self.logger.warning(f"The following emission objects do not have a correct category mapping: {missing_emit_cat}\n")

        df_col = list(df.columns)  # Gets names of all columns in df and places in list
        df_col.remove(0)
        df_col.insert(0, df_col.pop(df_col.index("timestamp")))  # move timestamp to start of df
        df.set_index(df_col, inplace=True)
        # downcast values to save on memory
        df[0] = pd.to_numeric(df[0].values, downcast='float')
        # convert to range index (otherwise h5 save will fail)
        df.columns = pd.RangeIndex(0, 1, step=1)
        return df

    def df_process_storage(self) -> pd.DataFrame:
        """
        Method for formatting data which comes form the PLEXOS Storage Class

        Returns
        -------
        df : pd.DataFrame
            Processed Output, single value column with multiindex.

        """
        df = self.df.droplevel(level=["band", "property", "category"])
        df = df.reset_index()  # unzip the levels in index
        df = df.merge(self.metadata.generator_storage(self.model), how='left', on='name')
        if self.metadata.region_generators(self.model).empty is False:
            df = df.merge(self.metadata.region_generators(self.model), how='left', on='gen_name')  # Merges in regions where generators are located
        if self.metadata.zone_generators(self.model).empty is False:
            df = df.merge(self.metadata.zone_generators(self.model), how='left', on='gen_name')  # Merges in zones where generators are located
        if not self.Region_Mapping.empty:  # checks if Region_Maping contains data to merge, skips if empty (Default)
            df = df.merge(self.Region_Mapping, how='left', on='region')  # Merges in all Region Mappings
        df.rename(columns={'name': 'storage_resource'}, inplace=True)
        df_col = list(df.columns)  # Gets names of all columns in df and places in list
        df_col.remove(0)  # Removes 0, the data column from the list
        df_col.insert(0, df_col.pop(df_col.index("timestamp")))  # move timestamp to start of df
        df.set_index(df_col, inplace=True)
        df[0] = pd.to_numeric(df[0], downcast='float')
        return df

    def df_process_region_regions(self) -> pd.DataFrame:
        """
        Method for formatting data which comes form the PLEXOS Region_Regions Relational Class

        Returns
        -------
        df : pd.DataFrame
            Processed Output, single value column with multiindex.

        """
        df = self.df.droplevel(level=["band", "property"])
        df = pd.DataFrame(data=df.values.reshape(-1), index=df.index)
        df_col = list(df.index.names)  # Gets names of all columns in df and places in list
        df_col.insert(0, df_col.pop(df_col.index("timestamp")))  # move timestamp to start of df
        df = df.reorder_levels(df_col, axis=0)
        df[0] = pd.to_numeric(df[0], downcast='float')
        return df

    def df_process_node(self) -> pd.DataFrame:
        """
        Method for formatting data which comes form the PLEXOS Node Class

        Returns
        -------
        df : pd.DataFrame
            Processed Output, single value column with multiindex.

        """
        df = self.df.droplevel(level=["band", "property", "category"])
        df.index.rename('node', level='name', inplace=True)
        df.sort_index(level=['node'], inplace=True)
        if self.metadata.node_region(self.model).empty is False:
            node_region_idx = pd.CategoricalIndex(self.metadata.node_region(self.model).index.get_level_values(0))
            node_region_idx = node_region_idx.repeat(len(df.index.get_level_values('timestamp').unique()))
            idx_region = pd.MultiIndex(levels=df.index.levels + [node_region_idx.categories],
                                       codes=df.index.codes + [node_region_idx.codes],
                                       names=df.index.names + node_region_idx.names)
        else:
            idx_region = df.index
        if self.metadata.node_zone(self.model).empty is False:
            node_zone_idx = pd.CategoricalIndex(self.metadata.node_zone(self.model).index.get_level_values(0))
            node_zone_idx = node_zone_idx.repeat(len(df.index.get_level_values('timestamp').unique()))
            idx_zone = pd.MultiIndex(levels=idx_region.levels + [node_zone_idx.categories],
                                     codes=idx_region.codes + [node_zone_idx.codes],
                                     names=idx_region.names + node_zone_idx.names)
        else:
            idx_zone = idx_region
        if not self.Region_Mapping.empty:
            region_mapping_idx = pd.MultiIndex.from_frame(self.metadata.node_region(self.model)
                                                          .merge(self.Region_Mapping,
                                                                 how="left",
                                                                 on='region')
                                                          .drop(['region', 'node'], axis=1)
                                                          )
            region_mapping_idx = region_mapping_idx.repeat(len(df.index.get_level_values('timestamp').unique()))

            idx_map = pd.MultiIndex(levels=idx_zone.levels + region_mapping_idx.levels,
                                    codes=idx_zone.codes + region_mapping_idx.codes,
                                    names=idx_zone.names + region_mapping_idx.names)
        else:
            idx_map = idx_zone

        df = pd.DataFrame(data=df.values.reshape(-1), index=idx_map)
        df_col = list(df.index.names)  # Gets names of all columns in df and places in list
        df_col.insert(0, df_col.pop(df_col.index("timestamp")))  # move timestamp to start of df
        df = df.reorder_levels(df_col, axis=0)
        df[0] = pd.to_numeric(df[0], downcast='float')
        return df


class MarmotFormat(SetupLogger):
    """Main module class to be instantiated to run the formatter.

    MarmotFormat reads in PLEXOS hdf5 files created with the h5plexos library
    and processes the output results to ready them for plotting.
    Once the outputs have been processed, they are saved to an intermediary hdf5 file
    which can then be read into the Marmot plotting code
    """

    def __init__(self, Scenario_name, PLEXOS_Solutions_folder, Plexos_Properties,
                 Marmot_Solutions_folder=None,
                 mapping_folder='mapping_folder',
                 Region_Mapping=pd.DataFrame(),
                 emit_names=pd.DataFrame(),
                 VoLL=10000,
                 **kwargs):
        """Marmotformat class __init__ method.

        Parameters
        ----------
        Scenario_name : string
            Name of sceanrio to process.
        PLEXOS_Solutions_folder : string directory
            Folder containing h5plexos results files.
        Plexos_Properties : string directory/pd.DataFrame
            PLEXOS properties to process, must follow format seen in Marmot directory.
        Marmot_Solutions_folder : string directory, optional
            Folder to save Marmot solution files. The default is None.
        mapping_folder : string directory, optional
            The location of the Marmot mapping folder. The default is 'mapping_folder'.
        Region_Mapping : string directory/pd.DataFrame, optional
            Mapping file to map custom regions/zones to create custom aggregations.
            Aggregations are created by grouping PLEXOS regions.
            The default is pd.DataFrame().
        emit_names : string directory/pd.DataFrame, optional
            Mapping file to reanme emissions types.
            The default is pd.DataFrame().
        VoLL : int, optional
            Value of lost load, used to calculate cost of unserved energy.
            The default is 10000.

        Returns
        -------
        None.

        """
        super().__init__(**kwargs) # Instantiation of SetupLogger

        self.Scenario_name = Scenario_name
        self.PLEXOS_Solutions_folder = PLEXOS_Solutions_folder
        self.Marmot_Solutions_folder = Marmot_Solutions_folder
        self.mapping_folder = mapping_folder
        self.VoLL = VoLL

        if self.Marmot_Solutions_folder is None:
            self.Marmot_Solutions_folder = self.PLEXOS_Solutions_folder

        if isinstance(Plexos_Properties, str):
            try:
                self.Plexos_Properties = pd.read_csv(Plexos_Properties)
            except FileNotFoundError:
                self.logger.warning('Could not find specified Plexos_Properties file; check file name. This is required to run Marmot, system will now exit')
                sys.exit()
        elif isinstance(Plexos_Properties, pd.DataFrame):
            self.Plexos_Properties = Plexos_Properties

        if isinstance(Region_Mapping, str):
            try:
                self.Region_Mapping = pd.read_csv(Region_Mapping)
                if not self.Region_Mapping.empty:
                    self.Region_Mapping = self.Region_Mapping.astype(str)
            except FileNotFoundError:
                self.logger.warning('Could not find specified Region Mapping file; check file name\n')
                self.Region_Mapping = pd.DataFrame()
        elif isinstance(Region_Mapping, pd.DataFrame):
            self.Region_Mapping = Region_Mapping
            if not self.Region_Mapping.empty:
                self.Region_Mapping = self.Region_Mapping.astype('string')
        try:
            self.Region_Mapping = self.Region_Mapping.drop(["category"], axis=1)  # delete category columns if exists
        except KeyError:
            pass

        if isinstance(emit_names, str):
            try:
                self.emit_names = pd.read_csv(emit_names)
                if not self.emit_names.empty:
                    self.emit_names.rename(columns={self.emit_names.columns[0]: 'Original',
                                                    self.emit_names.columns[1]: 'New'},
                                           inplace=True)
            except FileNotFoundError:
                self.logger.warning('Could not find specified emissions mapping file; check file name\n')
                self.emit_names = pd.DataFrame()
        elif isinstance(emit_names, pd.DataFrame):
            self.emit_names = emit_names
            if not self.emit_names.empty:
                self.emit_names.rename(columns={self.emit_names.columns[0]: 'Original',
                                                self.emit_names.columns[1]: 'New'},
                                       inplace=True)


    def output_metadata(self, files_list, hdf_out_folder, HDF5_output, HDF5_folder_in) -> None:
        """ 
        This function is used to output metadata from the original PLEXOS solutions
        file to the processed HDF5 file.  For each partition in a given scenario,
        the metadata from that partition is copied over and saved in the processed output file.
        This function is called within the run_formatter method of this class.
        """

        for partition in files_list:
            f = h5py.File(os.path.join(HDF5_folder_in, partition),'r')
            meta_keys = [key for key in f['metadata'].keys()]

            group_dict = {}
            for key in meta_keys:
                sub_dict = {}
                subkeys = [key for key in f['metadata'][key].keys()]
                for sub in subkeys:
                    dset = f['metadata'][key][sub]
                    sub_dict[sub] = dset
                group_dict[key] = sub_dict

            with h5py.File(os.path.join(hdf_out_folder, HDF5_output),"a") as g:
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
                        subgrp.create_dataset(name=key2,data=dset)
            f.close()


    def _get_data(self, plexos_class, plexos_prop, timescale, db, metadata) -> pd.DataFrame:
        """
        This method handles the pulling of the data from the H5plexos hdf5
        file and then passes the data to one of the formating functions

        Parameters
        ----------
        plexos_class : string
            PLEXOS calss e.g Region, Generator, Zone etc.
        plexos_prop : string
            PLEXOS property e.g Max Capacity, Generation etc.
        timescale : string
            Data timescale, e.g Hourly, Monthly, 5 minute etc.
        db : h5plexos.query.solution.PLEXOSSolution (class instantiation)
            Instantiation of h5plexos PLEXOSSolution for specific h5plexos file.
        metadata : meta_data.MetaData (class instantiation)
            Instantiation of MetaData for specific h5plexos file.

        Returns
        -------
        df : pd.DataFrame()
            Formatted results dataframe.

        """
        try:
            if "_" in plexos_class:
                df = db.query_relation_property(plexos_class, plexos_prop, timescale=timescale)
                object_class = plexos_class
            else:
                df = db.query_object_property(plexos_class, plexos_prop, timescale=timescale)
                
                # handles h5plexos naming discrepency 
                if ((0,6,0) <= db.version and db.version < (0,7,0)):
                    object_class = f"{plexos_class}s"
        
        except KeyError:
            df = self._report_prop_error(plexos_prop, plexos_class)
            return df
        
        # Get original units from h5plexos file 
        df_units = db.h5file[f'/data/ST/{timescale}/{object_class}/{plexos_prop}'].attrs['units'].decode('UTF-8')
        # find unit conversion values
        converted_units = UNITS_CONVERSION.get(df_units, (df_units, 1))


        # Instantiate instance of Process Class
        # metadata is used as a paramter to initialize process_cl
        process_cl = Process(df, metadata, db.h5file.filename, self.Region_Mapping, self.emit_names, self.logger)
        # Instantiate Method of Process Class
        process_att = getattr(process_cl, f'df_process_{plexos_class}')
        # Process attribute and return to df
        df = process_att()
        
        # Convert units and add unit column to index 
        df = df*converted_units[1]
        units_index = pd.Index([converted_units[0]] *len(df), name='units')
        df.set_index(units_index, append=True, inplace=True)

        if plexos_class == 'region' and plexos_prop == "Unserved Energy" and int(df.sum(axis=0)) > 0:
            self.logger.warning(f"Scenario contains Unserved Energy: {int(df.sum(axis=0))} MW\n")
        return df

    def _report_prop_error(self, plexos_prop, plexos_class) -> pd.DataFrame:
        """
        This method prints a warning message when the _get_data method
        cannot find the specified PLEXOS property in the h5plexos hdf5 file

        Parameters
        ----------
        plexos_class : string
            PLEXOS calss e.g Region, Generator, Zone etc.
        plexos_prop : string
            PLEXOS property e.g Max Capacity, Generation etc.

        Returns
        -------
        df : pd.DataFrame
            Empty DataFrame.

        """
        self.logger.warning(f'CAN NOT FIND "{plexos_class} {plexos_prop}". "{plexos_prop}" DOES NOT EXIST')
        self.logger.info('SKIPPING PROPERTY\n')
        df = pd.DataFrame()
        return df

    @staticmethod
    def _save_to_h5(df, file_name, key, 
                    mode="a", complevel=9, 
                    complib='blosc:zlib', **kwargs) -> None:
        """Method to save data to hdf5 file """

        df.to_hdf(file_name, key=key, mode=mode,
                    complevel=complevel,
                    complib=complib,
                    **kwargs)

    def run_formatter(self) -> None:
        """
        Main method to call to begin processing h5plexos files, this method takes
        no input variables, all required varibales are passed in via the __init__ method.

        Returns
        -------
        None.

        """

        self.logger.info(f"#### Processing {self.Scenario_name} PLEXOS Results ####")

        # ===============================================================================
        # Input and Output Directories
        # ===============================================================================

        HDF5_output = f"{self.Scenario_name}_formatted.h5"

        HDF5_folder_in = os.path.join(self.PLEXOS_Solutions_folder, str(self.Scenario_name))
        try:
            os.makedirs(HDF5_folder_in)
        except FileExistsError:
            # directory already exists
            pass

        hdf_out_folder = os.path.join(self.Marmot_Solutions_folder, 'Processed_HDF5_folder')
        try:
            os.makedirs(hdf_out_folder)
        except FileExistsError:
            # directory already exists
            pass

        startdir = os.getcwd()
        os.chdir(HDF5_folder_in)  # Due to a bug on eagle need to chdir before listdir

        files = []
        for names in os.listdir():
            if names.endswith(".h5"):
                files.append(names)  # Creates a list of only the hdf5 files

        # List of all hf files in hdf5 folder in alpha numeric order
        files_list = sorted(files, key=lambda x:int(re.sub('\D', '', x)))

        os.chdir(startdir)

        # Read in all HDF5 files into dictionary
        self.logger.info("Loading all HDF5 files to prepare for processing")
        hdf5_collection = {}
        for file in files_list:
            hdf5_collection[file] = PLEXOSSolution(os.path.join(HDF5_folder_in, file))

        # ===================================================================================
        # Process the Outputs
        # ===================================================================================

        # Creates Initial HDF5 file for ouputing formated data
        Processed_Data_Out = pd.DataFrame()
        if os.path.isfile(os.path.join(hdf_out_folder, HDF5_output)) is True:
            self.logger.info(f"'{hdf_out_folder}\{HDF5_output}' already exists: New variables will be added\n")
            # Skip properties that already exist in *formatted.h5 file.
            with h5py.File(os.path.join(hdf_out_folder, HDF5_output), 'r') as f:
                existing_keys = [key for key in f.keys()]

            # The processed HDF5 output file already exists.  If metadata is already in
            # this file, leave as is.  Otherwise, append it to the file.
            if 'metadata' not in existing_keys:
                self.logger.info('Adding metadata to processed HDF5 file.')
                self.output_metadata(files_list, hdf_out_folder, HDF5_output, HDF5_folder_in)

            if not mconfig.parser('skip_existing_properties'):
                existing_keys = []

        # The processed HDF5 file does not exist.  Create the file and add metadata to it.
        else:
            existing_keys = []
            
            # Create empty hdf5 file 
            f = h5py.File(os.path.join(hdf_out_folder, HDF5_output), "w")
            f.close()

            self.output_metadata(files_list, hdf_out_folder, HDF5_output, HDF5_folder_in)

        process_properties = self.Plexos_Properties.loc[self.Plexos_Properties["collect_data"] == True]
        
        # Create an instance of metadata, and pass that as a variable to get data.
        meta = MetaData(HDF5_folder_in, read_from_formatted_h5=False, Region_Mapping=self.Region_Mapping)
                    
        if not self.Region_Mapping.empty:
            # if any(meta.regions()['region'] not in Region_Mapping['region']):
            if set(meta.regions(files_list[0])['region']).issubset(self.Region_Mapping['region']) is False:
                missing_regions = list(set(meta.regions(files_list[0])['region']) - set(self.Region_Mapping['region']))

                self.logger.warning(f'The Following PLEXOS REGIONS are missing from the "region" column of your mapping file: {missing_regions}\n',)

        start = time.time()
        # Main loop to process each ouput and pass data to functions
        for index, row in process_properties.iterrows():
            Processed_Data_Out = pd.DataFrame()
            data_chunks = []

            self.logger.info(f'Processing {row["group"]} {row["data_set"]}')
            prop_underscore = row["data_set"].replace(' ', '_')
            key_path = row["group"] + "_" + prop_underscore
            if key_path not in existing_keys:

                for model in files_list:
                    self.logger.info(f"      {model}")

                    db = hdf5_collection.get(model)
                    processed_data = self._get_data(row["group"], row["data_set"], row["data_type"], db, meta)

                    if processed_data.empty is True:
                        break
                    
                    # Check if data is for year interval and of type capacity
                    if (row["data_type"] == "year") & (
                            (row["data_set"] == "Installed Capacity")
                            | (row["data_set"] == "Export Limit")
                            | (row["data_set"] == "Import Limit")
                            ):
                        data_chunks.append(processed_data)
                        self.logger.info(f"{row['data_set']} Year property reported from only the first partition")
                        break
                    else:
                        data_chunks.append(processed_data)

                if data_chunks:
                    Processed_Data_Out = pd.concat(data_chunks, copy=False)

                if Processed_Data_Out.empty is False:
                    if (row["data_type"] == "year"):
                        self.logger.info("Please Note: Year properties can not be checked for duplicates.\n\
                        Overlaping data cannot be removed from 'Year' grouped data.\n\
                        This will effect Year data that differs between partitions such as cost results.\n\
                        It will not effect Year data that is equal in all partitions such as Installed Capacity or Line Limit results")

                    else:
                        oldsize = Processed_Data_Out.size
                        Processed_Data_Out = Processed_Data_Out.loc[~Processed_Data_Out.index.duplicated(keep='first')]  # Remove duplicates; keep first entry
                        if (oldsize - Processed_Data_Out.size) > 0:
                            self.logger.info(f'Drop duplicates removed {oldsize-Processed_Data_Out.size} rows')

                    row["data_set"] = row["data_set"].replace(' ', '_')
                    
                    save_attempt=1
                    while save_attempt<=3:
                        try:
                            self.logger.info("Saving data to h5 file...")
                            MarmotFormat._save_to_h5(Processed_Data_Out,
                                        os.path.join(hdf_out_folder, HDF5_output), 
                                        key=f'{row["group"]}_{row["data_set"]}')

                            self.logger.info("Data saved to h5 file successfully\n")
                            save_attempt=4
                        except:
                            self.logger.warning("h5 File is probably in use, waiting to attempt to save again")
                            time.sleep(60)
                            save_attempt+=1
                else:
                    continue
            else:
                self.logger.info(f"{key_path} already exists in output .h5 file.")
                self.logger.info("PROPERTY ALREADY PROCESSED\n")
                continue

        # ===================================================================================
        # Calculate Extra Ouputs
        # ===================================================================================
        if "generator_Curtailment" not in h5py.File(os.path.join(hdf_out_folder, HDF5_output), 'r') or not mconfig.parser('skip_existing_properties'):
            try:
                self.logger.info("Processing generator Curtailment")
                try:
                    Avail_Gen_Out = pd.read_hdf(os.path.join(hdf_out_folder,
                                                             HDF5_output),
                                                'generator_Available_Capacity')
                    Total_Gen_Out = pd.read_hdf(os.path.join(hdf_out_folder,
                                                             HDF5_output),
                                                'generator_Generation')
                    if Total_Gen_Out.empty is True:
                        self.logger.warning("generator_Available_Capacity & generator_Generation are required for Curtailment calculation")
                except KeyError:
                    self.logger.warning("generator_Available_Capacity & generator_Generation are required for Curtailment calculation")

                Curtailment_Out = Avail_Gen_Out - Total_Gen_Out

                Upward_Available_Capacity = Curtailment_Out

                MarmotFormat._save_to_h5(Curtailment_Out,
                                    os.path.join(hdf_out_folder, HDF5_output), 
                                    key="generator_Curtailment")

                MarmotFormat._save_to_h5(Upward_Available_Capacity,
                                    os.path.join(hdf_out_folder, HDF5_output), 
                                    key="generator_Upward_Available_Capacity")

                self.logger.info("Data saved to h5 file successfully\n")
                # Clear Some Memory
                del Total_Gen_Out
                del Avail_Gen_Out
                del Curtailment_Out
            except Exception:
                self.logger.warning("NOTE!! Curtailment not calculated, processing skipped\n")

        if "region_Cost_Unserved_Energy" not in h5py.File(os.path.join(hdf_out_folder, HDF5_output), 'r') or not mconfig.parser('skip_existing_properties'):
            try:
                self.logger.info("Calculating Cost Unserved Energy: Regions")
                Cost_Unserved_Energy = pd.read_hdf(os.path.join(hdf_out_folder,
                                                                HDF5_output),
                                                   'region_Unserved_Energy')
                                                   
                Cost_Unserved_Energy = Cost_Unserved_Energy * self.VoLL

                MarmotFormat._save_to_h5(Cost_Unserved_Energy,
                                    os.path.join(hdf_out_folder, HDF5_output), 
                                    key="region_Cost_Unserved_Energy")
            except KeyError:
                self.logger.warning("NOTE!! Regional Unserved Energy not available to process, processing skipped\n")
                pass

        if "zone_Cost_Unserved_Energy" not in h5py.File(os.path.join(hdf_out_folder, HDF5_output), 'r') or not mconfig.parser('skip_existing_properties'):
            try:
                self.logger.info("Calculating Cost Unserved Energy: Zones")
                Cost_Unserved_Energy = pd.read_hdf(os.path.join(hdf_out_folder,
                                                                HDF5_output),
                                                   'zone_Unserved_Energy')
                Cost_Unserved_Energy = Cost_Unserved_Energy * self.VoLL

                MarmotFormat._save_to_h5(Cost_Unserved_Energy,
                                    os.path.join(hdf_out_folder, HDF5_output), 
                                    key="zone_Cost_Unserved_Energy")
            except KeyError:
                self.logger.warning("NOTE!! Zonal Unserved Energy not available to process, processing skipped\n")
                pass

        end = time.time()
        elapsed = end - start
        self.logger.info('Main loop took %s minutes', round(elapsed/60, 2))
        self.logger.info(f'Formatting COMPLETED for {self.Scenario_name}')


def main():
    '''
    The following code is run if the formatter is run directly,
    it does not run if the formatter is imported as a module.
    '''

    # ===============================================================================
    # Input Properties
    # ===============================================================================

    # Changes working directory to location of this python file
    os.chdir(FILE_DIR)

    Marmot_user_defined_inputs = pd.read_csv(mconfig.parser("user_defined_inputs_file"),
                                            usecols=['Input', 'User_defined_value'],
                                            index_col='Input',
                                            skipinitialspace=True)

    # File which determiens which plexos properties to pull from the h5plexos results and process, this file is in the repo
    Plexos_Properties = pd.read_csv(mconfig.parser('plexos_properties_file'))
    
    # Name of the Scenario(s) being run, must have the same name(s) as the folder holding the runs HDF5 file
    Scenario_List = pd.Series(Marmot_user_defined_inputs.loc['Scenario_process_list'].squeeze().split(",")).str.strip().tolist()
    # The folder that contains all PLEXOS h5plexos outputs - the h5 files should be contained in another folder with the Scenario_name
    PLEXOS_Solutions_folder = Marmot_user_defined_inputs.loc['PLEXOS_Solutions_folder'].to_string(index=False).strip()

    # Folder to save your processed solutions
    if pd.isna(Marmot_user_defined_inputs.loc['Marmot_Solutions_folder','User_defined_value']):
        Marmot_Solutions_folder = None
    else:
        Marmot_Solutions_folder = Marmot_user_defined_inputs.loc['Marmot_Solutions_folder'].to_string(index=False).strip()

    # This folder contains all the csv required for mapping and selecting outputs to process
    # Examples of these mapping files are within the Marmot repo, you may need to alter these to fit your needs
    Mapping_folder = 'mapping_folder'

    if pd.isna(Marmot_user_defined_inputs.loc['Region_Mapping.csv_name', 'User_defined_value']) is True:
        Region_Mapping = pd.DataFrame()
    else:
        Region_Mapping = pd.read_csv(os.path.join(Mapping_folder, Marmot_user_defined_inputs.loc['Region_Mapping.csv_name'].to_string(index=False).strip()))

    # Value of Lost Load for calculatinhg cost of unserved energy
    VoLL = pd.to_numeric(Marmot_user_defined_inputs.loc['VoLL'].to_string(index=False))

    # ===============================================================================
    # Standard Naming of Emissions types (optional)
    # ===============================================================================

    emit_names = os.path.join(Mapping_folder, Marmot_user_defined_inputs.loc['emit_names.csv_name'].to_string(index=False).strip())

    # ===============================================================================
    # Loop through scenarios in list
    # ===============================================================================

    for Scenario_name in Scenario_List:

        initiate = MarmotFormat(Scenario_name, PLEXOS_Solutions_folder, Plexos_Properties,
                                Marmot_Solutions_folder=Marmot_Solutions_folder,
                                mapping_folder=Mapping_folder,
                                Region_Mapping=Region_Mapping,
                                emit_names=emit_names,
                                VoLL=VoLL)

        initiate.run_formatter()


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

# test.index.get_level_values('region') = test.index.get_level_values('region').astype("category")

# test['timestamp'] = test['timestamp'].astype("category")

# test.index = test.index.set_levels(test.index.levels[-1].astype('category'), level=-1)

# test.memory_usage(deep=True)
# test[0] = pd.to_numeric(test[0], downcast='float')

# test.memory_usage(deep=False)

# Stacked_Gen_read = Stacked_Gen_read.reset_index() # unzip the levels in index
# Stacked_Gen_read.rename(columns={'name':'zone'}, inplace=True)
#         Stacked_Gen_read = Stacked_Gen_read.drop(["band", "property", "category"],axis=1)
    # if int(Stacked_Gen_read.sum(axis=0)) >= 0:
    #     print("WARNING! Scenario contains Unserved Energy: " + str(int(Stacked_Gen_read.sum(axis=0))) + "MW")

    #storage = db.storage("Generation")
    #storage = df_process_storage(storage, overlap)

# df_old = df
# t =df.loc[~df.index.duplicated()]
# df_old.equals(df)
