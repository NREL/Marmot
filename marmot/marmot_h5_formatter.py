# -*- coding: utf-8 -*-
"""
First Created on Wed May 22 14:29:48 2019

This code was written to process PLEXOS HDF5 outputs to get them ready for plotting.
Once the data is processed it is outputed as an intermediary HDF5 file format so that
it can be read into the marmot_plot_main.py file


@author: Daniel Levie
"""
#===============================================================================
# Import Python Libraries
#===============================================================================

import os
import sys
import pathlib
file_dir = pathlib.Path(__file__).parent.absolute()
if __name__ == '__main__': # Add Marmot directory to sys path if running from __main__
    if os.path.dirname(os.path.dirname(__file__)) not in sys.path:
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        os.chdir(pathlib.Path(__file__).parent.absolute().parent.absolute())
import time
import re
import pandas as pd
import h5py
import logging
import logging.config
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
    from h5plexos.h5plexos.query import PLEXOSSolution
except ModuleNotFoundError:
    from marmot.h5plexos.h5plexos.query import PLEXOSSolution

#===============================================================================
# Setup Logger
#===============================================================================

current_dir = os.getcwd()
os.chdir(file_dir)

with open('config/marmot_logging_config.yml', 'rt') as f:
    conf = yaml.safe_load(f.read())
    logging.config.dictConfig(conf)

logger = logging.getLogger('marmot_format')
# Creates a new log file for next run
logger.handlers[1].doRollover()
logger.handlers[2].doRollover()

os.chdir(current_dir)
#===============================================================================
# Create HDF5 file from PLEXOS zip solution
#===============================================================================
#This is only required if your output has not been processed already on Eagle

#from h5plexos.process import process_solution
#PLEXOS_Solution = '/path/to/PLEXOS/zipfile.zip'
#process_solution(PLEXOS_Solution,'/write/path/to/h5plexos/solution.h5') # Saves out to PLEXOS_Solution.h5

#===============================================================================

#A bug in pandas requires this to be included, otherwise df.to_string truncates long strings
#Fix available in Pandas 1.0 but leaving here in case user version not up to date
pd.set_option("display.max_colwidth", 1000)


class Process():
    '''
    Process Class contains methods for processing h5plexos query data
    '''

    def __init__(self, df, metadata, Region_Mapping, gen_names, emit_names):
        '''
        
        Parameters
        ----------
        df : pd.DataFrame
            Unprocessed h5plexos dataframe containing 
            class and property specifc data.
        metadata : meta_data.MetaData (class instantiation)
            Instantiation of MetaData for specific h5plexos file.
        Region_Mapping : pd.DataFrame
            DataFrame to map custom regions/zones to create custom aggregations.
        gen_names : pd.DataFrame
            DataFrame with 2 columns to rename generator technologies.
        emit_names : pd.DataFrame
            DataFrame with 2 columns to rename emmission names. 

        Returns
        -------
        None.

        '''
        
        # certain methods require information from metadata.  metadata is now
        # passed in as an instance of MetaData class for the appropriate model
        self.df = df
        self.metadata = metadata
        self.Region_Mapping = Region_Mapping
        self.gen_names = gen_names
        self.emit_names = emit_names
        
        if not self.emit_names.empty:
            self.emit_names_dict=self.emit_names[['Original','New']].set_index("Original").to_dict()["New"]
            
        self.gen_names_dict=self.gen_names[['Original','New']].set_index("Original").to_dict()["New"]



    def df_process_generator(self):
        '''
        Method for formatting data which comes form the PLEXOS Generator Class

        Returns
        -------
        df : pd.DataFrame
            Processed Output, single value column with multiindex.

        '''
        df = self.df.droplevel(level=["band", "property"])
        df.index.rename(['tech','gen_name'], level=['category','name'], inplace=True)

        if self.metadata.region_generator_category().empty == False:
            region_gen_idx = pd.CategoricalIndex(self.metadata.region_generator_category().index.get_level_values(0))
            region_gen_idx = region_gen_idx.repeat(len(df.index.get_level_values('timestamp').unique()))

            idx_region = pd.MultiIndex(levels= df.index.levels + [region_gen_idx.categories]
                                ,codes= df.index.codes +  [region_gen_idx.codes],
                                names= df.index.names + region_gen_idx.names)
        else:
            idx_region = df.index

        if self.metadata.zone_generator_category().empty == False:
            zone_gen_idx = pd.CategoricalIndex(self.metadata.zone_generator_category().index.get_level_values(0))
            zone_gen_idx = zone_gen_idx.repeat(len(df.index.get_level_values('timestamp').unique()))

            idx_zone = pd.MultiIndex(levels= idx_region.levels + [zone_gen_idx.categories]
                                ,codes= idx_region.codes + [zone_gen_idx.codes] ,
                                names= idx_region.names + zone_gen_idx.names)
        else:
            idx_zone = idx_region

        if not self.Region_Mapping.empty:
            region_gen_mapping_idx = pd.MultiIndex.from_frame(self.metadata.region_generator_category().merge(self.Region_Mapping,
                                how="left", on='region').sort_values(by=['tech','gen_name']).drop(['region','tech','gen_name'], axis=1))
            region_gen_mapping_idx = region_gen_mapping_idx.repeat(len(df.index.get_level_values('timestamp').unique()))

            idx_map = pd.MultiIndex(levels= idx_zone.levels + region_gen_mapping_idx.levels
                                ,codes= idx_zone.codes + region_gen_mapping_idx.codes,
                                names = idx_zone.names + region_gen_mapping_idx.names)
        else:
            idx_map = idx_zone

        idx_map = idx_map.droplevel(level=["tech"])
        df_tech = pd.CategoricalIndex(df.index.get_level_values('tech').map(lambda x: self.gen_names_dict.get(x,x)))

        idx =  pd.MultiIndex(levels= [df_tech.categories] + idx_map.levels
                                ,codes= [df_tech.codes] + idx_map.codes,
                                names = df_tech.names + idx_map.names)

        df = pd.DataFrame(data=df.values.reshape(-1), index=idx)
        df_col = list(df.index.names) # Gets names of all columns in df and places in list
        df_col.insert(0, df_col.pop(df_col.index("timestamp"))) #move timestamp to start of df
        df = df.reorder_levels(df_col, axis=0)
        df[0] = pd.to_numeric(df[0], downcast='float')

        # Checks if all generator tech categorieses have been identified and matched. If not, lists categories that need a match
        if set(df.index.unique(level="tech")).issubset(self.gen_names["New"].unique()) == False:
            missing_gen_cat = list((set(df.index.unique(level="tech"))) - (set(self.gen_names["New"].unique())))
            logger.warning("The Following Generators do not have a correct category mapping: %s\n",missing_gen_cat)
        return df

   
    def df_process_region(self):
        '''
        Function for formating data which comes from the PLEXOS Region Class

        Returns
        -------
        df : pd.DataFrame
            Processed Output, single value column with multiindex.

        '''
        df = self.df.droplevel(level=["band", "property", "category"])
        df.index.rename('region', level='name', inplace=True)
        if not self.Region_Mapping.empty: #checks if Region_Mapping contains data to merge, skips if empty
            mapping_idx = pd.MultiIndex.from_frame(self.metadata.regions().merge(self.Region_Mapping,
                                how="left", on='region').drop(['region','category'], axis=1))
            mapping_idx = mapping_idx.repeat(len(df.index.get_level_values('timestamp').unique()))

            idx = pd.MultiIndex(levels= df.index.levels + mapping_idx.levels
                                ,codes= df.index.codes + mapping_idx.codes,
                                names = df.index.names + mapping_idx.names)
        else:
            idx = df.index
        df = pd.DataFrame(data=df.values.reshape(-1), index=idx)
        df_col = list(df.index.names) # Gets names of all columns in df and places in list
        df_col.insert(0, df_col.pop(df_col.index("timestamp"))) #move timestamp to start of df
        df = df.reorder_levels(df_col, axis=0)
        df[0] = pd.to_numeric(df[0], downcast='float')
        return df

     
    def df_process_zone(self):
        '''
        Method for formating data which comes from the PLEXOS Zone Class

        Returns
        -------
        df : pd.DataFrame
            Processed Output, single value column with multiindex.

        '''
        df = self.df.droplevel(level=["band", "property", "category"])
        df.index.rename('zone', level='name', inplace=True)
        df = pd.DataFrame(data=df.values.reshape(-1), index=df.index)
        df_col = list(df.index.names) # Gets names of all columns in df and places in list
        df_col.insert(0, df_col.pop(df_col.index("timestamp"))) #move timestamp to start of df
        df = df.reorder_levels(df_col, axis=0)
        df[0] = pd.to_numeric(df[0], downcast='float')
        return df

    
    def df_process_line(self):
        '''
        Method for formatting data which comes form the PLEXOS Line Class

        Returns
        -------
        df : pd.DataFrame
            Processed Output, single value column with multiindex.

        '''
        df = self.df.droplevel(level=["band", "property", "category"])
        df.index.rename('line_name', level='name', inplace=True)
        df = pd.DataFrame(data=df.values.reshape(-1), index=df.index)
        df_col = list(df.index.names) # Gets names of all columns in df and places in list
        df_col.insert(0, df_col.pop(df_col.index("timestamp"))) #move timestamp to start of df
        df = df.reorder_levels(df_col, axis=0)
        df[0] = pd.to_numeric(df[0], downcast='float')
        return df

    
    def df_process_interface(self):
        '''
        Method for formatting data which comes form the PLEXOS Interface Class

        Returns
        -------
        df : pd.DataFrame
            Processed Output, single value column with multiindex.

        '''
        df = self.df.droplevel(level=["band", "property"])
        df.index.rename(['interface_name', 'interface_category'], level=['name','category'], inplace=True)
        df = pd.DataFrame(data=df.values.reshape(-1), index=df.index)
        df_col = list(df.index.names) # Gets names of all columns in df and places in list
        df_col.insert(0, df_col.pop(df_col.index("timestamp"))) #move timestamp to start of df
        df = df.reorder_levels(df_col, axis=0)
        df[0] = pd.to_numeric(df[0], downcast='float')
        return df

    
    def df_process_reserve(self):
        '''
        Method for formatting data which comes form the PLEXOS Reserve Class

        Returns
        -------
        df : pd.DataFrame
            Processed Output, single value column with multiindex.

        '''
        df = self.df.droplevel(level=["band", "property"])
        df.index.rename(['parent','Type'], level=['name','category'], inplace=True)
        df = df.reset_index() # unzip the levels in index
        if self.metadata.reserves_regions().empty == False:
            df = df.merge(self.metadata.reserves_regions(), how='left', on='parent') # Merges in regions where reserves are located
        if self.metadata.reserves_zones().empty == False:
            df = df.merge(self.metadata.reserves_zones(), how='left', on='parent') # Merges in zones where reserves are located
        df_col = list(df.columns) # Gets names of all columns in df and places in list
        df_col.remove(0)
        df_col.insert(0, df_col.pop(df_col.index("timestamp"))) #move timestamp to start of df
        df.set_index(df_col, inplace=True)
        df[0] = pd.to_numeric(df[0], downcast='float')
        return df

    
    def df_process_reserves_generators(self):
        '''
        Method for formatting data which comes form the PLEXOS Reserve_Generators Relational Class

        Returns
        -------
        df : pd.DataFrame
            Processed Output, single value column with multiindex.

        '''
        df = self.df.droplevel(level=["band", "property"])
        df.index.rename(['gen_name'], level=['child'], inplace=True)
        df = df.reset_index() # unzip the levels in index
        df = df.merge(self.metadata.generator_category(), how='left', on='gen_name')
         
        # merging in generator region/zones first prevents double counting in cases where multiple model regions are within a reserve region
        if self.metadata.region_generators().empty == False:
            df = df.merge(self.metadata.region_generators(), how='left', on='gen_name')
        if self.metadata.zone_generators().empty == False:
            df = df.merge(self.metadata.zone_generators(), how='left', on='gen_name')
        
        # now merge in reserve regions/zones
        if self.metadata.reserves_regions().empty == False:
            df = df.merge(self.metadata.reserves_regions(), how='left', on=['parent', 'region']) # Merges in regions where reserves are located
        if self.metadata.reserves_zones().empty == False:
            df = df.merge(self.metadata.reserves_zones(), how='left', on=['parent', 'zone']) # Merges in zones where reserves are located
        
        df['tech'] = df['tech'].map(lambda x: self.gen_names_dict.get(x,x))
        df_col = list(df.columns) # Gets names of all columns in df and places in list
        df_col.remove(0)
        df_col.insert(0, df_col.pop(df_col.index("timestamp"))) #move timestamp to start of df
        df.set_index(df_col, inplace=True)
        df[0] = pd.to_numeric(df[0], downcast='float')
        return df

    
    def df_process_fuel(self):
        '''
        Methodfor formatting data which comes form the PLEXOS Fuel Class

        Returns
        -------
        df : pd.DataFrame
            Processed Output, single value column with multiindex.

        '''
        df = self.df.droplevel(level=["band", "property", "category"])
        df.index.rename('fuel_type', level='name', inplace=True)
        df = pd.DataFrame(data=df.values.reshape(-1), index=df.index)
        df_col = list(df.index.names) # Gets names of all columns in df and places in list
        df_col.insert(0, df_col.pop(df_col.index("timestamp"))) #move timestamp to start of df
        df = df.reorder_levels(df_col, axis=0)
        df[0] = pd.to_numeric(df[0], downcast='float')
        return df

    
    def df_process_constraint(self):
        '''
        Method for formatting data which comes form the PLEXOS Constraint Class

        Returns
        -------
        df : pd.DataFrame
            Processed Output, single value column with multiindex.

        '''
        df = self.df.droplevel(level=["band", "property"])
        df.index.rename(['constraint_category', 'constraint'], level=['category', 'name'], inplace=True)
        df = pd.DataFrame(data=df.values.reshape(-1), index=df.index)
        df_col = list(df.index.names) # Gets names of all columns in df and places in list
        df_col.insert(0, df_col.pop(df_col.index("timestamp"))) #move timestamp to start of df
        df = df.reorder_levels(df_col, axis=0)
        df[0] = pd.to_numeric(df[0], downcast='float')
        return df

    
    def df_process_emission(self):
        '''
        Method for formatting data which comes form the PLEXOS Emission Class

        Returns
        -------
        df : pd.DataFrame
            Processed Output, single value column with multiindex.

        '''
        df = self.df.droplevel(level=["band", "property"])
        df.index.rename('emission_type', level='name', inplace=True)
        df = pd.DataFrame(data=df.values.reshape(-1), index=df.index)
        df_col = list(df.index.names) # Gets names of all columns in df and places in list
        df_col.insert(0, df_col.pop(df_col.index("timestamp"))) #move timestamp to start of df
        df = df.reorder_levels(df_col, axis=0)
        df[0] = pd.to_numeric(df[0], downcast='float')
        return df

    
    def df_process_emissions_generators(self):
        '''
        Method for formatting data which comes from the PLEXOS Emissions_Generators Relational Class

        Returns
        -------
        df : pd.DataFrame
            Processed Output, single value column with multiindex.

        '''
        df = self.df.droplevel(level=["band", "property"])
        df.index.rename(['gen_name'], level=['child'], inplace=True)
        df.index.rename(['pollutant'], level=['parent'], inplace=True)

        df = df.reset_index() # unzip the levels in index
        df = df.merge(self.metadata.generator_category(), how='left', on='gen_name') # merge in tech information

        # merge in region and zone information
        if self.metadata.region_generator_category().empty == False:
            # merge in region information
            df = df.merge(self.metadata.region_generator_category().reset_index(), how='left', on=['gen_name', 'tech'])
        if self.metadata.zone_generator_category().empty == False:
            df = df.merge(self.metadata.zone_generator_category().reset_index(), how='left', on=['gen_name', 'tech']) # Merges in zones where reserves are located

        if not self.Region_Mapping.empty:
            df = df.merge(self.Region_Mapping, how="left", on="region")

        # reclassify gen tech categories
        df['tech'] = pd.Categorical(df['tech'].map(lambda x: self.gen_names_dict.get(x,x)))
        
        if not self.emit_names.empty:
            # reclassify emissions as specified by user in mapping
            df['pollutant'] = pd.Categorical(df['pollutant'].map(lambda x: self.emit_names_dict.get(x,x)))

        # remove categoricals (otherwise h5 save will fail)
        df = df.astype({'tech':'object', 'pollutant':'object'})

        # Checks if all emissions categorieses have been identified and matched. If not, lists categories that need a match
        if not self.emit_names.empty:
            if self.emit_names_dict != {} and (set(df['pollutant'].unique()).issubset(self.emit_names["New"].unique())) == False:
                missing_emit_cat = list((set(df['pollutant'].unique())) - (set(self.emit_names["New"].unique())))
                logger.warning("The following emission objects do not have a correct category mapping: %s\n",missing_emit_cat)
        
        df_col = list(df.columns) # Gets names of all columns in df and places in list
        df_col.remove(0)
        df_col.insert(0, df_col.pop(df_col.index("timestamp"))) #move timestamp to start of df
        df.set_index(df_col, inplace=True)
        # downcast values to save on memory
        df[0] = pd.to_numeric(df[0].values, downcast='float')
        # convert to range index (otherwise h5 save will fail)
        df.columns = pd.RangeIndex(0, 1, step=1)
        return df

    
    def df_process_storage(self):
        '''
        Method for formatting data which comes form the PLEXOS Storage Class

        Returns
        -------
        df : pd.DataFrame
            Processed Output, single value column with multiindex.

        '''
        df = self.df.droplevel(level=["band", "property", "category"])
        df = df.reset_index() # unzip the levels in index
        df = df.merge(self.metadata.generator_storage(), how='left', on='name')
        if self.metadata.region_generators().empty == False:
            df = df.merge(self.metadata.region_generators(), how='left', on='gen_name') # Merges in regions where generators are located
        if self.metadata.zone_generators().empty == False:
            df = df.merge(self.metadata.zone_generators(), how='left', on='gen_name') # Merges in zones where generators are located
        if not self.Region_Mapping.empty: #checks if Region_Maping contains data to merge, skips if empty (Default)
            df = df.merge(self.Region_Mapping, how='left', on='region') # Merges in all Region Mappings
        df.rename(columns={'name':'storage_resource'}, inplace=True)
        df_col = list(df.columns) # Gets names of all columns in df and places in list
        df_col.remove(0) # Removes 0, the data column from the list
        df_col.insert(0, df_col.pop(df_col.index("timestamp"))) #move timestamp to start of df
        df.set_index(df_col, inplace=True)
        df[0] = pd.to_numeric(df[0], downcast='float')
        return df

    
    def df_process_region_regions(self):
        '''
        Method for formatting data which comes form the PLEXOS Region_Regions Relational Class

        Returns
        -------
        df : pd.DataFrame
            Processed Output, single value column with multiindex.

        '''
        df = self.df.droplevel(level=["band", "property"])
        df = pd.DataFrame(data=df.values.reshape(-1), index=df.index)
        df_col = list(df.index.names) # Gets names of all columns in df and places in list
        df_col.insert(0, df_col.pop(df_col.index("timestamp"))) #move timestamp to start of df
        df = df.reorder_levels(df_col, axis=0)
        df[0] = pd.to_numeric(df[0], downcast='float')
        return df

    def df_process_node(self):
        '''
        Method for formatting data which comes form the PLEXOS Node Class

        Returns
        -------
        df : pd.DataFrame
            Processed Output, single value column with multiindex.

        '''
        df = self.df.droplevel(level=["band","property","category"])
        df.index.rename('node', level='name', inplace=True)
        df.sort_index(level=['node'], inplace=True)
        if self.metadata.node_region().empty == False:
            node_region_idx = pd.CategoricalIndex(self.metadata.node_region().index.get_level_values(0))
            node_region_idx = node_region_idx.repeat(len(df.index.get_level_values('timestamp').unique()))
            idx_region = pd.MultiIndex(levels= df.index.levels + [node_region_idx.categories]
                                ,codes= df.index.codes +  [node_region_idx.codes],
                                names= df.index.names + node_region_idx.names)
        else:
            idx_region = df.index
        if self.metadata.node_zone().empty == False:
            node_zone_idx = pd.CategoricalIndex(self.metadata.node_zone().index.get_level_values(0))
            node_zone_idx = node_zone_idx.repeat(len(df.index.get_level_values('timestamp').unique()))
            idx_zone = pd.MultiIndex(levels= idx_region.levels + [node_zone_idx.categories]
                                ,codes= idx_region.codes + [node_zone_idx.codes] ,
                                names= idx_region.names + node_zone_idx.names)
        else:
            idx_zone = idx_region
        if not self.Region_Mapping.empty:
            region_mapping_idx = pd.MultiIndex.from_frame(self.metadata.node_region().merge(self.Region_Mapping,
                                how="left", on='region').drop(['region','node'], axis=1))
            region_mapping_idx = region_mapping_idx.repeat(len(df.index.get_level_values('timestamp').unique()))

            idx_map = pd.MultiIndex(levels= idx_zone.levels + region_mapping_idx.levels
                                ,codes= idx_zone.codes + region_mapping_idx.codes,
                                names = idx_zone.names + region_mapping_idx.names)
        else:
            idx_map = idx_zone

        df = pd.DataFrame(data=df.values.reshape(-1), index=idx_map)
        df_col = list(df.index.names) # Gets names of all columns in df and places in list
        df_col.insert(0, df_col.pop(df_col.index("timestamp"))) #move timestamp to start of df
        df = df.reorder_levels(df_col, axis=0)
        df[0] = pd.to_numeric(df[0], downcast='float')
        return df


class MarmotFormat():
    
    ''' 
    This is the main MarmotFormat class which needs to be instantiated to run the formatter.
    The fromatter reads in PLEXOS hdf5 files created with the h5plexos library 
    and processes the output results to ready them for plotting. 
    Once the outputs have been processed, they are saved to an intermediary hdf5 file 
    which can then be read into the Marmot plotting code 
    '''
    
    def __init__(self,Scenario_name, PLEXOS_Solutions_folder, gen_names, Plexos_Properties,
                 Marmot_Solutions_folder=None, mapping_folder='mapping_folder', Region_Mapping=pd.DataFrame(),
                  emit_names=pd.DataFrame(), VoLL=10000):
        '''
        
        Parameters
        ----------
        Scenario_name : string
            Name of sceanrio to process.
        PLEXOS_Solutions_folder : string directory
            Folder containing h5plexos results files.
        gen_names : string directory/pd.DataFrame
            Mapping file to rename generator technologies.
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

        '''
        
        self.Scenario_name = Scenario_name
        self.PLEXOS_Solutions_folder = PLEXOS_Solutions_folder
        self.Marmot_Solutions_folder = Marmot_Solutions_folder
        self.mapping_folder = mapping_folder
        self.VoLL = VoLL

        
        if self.Marmot_Solutions_folder == None:
            self.Marmot_Solutions_folder = self.PLEXOS_Solutions_folder
        
        if isinstance(gen_names, str):
            try:
                gen_names = pd.read_csv(gen_names)   
                self.gen_names = gen_names.rename(columns={gen_names.columns[0]:'Original',gen_names.columns[1]:'New'})
            except FileNotFoundError:
                logger.warning('Could not find specified gen_names file; check file name. This is required to run Marmot, system will now exit')
                sys.exit()
        elif isinstance(gen_names, pd.DataFrame):
            self.gen_names = gen_names.rename(columns={gen_names.columns[0]:'Original',gen_names.columns[1]:'New'})
        
        if isinstance(Plexos_Properties, str):
            try:
                self.Plexos_Properties = pd.read_csv(Plexos_Properties)   
            except FileNotFoundError:
                logger.warning('Could not find specified Plexos_Properties file; check file name. This is required to run Marmot, system will now exit')
                sys.exit()
        elif isinstance(Plexos_Properties, pd.DataFrame):
            self.Plexos_Properties = Plexos_Properties
        
        try:
            self.vre_gen_cat = pd.read_csv(os.path.join(self.mapping_folder, mconfig.parser('category_file_names','vre_gen_cat')),squeeze=True).str.strip().tolist()
        except FileNotFoundError:
            logger.warning(f'Could not find "{os.path.join(self.mapping_folder, "vre_gen_cat.csv")}"; Check file name in config file. This is required to calculate Curtailment')
            self.vre_gen_cat = []
        
        if isinstance(Region_Mapping, str):
            try:
                self.Region_Mapping = pd.read_csv(Region_Mapping)
                if not self.Region_Mapping.empty:  
                    self.Region_Mapping = self.Region_Mapping.astype(str)
            except FileNotFoundError:
                logger.warning('Could not find specified Region Mapping file; check file name\n')
                self.Region_Mapping = pd.DataFrame()
        elif isinstance(Region_Mapping, pd.DataFrame):
            self.Region_Mapping = Region_Mapping
            if not self.Region_Mapping.empty:           
                self.Region_Mapping = self.Region_Mapping.astype(str)
        
        try:
            self.Region_Mapping = self.Region_Mapping.drop(["category"],axis=1) # delete category columns if exists
        except KeyError:
            pass
        
        if isinstance(emit_names, str):
            try:
                self.emit_names = pd.read_csv(emit_names)
                if not self.emit_names.empty:
                    self.emit_names.rename(columns={self.emit_names.columns[0]:'Original',self.emit_names.columns[1]:'New'},inplace=True)
            except FileNotFoundError:
                logger.warning('Could not find specified emissions mapping file; check file name\n')
                self.emit_names = pd.DataFrame()
        elif isinstance(emit_names, pd.DataFrame):
            self.emit_names = emit_names
            if not self.emit_names.empty:
                self.emit_names.rename(columns={self.emit_names.columns[0]:'Original',self.emit_names.columns[1]:'New'},inplace=True)
        
    
    
    def _get_data(self, plexos_class, plexos_prop, timescale, db, metadata):
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
                df = db.query_relation_property(plexos_class,plexos_prop,timescale=timescale)
            else:
                df = db.query_object_property(plexos_class,plexos_prop,timescale=timescale)
    
        except KeyError:
            df = self._report_prop_error(plexos_prop,plexos_class)
            return df
        
        # Instantiate instance of Process Class
        # metadata is used as a paramter to initialize process_cl
        process_cl = Process(df, metadata, self.Region_Mapping, self.gen_names, self.emit_names)
        # Instantiate Method of Process Class
        process_att = getattr(process_cl,'df_process_' + plexos_class)
        # Process attribute and return to df
        df = process_att()
        if plexos_class == 'region' and plexos_prop == "Unserved Energy" and int(df.sum(axis=0)) > 0:
            logger.warning("Scenario contains Unserved Energy: %s MW\n", int(df.sum(axis=0)))
        return df
    
   
    def _report_prop_error(self, plexos_prop, plexos_class):
        '''
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

        '''
        logger.warning('CAN NOT FIND "%s %s". "%s" DOES NOT EXIST',plexos_class,plexos_prop,plexos_prop)
        logger.info('SKIPPING PROPERTY\n')
        df = pd.DataFrame()
        return df
        
    
    def run_formatter(self):
        '''
        
        Main method to call to begin processing h5plexos files, this method takes 
        no input variables, all required varibales are passed in via the __init__ method.

        Returns
        -------
        None.

        '''
            
        logger.info("#### Processing %s PLEXOS Results ####", self.Scenario_name)
        
        #===============================================================================
        # Input and Output Directories
        #===============================================================================
    
        HDF5_output = str(self.Scenario_name) + "_formatted.h5"

        HDF5_folder_in = os.path.join(self.PLEXOS_Solutions_folder, str(self.Scenario_name))
        try:
            os.makedirs(HDF5_folder_in)
        except FileExistsError:
            # directory already exists
            pass
        
        hdf_out_folder = os.path.join(self.Marmot_Solutions_folder,'Processed_HDF5_folder')
        try:
            os.makedirs(hdf_out_folder)
        except FileExistsError:
            # directory already exists
            pass    
        
        startdir=os.getcwd()
        os.chdir(HDF5_folder_in)     #Due to a bug on eagle need to chdir before listdir
        
        files = []
        for names in os.listdir():
            if names.endswith(".h5"):
                files.append(names) # Creates a list of only the hdf5 files
        
        # List of all hf files in hdf5 folder in alpha numeric order
        files_list = sorted(files, key=lambda x:int(re.sub('\D', '', os.path.splitext(x)[0]))) 
        
        os.chdir(startdir)

        # Read in all HDF5 files into dictionary
        logger.info("Loading all HDF5 files to prepare for processing")
        hdf5_collection = {}
        for file in files_list:
            hdf5_collection[file] = PLEXOSSolution(os.path.join(HDF5_folder_in, file))

        #===================================================================================
        # Process the Outputs
        #===================================================================================
    
        # Creates Initial HDF5 file for ouputing formated data
        Processed_Data_Out = pd.DataFrame()
        if os.path.isfile(os.path.join(hdf_out_folder,HDF5_output))==True:
            logger.info("'%s\%s' already exists: New variables will be added\n",hdf_out_folder,HDF5_output)
            #Skip properties that already exist in *formatted.h5 file.
            with h5py.File(os.path.join(hdf_out_folder,HDF5_output),'r') as f:
                existing_keys = [key for key in f.keys()]
        else:
            Processed_Data_Out.to_hdf(os.path.join(hdf_out_folder, HDF5_output), key= "generator_Generation" , mode="w", complevel=9, complib  ='blosc:zlib')
    
        process_properties = self.Plexos_Properties.loc[self.Plexos_Properties["collect_data"] == True]
    
        start = time.time()
        
        if not self.Region_Mapping.empty:
            #if any(meta.regions()['region'] not in Region_Mapping['region']):
            if set(MetaData(HDF5_folder_in, self.Region_Mapping).regions()['region']).issubset(self.Region_Mapping['region']) == False:
                missing_regions = list(set(MetaData(HDF5_folder_in, self.Region_Mapping).regions()['region']) - set(self.Region_Mapping['region']))
                logger.warning('The Following PLEXOS REGIONS are missing from the "region" column of your mapping file: %s\n',missing_regions)
        
        # Main loop to process each ouput and pass data to functions
        for index, row in process_properties.iterrows():
            Processed_Data_Out = pd.DataFrame()
            data_chunks = []
    
            logger.info("Processing %s %s",row["group"],row["data_set"])
            row["data_set"] = row["data_set"].replace(' ', '_')
            key_path = row["group"] + "_" + row["data_set"]
            if key_path not in existing_keys:
    
                for model in files_list:
                    logger.info("      %s",model)
                    
                    # Create an instance of metadata, and pass that as a variable to get data.
                    meta = MetaData(HDF5_folder_in, self.Region_Mapping,model)
                    db = hdf5_collection.get(model)
                    processed_data = self._get_data(row["group"], row["data_set"],row["data_type"], db, meta)
        
                    if processed_data.empty == True:
                        break
                    
                    # special units processing for emissions
                    if row["group"]=="emissions_generators":
                        if (row["Units"] == "lb") | (row["Units"] == "lbs"):
                            # convert lbs to kg
                            kg_per_lb = 0.453592
                            processed_data = processed_data*kg_per_lb
                        # convert kg to metric tons
                        kg_per_metric_ton = 1E3
                        data_chunks.append(processed_data/kg_per_metric_ton)
                    
                    # other unit multipliers
                    if (row["data_type"] == "year")&((row["data_set"]=="Installed Capacity")|(row["data_set"]=="Export Limit")|(row["data_set"]=="Import Limit")):
                        data_chunks.append(processed_data*row["unit_multiplier"])
                        logger.info("%s Year property reported from only the first partition",row["data_set"])
                        break
                    else:
                        data_chunks.append(processed_data*row["unit_multiplier"])
        
                if data_chunks:
                    Processed_Data_Out = pd.concat(data_chunks, copy=False)
        
                if Processed_Data_Out.empty == False:
                    if (row["data_type"]== "year"):
                        logger.info("Please Note: Year properties can not be checked for duplicates.\n\
                        Overlaping data can not be removed from 'Year' grouped data.\n\
                        This will effect Year data that differs between partitions such as cost results.\n\
                        It will not effect Year data that is equal in all partitions such as Installed Capacity or Line Limit results")
        
                    else:
                        oldsize=Processed_Data_Out.size
                        Processed_Data_Out = Processed_Data_Out.loc[~Processed_Data_Out.index.duplicated(keep='first')] #Remove duplicates; keep first entry^M
                        if  (oldsize-Processed_Data_Out.size) >0:
                            logger.info('Drop duplicates removed %s rows',oldsize-Processed_Data_Out.size)
        
                    row["data_set"] = row["data_set"].replace(' ', '_')
                    try:
                        logger.info("Saving data to h5 file...")
                        Processed_Data_Out.to_hdf(os.path.join(hdf_out_folder, HDF5_output), key= row["group"] + "_" + row["data_set"], mode="a", complevel=9, complib = 'blosc:zlib')
                        logger.info("Data saved to h5 file successfully\n")
                    except:
                        logger.warning("h5 File is probably in use, waiting to attempt save for a second time")
                        time.sleep(120)
                        try:
                              Processed_Data_Out.to_hdf(os.path.join(hdf_out_folder, HDF5_output), key= row["group"] + "_" + row["data_set"], mode="a", complevel=9, complib = 'blosc:zlib')
                              logger.info("h5 File save succeded on second attempt")
                        except:
                            logger.warning("h5 File is probably in use, waiting to attempt save for a third time")
                            time.sleep(240)
                            try:
                                Processed_Data_Out.to_hdf(os.path.join(hdf_out_folder, HDF5_output), key= row["group"] + "_" + row["data_set"], mode="a",  complevel=9, complib = 'blosc:zlib')
                                logger.info("h5 File save succeded on third attempt")
                            except:
                                logger.warning("h5 Save failed on third try; will not attempt again\n")
                    # del Processed_Data_Out
                else:
                    continue
            else:
                logger.info(f"{key_path} already exists in output .h5 file. Skipping property.")
                continue

            
        #===================================================================================
        # Calculate Extra Ouputs
        #===================================================================================
        if "generator_Curtailment" not in h5py.File(os.path.join(hdf_out_folder, HDF5_output),'r'):
            try:
                logger.info("Processing generator Curtailment")
                try:
                    Avail_Gen_Out = pd.read_hdf(os.path.join(hdf_out_folder, HDF5_output), 'generator_Available_Capacity')
                    Total_Gen_Out = pd.read_hdf(os.path.join(hdf_out_folder, HDF5_output), 'generator_Generation')
                    if Total_Gen_Out.empty==True:
                        logger.warning("generator_Available_Capacity & generator_Generation are required for Curtailment calculation")
                except KeyError:
                    logger.warning("generator_Available_Capacity & generator_Generation are required for Curtailment calculation")
                    
                # Adjust list of values to drop from vre_gen_cat depending on if it exhists in processed techs
                adjusted_vre_gen_list = [name for name in self.vre_gen_cat if name in Avail_Gen_Out.index.unique(level="tech")]
                
                if not adjusted_vre_gen_list:
                    logger.warning("vre_gen_cat.csv is not set up correctly with your gen_names.csv")
                    logger.warning("To Process Curtailment add correct names to vre_gen_cat.csv. \
                    For more information see Marmot Readme under 'Mapping Files'")
            
                # Output Curtailment#
                Curtailment_Out =  ((Avail_Gen_Out.loc[(slice(None), adjusted_vre_gen_list),:]) -
                                    (Total_Gen_Out.loc[(slice(None), adjusted_vre_gen_list),:]))
            
                Curtailment_Out.to_hdf(os.path.join(hdf_out_folder, HDF5_output), key="generator_Curtailment", mode="a", complevel=9, complib = 'blosc:zlib')
                logger.info("Data saved to h5 file successfully\n")
                #Clear Some Memory
                del Total_Gen_Out
                del Avail_Gen_Out
                del Curtailment_Out
            except Exception:
                logger.warning("NOTE!! Curtailment not calculated, processing skipped\n")

    
        if "region_Cost_Unserved_Energy" not in h5py.File(os.path.join(hdf_out_folder, HDF5_output),'r'):
            try:
                logger.info("Calculating Cost Unserved Energy: Regions")
                Cost_Unserved_Energy = pd.read_hdf(os.path.join(hdf_out_folder, HDF5_output), 'region_Unserved_Energy')
                Cost_Unserved_Energy = Cost_Unserved_Energy * self.VoLL
                Cost_Unserved_Energy.to_hdf(os.path.join(hdf_out_folder, HDF5_output), key="region_Cost_Unserved_Energy", mode="a", complevel=9, complib = 'blosc:zlib')
            except KeyError:
                logger.warning("NOTE!! Regional Unserved Energy not available to process, processing skipped\n")
                pass
    
        if "zone_Cost_Unserved_Energy" not in h5py.File(os.path.join(hdf_out_folder, HDF5_output),'r'):
            try:
                logger.info("Calculating Cost Unserved Energy: Zones")
                Cost_Unserved_Energy = pd.read_hdf(os.path.join(hdf_out_folder, HDF5_output), 'zone_Unserved_Energy')
                Cost_Unserved_Energy = Cost_Unserved_Energy * self.VoLL
                Cost_Unserved_Energy.to_hdf(os.path.join(hdf_out_folder, HDF5_output), key="zone_Cost_Unserved_Energy", mode="a", complevel=9, complib = 'blosc:zlib')
            except KeyError:
                logger.warning("NOTE!! Zonal Unserved Energy not available to process, processing skipped\n")
                pass
    
        end = time.time()
        elapsed = end - start
        logger.info('Main loop took %s minutes',round(elapsed/60,2))
        logger.info('Formatting COMPLETED for %s',self.Scenario_name)



if __name__ == '__main__':
    
    '''
    The following code is run if the formatter is run directly,
    it does not run if the formatter is imported as a module. 
    '''
    
    #===============================================================================
    # Input Properties
    #===============================================================================
    
    #changes working directory to location of this python file
    os.chdir(file_dir)
    
    Marmot_user_defined_inputs = pd.read_csv(mconfig.parser("user_defined_inputs_file"), usecols=['Input','User_defined_value'],
                                         index_col='Input', skipinitialspace=True)

    # File which determiens which plexos properties to pull from the h5plexos results and process, this file is in the repo
    Plexos_Properties = pd.read_csv('plexos_properties.csv')
    
    # Plexos_Properties = 'plexos_properties.csv'
        
    # Name of the Scenario(s) being run, must have the same name(s) as the folder holding the runs HDF5 file
    Scenario_List = pd.Series(Marmot_user_defined_inputs.loc['Scenario_process_list'].squeeze().split(",")).str.strip().tolist()
    # The folder that contains all PLEXOS h5plexos outputs - the h5 files should be contained in another folder with the Scenario_name
    PLEXOS_Solutions_folder = Marmot_user_defined_inputs.loc['PLEXOS_Solutions_folder'].to_string(index=False).strip()
    
    # Folder to save your processed solutions
    Marmot_Solutions_folder = Marmot_user_defined_inputs.loc['Marmot_Solutions_folder'].to_string(index=False).strip()
    
    # This folder contains all the csv required for mapping and selecting outputs to process
    # Examples of these mapping files are within the Marmot repo, you may need to alter these to fit your needs
    Mapping_folder = 'mapping_folder'
    
    Region_Mapping = pd.read_csv(os.path.join(Mapping_folder, Marmot_user_defined_inputs.loc['Region_Mapping.csv_name'].to_string(index=False).strip()))
    
    gen_names = pd.read_csv(os.path.join(Mapping_folder, Marmot_user_defined_inputs.loc['gen_names.csv_name'].to_string(index=False).strip()))
    
    # Value of Lost Load for calculatinhg cost of unserved energy
    VoLL = pd.to_numeric(Marmot_user_defined_inputs.loc['VoLL'].to_string(index=False))
    
    #===============================================================================
    # Standard Naming of Generation Data
    #===============================================================================
    
    
    #===============================================================================
    # Standard Naming of Emissions types (optional)
    #===============================================================================
    
    emit_names = os.path.join(Mapping_folder, Marmot_user_defined_inputs.loc['emit_names.csv_name'].to_string(index=False).strip())

    #===============================================================================
    # Loop through scenarios in list
    #===============================================================================
    
    for Scenario_name in Scenario_List:
        
        initiate = MarmotFormat(Scenario_name,PLEXOS_Solutions_folder,gen_names,Plexos_Properties,
                                Marmot_Solutions_folder = Marmot_Solutions_folder,
                                mapping_folder = 'mapping_folder',
                                Region_Mapping = Region_Mapping,
                                emit_names = emit_names,
                                VoLL = VoLL)
    
        initiate.run_formatter()
    
    
#===============================================================================
# Code that can be used to test PLEXOS_H5_results_formatter
#===============================================================================

    # test = pd.read_hdf(file, 'generator_Generation')
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
