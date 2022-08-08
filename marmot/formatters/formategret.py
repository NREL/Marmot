import logging
import pandas as pd
import os
import json
import datetime
from pathlib import Path
import re
import numpy as np

from marmot.formatters.formatbase import Process
from marmot.metamanagers.read_metadata import MetaData

logger = logging.getLogger("formatter." + __name__)


class ProcessEGRET(Process):
    """Process EGRET class specific data from a json database.
    """
    # Maps EGRET property names to Marmot names, 
    # unchanged names not included  
    PROPERTY_MAPPING: dict = {
        'generator_headroom':'generator_Available_Capacity',
        'generator_pg':'generator_Generation',
        'zone_load':'zone_Load',
        'region_load':'region_Load'
    }

    # Extra custom properties that are created based off existing properties. 
    # The dictionary keys are the existing properties and the values are the new
    # property names and methods used to create it.
    EXTRA_MARMOT_PROPERTIES: dict = {}


    # NOTE: not sure if I need to change this??
    def __init__(self, input_folder: Path, output_file_path: Path, 
                 Region_Mapping: pd.DataFrame, 
                 *args, process_subset_years: list=None, **kwargs):
        """
        Args:
            input_folder (Path): Folder containing csv files.
            output_file_path (Path): Path to formatted h5 output file.
            Region_Mapping (pd.DataFrame): DataFrame to map custom 
                regions/zones to create custom aggregations.
            process_subset_years (list, optional): If provided only process 
                years specified. Defaults to None.
        """
        #self.file_collection: dict = {}
        # Internal cached data is saved to the following variables.
        # To access the values use the public api e.g self.property_units
        self._property_units: dict = {}
        self._wind_resource_to_pca = None

        self.metadata = MetaData(output_file_path.parent, 
                                read_from_formatted_h5=True, 
                                Region_Mapping=Region_Mapping)
        
        if process_subset_years:
            # Ensure values are ints
            process_subset_years = list(map(int, process_subset_years)) 
            logger.info(f"Processing subset of EGRET years: {process_subset_years}")
        self.process_subset_years = process_subset_years

        # Instantiation of Process Base class
        super().__init__(input_folder, output_file_path, 
                        Region_Mapping, *args, **kwargs) 

    # NOTE: Skipping some reeds methods here; I'm not sure what they do.

    # NOTE: need to implement this for EGRET!!
    def output_metadata(self, files_list: list) -> None:
        """Add ReEDS specific metadata to formatted h5 file .  

        Args:
            files_list (list): List of all gdx files in inputs 
                folder in alpha numeric order.
        """
        for partition in files_list:
            # f = open(partition, 'r')
            f = open(self.input_folder.joinpath(partition), 'r')
            data = json.load(f)
            f.close()

            regions = list(data['elements']['area'].keys())

            region_df = pd.DataFrame({'name':regions,'category':['']*len(regions)}) 
            
            region_df.to_hdf(self.output_file_path, 
                             key=f'metadata/{partition}/objects/regions', 
                             mode='a')
    
    # # NOTE: might need to change this somehow
    # def get_input_files(self) -> list:
    #     """Gets a list of input files within the scenario folders
    #     """
    #     startdir = os.getcwd()
    #     os.chdir(self.input_folder) 
        
    #     files = []
    #     for names in os.listdir():
    #         if names.endswith(".json"):
    #             files.append(str(self.input_folder.absolute()) + '/' + names)  # Creates a list of only the json files

    #     # List of all files in alpha numeric order
    #     # files_list = sorted(files, key=lambda x:int(re.sub('\D', '', x)))
    #     # os.chdir(startdir)
    #     self._get_input_files = sorted(files, key=lambda x:int(re.sub('\D', '', x)))
    #     os.chdir(startdir)

    #     return self._get_input_files


    # NOTE: need to make sure we have the correct data_class
    def get_processed_data(self, data_class: str, prop: str, 
                  timescale: str, model_filename: str) -> pd.DataFrame:
        """Handles the pulling of data from the EGRET json
        file and then passes the data to one of the formating functions

        Args:
            data_class (str): Data class e.g Region, Generator, Zone etc
            prop (str): Property e.g gen_out, cap_out etc.
            timescale (str): Data timescale, e.g interval, summary.
            model_filename (str): name of model to process.

        Returns:
            pd.DataFrame: Formatted results dataframe.
        """
        # Read json file
        f = open(self.input_folder.joinpath(model_filename), 'r')
        data = json.load(f)
        f.close()
        
        # Get desired method, used for extra processing if needed
        process_att = getattr(self, f'df_process_{data_class}', None)
        if process_att:
            # Process attribute and return to df
            df = process_att(data,prop)

        return df


    def df_process_generator(self, data: dict, prop: str) -> pd.DataFrame:
        """Format PLEXOS Generator Class data.
        Args:
            data (dictionary): Egret json data read into a nested dictionary
            prop (string): Egret property name; key of json file
        Returns:
            pd.DataFrame: Processed output, single value column with multiindex.
        """
        # Initialize dataframe column lists
        timestamp = []
        tech = []
        gen_name = []
        region = []
        zone = []
        values = []

        # # Get time information specified by user in egret configs file
        # f = open("../marmot/input_files/egret_configs.txt")
        # start_day = int(f.readline().split("=")[-1].strip())
        # start_month = int(f.readline().split("=")[-1].strip())
        # start_year = int(f.readline().split("=")[-1].strip())
        # resolution = int(f.readline().split("=")[-1].strip())
        # f.close()

        # Loop through generators 
        for generator in data['elements']['generator'].keys():
            
            # Get generator time series values

            # Generation >> pg
            if prop == "pg":
                vals = data['elements']['generator'][generator][prop]['values']
                
            # Available Capacity >> headroom/p_max
            elif prop == "headroom":
                if "headroom" in data['elements']['generator'][generator].keys():
                    vals = data['elements']['generator'][generator]["headroom"]['values']
                else:
                    vals = [0]*len(data['elements']['generator'][generator]["p_max"]['values'])

            # Get zone and area labels if they exists
            if 'zone' in data['elements']['generator'][generator].keys():
                zone_val = data['elements']['generator'][generator]['zone']
            else:
                zone_val = '0'
            if 'area' in data['elements']['generator'][generator].keys():
                area_val = data['elements']['generator'][generator]['area']
            else:
                area_val = '0'

            # Get tech value
            tech_val = data['elements']['generator'][generator]['fuel'] + '_' + data['elements']['generator'][generator]['unit_type']

            # timestamp += [(datetime.datetime(start_year,start_month,start_day) + datetime.timedelta(minutes=i*resolution)).strftime("%Y-%m-%d %H:%M:%S") for i in range(len(vals))]
            timestamp += [dt + ":00" for dt in data["system"]["time_keys"]]
            tech += [tech_val]*len(vals)
            gen_name += [generator]*len(vals)
            region += [area_val]*len(vals)
            zone += [zone_val]*len(vals)
            values += vals

        gen_df = pd.DataFrame({'timestamp':timestamp,
                               'tech':tech,
                               'gen_name':gen_name,
                               'region':region,
                               'zone':zone,
                               '0':values})

        # Change timestamp datatype from string to pandas datetime
        gen_df['timestamp'] = pd.to_datetime(gen_df['timestamp'])

        # Merge mapping file
        if not self.Region_Mapping.empty:
            print(gen_df)
            print(self.Region_Mapping)
            gen_df = gen_df.merge(self.Region_Mapping,
                                  how="left",
                                  on='region')

        # Set index
        index_names = list(gen_df.columns)
        index_names.remove('0')
        gen_df = gen_df.set_index(index_names)

        # Need to change from string to integer column name 0
        gen_df = gen_df.rename(columns={'0':0})

        return gen_df


    def df_process_region(self, data: dict, prop: str) -> pd.DataFrame:
        """Format EGRET region data. I think currently this will only work for load.
        Args:
            data (dictionary): EGRET json data read into a nested dictionary
            egret_property (string): Egret property name; key of json file
        Returns:
            pd.DataFrame: Processed output, single value column with multiindex.
        """
        # # Get time information specified by user in egret configs file
        # f = open("../marmot/input_files/egret_configs.txt")
        # start_day = int(f.readline().split("=")[-1].strip())
        # start_month = int(f.readline().split("=")[-1].strip())
        # start_year = int(f.readline().split("=")[-1].strip())
        # resolution = int(f.readline().split("=")[-1].strip())
        # f.close()
        
        # If there is no region information, demand is the only key
        if 'demand' in data['elements'][prop].keys():
            values = data['elements'][prop]['demand']['p_load']['values']
            # timestamp = [(datetime.datetime(start_year,start_month,start_day) + datetime.timedelta(minutes=i*resolution)).strftime("%Y-%m-%d %H:%M:%S") for i in range(len(values))]
            timestamp += [dt + ":00" for dt in data["system"]["time_keys"]]
            region = [0]*len(values)

        # If there is region information, aggregate by region
        else:
            values_dict = {}
            for key in data['elements'][prop].keys():
                area = data['elements'][prop][key]['area']
                # Creat new key 
                if area not in values_dict.keys():
                    values_dict[area] = np.array(data['elements'][prop][key]['p_load']['values'])
                # Add values if key already exists
                else:
                    values_dict[area] += np.array(data['elements'][prop][key]['p_load']['values'])

            # Create actual columns        
            values = []
            timestamp = []
            region = [] # this is area in Egret
            for key in values_dict.keys():
                values += list(values_dict[key])
                # timestamp += [(datetime.datetime(start_year,start_month,start_day) + datetime.timedelta(minutes=i*resolution)).strftime("%Y-%m-%d %H:%M:%S") for i in range(len(values_dict[key]))]
                timestamp += [dt + ":00" for dt in data["system"]["time_keys"]]
                region += [key]*len(values_dict[key])
        
        # Create dataframe
        region_df = pd.DataFrame({'timestamp':timestamp,
                                  'region':region,
                                  '0':values})

        # Change timestamp datatype from string to pandas datetime
        region_df['timestamp'] = pd.to_datetime(region_df['timestamp'])

        # Merge mapping file
        if not self.Region_Mapping.empty:   
            region_df = region_df.merge(self.Region_Mapping,
                                        how="left",
                                        on='region')

        # Set index
        index_names = list(region_df.columns)
        index_names.remove('0')
        region_df = region_df.set_index(index_names)

        # Need to change from string to integer column name
        region_df = region_df.rename(columns={'0':0})
        
        return region_df

    def df_process_zone(self, data: dict, prop: str) -> pd.DataFrame:
        """Format EGRET zone data. I think currently this will only work for load.
        Args:
            data (dictionary): Egret json data read into a nested dictionary
            egret_property (string): Egret property name; key of json file
        Returns:
            pd.DataFrame: Processed output, single value column with multiindex.
        """
        # # Get time information specified by user in egret configs file
        # f = open("../marmot/input_files/egret_configs.txt")
        # start_day = int(f.readline().split("=")[-1].strip())
        # start_month = int(f.readline().split("=")[-1].strip())
        # start_year = int(f.readline().split("=")[-1].strip())
        # resolution = int(f.readline().split("=")[-1].strip())
        # f.close()
        
        # If there is no zone information, demand is the only key
        if 'demand' in data['elements'][prop].keys():
            values = data['elements'][prop]['demand']['p_load']['values']
            # timestamp = [(datetime.datetime(start_year,start_month,start_day) + datetime.timedelta(minutes=i*resolution)).strftime("%Y-%m-%d %H:%M:%S") for i in range(len(values))]
            timestamp += [dt + ":00" for dt in data["system"]["time_keys"]]
            zone = [0]*len(values)

        # If there is zone information, aggregate by zone
        else:
            values_dict = {}
            for key in data['elements'][prop].keys():
                zone = data['elements'][prop][key]['zone']
                # Creat new key 
                if zone not in values_dict.keys():
                    values_dict[zone] = np.array(data['elements'][prop][key]['p_load']['values'])
                # Add values if key already exists
                else:
                    values_dict[zone] += np.array(data['elements'][prop][key]['p_load']['values'])

            # Create actual columns        
            values = []
            timestamp = []
            zone = []
            for key in values_dict.keys():
                values += list(values_dict[key])
                # timestamp += [(datetime.datetime(start_year,start_month,start_day) + datetime.timedelta(minutes=i*resolution)).strftime("%Y-%m-%d %H:%M:%S") for i in range(len(values_dict[key]))]
                timestamp += [dt + ":00" for dt in data["system"]["time_keys"]]
                zone += [key]*len(values_dict[key])

        # Create dataframe
        zone_df = pd.DataFrame({'timestamp':timestamp,
                                'zone':zone,
                                '0':values})
        
        # Change timestamp datatype from string to pandas datetime
        zone_df['timestamp'] = pd.to_datetime(zone_df['timestamp'])
        
        # Set index
        zone_df = zone_df.set_index(['timestamp','zone'])

        # Need to change from string to integer column name
        zone_df = zone_df.rename(columns={'0':0})
        
        return zone_df
        

    #pass
