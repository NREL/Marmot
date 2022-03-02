
import logging
import re
import gdxpds
import pandas as pd
from pathlib import Path
from marmot.formatters.formatbase import Process

logger = logging.getLogger('marmot_format.'+__name__)

class ProcessReEDS(Process):
    """Process ReEDS  specific data from a ReEDS result set.
    """
    def __init__(self, input_folder: Path, Region_Mapping: pd.DataFrame, 
                *args, **kwargs):
        """
        Args:
            input_folder (Path): Folder containing csv files.
            Region_Mapping (pd.DataFrame): DataFrame to map custom 
                regions/zones to create custom aggregations.
            plexos_block (str, optional): PLEXOS results type. Defaults to 'ST'.
        """
        self.gdx_collection = {}
        self.gdx_data_units = {}

        # Instantiation of Process Base class
        super().__init__(input_folder, Region_Mapping, *args, **kwargs) 

    def get_input_files(self) -> list:
        """Gets a list of input files within the scenario folders
        """

        symbol_marker = "--(.*?)--"

        reeds_outputs_dir = self.input_folder.joinpath('outputs')

        files = []
        for names in reeds_outputs_dir.iterdir():
            if names.suffix == ".gdx":
                files.append(names.name)
                
                names = str(names)
                symbol_list = gdxpds.list_symbols(names)
                if names not in self.gdx_data_units:
                    self.gdx_data_units[names] = {}
                for symbol in symbol_list:
                    unit = re.search(symbol_marker, symbol.description)
                    if unit:
                        unit = unit.group(1)
                    self.gdx_data_units[names][symbol.name] = unit
                

        # List of all files in input folder in alpha numeric order
        files_list = sorted(files, key=lambda x:int(re.sub('\D', '', x)))

        return files_list

    def get_processed_data(self, data_class: str, prop: str, 
                  timescale: str, model_filename: str) -> pd.DataFrame:
        """Handles the pulling of data from the ReEDS gdx
        file and then passes the data to one of the formating functions

        Args:
            data_class (str): Data class e.g Region, Generator, Zone etc
            prop (str): Property e.g gen_out, cap_out etc.
            timescale (str): Data timescale, e.g interval, summary.
            model_filename (str): name of model to process.

        Returns:
            pd.DataFrame: Formatted results dataframe.
        """
        model_filename = str(model_filename)

        df = gdxpds.to_dataframe(model_filename, prop)

        # Get desired method
        process_att = getattr(self, f'df_process_{data_class}')
        # Process attribute and return to df
        df = process_att(df, timescale)

        df_units = self.gdx_data_units[model_filename][prop]
        # find unit conversion values
        converted_units = self.UNITS_CONVERSION.get(df_units, (df_units, 1))

        # Convert units and add unit column to index 
        df = df*converted_units[1]
        units_index = pd.Index([converted_units[0]] *len(df), name='units')
        df.set_index(units_index, append=True, inplace=True) 

    def df_process_generator(self, df: pd.DataFrame, 
                             timescale: str) -> pd.DataFrame:
        
        return
