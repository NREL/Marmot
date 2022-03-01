
import logging
import os
import re
import pandas as pd
from marmot.formatters.formatbase import Process

logger = logging.getLogger('marmot_format.'+__name__)

class ProcessReEDS(Process):
    """Process ReEDS  specific data from a ReEDS result set.
    """
    def __init__(self, input_folder: str, Region_Mapping: pd.DataFrame, 
                *args, **kwargs):
        """
        Args:
            input_folder (str): Folder containing csv files.
            Region_Mapping (pd.DataFrame): DataFrame to map custom 
                regions/zones to create custom aggregations.
            plexos_block (str, optional): PLEXOS results type. Defaults to 'ST'.
        """
        # self.hdf5_collection = {}

        # Instantiation of Process Base class
        super().__init__(input_folder, Region_Mapping, *args, **kwargs) 

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
