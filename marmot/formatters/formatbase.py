
import os
import re
import logging
import pandas as pd

logger = logging.getLogger('marmot_format.'+__name__)

class Process():
    """Base class for processing simulation model data.
    """

    # Conversion units dict, key values is a tuple of new unit name and 
    # conversion multiplier
    UNITS_CONVERSION: dict = {
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


    def __init__(self, input_folder: str, Region_Mapping: pd.DataFrame, 
                 emit_names: pd.DataFrame, **kwargs):
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
        logger.warning(f'CAN NOT FIND "{prop_class} {property}". ' 
                            f'"{property}" DOES NOT EXIST')
        logger.info('SKIPPING PROPERTY\n')
        df = pd.DataFrame()
        return df
