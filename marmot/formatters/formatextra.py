
"""Contains class and methods used to creates extra properties 
required by the Marmot plotter.

@author: Daniel Levie
"""

import pandas as pd
import logging
import marmot.utils.mconfig as mconfig
from marmot.formatters.formatbase import Process

logger = logging.getLogger('marmot_format.'+__name__)

class ExtraProperties():
    """Creates extra properties required by Marmots plotter.

    Properties can be created based off of existing properties, 
    e.g calculating generator Curtailment from generation and available capacity.

    The class takes a model specific instance of a Process class as an input.
    For example an instance of ProcessPLEXOS is passed when formatting PLEXOS 
    results. This allows access to all the Process class specific methods and 
    attributes. The list of input files are also passed to the class at 
    instantiation. 
    """

    def __init__(self, model: Process, files_list: list):
        """
        Args:
            model (Process): model specific instance of a Process class, 
                e.g ProcessPLEXOS, ProcessReEDS
            files_list (list): list of model input filenames, e.g PLEXOS h5plexos files
        """
        self.model = model
        self.files_list = files_list

    def plexos_generator_curtailment(self, df: pd.DataFrame, 
                                    timescale: str='interval'):
        """Creates a generator_Curtailment property for PLEXOS result sets 

        Args:
            df (pd.DataFrame): generator_Generation df
            timescale (str, optional): Data timescale, e.g Hourly, Monthly, 5 minute etc.
                Defaults to 'interval'.

        Returns:
            pd.DataFrame: generator_Curtailment df
        """
        data_chunks = []
        for file in self.files_list:
            logger.info(f"      {file}")
            processed_data = self.model.get_processed_data('generator', 
                                                    'Available Capacity',
                                                    timescale,
                                                    file)

            if processed_data.empty is True:
                logger.warning("generator_Available_Capacity & "
                                "generator_Generation are required "
                                "for Curtailment calculation")
                return pd.DataFrame()

            data_chunks.append(processed_data)   

        avail_gen = pd.concat(data_chunks, copy=False) 
        # Remove duplicates; keep first entry
        avail_gen = (avail_gen.loc
                        [~avail_gen
                        .index.duplicated(keep='first')])           
        
        return avail_gen - df        

    def plexos_cost_unserved_energy(self, df: pd.DataFrame, **_):
        """Creates a region_Cost_Unserved_Energy property for PLEXOS result sets 

        Args:
            df (pd.DataFrame): region_Unserved_Energy df

        Returns:
            pd.DataFrame: region_Cost_Unserved_Energy df
        """
        return df * mconfig.parser("formatter_settings", 'VoLL')

    def reeds_reserve_provision(self, df: pd.DataFrame, **_):
        """Creates a reserve_Provision property for ReEDS result sets 

        Args:
            df (pd.DataFrame): reserves_generators_Provision df

        Returns:
            pd.DataFrame: reserve_Provision df
        """
        return df.groupby(['timestamp', 'Type', 'parent',
                            'region', 'season', 'units']).sum()

    def reeds_generator_vom_cost(self, df: pd.DataFrame, **_):
        """Creates a generator_VO&M property for ReEDS result sets 

        Args:
            df (pd.DataFrame): generator_Total_Generation_Cost df

        Returns:
            pd.DataFrame: generator_VO&M df
        """
        return df.xs('op_vom_costs', level='cost_type')

    def reeds_generator_fuel_cost(self, df: pd.DataFrame, **_):
        """Creates a generator_Fuel_Cost property for ReEDS result sets 

        Args:
            df (pd.DataFrame): generator_Total_Generation_Cost df

        Returns:
            pd.DataFrame: generator_Fuel_Cost df
        """
        return df.xs('op_fuelcosts_objfn', level='cost_type')

    def reeds_generator_reserve_vom_cost(self, df: pd.DataFrame, **_):
        """Creates a generator_Reserves_VO&M_Cost property for ReEDS result sets 

        Args:
            df (pd.DataFrame): generator_Total_Generation_Cost df

        Returns:
            pd.DataFrame: generator_Reserves_VO&M_Cost df
        """
        return df.xs('op_operating_reserve_costs', level='cost_type')

    def reeds_generator_fom_cost(self, df: pd.DataFrame, **_):
        """Creates a generator_FO&M_Cost property for ReEDS result sets 

        Args:
            df (pd.DataFrame): generator_Total_Generation_Cost df

        Returns:
            pd.DataFrame: generator_FO&M_Cost df
        """
        return df.xs('op_fom_costs', level='cost_type')

    def annualize_property(self, df: pd.DataFrame, **_):
        """Annualizes any property, groups by year

        Args:
            df (pd.DataFrame): multiindex dataframe with timestamp level.

        Returns:
            pd.DataFrame: df with timestamp grouped by year.
        """
        index_names = list(df.index.names)
        index_names.remove('timestamp')
        timestamp_annualized = [pd.to_datetime(df.index.get_level_values('timestamp')
                                                .year.astype(str))]
        timestamp_annualized.extend(index_names)
        return df.groupby(timestamp_annualized).sum()


            
