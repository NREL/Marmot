"""Contains class and methods used to creates extra properties 
required by the Marmot plotter.

@author: Daniel Levie
"""

import pandas as pd
import logging
import marmot.utils.mconfig as mconfig
from marmot.formatters.formatbase import Process

logger = logging.getLogger("formatter." + __name__)


class ExtraProperties:
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

    def plexos_generator_curtailment(
        self, df: pd.DataFrame, timescale: str = "interval"
    ) -> pd.DataFrame:
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
            processed_data = self.model.get_processed_data(
                "generator", "Available Capacity", timescale, file
            )

            if processed_data.empty is True:
                logger.warning(
                    "generator_Available_Capacity & "
                    "generator_Generation are required "
                    "for Curtailment calculation"
                )
                return pd.DataFrame()

            data_chunks.append(processed_data)

        avail_gen = pd.concat(data_chunks, copy=False)
        # Remove duplicates; keep first entry
        avail_gen = avail_gen.loc[~avail_gen.index.duplicated(keep="first")]

        return avail_gen - df

    def plexos_demand(
        self, df: pd.DataFrame, timescale: str = "interval"
    ) -> pd.DataFrame:
        """Creates a region_Demand / zone_Demand property for PLEXOS result sets

        PLEXOS includes generator_Pumped_Load in total load
        This method subtracts generator_Pumped_Load from region_Demand / zone_Demand to get
        region_Demand / zone_Demand

        Args:
            df (pd.DataFrame): region_Load df
            timescale (str, optional): Data timescale, e.g Hourly, Monthly, 5 minute etc.
                Defaults to 'interval'.

        Returns:
            pd.DataFrame: region_Demand / zone_Demand df
        """
        data_chunks = []
        for file in self.files_list:
            processed_data = self.model.get_processed_data(
                "generator", "Pump Load", timescale, file
            )

            if processed_data.empty is True:
                logger.info("Total Demand will equal Total Load")
                return pd.DataFrame()

            data_chunks.append(processed_data)

        pump_load: pd.DataFrame = pd.concat(data_chunks, copy=False)
        # Remove duplicates; keep first entry
        pump_load = pump_load.loc[~pump_load.index.duplicated(keep="first")]

        pump_load = pump_load.groupby(df.index.names).sum()
        return df - pump_load

    def plexos_cost_unserved_energy(self, df: pd.DataFrame, **_) -> pd.DataFrame:
        """Creates a region_Cost_Unserved_Energy property for PLEXOS result sets

        Args:
            df (pd.DataFrame): region_Unserved_Energy df

        Returns:
            pd.DataFrame: region_Cost_Unserved_Energy df
        """
        return df * mconfig.parser("formatter_settings", "VoLL")

    def reeds_region_total_load(
        self, df: pd.DataFrame, timescale: str = "year"
    ) -> pd.DataFrame:
        """Creates a region_Load property for ReEDS results sets

        ReEDS does not include storage charging in total load
        This is added to region_Demand to get region_Load

        Args:
            df (pd.DataFrame): region_Demand df
            timescale (str, optional): Data timescale.
                Defaults to 'year'.

        Returns:
            pd.DataFrame: region_Load df
        """
        data_chunks = []
        for file in self.files_list:
            processed_data = self.model.get_processed_data(
                "region", "stor_in", "interval", file
            )

            if processed_data.empty is True:
                logger.info("region_Load will equal region_Demand")
                return df

            data_chunks.append(processed_data)

        pump_load = pd.concat(data_chunks, copy=False)
        if timescale == "year":
            pump_load = self.annualize_property(pump_load)
            all_col = list(pump_load.index.names)
            [all_col.remove(x) for x in ["tech", "sub-tech", "units", "season"]]
        else:
            all_col = list(pump_load.index.names)
            [all_col.remove(x) for x in ["tech", "sub-tech", "units"]]
        pump_load = pump_load.groupby(all_col).sum()

        load = df.merge(pump_load, on=all_col, how="outer")
        load[0] = load["0_x"] + load["0_y"]
        load[0] = load[0].fillna(load["0_x"])
        load = load.drop(["0_x", "0_y"], axis=1)
        return load

    def reeds_reserve_provision(self, df: pd.DataFrame, **_) -> pd.DataFrame:
        """Creates a reserve_Provision property for ReEDS result sets

        Args:
            df (pd.DataFrame): reserves_generators_Provision df

        Returns:
            pd.DataFrame: reserve_Provision df
        """
        return df.groupby(
            ["timestamp", "Type", "parent", "region", "season", "units"]
        ).sum()

    def reeds_generator_vom_cost(self, df: pd.DataFrame, **_) -> pd.DataFrame:
        """Creates a generator_VO&M property for ReEDS result sets

        Args:
            df (pd.DataFrame): generator_Total_Generation_Cost df

        Returns:
            pd.DataFrame: generator_VO&M df
        """
        return df.xs("op_vom_costs", level="cost_type")

    def reeds_generator_fuel_cost(self, df: pd.DataFrame, **_) -> pd.DataFrame:
        """Creates a generator_Fuel_Cost property for ReEDS result sets

        Args:
            df (pd.DataFrame): generator_Total_Generation_Cost df

        Returns:
            pd.DataFrame: generator_Fuel_Cost df
        """
        return df.xs("op_fuelcosts_objfn", level="cost_type")

    def reeds_generator_reserve_vom_cost(self, df: pd.DataFrame, **_) -> pd.DataFrame:
        """Creates a generator_Reserves_VOM_Cost property for ReEDS result sets

        Args:
            df (pd.DataFrame): generator_Total_Generation_Cost df

        Returns:
            pd.DataFrame: generator_Reserves_VOM_Cost df
        """
        return df.xs("op_operating_reserve_costs", level="cost_type")

    def reeds_generator_fom_cost(self, df: pd.DataFrame, **_) -> pd.DataFrame:
        """Creates a generator_FOM_Cost property for ReEDS result sets

        Args:
            df (pd.DataFrame): generator_Total_Generation_Cost df

        Returns:
            pd.DataFrame: generator_FOM_Cost df
        """
        return df.xs("op_fom_costs", level="cost_type")

    def annualize_property(self, df: pd.DataFrame, **_) -> pd.DataFrame:
        """Annualizes any property, groups by year

        Args:
            df (pd.DataFrame): multiindex dataframe with timestamp level.

        Returns:
            pd.DataFrame: df with timestamp grouped by year.
        """
        index_names = list(df.index.names)
        index_names.remove("timestamp")
        timestamp_annualized = [
            pd.to_datetime(df.index.get_level_values("timestamp").year.astype(str))
        ]
        timestamp_annualized.extend(index_names)
        return df.groupby(timestamp_annualized).sum()
