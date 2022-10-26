"""Contains class and methods used to creates extra properties 
required by the Marmot plotter.

@author: Daniel Levie
"""

import logging
import pandas as pd
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

        avail_gen = self.model.combine_models(data_chunks)
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
        pump_load_chunks = []
        batt_load_chunks = []
        for file in self.files_list:
            pump_load_data = self.model.get_processed_data(
                "generator", "Pump Load", timescale, file
            )

            batt_load_data = self.model.get_processed_data(
                "batterie", "Load", timescale, file
            )

            if pump_load_data.empty is True and batt_load_data.empty is True:
                logger.info("Total Demand will equal Total Load")
                return df

            if pump_load_data.empty is False:
                pump_load_chunks.append(pump_load_data)
                pump_load = self.model.combine_models(pump_load_chunks)
                pump_load = pump_load.groupby(df.index.names).sum()
                df = df - pump_load

            if batt_load_data.empty is False:
                batt_load_chunks.append(batt_load_data)
                batt_load = self.model.combine_models(batt_load_chunks)
                print(df)
                print(batt_load)
                batt_load = batt_load.groupby(df.index.names).sum()
                df = df - batt_load

        return df

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
        load["values"] = load["0_x"] + load["0_y"]
        load["values"] = load["values"].fillna(load["0_x"])
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

    def siip_generator_curtailment(
        self, df: pd.DataFrame, timescale: str = "interval"
    ) -> pd.DataFrame:
        """Creates a generator_Curtailment property for SIIP result sets

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
                "generator", "generation_availability", timescale, file
            )

            if processed_data.empty is True:
                logger.warning(
                    "generation_availability & "
                    "generation_actual are required "
                    "for Curtailment calculation"
                )
                return pd.DataFrame()

            data_chunks.append(processed_data)

        avail_gen = self.model.combine_models(data_chunks)

        # Only use gens unique to avail_gen and filter generator_Generation df
        unique_gens = avail_gen.index.get_level_values('gen_name').unique()
        map_gens = df.index.isin(unique_gens, level='gen_name')

        return avail_gen - df.loc[map_gens,:]

    def siip_region_total_load(
        self, df: pd.DataFrame, timescale: str = "interval"
    ) -> pd.DataFrame:
        """Creates a region_Load property for SIIP results sets

        SIIP does not include storage charging in total load
        This is added to region_Demand to get region_Load

        Args:
            df (pd.DataFrame): region_Demand df
            timescale (str, optional): Data timescale.
                Defaults to 'interval'.

        Returns:
            pd.DataFrame: region_Load df
        """
        data_chunks = []
        for file in self.files_list:
            processed_data = self.model.get_processed_data(
                "generator", "pump", timescale, file
            )

            if processed_data.empty is True:
                logger.info("region_Load will equal region_Demand")
                return df

            data_chunks.append(processed_data)

        pump_load = self.model.combine_models(data_chunks)
        pump_load = pump_load.groupby(df.index.names).sum()

        return df + pump_load