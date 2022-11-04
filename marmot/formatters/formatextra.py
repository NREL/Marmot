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

    # Extra custom properties that are created based off existing properties.
    # The dictionary keys are the existing properties and the values are the new
    # property names and methods used to create it.
    EXTRA_MARMOT_PROPERTIES: dict = {}

    def __init__(self, model: Process):
        """
        Args:
            model (Process): model specific instance of a Process class,
                e.g ProcessPLEXOS, ProcessReEDS
        """
        self.model = model
        self.files_list = model.get_input_data_paths

    def get_extra_properties(self, key):

        if key in self.EXTRA_MARMOT_PROPERTIES:
            extra_properties = []
            extra_prop_functions = (
                self.EXTRA_MARMOT_PROPERTIES[key]
            )
            for prop_function_tup in extra_prop_functions:
                prop_name, prop_function = prop_function_tup
                extra_properties.append((prop_name, getattr(self, prop_function)))

            return extra_properties
        else:
            return None

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


class ExtraPLEXOSProperties(ExtraProperties):

    EXTRA_MARMOT_PROPERTIES: dict = {
        "generator_Generation": [
            ("generator_Curtailment", "generator_curtailment"),
            ("generator_Generation_Annual", "annualize_property"),
        ],
        "region_Unserved_Energy": [
            ("region_Cost_Unserved_Energy", "cost_unserved_energy")
        ],
        "zone_Unserved_Energy": [
            ("zone_Cost_Unserved_Energy", "cost_unserved_energy")
        ],
        "region_Load": [
            ("region_Load_Annual", "annualize_property"),
            ("region_Demand", "demand"),
        ],
        "zone_Load": [
            ("zone_Load_Annual", "annualize_property"),
            ("zone_Demand", "demand"),
        ],
        "generator_Pump_Load": [
            ("generator_Pump_Load_Annual", "annualize_property")
        ],
        "reserves_generators_Provision": [
            ("reserves_generators_Provision_Annual", "annualize_property")
        ],
        "generator_Curtailment": [
            ("generator_Curtailment_Annual", "annualize_property")
        ],
        "region_Demand": [("region_Demand_Annual", "annualize_property")],
        "zone_Demand": [("zone_Demand_Annual", "annualize_property")],
    }
    """Dictionary of Extra custom properties that are created based off existing properties."""


    def generator_curtailment(
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

    def demand(
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
                batt_load = batt_load.groupby(df.index.names).sum()
                df = df - batt_load

        return df

    def cost_unserved_energy(self, df: pd.DataFrame, **_) -> pd.DataFrame:
        """Creates a region_Cost_Unserved_Energy property for PLEXOS result sets

        Args:
            df (pd.DataFrame): region_Unserved_Energy df

        Returns:
            pd.DataFrame: region_Cost_Unserved_Energy df
        """
        return df * mconfig.parser("formatter_settings", "VoLL")


class ExtraReEDSProperties(ExtraProperties):

    EXTRA_MARMOT_PROPERTIES: dict = {
        "generator_Total_Generation_Cost": [
            ("generator_VOM_Cost", "generator_vom_cost"),
            ("generator_Fuel_Cost", "generator_fuel_cost"),
            (
                "generator_Reserves_VOM_Cost",
                "generator_reserve_vom_cost",
            ),
            ("generator_FOM_Cost", "generator_fom_cost"),
        ],
        "reserves_generators_Provision": [
            ("reserve_Provision", "reserve_provision")
        ],
        "region_Demand": [
            ("region_Demand_Annual", "annualize_property"),
            ("region_Load", "region_total_load"),
        ],
        "generator_Curtailment": [
            ("generator_Curtailment_Annual", "annualize_property")
        ],
        "generator_Pump_Load": [
            ("generator_Pump_Load_Annual", "annualize_property")
        ],
        "region_Load": [("region_Load_Annual", "annualize_property")],

    }
    """Dictionary of Extra custom properties that are created based off existing properties."""


    def region_total_load(
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
        [all_col.remove(x) for x in ["tech", "sub-tech", "units", "season"] if x in all_col]
        pump_load = pump_load.groupby(all_col).sum()

        load = df.merge(pump_load, on=all_col, how="outer")

        load["values"] = load["values_x"] + load["values_y"]
        load["values"] = load["values"].fillna(load["values_x"])
        load = load.drop(["values_x", "values_y"], axis=1)
        return load

    def reserve_provision(self, df: pd.DataFrame, **_) -> pd.DataFrame:
        """Creates a reserve_Provision property for ReEDS result sets

        Args:
            df (pd.DataFrame): reserves_generators_Provision df

        Returns:
            pd.DataFrame: reserve_Provision df
        """
        return df.groupby(
            ["timestamp", "Type", "parent", "region", "season", "units"]
        ).sum()

    def generator_vom_cost(self, df: pd.DataFrame, **_) -> pd.DataFrame:
        """Creates a generator_VO&M property for ReEDS result sets

        Args:
            df (pd.DataFrame): generator_Total_Generation_Cost df

        Returns:
            pd.DataFrame: generator_VO&M df
        """
        return df.xs("op_vom_costs", level="cost_type")

    def generator_fuel_cost(self, df: pd.DataFrame, **_) -> pd.DataFrame:
        """Creates a generator_Fuel_Cost property for ReEDS result sets

        Args:
            df (pd.DataFrame): generator_Total_Generation_Cost df

        Returns:
            pd.DataFrame: generator_Fuel_Cost df
        """
        return df.xs("op_fuelcosts_objfn", level="cost_type")

    def generator_reserve_vom_cost(self, df: pd.DataFrame, **_) -> pd.DataFrame:
        """Creates a generator_Reserves_VOM_Cost property for ReEDS result sets

        Args:
            df (pd.DataFrame): generator_Total_Generation_Cost df

        Returns:
            pd.DataFrame: generator_Reserves_VOM_Cost df
        """
        return df.xs("op_operating_reserve_costs", level="cost_type")

    def generator_fom_cost(self, df: pd.DataFrame, **_) -> pd.DataFrame:
        """Creates a generator_FOM_Cost property for ReEDS result sets

        Args:
            df (pd.DataFrame): generator_Total_Generation_Cost df

        Returns:
            pd.DataFrame: generator_FOM_Cost df
        """
        return df.xs("op_fom_costs", level="cost_type")


class ExtraReEDSIndiaProperties(ExtraReEDSProperties):

    def region_total_load(
        self, df: pd.DataFrame, timescale: str = "year"
    ) -> pd.DataFrame:
        """Creates a region_Load property for ReEDS India results sets

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
                "generator", "stor_charge", "interval", file
            )

            if processed_data.empty is True:
                logger.info("region_Load will equal region_Demand")
                return df

            data_chunks.append(processed_data)

        pump_load = pd.concat(data_chunks, copy=False)
        if timescale == "year":
            pump_load = self.annualize_property(pump_load)

        all_col = list(pump_load.index.names)
        [all_col.remove(x) for x in ["tech", "gen_name", "units", "season"] if x in all_col]
        pump_load = pump_load.groupby(all_col).sum()

        load = df.merge(pump_load, on=all_col, how="outer")
        load["values"] = load["values_x"] + load["values_y"]
        load["values"] = load["values"].fillna(load["values_x"])
        load = load.drop(["values_x", "values_y"], axis=1)
        return load


class ExtraSIIProperties(ExtraProperties):

    EXTRA_MARMOT_PROPERTIES: dict = {
        "generator_Generation": [
            ("generator_Curtailment", "generator_curtailment"),
            ("generator_Generation_Annual", "annualize_property")],
        "generator_Curtailment": [
            ("generator_Curtailment_Annual", "annualize_property")],
        "region_Demand": [("region_Load", "region_total_load")]}
    """Dictionary of Extra custom properties that are created based off existing properties."""


    def generator_curtailment(
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

    def region_total_load(
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