"""Main formatting module for ReEDS results,
Contains classes and methods specific to ReEDS outputs.
Inherits the Process class.

@author: Daniel Levie
"""

import logging
import re
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass, field
import gdxpds
import pandas as pd

import marmot.utils.mconfig as mconfig
from marmot.metamanagers.read_metadata import MetaData
from marmot.formatters.formatbase import Process
from marmot.formatters.formatextra import ExtraProperties

logger = logging.getLogger("formatter." + __name__)
formatter_settings = mconfig.parser("formatter_settings")


class ProcessReEDS(Process):
    """Process ReEDS specific data from a ReEDS result set."""

    # Maps ReEDS property names to Marmot names
    PROPERTY_MAPPING: dict = {
        "generator_gen_out": "generator_Generation",
        "generator_gen_out_ann": "generator_Generation_Annual",
        "generator_cap_out": "generator_Installed_Capacity",
        "generator_curt_out": "generator_Curtailment",
        "region_load_rt": "region_Demand",
        "line_losses_tran_h": "line_Losses",
        "line_tran_flow_power": "line_Flow",
        "line_tran_out": "line_Import_Limit",
        "generator_stor_in": "generator_Pump_Load",
        "storage_stor_energy_cap": "storage_Max_Volume",
        "emission_emit_r": "emission_Production",
        "reserves_generators_opRes_supply_h": "reserves_generators_Provision",
        "reserves_generators_opRes_supply": "reserves_generators_Provision_Annual",
        "generator_systemcost_techba": "generator_Total_Generation_Cost",
    }
    """Maps simulation model property names to Marmot property names"""
    # Extra custom properties that are created based off existing properties.
    # The dictionary keys are the existing properties and the values are the new
    # property names and methods used to create it.
    EXTRA_MARMOT_PROPERTIES: dict = {
        "generator_Total_Generation_Cost": [
            ("generator_VOM_Cost", ExtraProperties.reeds_generator_vom_cost),
            ("generator_Fuel_Cost", ExtraProperties.reeds_generator_fuel_cost),
            (
                "generator_Reserves_VOM_Cost",
                ExtraProperties.reeds_generator_reserve_vom_cost,
            ),
            ("generator_FOM_Cost", ExtraProperties.reeds_generator_fom_cost),
        ],
        "reserves_generators_Provision": [
            ("reserve_Provision", ExtraProperties.reeds_reserve_provision)
        ],
        "region_Demand": [
            ("region_Demand_Annual", ExtraProperties.annualize_property),
            ("region_Load", ExtraProperties.reeds_region_total_load),
        ],
        "generator_Curtailment": [
            ("generator_Curtailment_Annual", ExtraProperties.annualize_property)
        ],
        "generator_Pump_Load": [
            ("generator_Pump_Load_Annual", ExtraProperties.annualize_property)
        ],
        "region_Load": [("region_Load_Annual", ExtraProperties.annualize_property)],
    }
    """Dictionary of Extra custom properties that are created based off existing properties."""

    def __init__(
        self,
        input_folder: Path,
        output_file_path: Path,
        *args,
        process_subset_years: list = None,
        Region_Mapping: pd.DataFrame = pd.DataFrame(),
        **kwargs,
    ):
        """
        Args:
            input_folder (Path): Folder containing csv files.
            output_file_path (Path): Path to formatted h5 output file.
            process_subset_years (list, optional): If provided only process
                years specified. Defaults to None.
            Region_Mapping (pd.DataFrame, optional): DataFrame to map custom
                regions/zones to create custom aggregations.
                Defaults to pd.DataFrame().
            **kwargs
                These parameters will be passed to the Process 
                class.
        """
        # Instantiation of Process Base class
        super().__init__(
            input_folder, output_file_path, *args, Region_Mapping=Region_Mapping, **kwargs
        )
        # Internal cached data is saved to the following variables.
        # To access the values use the public api e.g self.property_units
        self._property_units: dict = {}
        self._wind_resource_to_pca = None

        self.metadata = MetaData(
            output_file_path.parent,
            read_from_formatted_h5=True,
            Region_Mapping=Region_Mapping,
        )

        if process_subset_years:
            # Ensure values are ints
            process_subset_years = list(map(int, process_subset_years))
            logger.info(f"Processing subset of ReEDS years: {process_subset_years}")
        self.process_subset_years = process_subset_years

    @property
    def property_units(self) -> dict:
        """Gets the property units from data, e.g MW, MWh

        Returns:
            dict: property_units
        """
        return self._property_units

    @property_units.setter
    def property_units(self, gdx_filename: str):
        """Sets the property units, adds values to a dict

        Args:
            gdx_filename (str): Full path to gdx_file
        """
        # Extracts values between markers
        symbol_marker = "--(.*?)--"

        symbol_list = gdxpds.list_symbols(gdx_filename)
        for symbol in symbol_list:
            unit = re.search(symbol_marker, symbol.description)
            if unit:
                unit = unit.group(1)
            if symbol.name not in self._property_units:
                self._property_units[symbol.name] = unit

    @property
    def wind_resource_to_pca(self) -> dict:
        """Get the wind resource (s) to pca/region mapping

        Returns:
            dict: wind_resource_to_pca mapping
        """
        return self._wind_resource_to_pca

    @wind_resource_to_pca.setter
    def wind_resource_to_pca(self, h5_filename: str):
        """Sets the wind_resource_to_pca mapping

        Args:
            h5_filename (str): formatted h5 filename
        """
        if self.metadata.filename != h5_filename or self._wind_resource_to_pca is None:
            regions = self.metadata.regions(h5_filename)
            self._wind_resource_to_pca = (
                regions[["category", "region"]]
                .set_index("category")
                .to_dict()["region"]
            )

    @property
    def get_input_data_paths(self) -> list:
        """Gets a list of input gdx files within the scenario folders"""
        if self._get_input_data_paths is None:
            reeds_outputs_dir = self.input_folder.joinpath("outputs")
            files = []
            for names in reeds_outputs_dir.iterdir():
                if names.name == f"rep_{self.input_folder.name}.gdx":
                    files.append(names.name)

                    self.property_units = str(names)

            # List of all files in input folder in alpha numeric order
            self._get_input_data_paths = sorted(files, key=lambda x: int(re.sub("\D", "0", x)))
        return self._get_input_data_paths
    
    @property
    def data_collection(self) -> Dict[str, Path]:
        """Dictionary input file names to full filename path 

        Returns:
            dict: data_collection {filename: fullpath}
        """
        if self._data_collection is None:
            self._data_collection = {}
            for file in self.get_input_data_paths:
                self._data_collection[file] = self.input_folder.joinpath("outputs", file)
        return self._data_collection

    def output_metadata(self, files_list: list) -> None:
        """Add ReEDS specific metadata to formatted h5 file .

        Args:
            files_list (list): List of all gdx files in inputs
                folder in alpha numeric order.
        """
        for partition in files_list:
            region_df = pd.read_csv(
                self.input_folder.joinpath("inputs_case", "regions.csv")
            )
            region_df.rename(columns={"p": "name", "s": "category"}, inplace=True)
            region_df.to_hdf(
                self.output_file_path,
                key=f"metadata/{partition}/objects/regions",
                mode="a",
            )

    def get_processed_data(
        self, prop_class: str, prop: str, timescale: str, model_name: str
    ) -> pd.DataFrame:
        """Handles the pulling of data from the ReEDS gdx
        file and then passes the data to one of the formating functions

        Args:
            prop_class (str): Property class e.g Region, Generator, Zone etc
            prop (str): Property e.g gen_out, cap_out etc.
            timescale (str): Data timescale, e.g interval, summary.
            model_name (str): name of model to process.

        Returns:
            pd.DataFrame: Formatted results dataframe.
        """
        # Set wind_resource_to_pca dict
        self.wind_resource_to_pca = self.input_folder.name

        gdx_file = self.data_collection.get(model_name)
        logger.info(f"      {model_name}")
        try:
            df: pd.DataFrame = gdxpds.to_dataframe(str(gdx_file), prop)[prop]
        except gdxpds.tools.Error:
            df = self.report_prop_error(prop, prop_class)
            return df
        # Get column names
        reeds_prop_cols = PropertyColumns()
        df.columns = getattr(reeds_prop_cols, prop)
        if "region" in df.columns:
            df.region = df.region.map(lambda x: self.wind_resource_to_pca.get(x, x))
            if not self.Region_Mapping.empty:
                # Merge in region mapping, drop any na columns
                df = df.merge(self.Region_Mapping, how="left", on="region")
                df.dropna(axis=1, how="all", inplace=True)
        # Get desired method, used for extra processing if needed
        process_att = getattr(self, f"df_process_{prop_class}", None)
        if process_att:
            # Process attribute and return to df
            df = process_att(df, prop, str(gdx_file))

        df.year = df.year.astype(int)
        if self.process_subset_years:
            df = df.loc[df.year.isin(self.process_subset_years)]

        if timescale == "interval":
            df = self.merge_timeseries_block_data(df)
        else:
            df["timestamp"] = pd.to_datetime(df.year.astype(str))
        if "year" in df.columns:
            df = df.drop(["year"], axis=1)

        df_col = list(df.columns)
        df_col.remove("Value")
        df_col.insert(0, df_col.pop(df_col.index("timestamp")))
        df = df.groupby(df_col).sum()
        df = df.sort_index(level=["timestamp"])

        df_units = self.property_units[prop]
        # find unit conversion values
        converted_units = self.UNITS_CONVERSION.get(df_units, (df_units, 1))

        # Convert units and add unit column to index
        df = df * converted_units[1]
        units_index = pd.Index([converted_units[0]] * len(df), name="units")
        df.set_index(units_index, append=True, inplace=True)
        df.rename(columns={"Value": 0}, inplace=True)
        return df

    def merge_timeseries_block_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Merge chronological time intervals with reeds timeslices

        Args:
            df (pd.DataFrame): input dataframe

        Returns:
            pd.DataFrame: df with merged in timeseries data
        """
        timeslice_mapping_file = pd.read_csv(
            self.input_folder.joinpath("inputs_case", "h_dt_szn.csv")
        )

        # All year timeslice mappings are the same, defaulting to 2007
        timeslice_mapping_file = timeslice_mapping_file.loc[
            timeslice_mapping_file.year == 2007
        ]
        timeslice_mapping_file = timeslice_mapping_file.drop("year", axis=1)

        year_list = df.year.unique()
        year_list.sort()

        year_chunks = []
        for year in year_list:
            year_chunks.extend(pd.date_range(f"{year}-01-01", periods=8760, freq="H"))

        datetime_df = pd.DataFrame(
            data=list(range(1, 8761)) * len(year_list),
            index=pd.to_datetime(year_chunks),
            columns=["hour"],
        )
        datetime_df["year"] = datetime_df.index.year.astype(int)
        datetime_df = datetime_df.reset_index()
        datetime_df.rename(columns={"index": "timestamp"}, inplace=True)

        datetime_block = datetime_df.merge(timeslice_mapping_file, on="hour")
        datetime_block.sort_values(by=["timestamp", "h"], inplace=True)
        df_merged = df.merge(datetime_block, on=["year", "h"])
        return df_merged.drop(["h", "hour", "year"], axis=1)

    def df_process_generator(
        self, df: pd.DataFrame, prop: str = None, gdx_file: str = None
    ) -> pd.DataFrame:
        """Does any additional processing for generator properties

        Args:
            df (pd.DataFrame): input dataframe
            prop (str, optional): ReEDS input property.
                Defaults to None.
            gdx_file (str, optional): String path to GDX file.
                Defaults to None.

        Returns:
            pd.DataFrame:
        """

        if "tech" not in df.columns:
            df["tech"] = "reeds_vre"
        df["gen_name"] = df.tech + "_" + df.region

        if (
            prop == "gen_out"
            or prop == "gen_out_ann"
            and formatter_settings["exclude_pumping_from_reeds_storage_gen"]
        ):
            if prop == "gen_out":
                stor_prop_name = "stor_out"
                group_list = ["tech", "region", "h", "year"]
            else:
                stor_prop_name = "stor_inout"
                group_list = ["tech", "region", "year"]
            try:
                stor_out: pd.DataFrame = gdxpds.to_dataframe(gdx_file, stor_prop_name)[
                    stor_prop_name
                ]
            except gdxpds.tools.Error:
                stor_out = self.report_prop_error(stor_prop_name, "storage")
                return df
            reeds_prop_cols = PropertyColumns()
            stor_out.columns = getattr(reeds_prop_cols, stor_prop_name)
            if prop == "gen_out_ann":
                stor_out = stor_out.loc[stor_out.type == "out"]
            stor_out = stor_out.groupby(group_list).sum()

            df = df.merge(stor_out, on=group_list, how="outer")
            df["Value"] = df["Value_y"]
            df["Value"] = df["Value"].fillna(df["Value_x"])
            df["Value"].to_numpy()[df["Value"].to_numpy() < 0] = 0
            df = df.drop(["Value_x", "Value_y"], axis=1)

        return df

    def df_process_line(
        self, df: pd.DataFrame, prop: str = None, gdx_file: str = None
    ) -> pd.DataFrame:
        """Does any additional processing for line properties

        Args:
            df (pd.DataFrame): input dataframe
            prop (str, optional): ReEDS input property.
                Defaults to None.
            gdx_file (str, optional): String path to GDX file.
                Defaults to None.

        Returns:
            pd.DataFrame:
        """

        df["line_name"] = df["region_from"] + "_" + df["region_to"]
        return df

    def df_process_reserves_generators(
        self, df: pd.DataFrame, prop: str = None, gdx_file: str = None
    ) -> pd.DataFrame:
        """Does any additional processing for reserves_generators properties

        Args:
            df (pd.DataFrame): input dataframe
            prop (str, optional): ReEDS input property.
                Defaults to None.
            gdx_file (str, optional): String path to GDX file.
                Defaults to None.

        Returns:
            pd.DataFrame:
        """
        df["Type"] = "-"
        return df


@dataclass
class PropertyColumns:
    """ReEDS property column names"""

    gen_out: List = field(
        default_factory=lambda: ["tech", "region", "h", "year", "Value"]
    )
    """ReEDS 'gen_out' property columns (Marmot generator_Generation property)"""
    gen_out_ann: List = field(
        default_factory=lambda: ["tech", "region", "year", "Value"]
    )
    """ReEDS 'gen_out_ann' property columns (Marmot generator_Generation_Annual property)"""
    cap_out: List = field(default_factory=lambda: ["tech", "region", "year", "Value"])
    """ReEDS 'cap_out' property columns (Marmot generator_Installed_Capacity property)"""
    curt_out: List = field(default_factory=lambda: ["region", "h", "year", "Value"])
    """ReEDS 'curt_out' property columns (Marmot generator_Curtailment property)"""
    load_rt: List = field(default_factory=lambda: ["region", "year", "Value"])
    """ReEDS 'load_rt' property columns (Marmot region_Load_Annual property)"""
    losses_tran_h: List = field(
        default_factory=lambda: [
            "region_from",
            "region_to",
            "h",
            "year",
            "category",
            "Value",
        ]
    )
    """ReEDS 'losses_tran_h' property columns (Marmot line_Losses property)"""
    tran_flow_power: List = field(
        default_factory=lambda: [
            "region_from",
            "region_to",
            "h",
            "category",
            "year",
            "Value",
        ]
    )
    """ReEDS 'tran_flow_power' property columns (Marmot line_Flow property)"""
    tran_out: List = field(
        default_factory=lambda: [
            "region_from",
            "region_to",
            "category",
            "year",
            "Value",
        ]
    )
    """ReEDS 'tran_out' property columns (Marmot line_Import_Limit property)"""
    stor_in: List = field(
        default_factory=lambda: ["tech", "sub-tech", "region", "h", "year", "Value"]
    )
    """ReEDS 'stor_in' property columns (Marmot generator_Pumped_Load property)"""
    stor_out: List = field(
        default_factory=lambda: ["tech", "sub-tech", "region", "h", "year", "Value"]
    )
    """ReEDS 'stor_out' property columns (Marmot storage_Generation property)"""
    stor_inout: List = field(
        default_factory=lambda: ["tech", "sub-tech", "region", "year", "type", "Value"]
    )
    """ReEDS 'stor_inout' property columns (Marmot storage_In_Out property)"""
    stor_energy_cap: List = field(
        default_factory=lambda: ["tech", "sub-tech", "region", "year", "Value"]
    )
    """ReEDS 'stor_energy_cap' property columns (Marmot storage_Max_Volume property)"""
    emit_nat_tech: List = field(
        default_factory=lambda: ["emission_type", "tech", "year", "Value"]
    )
    """ReEDS 'emit_nat_tech' property columns (Marmot emissions property)"""
    emit_r: List = field(
        default_factory=lambda: ["emission_type", "region", "year", "Value"]
    )
    """ReEDS 'emit_r' property columns (Marmot emission_Production_Annual property)"""
    opRes_supply_h: List = field(
        default_factory=lambda: ["parent", "tech", "region", "h", "year", "Value"]
    )
    """ReEDS 'opRes_supply_h' property columns (Marmot reserves_generators_Provision property)"""
    opRes_supply: List = field(
        default_factory=lambda: ["parent", "tech", "region", "year", "Value"]
    )
    """ReEDS 'opRes_supply' property columns (Marmot reserves_generators_Provision_Annual property)"""
    # Marmot generator_Total Generation Cost
    systemcost_techba: List = field(
        default_factory=lambda: ["cost_type", "tech", "region", "year", "Value"]
    )
    """ReEDS 'systemcost_techba' property columns (Marmot generator_Total Generation Cost property)"""
