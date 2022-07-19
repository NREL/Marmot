"""Main formatting module for SIIP results,
Contains classes and methods specific to SIIP outputs.
Inherits the Process class.

@author: Daniel Levie
"""

import re
import logging
import pandas as pd
from pathlib import Path

import marmot.utils.mconfig as mconfig
import marmot.metamanagers.write_siip_metadata as write_siip_metadata
from marmot.metamanagers.read_metadata import MetaData
from marmot.formatters.formatbase import Process
from marmot.formatters.formatextra import ExtraProperties

logger = logging.getLogger("formatter." + __name__)
formatter_settings = mconfig.parser("formatter_settings")


class ProcessSIIP(Process):
    """Process SIIP specific data from a SIIP result set."""

    # Maps SIIP property names to Marmot names
    PROPERTY_MAPPING: dict = {
        "generator_generation_actual": "generator_Generation",
        "generator_generation_availability": "generator_Available_Capacity",
        "region_regional_load": "region_Demand",
        "reserves_generators_reserve_contribution": "reserves_generators_Provision"}
    """Maps simulation model property names to Marmot property names"""
    # Extra custom properties that are created based off existing properties.
    # The dictionary keys are the existing properties and the values are the new
    # property names and methods used to create it.
    EXTRA_MARMOT_PROPERTIES: dict = {
        "generator_Generation": [
            ("generator_Curtailment", ExtraProperties.siip_generator_curtailment),
            ("generator_Generation_Annual", ExtraProperties.annualize_property)],
        "generator_Curtailment": [
            ("generator_Curtailment_Annual", ExtraProperties.annualize_property)],
        "region_Demand": [("region_Load", ExtraProperties.siip_region_total_load)]}
    """Dictionary of Extra custom properties that are created based off existing properties."""

    def __init__(
        self,
        input_folder: Path,
        output_file_path: Path,
        *args,
        Region_Mapping: pd.DataFrame = pd.DataFrame(),
        **kwargs,
    ):
        """
        Args:
            input_folder (Path): Folder containing csv files.
            output_file_path (Path): Path to formatted h5 output file.
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

        self.metadata = MetaData(
            self.output_file_path.parent,
            read_from_formatted_h5=True,
            Region_Mapping=Region_Mapping,
        )

    @property
    def get_input_data_paths(self) -> list:
        """For SIIP returns paths to input folders
        
        SIIP places individual input files in a single folder,
        It does not combine all results into a single file 
        like PLEXOS or ReEDS.
        If models have been split into partitions, returns list 
        of partition folders, else returns scenario folder.
        """
        if self._get_input_data_paths is None:

            folders = []
            # First checks if input_folder contains partition folders
            # Adds to list of folders if a dir
            for names in self.input_folder.iterdir():
                if names.is_dir():
                    folders.append(names.name)
            # If no partition folders found, use input_folder as file directory
            # This is a non partitioned run
            if not folders:
                self._get_input_data_paths = [self.input_folder]
            else:
                # List of all partition folders in input folder in alpha numeric order
                self._get_input_data_paths = sorted(folders, key=lambda x: int(re.sub("\D", "0", x)))

        return self._get_input_data_paths

    def output_metadata(self, *_) -> None:
        """Add SIIP specific metadata to formatted h5 file .
        """
        json_metadata_file = self.input_folder.joinpath("metadata.json")
        write_siip_metadata.metadata_to_h5(json_metadata_file, 
            self.output_file_path)

    def get_processed_data(
        self, prop_class: str, prop: str, timescale: str, model_name: str
    ) -> pd.DataFrame:
        """Handles the pulling of data from the SIIP input folder 
        and then passes the data to one of the formating functions

        Args:
            prop_class (str): Property class e.g Region, Generator, Zone etc
            prop (str): Property e.g generation_actual, regional_load etc.
            timescale (str): Data timescale, e.g interval, month, year, etc.
            model_name (str): name of model to process.

        Returns:
            pd.DataFrame: Formatted results dataframe.
        """
        logger.info(f"      {model_name}")

        siip_partition = self.data_collection.get(model_name)
        try:
            df: pd.DataFrame = pd.read_csv(siip_partition.joinpath(prop + ".csv"))
        except FileNotFoundError:
            df = self.report_prop_error(prop, prop_class)
            return df
        # Convert to datetime64[ns] type
        df.DateTime = pd.to_datetime(df.DateTime)
        df = df.rename(columns={"DateTime": "timestamp"})

        # Get desired method
        process_att = getattr(self, f"df_process_{prop_class}")
        # Process attribute and return to df
        df = process_att(df)
        # Add region mapping
        if 'region' in df.columns and not self.Region_Mapping.empty:
            df = df.merge(self.Region_Mapping, how="left", on="region")
        # Set multiindex
        df_idx_col = list(df.columns)
        df_idx_col.pop(df_idx_col.index("values"))
        # move timestamp to start of df
        df_idx_col.insert(0, df_idx_col.pop(df_idx_col.index("timestamp")))
        df = df.set_index(df_idx_col)
        
        df_units = "MW"
        # find unit conversion values
        converted_units = self.UNITS_CONVERSION.get(df_units, (df_units, 1))
        # Convert units and add unit column to index
        df = df * converted_units[1]
        units_index = pd.Index([converted_units[0]] * len(df), name="units")
        df.set_index(units_index, append=True, inplace=True)
        df["values"] = pd.to_numeric(df["values"], downcast="float")
        return df

    def df_process_generator(self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Format SIIP Generator class data

        Args:
            df (pd.DataFrame): Data Frame to process

        Returns:
            pd.DataFrame: dataframe formatted to generator class spec
        """
        region_gen_cat_meta = self.metadata.region_generator_category(self.output_file_path.name).reset_index()

        df = df.melt(id_vars=["timestamp"], var_name="gen_name", value_name="values")
        df = df.merge(region_gen_cat_meta, how='left', on="gen_name")
        return df

    def df_process_region(self, df: pd.DataFrame) -> pd.DataFrame:
        """Format SIIP Region data

        Args:
            df (pd.DataFrame): Data Frame to process

        Returns:
            pd.DataFrame: dataframe formatted to region class spec
        """
        return df.melt(id_vars=["timestamp"], var_name="region", value_name="values")

    def df_process_reserves_generators(self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Format SIIP Reserves Generator data

        Args:
            df (pd.DataFrame): Data Frame to process

        Returns:
            pd.DataFrame: dataframe formatted to reserves generator class spec
        """
        reserves_generators = self.metadata.reserves_generators(self.output_file_path.name)
        region_gen_cat_meta = self.metadata.region_generator_category(self.output_file_path.name).reset_index()

        df = df.melt(id_vars=["timestamp"], var_name="gen_name_reserve", value_name="values")
        df = df.merge(reserves_generators, how='left', on="gen_name_reserve")
        df = df.drop("gen_name_reserve", axis=1)
        df = df.merge(region_gen_cat_meta, how='left', on="gen_name")
        return df


