"""Main formatting module for ReEDS results,
Contains classes and methods specific to ReEDS outputs.
Inherits the Process class.

@author: Daniel Levie
"""

import logging
import pandas as pd
from pathlib import Path
from typing import List

import marmot.utils.mconfig as mconfig
from marmot.metamanagers.read_metadata import MetaData
from marmot.metamanagers.write_siip_metadata import WriteSIIPMetaData
from marmot.formatters.formatbase import Process
from marmot.formatters.formatextra import ExtraProperties

logger = logging.getLogger("formatter." + __name__)
formatter_settings = mconfig.parser("formatter_settings")


class ProcessSIIP(Process):
    """Process SIIP specific data from a SIIP result set."""

    # Maps SIIP property names to Marmot names
    PROPERTY_MAPPING: dict = {
        "marmot_gen_UC": "generator_Generation",
        "marmot_load_UC": "region_Load",
        "marmot_reserve_contribution_UC": "reserve_Provision"}
    """Maps simulation model property names to Marmot property names"""
    # Extra custom properties that are created based off existing properties.
    # The dictionary keys are the existing properties and the values are the new
    # property names and methods used to create it.
    EXTRA_MARMOT_PROPERTIES: dict = {}
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
    def get_input_files(self) -> list:
        """For SIIP returns input_folder folder name
        
        SIIP places individual input files in a single folder,
        It does not combine all results into a single file 
        like PLEXOS or ReEDS. It also does not output 
        partition files for Marmot
        """
        if self._get_input_files == None:
            self._get_input_files = [self.input_folder.name]

        return self._get_input_files

    def output_metadata(self, *_) -> None:
        """Add SIIP specific metadata to formatted h5 file .
        """
        json_metadata_file = self.input_folder.joinpath("metadata.json")
    
        WriteSIIPMetaData.write_to_h5(json_metadata_file, 
        self.output_file_path)

    def get_processed_data(
        self, prop_class: str, prop: str, timescale: str, model_filename: str
    ) -> pd.DataFrame:
        """Handles the pulling of data from the SIIP input folder 
        and then passes the data to one of the formating functions

        Args:
            prop_class (str): Property class e.g Region, Generator, Zone etc
            prop (str): Property e.g marmot_gen_UC, marmot_load_UC etc.
            timescale (str): Data timescale, e.g interval, summary.
            model_filename (str): name of model to process.

        Returns:
            pd.DataFrame: Formatted results dataframe.
        """
        logger.info(f"      {model_filename}")

        try:
            df: pd.DataFrame = pd.read_csv(self.input_folder.joinpath(prop + ".csv"))
        except FileNotFoundError:
            df = self.report_prop_error(prop, prop_class)
            return df
        # Convert to datetime64[ns] type
        df.DateTime = pd.to_datetime(df.DateTime)
        df = df.rename(columns={"DateTime": "timestamp"})

        # Get desired method
        process_att = getattr(self, f"df_process_{prop_class}")
        # Process attribute and return to df
        df = process_att(df, model_filename)
        
        df_units = "MW"
        # find unit conversion values
        converted_units = self.UNITS_CONVERSION.get(df_units, (df_units, 1))

        # Convert units and add unit column to index
        df = df * converted_units[1]
        units_index = pd.Index([converted_units[0]] * len(df), name="units")
        df.set_index(units_index, append=True, inplace=True)
        return df


    def df_process_generator(self, df: pd.DataFrame, model_filename: str) -> pd.DataFrame:

        region_gen_cat_meta = self.metadata.region_generator_category(model_filename).reset_index()

        df = df.melt(id_vars=["timestamp"], var_name="gen_name", value_name=0)
        df = df.merge(region_gen_cat_meta, how='left', on="gen_name")
        if not self.Region_Mapping.empty:
            df = df.merge(self.Region_Mapping, how="left", on="region")

        df_idx_col = list(df.columns)
        df_idx_col.pop(df_idx_col.index(0))
        # move timestamp to start of df
        df_idx_col.insert(0, df_idx_col.pop(df_idx_col.index("timestamp")))
        df = df.set_index(df_idx_col)
        df[0] = pd.to_numeric(df[0], downcast="float")
        return df

    def df_process_region(self, df: pd.DataFrame, model_filename: str) -> pd.DataFrame:

        df = df.melt(id_vars=["timestamp"], var_name="region", value_name=0)
        if not self.Region_Mapping.empty:
            df = df.merge(self.Region_Mapping, how="left", on="region")
        
        df_idx_col = list(df.columns)
        df_idx_col.pop(df_idx_col.index(0))
        # move timestamp to start of df
        df_idx_col.insert(0, df_idx_col.pop(df_idx_col.index("timestamp")))
        df = df.set_index(df_idx_col)
        df[0] = pd.to_numeric(df[0], downcast="float")
        return df

    def df_process_reserves_generators(self, df: pd.DataFrame, model_filename: str) -> pd.DataFrame:
        
        reserves_generators = self.metadata.reserves_generators(model_filename)
        region_gen_cat_meta = self.metadata.region_generator_category(model_filename).reset_index()

        df = df.melt(id_vars=["timestamp"], var_name="gen_name_reserve", value_name=0)
        df = df.merge(reserves_generators, how='left', on="gen_name_reserve")
        df = df.drop("gen_name_reserve", axis=1)
        df = df.merge(region_gen_cat_meta, how='left', on="gen_name")
        if not self.Region_Mapping.empty:
            df = df.merge(self.Region_Mapping, how="left", on="region")

        df_idx_col = list(df.columns)
        df_idx_col.pop(df_idx_col.index(0))
        # move timestamp to start of df
        df_idx_col.insert(0, df_idx_col.pop(df_idx_col.index("timestamp")))
        df = df.set_index(df_idx_col)
        df[0] = pd.to_numeric(df[0], downcast="float")
        return df


