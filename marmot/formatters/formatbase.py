"""Base formatting module for all simulation model results,
Inherited by all model specific Process classes 

@author: Daniel Levie
"""

import re
import logging
import pandas as pd
from pathlib import Path
from typing import Dict

logger = logging.getLogger("formatter." + __name__)


class Process:
    """Base class for processing simulation model data."""

    # Maps sim model property names to Marmot names,
    # unchanged names not included
    PROPERTY_MAPPING: dict = {}
    """Maps simulation model property names to Marmot property names"""
    # Extra custom properties that are created based off existing properties.
    # The dictionary keys are the existing properties and the values are the new
    # property names and methods used to create it.
    EXTRA_MARMOT_PROPERTIES: dict = {}
    """Dictionary of Extra custom properties that are created based off existing properties."""
    # Conversion units dict, key values is a tuple of new unit name and
    # conversion multiplier
    UNITS_CONVERSION: dict = {
        "kW": ("MW", 1e-3),
        "MW": ("MW", 1),
        "GW": ("MW", 1e3),
        "TW": ("MW", 1e6),
        "kWh": ("MWh", 1e-3),
        "MWh": ("MWh", 1),
        "GWh": ("MWh", 1e3),
        "TWh": ("MWh", 1e6),
        "lb": ("kg", 0.453592),
        "ton": ("kg", 907.18474),
        "kg": ("kg", 1),
        "tonne": ("kg", 1000),
        "metric tons": ("kg", 1000),
        "$": ("$", 1),
        "2004$": ("$", 1),
        "$000": ("$", 1000),
        "h": ("h", 1),
        "MMBTU": ("MMBTU", 1),
        "GBTU": ("MMBTU", 1000),
        'GJ"': ("MMBTU", 0.947817),
        "TJ": ("MMBTU", 947.817120),
        "$/MW": ("$/MW", 1),
        "$/MWh": ("$/MWh", 1),
        "lb/MWh": ("kg/MWh", 0.453592),
        "Kg/MWh": ("Kg/MWh", 1),
        "Quads": ("Quads", 1),
        "MW/yr": ("MW/yr", 1),
        "frac": ("fraction", 1),
        "fraction": ("fraction", 1),
        None: ("unitless", 1),
        "unitless": ("unitless", 1),
    }
    """Dictionary to convert units to standard values used by Marmot"""

    def __init__(
        self,
        input_folder: Path,
        output_file_path: Path,
        *_,
        Region_Mapping : pd.DataFrame = pd.DataFrame(),
        emit_names : pd.DataFrame = pd.DataFrame(),
        **__,
    ):
        """
        Args:
            input_folder (Path): Folder containing model input files.
            output_file_path (Path): Path to formatted h5 output file.
            Region_Mapping (pd.DataFrame, optional): DataFrame to map custom
                regions/zones to create custom aggregations.
                Defaults to pd.DataFrame().
            emit_names (pd.DataFrame, optional): DataFrame with 2 columns to rename
                emission names.
                Defaults to pd.DataFrame().
        """
        self.input_folder = input_folder
        self.output_file_path = Path(output_file_path)
        self.Region_Mapping = Region_Mapping
        self.emit_names = emit_names

        if not self.emit_names.empty:
            self.emit_names_dict = (
                self.emit_names[["Original", "New"]]
                .set_index("Original")
                .to_dict()["New"]
            )

    @property
    def input_folder(self) -> Path:
        """Path to input folder

        Returns:
            Path: input_folder
        """
        return self._input_folder

    @input_folder.setter
    def input_folder(self, value):
        self._get_input_data_paths = None
        self._data_collection = None
        self._input_folder = Path(value)

    @property
    def get_input_data_paths(self) -> list:
        """Gets a list of input files within the scenario folders

        Returns:
            list: list of input filenames to process
        """
        if self._get_input_data_paths is None:
            files = []
            for names in self.input_folder.iterdir():
                files.append(names.name)

            # List of all files in input folder in alpha numeric order
            self._get_input_data_paths = sorted(files, key=lambda x: int(re.sub("\D", "0", x)))
        return self._get_input_data_paths 
    
    @property
    def data_collection(self) -> Dict[str, Path]:
        """Dictionary model names to full filename path 

        Returns:
            dict: data_collection {filename: fullpath}
        """
        if self._data_collection is None:
            self._data_collection = {}
            for file in self.get_input_data_paths:
                self._data_collection[file] = self.input_folder.joinpath(file)
        return self._data_collection

    def output_metadata(self, files_list: list) -> None:
        """method template for output_metadata

        Args:
            files_list (list): list of string files or filenames
        """
        raise NotImplementedError("No default implementation of this functionality")

    def get_processed_data(
        self, prop_class: str, property: str, timescale: str, model_name: str
    ) -> pd.DataFrame:
        """method template for get_processed_data

        Args:
            prop_class (str): class e.g Region, Generator, Zone etc
            property (str): Property e.g gen_out, cap_out etc.
            timescale (str): Data timescale, e.g interval, summary.
            model_name (str): name of model to process.

        Returns:
            pd.DataFrame: pd.DataFrame
        """
        raise NotImplementedError("No default implementation of this functionality")

    def report_prop_error(self, property: str, prop_class: str) -> pd.DataFrame:
        """Outputs a warning message when the get_processed_data method
        cannot find the specified property in the simulation model solution files.

        Args:
            property (str): property e.g Max Capacity, Generation etc.
            prop_class (str): property class e.g Region, Generator, Zone etc.

        Returns:
            pd.DataFrame: Empty DataFrame.
        """
        logger.warning(
            f'CAN NOT FIND "{prop_class} {property}". ' f'"{property}" DOES NOT EXIST'
        )
        logger.info("SKIPPING PROPERTY\n")
        df = pd.DataFrame()
        return df

    def combine_models(self, model_list: list, 
        drop_duplicates: bool = True) -> pd.DataFrame:
        """Combine temporally disaggregated model results.

        Will drop duplicate index entries by default.

        Args:
            model_list (list): list of df models to combine.
            drop_duplicates (bool, optional): Drop duplicate index entries.
                Defaults to True.

        Returns:
            pd.DataFrame: Combined df
        """
        df = pd.concat(model_list, copy=False)
        if drop_duplicates:
            origsize = df.size
            # Remove duplicates; keep first entry
            df = df.loc[~df.index.duplicated(keep="first")]

            if (origsize - df.size) > 0:
                logger.info(f"Drop duplicates removed {origsize-df.size} rows")
        return df