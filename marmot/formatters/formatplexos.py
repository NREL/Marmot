"""Main formatting module for PLEXOS results,
Contains classes and methods specific to PLEXOS outputs.
Inherits the Process class.

@author: Daniel Levie
"""

import re
import pandas as pd
import h5py
import logging
from typing import Dict
from pathlib import Path
from marmot.utils.error_handler import MissingH5PLEXOSDataError, PropertyNotFound
from marmot.metamanagers.read_metadata import MetaData
from marmot.formatters.formatbase import Process
from marmot.formatters.formatextra import ExtraPLEXOSProperties

try:
    # Import as Submodule
    from marmot.h5plexos.h5plexos.query import PLEXOSSolution
except ModuleNotFoundError:
    from h5plexos.query import PLEXOSSolution

logger = logging.getLogger("formatter." + __name__)


class ProcessPLEXOS(Process):
    """Process PLEXOS class specific data from a h5plexos database."""

    # Maps PLEXOS property names to Marmot names,
    # unchanged names not included
    PROPERTY_MAPPING: dict = {
        "generator_Start_&_Shutdown_Cost": "generator_Start_and_Shutdown_Cost",
        "generator_VO&M_Cost": "generator_VOM_Cost",
        "generator_Reserves_VO&M_Cost": "generator_Reserves_VOM_Cost",
    }
    """Maps simulation model property names to Marmot property names"""

    EXTRA_PROPERTIES_CLASS = ExtraPLEXOSProperties

    def __init__(
        self,
        input_folder: Path,
        output_file_path: Path,
        *args,
        plexos_block: str = "ST",
        Region_Mapping: pd.DataFrame = pd.DataFrame(),
        **kwargs,
    ):
        """
        Args:
            input_folder (Path): Folder containing h5plexos h5 files.
            output_file_path (Path): Path to formatted h5 output file.
            plexos_block (str, optional): PLEXOS results type. Defaults to 'ST'.
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
        self.plexos_block = plexos_block
        self.metadata = MetaData(
            self.input_folder, read_from_formatted_h5=False, Region_Mapping=Region_Mapping
        )
    
    @property
    def get_input_data_paths(self) -> list:
        """Gets a list of h5plexos input files within the scenario folders

        Returns:
            list: list of h5plexos input filenames to process
        """
        if self._get_input_data_paths is None:
            files = []
            for names in self.input_folder.iterdir():
                if names.suffix == ".h5":
                    files.append(names.name)  # Creates a list of only the hdf5 files

            # List of all hf files in hdf5 folder in alpha numeric order
            self._get_input_data_paths = sorted(files, key=lambda x: int(re.sub("\D", "0", x)))
        return self._get_input_data_paths

    @property
    def data_collection(self) -> Dict[str, PLEXOSSolution]:
        """Dictionary model file names to h5 PLEXOSSolution

        Returns:
            dict: data_collection {filename: PLEXOSSolution}
        """
        if self._data_collection is None:
            # Read in all HDF5 files into dictionary
            logger.info("Loading all HDF5 files to prepare for processing")
            regions = set()

            self._data_collection = {}
            for file in self.get_input_data_paths:
                plx_file = PLEXOSSolution(
                    self.input_folder.joinpath(file)
                )
                if not list(plx_file.h5file['data'].keys()):
                    raise MissingH5PLEXOSDataError(file)

                self._data_collection[file] = plx_file
                if not self.Region_Mapping.empty:
                    regions.update(list(self.metadata.regions(file)["region"]))

            if not self.Region_Mapping.empty:
                if regions.issubset(self.Region_Mapping["region"]) is False:
                    missing_regions = list(regions - set(self.Region_Mapping["region"]))
                    logger.warning(
                        "The Following PLEXOS REGIONS are missing from "
                        "the 'region' column of your mapping file: "
                        f"{missing_regions}\n"
                    )
        return self._data_collection

    def output_metadata(self, files_list: list) -> None:
        """Transfers metadata from original PLEXOS solutions file to processed HDF5 file.

        For each partition in a given scenario, the metadata from that partition
        is copied over and saved in the processed output file.

        Args:
            files_list (list): List of all h5 files in hdf5 folder in alpha numeric order
        """
        for partition in files_list:
            f = h5py.File(self.input_folder.joinpath(partition), "r")
            meta_keys = [key for key in f["metadata"].keys()]

            group_dict = {}
            for key in meta_keys:
                sub_dict = {}
                subkeys = [key for key in f["metadata"][key].keys()]
                for sub in subkeys:
                    dset = f["metadata"][key][sub]
                    sub_dict[sub] = dset
                group_dict[key] = sub_dict

            with h5py.File(self.output_file_path, "a") as g:
                # check if metadata group already exists
                existing_groups = [key for key in g.keys()]
                if "metadata" not in existing_groups:
                    grp = g.create_group("metadata")
                else:
                    grp = g["metadata"]

                partition_group = grp.create_group(partition)
                for key in list(group_dict.keys()):
                    subgrp = partition_group.create_group(key)
                    s_dict = group_dict[key]
                    for key2 in list(s_dict.keys()):
                        dset = s_dict[key2]
                        subgrp.create_dataset(name=key2, data=dset)
            f.close()

    def get_processed_data(
        self, prop_class: str, prop: str, timescale: str, model_name: str
    ) -> pd.DataFrame:
        """Handles the pulling of data from the h5plexos hdf5
        file and then passes the data to one of the formating functions

        Args:
            prop_class (str): PLEXOS class e.g Region, Generator, Zone etc
            prop (str): PLEXOS property e.g Max Capacity, Generation etc.
            timescale (str): Data timescale, e.g Hourly, Monthly, 5 minute etc.
            model_name (str): name of model to process.

        Returns:
            pd.DataFrame: Formatted results dataframe.
        """
        db = self.data_collection.get(model_name)
        logger.info(f"      {model_name}")
        try:
            if "_" in prop_class:
                df = db.query_relation_property(
                    prop_class,
                    prop,
                    timescale=timescale,
                    phase=self.plexos_block,
                )
                object_class = prop_class
            else:
                df = db.query_object_property(
                    prop_class,
                    prop,
                    timescale=timescale,
                    phase=self.plexos_block,
                )
                if (0, 6, 0) <= db.version and db.version < (0, 7, 0):
                    object_class = f"{prop_class}s"
                else:
                    object_class = prop_class

        except (ValueError, KeyError):
            raise PropertyNotFound(prop, prop_class)

        if self.plexos_block != "ST" and timescale == "interval":
            df = self.merge_timeseries_block_data(db, df)

        # handles h5plexos naming discrepency
        if (0, 6, 0) <= db.version and db.version < (0, 7, 0):
            # Get original units from h5plexos file
            df_units = (
                db.h5file[
                    f"/data/{self.plexos_block}/{timescale}"
                    f"/{object_class}/{prop}"
                ]
                .attrs["units"]
                .decode("UTF-8")
            )
        else:
            df_units = db.h5file[
                f"/data/{self.plexos_block}/{timescale}"
                f"/{object_class}/{prop}"
            ].attrs["unit"]
        # find unit conversion values
        converted_units = self.UNITS_CONVERSION.get(df_units, (df_units, 1))

        # Get desired method
        process_att = getattr(self, f"df_process_{prop_class}")
        # Process attribute and return to df
        df = process_att(df, model_name)

        # Convert units and add unit column to index
        df = df * converted_units[1]
        units_index = pd.Index([converted_units[0]] * len(df), name="units")
        df.set_index(units_index, append=True, inplace=True)
        df = df.rename(columns={0: "values"})
        if (
            prop_class == "region"
            and prop == "Unserved Energy"
            and int(df.sum(axis=0)) > 0
        ):
            logger.warning(
                f"Scenario contains Unserved Energy: " f"{int(df.sum(axis=0))} MW\n"
            )
        return df

    def merge_timeseries_block_data(
        self, db: PLEXOSSolution, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge chronological time intervals and block data found in LT, MT and PASA results

        Args:
            db (PLEXOSSolution): PLEXOSSolution instance for specific h5plexos file.
            df (pd.DataFrame): h5plexos dataframe

        Returns:
            pd.DataFrame: df with merged in timeseries data
        """

        block_mapping = db.blocks[self.plexos_block]
        block_mapping.index.rename("timestamp", inplace=True)
        df = df.reset_index()

        merged_data = df.merge(block_mapping.reset_index(), on="block")
        merged_data.drop("block", axis=1, inplace=True)
        index_cols = list(merged_data.columns)
        index_cols.remove(0)
        merged_data.set_index(index_cols, inplace=True)
        merged_data = merged_data.sort_index(level=["category", "name"])
        return merged_data

    def df_process_generator(
        self, df: pd.DataFrame, model_name: str
    ) -> pd.DataFrame:
        """Format PLEXOS Generator Class data.

        Args:
            df (pd.DataFrame): h5plexos dataframe to process
            model_name (str): name of h5plexos h5 file being processed

        Returns:
            pd.DataFrame: Processed output, single value column with multiindex.
        """
        df = df.droplevel(level=["band", "property"])
        df.index.rename(["tech", "gen_name"], level=["category", "name"], inplace=True)

        region_gen_cat_meta = self.metadata.region_generator_category(model_name)
        zone_gen_cat_meta = self.metadata.zone_generator_category(model_name)
        timeseries_len = len(df.index.get_level_values("timestamp").unique())

        if region_gen_cat_meta.empty is False:
            region_gen_idx = pd.CategoricalIndex(
                region_gen_cat_meta.index.get_level_values(0)
            )

            region_gen_idx = region_gen_idx.repeat(timeseries_len)

            idx_region = pd.MultiIndex(
                levels=df.index.levels + [region_gen_idx.categories],
                codes=df.index.codes + [region_gen_idx.codes],
                names=df.index.names + region_gen_idx.names,
            )
        else:
            idx_region = df.index

        if zone_gen_cat_meta.empty is False:
            zone_gen_idx = pd.CategoricalIndex(
                zone_gen_cat_meta.index.get_level_values(0)
            )

            zone_gen_idx = zone_gen_idx.repeat(timeseries_len)

            idx_zone = pd.MultiIndex(
                levels=idx_region.levels + [zone_gen_idx.categories],
                codes=idx_region.codes + [zone_gen_idx.codes],
                names=idx_region.names + zone_gen_idx.names,
            )
        else:
            idx_zone = idx_region

        if not self.Region_Mapping.empty:
            region_gen_mapping = (
                region_gen_cat_meta.merge(self.Region_Mapping, how="left", on="region")
                .sort_values(by=["tech", "gen_name"])
                .drop(["region", "tech", "gen_name"], axis=1)
            )
            region_gen_mapping.dropna(axis=1, how="all", inplace=True)

            if not region_gen_mapping.empty:
                region_gen_mapping_idx = pd.MultiIndex.from_frame(region_gen_mapping)
                region_gen_mapping_idx = region_gen_mapping_idx.repeat(timeseries_len)

                idx_map = pd.MultiIndex(
                    levels=idx_zone.levels + region_gen_mapping_idx.levels,
                    codes=idx_zone.codes + region_gen_mapping_idx.codes,
                    names=idx_zone.names + region_gen_mapping_idx.names,
                )
            else:
                idx_map = idx_zone
        else:
            idx_map = idx_zone

        df = pd.DataFrame(data=df.values.reshape(-1), index=idx_map)
        # Gets names of all columns in df and places in list
        df_col = list(df.index.names)
        # move timestamp to start of df
        df_col.insert(0, df_col.pop(df_col.index("timestamp")))
        df = df.reorder_levels(df_col, axis=0)
        df[0] = pd.to_numeric(df[0], downcast="float")

        return df

    def df_process_region(self, df: pd.DataFrame, model_name: str) -> pd.DataFrame:
        """Format PLEXOS Region Class data.

        Args:
            df (pd.DataFrame): h5plexos dataframe to process
            model_name (str): name of h5plexos h5 file being processed

        Returns:
            pd.DataFrame: Processed output, single value column with multiindex.
        """
        df = df.droplevel(level=["band", "property", "category"])
        df.index.rename("region", level="name", inplace=True)

        timeseries_len = len(df.index.get_level_values("timestamp").unique())

        # checks if Region_Mapping contains data to merge, skips if empty
        if not self.Region_Mapping.empty:
            region_gen_mapping = (
                self.metadata.regions(model_name)
                .merge(self.Region_Mapping, how="left", on="region")
                .drop(["region", "category"], axis=1)
            )
            region_gen_mapping.dropna(axis=1, how="all", inplace=True)

            if not region_gen_mapping.empty:
                mapping_idx = pd.MultiIndex.from_frame(region_gen_mapping)
                mapping_idx = mapping_idx.repeat(timeseries_len)

                idx = pd.MultiIndex(
                    levels=df.index.levels + mapping_idx.levels,
                    codes=df.index.codes + mapping_idx.codes,
                    names=df.index.names + mapping_idx.names,
                )
            else:
                idx = df.index
        else:
            idx = df.index

        df = pd.DataFrame(data=df.values.reshape(-1), index=idx)
        df_col = list(df.index.names)
        df_col.insert(0, df_col.pop(df_col.index("timestamp")))
        df = df.reorder_levels(df_col, axis=0)
        df[0] = pd.to_numeric(df[0], downcast="float")
        return df

    def df_process_zone(self, df: pd.DataFrame, model_name: str) -> pd.DataFrame:
        """Format PLEXOS Zone Class data.

        Args:
            df (pd.DataFrame): h5plexos dataframe to process
            model_name (str): name of h5plexos h5 file being processed

        Returns:
            pd.DataFrame: Processed output, single value column with multiindex.
        """
        df = df.droplevel(level=["band", "property", "category"])
        df.index.rename("zone", level="name", inplace=True)
        df = pd.DataFrame(data=df.values.reshape(-1), index=df.index)
        df_col = list(df.index.names)  #
        df_col.insert(0, df_col.pop(df_col.index("timestamp")))
        df = df.reorder_levels(df_col, axis=0)
        df[0] = pd.to_numeric(df[0], downcast="float")
        return df

    def df_process_line(self, df: pd.DataFrame, model_name: str) -> pd.DataFrame:
        """Format PLEXOS Line Class data.

        Args:
            df (pd.DataFrame): h5plexos dataframe to process
            model_name (str): name of h5plexos h5 file being processed

        Returns:
            pd.DataFrame: Processed output, single value column with multiindex.
        """
        df = df.droplevel(level=["band", "property"])
        df.index.rename("line_name", level="name", inplace=True)
        df = pd.DataFrame(data=df.values.reshape(-1), index=df.index)
        df_col = list(df.index.names)
        df_col.insert(0, df_col.pop(df_col.index("timestamp")))
        df = df.reorder_levels(df_col, axis=0)
        df[0] = pd.to_numeric(df[0], downcast="float")
        return df

    def df_process_interface(
        self, df: pd.DataFrame, model_name: str
    ) -> pd.DataFrame:
        """Format PLEXOS PLEXOS Interface Class data.

        Args:
            df (pd.DataFrame): h5plexos dataframe to process
            model_name (str): name of h5plexos h5 file being processed

        Returns:
            pd.DataFrame: Processed output, single value column with multiindex.
        """
        df = df.droplevel(level=["band", "property"])
        df.index.rename(
            ["interface_name", "interface_category"],
            level=["name", "category"],
            inplace=True,
        )
        df = pd.DataFrame(data=df.values.reshape(-1), index=df.index)
        df_col = list(df.index.names)
        df_col.insert(0, df_col.pop(df_col.index("timestamp")))
        df = df.reorder_levels(df_col, axis=0)
        df[0] = pd.to_numeric(df[0], downcast="float")
        return df

    def df_process_reserve(self, df: pd.DataFrame, model_name: str) -> pd.DataFrame:
        """Format PLEXOS Reserve Class data.

        Args:
            df (pd.DataFrame): h5plexos dataframe to process
            model_name (str): name of h5plexos h5 file being processed

        Returns:
            pd.DataFrame: Processed output, single value column with multiindex.
        """
        df = df.droplevel(level=["band", "property"])
        df.index.rename(["parent", "Type"], level=["name", "category"], inplace=True)
        df = df.reset_index()  # unzip the levels in index
        if self.metadata.reserves_regions(model_name).empty is False:
            # Merges in regions where reserves are located
            df = df.merge(
                self.metadata.reserves_regions(model_name), how="left", on="parent"
            )

        if self.metadata.reserves_zones(model_name).empty is False:
            # Merges in zones where reserves are located
            df = df.merge(
                self.metadata.reserves_zones(model_name), how="left", on="parent"
            )
        df_col = list(df.columns)
        df_col.remove(0)
        # move timestamp to start of df
        df_col.insert(0, df_col.pop(df_col.index("timestamp")))
        df.set_index(df_col, inplace=True)
        df[0] = pd.to_numeric(df[0], downcast="float")
        return df

    def df_process_reserves_generators(
        self, df: pd.DataFrame, model_name: str
    ) -> pd.DataFrame:
        """Format PLEXOS Reserve_Generators Relational Class data.

        Args:
            df (pd.DataFrame): h5plexos dataframe to process
            model_name (str): name of h5plexos h5 file being processed

        Returns:
            pd.DataFrame: Processed output, single value column with multiindex.
        """
        df = df.droplevel(level=["band", "property"])
        df.index.rename(["gen_name"], level=["child"], inplace=True)
        df = df.reset_index()  # unzip the levels in index
        df = df.merge(
            self.metadata.generator_category(model_name), how="left", on="gen_name"
        )

        # merging in generator region/zones first prevents double
        # counting in cases where multiple model regions are within a reserve region
        if self.metadata.region_generators(model_name).empty is False:
            df = df.merge(
                self.metadata.region_generators(model_name),
                how="left",
                on="gen_name",
            )
        if self.metadata.zone_generators(model_name).empty is False:
            df = df.merge(
                self.metadata.zone_generators(model_name), how="left", on="gen_name"
            )

        # now merge in reserve regions/zones
        if self.metadata.reserves_regions(model_name).empty is False:
            # Merges in regions where reserves are located
            df = df.merge(
                self.metadata.reserves_regions(model_name),
                how="left",
                on=["parent", "region"],
            )
        if self.metadata.reserves_zones(model_name).empty is False:
            # Merges in zones where reserves are located
            df = df.merge(
                self.metadata.reserves_zones(model_name),
                how="left",
                on=["parent", "zone"],
            )

        df_col = list(df.columns)
        df_col.remove(0)
        df_col.insert(0, df_col.pop(df_col.index("timestamp")))
        df.set_index(df_col, inplace=True)
        df[0] = pd.to_numeric(df[0], downcast="float")
        return df

    def df_process_fuel(self, df: pd.DataFrame, model_name: str) -> pd.DataFrame:
        """Format PLEXOS Fuel Class data.

        Args:
            df (pd.DataFrame): h5plexos dataframe to process
            model_name (str): name of h5plexos h5 file being processed

        Returns:
            pd.DataFrame: Processed output, single value column with multiindex.
        """
        df = df.droplevel(level=["band", "property", "category"])
        df.index.rename("fuel_type", level="name", inplace=True)
        df = pd.DataFrame(data=df.values.reshape(-1), index=df.index)
        df_col = list(df.index.names)
        df_col.insert(0, df_col.pop(df_col.index("timestamp")))
        df = df.reorder_levels(df_col, axis=0)
        df[0] = pd.to_numeric(df[0], downcast="float")
        return df

    def df_process_constraint(
        self, df: pd.DataFrame, model_name: str
    ) -> pd.DataFrame:
        """Format PLEXOS Constraint Class data.

        Args:
            df (pd.DataFrame): h5plexos dataframe to process
            model_name (str): name of h5plexos h5 file being processed

        Returns:
            pd.DataFrame: Processed output, single value column with multiindex.
        """
        df = df.droplevel(level=["band", "property"])
        df.index.rename(
            ["constraint_category", "constraint"],
            level=["category", "name"],
            inplace=True,
        )
        df = pd.DataFrame(data=df.values.reshape(-1), index=df.index)
        df_col = list(df.index.names)
        df_col.insert(0, df_col.pop(df_col.index("timestamp")))
        df = df.reorder_levels(df_col, axis=0)
        df[0] = pd.to_numeric(df[0], downcast="float")
        return df

    def df_process_emission(
        self, df: pd.DataFrame, model_name: str
    ) -> pd.DataFrame:
        """Format PLEXOS Emission Class data.

        Args:
            df (pd.DataFrame): h5plexos dataframe to process
            model_name (str): name of h5plexos h5 file being processed

        Returns:
            pd.DataFrame: Processed output, single value column with multiindex.
        """
        df = df.droplevel(level=["band", "property"])
        df.index.rename("emission_type", level="name", inplace=True)
        df = pd.DataFrame(data=df.values.reshape(-1), index=df.index)
        df_col = list(df.index.names)
        df_col.insert(0, df_col.pop(df_col.index("timestamp")))
        df = df.reorder_levels(df_col, axis=0)
        df[0] = pd.to_numeric(df[0], downcast="float")
        return df

    def df_process_emissions_generators(
        self, df: pd.DataFrame, model_name: str
    ) -> pd.DataFrame:
        """Format PLEXOS Emissions_Generators Relational Class data.

        Args:
            df (pd.DataFrame): h5plexos dataframe to process
            model_name (str): name of h5plexos h5 file being processed

        Returns:
            pd.DataFrame: Processed output, single value column with multiindex.
        """
        df = df.droplevel(level=["band", "property"])
        df.index.rename(["gen_name"], level=["child"], inplace=True)
        df.index.rename(["pollutant"], level=["parent"], inplace=True)

        df = df.reset_index()  # unzip the levels in index
        # merge in tech information
        df = df.merge(
            self.metadata.generator_category(model_name), how="left", on="gen_name"
        )
        # merge in region and zone information
        if self.metadata.region_generator_category(model_name).empty is False:
            # merge in region information
            df = df.merge(
                self.metadata.region_generator_category(model_name).reset_index(),
                how="left",
                on=["gen_name", "tech"],
            )

        if self.metadata.zone_generator_category(model_name).empty is False:
            # Merges in zones where reserves are located
            df = df.merge(
                self.metadata.zone_generator_category(model_name).reset_index(),
                how="left",
                on=["gen_name", "tech"],
            )

        if not self.Region_Mapping.empty:
            df = df.merge(self.Region_Mapping, how="left", on="region")
            df.dropna(axis=1, how="all", inplace=True)

        if not self.emit_names.empty:
            # reclassify emissions as specified by user in mapping
            df["pollutant"] = pd.Categorical(
                df["pollutant"].map(lambda x: self.emit_names_dict.get(x, x))
            )

        # remove categoricals (otherwise h5 save will fail)
        df = df.astype({"tech": "object", "pollutant": "object"})

        # Checks if all emissions categories have been identified and matched.
        # If not, lists categories that need a match
        if not self.emit_names.empty:
            if (
                self.emit_names_dict != {}
                and (
                    set(df["pollutant"].unique()).issubset(
                        self.emit_names["New"].unique()
                    )
                )
                is False
            ):
                missing_emit_cat = list(
                    (set(df["pollutant"].unique()))
                    - (set(self.emit_names["New"].unique()))
                )
                logger.warning(
                    "The following emission objects do not have a "
                    f"correct category mapping: {missing_emit_cat}\n"
                )

        df_col = list(df.columns)
        df_col.remove(0)
        df_col.insert(0, df_col.pop(df_col.index("timestamp")))
        df.set_index(df_col, inplace=True)
        # downcast values to save on memory
        df[0] = pd.to_numeric(df[0].values, downcast="float")
        # convert to range index (otherwise h5 save will fail)
        df.columns = pd.RangeIndex(0, 1, step=1)
        return df

    def df_process_storage(self, df: pd.DataFrame, model_name: str) -> pd.DataFrame:
        """Format PLEXOS Storage Class data.

        Args:
            df (pd.DataFrame): h5plexos dataframe to process
            model_name (str): name of h5plexos h5 file being processed

        Returns:
            pd.DataFrame: Processed output, single value column with multiindex.
        """
        df = df.droplevel(level=["band", "property", "category"])
        df = df.reset_index()  # unzip the levels in index
        df = df.merge(
            self.metadata.generator_storage(model_name), how="left", on="name"
        )
        if self.metadata.region_generators(model_name).empty is False:
            # Merges in regions where generators are located
            df = df.merge(
                self.metadata.region_generators(model_name),
                how="left",
                on="gen_name",
            )
        if self.metadata.zone_generators(model_name).empty is False:
            # Merges in zones where generators are located
            df = df.merge(
                self.metadata.zone_generators(model_name), how="left", on="gen_name"
            )
        # checks if Region_Maping contains data to merge, skips if empty (Default)
        if not self.Region_Mapping.empty:
            # Merges in all Region Mappings
            df = df.merge(self.Region_Mapping, how="left", on="region")
            df.dropna(axis=1, how="all", inplace=True)

        df.rename(columns={"name": "storage_resource"}, inplace=True)
        df_col = list(df.columns)
        df_col.remove(0)
        df_col.insert(0, df_col.pop(df_col.index("timestamp")))
        df.set_index(df_col, inplace=True)
        df[0] = pd.to_numeric(df[0], downcast="float")
        return df

    def df_process_region_regions(
        self, df: pd.DataFrame, model_name: str
    ) -> pd.DataFrame:
        """Format PLEXOS Region_Regions Relational Class data.

        Args:
            df (pd.DataFrame): h5plexos dataframe to process
            model_name (str): name of h5plexos h5 file being processed

        Returns:
            pd.DataFrame: Processed output, single value column with multiindex.
        """
        df = df.droplevel(level=["band", "property"])
        df = pd.DataFrame(data=df.values.reshape(-1), index=df.index)
        df_col = list(df.index.names)
        df_col.insert(0, df_col.pop(df_col.index("timestamp")))
        df = df.reorder_levels(df_col, axis=0)
        df[0] = pd.to_numeric(df[0], downcast="float")
        return df

    def df_process_node(self, df: pd.DataFrame, model_name: str) -> pd.DataFrame:
        """Format PLEXOS Node Class data.

        Args:
            df (pd.DataFrame): h5plexos dataframe to process
            model_name (str): name of h5plexos h5 file being processed

        Returns:
            pd.DataFrame: Processed output, single value column with multiindex.
        """
        df = df.droplevel(level=["band", "property", "category"])
        df.index.rename("node", level="name", inplace=True)
        df.sort_index(level=["node"], inplace=True)

        node_region_meta = self.metadata.node_region(model_name)
        node_zone_meta = self.metadata.node_zone(model_name)
        timeseries_len = len(df.index.get_level_values("timestamp").unique())

        if node_region_meta.empty is False:
            node_region_idx = pd.CategoricalIndex(
                node_region_meta.index.get_level_values(0)
            )

            node_region_idx = node_region_idx.repeat(timeseries_len)

            idx_region = pd.MultiIndex(
                levels=df.index.levels + [node_region_idx.categories],
                codes=df.index.codes + [node_region_idx.codes],
                names=df.index.names + node_region_idx.names,
            )
        else:
            idx_region = df.index

        if node_zone_meta.empty is False:
            node_zone_idx = pd.CategoricalIndex(
                node_zone_meta.index.get_level_values(0)
            )

            node_zone_idx = node_zone_idx.repeat(timeseries_len)

            idx_zone = pd.MultiIndex(
                levels=idx_region.levels + [node_zone_idx.categories],
                codes=idx_region.codes + [node_zone_idx.codes],
                names=idx_region.names + node_zone_idx.names,
            )
        else:
            idx_zone = idx_region

        if not self.Region_Mapping.empty:
            region_mapping = node_region_meta.merge(
                self.Region_Mapping, how="left", on="region"
            ).drop(["region", "node"], axis=1)
            region_mapping.dropna(axis=1, how="all", inplace=True)

            if not region_mapping.empty:
                region_mapping_idx = pd.MultiIndex.from_frame(region_mapping)
                region_mapping_idx = region_mapping_idx.repeat(timeseries_len)

                idx_map = pd.MultiIndex(
                    levels=idx_zone.levels + region_mapping_idx.levels,
                    codes=idx_zone.codes + region_mapping_idx.codes,
                    names=idx_zone.names + region_mapping_idx.names,
                )
            else:
                idx_map = idx_zone
        else:
            idx_map = idx_zone

        df = pd.DataFrame(data=df.values.reshape(-1), index=idx_map)
        df_col = list(df.index.names)
        df_col.insert(0, df_col.pop(df_col.index("timestamp")))
        df = df.reorder_levels(df_col, axis=0)
        df[0] = pd.to_numeric(df[0], downcast="float")
        return df

    def df_process_abatement(
        self, df: pd.DataFrame, model_name: str
    ) -> pd.DataFrame:
        """Format PLEXOS Abatement Class data.

        Args:
            df (pd.DataFrame): h5plexos dataframe to process
            model_name (str): name of h5plexos h5 file being processed

        Returns:
            pd.DataFrame: Processed output, single value column with multiindex.
        """
        df = df.droplevel(level=["band", "property"])
        df.index.rename("abatement_name", level="name", inplace=True)
        df = pd.DataFrame(data=df.values.reshape(-1), index=df.index)
        df_col = list(df.index.names)
        df_col.insert(0, df_col.pop(df_col.index("timestamp")))
        df = df.reorder_levels(df_col, axis=0)
        df[0] = pd.to_numeric(df[0], downcast="float")
        return df

    def df_process_batterie(
        self, df: pd.DataFrame, model_name: str
    ) -> pd.DataFrame:
        """
        Method for formatting data which comes form the PLEXOS Batteries Class

        Returns
        -------
        df : pd.DataFrame
            Processed Output, single value column with multiindex.

        """
        df = self.df.droplevel(level=["band", "property"])
        df.index.rename("battery_name", level="name", inplace=True)
        df = pd.DataFrame(data=df.values.reshape(-1), index=df.index)
        df_col = list(
            df.index.names
        )  # Gets names of all columns in df and places in list
        df_col.insert(
            0, df_col.pop(df_col.index("timestamp"))
        )  # move timestamp to start of df
        df = df.reorder_levels(df_col, axis=0)
        df[0] = pd.to_numeric(df[0], downcast="float")
        return df

    def df_process_waterway(
        self, df: pd.DataFrame, model_name: str
    ) -> pd.DataFrame:
        """Format PLEXOS Waterway Class data.

        Args:
            df (pd.DataFrame): h5plexos dataframe to process
            model_name (str): name of h5plexos h5 file being processed

        Returns:
            pd.DataFrame: Processed output, single value column with multiindex.
        """
        df = df.droplevel(level=["band", "property"])
        df.index.rename("waterway_name", level="name", inplace=True)
        df = pd.DataFrame(data=df.values.reshape(-1), index=df.index)
        df_col = list(df.index.names)
        df_col.insert(0, df_col.pop(df_col.index("timestamp")))
        df = df.reorder_levels(df_col, axis=0)
        df[0] = pd.to_numeric(df[0], downcast="float")
        return df
