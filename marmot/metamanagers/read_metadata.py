# -*- coding: utf-8 -*-
"""Retrieve metadata from modelling results.

Database can be either a h5plexos file or a formatted Marmot hdf5 file.

@author: Ryan Houseman
"""

import sys
import h5py
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger("formatter." + __name__)

class MetaData:
    """Handle the retrieval of metadata from the formatted or original solution h5 files.
    """

    filename: str = None
    """The name of the h5 file to retrieve data from."""
    h5_filepath: Path = None
    """The path to the h5 file"""
    h5_data: h5py.File = None
    """h5 file loaded in memory."""

    def __init__(
        self,
        HDF5_folder_in: Path,
        read_from_formatted_h5: bool = True,
        Region_Mapping: pd.DataFrame = pd.DataFrame(),
        partition_number: int = 0,
    ):
        """
        Args:
            HDF5_folder_in (Path): Folder containing h5 file.
            read_from_formatted_h5 (bool, optional): Boolean for whether the metadata is
                being read from the formatted hdf5 file or the original PLEXOS solution file.
                Defaults to True.
            Region_Mapping (pd.DataFrame, optional): DataFrame of extra regions to map.
                Defaults to pd.DataFrame().
            partition_number (int, optional): Which temporal partition of h5 data to retrieve
                metadata from in the formatted h5 file. Defaults to 0.
        """
        self.HDF5_folder_in = Path(HDF5_folder_in)
        self.Region_Mapping = Region_Mapping
        self.read_from_formatted_h5 = read_from_formatted_h5
        self.partition_number = partition_number
        self.start_index = None

    def _check_if_existing_filename(self, filename: str) -> bool:
        """Check if the passed filename is the same or different from previous calls.

        If file is different replaces the filename with new value
        and closes old file

        Args:
            filename (str): The name of the h5 file to retreive data from.

        Returns:
            bool: False if new file, True if existing
        """
        if self.filename != filename:
            self.filename = filename
            self.close_h5()
            return False
        elif self.filename == filename:
            return True

    @classmethod
    def close_h5(cls) -> None:
        """Closes h5 file open in memory."""
        if cls.h5_data:
            cls.h5_data.close()

    def _read_data(self, filename: str) -> None:
        """Reads h5 file into memory.

        Args:
            filename (str): The name of the h5 file to retreive
                data from.
        """
        logger.debug(f"Reading New h5 file: {filename}")
        processed_file_format = "{}_formatted.h5"

        try:
            if self.read_from_formatted_h5:
                
                if "_formatted.h5" not in filename:
                    filename = processed_file_format.format(filename)
                self.h5_filepath = self.HDF5_folder_in.joinpath(filename)
                with h5py.File(self.HDF5_folder_in.joinpath(filename), "r") as f:
                    self.h5_data = f
                    partitions = [key for key in self.h5_data["metadata"].keys()]
                if self.partition_number > len(partitions):
                    logger.warning(
                        "\nYou have chosen to use metadata partition_number "
                        f"{self.partition_number}, But there are only {len(partitions)} "
                        "partitions in your formatted h5 file.\n"
                        "Defaulting to partition_number 0"
                    )
                    self.partition_number = 0

                self.start_index = f"metadata/{partitions[self.partition_number]}/"
            else:
                self.h5_filepath = self.HDF5_folder_in.joinpath(filename)
                with h5py.File(self.HDF5_folder_in.joinpath(filename), "r") as f:
                    self.h5_data = f
                self.start_index = "metadata/"

        except OSError:
            if self.read_from_formatted_h5:
                logger.warning(
                    "Unable to find processed HDF5 file to retrieve metadata.\n"
                    "Check scenario name."
                )
                return
            else:
                logger.info(
                    "\nIn order to initialize your database's metadata, "
                    "Marmot is looking for a h5plexos solution file.\n"
                    f"It is looking in {self.HDF5_folder_in}, but it cannot "
                    "find any *.h5 files there.\n"
                    "Please check the 'Model_Solutions_folder' input in your "
                    "'Marmot_user_defined_inputs.csv'.\n"
                    "Ensure that it matches the filepath containing the *.h5 files "
                    "created by h5plexos.\n\nMarmot will now quit."
                )
                sys.exit()

    def generator_category(self, filename: str) -> pd.DataFrame:
        """Generator categories mapping.

        Args:
            filename (str): The name of the h5 file to retreive data from.
                If retreiving from fromatted h5 file, just pass scenario name.
        """
        if not self._check_if_existing_filename(filename):
            self._read_data(filename)

        try:
            try:
                gen_category = pd.read_hdf(
                    self.h5_filepath, key=f"{self.start_index}/objects/generator"
                )
            except KeyError:
                gen_category = pd.read_hdf(
                    self.h5_filepath, key=f"{self.start_index}/objects/generators"
                )
            gen_category.rename(
                columns={"name": "gen_name", "category": "tech"}, inplace=True
            )
            gen_category = gen_category.applymap(
                lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
            )
        except KeyError:
            gen_category = pd.DataFrame()

        return gen_category

    def region_generators(self, filename: str) -> pd.DataFrame:
        """Region generators mapping.

        Args:
            filename (str): The name of the h5 file to retreive data from.
                If retreiving from fromatted h5 file, just pass scenario name.
        """
        if not self._check_if_existing_filename(filename):
            self._read_data(filename)

        try:
            try:
                region_gen = pd.read_hdf(
                    self.h5_filepath,
                    key=f"{self.start_index}/relations/regions_generators",
                )
            except KeyError:
                region_gen = pd.read_hdf(
                    self.h5_filepath,
                    key=f"{self.start_index}/relations/region_generators",
                )
            region_gen.rename(
                columns={"child": "gen_name", "parent": "region"}, inplace=True
            )
            region_gen = region_gen.applymap(
                lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
            )
            region_gen.drop_duplicates(
                subset=["gen_name"], keep="first", inplace=True
            )  # For generators which belong to more than 1 region, drop duplicates.
        except KeyError:
            region_gen = pd.DataFrame()

        return region_gen

    def region_generator_category(self, filename: str) -> pd.DataFrame:
        """Region generators category mapping.

        Args:
            filename (str): The name of the h5 file to retreive data from.
                If retreiving from fromatted h5 file, just pass scenario name.
        """
        try:
            region_gen = self.region_generators(filename)
            gen_category = self.generator_category(filename)
            region_gen_cat = (
                region_gen.merge(gen_category, how="left", on="gen_name")
                .sort_values(by=["tech", "gen_name"])
                .set_index("region")
            )
        except KeyError:
            region_gen_cat = pd.DataFrame()

        return region_gen_cat

    def zone_generators(self, filename: str) -> pd.DataFrame:
        """Zone generators mapping.

        Args:
            filename (str): The name of the h5 file to retreive data from.
                If retreiving from fromatted h5 file, just pass scenario name.
        """
        if not self._check_if_existing_filename(filename):
            self._read_data(filename)
        try:
            try:
                zone_gen = pd.read_hdf(
                    self.h5_filepath,
                    key=f"{self.start_index}/relations/zones_generators",
                )
            except KeyError:
                zone_gen = pd.read_hdf(
                    self.h5_filepath,
                    key=f"{self.start_index}/relations/zone_generators",
                )
            zone_gen.rename(
                columns={"child": "gen_name", "parent": "zone"}, inplace=True
            )
            zone_gen = zone_gen.applymap(
                lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
            )
            zone_gen.drop_duplicates(
                subset=["gen_name"], keep="first", inplace=True
            )  # For generators which belong to more than 1 region, drop duplicates.
        except KeyError:
            zone_gen = pd.DataFrame()

        return zone_gen

    def zone_generator_category(self, filename: str) -> pd.DataFrame:
        """Zone generators category mapping.

        Args:
            filename (str): The name of the h5 file to retreive data from.
                If retreiving from fromatted h5 file, just pass scenario name.
        """
        try:
            zone_gen = self.zone_generators(filename)
            gen_category = self.generator_category(filename)
            zone_gen_cat = (
                zone_gen.merge(gen_category, how="left", on="gen_name")
                .sort_values(by=["tech", "gen_name"])
                .set_index("zone")
            )
        except KeyError:
            zone_gen_cat = pd.DataFrame()

        return zone_gen_cat

    def region_batteries(self, filename: str) -> pd.DataFrame:
        """Region batteries mapping.

        Args:
            filename (str): The name of the h5 file to retreive data from.
                If retreiving from fromatted h5 file, just pass scenario name.
        """
        if not self._check_if_existing_filename(filename):
            self._read_data(filename)

        try:
            region_batt = pd.read_hdf(
                self.h5_filepath,
                key=f"{self.start_index}/relations/regions_batteries",
            )
            region_batt.rename(
                columns={"child": "battery_name", "parent": "region"}, inplace=True
            )
            region_batt = region_batt.applymap(
                lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
            )
            region_batt.drop_duplicates(
                subset=["battery_name"], keep="first", inplace=True
            )  # For batteries which belong to more than 1 region, drop duplicates.

            #Merge in region mapping.
            if not self.Region_Mapping.empty:
                region_batt = pd.merge(
                    region_batt,
                    self.Region_Mapping,
                    how="left",
                    on="region",
                )
                region_batt.dropna(axis=1, how="all", inplace=True)

        except KeyError:
            region_batt = pd.DataFrame()

        return region_batt

    # Generator storage has been updated so that only one of 
    # tail_storage & head_storage is required
    # If both are available, both are used
    def generator_storage(self, filename: str) -> pd.DataFrame:
        """Generator Storage mapping.

        Args:
            filename (str): The name of the h5 file to retreive data from.
                If retreiving from fromatted h5 file, just pass scenario name.
        """
        if not self._check_if_existing_filename(filename):
            self._read_data(filename)
        head_tail = [0, 0]
        try:
            generator_headstorage = pd.DataFrame()
            generator_tailstorage = pd.DataFrame()
            try:
                generator_headstorage = pd.read_hdf(
                    self.h5_filepath,
                    key=f"{self.start_index}/relations/generators_headstorage",
                )
                head_tail[0] = 1
            except KeyError:
                pass
            try:
                generator_headstorage = pd.read_hdf(
                    self.h5_filepath,
                    key=f"{self.start_index}/relations/generator_headstorage",
                )
                head_tail[0] = 1
            except KeyError:
                pass
            try:
                generator_headstorage = pd.read_hdf(
                    self.h5_filepath,
                    key=f"{self.start_index}/relations/exportinggenerators_headstorage",
                )
                head_tail[0] = 1
            except KeyError:
                pass
            try:
                generator_tailstorage = pd.read_hdf(
                    self.h5_filepath,
                    key=f"{self.start_index}/relations/generators_tailstorage",
                )
                head_tail[1] = 1
            except KeyError:
                pass
            try:
                generator_tailstorage = pd.read_hdf(
                    self.h5_filepath,
                    key=f"{self.start_index}/relations/generator_tailstorage",
                )
                head_tail[1] = 1
            except KeyError:
                pass
            try:
                generator_tailstorage = pd.read_hdf(
                    self.h5_filepath,
                    key=f"{self.start_index}/relations/importinggenerators_tailstorage",
                )
                head_tail[1] = 1
            except KeyError:
                pass
            if head_tail[0] == 1:
                if head_tail[1] == 1:
                    gen_storage = pd.concat(
                        [generator_headstorage, generator_tailstorage]
                    )
                else:
                    gen_storage = generator_headstorage
            else:
                gen_storage = generator_tailstorage
            gen_storage.rename(
                columns={"child": "name", "parent": "gen_name"}, inplace=True
            )
            gen_storage = gen_storage.applymap(
                lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
            )
        except:
            gen_storage = pd.DataFrame()

        return gen_storage

    def node_region(self, filename: str) -> pd.DataFrame:
        """Node Region mapping.

        Args:
            filename (str): The name of the h5 file to retreive data from.
                If retreiving from fromatted h5 file, just pass scenario name.
        """
        if not self._check_if_existing_filename(filename):
            self._read_data(filename)
        try:
            try:
                node_region = pd.read_hdf(
                    self.h5_filepath, key=f"{self.start_index}/relations/nodes_region"
                )
            except KeyError:
                node_region = pd.read_hdf(
                    self.h5_filepath, key=f"{self.start_index}/relations/node_region"
                )
            node_region.rename(
                columns={"child": "region", "parent": "node"}, inplace=True
            )
            node_region = node_region.applymap(
                lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
            )
            node_region = node_region.sort_values(by=["node"]).set_index("region")
        except:
            node_region = pd.DataFrame()

        return node_region

    def node_zone(self, filename: str) -> pd.DataFrame:
        """Node zone mapping.

        Args:
            filename (str): The name of the h5 file to retreive data from.
                If retreiving from fromatted h5 file, just pass scenario name.
        """
        if not self._check_if_existing_filename(filename):
            self._read_data(filename)
        try:
            try:
                node_zone = pd.read_hdf(
                    self.h5_filepath, key=f"{self.start_index}/relations/nodes_zone"
                )
            except KeyError:
                node_zone = pd.read_hdf(
                    self.h5_filepath, key=f"{self.start_index}/relations/node_zone"
                )
            node_zone.rename(columns={"child": "zone", "parent": "node"}, inplace=True)
            node_zone = node_zone.applymap(
                lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
            )
            node_zone = node_zone.sort_values(by=["node"]).set_index("zone")
        except:
            node_zone = pd.DataFrame()

        return node_zone

    def generator_node(self, filename: str) -> pd.DataFrame:
        """generator node mapping.

        Args:
            filename (str): The name of the h5 file to retreive data from.
                If retreiving from fromatted h5 file, just pass scenario name.
        """
        if not self._check_if_existing_filename(filename):
            self._read_data(filename)
        try:
            try:
                generator_node = pd.read_hdf(
                    self.h5_filepath,
                    key=f"{self.start_index}/relations/generators_nodes",
                )
            except KeyError:
                generator_node = pd.read_hdf(
                    self.h5_filepath,
                    key=f"{self.start_index}/relations/generator_nodes",
                )
            generator_node.rename(
                columns={"child": "node", "parent": "gen_name"}, inplace=True
            )
            generator_node = generator_node.applymap(
                lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
            )
            # generators_nodes = generators_nodes.sort_values(by=['generator'
        except:
            generator_node = pd.DataFrame()

        return generator_node

    def regions(self, filename: str) -> pd.DataFrame:
        """Region objects.

        Args:
            filename (str): The name of the h5 file to retreive data from.
                If retreiving from fromatted h5 file, just pass scenario name.
        """
        if not self._check_if_existing_filename(filename):
            self._read_data(filename)

        try:
            try:
                regions = pd.read_hdf(
                    self.h5_filepath, key=f"{self.start_index}/objects/regions"
                )
            except KeyError:
                regions = pd.read_hdf(
                    self.h5_filepath, key=f"{self.start_index}/objects/region"
                )
            regions = regions.applymap(
                lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
            )
            regions.rename(columns={"name": "region"}, inplace=True)
            regions.sort_values(["category", "region"], inplace=True)
        except KeyError:
            logger.warning("Regional data not included in h5plexos results")
            regions = pd.DataFrame()

        return regions

    def zones(self, filename: str) -> pd.DataFrame:
        """Zone objects.

        Args:
            filename (str): The name of the h5 file to retreive data from.
                If retreiving from fromatted h5 file, just pass scenario name.
        """
        if not self._check_if_existing_filename(filename):
            self._read_data(filename)
        try:
            try:
                zones = pd.read_hdf(
                    self.h5_filepath, key=f"{self.start_index}/objects/zones"
                )
            except KeyError:
                zones = pd.read_hdf(
                    self.h5_filepath, key=f"{self.start_index}/objects/zone"
                )
            zones = zones.applymap(
                lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
            )
        except KeyError:
            logger.warning("Zonal data not included in h5plexos results")
            zones = pd.DataFrame()

        return zones

    def lines(self, filename: str) -> pd.DataFrame:
        """Line objects.

        Args:
            filename (str): The name of the h5 file to retreive data from.
                If retreiving from fromatted h5 file, just pass scenario name.
        """
        if not self._check_if_existing_filename(filename):
            self._read_data(filename)
        try:
            try:
                lines = pd.read_hdf(
                    self.h5_filepath, key=f"{self.start_index}/objects/lines"
                )
            except KeyError:
                lines = pd.read_hdf(
                    self.h5_filepath, key=f"{self.start_index}/objects/line"
                )
            lines = lines.applymap(
                lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
            )
            lines.rename(columns={"name": "line_name"}, inplace=True)
        except KeyError:
            logger.warning("Line data not included in h5plexos results")

        return lines

    def region_regions(self, filename: str) -> pd.DataFrame:
        """Region-region mapping.

        Args:
            filename (str): The name of the h5 file to retreive data from.
                If retreiving from fromatted h5 file, just pass scenario name.
        """
        if not self._check_if_existing_filename(filename):
            self._read_data(filename)
        try:
            region_regions = pd.read_hdf(
                self.h5_filepath, key=f"{self.start_index}/relations/region_regions"
            )
            region_regions = region_regions.applymap(
                lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
            )
        except KeyError:
            logger.warning("region_regions data not included in h5plexos results")

        return region_regions

    def region_interregionallines(self, filename: str) -> pd.DataFrame:
        """Region inter-regional lines mapping.

        Args:
            filename (str): The name of the h5 file to retreive data from.
                If retreiving from fromatted h5 file, just pass scenario name.
        """
        if not self._check_if_existing_filename(filename):
            self._read_data(filename)
        try:
            try:
                region_interregionallines = pd.read_hdf(
                    self.h5_filepath,
                    key=f"{self.start_index}/relations/region_interregionallines",
                )
            except KeyError:
                region_interregionallines = pd.read_hdf(
                    self.h5_filepath,
                    key=f"{self.start_index}/relations/region_interregionalline",
                )

            region_interregionallines = region_interregionallines.applymap(
                lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
            )
            region_interregionallines.rename(
                columns={"parent": "region", "child": "line_name"}, inplace=True
            )
            if not self.Region_Mapping.empty:
                region_interregionallines = pd.merge(
                    region_interregionallines,
                    self.Region_Mapping,
                    how="left",
                    on="region",
                )
                region_interregionallines.dropna(axis=1, how="all", inplace=True)
        except KeyError:
            region_interregionallines = pd.DataFrame()
            logger.warning(
                "Region Interregionallines data not included in h5plexos results"
            )

        return region_interregionallines

    def region_intraregionallines(self, filename: str) -> pd.DataFrame:
        """Region intra-regional lines mapping.

        Args:
            filename (str): The name of the h5 file to retreive data from.
                If retreiving from fromatted h5 file, just pass scenario name.
        """
        if not self._check_if_existing_filename(filename):
            self._read_data(filename)
        try:
            try:
                region_intraregionallines = pd.read_hdf(
                    self.h5_filepath,
                    key=f"{self.start_index}/relations/region_intraregionallines",
                )
            except KeyError:
                try:
                    region_intraregionallines = pd.read_hdf(
                        self.h5_filepath,
                        key=f"{self.start_index}/relations/region_intraregionalline",
                    )
                except KeyError:
                    region_intraregionallines = pd.concat(
                        [
                            pd.read_hdf(
                                self.h5_filepath,
                                key=f"{self.start_index}/relations/region_importinglines",
                            ),
                            pd.read_hdf(
                                self.h5_filepath,
                                key=f"{self.start_index}/relations/region_exportinglines",
                            ),
                        ]
                    ).drop_duplicates()
            region_intraregionallines = region_intraregionallines.applymap(
                lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
            )
            region_intraregionallines.rename(
                columns={"parent": "region", "child": "line_name"}, inplace=True
            )
            if not self.Region_Mapping.empty:
                region_intraregionallines = pd.merge(
                    region_intraregionallines,
                    self.Region_Mapping,
                    how="left",
                    on="region",
                )
                region_intraregionallines.dropna(axis=1, how="all", inplace=True)
        except KeyError:
            region_intraregionallines = pd.DataFrame()
            logger.warning(
                "Region Intraregionallines Lines data not included in h5plexos results"
            )

        return region_intraregionallines

    def region_exporting_lines(self, filename: str) -> pd.DataFrame:
        """Region exporting lines mapping.

        Args:
            filename (str): The name of the h5 file to retreive data from.
                If retreiving from fromatted h5 file, just pass scenario name.
        """
        if not self._check_if_existing_filename(filename):
            self._read_data(filename)
        try:
            try:
                region_exportinglines = pd.read_hdf(
                    self.h5_filepath,
                    key=f"{self.start_index}/relations/region_exportinglines",
                )
            except KeyError:
                region_exportinglines = pd.read_hdf(
                    self.h5_filepath,
                    key=f"{self.start_index}/relations/region_exportingline",
                )
            region_exportinglines = region_exportinglines.applymap(
                lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
            )
            region_exportinglines = region_exportinglines.rename(
                columns={"parent": "region", "child": "line_name"}
            )
            if not self.Region_Mapping.empty:
                region_exportinglines = pd.merge(
                    region_exportinglines, self.Region_Mapping, how="left", on="region"
                )
                region_exportinglines.dropna(axis=1, how="all", inplace=True)
        except KeyError:
            logger.warning(
                "Region Exporting Lines data not included in h5plexos results"
            )

        return region_exportinglines

    def region_importing_lines(self, filename: str) -> pd.DataFrame:
        """Region importing lines mapping.

        Args:
            filename (str): The name of the h5 file to retreive data from.
                If retreiving from fromatted h5 file, just pass scenario name.
        """
        if not self._check_if_existing_filename(filename):
            self._read_data(filename)
        try:
            try:
                region_importinglines = pd.read_hdf(
                    self.h5_filepath,
                    key=f"{self.start_index}/relations/region_importinglines",
                )
            except KeyError:
                region_importinglines = pd.read_hdf(
                    self.h5_filepath,
                    key=f"{self.start_index}/relations/region_importingline",
                )
            region_importinglines = region_importinglines.applymap(
                lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
            )
            region_importinglines = region_importinglines.rename(
                columns={"parent": "region", "child": "line_name"}
            )
            if not self.Region_Mapping.empty:
                region_importinglines = pd.merge(
                    region_importinglines, self.Region_Mapping, how="left", on="region"
                )
                region_importinglines.dropna(axis=1, how="all", inplace=True)
        except KeyError:
            logger.warning(
                "Region Importing Lines data not included in h5plexos results"
            )

        return region_importinglines

    def zone_interzonallines(self, filename: str) -> pd.DataFrame:
        """Zone inter-zonal lines mapping.

        Args:
            filename (str): The name of the h5 file to retreive data from.
                If retreiving from fromatted h5 file, just pass scenario name.
        """
        if not self._check_if_existing_filename(filename):
            self._read_data(filename)
        try:
            try:
                zone_interzonallines = pd.read_hdf(
                    self.h5_filepath,
                    key=f"{self.start_index}/relations/zone_interzonallines",
                )
            except KeyError:
                zone_interzonallines = pd.read_hdf(
                    self.h5_filepath,
                    key=f"{self.start_index}/relations/zone_interzonalline",
                )

            zone_interzonallines = zone_interzonallines.applymap(
                lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
            )
            zone_interzonallines.rename(
                columns={"parent": "region", "child": "line_name"}, inplace=True
            )
        except KeyError:
            zone_interzonallines = pd.DataFrame()
            logger.warning("Zone Interzonallines data not included in h5plexos results")

        return zone_interzonallines

    def zone_intrazonallines(self, filename: str) -> pd.DataFrame:
        """Zone intra-zonal lines mapping.

        Args:
            filename (str): The name of the h5 file to retreive data from.
                If retreiving from fromatted h5 file, just pass scenario name.
        """
        if not self._check_if_existing_filename(filename):
            self._read_data(filename)
        try:
            try:
                zone_intrazonallines = pd.read_hdf(
                    self.h5_filepath,
                    key=f"{self.start_index}/relations/zone_intrazonallines",
                )
            except KeyError:
                zone_intrazonallines = pd.read_hdf(
                    self.h5_filepath,
                    key=f"{self.start_index}/relations/zone_intrazonalline",
                )
            zone_intrazonallines = zone_intrazonallines.applymap(
                lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
            )
            zone_intrazonallines.rename(
                columns={"parent": "region", "child": "line_name"}, inplace=True
            )
        except KeyError:
            zone_intrazonallines = pd.DataFrame()
            logger.warning(
                "Zone Intrazonallines Lines data not included in h5plexos results"
            )

        return zone_intrazonallines

    def zone_exporting_lines(self, filename: str) -> pd.DataFrame:
        """Zone exporting lines mapping.

        Args:
            filename (str): The name of the h5 file to retreive data from.
                If retreiving from fromatted h5 file, just pass scenario name.
        """
        if not self._check_if_existing_filename(filename):
            self._read_data(filename)
        try:
            try:
                zone_exportinglines = pd.read_hdf(
                    self.h5_filepath,
                    key=f"{self.start_index}/relations/zone_exportinglines",
                )
            except KeyError:
                zone_exportinglines = pd.read_hdf(
                    self.h5_filepath,
                    key=f"{self.start_index}/relations/zone_exportingline",
                )
            zone_exportinglines = zone_exportinglines.applymap(
                lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
            )
            zone_exportinglines = zone_exportinglines.rename(
                columns={"parent": "region", "child": "line_name"}
            )
        except KeyError:
            logger.warning("zone exporting lines data not included in h5plexos results")
            zone_exportinglines = pd.DataFrame()

        return zone_exportinglines

    def zone_importing_lines(self, filename: str) -> pd.DataFrame:
        """Zone importing lines mapping.

        Args:
            filename (str): The name of the h5 file to retreive data from.
                If retreiving from fromatted h5 file, just pass scenario name.
        """
        if not self._check_if_existing_filename(filename):
            self._read_data(filename)
        try:
            try:
                zone_importinglines = pd.read_hdf(
                    self.h5_filepath,
                    key=f"{self.start_index}/relations/zone_importinglines",
                )
            except KeyError:
                zone_importinglines = pd.read_hdf(
                    self.h5_filepath,
                    key=f"{self.start_index}/relations/zone_importingline",
                )
            zone_importinglines = zone_importinglines.applymap(
                lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
            )
            zone_importinglines = zone_importinglines.rename(
                columns={"parent": "region", "child": "line_name"}
            )
        except KeyError:
            logger.warning("zone importing lines data not included in h5plexos results")
            zone_importinglines = pd.DataFrame()

        return zone_importinglines

    def interface_lines(self, filename: str) -> pd.DataFrame:
        """Interface to lines mapping.

        Args:
            filename (str): The name of the h5 file to retreive data from.
                If retreiving from fromatted h5 file, just pass scenario name.
        """
        if not self._check_if_existing_filename(filename):
            self._read_data(filename)
        try:
            try:
                interface_lines = pd.read_hdf(
                    self.h5_filepath,
                    key=f"{self.start_index}/relations/interface_lines",
                )
            except KeyError:
                interface_lines = pd.read_hdf(
                    self.h5_filepath,
                    key=f"{self.start_index}/relations/interfaces_lines",
                )
            interface_lines = interface_lines.applymap(
                lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
            )
            interface_lines = interface_lines.rename(
                columns={"parent": "interface", "child": "line"}
            )
        except KeyError:
            logger.warning("Interface Lines data not included in h5plexos results")

        return interface_lines

    def region_lines(self, filename: str) -> pd.DataFrame:
        """Region to Lines mapping.

        Args:
            filename (str): The name of the h5 file to retreive data from.
                If retreiving from fromatted h5 file, just pass scenario name.
        """
        region_interregionallines = self.region_interregionallines(filename)
        region_intraregionallines = self.region_intraregionallines(filename)
        region_lines = pd.concat([region_interregionallines, region_intraregionallines])
        return region_lines

    def zone_lines(self, filename: str) -> pd.DataFrame:
        """Zone to Lines mapping.

        Args:
            filename (str): The name of the h5 file to retreive data from.
                If retreiving from fromatted h5 file, just pass scenario name.
        """
        zone_interzonallines = self.zone_interzonallines(filename)
        zone_intrazonallines = self.zone_intrazonallines(filename)
        zone_lines = pd.concat([zone_interzonallines, zone_intrazonallines])
        zone_lines = zone_lines.rename(columns={"region": "zone"})
        return zone_lines

    def reserves(self, filename: str) -> pd.DataFrame:
        """Reserves objects.

        Args:
            filename (str): The name of the h5 file to retreive data from.
                If retreiving from fromatted h5 file, just pass scenario name.
        """
        if not self._check_if_existing_filename(filename):
            self._read_data(filename)
        try:
            try:
                reserves = pd.read_hdf(
                    self.h5_filepath, key=f"{self.start_index}/objects/reserves"
                )
            except KeyError:
                reserves = pd.read_hdf(
                    self.h5_filepath, key=f"{self.start_index}/objects/reserve"
                )
            reserves = reserves.applymap(
                lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
            )
        except KeyError:
            logger.warning("Reserves data not included in h5plexos results")

        return reserves

    def reserves_generators(self, filename: str) -> pd.DataFrame:
        """Reserves to generators mapping.

        Args:
            filename (str): The name of the h5 file to retreive data from.
                If retreiving from fromatted h5 file, just pass scenario name.
        """
        if not self._check_if_existing_filename(filename):
            self._read_data(filename)
        try:
            try:
                reserves_generators = pd.read_hdf(
                    self.h5_filepath,
                    key=f"{self.start_index}/relations/reserves_generators",
                )
            except KeyError:
                reserves_generators = pd.read_hdf(
                    self.h5_filepath,
                    key=f"{self.start_index}/relations/reserve_generators",
                )
            reserves_generators = reserves_generators.applymap(
                lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
            )
            reserves_generators = reserves_generators.rename(
                columns={"child": "gen_name"}
            )
        except KeyError:
            logger.warning("Reserves data not included in h5plexos results")
            reserves_generators = pd.DataFrame()

        return reserves_generators

    def reserves_regions(self, filename: str) -> pd.DataFrame:
        """Reserves to regions mapping.

        Args:
            filename (str): The name of the h5 file to retreive data from.
                If retreiving from fromatted h5 file, just pass scenario name.
        """
        reserves_generators = self.reserves_generators(filename)
        region_generators = self.region_generators(filename)
        try:
            reserves_regions = reserves_generators.merge(
                region_generators, how="left", on="gen_name"
            )
        except KeyError:
            logger.warning("Reserves Region data not available in h5plexos results")
            return pd.DataFrame()
        if not self.Region_Mapping.empty:
            reserves_regions = pd.merge(
                reserves_regions, self.Region_Mapping, how="left", on="region"
            )
            reserves_regions.dropna(axis=1, how="all", inplace=True)
        reserves_regions.drop("gen_name", axis=1, inplace=True)
        reserves_regions.drop_duplicates(inplace=True)
        reserves_regions.reset_index(drop=True, inplace=True)
        return reserves_regions

    def reserves_zones(self, filename: str) -> pd.DataFrame:
        """Reserves to zones mapping.

        Args:
            filename (str): The name of the h5 file to retreive data from.
                If retreiving from fromatted h5 file, just pass scenario name.
        """
        reserves_generators = self.reserves_generators(filename)
        zone_generators = self.zone_generators(filename)
        try:
            reserves_zones = reserves_generators.merge(
                zone_generators, how="left", on="gen_name"
            )
        except KeyError:
            logger.warning("Reserves Zone data not available in h5plexos results")
            return pd.DataFrame()
        reserves_zones.drop("gen_name", axis=1, inplace=True)
        reserves_zones.drop_duplicates(inplace=True)
        reserves_zones.reset_index(drop=True, inplace=True)
        return reserves_zones
