# -*- coding: utf-8 -*-
"""Main formatting source code to format modelling results for plotting.

This code was orginally written to process PLEXOS HDF5 outputs to get them ready for plotting,
but has since been expanded to allow class additions to process results from any energy 
simulation model. 
Once the data is processed it is outputted as an intermediary HDF5 file format so that
it can be read into the marmot_plot_main.py file

@author: Daniel Levie
"""
# =======================================================================================
# Import Python Libraries
# =======================================================================================

import sys
from pathlib import Path
import time
from typing import Union
import pandas as pd
import h5py

try:
    import marmot.utils.mconfig as mconfig
except ModuleNotFoundError:
    from utils.definitions import INCORRECT_ENTRY_POINT

    print(INCORRECT_ENTRY_POINT.format(Path(__file__).name))
    sys.exit()
from marmot.utils.definitions import INPUT_DIR, PLEXOS_YEAR_WARNING
from marmot.utils.loggersetup import SetupLogger
from marmot.formatters import PROCESS_LIBRARY
from marmot.formatters.formatextra import ExtraProperties

# A bug in pandas requires this to be included,
# otherwise df.to_string truncates long strings. Fix available in Pandas 1.0
# but leaving here in case user version not up to date
pd.set_option("display.max_colwidth", 1000)

formatter_settings = mconfig.parser("formatter_settings")


class MarmotFormat(SetupLogger):
    """Main module class to be instantiated to run the formatter.

    MarmotFormat handles the passing on information to the various
    Process classes and handles the saving of formatted results.
    Once the outputs have been processed, they are saved to an intermediary hdf5 file
    which can then be read into the Marmot plotting code
    """

    def __init__(
        self,
        Scenario_name: str,
        Model_Solutions_folder: Union[str, Path],
        Properties_File: Union[str, Path, pd.DataFrame],
        Marmot_Solutions_folder: Union[str, Path] = None,
        mapping_folder: Union[str, Path] = INPUT_DIR.joinpath("mapping_folder"),
        Region_Mapping: Union[str, Path, pd.DataFrame] = pd.DataFrame(),
        emit_names: Union[str, Path, pd.DataFrame] = pd.DataFrame(),
        **kwargs,
    ):
        """
        Args:
            Scenario_name (str): Name of scenario to process.
            Model_Solutions_folder (Union[str, Path]): Folder containing model simulation
                results subfolders and their files.
            Properties_File (Union[str, Path, pd.DataFrame]): Properties
                to process, must follow format seen in Marmot directory.
            Marmot_Solutions_folder (Union[str, Path], optional): Folder to save Marmot
                solution files.
                Defaults to None.
            mapping_folder (Union[str, Path], optional): The location of the Marmot
                mapping folder.
                Defaults to INPUT_DIR.joinpath('mapping_folder').
            Region_Mapping (Union[str, Path, pd.DataFrame], optional): Mapping file
                to map custom regions/zones to create custom aggregations.
                Aggregations are created by grouping PLEXOS regions.
                Defaults to pd.DataFrame().
            emit_names (Union[str, Path, pd.DataFrame], optional): Mapping file
                to rename emissions types.
                Defaults to pd.DataFrame().
            **kwargs
                These parameters will be passed to the 
                marmot.utils.loggersetup.SetupLogger class.
        """
        super().__init__("formatter", **kwargs)  # Instantiation of SetupLogger

        self.Scenario_name = Scenario_name
        self.Model_Solutions_folder = Path(Model_Solutions_folder)

        self.mapping_folder = Path(mapping_folder)

        if Marmot_Solutions_folder is None:
            self.Marmot_Solutions_folder = self.Model_Solutions_folder
        else:
            self.Marmot_Solutions_folder = Path(Marmot_Solutions_folder)
            self.Marmot_Solutions_folder.mkdir(exist_ok=True)

        if isinstance(Properties_File, (str, Path)):
            try:
                self.Properties_File = pd.read_csv(Properties_File)
            except FileNotFoundError:
                self.logger.error(
                    "Could not find specified "
                    "Properties_File; check file name. "
                    "This is required to run Marmot, "
                    "system will now exit"
                )
                sys.exit()
        elif isinstance(Properties_File, pd.DataFrame):
            self.Properties_File = Properties_File

        if isinstance(Region_Mapping, (str, Path)):
            try:
                Region_Mapping = pd.read_csv(Region_Mapping)
                if not self.Region_Mapping.empty:
                    Region_Mapping = Region_Mapping.astype(object)
            except FileNotFoundError:
                self.logger.warning(
                    "Could not find specified "
                    "Region Mapping file; "
                    "check file name\n"
                )
                Region_Mapping = pd.DataFrame()
            self.Region_Mapping = Region_Mapping
        elif isinstance(Region_Mapping, pd.DataFrame):
            if not Region_Mapping.empty:
                Region_Mapping = Region_Mapping.astype(object)
            self.Region_Mapping = Region_Mapping
        try:
            # delete category columns if exists
            self.Region_Mapping = self.Region_Mapping.drop(["category"], axis=1)
        except KeyError:
            pass

        if isinstance(emit_names, (str, Path)):
            try:
                self.emit_names = pd.read_csv(emit_names)
                if not self.emit_names.empty:
                    self.emit_names.rename(
                        columns={
                            self.emit_names.columns[0]: "Original",
                            self.emit_names.columns[1]: "New",
                        },
                        inplace=True,
                    )
            except FileNotFoundError:
                self.logger.warning(
                    "Could not find specified emissions "
                    "mapping file; check file name\n"
                )
                self.emit_names = pd.DataFrame()
        elif isinstance(emit_names, pd.DataFrame):
            self.emit_names = emit_names
            if not self.emit_names.empty:
                self.emit_names.rename(
                    columns={
                        self.emit_names.columns[0]: "Original",
                        self.emit_names.columns[1]: "New",
                    },
                    inplace=True,
                )

    def save_to_h5(
        self,
        df: pd.DataFrame,
        file_name: Path,
        key: str,
        mode: str = "a",
        complevel: int = 9,
        complib: str = "blosc:zlib",
        **kwargs,
    ) -> None:
        """Saves data to formatted hdf5 file

        Args:
            df (pd.DataFrame): Dataframe to save
            file_name (Path): name of hdf5 file
            key (str): formatted property identifier,
                e.g generator_Generation
            mode (str, optional): file access mode.
                Defaults to "a".
            complevel (int, optional): compression level.
                Defaults to 9.
            complib (str, optional): compression library.
                Defaults to 'blosc:zlib'.
            **kwargs
                These parameters will be passed pandas.to_hdf function.
        """
        self.logger.info("Saving data to h5 file...")
        df.to_hdf(
            file_name,
            key=key,
            mode=mode,
            complevel=complevel,
            complib=complib,
            **kwargs,
        )

        self.logger.info("Data saved to h5 file successfully\n")

    def run_formatter(
        self,
        sim_model: str = "PLEXOS",
        plexos_block: str = "ST",
        append_block_name: bool = False,
        process_subset_years: list = None,
    ) -> None:
        """Main method to call to begin formatting simulation model results

        Args:
            sim_model (str, optional): Name of simulation model to
                process data for.
                Defaults to 'PLEXOS'.
            plexos_block (str, optional): PLEXOS results type.
                Defaults to 'ST'.
            append_block_name (bool, optional): Append block type to
                scenario name.
                Defaults to False.
            process_subset_years (list, optional): If provided only process
                years specified. (Only used for sim_model = ReEDS)
                Defaults to None.
        """
        if append_block_name:
            scen_name = f"{self.Scenario_name} {plexos_block}"
        else:
            scen_name = self.Scenario_name

        try:
            process_class = PROCESS_LIBRARY[sim_model]
            if process_class is None:
                self.logger.error(
                    "A required module was not found to " f"process {sim_model} results"
                )
                self.logger.error(PROCESS_LIBRARY["Error"])
                sys.exit()
        except KeyError:
            self.logger.error(f"No formatter found for model: {sim_model}")
            sys.exit()

        self.logger.info(f"#### Processing {scen_name} {sim_model} " "Results ####")

        hdf5_output_name = f"{scen_name}_formatted.h5"
        input_folder = self.Model_Solutions_folder.joinpath(str(self.Scenario_name))
        output_folder = self.Marmot_Solutions_folder.joinpath("Processed_HDF5_folder")
        output_folder.mkdir(exist_ok=True)

        output_file_path = output_folder.joinpath(hdf5_output_name)

        process_sim_model = process_class(
            input_folder,
            output_file_path,
            self.Region_Mapping,
            self.emit_names,
            plexos_block=plexos_block,
            process_subset_years=process_subset_years,
        )

        files_list = process_sim_model.get_input_files

        # init of ExtraProperties class
        extraprops_init = ExtraProperties(process_sim_model, files_list)
        # =====================================================================
        # Process the Outputs
        # =====================================================================

        # Creates Initial HDF5 file for outputting formated data
        Processed_Data_Out = pd.DataFrame()
        if output_file_path.is_file():
            self.logger.info(
                f"'{output_file_path}' already exists: New " "variables will be added\n"
            )
            # Skip properties that already exist in *formatted.h5 file.
            with h5py.File(output_file_path, "r") as f:
                existing_keys = [key for key in f.keys()]
            # The processed HDF5 output file already exists. If metadata is already in
            # this file, leave as is. Otherwise, append it to the file.
            if "metadata" not in existing_keys:
                self.logger.info("Adding metadata to processed HDF5 file.")
                process_sim_model.output_metadata(files_list)

            if not formatter_settings["skip_existing_properties"]:
                existing_keys = []

        # The processed HDF5 file does not exist.
        # Create the file and add metadata to it.
        else:
            existing_keys = []
            # Create empty hdf5 file
            f = h5py.File(output_file_path, "w")
            f.close()
            process_sim_model.output_metadata(files_list)

        process_properties = self.Properties_File.loc[
            self.Properties_File["collect_data"] == True
        ]

        start = time.time()
        # Main loop to process each output and pass data to functions
        for index, row in process_properties.iterrows():
            Processed_Data_Out = pd.DataFrame()
            data_chunks = []

            self.logger.info(f'Processing {row["group"]} {row["data_set"]}')

            prop_underscore = row["data_set"].replace(" ", "_")
            key_path = row["group"] + "_" + prop_underscore
            # Get name to save property as in formatted h5 file
            property_key_name = process_sim_model.PROPERTY_MAPPING.get(
                key_path, key_path
            )

            if property_key_name not in existing_keys:
                for model in files_list:
                    processed_data = process_sim_model.get_processed_data(
                        row["group"], row["data_set"], row["data_type"], model
                    )
                    if processed_data.empty is True:
                        data_chunks.append(processed_data)
                        break

                    # Check if data is for year interval and of type capacity
                    if (
                        row["data_type"] == "year"
                        and sim_model == "PLEXOS"
                        and (
                            (row["data_set"] == "Installed Capacity")
                            | (row["data_set"] == "Export Limit")
                            | (row["data_set"] == "Import Limit")
                        )
                    ):
                        data_chunks.append(processed_data)
                        self.logger.info(
                            f"{row['data_set']} Year property reported "
                            "from only the first partition"
                        )
                        break
                    else:
                        data_chunks.append(processed_data)

                # Combine models
                Processed_Data_Out = process_sim_model.combine_models(data_chunks)
                if Processed_Data_Out.empty is False:
                    if row["data_type"] == "year" and sim_model == "PLEXOS":
                        self.logger.info(PLEXOS_YEAR_WARNING)
                    save_attempt = 1
                    while save_attempt <= 3:
                        try:
                            self.save_to_h5(
                                Processed_Data_Out,
                                output_file_path,
                                key=property_key_name,
                            )
                            save_attempt = 4
                        except OSError:
                            self.logger.warning(
                                "h5 File is probably in use, "
                                "waiting to attempt to save again"
                            )
                            time.sleep(60)
                            save_attempt += 1

                    # Calculate any extra properties
                    if property_key_name in process_sim_model.EXTRA_MARMOT_PROPERTIES:

                        extra_prop_functions = (
                            process_sim_model.EXTRA_MARMOT_PROPERTIES[property_key_name]
                        )
                        for prop_function_tup in extra_prop_functions:
                            prop_name, prop_function = prop_function_tup

                            if (
                                prop_name not in h5py.File(output_file_path, "r")
                                or not formatter_settings["skip_existing_properties"]
                            ):

                                self.logger.info(f"Processing {prop_name}")
                                prop = prop_function(
                                    extraprops_init,
                                    Processed_Data_Out,
                                    timescale=row["data_type"],
                                )

                                if prop.empty is False:
                                    self.save_to_h5(
                                        prop, output_file_path, key=prop_name
                                    )
                                else:
                                    self.logger.warning(f"{prop_name} was not saved")
                            # Run again to check for properties based of new properties
                            if prop_name in process_sim_model.EXTRA_MARMOT_PROPERTIES:

                                extra2_prop_functions = (
                                    process_sim_model.EXTRA_MARMOT_PROPERTIES[prop_name]
                                )
                                for prop_function_tup2 in extra2_prop_functions:
                                    prop_name2, prop_function2 = prop_function_tup2

                                    if (
                                        prop_name2
                                        not in h5py.File(output_file_path, "r")
                                        or not formatter_settings[
                                            "skip_existing_properties"
                                        ]
                                    ):

                                        self.logger.info(f"Processing {prop_name2}")
                                        prop2 = prop_function2(
                                            extraprops_init,
                                            prop,
                                            timescale=row["data_type"],
                                        )

                                        if prop2.empty is False:
                                            self.save_to_h5(
                                                prop2, output_file_path, key=prop_name2
                                            )
                                        else:
                                            self.logger.warning(
                                                f"{prop_name2} was not saved"
                                            )

                else:
                    continue

            else:
                self.logger.info(f"{key_path} already exists in output .h5 file.")
                self.logger.info("PROPERTY ALREADY PROCESSED\n")
                continue

        end = time.time()
        elapsed = end - start
        self.logger.info("Main loop took %s minutes", round(elapsed / 60, 2))
        self.logger.info(f"Formatting COMPLETED for {scen_name}")


def main():
    """Run the formatting code and format desired properties based on user input files."""

    # ===================================================================================
    # Input Properties
    # ===================================================================================

    Marmot_user_defined_inputs = pd.read_csv(
        INPUT_DIR.joinpath(mconfig.parser("user_defined_inputs_file")),
        usecols=["Input", "User_defined_value"],
        index_col="Input",
        skipinitialspace=True,
    )

    simulation_model = (
        Marmot_user_defined_inputs.loc["Simulation_model"]
        .to_string(index=False)
        .strip()
    )

    if pd.isna(
        Marmot_user_defined_inputs.loc["PLEXOS_data_blocks", "User_defined_value"]
    ):
        plexos_data_blocks = ["ST"]
    else:
        plexos_data_blocks = (
            pd.Series(
                Marmot_user_defined_inputs.loc["PLEXOS_data_blocks"]
                .squeeze()
                .split(",")
            )
            .str.strip()
            .tolist()
        )

    # File which determiens which plexos properties to pull from the h5plexos results and
    # process, this file is in the repo
    Properties_File = pd.read_csv(
        INPUT_DIR.joinpath(
            mconfig.parser(f"{simulation_model.lower()}_properties_file")
        )
    )

    # Name of the Scenario(s) being run, must have the same name(s) as the folder
    # holding the runs HDF5 file
    Scenario_List = (
        pd.Series(
            Marmot_user_defined_inputs.loc["Scenario_process_list"].squeeze().split(",")
        )
        .str.strip()
        .tolist()
    )
    # The folder that contains all the simulation model outputs - the files should
    # be contained in another folder with the Scenario_name
    Model_Solutions_folder = (
        Marmot_user_defined_inputs.loc["Model_Solutions_folder"]
        .to_string(index=False)
        .strip()
    )

    # Folder to save your processed solutions
    if pd.isna(
        Marmot_user_defined_inputs.loc["Marmot_Solutions_folder", "User_defined_value"]
    ):
        Marmot_Solutions_folder = None
    else:
        Marmot_Solutions_folder = (
            Marmot_user_defined_inputs.loc["Marmot_Solutions_folder"]
            .to_string(index=False)
            .strip()
        )

    # This folder contains all the csv required for mapping and selecting outputs
    # to process. Examples of these mapping files are within the Marmot repo, you
    # may need to alter these to fit your needs
    Mapping_folder = INPUT_DIR.joinpath("mapping_folder")

    if (
        pd.isna(
            Marmot_user_defined_inputs.loc[
                "Region_Mapping.csv_name", "User_defined_value"
            ]
        )
        is True
    ):
        Region_Mapping = pd.DataFrame()
    else:
        Region_Mapping = pd.read_csv(
            INPUT_DIR.joinpath(
                Mapping_folder,
                Marmot_user_defined_inputs.loc["Region_Mapping.csv_name"]
                .to_string(index=False)
                .strip(),
            )
        )

    # Subset of years to process
    if pd.isna(
        Marmot_user_defined_inputs.loc["process_subset_years", "User_defined_value"]
    ):
        process_subset_years = None
    else:
        process_subset_years = (
            pd.Series(
                Marmot_user_defined_inputs.loc["process_subset_years"]
                .squeeze()
                .split(",")
            )
            .str.strip()
            .tolist()
        )
    # ===================================================================================
    # Standard Naming of Emissions types (optional)
    # ===================================================================================

    emit_names = INPUT_DIR.joinpath(
        Mapping_folder,
        Marmot_user_defined_inputs.loc["emit_names.csv_name"]
        .to_string(index=False)
        .strip(),
    )

    # ===================================================================================
    # Loop through scenarios in list
    # ===================================================================================

    for Scenario_name in Scenario_List:

        initiate = MarmotFormat(
            Scenario_name,
            Model_Solutions_folder,
            Properties_File,
            Marmot_Solutions_folder=Marmot_Solutions_folder,
            mapping_folder=Mapping_folder,
            Region_Mapping=Region_Mapping,
            emit_names=emit_names,
        )

        if simulation_model == "PLEXOS":
            for block in plexos_data_blocks:
                initiate.run_formatter(
                    plexos_block=block,
                    append_block_name=formatter_settings["append_plexos_block_name"],
                )
        else:
            initiate.run_formatter(
                sim_model=simulation_model, process_subset_years=process_subset_years
            )


if __name__ == "__main__":
    main()
