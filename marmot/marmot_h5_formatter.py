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
from marmot.utils.definitions import INPUT_DIR
from marmot.utils.loggersetup import SetupLogger
from marmot.formatters import PROCESS_LIBRARY

# A bug in pandas requires this to be included,
# otherwise df.to_string truncates long strings. Fix available in Pandas 1.0
# but leaving here in case user version not up to date
pd.set_option("display.max_colwidth", 1000)


class MarmotFormat(SetupLogger):
    """Main module class to be instantiated to run the formatter.

    MarmotFormat handles the passing on information to the various
    Process classes and handles the saving of formatted results.
    Once the outputs have been processed, they are saved to an intermediary hdf5 file
    which can then be read into the Marmot plotting code
    """

    def __init__(self, Scenario_name: str, 
                 Model_Solutions_folder: Union[str, Path], 
                 Plexos_Properties: Union[str, Path, pd.DataFrame],
                 Marmot_Solutions_folder: Union[str, Path] = None,
                 mapping_folder: Union[str, Path] = 'mapping_folder',
                 Region_Mapping: Union[str, Path, pd.DataFrame] = pd.DataFrame(),
                 emit_names: Union[str, Path, pd.DataFrame] = pd.DataFrame(),
                 VoLL: int = 10000,
                 **kwargs):
        """
        Args:
            Scenario_name (str): Name of scenario to process.
            Model_Solutions_folder (Union[str, Path]): Folder containing model simulation 
                results subfolders and their files.
            Plexos_Properties (Union[str, Path, pd.DataFrame]): PLEXOS properties 
                to process, must follow format seen in Marmot directory.
            Marmot_Solutions_folder (Union[str, Path], optional): Folder to save Marmot 
                solution files.
                Defaults to None.
            mapping_folder (Union[str, Path], optional): The location of the Marmot 
                mapping folder.
                Defaults to 'mapping_folder'.
            Region_Mapping (Union[str, Path, pd.DataFrame], optional): Mapping file 
                to map custom regions/zones to create custom aggregations.
                Aggregations are created by grouping PLEXOS regions.
                Defaults to pd.DataFrame().
            emit_names (Union[str, Path, pd.DataFrame], optional): Mapping file 
                to rename emissions types. 
                Defaults to pd.DataFrame().
            VoLL (int, optional): Value of lost load, used to calculate 
                cost of unserved energy. 
                Defaults to 10000.
        """
        super().__init__('marmot_format', **kwargs) # Instantiation of SetupLogger

        self.Scenario_name = Scenario_name
        self.Model_Solutions_folder = Path(Model_Solutions_folder)
        
        self.mapping_folder = Path(mapping_folder)
        self.VoLL = VoLL

        if Marmot_Solutions_folder is None:
            self.Marmot_Solutions_folder = self.Model_Solutions_folder
        else:
            self.Marmot_Solutions_folder = Path(Marmot_Solutions_folder)

        if isinstance(Plexos_Properties, (str, Path)):
            try:
                self.Plexos_Properties = pd.read_csv(Plexos_Properties)
            except FileNotFoundError:
                self.logger.error("Could not find specified "
                                    "Plexos_Properties file; check file name. "
                                    "This is required to run Marmot, "
                                    "system will now exit")
                sys.exit()
        elif isinstance(Plexos_Properties, pd.DataFrame):
            self.Plexos_Properties = Plexos_Properties

        if isinstance(Region_Mapping, (str, Path)):
            try:
                self.Region_Mapping = pd.read_csv(Region_Mapping)
                if not self.Region_Mapping.empty:
                    self.Region_Mapping = self.Region_Mapping.astype(str)
            except FileNotFoundError:
                self.logger.warning("Could not find specified "
                                    "Region Mapping file; "
                                    "check file name\n")
                self.Region_Mapping = pd.DataFrame()
        elif isinstance(Region_Mapping, pd.DataFrame):
            self.Region_Mapping = Region_Mapping
            if not self.Region_Mapping.empty:
                self.Region_Mapping = self.Region_Mapping.astype('string')
        try:
            # delete category columns if exists
            self.Region_Mapping = self.Region_Mapping.drop(["category"], axis=1)  
        except KeyError:
            pass
        
        if isinstance(emit_names, (str, Path)):
            try:
                self.emit_names = pd.read_csv(emit_names)
                if not self.emit_names.empty:
                    self.emit_names.rename(columns=
                                           {self.emit_names.columns[0]: 'Original',
                                           self.emit_names.columns[1]: 'New'},
                                           inplace=True)
            except FileNotFoundError:
                self.logger.warning("Could not find specified emissions "
                                    "mapping file; check file name\n")
                self.emit_names = pd.DataFrame()
        elif isinstance(emit_names, pd.DataFrame):
            self.emit_names = emit_names
            if not self.emit_names.empty:
                self.emit_names.rename(columns={self.emit_names.columns[0]: 'Original',
                                                self.emit_names.columns[1]: 'New'},
                                       inplace=True)

    @staticmethod
    def _save_to_h5(df: pd.DataFrame, file_name: Path, key: str, 
                    mode: str = "a", complevel: int = 9, 
                    complib: str ='blosc:zlib', **kwargs) -> None:
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
        """
        df.to_hdf(file_name, key=key, mode=mode,
                    complevel=complevel,
                    complib=complib,
                    **kwargs)

    def run_formatter(self, sim_model='PLEXOS', plexos_block='ST', 
                      append_block_name=False) -> None:
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
        """
        if append_block_name:
            scen_name = f"{self.Scenario_name} {plexos_block}"
        else:
            scen_name = self.Scenario_name
        
        try:
            process_class = PROCESS_LIBRARY[sim_model]
            if process_class is None:
                self.logger.error("A required module was not found to "
                                  f"process {sim_model} results")
                self.logger.error(PROCESS_LIBRARY['Error'])
                sys.exit()
        except KeyError:
            self.logger.error(f"No formatter found for model: {sim_model}")
            sys.exit()

        self.logger.info(f"#### Processing {scen_name} {sim_model} "
                         "Results ####")

        hdf5_output_name = f"{scen_name}_formatted.h5"
        input_folder = self.Model_Solutions_folder.joinpath( 
                                      str(self.Scenario_name))
        output_folder = self.Marmot_Solutions_folder.joinpath( 
                                      'Processed_HDF5_folder')
        output_folder.mkdir(exist_ok=True)

        output_file_path = output_folder.joinpath(hdf5_output_name)
        
        process_sim_model = process_class(input_folder, 
                                            self.Region_Mapping,
                                            self.emit_names,
                                            plexos_block=plexos_block)
        files_list = process_sim_model.get_input_files()

        # =====================================================================
        # Process the Outputs
        # =====================================================================

        # Creates Initial HDF5 file for outputting formated data
        Processed_Data_Out = pd.DataFrame()
        if output_file_path.is_file():
            self.logger.info(f"'{output_file_path}' already exists: New "
                             "variables will be added\n")
            # Skip properties that already exist in *formatted.h5 file.
            with h5py.File(output_file_path, 'r') as f:
                existing_keys = [key for key in f.keys()]
            # The processed HDF5 output file already exists. If metadata is already in
            # this file, leave as is. Otherwise, append it to the file.
            if 'metadata' not in existing_keys:
                self.logger.info('Adding metadata to processed HDF5 file.')
                process_sim_model.output_metadata(files_list, output_file_path)

            if not mconfig.parser('skip_existing_properties'):
                existing_keys = []

        # The processed HDF5 file does not exist. 
        # Create the file and add metadata to it.
        else:
            existing_keys = []
            # Create empty hdf5 file 
            f = h5py.File(output_file_path, "w")
            f.close()
            process_sim_model.output_metadata(files_list, output_file_path)

        process_properties = (self.Plexos_Properties
                                  .loc[self.Plexos_Properties["collect_data"] == True])
        
        start = time.time()
        # Main loop to process each output and pass data to functions
        for index, row in process_properties.iterrows():
            Processed_Data_Out = pd.DataFrame()
            data_chunks = []

            self.logger.info(f'Processing {row["group"]} {row["data_set"]}')
            prop_underscore = row["data_set"].replace(' ', '_')
            key_path = row["group"] + "_" + prop_underscore
            
            if key_path not in existing_keys:
                for model in files_list:
                    self.logger.info(f"      {model}")
                    processed_data = process_sim_model.get_processed_data(row["group"], 
                                                                  row["data_set"], 
                                                                  row["data_type"], 
                                                                  model)
                    if processed_data.empty is True:
                        break
                    
                    # Check if data is for year interval and of type capacity
                    if (row["data_type"] == "year") & (
                            (row["data_set"] == "Installed Capacity")
                            | (row["data_set"] == "Export Limit")
                            | (row["data_set"] == "Import Limit")
                            ):
                        data_chunks.append(processed_data)
                        self.logger.info(f"{row['data_set']} Year property reported "
                                         "from only the first partition")
                        break
                    else:
                        data_chunks.append(processed_data)

                if data_chunks:
                    Processed_Data_Out = pd.concat(data_chunks, copy=False)

                if Processed_Data_Out.empty is False:
                    if (row["data_type"] == "year"):
                        self.logger.info("Please Note: Year properties can not "
                                         "be checked for duplicates.\n"
                                         "Overlaping data cannot be removed from "
                                         "'Year' grouped data.\nThis will effect "
                                         "Year data that differs between partitions "
                                         "such as cost results.\nIt will not effect "
                                         "Year data that is equal in all partitions "
                                         "such as Installed Capacity or "
                                         "Line Limit results")
                    else:
                        oldsize = Processed_Data_Out.size
                        # Remove duplicates; keep first entry
                        Processed_Data_Out = (Processed_Data_Out.loc
                                              [~Processed_Data_Out
                                              .index.duplicated(keep='first')])

                        if (oldsize - Processed_Data_Out.size) > 0:
                            self.logger.info("Drop duplicates removed "
                                             f"{oldsize-Processed_Data_Out.size} rows")

                    row["data_set"] = row["data_set"].replace(' ', '_')
                    
                    save_attempt=1
                    while save_attempt<=3:
                        try:
                            self.logger.info("Saving data to h5 file...")
                            MarmotFormat._save_to_h5(Processed_Data_Out,
                                                     output_file_path, 
                                                     key=(f'{row["group"]}_'
                                                          f'{row["data_set"]}'))

                            self.logger.info("Data saved to h5 file successfully\n")
                            save_attempt=4
                        except:
                            self.logger.warning("h5 File is probably in use, "
                                                "waiting to attempt to save again")
                            time.sleep(60)
                            save_attempt+=1
                else:
                    continue
            else:
                self.logger.info(f"{key_path} already exists in output .h5 file.")
                self.logger.info("PROPERTY ALREADY PROCESSED\n")
                continue

        # ===============================================================================
        # Calculate Extra Outputs
        # ===============================================================================
        if "generator_Curtailment" not in \
            h5py.File(output_file_path, 'r') or not \
            mconfig.parser('skip_existing_properties'):
            try:
                self.logger.info("Processing generator Curtailment")
                try:
                    Avail_Gen_Out = pd.read_hdf(output_file_path,
                                                'generator_Available_Capacity')
                    Total_Gen_Out = pd.read_hdf(output_file_path,
                                                'generator_Generation')
                    if Total_Gen_Out.empty is True:
                        self.logger.warning("generator_Available_Capacity & "
                                            "generator_Generation are required "
                                            "for Curtailment calculation")
                except KeyError:
                    self.logger.warning("generator_Available_Capacity & "
                                        "generator_Generation are required "
                                        "for Curtailment calculation")

                Curtailment_Out = Avail_Gen_Out - Total_Gen_Out

                Upward_Available_Capacity = Curtailment_Out

                MarmotFormat._save_to_h5(Curtailment_Out,
                                    output_file_path, 
                                    key="generator_Curtailment")

                MarmotFormat._save_to_h5(Upward_Available_Capacity,
                                    output_file_path, 
                                    key="generator_Upward_Available_Capacity")

                self.logger.info("Data saved to h5 file successfully\n")
                # Clear Some Memory
                del Total_Gen_Out
                del Avail_Gen_Out
                del Curtailment_Out
            except Exception:
                self.logger.warning("NOTE!! Curtailment not calculated, "
                                    "processing skipped\n")

        if "region_Cost_Unserved_Energy" not in \
            h5py.File(output_file_path, 'r') or not \
            mconfig.parser('skip_existing_properties'):
            try:
                self.logger.info("Calculating Cost Unserved Energy: Regions")
                Cost_Unserved_Energy = pd.read_hdf(output_file_path,
                                                   'region_Unserved_Energy')
                                                   
                Cost_Unserved_Energy = Cost_Unserved_Energy * self.VoLL

                MarmotFormat._save_to_h5(Cost_Unserved_Energy,
                                    output_file_path, 
                                    key="region_Cost_Unserved_Energy")
            except KeyError:
                self.logger.warning("NOTE!! Regional Unserved Energy not available "
                                    "to process, processing skipped\n")
                pass

        if "zone_Cost_Unserved_Energy" not in \
            h5py.File(output_file_path, 'r') or not \
            mconfig.parser('skip_existing_properties'):
            try:
                self.logger.info("Calculating Cost Unserved Energy: Zones")
                Cost_Unserved_Energy = pd.read_hdf(output_file_path,
                                                   'zone_Unserved_Energy')
                Cost_Unserved_Energy = Cost_Unserved_Energy * self.VoLL

                MarmotFormat._save_to_h5(Cost_Unserved_Energy,
                                    output_file_path, 
                                    key="zone_Cost_Unserved_Energy")
            except KeyError:
                self.logger.warning("NOTE!! Zonal Unserved Energy not available to "
                                    "process, processing skipped\n")
                pass

        end = time.time()
        elapsed = end - start
        self.logger.info('Main loop took %s minutes', round(elapsed/60, 2))
        self.logger.info(f'Formatting COMPLETED for {scen_name}')


def main():
    """Run the formatting code and format desired properties based on user input files."""

    # ===================================================================================
    # Input Properties
    # ===================================================================================

    Marmot_user_defined_inputs = pd.read_csv(INPUT_DIR.joinpath(mconfig.parser("user_defined_inputs_file")),
                                             usecols=['Input', 'User_defined_value'],
                                             index_col='Input',
                                             skipinitialspace=True)

    simulation_model = (Marmot_user_defined_inputs.loc['Simulation_model']
                                                        .to_string(index=False).strip())

    if pd.isna(Marmot_user_defined_inputs.loc['PLEXOS_data_blocks',
                                              'User_defined_value']):
        plexos_data_blocks = ['ST']
    else:
        plexos_data_blocks = (pd.Series(Marmot_user_defined_inputs.loc['PLEXOS_data_blocks']
                                                                  .squeeze().split(","))
                                                                  .str.strip().tolist())

    # File which determiens which plexos properties to pull from the h5plexos results and 
    # process, this file is in the repo
    Plexos_Properties = pd.read_csv(INPUT_DIR.joinpath(mconfig.parser('plexos_properties_file')))
    
    # Name of the Scenario(s) being run, must have the same name(s) as the folder 
    # holding the runs HDF5 file
    Scenario_List = (pd.Series(Marmot_user_defined_inputs.loc['Scenario_process_list']
                                                         .squeeze().split(","))
                                                         .str.strip().tolist())
    # The folder that contains all the simulation model outputs - the files should 
    # be contained in another folder with the Scenario_name
    Model_Solutions_folder = (Marmot_user_defined_inputs.loc['Model_Solutions_folder']
                                                         .to_string(index=False).strip())

    # Folder to save your processed solutions
    if pd.isna(Marmot_user_defined_inputs.loc['Marmot_Solutions_folder',
                                              'User_defined_value']):
        Marmot_Solutions_folder = None
    else:
        Marmot_Solutions_folder = (Marmot_user_defined_inputs.loc
                                                             ['Marmot_Solutions_folder']
                                                             .to_string(index=False)
                                                             .strip())

    # This folder contains all the csv required for mapping and selecting outputs 
    # to process. Examples of these mapping files are within the Marmot repo, you 
    # may need to alter these to fit your needs
    Mapping_folder = INPUT_DIR.joinpath('mapping_folder')

    if pd.isna(Marmot_user_defined_inputs.loc['Region_Mapping.csv_name', 
                                              'User_defined_value']) is True:
        Region_Mapping = pd.DataFrame()
    else:
        Region_Mapping = (pd.read_csv(INPUT_DIR.joinpath(Mapping_folder, 
                                                   Marmot_user_defined_inputs
                                                   .loc['Region_Mapping.csv_name']
                                                   .to_string(index=False).strip())))

    # Value of Lost Load for calculating cost of unserved energy
    VoLL = pd.to_numeric(Marmot_user_defined_inputs.loc['VoLL'].to_string(index=False))

    # ===================================================================================
    # Standard Naming of Emissions types (optional)
    # ===================================================================================

    emit_names = INPUT_DIR.joinpath(Mapping_folder, Marmot_user_defined_inputs
                              .loc['emit_names.csv_name']
                              .to_string(index=False).strip())

    # ===================================================================================
    # Loop through scenarios in list
    # ===================================================================================

    for Scenario_name in Scenario_List:
        
        initiate = MarmotFormat(Scenario_name, Model_Solutions_folder, 
                                Plexos_Properties,
                                Marmot_Solutions_folder=Marmot_Solutions_folder,
                                mapping_folder=Mapping_folder,
                                Region_Mapping=Region_Mapping,
                                emit_names=emit_names,
                                VoLL=VoLL)

        if simulation_model=='PLEXOS':
            for block in plexos_data_blocks:
                initiate.run_formatter(plexos_block=block, 
                                       append_block_name=mconfig.parser('append_plexos_block_name'))
        else:
            initiate.run_formatter(sim_model=simulation_model)


if __name__ == '__main__':
    main()

#===============================================================================
# Code that can be used to test PLEXOS_H5_results_formatter
#===============================================================================
# test = test.xs("p60",level='region')
# test = test.xs("gas-ct",level='tech')
# test = test.reset_index(['timestamp','node'])
# test = test.groupby(["timestamp", "node"], as_index=False).sum()
# test = test.pivot(index='timestamp', columns='node', values=0)

# test = test[['600003_PR IS31G_20','600005_MNTCE31G_22']]
# test = test.reset_index()

# test.index.get_level_values('region') = (test.index.get_level_values('region')
#                                                       .astype("category"))

# test['timestamp'] = test['timestamp'].astype("category")

# test.index = (test.index.set_levels(test.index.levels[-1].
#                                           astype('category'), level=-1))

# test.memory_usage(deep=True)
# test[0] = pd.to_numeric(test[0], downcast='float')

# test.memory_usage(deep=False)

# Stacked_Gen_read = Stacked_Gen_read.reset_index() 
# Stacked_Gen_read.rename(columns={'name':'zone'}, inplace=True)
#         Stacked_Gen_read = Stacked_Gen_read.drop(["band", 
#                                               "property", "category"],axis=1)

    #storage = db.storage("Generation")
    #storage = df_process_storage(storage, overlap)

# df_old = df
# t =df.loc[~df.index.duplicated()]
# df_old.equals(df)
