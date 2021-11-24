# -*- coding: utf-8 -*-
"""Main plotting source code, creates output figures and data-tables.

marmot_plot_main.py is the main plotting script within Marmot which calls on 
supporting files to read in data, create the plot, and then return the plot and 
data to marmot_plot_main.py. The supporting modules can be viewed within the repo 
plottingmodules folder and have descriptive names such as total_generation.py, 
generation_stack.py, curtailment.py etc.

@author: Daniel Levie
"""
#========================================================================================
# Import Python Libraries
#========================================================================================

import os
import pathlib
import sys

FILE_DIR = pathlib.Path(__file__).parent.absolute() # Location of this module
# Add Marmot directory to sys path if running from __main__
if __name__ == '__main__': 
    #If running from top level of repo.
    if os.path.dirname(os.path.dirname(__file__)) not in sys.path:
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        os.chdir(pathlib.Path(__file__).parent.absolute().parent.absolute())
import importlib
import logging
import logging.config
import time
from typing import Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import yaml

try:
    from marmot.meta_data import MetaData
except ModuleNotFoundError:
    print("Attempted import of Marmot as a module from a Git directory. "
          "Import of Marmot will not function in this way. "
          "To import Marmot as a module use the preferred method of pip "
          "installing Marmot, or add the Marmot directory to the system path, "
          "see ReadME for details.\nSystem will now exit")
    sys.exit()
import marmot.config.mconfig as mconfig
from marmot.plottingmodules.plotutils.plot_exceptions import (
    DataSavedInModule, InputSheetError, MissingInputData, MissingMetaData,
    MissingZoneData, UnderDevelopment, UnsupportedAggregation)

#A bug in pandas requires this to be included, otherwise df.to_string truncates 
# long strings.
#Fix available in Pandas 1.0 but leaving here in case user version not up to date
pd.set_option("display.max_colwidth", 1000)


class SetupLogger():
    """Sets up the python logger.

    This class handles the following.

    1. Configures logger from marmot_logging_config.yml file.
    2. Handles rollover of log file on each instantiation.
    3. Sets log_directory.
    4. Append optional suffix to the end of the log file name

    Optional suffix is useful when running multiple processes in parallel to 
    allow logging to separate files.
    """

    def __init__(self, log_directory: str = 'logs', log_suffix: str = None):
        """
        Args:
            log_directory (str, optional): log directory to save logs. 
                Defaults to 'logs'.
            log_suffix (str, optional): Optional suffix to add to end of log file. 
                Defaults to None.
        """
        if log_suffix is None:
            self.log_suffix = ''
        else:
             self.log_suffix = f'_{log_suffix}'
                 
        current_dir = os.getcwd()
        os.chdir(FILE_DIR)

        try:
            os.makedirs(log_directory)
        except FileExistsError:
            # log directory already exists
            pass
        
        with open('config/marmot_logging_config.yml', 'rt') as f:
            conf = yaml.safe_load(f.read())
            conf['handlers']['warning_handler']['filename'] = \
                (conf['handlers']['warning_handler']['filename']
                .format(log_directory, 'plotter', self.log_suffix))
            conf['handlers']['info_handler']['filename'] = \
                (conf['handlers']['info_handler']['filename']
                .format(log_directory, 'plotter', self.log_suffix))
            logging.config.dictConfig(conf)
            
        self.logger = logging.getLogger('marmot_plot')
        # Creates a new log file for next run 
        self.logger.handlers[1].doRollover()
        self.logger.handlers[2].doRollover()
        
        os.chdir(current_dir)


class MarmotPlot(SetupLogger):
    """Main module class to be instantiated to run the plotter.
    
    MarmotPlot handles the selection of plotting module to 
    create the desired figure and saving of outputs. 
    It also handles the area aggregation selection
    """
    
    def __init__(self, Scenarios: Union[str, list],
                 AGG_BY: str, 
                 Model_Solutions_folder: str, 
                 gen_names: Union[str, pd.DataFrame],
                 Marmot_plot_select: Union[str, pd.DataFrame], 
                 Marmot_Solutions_folder: str = None,
                 mapping_folder: str = 'mapping_folder',
                 Scenario_Diff: Union[str, list] = None,
                 zone_region_sublist: Union[str, list] = None,
                 xlabels: Union[str, list] = None,
                 ylabels: Union[str, list] = None,
                 ticklabels: Union[str, list] = None,
                 Region_Mapping: Union[str, pd.DataFrame] = pd.DataFrame(),
                 TECH_SUBSET: Union[str, list] = None,
                 **kwargs):
        """
        Args:
            Scenarios (Union[str, list]): Name of scenarios 
                to process.
            AGG_BY (str): Informs region type to aggregate by 
                when creating plots.
            Model_Solutions_folder (str): Folder containing model 
                simulation results subfolders and their files.
            gen_names (Union[str, pd.DataFrame]): Mapping file to rename 
                generator technologies.
            Marmot_plot_select (Union[str, pd.DataFrame]): Selection of plots 
                to plot.
            Marmot_Solutions_folder (str, optional): Folder to save 
                Marmot solution files.
                Defaults to None.
            mapping_folder (str, optional): The location of the 
                Marmot mapping folder.
                Defaults to 'mapping_folder'.
            Scenario_Diff (Union[str, list], optional): 2 value string 
                or list, used to compare 2 scenarios. 
                Defaults to None.
            zone_region_sublist (Union[str, list], optional): Subset of regions 
                to plot from AGG_BY. 
                Defaults to None.
            xlabels (Union[str, list], optional): x axis labels for facet plots. 
                Defaults to None.
            ylabels (Union[str, list], optional): y axis labels for facet plots. 
                Defaults to None.
            ticklabels (Union[str, list], optional): custom ticklabels for plots, 
                not available for every plot type.
                Defaults to None.
            Region_Mapping (Union[str, pd.DataFrame], optional): Mapping file 
                to map custom regions/zones to create custom aggregations. 
                Aggregations are created by grouping PLEXOS regions. 
                Defaults to pd.DataFrame().
            TECH_SUBSET (Union[str, list], optional): Tech subset category to plot.
                The TECH_SUBSET value should be a column in the 
                ordered_gen_categories.csv. If left None all techs will be plotted
                Defaults to None.
        """
        super().__init__(**kwargs) # Instantiation of SetupLogger
        
        if isinstance(Scenarios, str):
            self.Scenarios = pd.Series(Scenarios.split(",")).str.strip().tolist()
        elif isinstance(Scenarios, list):
            self.Scenarios = Scenarios
        
        self.AGG_BY = AGG_BY
        self.TECH_SUBSET = TECH_SUBSET
        self.Model_Solutions_folder = Model_Solutions_folder
        
        if isinstance(gen_names, str):
            try:
                gen_names = pd.read_csv(gen_names) 
                self.gen_names = gen_names.rename(columns=
                                                  {gen_names.columns[0]:'Original',
                                                  gen_names.columns[1]:'New'})
            except FileNotFoundError:
                self.logger.warning('Could not find specified gen_names file; '
                                    'check file name. This is required to run Marmot, '
                                    'system will now exit')
                sys.exit()
        elif isinstance(gen_names, pd.DataFrame):
            self.gen_names = gen_names.rename(columns=
                                             {gen_names.columns[0]:'Original',
                                             gen_names.columns[1]:'New'})
        
        if isinstance(Marmot_plot_select, str):
            try:
                self.Marmot_plot_select = pd.read_csv(Marmot_plot_select)   
            except FileNotFoundError:
                self.logger.warning('Could not find specified Marmot_plot_select file; '
                                    'check file name. This is required to run Marmot, '
                                    'system will now exit')
                sys.exit()
        elif isinstance(Marmot_plot_select, pd.DataFrame):
            self.Marmot_plot_select = Marmot_plot_select
        
        self.Marmot_Solutions_folder = Marmot_Solutions_folder
        
        if self.Marmot_Solutions_folder is None:
            self.Marmot_Solutions_folder = self.Model_Solutions_folder
            
        self.mapping_folder = mapping_folder
        
        if isinstance(Scenario_Diff, str):
            self.Scenario_Diff = (pd.Series(Scenario_Diff.split(","))
                                    .str.strip().tolist())
        elif isinstance(Scenario_Diff, list):
            self.Scenario_Diff = Scenario_Diff

        if Scenario_Diff == ['nan'] or Scenario_Diff is None: 
            self.Scenario_Diff = [""]
        
        if isinstance(zone_region_sublist, str):
            self.zone_region_sublist = (pd.Series(zone_region_sublist.split(","))
                                          .str.strip().tolist())
        elif isinstance(zone_region_sublist, list):
            self.zone_region_sublist = zone_region_sublist
        else:
            self.zone_region_sublist = []
        
        if isinstance(xlabels, str):
            self.xlabels = (pd.Series(xlabels.split(","))
                              .str.strip().tolist())                   
        elif isinstance(xlabels, list):
            self.xlabels = xlabels
        if xlabels == ['nan'] or xlabels is None: 
            self.xlabels = [""]
        
        if isinstance(ylabels, str):
            self.ylabels = (pd.Series(ylabels.split(","))
                              .str.strip().tolist())                                       
        elif isinstance(ylabels, list):
            self.ylabels = ylabels
        if ylabels == ['nan'] or ylabels is None:
            self.ylabels = [""]
        
        if isinstance(ticklabels, str):
            self.custom_xticklabels = (pd.Series(ticklabels.split(","))
                                         .str.strip().tolist())
        elif isinstance(ticklabels, list):
            self.custom_xticklabels = ticklabels
        if ticklabels == ['nan'] or ticklabels is None:
            self.custom_xticklabels = [""]

        if isinstance(Region_Mapping, str):
            try:
                self.Region_Mapping = pd.read_csv(Region_Mapping)
                if not self.Region_Mapping.empty:  
                    self.Region_Mapping = self.Region_Mapping.astype(str)
            except FileNotFoundError:
                self.logger.warning('Could not find specified Region Mapping file; '
                                    'check file name\n')
                self.Region_Mapping = pd.DataFrame()
        elif isinstance(Region_Mapping, pd.DataFrame):
            self.Region_Mapping = Region_Mapping
            if not self.Region_Mapping.empty:           
                self.Region_Mapping = self.Region_Mapping.astype(str)
        try:
            # delete category columns if exists
            self.Region_Mapping = self.Region_Mapping.drop(["category"],axis=1) 
        except KeyError:
            pass     
                        
    def run_plotter(self):
        """Main method to call to begin plotting figures. 
        
        This method takes no input variables, all required 
        variables are passed in via the __init__ method.
        """
        
        self.logger.info(f"Area Aggregation selected: {self.AGG_BY}")
        
        if self.zone_region_sublist != ['nan'] and \
        self.zone_region_sublist !=[]:
            self.logger.info(f"Only plotting {self.AGG_BY}: "
                            f"{self.zone_region_sublist}")

        metadata_HDF5_folder_in = os.path.join(self.Marmot_Solutions_folder,
                                               'Processed_HDF5_folder')
        
        figure_format = mconfig.parser("figure_file_format")
        if figure_format == 'nan':
            figure_format = 'png'
        
        shift_leapday = str(mconfig.parser("shift_leapday")).upper()
        font_defaults = mconfig.parser("font_settings")
        text_position = mconfig.parser("text_position")
        
        #================================================================================
        # Input and Output Directories
        #================================================================================
        
        figure_folder = os.path.join(self.Marmot_Solutions_folder,
                                     'Figures_Output')
        try:
            os.makedirs(figure_folder)
        except FileExistsError:
            # directory already exists
            pass
        
        hdf_out_folder = os.path.join(self.Marmot_Solutions_folder,
                                      'Processed_HDF5_folder')
        
        #================================================================================
        # Standard Generation Order, Gen Categorization Lists, Plotting Colors
        #================================================================================
        
        try:
            ordered_gen_categories = pd.read_csv(os.path.join(self.mapping_folder,
                                                 mconfig.parser('ordered_gen_categories')))
        except FileNotFoundError:
            self.logger.warning('Could not find '
                                f'"{os.path.join(self.mapping_folder, "ordered_gen.csv")}"; '
                                'Check file name in config file. This is required to '
                                'run Marmot, system will now exit')
            sys.exit()
        
        if (set(self.gen_names["New"].unique())
        .issubset(ordered_gen_categories['Ordered_Gen'].str.strip().tolist())) == False:
            missing_gen = (set(self.gen_names.New.unique()) 
                           - (set(ordered_gen_categories['Ordered_Gen'].str
                                                                       .strip()
                                                                       .tolist())))
            self.logger.warning("The following tech categories from the "
                                "gen_names csv do not exist in "
                                "ordered_gen_categories.csv!: "
                                f"{missing_gen}")
            
        # Subset ordered_gen to user desired generation
        if self.TECH_SUBSET:
            if self.TECH_SUBSET not in ordered_gen_categories.columns:
                self.logger.warning(f"{self.TECH_SUBSET} column was not found "
                                    "in the ordered_gen_categories.csv. "
                                    "All generator technologies will be plotted")
                ordered_gen = (ordered_gen_categories['Ordered_Gen'].str
                                                                    .strip()
                                                                    .tolist())
            else:
                ordered_gen = ordered_gen_categories.loc[
                                ordered_gen_categories[self.TECH_SUBSET] == True]
                ordered_gen = ordered_gen['Ordered_Gen'].str.strip().tolist()
                self.logger.info(f"Tech Aggregation selected: {self.TECH_SUBSET}")
        else:
            ordered_gen = ordered_gen_categories['Ordered_Gen'].str.strip().tolist()
        
        # If Other category does not exist in ordered_gen, create entry 
        if 'Other' not in ordered_gen:
            ordered_gen.append('Other')

        if 'pv' not in ordered_gen_categories.columns:
            pv_gen_cat = []
            self.logger.warning('"pv" column was not found in the '
                                'ordered_gen_categories.csv. Check if the column '
                                'exists in the csv file. This is required for '
                                'certain plots to display correctly')
        else:
            pv_gen_cat = ordered_gen_categories.loc[
                            ordered_gen_categories['pv'] == True]
            pv_gen_cat = pv_gen_cat['Ordered_Gen'].str.strip().tolist()
            
        if 're' not in ordered_gen_categories.columns:
            re_gen_cat = []
            self.logger.warning('"re" column was not found in the '
                                'ordered_gen_categories.csv. Check if the column '
                                'exists in the csv file. This is required for '
                                'certain plots to display correctly')
        else:
            re_gen_cat = ordered_gen_categories.loc[
                            ordered_gen_categories['re'] == True]
            re_gen_cat = re_gen_cat['Ordered_Gen'].str.strip().tolist()

        if 'vre' not in ordered_gen_categories.columns:
            vre_gen_cat = []
            self.logger.warning('"vre" column was not found in the '
                                'ordered_gen_categories.csv. Check if the column '
                                'exists in the csv file. This is required for '
                                'certain plots to display correctly')
        else:
            vre_gen_cat = ordered_gen_categories.loc[
                            ordered_gen_categories['vre'] == True]
            vre_gen_cat = vre_gen_cat['Ordered_Gen'].str.strip().tolist()

        if 'thermal' not in ordered_gen_categories.columns:
            thermal_gen_cat = []
            self.logger.warning('"thermal" column was not found in the '
                                'ordered_gen_categories.csv. Check if the column '
                                'exists in the csv file. This is required for '
                                'certain plots to display correctly')
        else:
            thermal_gen_cat = ordered_gen_categories.loc[
                                ordered_gen_categories['thermal'] == True]
            thermal_gen_cat = thermal_gen_cat['Ordered_Gen'].str.strip().tolist()
        
        try:
            PLEXOS_color_dict = pd.read_csv(os.path.join(self.mapping_folder,
                                            mconfig.parser('color_dictionary_file')))

            PLEXOS_color_dict = PLEXOS_color_dict.rename(columns=
                                                         {PLEXOS_color_dict.columns[0]: 
                                                         'Generator',
                                                         PLEXOS_color_dict.columns[1]:
                                                         'Colour'})

            PLEXOS_color_dict["Generator"] = (PLEXOS_color_dict["Generator"].str
                                                                            .strip())
            PLEXOS_color_dict["Colour"] = PLEXOS_color_dict["Colour"].str.strip()
            PLEXOS_color_dict = (PLEXOS_color_dict[['Generator','Colour']].set_index("Generator")
                                                                          .to_dict()["Colour"])
        except FileNotFoundError:
            self.logger.warning('Could not find '
                                f'"{os.path.join(self.mapping_folder, "colour_dictionary.csv")}"; '
                                'Check file name in config file. Random colors will now be used')
            cmap = plt.cm.get_cmap(lut=len(ordered_gen))
            colors = []
            for i in range(cmap.N):
                colors.append(mpl.colors.rgb2hex(cmap(i)))  
            PLEXOS_color_dict = dict(zip(ordered_gen,colors))
        
        
        color_list = ['#396AB1', '#CC2529','#3E9651','#ff7f00','#6B4C9A',
                      '#922428','#cab2d6', '#6a3d9a', '#fb9a99', '#b15928']
        marker_style = ["^", "*", "o", "D", "x", "<", "P", "H", "8", "+"]
        
        gen_names_dict=(self.gen_names[['Original','New']]
                            .set_index("Original")
                            .to_dict()["New"])

        # If curtailment category does not exist in ordered_gen, create entry 
        curtailment_name = gen_names_dict.get('Curtailment','Curtailment')
        if curtailment_name not in ordered_gen:
            ordered_gen.append(curtailment_name)

        #================================================================================
        # Set aggregation
        #================================================================================
        
        # Create an instance of MetaData.
        meta = MetaData(metadata_HDF5_folder_in, Region_Mapping=self.Region_Mapping)
        
        if self.AGG_BY in {"zone", "zones", "Zone", "Zones"}:
            self.AGG_BY = 'zone'
            zones = pd.concat([meta.zones(scenario) for scenario in self.Scenarios])
            if zones.empty == True:
                self.logger.warning("Input Sheet Data Incorrect! Your model does "
                                    "not contain Zones, enter a different aggregation")
                sys.exit()
            Zones = zones['name'].unique()
        
            if self.zone_region_sublist != ['nan'] and self.zone_region_sublist !=[]:
                zsub = []
                for zone in self.zone_region_sublist:
                    if zone in Zones:
                        zsub.append(zone)
                    else:
                        self.logger.info("metadata does not contain zone: "
                                         f"{zone}, SKIPPING ZONE")
                if zsub:
                    Zones = zsub
                else:
                    self.logger.warning(f"None of: {self.zone_region_sublist} "
                                        "in model Zones. Plotting all Zones")
        
        elif self.AGG_BY in {"region", "regions", "Region", "Regions"}:
            self.AGG_BY = 'region'
            regions = pd.concat([meta.regions(scenario) for scenario in self.Scenarios])
            if regions.empty == True:
                self.logger.warning("Input Sheet Data Incorrect! Your model does "
                                    "not contain Regions, enter a different aggregation")
                sys.exit()

            Zones = regions['region'].unique()
            if self.zone_region_sublist != ['nan'] and self.zone_region_sublist !=[]:
                zsub = []
                for region in self.zone_region_sublist:
                    if region in Zones:
                        zsub.append(region)
                    else:
                        self.logger.info("metadata does not contain region: "
                                         f"{region}, SKIPPING REGION")
                if zsub:
                    Zones = zsub
                else:
                    self.logger.warning(f"None of: {self.zone_region_sublist} "
                                        "in model Regions. Plotting all Regions")
        
        elif not self.Region_Mapping.empty:
            self.logger.info("Plotting Custom region aggregation from "
                             "Region_Mapping File")
            regions = pd.concat([meta.regions(scenario) for scenario in self.Scenarios])
            self.Region_Mapping = regions.merge(self.Region_Mapping, 
                                                how='left', 
                                                on='region')
            self.Region_Mapping.dropna(axis=1, how='all', inplace=True)
            
            try:
                Zones = self.Region_Mapping[self.AGG_BY].unique()
            except KeyError:
                self.logger.warning(f"AGG_BY = '{self.AGG_BY}' is not in the "
                                    "Region_Mapping File, enter a different aggregation")
                sys.exit()
            
            # remove any nan that might end  up in list
            Zones = [x for x in Zones if str(x) != 'nan']
            
            if self.zone_region_sublist != ['nan'] and self.zone_region_sublist !=[]:
                zsub = []
                for region in self.zone_region_sublist:
                    if region in Zones:
                        zsub.append(region)
                    else:
                        self.logger.info("Region_Mapping File does not contain region: "
                                         f"{region}, SKIPPING REGION")
                if zsub:
                    Zones = zsub
                else:
                    self.logger.warning(f"None of: {self.zone_region_sublist} "
                                        "in Region_Mapping File. Plotting all "
                                        f"Regions of aggregation '{self.AGG_BY}'")
        else:
            self.logger.warning("AGG_BY is not defined correctly, aggregation "
                                "specified was not found, system will now exit")
            sys.exit()
        
        #================================================================================
        # Start Main plotting loop
        #================================================================================
        
        # Filter for chosen figures to plot
        plot_selection = self.Marmot_plot_select.loc[
                            self.Marmot_plot_select["Plot Graph"] == True]
        plot_selection = plot_selection.sort_values(by=['Marmot Module','Method'])
        
        list_modules = plot_selection['Marmot Module'].unique()
        
        start_timer = time.time()
        
        for module in list_modules:
            module_plots = plot_selection.loc[plot_selection['Marmot Module'] == module]
            # dictionary of arguments passed to plotting modules; 
            # key names match the instance variables in each module            
            argument_dict = {
                "hdf_out_folder": hdf_out_folder,
                "Zones": Zones,
                "AGG_BY": self.AGG_BY,
                "ordered_gen": ordered_gen,
                "PLEXOS_color_dict": PLEXOS_color_dict,
                "Scenarios": self.Scenarios,
                "Scenario_Diff": self.Scenario_Diff,
                "Marmot_Solutions_folder": self.Marmot_Solutions_folder,
                "ylabels": self.ylabels,
                "xlabels": self.xlabels,
                "custom_xticklabels": self.custom_xticklabels,
                "color_list": color_list,
                "marker_style": marker_style,
                "gen_names_dict": gen_names_dict,
                "pv_gen_cat": pv_gen_cat,
                "re_gen_cat": re_gen_cat,
                "vre_gen_cat": vre_gen_cat,
                "thermal_gen_cat": thermal_gen_cat,
                "Region_Mapping": self.Region_Mapping,
                "figure_folder": figure_folder,
                "meta": meta,
                "shift_leapday": shift_leapday
                }
            
            # Create output folder for each plotting module
            figures = os.path.join(figure_folder, f"{self.AGG_BY}_{module}")
            try:
                os.makedirs(figures)
            except FileExistsError:
                pass
            
            # Import plot module from plottingmodules package
            plot_module = importlib.import_module('marmot.plottingmodules.' + module)
            # Instantiate the module class
            instantiate_mplot = plot_module.MPlot(argument_dict)
            
            # Main loop to process each figure and pass 
            # plot specific variables to methods
            for index, row in module_plots.iterrows(): 
                
                # Set Plot defaults
                mpl.rc('xtick', labelsize=font_defaults['xtick_size'])
                mpl.rc('ytick', labelsize=font_defaults['ytick_size'])
                mpl.rc('axes', labelsize=font_defaults['axes_label_size'])
                mpl.rc('legend', fontsize=font_defaults['legend_size'])
                mpl.rc('font', family=font_defaults['font_family'])
                mpl.rc('figure', max_open_warning = 0)
                mpl.rc('axes', titlesize=font_defaults['title_size'], 
                       titlepad=text_position['title_height'])
                
                print("\n\n\n")
                self.logger.info(f"Plot =  {row['Figure Output Name']}")
                
                # Modifies timezone string before plotting
                if pd.isna(row.iloc[6]):
                    row.iloc[6] = "Date"
                else:
                    row.iloc[6] = f"Date ({row.iloc[6]})"

                # Get figure method and run plot
                figure_method = getattr(instantiate_mplot, row['Method'])
                Figure_Out = figure_method(figure_name = row.iloc[0], 
                                           prop = row.iloc[2],
                                           y_axis_max = float(row.iloc[3]),
                                           start = float(row.iloc[4]),
                                           end = float(row.iloc[5]),
                                           timezone = row.iloc[6],
                                           start_date_range = row.iloc[7],
                                           end_date_range = row.iloc[8],
                                           custom_data_file_path = row['Custom Data File'])
                
                if isinstance(Figure_Out, MissingInputData):
                    self.logger.info("Add Inputs With Formatter Before "
                                     "Attempting to Plot!\n")
                    continue
                
                if isinstance(Figure_Out, DataSavedInModule):
                    self.logger.info(f'Plotting Completed for '
                                     f'{row["Figure Output Name"]}\n')
                    self.logger.info("Plots & Data Saved Within Module!\n")
                    continue
                
                if isinstance(Figure_Out, UnderDevelopment):
                    self.logger.info("Plot is Under Development, Plotting Skipped!\n")
                    continue
                
                if isinstance(Figure_Out, InputSheetError):
                    self.logger.info("Input Sheet Data Incorrect!\n")
                    continue
                
                if isinstance(Figure_Out, MissingMetaData):
                    self.logger.info("Required Meta Data Not Available For "
                                     "This Plot!\n")
                    continue
                
                if isinstance(Figure_Out, UnsupportedAggregation):
                    self.logger.info(f"Aggregation Type: '{self.AGG_BY}' "
                                     "not supported for This plot!\n")
                    continue
                
                for zone_input in Zones:
                    if isinstance(Figure_Out[zone_input], MissingZoneData):

                        self.logger.info(f"No Data to Plot in {zone_input}")

                    else:
                        # Save figures
                        Figure_Out[zone_input]["fig"].savefig(os.path.join(figures,
                                                              f'{zone_input}_'
                                                              f'{row["Figure Output Name"]}'
                                                              f'.{figure_format}'),
                                                              dpi=600,
                                                              bbox_inches='tight')

                        # Save .csv's.
                        if Figure_Out[zone_input]['data_table'].empty:
                            self.logger.info(f'{row["Figure Output Name"]} '
                                             'does not return a data table')
                        else:
                            Figure_Out[zone_input]["data_table"].to_csv(os.path.join(figures,
                                                                        f'{zone_input}_'
                                                                        f'{row["Figure Output Name"]}.csv'))

                self.logger.info('Plotting Completed for '
                                 f'{row["Figure Output Name"]}\n')

                mpl.pyplot.close('all')
        
        end_timer = time.time()
        time_elapsed = end_timer - start_timer
        self.logger.info(f'Main Plotting loop took {round(time_elapsed/60,2)} minutes')
        self.logger.info('All Plotting COMPLETED')
        meta.close_h5()


def main():
    """Run the plotting code and create desired plots and data-tables based on user input files.
    """
    #====================================================================================
    # Load Input Properties
    #====================================================================================
    
    #changes working directory to location of this python file
    os.chdir(FILE_DIR)

    Marmot_user_defined_inputs = pd.read_csv(mconfig.parser("user_defined_inputs_file"), 
                                             usecols=['Input','User_defined_value'],
                                             index_col='Input', skipinitialspace=True)

    Marmot_plot_select = pd.read_csv(mconfig.parser("plot_select_file"))
    
    # Folder to save your processed solutions
    if pd.isna(Marmot_user_defined_inputs.loc['Marmot_Solutions_folder',
                                              'User_defined_value']):
        Marmot_Solutions_folder = None
    else:
        Marmot_Solutions_folder = Marmot_user_defined_inputs.loc[
                                    'Marmot_Solutions_folder'].to_string(index=False).strip()
        
    Scenarios = pd.Series(Marmot_user_defined_inputs.loc[
                            'Scenarios'].squeeze().split(",")).str.strip().tolist()
    
    # These variables (along with Region_Mapping) are used to initialize MetaData
    Model_Solutions_folder = Marmot_user_defined_inputs.loc[
                                'Model_Solutions_folder'].to_string(index=False).strip()
    
    # For plots using the difference of the values between two scenarios.
    # Max two entries, the second scenario is subtracted from the first.
    Scenario_Diff = pd.Series(str(Marmot_user_defined_inputs.loc[
                            'Scenario_Diff_plot'].squeeze()).split(",")).str.strip().tolist()
    if Scenario_Diff == ['nan']: Scenario_Diff = [""]
    
    Mapping_folder = 'mapping_folder'
    
    if pd.isna(Marmot_user_defined_inputs.loc['Region_Mapping.csv_name', 
                                              'User_defined_value']) is True:
        Region_Mapping = pd.DataFrame()
    else:
        Region_Mapping = pd.read_csv(os.path.join(Mapping_folder, 
                                     Marmot_user_defined_inputs.loc[
                                         'Region_Mapping.csv_name'].to_string(index=False).strip()))

        Region_Mapping = Region_Mapping.astype(str)
    
    gen_names = pd.read_csv(os.path.join(Mapping_folder, 
                            Marmot_user_defined_inputs.loc[
                                'gen_names.csv_name'].to_string(index=False).strip()))
    
    AGG_BY = Marmot_user_defined_inputs.loc['AGG_BY'].squeeze().strip()

    if pd.notna(Marmot_user_defined_inputs.loc['TECH_SUBSET', 'User_defined_value']):
        TECH_SUBSET = Marmot_user_defined_inputs.loc['TECH_SUBSET'].squeeze().strip()
    else:
        TECH_SUBSET = None

    # Facet Grid Labels (Based on Scenarios)
    zone_region_sublist = pd.Series(str(Marmot_user_defined_inputs.loc[
                                'zone_region_sublist'].squeeze()).split(",")).str.strip().tolist()

    ylabels = pd.Series(str(Marmot_user_defined_inputs.loc[
                    'Facet_ylabels'].squeeze()).split(",")).str.strip().tolist()
    if ylabels == ['nan']: ylabels = [""]
    xlabels = pd.Series(str(Marmot_user_defined_inputs.loc[
                    'Facet_xlabels'].squeeze()).split(",")).str.strip().tolist()
    if xlabels == ['nan']: xlabels = [""]
    
    # option to change tick labels on plot
    ticklabels = pd.Series(str(Marmot_user_defined_inputs.loc[
                    'Tick_labels'].squeeze()).split(",")).str.strip().tolist()
    if ticklabels == ['nan']: ticklabels = [""]
    
    initiate = MarmotPlot(Scenarios, AGG_BY, Model_Solutions_folder, 
                          gen_names, Marmot_plot_select,
                          Marmot_Solutions_folder=Marmot_Solutions_folder,
                          mapping_folder=Mapping_folder,
                          Scenario_Diff=Scenario_Diff,
                          zone_region_sublist=zone_region_sublist,
                          xlabels=xlabels,
                          ylabels=ylabels,
                          ticklabels=ticklabels,
                          Region_Mapping=Region_Mapping,
                          TECH_SUBSET=TECH_SUBSET)
    
    initiate.run_plotter()


if __name__ == '__main__':
    main()
    
