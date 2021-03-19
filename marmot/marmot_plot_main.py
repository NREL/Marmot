# -*- coding: utf-8 -*-
"""
First Created on Thu Dec  5 14:16:30 2019

marmot_plot_main.py is the main plotting script within Marmot which calls on supporting files to read in data, 
create the plot, and then return the plot and data to marmot_plot_main.py. 
The supporting modules can be viewed within the repo plottingmodules folder and 
have descriptive names such as total_generation.py, generation_stack.py, curtaiment.py etc.

@author: Daniel Levie
"""

#===============================================================================
# Import Python Libraries
#===============================================================================

import os
import sys
import pathlib
import pandas as pd
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
import importlib
import logging 
import logging.config
import yaml
from marmot.meta_data import MetaData
import marmot.plottingmodules.marmot_plot_functions as mfunc
import marmot.config.mconfig as mconfig

#===============================================================================
# Setup Logger
#===============================================================================

current_dir = os.getcwd()
os.chdir(pathlib.Path(__file__).parent.absolute())

with open('config/marmot_logging_config.yml', 'rt') as f:
    conf = yaml.safe_load(f.read())
    logging.config.dictConfig(conf)
    
logger = logging.getLogger('marmot_plot')
# Creates a new log file for next run 
logger.handlers[1].doRollover()
logger.handlers[2].doRollover()

os.chdir(current_dir)

#A bug in pandas requires this to be included, otherwise df.to_string truncates long strings
#Fix available in Pandas 1.0 but leaving here in case user version not up to date
pd.set_option("display.max_colwidth", 1000)


class PlotTypes:
    '''
    Class to handle the selection of plottingmodule and correct method for specifed figure
    '''

    def __init__(self, figure_type, figure_method, argument_dict, font_defaults):
        '''
        
        Parameters
        ----------
        figure_type : string
            figure module name.
        figure_method : string
            figure method name.
        argument_dict : dictionary
            dictioanry of input variables for plottingmodules.
        font_defaults : dictionary
            default font settings passed from config file.

        Returns
        -------
        None.

        '''
        self.figure_type = figure_type
        self.figure_method = figure_method
        self.argument_dict = argument_dict
        self.font_defaults = font_defaults


    def runmplot(self):
        mpl.rc('xtick', labelsize=self.font_defaults['xtick_size'])
        mpl.rc('ytick', labelsize=self.font_defaults['ytick_size'])
        mpl.rc('axes', labelsize=self.font_defaults['axes_label_size'])
        mpl.rc('legend', fontsize=self.font_defaults['legend_size'])
        mpl.rc('font', family=self.font_defaults['font_family'])
        mpl.rc('figure', max_open_warning = 0)
        
        # Import plot module from plottingmodules package
        plot = importlib.import_module('marmot.plottingmodules.' + self.figure_type)
        fig = plot.mplot(self.argument_dict)

        process_attr = getattr(fig, self.figure_method)

        Figure_Out = process_attr()
        return Figure_Out


class MarmotPlot():
    
    def __init__(self,Scenarios, AGG_BY, PLEXOS_Solutions_folder, gen_names, Marmot_plot_select, 
                 Marmot_Solutions_folder=None,
                 marmot_mapping_folder='mapping_folder',Scenario_Diff=[],
                 zone_region_sublist=[],xlabels=[], ylabels=[],ticklabels=[],
                 Region_Mapping=pd.DataFrame()):
        '''

        Parameters
        ----------
        Scenarios : string/list
            Name of scenarios to process.
        AGG_BY : string
            Informs region type to aggregate by when creating plots.
        PLEXOS_Solutions_folder : string directory
            Folder containing h5plexos results files.
        gen_names : string directory/pd.DataFrame
            Mapping file to rename generator technologies.
        Marmot_plot_select : string directory/pd.DataFrame
            selection of plots to plot.
        Marmot_Solutions_folder : string directory, optional
            Folder to save Marmot solution files. The default is None.
        marmot_mapping_folder : string directory, optional
            The location of the Marmot mapping folder. The default is 'mapping_folder'.
        Scenario_Diff : string/list, optional
            2 value string or list, used to compare 2 sceanrios. The default is [].
        zone_region_sublist : string/list, optional
            subset of regions to plot from AGG_BY. The default is [].
        xlabels : string/list, optional
            x axis labels for facet plots. The default is [].
        ylabels : string/list, optional
            y axis labels for facet plots. The default is [].
        ticklabels : string/list, optional
            custom ticklabels for plots, not available for every plot type. The default is [].
        Region_Mapping : string directory/pd.DataFrame, optional
            Mapping file to map custom regions/zones to create custom aggregations. 
            Aggregations are created by grouping PLEXOS regions.
            The default is pd.DataFrame().

        Returns
        -------
        None.

        '''
        
        if isinstance(Scenarios, str):
            self.Scenarios = pd.Series(Scenarios.split(",")).str.strip().tolist()
        elif isinstance(Scenarios, list):
            self.Scenarios = Scenarios
        
        self.AGG_BY = AGG_BY
        self.PLEXOS_Solutions_folder = PLEXOS_Solutions_folder
        
        if isinstance(gen_names, str):
            try:
                self.gen_names = pd.read_csv(gen_names)   
            except FileNotFoundError:
                logger.warning('Could not find specified gen_names file; check file name. This is required to run Marmot, system will now exit')
                sys.exit()
        elif isinstance(gen_names, pd.DataFrame):
            self.gen_names = gen_names.rename(columns={gen_names.columns[0]:'Original',gen_names.columns[1]:'New'})
            self.gen_names_dict=self.gen_names[['Original','New']].set_index("Original").to_dict()["New"]
        
        if isinstance(Marmot_plot_select, str):
            try:
                self.Marmot_plot_select = pd.read_csv(Marmot_plot_select)   
            except FileNotFoundError:
                logger.warning('Could not find specified Marmot_plot_select file; check file name. This is required to run Marmot, system will now exit')
                sys.exit()
        elif isinstance(Marmot_plot_select, pd.DataFrame):
            self.Marmot_plot_select = Marmot_plot_select
        
        self.Marmot_Solutions_folder = Marmot_Solutions_folder
        
        if self.Marmot_Solutions_folder == None:
            self.Marmot_Solutions_folder = self.PLEXOS_Solutions_folder
            
        self.marmot_mapping_folder = marmot_mapping_folder
        
        if isinstance(Scenario_Diff, str):
            self.Scenario_Diff = pd.Series(Scenario_Diff.split(",")).str.strip().tolist() 
        elif isinstance(Scenario_Diff, list):
            self.Scenario_Diff = Scenario_Diff
        if self.Scenario_Diff == ['nan'] or self.Scenario_Diff == [] : self.Scenario_Diff = [""]
        
        if isinstance(zone_region_sublist, str):
            self.zone_region_sublist = pd.Series(zone_region_sublist.split(",")).str.strip().tolist()
        elif isinstance(zone_region_sublist, list):
            self.zone_region_sublist = zone_region_sublist
        
        if isinstance(xlabels, str):
            self.xlabels = pd.Series(xlabels.split(",")).str.strip().tolist()
        elif isinstance(xlabels, list):
            self.xlabels = xlabels
        if self.xlabels == ['nan'] or self.xlabels == [] : self.xlabels = [""]
        
        if isinstance(ylabels, str):
            self.ylabels = pd.Series(ylabels.split(",")).str.strip().tolist()
        elif isinstance(ylabels, list):
            self.ylabels = ylabels
        if self.ylabels == ['nan'] or self.ylabels == [] : self.ylabels = [""]
        
        if isinstance(ticklabels, str):
            self.ticklabels = pd.Series(ticklabels.split(",")).str.strip().tolist()
        elif isinstance(ticklabels, list):
            self.ticklabels = ticklabels
        if self.ticklabels == ['nan'] or self.ticklabels == [] : self.ticklabels = [""]
        
        if isinstance(Region_Mapping, str):
            try:
                self.Region_Mapping = pd.read_csv(Region_Mapping)
                if not self.Region_Mapping.empty:  
                    self.Region_Mapping = self.Region_Mapping.astype(str)
            except FileNotFoundError:
                logger.warning('Could not find specified Region Mapping file; check file name\n')
                self.Region_Mapping = pd.DataFrame()
        elif isinstance(Region_Mapping, pd.DataFrame):
            self.Region_Mapping = Region_Mapping
            if not self.Region_Mapping.empty:           
                self.Region_Mapping = self.Region_Mapping.astype(str)
        try:
            self.Region_Mapping = self.Region_Mapping.drop(["category"],axis=1) # delete category columns if exists
        except KeyError:
            pass     
                        
        
    def run_plotter(self):
        '''
        Main method to call to begin plotting figures, this method takes 
        no input variables, all required varibales are passed in via the __init__ method.

        Returns
        -------
        None.

        '''
        
        logger.info(f"Aggregation selected: {self.AGG_BY}")
        
        if self.zone_region_sublist != ['nan'] and self.zone_region_sublist !=[]:
            logger.info(f"Only plotting {self.AGG_BY}: {self.zone_region_sublist}")

        metadata_HDF5_folder_in = os.path.join(self.PLEXOS_Solutions_folder, self.Scenarios[0])
        
        figure_format = mconfig.parser("figure_file_format")
        if figure_format == 'nan':
            figure_format = 'png'
        
        shift_leap_day = str(mconfig.parser("shift_leap_day")).upper()
        font_defaults = mconfig.parser("font_settings")

        #===============================================================================
        # Input and Output Directories
        #===============================================================================
        
        figure_folder = os.path.join(self.Marmot_Solutions_folder,'Figures_Output')
        try:
            os.makedirs(figure_folder)
        except FileExistsError:
            # directory already exists
            pass
        
        hdf_out_folder = os.path.join(self.Marmot_Solutions_folder,'Processed_HDF5_folder')
        try:
            os.makedirs(hdf_out_folder)
        except FileExistsError:
            # directory already exists
            pass
        
        #===============================================================================
        # Standard Generation Order, Gen Categorization Lists, Plotting Colors
        #===============================================================================
        
        try:
            ordered_gen = pd.read_csv(os.path.join(self.marmot_mapping_folder, 'ordered_gen.csv'),squeeze=True).str.strip().tolist()
        except FileNotFoundError:
            logger.warning(f'Could not find "{os.path.join(self.marmot_mapping_folder, "ordered_gen.csv")}"; Check file name. This is required to run Marmot, system will now exit')
            sys.exit()
        
        try:
            pv_gen_cat = pd.read_csv(os.path.join(self.marmot_mapping_folder, 'pv_gen_cat.csv'),squeeze=True).str.strip().tolist()
        except FileNotFoundError:
            logger.warning(f'Could not find "{os.path.join(self.marmot_mapping_folder, "pv_gen_cat.csv")}"; Check file name. This is required for certain plots to display correctly')
            pv_gen_cat = []
        
        try:
            re_gen_cat = pd.read_csv(os.path.join(self.marmot_mapping_folder, 're_gen_cat.csv'),squeeze=True).str.strip().tolist()
        except FileNotFoundError:
            logger.warning(f'Could not find "{os.path.join(self.marmot_mapping_folder, "re_gen_cat.csv")}"; Check file name. This is required for certain plots to display correctly')
            re_gen_cat = []
        
        try:
            vre_gen_cat = pd.read_csv(os.path.join(self.marmot_mapping_folder, 'vre_gen_cat.csv'),squeeze=True).str.strip().tolist()
        except FileNotFoundError:
            logger.warning(f'Could not find "{os.path.join(self.marmot_mapping_folder, "vre_gen_cat.csv")}"; Check file name. This is required for certain plots to display correctly')
            vre_gen_cat = []
        
        try:
            thermal_gen_cat = pd.read_csv(os.path.join(self.marmot_mapping_folder, 'thermal_gen_cat.csv'), squeeze = True).str.strip().tolist()
        except FileNotFoundError:
            logger.warning(f'Could not find "{os.path.join(self.marmot_mapping_folder, "thermal_gen_cat.csv")}"; Check file name. This is required for certain plots to display correctly')
            thermal_gen_cat = []
        
        
        if set(self.gen_names["New"].unique()).issubset(ordered_gen) == False:
                            logger.warning(f"The new categories from the gen_names csv do not exist in ordered_gen!:{set(self.gen_names.New.unique()) - (set(ordered_gen))}")
        
        try:
            PLEXOS_color_dict = pd.read_csv(os.path.join(self.marmot_mapping_folder, 'colour_dictionary.csv'))
            PLEXOS_color_dict["Generator"] = PLEXOS_color_dict["Generator"].str.strip()
            PLEXOS_color_dict["Colour"] = PLEXOS_color_dict["Colour"].str.strip()
            PLEXOS_color_dict = PLEXOS_color_dict[['Generator','Colour']].set_index("Generator").to_dict()["Colour"]
        except FileNotFoundError:
            logger.warning(f'Could not find "{os.path.join(self.marmot_mapping_folder, "colour_dictionary.csv")}"; Check file name. Random colors will now be used')
            cmap = plt.cm.get_cmap(lut=len(ordered_gen))
            colors = []
            for i in range(cmap.N):
                colors.append(mpl.colors.rgb2hex(cmap(i)))  
            PLEXOS_color_dict = dict(zip(ordered_gen,colors))
        
        
        color_list = ['#396AB1', '#CC2529','#3E9651','#ff7f00','#6B4C9A','#922428','#cab2d6', '#6a3d9a', '#fb9a99', '#b15928']
        marker_style = ["^", "*", "o", "D", "x", "<", "P", "H", "8", "+"]
        
        gen_names_dict=self.gen_names[['Original','New']].set_index("Original").to_dict()["New"]
        
        #===============================================================================
        # Set aggregation
        #===============================================================================
        
        # Create an instance of MetaData, and pass that as a variable to get data.
        meta = MetaData(metadata_HDF5_folder_in, self.Region_Mapping)
        
        if self.AGG_BY in {"zone", "zones", "Zone", "Zones"}:
            self.AGG_BY = 'zone'
            zones = meta.zones()
            if zones.empty == True:
                logger.warning("Input Sheet Data Incorrect! Your model does not contain Zones, enter a different aggregation")
                sys.exit()
            Zones = zones['name'].unique()
        
            if self.zone_region_sublist != ['nan']:
                zsub = []
                for zone in self.zone_region_sublist:
                    if zone in Zones:
                        zsub.append(zone)
                    else:
                        logger.info(f"metadata does not contain zone: {zone}, SKIPPING ZONE")
                if zsub:
                    Zones = zsub
                else:
                    logger.warning(f"None of: {self.zone_region_sublist} in model Zones. Plotting all Zones")
        
        elif self.AGG_BY in {"region", "regions", "Region", "Regions"}:
            self.AGG_BY = 'region'
            regions = meta.regions()
            if regions.empty == True:
                logger.warning("Input Sheet Data Incorrect! Your model does not contain Regions, enter a different aggregation")
                sys.exit()
            Zones = regions['region'].unique()
            if self.zone_region_sublist != ['nan']:
                zsub = []
                for region in self.zone_region_sublist:
                    if region in Zones:
                        zsub.append(region)
                    else:
                        logger.info(f"metadata does not contain region: {region}, SKIPPING REGION")
                if zsub:
                    Zones = zsub
                else:
                    logger.warning(f"None of: {self.zone_region_sublist} in model Regions. Plotting all Regions")
        
        elif not self.Region_Mapping.empty:
            logger.info("Plotting Custom region aggregation from Region_Mapping File")
            regions = meta.regions()
            self.Region_Mapping = regions.merge(self.Region_Mapping, how='left', on='region')
            self.Region_Mapping.dropna(axis=1, how='all', inplace=True)
            
            try:
                Zones = self.Region_Mapping[self.AGG_BY].unique()
            except KeyError:
                logger.warning(f"AGG_BY = '{self.AGG_BY}' is not in the Region_Mapping File, enter a different aggregation")
                sys.exit()
            
            # remove any nan that might end  up in list
            Zones = [x for x in Zones if str(x) != 'nan']
            
            if self.zone_region_sublist != ['nan']:
                zsub = []
                for region in self.zone_region_sublist:
                    if region in Zones:
                        zsub.append(region)
                    else:
                        logger.info(f"Region_Mapping File does not contain region: {region}, SKIPPING REGION")
                if zsub:
                    Zones = zsub
                else:
                    logger.warning(f"None of: {self.zone_region_sublist} in Region_Mapping File. Plotting all Regions of aggregation '{self.AGG_BY}'")
        else:
            logger.warning("AGG_BY is not defined correctly, aggregation specified was not found, system will now exit")
            sys.exit()
        
        #===============================================================================
        # Start Main plotting loop
        #===============================================================================
        
        # Filter for chosen figures to plot
        plot_selection = self.Marmot_plot_select.loc[self.Marmot_plot_select["Plot Graph"] == True]
        
        start_timer = time.time()
        # Main loop to process each figure and pass data to functions
        for index, row in plot_selection.iterrows(): 
        
            print("\n\n\n")
            logger.info(f"Plot =  {row['Figure Output Name']}")
        
            module = row['Marmot Module']
            method = row['Method']
        
            facet = False
            if 'Facet' in row["Figure Output Name"]:
                facet = True
        
            duration_curve = False
            if 'duration_curve' in row["Figure Output Name"]:
            	duration_curve = True
            # dictionary of arguments passed to plotting modules; key_list names match the property names in each module
            # while arguments contains the property value
            key_list = ["prop", "start", "end", "timezone", "start_date", "end_date",
                        "hdf_out_folder", "Zones", "AGG_BY", "ordered_gen", "PLEXOS_color_dict",
                        "Scenarios", "Scenario_Diff", "Marmot_Solutions_folder",
                        "ylabels", "xlabels", "ticklabels",
                        "color_list", "marker_style", "gen_names_dict", "pv_gen_cat",
                        "re_gen_cat", "vre_gen_cat", "thermal_gen_cat", "Region_Mapping", "figure_folder", "meta", "facet","shift_leap_day","duration_curve"]
        
            argument_list = [row.iloc[2], row.iloc[3], row.iloc[4], row.iloc[5],row.iloc[6], row.iloc[7],
                             hdf_out_folder, Zones, self.AGG_BY, ordered_gen, PLEXOS_color_dict,
                             self.Scenarios, self.Scenario_Diff, self.Marmot_Solutions_folder,
                             self.ylabels, self.xlabels, self.ticklabels,
                             color_list, marker_style, gen_names_dict, pv_gen_cat,
                             re_gen_cat, vre_gen_cat, thermal_gen_cat,self.Region_Mapping,figure_folder, meta,facet,shift_leap_day,duration_curve]
        
            argument_dict = {key_list[i]: argument_list[i] for i in range(len(key_list))}
        
            # Use run_plot_types to run any plotting module
            figures = os.path.join(figure_folder, self.AGG_BY + '_' + module)
            try:
                os.makedirs(figures)
            except FileExistsError:
                pass
            fig = PlotTypes(module, method, argument_dict, font_defaults)
            Figure_Out = fig.runmplot()
            
            if isinstance(Figure_Out, mfunc.MissingInputData):
                logger.info("Add Inputs With Formatter Before Attempting to Plot!\n")
                continue
            
            if isinstance(Figure_Out, mfunc.DataSavedInModule):
                logger.info(f'Plotting Completed for {row["Figure Output Name"]}\n')
                logger.info("Plots & Data Saved Within Module!\n")
                continue
            
            if isinstance(Figure_Out, mfunc.UnderDevelopment):
                logger.info("Plot is Under Development, Plotting Skipped!\n")
                continue
            
            if isinstance(Figure_Out, mfunc.InputSheetError):
                logger.info("Input Sheet Data Incorrect!\n")
                continue
            
            for zone_input in Zones:
                if isinstance(Figure_Out[zone_input], mfunc.MissingZoneData):
                    logger.info(f"No Data to Plot in {zone_input}")
        
                else:
                    # Save figures
                    try:
                        Figure_Out[zone_input]["fig"].figure.savefig(os.path.join(figures, zone_input.replace('.','') + "_" + row["Figure Output Name"] + '.' + figure_format), dpi=600, bbox_inches='tight')
                    except AttributeError:
                        Figure_Out[zone_input]["fig"].savefig(os.path.join(figures, zone_input.replace('.','') + "_" + row["Figure Output Name"] + '.' + figure_format), dpi=600, bbox_inches='tight')
        
                    #Save .csv's.
                    if not facet:
                        if Figure_Out[zone_input]['data_table'].empty:
                            logger.info(f'{row["Figure Output Name"]} does not return a data table')
                        else:
                            Figure_Out[zone_input]["data_table"].to_csv(os.path.join(figures, zone_input.replace('.','') + "_" + row["Figure Output Name"] + ".csv"))
        
                    else: #Facetted plot, save multiple tables
                        tables_folder = os.path.join(figures, zone_input.replace('.','') + "_" + row["Figure Output Name"] + "_data_tables")
                        try:
                             os.makedirs(tables_folder)
                        except FileExistsError:
                             # directory already exists
                            pass
                        for scenario in self.Scenarios:
                            #CSV output file name cannot exceed 75 characters!!  Scenario names may need to be shortened
                            s = zone_input.replace('.','') + "_" + scenario + ".csv"
                            Figure_Out[zone_input]["data_table"][scenario].to_csv(os.path.join(tables_folder, s))
        
            logger.info(f'Plotting Completed for {row["Figure Output Name"]}\n')
        
            mpl.pyplot.close('all')
        
        end_timer = time.time()
        time_elapsed = end_timer - start_timer
        logger.info(f'Main Plotting loop took {round(time_elapsed/60,2)} minutes')
        logger.info('All Plotting COMPLETED')




if __name__ == '__main__':
    
    '''
    The following code is run if the formatter is run directly,
    it does not run if the formatter is imported as a module. 
    '''
    
    #===============================================================================
    # Load Input Properties
    #===============================================================================
    
    #changes working directory to location of this python file
    os.chdir(pathlib.Path(__file__).parent.absolute())
    
    Marmot_user_defined_inputs = pd.read_csv(mconfig.parser("user_defined_inputs_file"), usecols=['Input','User_defined_value'],
                                         index_col='Input', skipinitialspace=True)

    Marmot_plot_select = pd.read_csv("Marmot_plot_select.csv")
    
    # Folder to save your processed solutions
    Marmot_Solutions_folder = Marmot_user_defined_inputs.loc['Marmot_Solutions_folder'].to_string(index=False).strip()
    
    Scenarios = pd.Series(Marmot_user_defined_inputs.loc['Scenarios'].squeeze().split(",")).str.strip().tolist()
    
    # These variables (along with Region_Mapping) are used to initialize MetaData
    PLEXOS_Solutions_folder = Marmot_user_defined_inputs.loc['PLEXOS_Solutions_folder'].to_string(index=False).strip()
    
    # For plots using the differnec of the values between two scenarios.
    # Max two entries, the second scenario is subtracted from the first.
    Scenario_Diff = pd.Series(str(Marmot_user_defined_inputs.loc['Scenario_Diff_plot'].squeeze()).split(",")).str.strip().tolist()
    if Scenario_Diff == ['nan']: Scenario_Diff = [""]
    
    Mapping_folder = 'mapping_folder'
    
    Region_Mapping = pd.read_csv(os.path.join(Mapping_folder, Marmot_user_defined_inputs.loc['Region_Mapping.csv_name'].to_string(index=False).strip()))
    Region_Mapping = Region_Mapping.astype(str)
    
    gen_names = pd.read_csv(os.path.join(Mapping_folder, Marmot_user_defined_inputs.loc['gen_names.csv_name'].to_string(index=False).strip()))
    
    AGG_BY = Marmot_user_defined_inputs.loc['AGG_BY'].squeeze().strip()
    # Facet Grid Labels (Based on Scenarios)
    zone_region_sublist = pd.Series(str(Marmot_user_defined_inputs.loc['zone_region_sublist'].squeeze()).split(",")).str.strip().tolist()

    ylabels = pd.Series(str(Marmot_user_defined_inputs.loc['Facet_ylabels'].squeeze()).split(",")).str.strip().tolist()
    if ylabels == ['nan']: ylabels = [""]
    xlabels = pd.Series(str(Marmot_user_defined_inputs.loc['Facet_xlabels'].squeeze()).split(",")).str.strip().tolist()
    if xlabels == ['nan']: xlabels = [""]
    
    # option to change tick labels on plot
    ticklabels = pd.Series(str(Marmot_user_defined_inputs.loc['Tick_labels'].squeeze()).split(",")).str.strip().tolist()
    if ticklabels == ['nan']: ticklabels = [""]
    
    
    initiate = MarmotPlot(Scenarios,AGG_BY,PLEXOS_Solutions_folder,gen_names,Marmot_plot_select,
                          Marmot_Solutions_folder,Mapping_folder,Scenario_Diff,zone_region_sublist,
                          xlabels,ylabels,ticklabels,Region_Mapping)
    
    initiate.run_plotter()
        