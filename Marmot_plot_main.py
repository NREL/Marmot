# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 14:16:30 2019

@author: Daniel Levie
"""
#%%

import pandas as pd
import os
import pathlib
import matplotlib as mpl
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/plottingmodules")
import importlib
from meta_data import MetaData
import logging 
import logging.config
import yaml
import time
import marmot_plot_functions as mfunc

#changes working directory to location of this python file
os.chdir(pathlib.Path(__file__).parent.absolute()) #If running in sections you have to manually change the current directory to where Marmot is

with open('marmot_logging_config.yml', 'rt') as f:
    conf = yaml.safe_load(f.read())
    logging.config.dictConfig(conf)
    
logger = logging.getLogger('marmot_plot')
# Creates a new log file for next run 
logger.handlers[1].doRollover()
logger.handlers[2].doRollover()

class plottypes:

    def __init__(self, figure_type, figure_output_name, argument_dict, font_defaults):
        self.figure_type = figure_type
        self.figure_output_name = figure_output_name
        self.argument_dict = argument_dict
        self.font_defaults = font_defaults

    def runmplot(self):
        mpl.rc('xtick', labelsize=self.font_defaults['xtick_size'])
        mpl.rc('ytick', labelsize=self.font_defaults['ytick_size'])
        mpl.rc('axes', labelsize=self.font_defaults['axes_size'])
        mpl.rc('legend', fontsize=self.font_defaults['legend_size'])
        mpl.rc('font', family=self.font_defaults['font_family'])
        mpl.rc('figure', max_open_warning = 0)

        plot = importlib.import_module(self.figure_type)
        fig = plot.mplot(self.argument_dict)

        process_attr = getattr(fig, self.figure_output_name)

        Figure_Out = process_attr()
        return Figure_Out

try:
    logger.info("Will plot row:" +(sys.argv[1]))
    logger.info(str(len(sys.argv)-1)+" arguments were passed from commmand line.")
except IndexError:
    #No arguments passed
    pass

#===============================================================================
# Set Graphing Font Defaults
#===============================================================================

font_defaults = {'xtick_size':11,
                 'ytick_size':12,
                 'axes_size':16,
                 'legend_size':11,
                 'font_family':'serif'}

#===============================================================================
# Load Input Properties
#===============================================================================

#A bug in pandas requires this to be included, otherwise df.to_string truncates long strings
#Fix available in Pandas 1.0 but leaving here in case user version not up to date
pd.set_option("display.max_colwidth", 1000)

Marmot_user_defined_inputs = pd.read_csv('Marmot_user_defined_inputs.csv', usecols=['Input','User_defined_value'],
                                         index_col='Input', skipinitialspace=True)

Marmot_plot_select = pd.read_csv("Marmot_plot_select.csv")

Scenario_name = Marmot_user_defined_inputs.loc['Main_scenario_plot'].squeeze().strip()

# Folder to save your processed solutions
Marmot_Solutions_folder = Marmot_user_defined_inputs.loc['Marmot_Solutions_folder'].to_string(index=False).strip()

# These variables (along with Region_Mapping) are used to initialize MetaData
PLEXOS_Solutions_folder = Marmot_user_defined_inputs.loc['PLEXOS_Solutions_folder'].to_string(index=False).strip()
HDF5_folder_in = os.path.join(PLEXOS_Solutions_folder, Scenario_name)



Multi_Scenario = pd.Series(Marmot_user_defined_inputs.loc['Multi_scenario_plot'].squeeze().split(",")).str.strip().tolist()

# For plots using the differnec of the values between two scenarios.
# Max two entries, the second scenario is subtracted from the first.
Scenario_Diff = pd.Series(str(Marmot_user_defined_inputs.loc['Scenario_Diff_plot'].squeeze()).split(",")).str.strip().tolist()
if Scenario_Diff == ['nan']: Scenario_Diff = [""]

Mapping_folder = 'mapping_folder'

Region_Mapping = pd.read_csv(os.path.join(Mapping_folder, Marmot_user_defined_inputs.loc['Region_Mapping.csv_name'].to_string(index=False).strip()))
Region_Mapping = Region_Mapping.astype(str)

gen_names = pd.read_csv(os.path.join(Mapping_folder, Marmot_user_defined_inputs.loc['gen_names.csv_name'].to_string(index=False).strip()))

AGG_BY = Marmot_user_defined_inputs.loc['AGG_BY'].squeeze().strip()
logger.info("Aggregation selected: %s",AGG_BY)
# Facet Grid Labels (Based on Scenarios)
zone_region_sublist = pd.Series(str(Marmot_user_defined_inputs.loc['zone_region_sublist'].squeeze()).split(",")).str.strip().tolist()
if zone_region_sublist != ['nan']:
    logger.info("Only plotting %s: %s",AGG_BY,zone_region_sublist)

ylabels = pd.Series(str(Marmot_user_defined_inputs.loc['Facet_ylabels'].squeeze()).split(",")).str.strip().tolist()
if ylabels == ['nan']: ylabels = [""]
xlabels = pd.Series(str(Marmot_user_defined_inputs.loc['Facet_xlabels'].squeeze()).split(",")).str.strip().tolist()
if xlabels == ['nan']: xlabels = [""]

# option to change tick labels on plot
ticklabels = pd.Series(str(Marmot_user_defined_inputs.loc['Tick_labels'].squeeze()).split(",")).str.strip().tolist()
if ticklabels == ['nan']: ticklabels = [""]

figure_format = str(Marmot_user_defined_inputs.loc['Figure_Format'].squeeze()).strip()
if figure_format == 'nan':
    figure_format = 'png'

shift_leap_day = str(Marmot_user_defined_inputs.loc['shift_leap_day'].squeeze())
#===============================================================================
# Input and Output Directories
#===============================================================================

figure_folder = os.path.join(Marmot_Solutions_folder, Scenario_name, 'Figures_Output')
try:
    os.makedirs(figure_folder)
except FileExistsError:
    # directory already exists
    pass

hdf_out_folder = os.path.join(Marmot_Solutions_folder, Scenario_name,'Processed_HDF5_folder')
try:
    os.makedirs(hdf_out_folder)
except FileExistsError:
    # directory already exists
    pass

#===============================================================================
# Standard Generation Order
#===============================================================================

ordered_gen = pd.read_csv(os.path.join(Mapping_folder, 'ordered_gen.csv'),squeeze=True).str.strip().tolist()

pv_gen_cat = pd.read_csv(os.path.join(Mapping_folder, 'pv_gen_cat.csv'),squeeze=True).str.strip().tolist()

re_gen_cat = pd.read_csv(os.path.join(Mapping_folder, 're_gen_cat.csv'),squeeze=True).str.strip().tolist()

vre_gen_cat = pd.read_csv(os.path.join(Mapping_folder, 'vre_gen_cat.csv'),squeeze=True).str.strip().tolist()

thermal_gen_cat = pd.read_csv(os.path.join(Mapping_folder, 'thermal_gen_cat.csv'), squeeze = True).str.strip().tolist()

# facet_gen_cat = pd.read_csv(os.path.join(Mapping_folder, 'facet_gen_cat.csv'), squeeze = True).str.strip().tolist()

if set(gen_names["New"].unique()).issubset(ordered_gen) == False:
                    logger.warning("The new categories from the gen_names csv do not exist in ordered_gen!: \
                    %s",set(gen_names["New"].unique()) - (set(ordered_gen)))

#===============================================================================
# Colours and styles
#===============================================================================

#ORIGINAL MARMOT COLORS
# PLEXOS_color_dict = {'Nuclear':'#B22222',
#                     'Coal':'#333333',
#                     'Gas-CC':'#6E8B3D',
#                     'Gas-CC CCS':'#396AB1',
#                     'Gas-CT':'#FFB6C1',
#                     'DualFuel':'#000080',
#                     'Oil-Gas-Steam':'#cd5c5c',
#                     'Hydro':'#ADD8E6',
#                     'Ocean':'#000080',
#                     'Geothermal':'#eedc82',
#                     'Biopower':'#008B00',
#                     'Wind':'#4F94CD',
#                     'CSP':'#EE7600',
#                     'PV':'#FFC125',
#                     'PV-Battery':'#CD950C',
#                     'Storage':'#dcdcdc',
#                     'Other': '#9370DB',
#                     'Net Imports':'#efbbff',
#                     'Curtailment': '#FF0000'}

#STANDARD SEAC COLORS (AS OF MARCH 9, 2020)
PLEXOS_color_dict = pd.read_csv(os.path.join(Mapping_folder, 'colour_dictionary.csv'))
PLEXOS_color_dict["Generator"] = PLEXOS_color_dict["Generator"].str.strip()
PLEXOS_color_dict["Colour"] = PLEXOS_color_dict["Colour"].str.strip()
PLEXOS_color_dict = PLEXOS_color_dict[['Generator','Colour']].set_index("Generator").to_dict()["Colour"]

color_list = ['#396AB1', '#CC2529','#3E9651','#ff7f00','#6B4C9A','#922428','#cab2d6', '#6a3d9a', '#fb9a99', '#b15928']

marker_style = ["^", "*", "o", "D", "x", "<", "P", "H", "8", "+"]

#===============================================================================
# Main
#===============================================================================

gen_names_dict=gen_names[['Original','New']].set_index("Original").to_dict()["New"]


try:
    Region_Mapping = Region_Mapping.drop(["category"],axis=1) # delete category columns if exists
except Exception:
    pass

# Instead of reading in pickle files, an instance of metadata is initialized with the appropriate parameters
# Methods within that class are used to retreive the data that was stored in pickle files

meta = MetaData(HDF5_folder_in, Region_Mapping)

# Zones_pkl = pd.read_pickle(os.path.join(Marmot_Solutions_folder, Scenario_name,"zones.pkl"))
# Regions_pkl = pd.read_pickle(os.path.join(Marmot_Solutions_folder, Scenario_name,'regions.pkl'))

if AGG_BY in {"zone", "zones", "Zone", "Zones"}:
    AGG_BY = 'zone'
    zones = meta.zones()
    if zones.empty == True:
        logger.warning("Input Sheet Data Incorrect! Your model does not contain Zones, enter a different aggregation")
        sys.exit()
    Zones = zones['name'].unique()

    if zone_region_sublist != ['nan']:
        zsub = []
        for zone in zone_region_sublist:
            if zone in Zones:
                zsub.append(zone)
            else:
                logger.info("metadata does not contain zone: %s, SKIPPING ZONE",zone)
        if zsub:
            Zones = zsub
        else:
            logger.warning("None of: %s in model Zones. Plotting all Zones",zone_region_sublist)

elif AGG_BY in {"region", "regions", "Region", "Regions"}:
    AGG_BY = 'region'
    regions = meta.regions()
    if regions.empty == True:
        logger.warning("Input Sheet Data Incorrect! Your model does not contain Regions, enter a different aggregation")
        sys.exit()
    Zones = regions['region'].unique()
    if zone_region_sublist != ['nan']:
        zsub = []
        for region in zone_region_sublist:
            if region in Zones:
                zsub.append(region)
            else:
                logger.info("metadata does not contain region: %s, SKIPPING REGION",region)
        if zsub:
            Zones = zsub
        else:
            logger.warning("None of: %s in model Regions. Plotting all Regions",zone_region_sublist)

else:
    logger.info("Plotting Custom region aggregation from Region_Mapping File")
    regions = meta.regions()
    Region_Mapping = regions.merge(Region_Mapping, how='left', on='region')
    Region_Mapping.dropna(axis=1, how='all', inplace=True)
    
    try:
        Zones = Region_Mapping[AGG_BY].unique()
    except KeyError:
        logger.warning("AGG_BY = '%s' is not in the Region_Mapping File, enter a different aggregation",AGG_BY)
        sys.exit()
    
    # remove any nan that might end  up in list
    Zones = [x for x in Zones if str(x) != 'nan']
    
    if zone_region_sublist != ['nan']:
        zsub = []
        for region in zone_region_sublist:
            if region in Zones:
                zsub.append(region)
            else:
                logger.info("Region_Mapping File does not contain region: %s, SKIPPING REGION",region)
        if zsub:
            Zones = zsub
        else:
            logger.warning("None of: %s in Region_Mapping File. Plotting all Regions of aggregation '%s'",zone_region_sublist, AGG_BY)


# Zones = Region_Mapping[AGG_BY].unique()   #If formated H5 is from an older version of Marmot may need this line instead.

# Filter for chosen figures to plot
if (len(sys.argv)-1) == 1: # If passed one argument (not including file name which is automatic)
    logger.info("Will plot row " +(sys.argv[1])+" of Marmot plot select regardless of T/F.")
    Marmot_plot_select = Marmot_plot_select.iloc[int(sys.argv[1])-1].to_frame().T
else:
    Marmot_plot_select = Marmot_plot_select.loc[Marmot_plot_select["Plot Graph"] == True]


#%%
start_timer = time.time()
# Main loop to process each figure and pass data to functions
for index, row in Marmot_plot_select.iterrows():

    print("\n\n\n")
    logger.info("Plot =  %s",row["Figure Output Name"])

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
                "Multi_Scenario", "Scenario_Diff", "Scenario_name", "Marmot_Solutions_folder",
                "ylabels", "xlabels", "ticklabels",
                "color_list", "marker_style", "gen_names_dict", "pv_gen_cat",
                "re_gen_cat", "vre_gen_cat", "thermal_gen_cat", "Region_Mapping", "figure_folder", "meta", "facet","shift_leap_day","duration_curve"]

    argument_list = [row.iloc[3], row.iloc[4], row.iloc[5], row.iloc[6],row.iloc[7], row.iloc[8],
                     hdf_out_folder, Zones, AGG_BY, ordered_gen, PLEXOS_color_dict,
                     Multi_Scenario, Scenario_Diff, Scenario_name, Marmot_Solutions_folder,
                     ylabels, xlabels, ticklabels,
                     color_list, marker_style, gen_names_dict, pv_gen_cat,
                     re_gen_cat, vre_gen_cat, thermal_gen_cat,Region_Mapping,figure_folder, meta,facet,shift_leap_day,duration_curve]

    argument_dict = {key_list[i]: argument_list[i] for i in range(len(key_list))}


##############################################################################

    # Use run_plot_types to run any plotting module
    figures = os.path.join(figure_folder, AGG_BY + '_' + module)
    try:
        os.makedirs(figures)
    except FileExistsError:
        pass
    fig = plottypes(module, method, argument_dict, font_defaults)
    Figure_Out = fig.runmplot()
    
    if isinstance(Figure_Out, mfunc.MissingInputData):
        logger.info("Add Inputs With Formatter Before Attempting to Plot!\n")
        continue
    
    if isinstance(Figure_Out, mfunc.DataSavedInModule):
        logger.info('Plotting Completed for %s\n',row["Figure Output Name"])
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
            logger.info("No Data to Plot in %s",zone_input)

        else:
            # Save figures
            try:
                Figure_Out[zone_input]["fig"].figure.savefig(os.path.join(figures, zone_input.replace('.','') + "_" + row["Figure Output Name"] + "_" + Scenario_name + '.' + figure_format), dpi=600, bbox_inches='tight')
            except AttributeError:
                Figure_Out[zone_input]["fig"].savefig(os.path.join(figures, zone_input.replace('.','') + "_" + row["Figure Output Name"] + "_" + Scenario_name + '.' + figure_format), dpi=600, bbox_inches='tight')
        
            # Save data tables to csv
            if not facet:
                if Figure_Out[zone_input]['data_table'].empty:
                    logger.info('%s does not return a data table',row["Figure Output Name"])
                    continue
                else:
                    Figure_Out[zone_input]["data_table"].to_csv(os.path.join(figures, zone_input.replace('.','') + "_" + row["Figure Output Name"] + "_" + Scenario_name + ".csv"))
                
            else:
                try:
                    if not Figure_Out[zone_input]['data_table']:
                        logger.info('%s does not return a data table',row["Figure Output Name"])
                except ValueError:
                    if Figure_Out[zone_input]['data_table'].empty:
                        logger.info('%s does not return a data table',row["Figure Output Name"])
                else:
                    tables_folder = os.path.join(figures, zone_input.replace('.','') + "_" + row["Figure Output Name"] + "_data_tables")
                    try:
                         os.makedirs(tables_folder)
                    except FileExistsError:
                         # directory already exists
                        pass
                    for scenario in Multi_Scenario:
                        #CSV output file name cannot exceed 75 characters!!  Scenario names may need to be shortened
                        s = zone_input.replace('.','') + "_" + scenario + ".csv"
                        Figure_Out[zone_input]["data_table"][scenario].to_csv(os.path.join(tables_folder, s))
    
    logger.info('Plotting Completed for %s\n',row["Figure Output Name"])

###############################################################################
    mpl.pyplot.close('all')

end_timer = time.time()
time_elapsed = end_timer - start_timer
logger.info('Main Plotting loop took %s minutes',round(time_elapsed/60,2))
logger.info('All Plotting COMPLETED')
 #%%
#subprocess.call("/usr/bin/Rscript --vanilla /Users/mschwarz/EXTREME EVENTS/PLEXOS results analysis/Marmot/run_html_output.R", shell=True)
