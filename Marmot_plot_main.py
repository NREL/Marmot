# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 14:16:30 2019

@author: dlevie
"""

import pandas as pd
import os
os.chdir(r"C:\Users\DLEVIE\Documents\Marmot")

import matplotlib as mpl

import generation_stack
import total_generation 
import total_installed_capacity
import curtailment

#===============================================================================
# Graphing Defaults
#===============================================================================

mpl.rc('xtick', labelsize=11) 
mpl.rc('ytick', labelsize=12) 
mpl.rc('axes', labelsize=16)
mpl.rc('legend', fontsize=11)
mpl.rc('font', family='serif')


#===============================================================================
""" User Defined Names, Directories and Settings """
#===============================================================================

Marmot_plot_select = pd.read_csv(r"C:\Users\DLEVIE\Documents\Marmot\Marmot_plot_select.csv")

Scenario_name = "BAU_No_VG_Reserves"

Run_folder = r"\\nrelqnap02\PLEXOS\Projects\Drivers_of_Curtailment"

# Multi_Scenario = ["High_PV", "High_PV_2x_Mingen", "High_PV_2x_Mintime", "High_PV_2x_Ramp_Rate", "High_PV_Copperplate", "High_PV_No_Mingen", 
#                   "High_PV_No_Mintime", "High_PV_No_VG_Reserves", "High_PV_10prct_Ramp_Rate", "Low_PV", "Low_PV_2x_Mingen",
#                   "Low_PV_2x_Mintime", "Low_PV_2x_Ramp_Rate", "Low_PV_Copperplate", "Low_PV_No_Mingen", "Low_PV_No_Mintime", "Low_PV_No_VG_Reserves",
#                   "Low_PV_10prct_Ramp_Rate", "BAU", "BAU_2x_Mingen", "BAU_2x_Mintime", "BAU_2x_Ramp_Rate", "BAU_Copperplate", "BAU_No_Mingen", "BAU_No_Mintime",
#                   "BAU_No_VG_Reserves",  "BAU2", "BAU2_2x_Mingen", "BAU2_2x_Mintime", "BAU2_2x_Ramp_Rate", "BAU2_Copperplate", 
#                   "BAU2_No_Mingen", "BAU2_No_Mintime", "BAU2_No_VG_Reserves"]

Multi_Scenario = ["BAU_No_VG_Reserves", "BAU_VG_Reserves", "BAU_Copperplate",  
                  "BAU2_No_VG_Reserves", "BAU2_VG_Reserves", "BAU2_Copperplate"]


region_mapping = pd.read_csv(r"\\nrelqnap02\PLEXOS\Projects\Drivers_of_Curtailment\Region_Mapping\Region_mapping_LA.csv")
Reserve_Regions = pd.read_csv(r"\\nrelqnap02\PLEXOS\Projects\Drivers_of_Curtailment\Region_Mapping\reserve_region_type_LA.csv")
gen_names = pd.read_csv(r"\\nrelqnap02\PLEXOS\Projects\Drivers_of_Curtailment\Region_Mapping\gen_names_LA.csv")

AGG_BY ="Usual"

# Facet Grid Labels (Based on Scenarios)
ylabels = ["BAU", "BAU2"]
xlabels = ["BAU No VG Reserves", "VG Reserves", "Copperplate"]

#===============================================================================
# Input and Output Directories 
#===============================================================================


PLEXOS_Scenarios = Run_folder + r"\PLEXOS_Scenarios\LA_Hourly_1212" 

figure_folder = PLEXOS_Scenarios + "/" + Scenario_name + r"\Figures_Output"
try:
    os.makedirs(figure_folder)
except FileExistsError:
    # directory already exists
    pass


hdf_out_folder = PLEXOS_Scenarios + "/" + Scenario_name + r"\Processed_HDF5_folder"
try:
    os.makedirs(hdf_out_folder)
except FileExistsError:
    # directory already exists
    pass

gen_stack_figures = figure_folder + "/" + AGG_BY +"_Gen_Stack\\"
try:
    os.makedirs(gen_stack_figures)
except FileExistsError:
    # directory already exists
    pass    
tot_gen_stack_figures = figure_folder + "/" + AGG_BY +"_Total_Gen_Stack\\"
try:
    os.makedirs(tot_gen_stack_figures)
except FileExistsError:
    # directory already exists
    pass    
installed_cap_figures = figure_folder + "/" + AGG_BY +"_Total_Installed_Capacity\\"
try:
    os.makedirs(installed_cap_figures)
except FileExistsError:
    # directory already exists
    pass                           
system_cost_figures = figure_folder + "/" + AGG_BY +"_Total_System_Cost\\"
try:
    os.makedirs(system_cost_figures)
except FileExistsError:
    # directory already exists
    pass                
reserve_timeseries_figures = figure_folder + "/" + AGG_BY + "_Reserve_Timeseries\\"
try:
    os.makedirs(reserve_timeseries_figures)
except FileExistsError:
    # directory already exists
    pass   
reserve_total_figures = figure_folder + "/" + AGG_BY + "_Reserve_Total\\"
try:
    os.makedirs(reserve_total_figures)
except FileExistsError:
    # directory already exists
    pass                          


HDF5_output = os.listdir(hdf_out_folder)
HDF5_output = str(HDF5_output[0])


#===============================================================================
# Standard Generation Order
#===============================================================================

ordered_gen = ['Nuclear',
               'Coal',
               'Gas-CC',
               'Gas-CT',
               'Gas',
               'Gas-Steam',
               'DualFuel',
               'Oil-Gas-Steam',
               'Oil',
               'Hydro',
               'Ocean', 
               'Geothermal',
               'Biomass',
               'Biopower',
               'Other',
               'Wind',
               'Solar',
               'CSP',
               'PV',
               'PV-Battery',
               'Battery',
               'PHS',
               'Storage',
               'Net Imports',
               'Curtailment']

pv_gen_cat = ['Solar',
              'PV']

re_gen_cat = ['Wind',
              'PV']

vre_gen_cat = ['Hydro',
               'Ocean',
               'Geothermal',
               'Biomass',
               'Biopwoer',
               'Wind',
               'Solar',
               'CSP',
               'PV']



#===============================================================================
# Colours and styles
#===============================================================================

                    
PLEXOS_color_dict = {'Nuclear':'#B22222',
                    'Coal':'#333333',
                    'Gas-CC':'#6E8B3D',
                    'Gas-CT':'#FFB6C1',
                    'DualFuel':'#000080',
                    'Oil-Gas-Steam':'#cd5c5c',
                    'Hydro':'#ADD8E6',
                    'Ocean':'#000080',
                    'Geothermal':'#eedc82',
                    'Biopower':'#008B00',
                    'Wind':'#4F94CD',
                    'CSP':'#EE7600',
                    'PV':'#FFC125',
                    'PV-Battery':'#CD950C',
                    'Storage':'#dcdcdc',
                    'Other': '#9370DB',
                    'Net Imports':'#efbbff',
                    'Curtailment': '#FF0000'}  
                    

color_list = ['#396AB1', '#CC2529','#3E9651','#535154','#6B4C9A','#922428','#948B3D']
 


marker_style = ["^", "*", "o", "D", "x", "<", "P"]
#===============================================================================
# Main          
#===============================================================================                   
 
gen_names_dict=gen_names[['Original','New']].set_index("Original").to_dict()["New"]
             
region_mapping.rename(columns={'name':'Region'}, inplace=True)
Zones = region_mapping[AGG_BY].unique()

Reserve_Regions = Reserve_Regions["Reserve_Region"].unique()


def pass_data(figure, prop, start, end, timezone, hdf_out_folder, HDF5_output, 
              zone_input, AGG_BY, ordered_gen, PLEXOS_color_dict, Multi_Scenario, 
              PLEXOS_Scenarios, ylabels, xlabels, color_list, marker_style, gen_names_dict, pv_gen_cat, re_gen_cat, vre_gen_cat):
    
    if figure == 'Generation Stack': 
        fig = generation_stack.mplot(prop, start, end, timezone, hdf_out_folder, HDF5_output, 
                                    zone_input, AGG_BY, ordered_gen, PLEXOS_color_dict, Multi_Scenario, 
                                    PLEXOS_Scenarios, ylabels, xlabels, gen_names_dict, re_gen_cat) 
        Figure_Out = fig.gen_stack()
        return Figure_Out
    
    if figure == 'Generation Stack Facet Grid': 
        fig = generation_stack.mplot(prop, start, end, timezone, hdf_out_folder, HDF5_output, 
                                    zone_input, AGG_BY, ordered_gen, PLEXOS_color_dict, Multi_Scenario, 
                                    PLEXOS_Scenarios, ylabels, xlabels, gen_names_dict, re_gen_cat) 
        Figure_Out = fig.gen_stack_facet()
        return Figure_Out

        
    elif figure == 'Total Generation':
        fig = total_generation.mplot(hdf_out_folder, HDF5_output, 
                                    zone_input, AGG_BY, ordered_gen, PLEXOS_color_dict, 
                                    Multi_Scenario, PLEXOS_Scenarios, ylabels, xlabels, gen_names_dict) 
        Figure_Out = fig.total_gen()
        return Figure_Out
    
    elif figure == 'Total Generation Facet Grid':
        fig = total_generation.mplot(hdf_out_folder, HDF5_output, 
                                zone_input, AGG_BY, ordered_gen, PLEXOS_color_dict, 
                                Multi_Scenario, PLEXOS_Scenarios, ylabels, xlabels, gen_names_dict) 
        Figure_Out = fig.total_gen_facet()
        return Figure_Out

    
    elif figure == 'Total Installed Capacity':
        fig = total_installed_capacity.mplot(hdf_out_folder, HDF5_output, 
                                    zone_input, AGG_BY, ordered_gen, PLEXOS_color_dict, 
                                    Multi_Scenario, PLEXOS_Scenarios, gen_names_dict)
        Figure_Out = fig.total_cap()
        return Figure_Out
    
    elif figure == 'Curtailment vs Penetration':
        fig = curtailment.mplot(prop, hdf_out_folder, HDF5_output, 
                                    zone_input, AGG_BY, ordered_gen, PLEXOS_color_dict, 
                                    Multi_Scenario, PLEXOS_Scenarios, color_list, marker_style, gen_names_dict, pv_gen_cat, re_gen_cat, vre_gen_cat)
        Figure_Out = fig.curt_pen()
        return Figure_Out

    

# Filter for chosen figures to plot
Marmot_plot_select = Marmot_plot_select.loc[Marmot_plot_select["Plot Graph"] == True]


# Main loop to process each figure and pass data to functions
for index, row in Marmot_plot_select.iterrows():
   
    print("Processing " + row["Figure Type"])

    for zone_input in Zones:
        Figure_Out = pass_data(row["Figure Type"], row.iloc[3], row.iloc[4], row.iloc[5], row.iloc[6],
                              hdf_out_folder, HDF5_output, zone_input, AGG_BY, ordered_gen, PLEXOS_color_dict, 
                              Multi_Scenario, PLEXOS_Scenarios, ylabels, xlabels, color_list, marker_style, gen_names_dict, pv_gen_cat, re_gen_cat, vre_gen_cat)
       
        if row["Figure Type"] == "Generation Stack":
            Figure_Out.savefig(gen_stack_figures + zone_input + "_" + row["Figure Output Name"] + "_" + Scenario_name, dpi=600, bbox_inches='tight')
       
        if row["Figure Type"] == "Generation Stack Facet Grid":
            Figure_Out.savefig(gen_stack_figures + zone_input + "_" + row["Figure Output Name"], dpi=600, bbox_inches='tight')
        
        elif row["Figure Type"] == "Total Generation": 
            Figure_Out["fig"].figure.savefig(tot_gen_stack_figures + zone_input + "_" + row["Figure Output Name"] , dpi=600, bbox_inches='tight')
            Figure_Out["data_table"].to_csv(figure_folder + "/" + zone_input + "_" + row["Figure Output Name"] + ".csv")
            
        elif row["Figure Type"] == "Total Generation Facet Grid": 
            Figure_Out["fig"].savefig(tot_gen_stack_figures + zone_input + "_" + row["Figure Output Name"] , dpi=600, bbox_inches='tight')    
            Figure_Out["data_table"].to_csv(figure_folder + "/" + zone_input + "_" + row["Figure Output Name"] + ".csv")
            
        elif row["Figure Type"] == "Total Installed Capacity": 
            Figure_Out["fig"].figure.savefig(installed_cap_figures + zone_input + "_" + row["Figure Output Name"] , dpi=600, bbox_inches='tight')
            Figure_Out["data_table"].to_csv(figure_folder + "/" + zone_input + "_" + row["Figure Output Name"] + ".csv")
            
        elif row["Figure Type"] == "Curtailment vs Penetration": 
            Figure_Out["fig"].savefig(figure_folder  + "/" + zone_input + "_" + row["Figure Output Name"] , dpi=600, bbox_inches='tight')
            Figure_Out["data_table"].to_csv(figure_folder + "/" + zone_input + "_" + row["Figure Output Name"] + ".csv")
            