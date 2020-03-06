# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 14:16:30 2019

@author: dlevie
"""
#%%
import pandas as pd
import os
import matplotlib as mpl
import generation_stack
import total_generation 
import total_installed_capacity
import curtailment
import production_cost
import unserved_energy
import reserves
import generation_unstack
import transmission

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

Marmot_plot_select = pd.read_csv("Marmot_plot_select_test.csv")

Scenario_name = 'Cold Wave 2011' # 'BAU' # "BAU_No_VG_Reserves"

Solutions_folder = '../TB_2024/StageA_DA'

# Multi_Scenario = ["BAU_No_VG_Reserves", "BAU_VG_Reserves", "BAU_Copperplate",  
#                   "BAU2_No_VG_Reserves", "BAU2_VG_Reserves", "BAU2_Copperplate"]
Multi_Scenario = ['Cold Wave 2011'] # ['BAU']

# For plots using the differnec of the values between two scenarios. 
# Max two entries, the second scenario is subtracted from the first. 
Scenario_Diff = [] # ["Gas_Outage_+_Icing", "Base_Case"]

Mapping_folder = 'mapping_folder'

Region_Mapping = pd.read_csv(os.path.join(Mapping_folder, 'Region_mapping.csv'))
Reserve_Regions = pd.read_csv(os.path.join(Mapping_folder, 'reserve_region_type.csv'))
gen_names = pd.read_csv(os.path.join(Mapping_folder, 'gen_names.csv'))


AGG_BY = 'Interconnection' # "Usual"

# Facet Grid Labels (Based on Scenarios)
ylabels = [] # ["BAU", "BAU2"]
xlabels = [] # ["No VG Reserves", "VG Reserves", "Copperplate"]

#===============================================================================
# Input and Output Directories 
#===============================================================================


PLEXOS_Scenarios = os.path.join(Solutions_folder, 'PLEXOS_Scenarios')
# PLEXOS_Scenarios = '/Volumes/PLEXOS/Projects/Drivers_of_Curtailment/PLEXOS_Scenarios'

figure_folder = os.path.join(PLEXOS_Scenarios, Scenario_name, 'Figures_Output_test')
try:
    os.makedirs(figure_folder)
except FileExistsError:
    # directory already exists
    pass


hdf_out_folder = os.path.join(PLEXOS_Scenarios, Scenario_name,'Processed_HDF5_folder')
try:
    os.makedirs(hdf_out_folder)
except FileExistsError:
    # directory already exists
    pass

gen_stack_figures = os.path.join(figure_folder, AGG_BY + '_Gen_Stack')
try:
    os.makedirs(gen_stack_figures)
except FileExistsError:
    # directory already exists
    pass    
tot_gen_stack_figures = os.path.join(figure_folder, AGG_BY + '_Total_Gen_Stack')
try:
    os.makedirs(tot_gen_stack_figures)
except FileExistsError:
    # directory already exists
    pass    
installed_cap_figures = os.path.join(figure_folder, AGG_BY + '_Total_Installed_Capacity')
try:
    os.makedirs(installed_cap_figures)
except FileExistsError:
    # directory already exists
    pass                           
system_cost_figures = os.path.join(figure_folder, AGG_BY + '_Total_System_Cost')
try:
    os.makedirs(system_cost_figures)
except FileExistsError:
    # directory already exists
    pass                
reserve_timeseries_figures = os.path.join(figure_folder, AGG_BY + '_Reserve_Timeseries')
try:
    os.makedirs(reserve_timeseries_figures)
except FileExistsError:
    # directory already exists
    pass   
reserve_total_figures = os.path.join(figure_folder, AGG_BY + '_Reserve_Total')
try:
    os.makedirs(reserve_total_figures)
except FileExistsError:
    # directory already exists
    pass          
transmission_figures = os.path.join(figure_folder, AGG_BY + '_Transmission')
try:
    os.makedirs(transmission_figures)
except FileExistsError:
    pass                


#===============================================================================
# Standard Generation Order
#===============================================================================

ordered_gen = ['Nuclear',
               'Coal',
               'Gas-CC',
               'Gas-CC CCS',
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

if set(gen_names["New"].unique()).issubset(ordered_gen) == False:
                    print("\n WARNING!! The new categories from the gen_names csv do not exist in ordered_gen \n")
                    print(set(gen_names["New"].unique()) - (set(ordered_gen)))

#===============================================================================
# Colours and styles
#===============================================================================

                    
PLEXOS_color_dict = {'Nuclear':'#B22222',
                    'Coal':'#333333',
                    'Gas-CC':'#6E8B3D',
                    'Gas-CC CCS':'#396AB1',
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
                    

color_list = ['#396AB1', '#CC2529','#3E9651','#ff7f00','#6B4C9A','#922428','#cab2d6', '#6a3d9a', '#fb9a99', '#b15928']
 


marker_style = ["^", "*", "o", "D", "x", "<", "P", "H", "8", "+"]
#%%
#===============================================================================
# Main          
#===============================================================================                   
 
gen_names_dict=gen_names[['Original','New']].set_index("Original").to_dict()["New"]

if AGG_BY=="zone":
    Zones = pd.read_pickle('zones.pkl')
    Zones = Zones['name'].unique()
elif Region_Mapping.empty==True:
    Zones = pd.read_pickle('regions.pkl') 
    Zones = Zones['name'].unique()
else:     
    Zones = Region_Mapping[AGG_BY].unique()


Reserve_Regions = Reserve_Regions["Reserve_Region"].unique()


def pass_data(figure,argument_list):
    
    if figure == 'Generation Stack': 
        fig = generation_stack.mplot(argument_list) 
        Figure_Out = fig.gen_stack()
        return Figure_Out
    
    elif figure == 'Generation Stack Facet Grid': 
        fig = generation_stack.mplot(argument_list) 
        Figure_Out = fig.gen_stack_facet()
        return Figure_Out
    
    elif figure == 'Generation Timeseries Difference': 
        fig = generation_stack.mplot(argument_list) 
        Figure_Out = fig.gen_diff()
        return Figure_Out

        
    elif figure == 'Total Generation':
        fig = total_generation.mplot(argument_list) 
        Figure_Out = fig.total_gen()
        return Figure_Out
    
    elif figure == 'Total Generation Facet Grid':
        fig = total_generation.mplot(argument_list) 
        Figure_Out = fig.total_gen_facet()
        return Figure_Out

    
    elif figure == 'Total Installed Capacity':
        fig = total_installed_capacity.mplot(argument_list)
        Figure_Out = fig.total_cap()
        return Figure_Out
    
    elif figure == 'Curtailment vs Penetration':
        fig = curtailment.mplot(argument_list)
        Figure_Out = fig.curt_pen()
        return Figure_Out
    
    elif figure == 'Curtailment Duration Curve':
        fig = curtailment.mplot(argument_list)
        Figure_Out = fig.curt_duration_curve()
        return Figure_Out
    
    elif figure == 'Production Cost':
        fig = production_cost.mplot(argument_list)
        Figure_Out = fig.prod_cost()
        return Figure_Out
    
    elif figure == 'Total System Cost':
        fig = production_cost.mplot(argument_list)
        Figure_Out = fig.sys_cost()
        return Figure_Out
    
    elif figure == 'Unserved Energy Timeseries':
        fig = unserved_energy.mplot(argument_list)
        
        Figure_Out = fig.unserved_energy_timeseries()
        return Figure_Out
    
    elif figure == 'Total Unserved Energy':
        fig = unserved_energy.mplot(argument_list)
        
        Figure_Out = fig.tot_unserved_energy()
        return Figure_Out
    
    elif figure == 'Reserve Timeseries':
        fig = reserves.mplot(argument_list)
        
        Figure_Out = fig.reserve_timeseries()
        return Figure_Out
    
    elif figure == 'Reserve Timeseries Facet Grid':
        fig = reserves.mplot(argument_list)
        
        Figure_Out = fig.reserve_timeseries_facet()
        return Figure_Out
    
    if figure == 'Generation Unstacked': 
        fig = generation_unstack.mplot(argument_list) 
        Figure_Out = fig.gen_unstack()
        return Figure_Out
    
    elif figure == 'Generation Unstack Facet Grid': 
        fig = generation_unstack.mplot(argument_list) 
        Figure_Out = fig.gen_unstack_facet()
        return Figure_Out
    
    elif figure == 'Net Interchange':
         fig = transmission.mplot(argument_list) 
         Figure_Out = fig.net_interchange()
         return Figure_Out
        
# Filter for chosen figures to plot
Marmot_plot_select = Marmot_plot_select.loc[Marmot_plot_select["Plot Graph"] == True]


# Main loop to process each figure and pass data to functions
for index, row in Marmot_plot_select.iterrows():
   
    print("Processing " + row["Figure Type"])
    
# Checks if figure type is a reserve figure. This is required as reserve regions dont always match generator regions/zones    
    if "Reserve" in row["Figure Type"]:
        
        for region in Reserve_Regions:
            
            argument_list = [row.iloc[3], row.iloc[4], row.iloc[5], row.iloc[6], row.iloc[7], row.iloc[8],
                                  hdf_out_folder, Zones, AGG_BY, ordered_gen, PLEXOS_color_dict, Multi_Scenario,
                                  Scenario_Diff, PLEXOS_Scenarios, ylabels, xlabels, color_list, marker_style, gen_names_dict, pv_gen_cat, 
                                  re_gen_cat, vre_gen_cat, region]
            
            Figure_Out = pass_data(row["Figure Type"],argument_list)
            
            if row["Figure Type"] == "Reserve Timeseries":
                Figure_Out["fig"].savefig(reserve_timeseries_figures + region + "_" + row["Figure Output Name"] + "_" + Scenario_name, dpi=600, bbox_inches='tight')
                Figure_Out["data_table"].to_csv(os.path.join(reserve_timeseries_figures, region + "_" + row["Figure Output Name"] + "_" + Scenario_name + ".csv"))
                
            if row["Figure Type"] == "Reserve Timeseries Facet Grid":
                Figure_Out.savefig(reserve_timeseries_figures + region + "_" + row["Figure Output Name"], dpi=600, bbox_inches='tight')

    else:
        
       
        
        for zone_input in Zones:
            argument_list =  [row.iloc[3], row.iloc[4], row.iloc[5], row.iloc[6],row.iloc[7], row.iloc[8],
                                  hdf_out_folder, zone_input, AGG_BY, ordered_gen, PLEXOS_color_dict, Multi_Scenario,
                                  Scenario_Diff, PLEXOS_Scenarios, ylabels, xlabels, color_list, marker_style, gen_names_dict, pv_gen_cat, 
                                  re_gen_cat, vre_gen_cat, Reserve_Regions]
             
            Figure_Out = pass_data(row["Figure Type"],argument_list)
           
            if row["Figure Type"] == "Generation Stack":
                Figure_Out["fig"].savefig(os.path.join(gen_stack_figures, zone_input + "_" + row["Figure Output Name"] + "_" + Scenario_name), dpi=600, bbox_inches='tight')
                Figure_Out["data_table"].to_csv(os.path.join(gen_stack_figures, zone_input + "_" + row["Figure Output Name"] + "_" + Scenario_name + ".csv"))
                
            if row["Figure Type"] == "Generation Stack Facet Grid":
                Figure_Out.savefig(os.path.join(gen_stack_figures, zone_input + "_" + row["Figure Output Name"]), dpi=600, bbox_inches='tight')
            
            elif row["Figure Type"] == "Total Generation": 
                Figure_Out["fig"].figure.savefig(os.path.join(tot_gen_stack_figures, zone_input + "_" + row["Figure Output Name"]), dpi=600, bbox_inches='tight')
                Figure_Out["data_table"].to_csv(os.path.join(tot_gen_stack_figures, zone_input + "_" + row["Figure Output Name"] + ".csv"))
                
            elif row["Figure Type"] == "Total Generation Facet Grid": 
                Figure_Out["fig"].savefig(os.path.join(tot_gen_stack_figures, zone_input + "_" + row["Figure Output Name"]), dpi=600, bbox_inches='tight')
                Figure_Out["data_table"].to_csv(os.path.join(tot_gen_stack_figures, zone_input + "_" + row["Figure Output Name"] + ".csv"))
                
            elif row["Figure Type"] == "Total Installed Capacity": 
                Figure_Out["fig"].figure.savefig(os.path.join(installed_cap_figures, zone_input + "_" + row["Figure Output Name"]) , dpi=600, bbox_inches='tight')
                Figure_Out["data_table"].to_csv(os.path.join(installed_cap_figures, zone_input + "_" + row["Figure Output Name"] + ".csv"))
                
            # Continue here (NSG)
            elif row["Figure Type"] == "Curtailment vs Penetration": 
                Figure_Out["fig"].savefig(os.path.join(figure_folder, zone_input + "_" + row["Figure Output Name"]) , dpi=600, bbox_inches='tight')
                Figure_Out["data_table"].to_csv(os.path.join(figure_folder, zone_input + "_" + row["Figure Output Name"] + ".csv"))
            
            elif row["Figure Type"] == "Curtailment Duration Curve": 
                Figure_Out["fig"].savefig(os.path.join(figure_folder, zone_input + "_" + row["Figure Output Name"]) , dpi=600, bbox_inches='tight')
                Figure_Out["data_table"].to_csv(os.path.join(figure_folder, zone_input + "_" + row["Figure Output Name"] + ".csv"))
                
            elif row["Figure Type"] == "Production Cost": 
                Figure_Out["fig"].savefig(os.path.join(system_cost_figures, zone_input + "_" + row["Figure Output Name"]) , dpi=600, bbox_inches='tight')
                Figure_Out["data_table"].to_csv(os.path.join(system_cost_figures, zone_input + "_" + row["Figure Output Name"] + ".csv"))
                
            elif row["Figure Type"] == "Total System Cost": 
                Figure_Out["fig"].savefig(os.path.join(system_cost_figures, zone_input + "_" + row["Figure Output Name"]) , dpi=600, bbox_inches='tight')
                Figure_Out["data_table"].to_csv(os.path.join(system_cost_figures, zone_input + "_" + row["Figure Output Name"] + ".csv"))
                
            elif row["Figure Type"] == "Generation Timeseries Difference": 
                Figure_Out["fig"].savefig(os.path.join(figure_folder, zone_input + "_" + row["Figure Output Name"] + "_" + Scenario_Diff[0]+"_vs_"+Scenario_Diff[1]), dpi=600, bbox_inches='tight')
                Figure_Out["data_table"].to_csv(os.path.join(figure_folder, zone_input + "_" + row["Figure Output Name"] + "_" + Scenario_Diff[0]+"_vs_"+Scenario_Diff[1] + ".csv"))
        
            elif row["Figure Type"] == "Unserved Energy Timeseries" or row["Figure Type"] == 'Total Unserved Energy': 
                Figure_Out["fig"].savefig(os.path.join(figure_folder, zone_input + "_" + row["Figure Output Name"]) , dpi=600, bbox_inches='tight')
                Figure_Out["data_table"].to_csv(os.path.join(figure_folder, zone_input + "_" + row["Figure Output Name"] + ".csv"))
                
            elif row["Figure Type"] == "Generation Unstacked":
                Figure_Out["fig"].savefig(os.path.join(gen_stack_figures, zone_input + "_" + row["Figure Output Name"] + "_" + Scenario_name), dpi=600, bbox_inches='tight')
                Figure_Out["data_table"].to_csv(os.path.join(gen_stack_figures, zone_input + "_" + row["Figure Output Name"] + "_" + Scenario_name + ".csv"))
                
            elif row["Figure Type"] == "Generation Unstacked Facet Grid":
                Figure_Out.savefig(os.path.join(gen_stack_figures, zone_input + "_" + row["Figure Output Name"]), dpi=600, bbox_inches='tight')
#%%
 
list_test = [1,2,3,4]
               
def argument_list(list):
    print(list[1])
    
argument_list(list_test)
                
                
                