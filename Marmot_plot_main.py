# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 14:16:30 2019

@author: dlevie
"""
#%%

import pandas as pd
import os
import pathlib
import matplotlib as mpl
import sys

#changes working directory to location of this python file
os.chdir(pathlib.Path(__file__).parent.absolute()) #If running in sections you have to manually change the current directory to where Marmot is

import generation_stack
import total_generation 
import total_installed_capacity
import capacity_factor
import curtailment
import production_cost
import unserved_energy
import reserves
import generation_unstack
import transmission
import ramping
import utilization_factor
import prices
# import constraints

try:
    print("Will plot row:" +(sys.argv[1]))
    print(str(len(sys.argv)-1)+" arguments were passed from commmand line.")
except IndexError:
    #No arguments passed
    pass


#===============================================================================
# Graphing Defaults
#===============================================================================

mpl.rc('xtick', labelsize=11) 
mpl.rc('ytick', labelsize=12) 
mpl.rc('axes', labelsize=16)
mpl.rc('legend', fontsize=11)
mpl.rc('font', family='serif')


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
Processed_Solutions_folder = Marmot_user_defined_inputs.loc['Processed_Solutions_folder'].to_string(index=False).strip()

Multi_Scenario = pd.Series(Marmot_user_defined_inputs.loc['Multi_scenario_plot'].squeeze().split(",")).str.strip().tolist()

# For plots using the differnec of the values between two scenarios. 
# Max two entries, the second scenario is subtracted from the first. 
Scenario_Diff = pd.Series(str(Marmot_user_defined_inputs.loc['Scenario_Diff_plot'].squeeze()).split(",")).str.strip().tolist()  
if Scenario_Diff == ['nan']: Scenario_Diff = [""]

Mapping_folder = 'mapping_folder'

Region_Mapping = pd.read_csv(os.path.join(Mapping_folder, Marmot_user_defined_inputs.loc['Region_Mapping.csv_name'].to_string(index=False).strip()))
Reserve_Regions = pd.read_csv(os.path.join(Mapping_folder, Marmot_user_defined_inputs.loc['reserve_region_type.csv_name'].to_string(index=False).strip()))
gen_names = pd.read_csv(os.path.join(Mapping_folder, Marmot_user_defined_inputs.loc['gen_names.csv_name'].to_string(index=False).strip()))

AGG_BY = Marmot_user_defined_inputs.loc['AGG_BY'].squeeze().strip()

# Facet Grid Labels (Based on Scenarios)
ylabels = pd.Series(str(Marmot_user_defined_inputs.loc['Facet_ylabels'].squeeze()).split(",")).str.strip().tolist() 
if ylabels == ['nan']: ylabels = [""]
xlabels = pd.Series(str(Marmot_user_defined_inputs.loc['Facet_xlabels'].squeeze()).split(",")).str.strip().tolist() 
if xlabels == ['nan']: xlabels = [""]

#===============================================================================
# Input and Output Directories 
#===============================================================================


PLEXOS_Scenarios = os.path.join(Processed_Solutions_folder)

figure_folder = os.path.join(PLEXOS_Scenarios, Scenario_name, 'Figures_Output')
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
capacity_factor_figures = os.path.join(figure_folder, AGG_BY + '_Capacity_Factor')
try:
    os.makedirs(capacity_factor_figures)
except FileExistsError:
    # directory already exists
    pass          
utilization_factor_figures = os.path.join(figure_folder, AGG_BY + '_Utilization_Factor')
try:
    os.makedirs(utilization_factor_figures)
except FileExistsError:
    # directory already exists
    pass
line_utilization_figures = os.path.join(figure_folder, 'Line_Utilization')
try:
    os.makedirs(line_utilization_figures)
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
ramping_figures = os.path.join(figure_folder, AGG_BY + '_Ramping')
try:
    os.makedirs(ramping_figures)
except FileExistsError:
    pass           
unserved_energy_figures = os.path.join(figure_folder, AGG_BY + '_Unserved_Energy')
try:
    os.makedirs(unserved_energy_figures)
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
    
thermal_gen_cat = pd.read_csv(os.path.join(Mapping_folder, 'thermal_gen_cat.csv'), squeeze = True).str.strip().tolist()

if set(gen_names["New"].unique()).issubset(ordered_gen) == False:
                    print("\n WARNING!! The new categories from the gen_names csv do not exist in ordered_gen \n")
                    print(set(gen_names["New"].unique()) - (set(ordered_gen)))

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

if AGG_BY=="zone":
    Zones = pd.read_pickle('zones.pkl')
    Zones = Zones['name'].unique()
elif Region_Mapping.empty==True:
    Zones = pd.read_pickle('regions.pkl') 
    Zones = Zones['name'].unique()
else:     
    Zones = Region_Mapping[AGG_BY].unique()

Reserve_Regions = Reserve_Regions["Reserve_Region"].unique()

# Filter for chosen figures to plot
if (len(sys.argv)-1) == 1: # If passed one argument (not including file name which is automatic)
    print("Will plot row " +(sys.argv[1])+" of Marmot plot select regardless of T/F.")
    Marmot_plot_select = Marmot_plot_select.iloc[int(sys.argv[1])-1].to_frame().T
else:
    Marmot_plot_select = Marmot_plot_select.loc[Marmot_plot_select["Plot Graph"] == True]



#%%
# Main loop to process each figure and pass data to functions
for index, row in Marmot_plot_select.iterrows():
    
    print("\n\n\n")
    print("Plot =  " + row["Figure Output Name"])
    
# Checks if figure type is a reserve figure. This is required as reserve regions dont always match generator regions/zones    
    if "Reserve" in row["Figure Type"]:
        
        for region in Reserve_Regions:
            
            argument_list = [row.iloc[3], row.iloc[4], row.iloc[5], row.iloc[6], row.iloc[7], row.iloc[8],
                                  hdf_out_folder, Zones, AGG_BY, ordered_gen, PLEXOS_color_dict, Multi_Scenario,
                                  Scenario_Diff, PLEXOS_Scenarios, ylabels, xlabels, color_list, marker_style, gen_names_dict, pv_gen_cat, 
                                  re_gen_cat, vre_gen_cat, region, thermal_gen_cat]
            
            if row["Figure Type"] == "Reserve Timeseries":
                fig = reserves.mplot(argument_list)
                Figure_Out = fig.reserve_timeseries()
                Figure_Out["fig"].savefig(reserve_timeseries_figures + region + "_" + row["Figure Output Name"] + "_" + Scenario_name, dpi=600, bbox_inches='tight')
                Figure_Out["data_table"].to_csv(os.path.join(reserve_timeseries_figures, region + "_" + row["Figure Output Name"] + "_" + Scenario_name + ".csv"))
                
            if row["Figure Type"] == "Reserve Timeseries Facet Grid":
                fig = reserves.mplot(argument_list)
                Figure_Out = fig.reserve_timeseries_facet()
                Figure_Out.savefig(reserve_timeseries_figures + region + "_" + row["Figure Output Name"], dpi=600, bbox_inches='tight')
            if row["Figure Type"] == "Reserve Shortage Region":
                fig = reserves.mplot(argument_list)
                Figure_Out = fig.reg_reserve_shortage()
                Figure_Out["fig"].savefig(os.path.join(reserve_total_figures , region + "_" + row["Figure Output Name"] + "_" + Scenario_name), dpi=600, bbox_inches='tight')
#                Figure_Out["data_table"].to_csv(os.path.join(reserve_timeseries_figures, region + "_" + row["Figure Output Name"] + "_" + Scenario_name + ".csv"))
            
            if row["Figure Type"] == "Reserve Provision Region":
                fig = reserves.mplot(argument_list)
                Figure_Out = fig.reg_reserve_provision()
                Figure_Out["fig"].savefig(os.path.join(reserve_total_figures , region + "_" + row["Figure Output Name"] + "_" + Scenario_name), dpi=600, bbox_inches='tight')
#                Figure_Out["data_table"].to_csv(os.path.join(reserve_timeseries_figures, region + "_" + row["Figure Output Name"] + "_" + Scenario_name + ".csv"))

            mpl.pyplot.close('all')

    else:
        
        for zone_input in Zones:
            
            argument_list =  [row.iloc[3], row.iloc[4], row.iloc[5], row.iloc[6],row.iloc[7], row.iloc[8],
               hdf_out_folder, zone_input, AGG_BY, ordered_gen, PLEXOS_color_dict, Multi_Scenario,
               Scenario_Diff, PLEXOS_Scenarios, ylabels, xlabels, color_list, marker_style, gen_names_dict, pv_gen_cat, 
               re_gen_cat, vre_gen_cat, Reserve_Regions, thermal_gen_cat,Region_Mapping]

            if row["Figure Type"] == "Generation Stack":
                fig = generation_stack.mplot(argument_list) 
                Figure_Out = fig.gen_stack()
                Figure_Out["fig"].savefig(os.path.join(gen_stack_figures, zone_input + "_" + row["Figure Output Name"] + "_" + Scenario_name), dpi=600, bbox_inches='tight')
                Figure_Out["data_table"].to_csv(os.path.join(gen_stack_figures, zone_input + "_" + row["Figure Output Name"] + "_" + Scenario_name + ".csv"))
                
            elif row["Figure Type"] == "Generation Stack Facet Grid":
                fig = generation_stack.mplot(argument_list) 
                Figure_Out = fig.gen_stack_facet()
                Figure_Out.savefig(os.path.join(gen_stack_figures, zone_input + "_" + row["Figure Output Name"]), dpi=600, bbox_inches='tight')
            
            elif row["Figure Type"] == "Total Generation": 
                fig = total_generation.mplot(argument_list) 
                Figure_Out = fig.total_gen()
                Figure_Out["fig"].figure.savefig(os.path.join(tot_gen_stack_figures, zone_input + "_" + row["Figure Output Name"]), dpi=600, bbox_inches='tight')
                Figure_Out["data_table"].to_csv(os.path.join(tot_gen_stack_figures, zone_input + "_" + row["Figure Output Name"] + ".csv"))
                
            elif row["Figure Type"] == "Total Generation Facet Grid": 
                print("Total Generation Facet Grid currently unavailable for plotting, code not stable and needs testing")
                # fig = total_generation.mplot(argument_list) 
                # Figure_Out = fig.total_gen_facet()
                # Figure_Out["fig"].savefig(os.path.join(tot_gen_stack_figures, zone_input + "_" + row["Figure Output Name"]), dpi=600, bbox_inches='tight')
                # Figure_Out["data_table"].to_csv(os.path.join(tot_gen_stack_figures, zone_input + "_" + row["Figure Output Name"] + ".csv"))
                
            elif row["Figure Type"] == "Total Installed Capacity":
                fig = total_installed_capacity.mplot(argument_list)
                Figure_Out = fig.total_cap()
                Figure_Out["fig"].figure.savefig(os.path.join(installed_cap_figures, zone_input + "_" + row["Figure Output Name"]) , dpi=600, bbox_inches='tight')
                Figure_Out["data_table"].to_csv(os.path.join(installed_cap_figures, zone_input + "_" + row["Figure Output Name"] + ".csv"))
                
            elif row["Figure Type"] == "Capacity Factor": 
                fig = capacity_factor.mplot(argument_list)
                Figure_Out = fig.cf()
                Figure_Out["fig"].figure.savefig(os.path.join(capacity_factor_figures, zone_input + "_" + row["Figure Output Name"]) , dpi=600, bbox_inches='tight')
                Figure_Out["data_table"].to_csv(os.path.join(capacity_factor_figures, zone_input + "_" + row["Figure Output Name"] + ".csv"))
                
            elif row["Figure Type"] == "Average Output When Committed": 
                fig = capacity_factor.mplot(argument_list)
                Figure_Out = fig.avg_output_when_committed()
                Figure_Out["fig"].figure.savefig(os.path.join(capacity_factor_figures, zone_input + "_" + row["Figure Output Name"]) , dpi=600, bbox_inches='tight')
                Figure_Out["data_table"].to_csv(os.path.join(capacity_factor_figures, zone_input + "_" + row["Figure Output Name"] + ".csv"))
                
            elif row["Figure Type"] == "Time at Minimum Generation": 
                fig = capacity_factor.mplot(argument_list)
                Figure_Out = fig.time_at_min_gen()
                Figure_Out["fig"].figure.savefig(os.path.join(capacity_factor_figures, zone_input + "_" + row["Figure Output Name"]) , dpi=600, bbox_inches='tight')
                Figure_Out["data_table"].to_csv(os.path.join(capacity_factor_figures, zone_input + "_" + row["Figure Output Name"] + ".csv"))
                
            elif row["Figure Type"] == "Capacity Started": 
                fig = ramping.mplot(argument_list)
                Figure_Out = fig.capacity_started()
                Figure_Out["fig"].figure.savefig(os.path.join(ramping_figures, zone_input + "_" + row["Figure Output Name"]) , dpi=600, bbox_inches='tight')
                Figure_Out["data_table"].to_csv(os.path.join(ramping_figures, zone_input + "_" + row["Figure Output Name"] + ".csv"))     

            elif row["Figure Type"] == "Utilization Factor Fleet": 
                fig = utilization_factor.mplot(argument_list)
                Figure_Out = fig.uf_fleet()
                Figure_Out["fig"].savefig(os.path.join(utilization_factor_figures, zone_input + "_" + row["Figure Output Name"]) , dpi=600, bbox_inches='tight')
                Figure_Out["data_table"].to_csv(os.path.join(utilization_factor_figures, zone_input + "_" + row["Figure Output Name"] + ".csv"))

            elif row["Figure Type"] == "Utilization Factor Generators": 
                fig = utilization_factor.mplot(argument_list)
                Figure_Out = fig.uf_gen()
                Figure_Out["fig"].savefig(os.path.join(utilization_factor_figures, zone_input + "_" + row["Figure Output Name"]) , dpi=600, bbox_inches='tight')
            
            elif row["Figure Type"] == "Line Utilization Hourly": 
                if zone_input == Zones[0]: # Only do this once. Not differentiated by zone.
                    fig = transmission.mplot(argument_list)
                    Figure_Out = fig.line_util()
                    Figure_Out["fig"].savefig(os.path.join(line_utilization_figures, row["Figure Output Name"]) , dpi=600, bbox_inches='tight')
                
            elif row["Figure Type"] == "Line Utilization Annual": 
                if zone_input == Zones[0]: # Only do this once. Not differentiated by zone.
                    fig = transmission.mplot(argument_list)
                    Figure_Out = fig.line_hist()
                    Figure_Out["fig"].savefig(os.path.join(line_utilization_figures, row["Figure Output Name"]) , dpi=200, bbox_inches='tight')
              
            elif row["Figure Type"] == "Region Price": 
                if zone_input == Zones[0]: # Only do this once. Not differentiated by zone.
                    fig = prices.mplot(argument_list)
                    Figure_Out = fig.price_region()
                    Figure_Out["fig"].savefig(os.path.join(figure_folder, row["Figure Output Name"]) , dpi=600, bbox_inches='tight')
             
            elif row["Figure Type"] == "Region Price Timeseries": 
                if zone_input == Zones[0]: # Only do this once. Not differentiated by zone.
                    fig = prices.mplot(argument_list)
                    Figure_Out = fig.price_region_chron()
                    Figure_Out["fig"].savefig(os.path.join(figure_folder, row["Figure Output Name"]) , dpi=600, bbox_inches='tight')
             
            elif row["Figure Type"] == "Constraint Violation": 
                if zone_input == Zones[0]: # Only do this once. Not differentiated by zone.
                    fig = constraints.mplot(argument_list)
                    Figure_Out = fig.constraint_violation()
                    Figure_Out["fig"].savefig(os.path.join(figure_folder, row["Figure Output Name"]) , dpi=600, bbox_inches='tight')               
            
            # Continue here (NSG)
            elif row["Figure Type"] == "Curtailment vs Penetration": 
                fig = curtailment.mplot(argument_list)
                Figure_Out = fig.curt_pen()
                Figure_Out["fig"].savefig(os.path.join(figure_folder, zone_input + "_" + row["Figure Output Name"]) , dpi=600, bbox_inches='tight')
                Figure_Out["data_table"].to_csv(os.path.join(figure_folder, zone_input + "_" + row["Figure Output Name"] + ".csv"))
            
            elif row["Figure Type"] == "Curtailment Duration Curve": 
                fig = curtailment.mplot(argument_list)
                Figure_Out = fig.curt_duration_curve()
                Figure_Out["fig"].savefig(os.path.join(figure_folder, zone_input + "_" + row["Figure Output Name"]) , dpi=600, bbox_inches='tight')
                Figure_Out["data_table"].to_csv(os.path.join(figure_folder, zone_input + "_" + row["Figure Output Name"] + ".csv"))
                
            elif row["Figure Type"] == "Production Cost": 
                fig = production_cost.mplot(argument_list)
                Figure_Out = fig.prod_cost()
                Figure_Out["fig"].savefig(os.path.join(system_cost_figures, zone_input + "_" + row["Figure Output Name"]) , dpi=600, bbox_inches='tight')
                Figure_Out["data_table"].to_csv(os.path.join(system_cost_figures, zone_input + "_" + row["Figure Output Name"] + ".csv"))
                
            elif row["Figure Type"] == "Total System Cost": 
                fig = production_cost.mplot(argument_list)
                Figure_Out = fig.sys_cost()
                Figure_Out["fig"].savefig(os.path.join(system_cost_figures, zone_input + "_" + row["Figure Output Name"]) , dpi=600, bbox_inches='tight')
                Figure_Out["data_table"].to_csv(os.path.join(system_cost_figures, zone_input + "_" + row["Figure Output Name"] + ".csv"))
            
            elif row["Figure Type"] == "Detailed Total Generation Cost": 
                fig = production_cost.mplot(argument_list)
                Figure_Out = fig.detailed_gen_cost()
                Figure_Out["fig"].savefig(os.path.join(system_cost_figures, zone_input + "_" + row["Figure Output Name"]) , dpi=600, bbox_inches='tight')
                Figure_Out["data_table"].to_csv(os.path.join(system_cost_figures, zone_input + "_" + row["Figure Output Name"] + ".csv"))
            
            elif row["Figure Type"] == "Generation Timeseries Difference": 
                fig = generation_stack.mplot(argument_list) 
                Figure_Out = fig.gen_diff()
                Figure_Out["fig"].savefig(os.path.join(figure_folder, zone_input + "_" + row["Figure Output Name"] + "_" + Scenario_Diff[0]+"_vs_"+Scenario_Diff[1]), dpi=600, bbox_inches='tight')
                Figure_Out["data_table"].to_csv(os.path.join(figure_folder, zone_input + "_" + row["Figure Output Name"] + "_" + Scenario_Diff[0]+"_vs_"+Scenario_Diff[1] + ".csv"))
        
            elif row["Figure Type"] == "Unserved Energy Timeseries" :
                fig = unserved_energy.mplot(argument_list)
                Figure_Out = fig.unserved_energy_timeseries()
                if isinstance(Figure_Out, pd.DataFrame):
                    print("No unserved energy in any scenario in "+zone_input)
                else:    
                    Figure_Out["fig"].savefig(os.path.join(unserved_energy_figures, zone_input + "_" + row["Figure Output Name"]) , dpi=600, bbox_inches='tight')
                    Figure_Out["data_table"].to_csv(os.path.join(unserved_energy_figures, zone_input + "_" + row["Figure Output Name"] + ".csv"))
                                
            elif row["Figure Type"] == 'Total Unserved Energy': 
                fig = unserved_energy.mplot(argument_list)
                Figure_Out = fig.tot_unserved_energy()
                if isinstance(Figure_Out, pd.DataFrame):
                    print("No unserved energy in any scenario in "+zone_input)
                else:    
                    Figure_Out["fig"].savefig(os.path.join(unserved_energy_figures, zone_input + "_" + row["Figure Output Name"]) , dpi=600, bbox_inches='tight')
                    Figure_Out["data_table"].to_csv(os.path.join(unserved_energy_figures, zone_input + "_" + row["Figure Output Name"] + ".csv"))
                
            elif row["Figure Type"] == "Generation Unstacked":
                fig = generation_unstack.mplot(argument_list) 
                Figure_Out = fig.gen_unstack()
                Figure_Out["fig"].savefig(os.path.join(gen_stack_figures, zone_input + "_" + row["Figure Output Name"] + "_" + Scenario_name), dpi=600, bbox_inches='tight')
                Figure_Out["data_table"].to_csv(os.path.join(gen_stack_figures, zone_input + "_" + row["Figure Output Name"] + "_" + Scenario_name + ".csv"))
                
            elif row["Figure Type"] == "Generation Unstacked Facet Grid":
                fig = generation_unstack.mplot(argument_list) 
                Figure_Out = fig.gen_unstack_facet()
                Figure_Out.savefig(os.path.join(gen_stack_figures, zone_input + "_" + row["Figure Output Name"]), dpi=600, bbox_inches='tight')
                
            elif row["Figure Type"] == 'Net Export':
                fig = transmission.mplot(argument_list) 
                Figure_Out = fig.net_export()
                Figure_Out["fig"].savefig(os.path.join(transmission_figures, zone_input + "_" + row["Figure Output Name"] + "_" + Scenario_name), dpi=600, bbox_inches='tight')
                Figure_Out["data_table"].to_csv(os.path.join(transmission_figures, zone_input + "_" + row["Figure Output Name"] + "_" + Scenario_name + ".csv"))
        
            #This property facets by zone aggregation, so it shouldn't be looped over for every zone_input. Keep at the end for now.
            elif row["Figure Type"] == 'Region-Region Net Interchange':
                fig = transmission.mplot(argument_list) 
                Figure_Out = fig.region_region_interchange()
                Figure_Out["fig"].savefig(os.path.join(transmission_figures, row["Figure Output Name"] + "_" + Scenario_name), dpi=600, bbox_inches='tight')
                Figure_Out["data_table"].to_csv(os.path.join(transmission_figures, row["Figure Output Name"] + "_" + Scenario_name + ".csv"))
                break

            mpl.pyplot.close('all')


 #%%
#subprocess.call("/usr/bin/Rscript --vanilla /Users/mschwarz/EXTREME EVENTS/PLEXOS results analysis/Marmot/run_html_output.R", shell=True)
                
