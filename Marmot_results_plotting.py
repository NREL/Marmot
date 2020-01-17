# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 14:50:33 2019

@author: Daniel Levie

"""

import pandas as pd
import os
#import h5py
import numpy as np
#import datetime as dt
#import glob
#import seaborn as sns
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
#from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
#                               AutoMinorLocator, FixedLocator)

import pickle
#===============================================================================
# Graphing Defaults
#===============================================================================

mpl.rc('xtick', labelsize=14) 
mpl.rc('ytick', labelsize=14) 
mpl.rc('axes', labelsize=16)
mpl.rc('legend', fontsize=11)
mpl.rc('font', family='serif')

#===============================================================================
""" User Defined Names, Directories and Settings """
#===============================================================================

Scenario_name = "LA_2020"

Run_folder = r"\\nrelqnap02\PLEXOS\Projects\Drivers_of_Curtailment"

#Multi_Scenario = ["0% DualFuel RT", "25% DualFuel RT", "50% DualFuel RT", "75% DualFuel RT", "100% DualFuel RT"]
Multi_Scenario = ["LA_2020"]

#Wind_Compare = ["Base_Case", "Gas_Outage_+_Turbine_Icing_+_Oil_Limitations"]

region_mapping = pd.read_csv(r"\\nrelqnap02\PLEXOS\Projects\Drivers_of_Curtailment\Region_Mapping\Region_mapping_LA.csv")
Reserve_Regions = pd.read_csv(r"\\nrelqnap02\PLEXOS\Projects\Drivers_of_Curtailment\Region_Mapping\reserve_region_type_LA.csv")

AGG_BY ="Usual"


""" This is used if you have created your own aggregated zone"""
#zone_select_name = "North East"


#===============================================================================
# Date ranges
#===============================================================================

#outage_date_from = "2024-01-02 14:15:00"
#outage_date_to= "2024-01-09 20:35:00"

#===============================================================================
# Input and Output Directories 
#===============================================================================


PLEXOS_Scenarios = Run_folder + r"\PLEXOS_Scenarios" 

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
               'Storage',
               'Net Imports',
               'Curtailment']


#===============================================================================
# Standard Colour dictionary 
#===============================================================================

ReEDS_color_dict = {'Nuclear':'#8B0000',
                    'Coal':'#000000',
                    'Gas-CC':'Chocolate',
                    'Gas-CT':'#d2b48c',
                    'Gas-Steam':'indianred',
                    'DualFuel':'#a85418',
                    'Oil-Gas-Steam':'#cd5c5c',
                    'Oil':'indianred',
                    'Hydro':'#27408b',
                    'Ocean':'#000080',
                    'Geothermal':'#eedc82',
                    'Biopower':'#54ff9f',
                    'Landfill-gas':'#3cb371',
                    'Wind':'#4f94cd',
                    'CSP':'#ee4000',
                    'PV':'#ffa500',
                    'Storage':'#dcdcdc',
                    'Other': 'black',
                    'Net Imports':'#efbbff'}
                    
PLEXOS_color_dict = {'Nuclear':'#B22222',
                    'Coal':'#333333',
                    'Gas-CC':'#6E8B3D',
                    'Gas-CT':'#FFB6C1',
                    'DualFuel':'#000080',
                    'Oil-Gas-Steam':'#cd5c5c',
                    'Hydro':'#ADD8E6',
                    'Ocean':'#000080',
                    'Geothermal':'#eedc82',
                    'Biopower':'#eedc82',
                    'Wind':'#4F94CD',
                    'CSP':'#EE7600',
                    'PV':'#FFC125',
                    'PV-Battery':'#CD950C',
                    'Storage':'#dcdcdc',
                    'Other': '#9370DB',
                    'Net Imports':'#efbbff',
                    'Curtailment': '#FF0000'}  
                    

color_list = ['#396AB1', '#CC2529','#3E9651','#CC2529','#535154','#6B4C9A','#922428','#948B3D']
 
                            
color_dict = {'0% DualFuel RT':'#396AB1', '25% DualFuel RT':'#DA7C30', '50% DualFuel RT':'#3E9651', '75% DualFuel RT':'#CC2529', 
              '100% DualFuel RT':'#535154', '0':'#6B4C9A'}                            

#===============================================================================
# Main          
#===============================================================================                   
              
region_mapping.rename(columns={'name':'Region'}, inplace=True)
Zones = region_mapping[AGG_BY].unique()

Reserve_Regions = Reserve_Regions["Reserve_Region"].unique()


def df_process_gen_inputs(df): 
        df = df.reset_index()
        df = df.groupby(["timestamp", "tech"], as_index=False).sum()
        df.tech = df.tech.astype("category")
        df.tech.cat.set_categories(ordered_gen, inplace=True)
        df = df.sort_values(["tech"]) 
        df = df.pivot(index='timestamp', columns='tech', values=0)
        return df  
    
def df_process_categorise_gen(df): 
        df=df
        df.tech = df.tech.astype("category")
        df.tech.cat.set_categories(ordered_gen, inplace=True)
        df = df.sort_values(["tech"]) 
        return df      


        
#===============================================================================
# Timeseries Stacked Generation Graph
#===============================================================================


Stacked_Gen_read = pd.read_hdf(hdf_out_folder + "/" + HDF5_output, 'generator_Generation')
Avail_Gen_read = pd.read_hdf(hdf_out_folder + "/" + HDF5_output, "generator_Available_Capacity")
Pump_Load_read =pd.read_hdf(hdf_out_folder + "/" + HDF5_output, "generator_Pump_Load" )
Load_read = pd.read_hdf(hdf_out_folder + "/" + HDF5_output, "region_Load")
Unserved_Energy_read = pd.read_hdf(hdf_out_folder + "/" + HDF5_output, "region_Unserved_Energy" )
Stacked_Curt_read = pd.read_hdf(hdf_out_folder + "/" + HDF5_output, "generation_Curtailment" )


print("Plotting Generation Stack by Zone")
for zone_input in Zones:
    

#    zone_input ="ISONE"
    print(zone_input)
    Pump_Load = pd.Series() # Initiate pump load 

    Stacked_Gen = Stacked_Gen_read.xs(zone_input,level=AGG_BY)        
    if Stacked_Gen.empty == True:
        continue
    Stacked_Gen = df_process_gen_inputs(Stacked_Gen)
    
    Avail_Gen = Avail_Gen_read.xs(zone_input,level=AGG_BY)
    Avail_Gen = df_process_gen_inputs(Avail_Gen)
    
    try:
        Stacked_Curt = Stacked_Curt_read.xs(zone_input,level=AGG_BY)
        Stacked_Curt = df_process_gen_inputs(Stacked_Curt)
        Stacked_Curt = Stacked_Curt.sum(axis=1)
        Stacked_Curt[Stacked_Curt<0.05] = 0 #Remove values less than 0.05 MW
        Stacked_Gen.insert(len(Stacked_Gen.columns),column='Curtailment',value=Stacked_Curt) #Insert curtailment into 
        Stacked_Gen = Stacked_Gen.loc[:, (Stacked_Gen != 0).any(axis=0)]
    except Exception:
        pass
    
    Load = Load_read.xs(zone_input,level=AGG_BY)
    Load = Load.groupby(["timestamp"]).sum()
    Load = Load.squeeze() #Convert to Series
    
    peak_load = Load.idxmax()
    end_date = peak_load + dt.timedelta(days=3)
    start_date = peak_load - dt.timedelta(days=3)

    try:
        Pump_Load = Pump_Load_read.xs(zone_input,level=AGG_BY)
        Pump_Load = Pump_Load.groupby(["timestamp"]).sum()
        Pump_Load = Pump_Load.squeeze() #Convert to Series
        if (Pump_Load == 0).all() == False:
            Pump_Load = Load - Pump_Load
    except Exception:
        pass
    
    Unserved_Energy = Unserved_Energy_read.xs(zone_input,level=AGG_BY)
    Unserved_Energy = Unserved_Energy.groupby(["timestamp"]).sum()
    Unserved_Energy = Unserved_Energy.squeeze() #Convert to Series
    if (Unserved_Energy == 0).all() == False:
        Unserved_Energy = Load - Unserved_Energy
  
   
########## Filter data by start and end date ###############
    Stacked_Gen = Stacked_Gen[start_date : end_date]
    Load = Load[start_date : end_date]
    Unserved_Energy = Unserved_Energy[start_date : end_date]
    Pump_Load = Pump_Load[start_date : end_date]
############################################################
    
    fig1, ax = plt.subplots(figsize=(9,6))
    sp = ax.stackplot(Stacked_Gen.index.values, Stacked_Gen.values.T, labels=Stacked_Gen.columns, linewidth=5,
                 colors=[PLEXOS_color_dict.get(x, '#333333') for x in Stacked_Gen.T.index])
    
    if (Unserved_Energy == 0).all() == False:
        lp2 = plt.plot(Unserved_Energy, color='#EE1289')
    
    lp = plt.plot(Load, color='black')
    
    if (Pump_Load == 0).all() == False:
        lp3 = plt.plot(Pump_Load, color='black', linestyle="--")
    
    ax.set_ylabel('Generation (MW)',  color='black', rotation='vertical')
    ax.set_xlabel('Date (EST)',  color='black', rotation='horizontal')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='y', which='major', length=5, width=1)
    ax.tick_params(axis='x', which='major', length=5, width=1)
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    ax.margins(x=0.01)
    ax.annotate('Peak Load', xy=(peak_load, Load[peak_load]), xytext=((peak_load + dt.timedelta(days=0.5)), (Load[peak_load] + Load[peak_load]*0.1)),
                fontsize=13, arrowprops=dict(facecolor='black', width=3, shrink=0.1))
    
    locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
    formatter = mdates.ConciseDateFormatter(locator)
    formatter.formats[2] = '%d\n %b'
    formatter.zero_formats[1] = '%b\n %Y'
    formatter.zero_formats[2] = '%d\n %b'
    formatter.zero_formats[3] = '%H:%M\n %d-%b'
    formatter.offset_formats[3] = '%b %Y'
    formatter.show_offset = False
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    
    if (Unserved_Energy == 0).all() == False:
        ax.fill_between(Load.index, Load,Unserved_Energy, facecolor='#EE1289')
    
    handles, labels = ax.get_legend_handles_labels()
    
 
    #Legend 1
    leg1 = ax.legend(reversed(handles), reversed(labels), loc='lower left',bbox_to_anchor=(1,0), 
                  facecolor='inherit', frameon=True)  
    #Legend 2
    leg2 = ax.legend(lp, ['Demand + Pumped Load'], loc='center left',bbox_to_anchor=(1, 0.9), 
                  facecolor='inherit', frameon=True)
    
    #Legend 3
    if (Unserved_Energy == 0).all() == False:
        leg3 = ax.legend(lp2, ['Unserved Energy'], loc='upper left',bbox_to_anchor=(1, 0.82), 
                  facecolor='inherit', frameon=True)
        
    #Legend 4
    if (Pump_Load == 0).all() == False:
        leg4 = ax.legend(lp3, ['Demand'], loc='upper left',bbox_to_anchor=(1, 0.885), 
                  facecolor='inherit', frameon=True)
    
    # Manually add the first legend back
    ax.add_artist(leg1)
    ax.add_artist(leg2)
    if (Unserved_Energy == 0).all() == False:
        ax.add_artist(leg3)
    
    fig1.savefig(gen_stack_figures + zone_input + "_Stacked_Gen_" + Scenario_name, dpi=600, bbox_inches='tight')

 
#Clear Some Memory
del Stacked_Gen_read 
del Avail_Gen_read
del Pump_Load_read
del Load_read
del Unserved_Energy_read
del Stacked_Curt_read

#===============================================================================
# Scenario Total Generation by Category
#===============================================================================

print("Plotting Total Generation Stack by Scenario and Zone")

# Create Dictionary to hold Datframes for each scenario 
Stacked_Gen_Collection = {} 
Stacked_Load_Collection = {}
for scenario in Multi_Scenario:
    Stacked_Gen_Collection[scenario] = pd.read_hdf(PLEXOS_Scenarios + r"\\" + scenario + r"\Processed_HDF5_folder" + "/" + HDF5_output, "generator_Generation")
    Stacked_Load_Collection[scenario] = pd.read_hdf(PLEXOS_Scenarios + r"\\" + scenario + r"\Processed_HDF5_folder" + "/" + HDF5_output,  "region_Load")

for zone_input in Zones:
    
    Total_Generation_Stack_Out = pd.DataFrame()
    Total_Load_Out = pd.DataFrame()
    print(zone_input)
    
    
    for scenario in Multi_Scenario:
        
        Total_Gen_Stack = Stacked_Gen_Collection.get(scenario)
        Total_Gen_Stack = Total_Gen_Stack.xs(zone_input,level=AGG_BY)
        Total_Gen_Stack = df_process_gen_inputs(Total_Gen_Stack)
        Total_Gen_Stack = Total_Gen_Stack.sum(axis=0)
        Total_Gen_Stack.rename(scenario, inplace=True)
        Total_Generation_Stack_Out = pd.concat([Total_Generation_Stack_Out, Total_Gen_Stack], axis=1, sort=False).fillna(0)
        
        Total_Load = Stacked_Load_Collection.get(scenario)
        Total_Load = Total_Load.xs(zone_input,level=AGG_BY)
        Total_Load = Total_Load.groupby(["timestamp"]).sum()
        Total_Load = Total_Load.rename(columns={0:scenario}).sum(axis=0)
        Total_Load_Out = pd.concat([Total_Load_Out, Total_Load], axis=0, sort=False)
        

    Total_Generation_Stack_Out = Total_Generation_Stack_Out.T/1000 
    Total_Generation_Stack_Out = Total_Generation_Stack_Out.loc[:, (Total_Generation_Stack_Out != 0).any(axis=0)]
    Total_Generation_Stack_Out.index = Total_Generation_Stack_Out.index.str.replace('_',' ')

    Total_Load_Out = Total_Load_Out.T/1000       
    
    if Total_Generation_Stack_Out.empty == True:
        continue
    

    fig4 = Total_Generation_Stack_Out.plot.bar(stacked=True, figsize=(9,6), rot=0, 
                         color=[PLEXOS_color_dict.get(x, '#333333') for x in Total_Generation_Stack_Out.columns], edgecolor='black', linewidth='0.1')
    fig4.spines['right'].set_visible(False)
    fig4.spines['top'].set_visible(False)
    fig4.set_ylabel('Total Genertaion (GWh)',  color='black', rotation='vertical')
#    fig4.set_xticklabels(["0% \nDualFuel", "25% \nDualFuel", "50% \nDualFuel", "75% \nDualFuel", "100% \nDualFuel"])
   
    handles, labels = fig4.get_legend_handles_labels()
    leg1 = fig4.legend(reversed(handles), reversed(labels), loc='center left',bbox_to_anchor=(1,0.5), 
                  facecolor='inherit', frameon=True)
    #adds comma to y axis data 
    fig4.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    fig4.tick_params(axis='y', which='major', length=3, width=1)   
    
    n=0
    for scenario in Multi_Scenario:
        x = [fig4.patches[n].get_x(), fig4.patches[n].get_x() + fig4.patches[n].get_width()]
        y = [int(Total_Load_Out[scenario])]*2
        lp2 = plt.plot(x,y, 'r--', c='black', linewidth=1.5)
        n=n+1

    
    #Legend 2
    leg2 = fig4.legend(lp2, ['Load'], loc='upper left',bbox_to_anchor=(1, 0.9), 
                  facecolor='inherit', frameon=True)
    
    fig4.add_artist(leg1)
    
    fig4.figure.savefig(tot_gen_stack_figures + zone_input + "_Total_Gen_Stack" , dpi=600, bbox_inches='tight')

#Deletes dataframes from dictionary to free up memory     
Stacked_Gen_Collection.clear() 
Stacked_Load_Collection.clear()


#===============================================================================
# Scenario Total Installed Capacity by Category
#===============================================================================


print("Plotting Total Installed Capacity by Scenario and Zone")

# Create Dictionary to hold Datframes for each scenario 
Installed_Capacity_Collection = {} 
for scenario in Multi_Scenario:
    Installed_Capacity_Collection[scenario] = pd.read_hdf(PLEXOS_Scenarios + r"\\" + scenario + r"\Processed_HDF5_folder" + "/" + (HDF5_output),  "generator_Installed_Capacity")
    
    
    
for zone_input in Zones:
#    zone_input="CA"
    
    Total_Installed_Capacity_Out = pd.DataFrame()
    print(zone_input)
    
    for scenario in Multi_Scenario:
        
        Total_Installed_Capacity = Installed_Capacity_Collection.get(scenario)
        Total_Installed_Capacity = Total_Installed_Capacity.xs(zone_input,level=AGG_BY)
        Total_Installed_Capacity = df_process_gen_inputs(Total_Installed_Capacity)
        Total_Installed_Capacity.reset_index(drop=True, inplace=True)
        Total_Installed_Capacity.rename(index={0:scenario}, inplace=True)
        Total_Installed_Capacity_Out = pd.concat([Total_Installed_Capacity_Out, Total_Installed_Capacity], axis=0, sort=False).fillna(0)
    
#    Total_Installed_Capacity_Out = Total_Installed_Capacity_Out.iloc[0:1]
    Total_Installed_Capacity_Out = Total_Installed_Capacity_Out/1000
    Total_Installed_Capacity_Out = Total_Installed_Capacity_Out.loc[:, (Total_Installed_Capacity_Out != 0).any(axis=0)]
    Total_Installed_Capacity_Out.index = Total_Installed_Capacity_Out.index.str.replace('_',' ')
    
    
    
    
    fig11 = Total_Installed_Capacity_Out.plot.bar(stacked=True, figsize=(9,6), rot=0, 
                         color=[PLEXOS_color_dict.get(x, '#333333') for x in Total_Installed_Capacity_Out.columns], edgecolor='black', linewidth='0.1')
    fig11.spines['right'].set_visible(False)
    fig11.spines['top'].set_visible(False)
    fig11.set_ylabel('Total Installed Capacity (GW)',  color='black', rotation='vertical')
#    fig4.set_xticklabels(["0% \nDualFuel", "25% \nDualFuel", "50% \nDualFuel", "75% \nDualFuel", "100% \nDualFuel"])
   
    handles, labels = fig11.get_legend_handles_labels()
    leg1 = fig11.legend(reversed(handles), reversed(labels), loc='center left',bbox_to_anchor=(1,0.5), 
                  facecolor='inherit', frameon=True)
    #adds comma to y axis data 
    fig11.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    fig11.tick_params(axis='y', which='major', length=3, width=1)   

    fig11.figure.savefig(installed_cap_figures + zone_input + "_Total_Installed_Capacity" , dpi=600, bbox_inches='tight')


#Deletes dataframes from dictionary to free up memory     
Installed_Capacity_Collection.clear() 
  

#===============================================================================
# Timeseries Stacked Reserve 
#===============================================================================

# Create Dictionary to hold Datframes for each scenario 
Reserve_Provision_Collection = {} 
for scenario in Multi_Scenario:
    Reserve_Provision_Collection[scenario] = pd.read_hdf(PLEXOS_Scenarios + r"\\" + scenario + r"\Processed_HDF5_folder" + "/" + HDF5_output,  "reserve_generators_Provision")
    

print("Plotting Timesereies Reserve by Region")

for region in Reserve_Regions:
    
    for scenario in Multi_Scenario:
        
        Reserve_Provision_Timeseries = Reserve_Provision_Collection.get(scenario)
        Reserve_Provision_Timeseries = Reserve_Provision_Timeseries.xs(region,level="Reserve_Region")          
        Reserve_Provision_Timeseries = df_process_gen_inputs(Reserve_Provision_Timeseries)
    
    peak_reserve =  Reserve_Provision_Timeseries.sum(axis=1).idxmax()
    start_date = peak_reserve - dt.timedelta(days=1)
    end_date = peak_reserve + dt.timedelta(days=1)

########## Filter data by start and end date ####################################### 
    Reserve_Provision_Timeseries = Reserve_Provision_Timeseries[start_date : end_date]
#####################################################################################    
   
    fig9, ax = plt.subplots(figsize=(9,6))
    sp = ax.stackplot(Reserve_Provision_Timeseries.index.values, Reserve_Provision_Timeseries.values.T, labels=Reserve_Provision_Timeseries.columns, linewidth=5,
                 colors=[PLEXOS_color_dict.get(x, '#333333') for x in Reserve_Provision_Timeseries.T.index])
    
    
    ax.set_ylabel('Reserve Provision (MW)',  color='black', rotation='vertical')
    ax.set_xlabel('Date (PST)',  color='black', rotation='horizontal')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='y', which='major', length=5, width=1)
    ax.tick_params(axis='x', which='major', length=5, width=1)
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    ax.margins(x=0.01)
    
    locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
    formatter = mdates.ConciseDateFormatter(locator)
    formatter.formats[2] = '%d\n %b'
    formatter.zero_formats[1] = '%b\n %Y'
    formatter.zero_formats[2] = '%d\n %b'
    formatter.zero_formats[3] = '%H:%M\n %d-%b'
    formatter.offset_formats[3] = '%b %Y'
    formatter.show_offset = False
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    
    handles, labels = ax.get_legend_handles_labels()
    
 
    #Legend 1
    leg1 = ax.legend(reversed(handles), reversed(labels), loc='lower left',bbox_to_anchor=(1,0), 
                  facecolor='inherit', frameon=True)  
    
    fig9.savefig(reserve_timeseries_figures + region + "_Reserve_Timeseries_" + Scenario_name, dpi=600, bbox_inches='tight')


#===============================================================================
# Total Reserve by Type
#===============================================================================

print("Plotting Total Reserve by Region")

for region in Reserve_Regions:
    print(region)
    
    for scenario in Multi_Scenario:
        
        Reserve_Provision_Type = Reserve_Provision_Collection.get(scenario)
        Reserve_Provision_Type = Reserve_Provision_Type.xs(region,level="Reserve_Region")
        Reserve_Provision_Type.reset_index(inplace=True)
        Reserve_Provision_Type = Reserve_Provision_Type.groupby(["Type", "tech"], as_index=False).sum()
        Reserve_Provision_Type["Type"] = Reserve_Provision_Type["Type"].str.wrap(11)
        Reserve_Provision_Type = df_process_categorise_gen(Reserve_Provision_Type)  
        Reserve_Provision_Type = Reserve_Provision_Type.pivot(index='Type', columns='tech', values=0)
        Reserve_Provision_Type.fillna(0, inplace=True)
        Reserve_Provision_Type = Reserve_Provision_Type.loc[:, (Reserve_Provision_Type != 0).any(axis=0)]
        Reserve_Provision_Type = Reserve_Provision_Type/1000
    
    
    fig10 = Reserve_Provision_Type.plot.bar(stacked=True, figsize=(9,6), rot=0, 
                         color=[PLEXOS_color_dict.get(x, '#333333') for x in Reserve_Provision_Type.columns], edgecolor='black', linewidth='0.1')
    fig10.spines['right'].set_visible(False)
    fig10.spines['top'].set_visible(False)
    fig10.set_ylabel('Total Reserve Provision (GWh)',  color='black', rotation='vertical')

    handles, labels = fig10.get_legend_handles_labels()
    leg1 = fig10.legend(reversed(handles), reversed(labels), loc='center left',bbox_to_anchor=(1,0.5), 
                  facecolor='inherit', frameon=True)
    #adds comma to y axis data 
    fig10.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    fig10.tick_params(axis='y', which='major', length=3, width=1)   
    
    fig10.figure.savefig(reserve_total_figures + region + "_Reserve_Total_" + Scenario_name, dpi=600, bbox_inches='tight')

#Deletes dataframes from dictionary to free up memory      
Reserve_Provision_Collection.clear()

#===============================================================================
# Scenario System Cost Compare
#===============================================================================


print("Plotting Total System Net Rev, Rev, & Cost by Scenario and Zone")

# Create Dictionary to hold Datframes for each scenario 
Total_Gen_Cost_Collection = {}
Pool_Revenues_Collection = {}
Reserve_Revenues_Collection = {}
Installed_Capacity_Collection = {} 
for scenario in Multi_Scenario:
    Total_Gen_Cost_Collection[scenario] = pd.read_hdf(PLEXOS_Scenarios + r"\\" + scenario + r"\Processed_HDF5_folder" + "/" + HDF5_output, "Total_Generation_Cost")
    Pool_Revenues_Collection[scenario] = pd.read_hdf(PLEXOS_Scenarios + r"\\" + scenario + r"\Processed_HDF5_folder" + "/" + HDF5_output,  "Pool_Revenues")
    Reserve_Revenues_Collection[scenario] = pd.read_hdf(PLEXOS_Scenarios + r"\\" + scenario + r"\Processed_HDF5_folder" + "/" + HDF5_output,  "Reserves_Revenue")
    Installed_Capacity_Collection[scenario] = pd.read_hdf(PLEXOS_Scenarios + r"\\" + scenario + r"\Processed_HDF5_folder" + "/" + HDF5_output,  "Installed_Capacity")
    
    
for zone_input in Zones:
    
    zone_input="NYISO"
    Total_Systems_Cost_Out = pd.DataFrame()
    print(zone_input)
    
    for scenario in Multi_Scenario:
        
        Total_Installed_Capacity = Installed_Capacity_Collection.get(scenario)
        Total_Installed_Capacity = Total_Installed_Capacity.xs(zone_input,level=AGG_BY)
        Total_Installed_Capacity = df_process_gen_inputs(Total_Installed_Capacity)
        Total_Installed_Capacity.reset_index(drop=True, inplace=True)
        Total_Installed_Capacity = Total_Installed_Capacity.iloc[0]
        
        Total_Gen_Cost = Total_Gen_Cost_Collection.get(scenario)
        Total_Gen_Cost = Total_Gen_Cost.xs(zone_input,level=AGG_BY)
        Total_Gen_Cost = df_process_gen_inputs(Total_Gen_Cost)
        Total_Gen_Cost = Total_Gen_Cost.sum(axis=0)*-1
#        Total_Gen_Cost = Total_Gen_Cost/Total_Installed_Capacity #Change to $/kW-year
        Total_Gen_Cost.rename("Total_Gen_Cost", inplace=True)
        
        Pool_Revenues = Pool_Revenues_Collection.get(scenario)
        Pool_Revenues = Pool_Revenues.xs(zone_input,level=AGG_BY)
        Pool_Revenues = df_process_gen_inputs(Pool_Revenues)
        Pool_Revenues = Pool_Revenues.sum(axis=0)
#        Pool_Revenues = Pool_Revenues/Total_Installed_Capacity #Change to $/kW-year
        Pool_Revenues.rename("Energy_Revenues", inplace=True)
        
#        Reserve_Revenues = Reserve_Revenues_Collection.get(scenario)
##        Reserve_Revenues = Reserve_Revenues.xs(zone_input,level=AGG_BY)
#        Reserve_Revenues = df_process_gen_inputs(Reserve_Revenues)
#        Reserve_Revenues = Reserve_Revenues.sum(axis=0)
#        Reserve_Revenues = Reserve_Revenues/Total_Installed_Capacity #Change to $/kW-year
#        Reserve_Revenues.rename("Reserve_Revenues", inplace=True)
#        
#        Total_Systems_Cost_Out = pd.concat([Total_Systems_Cost_Out, Total_Gen_Cost, Pool_Revenues, Reserve_Revenues], axis=1, sort=False) 
    
        Total_Systems_Cost_Out = pd.concat([Total_Systems_Cost_Out, Total_Gen_Cost, Pool_Revenues], axis=1, sort=False) 
    
        Total_Systems_Cost_Out.columns = Total_Systems_Cost_Out.columns.str.replace('_',' ')    
        Total_Systems_Cost_Out = Total_Systems_Cost_Out.sum(axis=0)
        Total_Systems_Cost_Out = Total_Systems_Cost_Out.rename(scenario)
        
    if Total_Systems_Cost_Out.empty == True:
        continue
        
    
    fig2 = Total_Systems_Cost_Out.plot.bar(stacked=True, figsize=(15,6), rot=0)
    fig2.spines['right'].set_visible(False)
    fig2.spines['top'].set_visible(False)
    fig2.set_ylabel('Total System Net Rev, Rev, & Cost (Million $)',  color='black', rotation='vertical')
#    fig2.set_xticklabels(["0% \nDualFuel", "25% \nDualFuel", "50% \nDualFuel", "75% \nDualFuel", "100% \nDualFuel"])
   
    handles, labels = fig2.get_legend_handles_labels()
    fig2.legend(reversed(handles), reversed(labels), loc='upper center',bbox_to_anchor=(0.5,-0.15), 
                 facecolor='inherit', frameon=True, ncol=3)
    #adds comma to y axis data 
    fig2.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    fig2.tick_params(axis='y', which='major', length=3, width=1)   


    """adds annotations to bar plots"""
    cost_values=[]  #holds cost of each stack
    cost_totals=[]  #holds total of each bar
    
    for i in fig2.patches:
        cost_values.append(i.get_height())
    
    #calculates total value of bar
    q=0    
    j = int(len(cost_values)/2)   #total number of bars in plot
    for cost in cost_values: 
        out = cost + cost_values[q+j]
        cost_totals.append(out)
        q=q+1
        if q>=j:
            break
        
    #inserts values into bar stacks
    for i in fig2.patches:
       width, height = i.get_width(), i.get_height()
       if height<=1:
           continue
       x, y = i.get_xy() 
       fig2.text(x+width/2, 
            y+height/2, 
            '{:,.0f}'.format(height), 
            horizontalalignment='center', 
            verticalalignment='center', fontsize=13)
   
    #inserts total bar value above each bar
    k=0   
    for i in fig2.patches:
        height = cost_totals[k]
        width = 0.5
        x, y = i.get_xy() 
        fig2.text(x+width/2, 
            y+height + 0.05*max(fig2.get_ylim()), 
            '{:,.0f}'.format(height), 
            horizontalalignment='center', 
            verticalalignment='center', fontsize=15, color='red') 
        k=k+1
        if k>=j:
            break

    fig2.figure.savefig(system_cost_figures + zone_input + "_System_Cost" , dpi=600, bbox_inches='tight')


#===============================================================================
# Scenario Oil Offtake North East
#===============================================================================

Oil_Offtake_Out = pd.DataFrame()
for scenario in Multi_Scenario:
    try:
        with open(PLEXOS_Scenarios + r"\\" + scenario + r"\Pickle_folder" + "\\NE_Oil_Offtake.pkl", 'rb') as f:
            oil_offtake = pickle.load(f)
            Oil_Offtake_Out = pd.concat([Oil_Offtake_Out, oil_offtake], axis=1, sort=False) 
    except Exception:
        print("WARNING Scenario", scenario, "data is Missing")
        break 
Oil_Offtake_Out.columns = Oil_Offtake_Out.columns.str.replace('_',' ')   
Oil_Offtake_Out =  Oil_Offtake_Out.T    
    
fig3 = Oil_Offtake_Out.plot.bar(figsize=(9,6), rot=0, color="#cd5c5c")
fig3.spines['right'].set_visible(False)
fig3.spines['top'].set_visible(False)
fig3.set_ylabel('Total Oil Usage in North East (GBtu)',  color='black', rotation='vertical')
fig3.get_legend().remove()
fig3.set_xticklabels(["Base Case", "Gas Outage", "Gas Outage +\nIcing", "Gas Outage + \nIcing + \nOil Limitations" ])
   
   
#adds comma to y axis data 
fig3.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
fig3.tick_params(axis='y', which='major', length=3, width=1)   


for i in fig3.patches:
       width, height = i.get_width(), i.get_height()
       if height<=1:
           continue
       x, y = i.get_xy() 
       fig3.text(x+width/2, 
            y+height/2, 
            '{:,.0f}'.format(height), 
            horizontalalignment='center', 
            verticalalignment='center', fontsize=13)
    
fig3.figure.savefig(figure_folder + "\\" + zone_select_name + "_Oil_Usage", dpi=600, bbox_inches='tight')    
    
  
#===============================================================================
# Scenario Wind Output Compare North East
#===============================================================================  
   

Wind_Gen_Out = pd.DataFrame()

 
for scenario in Wind_Compare:
    print(scenario)
    wind_gen = pd.read_pickle(PLEXOS_Scenarios + r"\\" + scenario + r"\Pickle_folder" + "\\North_East_Total_Wind_Generation.pkl")    
    Wind_Gen_Out = pd.concat([Wind_Gen_Out, wind_gen], axis=1, sort=False)
    
Wind_Gen_Out = Wind_Gen_Out.sum()/1000

    
fig4 = Wind_Gen_Out.plot.bar(figsize=(9,6), rot=0, color="#4F94CD")
fig4.spines['right'].set_visible(False)
fig4.spines['top'].set_visible(False)
fig4.set_ylabel('Total Wind Generation in North East (GWh)',  color='black', rotation='vertical')
fig4.set_xticklabels(["Wind - BAU", "Wind - Turbine Icing"])
   
#adds comma to y axis data 
fig4.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
fig4.tick_params(axis='y', which='major', length=3, width=1)   

for i in fig4.patches:
       width, height = i.get_width(), i.get_height()
       if height<=1:
           continue
       x, y = i.get_xy() 
       fig4.text(x+width/2, 
            y+height/2, 
            '{:,.0f}'.format(height), 
            horizontalalignment='center', 
            verticalalignment='center', fontsize=13)
       
fig4.figure.savefig(figure_folder + "\\" + zone_select_name + "_Wind_Generation", dpi=600, bbox_inches='tight') 



#===============================================================================
# Scenario Unserved Energy 
#===============================================================================  

print("Plotting Unserved by Scenario and Zone")

Unserved_Energy_Collection = {}
for scenario in Multi_Scenario:
    Unserved_Energy_Collection[scenario] = pd.read_hdf(PLEXOS_Scenarios + r"\\" + scenario + r"\Processed_HDF5_folder" + "/" + HDF5_output, "region_Unserved_Energy")

Zones =["North East"]  
for zone_input in Zones:
    Unserved_Energy_Timeseries_Out = pd.DataFrame()
    Total_Unserved_Energy_Out = pd.DataFrame()

    
    for scenario in Multi_Scenario:
        unserved_eng_timeseries = Unserved_Energy_Collection.get(scenario)
        unserved_eng_timeseries = unserved_eng_timeseries.xs(zone_input,level=AGG_BY)
        unserved_eng_timeseries = unserved_eng_timeseries.groupby(["timestamp"]).sum()
        unserved_eng_timeseries = unserved_eng_timeseries.squeeze() #Convert to Series
        unserved_eng_timeseries.rename(scenario, inplace=True)
        Unserved_Energy_Timeseries_Out = pd.concat([Unserved_Energy_Timeseries_Out, unserved_eng_timeseries], axis=1, sort=False).fillna(0)
    
    Unserved_Energy_Timeseries_Out.columns = Unserved_Energy_Timeseries_Out.columns.str.replace('_',' ')     
    Unserved_Energy_Timeseries_Out = Unserved_Energy_Timeseries_Out.loc[:, (Unserved_Energy_Timeseries_Out >= 1).any(axis=0)]
    Total_Unserved_Energy_Out = Unserved_Energy_Timeseries_Out.sum(axis=0)
    

    fig5, (ax1, ax2) = plt.subplots(2,1, sharex=False,  figsize=(9,9))
    
    # Converts color_list into an iterable list for use in a loop
    iter_colour = iter(color_list)
    
    for column in Unserved_Energy_Timeseries_Out:
        ax1.plot(Unserved_Energy_Timeseries_Out[column], linewidth=3, antialiased=True, 
                 color=next(iter_colour), label=column)
        # ax1.legend(loc='lower right', 
                 # facecolor='inherit', frameon=True)

    ax1.set_ylabel('Unserved Energy (MW)',  color='black', rotation='vertical')
    ax1.set_ylim(bottom=0)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.tick_params(axis='y', which='major', length=5, width=1)
    ax1.tick_params(axis='x', which='major', length=5, width=1)
    ax1.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    ax1.margins(x=0.01)
    
    ax1.axvline(dt.datetime(2024, 1, 2, 2, 0), color='black', linestyle='--')
#    ax1.axvline(outage_date_from, color='black', linestyle='--')
    ax1.text(dt.datetime(2024, 1, 1, 5, 15), 0.8*max(ax1.get_ylim()), "Outage \nBegins", fontsize=13)
    ax1.axvline(dt.datetime(2024, 1, 6, 23, 0), color='black', linestyle='--')
    ax1.text(dt.datetime(2024, 1, 7, 1, 30), 0.8*max(ax1.get_ylim()), "Outage \nEnds", fontsize=13)
    
    locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
    formatter = mdates.ConciseDateFormatter(locator)
    formatter.formats[2] = '%d\n %b'
    formatter.zero_formats[1] = '%b\n %Y'
    formatter.zero_formats[2] = '%d\n %b'
    formatter.zero_formats[3] = '%H:%M\n %d-%b'
    formatter.offset_formats[3] = '%b %Y'
    formatter.show_offset = False
    ax1.xaxis.set_major_locator(locator)
    ax1.xaxis.set_major_formatter(formatter)
   
    # Converts color_list into an iterable list for use in a loop
    iter_colour = iter(color_list)
    
    bp = Total_Unserved_Energy_Out.plot.bar(stacked=False, rot=0, edgecolor='black', 
                                            color=next(iter_colour), linewidth='0.1', 
                                            width=0.35, ax=ax2)
   
    ax2.spines['right'].set_visible(False)
#    ax2.spines['top'].set_visible(False)
    ax2.set_ylabel('Total Unserved Energy (MWh)',  color='black', rotation='vertical')
    ax2.set_xticks(ind)
    ax2.set_xticklabels(["Gas Outage + Icing"])
    ax2.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    ax2.margins(x=0.01)
    
    for i in ax2.patches:
       width, height = i.get_width(), i.get_height()
       if height<=1:
           continue
       x, y = i.get_xy() 
       ax2.text(x+width/2, 
            y+(height+100)/2, 
            '{:,.0f}'.format(height), 
            horizontalalignment='center', 
            verticalalignment='center', fontsize=13)
       
    fig5.savefig(figure_folder + "\\" + zone_input + "_Unserved_Energy", dpi=600, bbox_inches='tight') 



#===============================================================================
# CSV PLots Fuel Swicthing 
#===============================================================================  

fuel_switching = pd.read_csv(r"\\nrelqnap02\PLEXOS\Projects\NAERM\Draft_paper\PESGM_2020\Graphing_Data\fuel_switching.csv")
fuel_switching['timestamp'] = pd.to_datetime(fuel_switching['timestamp'])
fuel_switching.set_index('timestamp', inplace=True)


fig6, ax = plt.subplots(figsize=(9,6))

# Converts color_list into an iterable list for use in a loop
iter_colour = iter(color_list)

for column in fuel_switching:
    print(column)
    fp = ax.plot(fuel_switching[column], antialiased=True, 
                  linewidth=3 , color=next(iter_colour), label=column)
    ax.legend(loc='upper center',bbox_to_anchor=(0.5,-0.15), 
                 facecolor='inherit', frameon=True, ncol=2)


ax.set_ylabel('Real Time Fuel Offtake (MMBtu)',  color='black', rotation='vertical')
ax.set_ylim(bottom=0)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='y', which='major', length=5, width=1)
ax.tick_params(axis='x', which='major', length=5, width=1)
ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
ax.margins(x=0.01)


locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
formatter = mdates.ConciseDateFormatter(locator)
formatter.formats[2] = '%d\n %b'
formatter.zero_formats[1] = '%b\n %Y'
formatter.zero_formats[2] = '%d\n %b'
formatter.zero_formats[3] = '%H:%M\n %d-%b'
formatter.offset_formats[3] = '%b %Y'
formatter.show_offset = False
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)

ax.axvline(dt.datetime(2024, 1, 2, 14, 15), color='black', linestyle='--')
ax.text(dt.datetime(2024, 1, 2, 16, 30), 0.80*max(ax.get_ylim()), "Outage \nBegins", fontsize=13)
ax.axvline(dt.datetime(2024, 1, 9, 20, 35), color='black', linestyle='--')
ax.text(dt.datetime(2024, 1, 9, 22, 50), 0.80*max(ax.get_ylim()), "Outage \nEnds", fontsize=13)

fig6.savefig(figure_folder + "\\" + zone_input + "_Fuel_Offtake", dpi=600, bbox_inches='tight') 



#===============================================================================
# CSV PLots Tank Capacity
#===============================================================================  


tank_capacity = pd.read_csv(r"\\nrelqnap02\PLEXOS\Projects\NAERM\Draft_paper\PESGM_2020\Graphing_Data\tank_capacity.csv")
tank_capacity['timestamp'] = pd.to_datetime(tank_capacity['timestamp'])
tank_capacity.set_index('timestamp', inplace=True)


fig7, ax = plt.subplots(figsize=(9,6))
             
tc = ax.plot(tank_capacity, antialiased=True, 
             linewidth=3 , color=color_list[1])

ax.set_ylabel('Distillate Tank Capacity (GBtu)',  color='black', rotation='vertical')
ax.set_ylim(bottom=0, top=100)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='y', which='major', length=5, width=1)
ax.tick_params(axis='x', which='major', length=5, width=1)
ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
ax.margins(x=0.01)


locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
formatter = mdates.ConciseDateFormatter(locator)
formatter.formats[2] = '%d\n %b'
formatter.zero_formats[1] = '%b\n %Y'
formatter.zero_formats[2] = '%d\n %b'
formatter.zero_formats[3] = '%H:%M\n %d-%b'
formatter.offset_formats[3] = '%b %Y'
formatter.show_offset = False
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)

ax.axvline(dt.datetime(2024, 1, 7,), color='black', linestyle='--')
ax.text(dt.datetime(2024, 1, 5, 18), 0.80*max(ax.get_ylim()), "Tank \nRefuel", fontsize=13)
ax.axvline(dt.datetime(2024, 1, 14), color='black', linestyle='--')
ax.text(dt.datetime(2024, 1, 12, 18), 0.80*max(ax.get_ylim()), "Tank \nRefuel", fontsize=13)

fig7.savefig(figure_folder + "\\" + zone_input + "_Tank_Capacity", dpi=600, bbox_inches='tight') 



#===============================================================================
# CSV PLots DA vs RT
#===============================================================================  

DA_RT_gas = pd.read_csv(r"\\nrelqnap02\PLEXOS\Projects\NAERM\Draft_paper\PESGM_2020\Graphing_Data\DA-RT_Gas_Gen.csv")
DA_RT_gas['timestamp'] = pd.to_datetime(DA_RT_gas['timestamp'])
DA_RT_gas.set_index('timestamp', inplace=True)


fig8, ax = plt.subplots(figsize=(9,6))

# Converts color_list into an iterable list for use in a loop
iter_colour = iter(color_list)

for column in DA_RT_gas:
    fp = ax.plot(DA_RT_gas[column], antialiased=True, 
                  linewidth=3 , color=next(iter_colour), label=column)
    ax.legend(loc='upper center',bbox_to_anchor=(0.5,-0.15), 
                 facecolor='inherit', frameon=True, ncol=2)
    ax.fill_between(DA_RT_gas.index, DA_RT_gas[column])   
    
ax.set_ylabel('Generation (MW)',  color='black', rotation='vertical')
ax.set_ylim(bottom=0)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='y', which='major', length=5, width=1)
ax.tick_params(axis='x', which='major', length=5, width=1)
ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
ax.margins(x=0.01)


locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
formatter = mdates.ConciseDateFormatter(locator)
formatter.formats[2] = '%d\n %b'
formatter.zero_formats[1] = '%b\n %Y'
formatter.zero_formats[2] = '%d\n %b'
formatter.zero_formats[3] = '%H:%M\n %d-%b'
formatter.offset_formats[3] = '%b %Y'
formatter.show_offset = False
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)

ax.axvline(dt.datetime(2024, 1, 2, 13, 30), color='black', linestyle='--')
ax.text(dt.datetime(2024, 1, 2, 11), 0.80*max(ax.get_ylim()), "Outage \nBegins", fontsize=13)

fig8.savefig(figure_folder + "\\" + zone_input + "_DA-RT_Gas_Gen", dpi=600, bbox_inches='tight') 

##===============================================================================

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


Gen_Collection = {} 
Avail_Gen_Collection = {}
Curtailment_Collection = {}
Installed_Capacity_Collection = {} 
Total_Gen_Cost_Collection = {}
        
for scenario in Multi_Scenario:
    Installed_Capacity_Collection[scenario] = pd.read_hdf(PLEXOS_Scenarios + r"\\" + scenario + r"\Processed_HDF5_folder" + "/" + HDF5_output,  "generator_Installed_Capacity")
    Gen_Collection[scenario] = pd.read_hdf(PLEXOS_Scenarios + r"\\" + scenario + r"\Processed_HDF5_folder" + "/" + HDF5_output, "generator_Generation")
    Avail_Gen_Collection[scenario] = pd.read_hdf(PLEXOS_Scenarios + r"\\" + scenario + r"\Processed_HDF5_folder" + "/" + HDF5_output, "generator_Available_Capacity")
    Curtailment_Collection[scenario] = pd.read_hdf(PLEXOS_Scenarios + r"\\" + scenario + r"\Processed_HDF5_folder" + "/" + HDF5_output,  "generator_Curtailment")
    Total_Gen_Cost_Collection[scenario] = pd.read_hdf(PLEXOS_Scenarios + r"\\" + scenario + r"\Processed_HDF5_folder" + "/" + HDF5_output, "generator_Total_Generation_Cost")
    
        
Penetration_Curtailment_out = pd.DataFrame()

for zone_input in Zones: 
    print(zone_input)
    
    # Multi_Scenario = ['BAU_No_VG_Reserves']
    for scenario in Multi_Scenario:
        print("     " + scenario)
    
    
        gen = Gen_Collection.get(scenario)
        gen = gen.xs(zone_input,level=AGG_BY)
        
        avail_gen = Avail_Gen_Collection.get(scenario)
        avail_gen = avail_gen.xs(zone_input,level=AGG_BY) 
        
        re_curt = Curtailment_Collection.get(scenario)
        re_curt = re_curt.xs(zone_input,level=AGG_BY)

        # Finds the number of unique hours in the year
        no_hours_year = len(gen.index.unique(level="timestamp"))
        
        # Total generation across all technologies [MWh]
        total_gen = float(gen.sum())
        
        # Timeseries [MW] and Total VRE generation [MWh]
        vre_gen = (gen.loc[(slice(None), vre_gen_cat),:])
        total_vre_gen = float(vre_gen.sum())
        
        # Timeseries [MW] and Total RE generation [MWh]
        re_gen = (gen.loc[(slice(None), re_gen_cat),:])
        total_re_gen = float(re_gen.sum())
        
        # Timeseries [MW] and Total PV generation [MWh]
        pv_gen = (gen.loc[(slice(None), pv_gen_cat),:])
        total_pv_gen = float(pv_gen.sum())
        
        # % Penetration of generation classes across the year
        VRE_Penetration = (total_vre_gen/total_gen)*100
        RE_Penetration = (total_re_gen/total_gen)*100
        PV_Penetration = (total_pv_gen/total_gen)*100
        
        # Timeseries [MW] and Total RE available [MWh]
        re_avail = (avail_gen.loc[(slice(None), re_gen_cat),:])
        total_re_avail = float(re_avail.sum())
        
        # Timeseries [MW] and Total PV available [MWh]
        pv_avail = (avail_gen.loc[(slice(None), pv_gen_cat),:])
        total_pv_avail = float(pv_avail.sum())
    
        # Total RE curtailment [MWh]
        total_re_curt = float(re_curt.sum())
        
        # Timeseries [MW] and Total PV curtailment [MWh]
        pv_curt = (re_curt.loc[(slice(None), pv_gen_cat),:])
        total_pv_curt = float(pv_curt.sum())
        
        # % of hours with curtailment
        Prct_hr_RE_curt = (len((re_curt.sum(axis=1)).loc[(re_curt.sum(axis=1))>0])/no_hours_year)*100
        Prct_hr_PV_curt = (len((pv_curt.sum(axis=1)).loc[(pv_curt.sum(axis=1))>0])/no_hours_year)*100
        
        # Max instantaneous curtailment 
        Max_RE_Curt = max(re_curt.sum(axis=1))
        Max_PV_Curt = max(pv_curt.sum(axis=1))

        # % RE and PV Curtailment Capacity Factor
        RE_Curt_Cap_factor = (total_re_curt/Max_RE_Curt)/no_hours_year
        PV_Curt_Cap_factor = (total_pv_curt/Max_PV_Curt)/no_hours_year
        
        # % Curtailment across the year
        Prct_RE_curt = (total_re_curt/total_re_avail)*100
        Prct_PV_curt = (total_pv_curt/total_pv_avail)*100
        
        # Total generation cost
        Total_Gen_Cost = Total_Gen_Cost_Collection.get(scenario)
        Total_Gen_Cost = Total_Gen_Cost.xs(zone_input,level=AGG_BY)
        Total_Gen_Cost = float(Total_Gen_Cost.sum())
    
        
        vg_out = pd.Series([PV_Penetration ,RE_Penetration, VRE_Penetration, Max_PV_Curt, 
                            Max_RE_Curt, Prct_PV_curt, Prct_RE_curt, Prct_hr_PV_curt,
                            Prct_hr_RE_curt, PV_Curt_Cap_factor, RE_Curt_Cap_factor, Total_Gen_Cost], 
                           index=["% PV Penetration", "% RE Penetration", "% VRE Penetration",
                                  "Max PV Curtailment [MW]", "Max RE Curtailment [MW]",
                                  "% PV Curtailment", '% RE Curtailment',"% PV hrs Curtailed", 
                                  "% RE hrs Curtailed", "PV Curtailment Capacity Factor", 
                                  "RE Curtailment Capacity Factor", "Gen Cost"])
        vg_out = vg_out.rename(scenario)
        
        Penetration_Curtailment_out = pd.concat([Penetration_Curtailment_out, vg_out], axis=1, sort=False)
    
    Penetration_Curtailment_out = Penetration_Curtailment_out.T
        
    # Data table of values to return to main program
    Data_Table_Out = Penetration_Curtailment_out 
    
    VG_index = pd.Series(Penetration_Curtailment_out.index)
    # VG_index = VG_index.str.split(n=1, pat="_", expand=True)
    # VG_index.rename(columns = {0:"Scenario"}, inplace=True) 
    VG_index.rename("Scenario", inplace=True) 
    # VG_index = VG_index["Scenario"]
    Penetration_Curtailment_out.loc[:, "Scenario"] = VG_index[:,].values     
        
    marker_dict = dict(zip(VG_index.unique(), marker_style))
    colour_dict = dict(zip(VG_index.unique(), color_list))
    
    Penetration_Curtailment_out["colour"] = [colour_dict.get(x, '#333333') for x in Penetration_Curtailment_out.Scenario]
    Penetration_Curtailment_out["marker"] = [marker_dict.get(x, '+') for x in Penetration_Curtailment_out.Scenario]
    
    fig1, ax = plt.subplots(figsize=(9,6))
    for index, row in Penetration_Curtailment_out.iterrows():      
        # if self.prop == "PV":
        sp = ax.scatter(row["% PV Penetration"], row["% PV Curtailment"],
              marker=row["marker"],  c=row["colour"], s=100, label = row["Scenario"])
        ax.set_ylabel('% PV Curtailment',  color='black', rotation='vertical')
        ax.set_xlabel('% PV Penetration',  color='black', rotation='horizontal')
    
        # elif self.prop == "PV+Wind":
        #     sp = ax.scatter(row["% RE Penetration"], row["% RE Curtailment"],
        #           marker=row["marker"],  c=row["colour"], s=100, label = row["Scenario"])
        #     ax.set_ylabel('% PV + Wind Curtailment',  color='black', rotation='vertical')
        #     ax.set_xlabel('% PV + Wind Penetration',  color='black', rotation='horizontal')
    
    ax.set_ylim(bottom=0)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='y', which='major', length=5, width=1)
    ax.tick_params(axis='x', which='major', length=5, width=1)
    ax.margins(x=0.01)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())