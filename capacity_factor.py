# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 13:20:56 2019

This code creates total generation stacked bar plots and is called from Marmot_plot_main.py


@author: dlevie
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mtick
import numpy as np 
import os


#===============================================================================

def df_process_gen_inputs(df, self):
    df = df.reset_index()
    df['tech'].replace(self.gen_names_dict, inplace=True)
    df = df.groupby(["timestamp", "tech"], as_index=False).sum()
    df.tech = df.tech.astype("category")
    df.tech.cat.set_categories(self.ordered_gen, inplace=True)
    df = df.sort_values(["tech"]) 
    df = df.pivot(index='timestamp', columns='tech', values=0)
    return df  

class mplot(object):
        
    def __init__(self, argument_list):
        
        self.prop = argument_list[0]
        self.start_date = argument_list[4]
        self.end_date = argument_list[5]
        self.hdf_out_folder = argument_list[6]
        self.zone_input = argument_list[7]
        self.AGG_BY = argument_list[8]
        self.ordered_gen = argument_list[9]
        self.PLEXOS_color_dict = argument_list[10]
        self.Multi_Scenario = argument_list[11]
        self.PLEXOS_Scenarios = argument_list[13]
        self.ylabels = argument_list[14]
        self.xlabels = argument_list[15]
        self.color_list = argument_list[16]
        self.gen_names_dict = argument_list[18]
        
    def cf(self):
        # Create Dictionary to hold Datframes for each scenario 
        Gen_Collection = {} 
        Cap_Collection = {}
        
        for scenario in self.Multi_Scenario:
            Gen_Collection[scenario] = pd.read_hdf(os.path.join(self.PLEXOS_Scenarios, scenario,"Processed_HDF5_folder", scenario+ "_formatted.h5"),"generator_Generation")
            Cap_Collection[scenario] = pd.read_hdf(os.path.join(self.PLEXOS_Scenarios, scenario,"Processed_HDF5_folder", scenario+ "_formatted.h5"),"generator_Installed_Capacity")

        CF_all_scenarios = pd.DataFrame()
        print("     "+ self.zone_input)
            
        for scenario in self.Multi_Scenario:
            
            Gen = Gen_Collection.get(scenario)
            Gen = Gen.xs(self.zone_input,level = self.AGG_BY)
            Gen = df_process_gen_inputs(Gen,self)
            
            if self.prop == 'Date Range':
                print("Plotting specific date range:")
                print(self.start_date)
                print('    to')
                print(self.end_date)
                print('    ')
                
                Gen = Gen[self.start_date : self.end_date]
         
            # Calculates interval step to correct for MWh of generation
            time_delta = Gen.index[1] - Gen.index[0]
            duration = Gen.index[len(Gen)-1] - Gen.index[0]
            duration = duration + time_delta #Account for last timestep.
            # Finds intervals in 60 minute period
            interval_count = 60/(time_delta/np.timedelta64(1, 'm'))
            duration_hours = duration/np.timedelta64(1,'h')     #Get length of time series in hours for CF calculation.     
                
            Gen = Gen/interval_count
            Total_Gen = Gen.sum(axis=0)
            Total_Gen.rename(scenario, inplace = True)
            
            Cap = Cap_Collection.get(scenario)
            Cap = Cap.xs(self.zone_input,level = self.AGG_BY)
            Cap = df_process_gen_inputs(Cap, self)
            Cap = Cap.T.sum(axis = 1)  #Rotate and force capacity to a series.
            Cap.rename(scenario, inplace = True)
                      
            #Calculate CF
            CF = Total_Gen/(Cap * duration_hours) 
            CF.rename(scenario, inplace = True)
            CF_all_scenarios = pd.concat([CF_all_scenarios, CF], axis=1, sort=False)
            CF_all_scenarios = CF_all_scenarios.dropna(axis = 0)

        CF_all_scenarios.index = CF_all_scenarios.index.str.wrap(10, break_long_words = False)
        
        fig1 = CF_all_scenarios.plot.bar(stacked = False, figsize=(9,6), rot=0, 
                             color = self.color_list,edgecolor='black', linewidth='0.1')
        
        fig1.spines['right'].set_visible(False)
        fig1.spines['top'].set_visible(False)
        fig1.set_ylabel('Capacity Factor',  color='black', rotation='vertical')
        #adds % to y axis data 
        fig1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        fig1.tick_params(axis='y', which='major', length=5, width=1)
        fig1.tick_params(axis='x', which='major', length=5, width=1)
                           
        return {'fig': fig1, 'data_table': CF_all_scenarios}
    
def df_process_gen_inputs(df):
    df = df.reset_index()
    df['tech'].replace(gen_names_dict, inplace=True)
    df = df.groupby(["timestamp","tech"], as_index=False).sum()
    df.tech = df.tech.astype("category")
    df.tech.cat.set_categories(ordered_gen, inplace=True)
    df = df.sort_values(["tech"]) 
    df = df.pivot(index='timestamp', columns='tech', values=0)
    return df  
   
    
    
    

    def avg_output_when_committed(self):
        # Create Dictionary to hold Datframes for each scenario 
        Gen_Collection = {} 
        Cap_Collection = {}
        
        for scenario in Multi_Scenario:
            Gen_Collection[scenario] = pd.read_hdf(os.path.join(PLEXOS_Scenarios, scenario,"Processed_HDF5_folder", scenario+ "_formatted.h5"),"generator_Generation")
            Cap_Collection[scenario] = pd.read_hdf(os.path.join(PLEXOS_Scenarios, scenario,"Processed_HDF5_folder", scenario+ "_formatted.h5"),"generator_Installed_Capacity")

        Gen_all_scenarios = pd.DataFrame()
        Cap_all_scenarios = pd.DataFrame()
        print("     "+ self.zone_input)
            
        for scenario in self.Multi_Scenario:
            
            Gen = Gen_Collection.get(scenario)
            Gen = Gen.xs(zone_input,level = AGG_BY)
            Cap = Cap_Collection.get(scenario)
            Cap = Cap.xs(zone_input,level = AGG_BY)
            
            
            
            Gen
            
            Gen = df_process_gen_inputs(Gen)
            
            if self.prop == 'Date Range':
                print("Plotting specific date range:")
                print(self.start_date)
                print('    to')
                print(self.end_date)
                print('    ')
                
                Gen = Gen[self.start_date : self.end_date]
                
                
                       
            # Calculates interval step to correct for MWh of generation
            time_delta = Gen.index[1] - Gen.index[0]
            duration = Gen.index[len(Gen)-1] - Gen.index[0]
            duration = duration + time_delta #Account for last timestep.
            # Finds intervals in 60 minute period
            interval_count = 60/(time_delta/np.timedelta64(1, 'm'))
            duration_hours = duration/np.timedelta64(1,'h')     #Get length of time series in hours for CF calculation.

            Gen = Gen/interval_count
            
            
         
        
    #     Gen_Collection = {} 
    #     Load_Collection = {}
    #     Curtailment_Collection = {}
        
    #     for scenario in self.Multi_Scenario:
    #         try:
    #             Gen_Collection[scenario] = pd.read_hdf(os.path.join(self.PLEXOS_Scenarios, scenario,"Processed_HDF5_folder", scenario+ "_formatted.h5"), "generator_Generation")
    #             Curtailment_Collection[scenario] = pd.read_hdf(os.path.join(self.PLEXOS_Scenarios, scenario,"Processed_HDF5_folder", scenario+ "_formatted.h5"),  "generator_Curtailment")
    #             # If data is to be agreagted by zone, then zone properties are loaded, else region properties are loaded
    #             if self.AGG_BY == "zone":
    #                 Load_Collection[scenario] = pd.read_hdf(os.path.join(self.PLEXOS_Scenarios, scenario,"Processed_HDF5_folder", scenario+ "_formatted.h5"), "zone_Load")
    #             else:
    #                 Load_Collection[scenario] = pd.read_hdf(os.path.join(self.PLEXOS_Scenarios, scenario,"Processed_HDF5_folder", scenario+ "_formatted.h5"),  "region_Load")
                
    #         except Exception:
    #             pass
            
    
    #     Total_Generation_Stack_Out = pd.DataFrame()
    #     Total_Load_Out = pd.DataFrame()
    #     print("     "+ self.zone_input)
        
        
    #     for scenario in self.Multi_Scenario:      
    #         print("     " + scenario)
    #         try:
    #             Total_Gen_Stack = Gen_Collection.get(scenario)
    #             Total_Gen_Stack = Total_Gen_Stack.xs(self.zone_input,level=self.AGG_BY)
    #             Total_Gen_Stack = df_process_gen_inputs(Total_Gen_Stack, self)            
    #             Stacked_Curt = Curtailment_Collection.get(scenario)
    #             Stacked_Curt = Stacked_Curt.xs(self.zone_input,level=self.AGG_BY)
    #             Stacked_Curt = df_process_gen_inputs(Stacked_Curt, self)
    #             Stacked_Curt = Stacked_Curt.sum(axis=1)
    #             Total_Gen_Stack.insert(len(Total_Gen_Stack.columns),column='Curtailment',value=Stacked_Curt) #Insert curtailment into 
    #             Total_Gen_Stack = Total_Gen_Stack.loc[:, (Total_Gen_Stack != 0).any(axis=0)]
                
    #             Total_Gen_Stack = Total_Gen_Stack.sum(axis=0)
    #             Total_Gen_Stack.rename(scenario, inplace=True)
    
    #             Total_Generation_Stack_Out = pd.concat([Total_Generation_Stack_Out, Total_Gen_Stack], axis=1, sort=False).fillna(0)
            
    #             Total_Load = Load_Collection.get(scenario)
    #             Total_Load = Total_Load.xs(self.zone_input,level=self.AGG_BY)
    #             Total_Load = Total_Load.groupby(["timestamp"]).sum()
    #             Total_Load = Total_Load.rename(columns={0:scenario}).sum(axis=0)
    #             Total_Load_Out = pd.concat([Total_Load_Out, Total_Load], axis=0, sort=False)
    #         except Exception:
    #             print("Error: Skipping " + scenario)
    #             pass
        
    #     Total_Load_Out = Total_Load_Out.rename(columns={0:'Total Load'})
        
    #     Total_Generation_Stack_Out = df_process_categorical_index(Total_Generation_Stack_Out, self)
    #     Total_Generation_Stack_Out = Total_Generation_Stack_Out.T/1000 
    #     Total_Generation_Stack_Out = Total_Generation_Stack_Out.loc[:, (Total_Generation_Stack_Out != 0).any(axis=0)]
        
    #     # Data table of values to return to main program
    #     Data_Table_Out = pd.concat([Total_Load_Out/1000, Total_Generation_Stack_Out],  axis=1, sort=False)
        
    #     Total_Generation_Stack_Out.index = Total_Generation_Stack_Out.index.str.replace('_',' ')
    #     Total_Generation_Stack_Out.index = Total_Generation_Stack_Out.index.str.wrap(11, break_long_words=False)
        
    #     Total_Load_Out.index = Total_Load_Out.index.str.replace('_',' ')
    #     Total_Load_Out.index = Total_Load_Out.index.str.wrap(11, break_long_words=False)
    
    #     Total_Load_Out = Total_Load_Out.T/1000             
    
    #     xdimension=len(self.xlabels)
    #     ydimension=len(self.ylabels)
    #     grid_size = xdimension*ydimension
    
    #     fig2, axs = plt.subplots(ydimension,xdimension, figsize=((2*xdimension), (4*ydimension)), sharey=True)
    #     axs = axs.ravel()
    #     plt.subplots_adjust(wspace=0, hspace=0.01)
        
    #     i=0
    #     for index in Total_Generation_Stack_Out.index:
        
    #         sb = Total_Generation_Stack_Out.iloc[i:i+1].plot.bar(stacked=True, rot=0,
    #         color=[self.PLEXOS_color_dict.get(x, '#333333') for x in Total_Generation_Stack_Out.columns], edgecolor='black', linewidth='0.1', 
    #                                      ax=axs[i])
                
    #         axs[i].get_legend().remove()
    #         axs[i].spines['right'].set_visible(False)
    #         axs[i].spines['top'].set_visible(False)
    #         axs[i].xaxis.set_ticklabels([])
    #         axs[i].tick_params(axis='y', which='major', length=5, width=1)
    #         axs[i].tick_params(axis='x', which='major', length=5, width=1)
    #         axs[i].yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    #         axs[i].margins(x=0.01)
            
    #         height = [int(Total_Load_Out[index])]
    #         axs[i].axhline(y=height,xmin=0.25,xmax=0.75, linestyle ='--', c="black",linewidth=1.5)
            
    #         handles, labels = axs[1].get_legend_handles_labels()
            
    #         leg1 = axs[grid_size-1].legend(reversed(handles), reversed(labels), loc='lower left',bbox_to_anchor=(1,0), 
    #                   facecolor='inherit', frameon=True)
            
    #         #Legend 2
    #         leg2 = axs[grid_size-1].legend(['Load'], loc='upper left',bbox_to_anchor=(1, 0.95), 
    #                       facecolor='inherit', frameon=True)
            
    #         fig2.add_artist(leg1)
            
    #         i=i+1
     
    #     all_axes = fig2.get_axes()
        
    #     self.xlabels = pd.Series(self.xlabels).str.replace('_',' ').str.wrap(10, break_long_words=False)
        
    #     j=0
    #     k=0
    #     for ax in all_axes:
    #         if ax.is_last_row():
    #             ax.set_xlabel(xlabel=(self.xlabels[j]),  color='black')
    #             j=j+1
    #         if ax.is_first_col(): 
    #             ax.set_ylabel(ylabel=(self.ylabels[k]),  color='black', rotation='vertical')
    #             k=k+1
                
    #     fig2.add_subplot(111, frameon=False)
    #     plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    #     plt.ylabel('Total Genertaion (GWh)',  color='black', rotation='vertical', labelpad=60)
        
    #     return {'fig': fig2, 'data_table': Data_Table_Out}
        
        