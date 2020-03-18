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
            #interval_count = 60/(time_delta/np.timedelta64(1, 'm'))
            duration_hours = duration/np.timedelta64(1,'h')     #Get length of time series in hours for CF calculation.     
                
            #Gen = Gen/interval_count
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
    
    
    def avg_output_when_committed(self):

        # Create Dictionary to hold Datframes for each scenario 
        Gen_Collection = {} 
        Cap_Collection = {}
        
        for scenario in Multi_Scenario:
            Gen_Collection[scenario] = pd.read_hdf(os.path.join(PLEXOS_Scenarios, scenario,"Processed_HDF5_folder", scenario + "_formatted.h5"),"generator_Generation")
            Cap_Collection[scenario] = pd.read_hdf(os.path.join(PLEXOS_Scenarios, scenario,"Processed_HDF5_folder", scenario + "_formatted.h5"),"generator_Installed_Capacity")

        
        CF_all_scenarios = pd.DataFrame()
        
        print("Processing zone " + zone_input)
            
        for scenario in Multi_Scenario:
            print(scenario)
            Gen = Gen_Collection.get(scenario)
            Gen = Gen.xs(zone_input,level = AGG_BY)
      
            Gen = Gen.reset_index()
            Gen.tech = Gen.tech.astype("category")
            Gen.tech.cat.set_categories(ordered_gen, inplace=True)
            Gen = Gen.drop(columns = ['region'])
            Gen = Gen.rename(columns = {0:"Output (MWh)"})
    
            
            Cap = Cap_Collection.get(scenario)
            Cap = Cap.xs(zone_input,level =AGG_BY)
            Cap = Cap.reset_index()
            Cap = Cap.drop(columns = ['timestamp','region','tech'])
            Cap = Cap.rename(columns = {0:"Installed Capacity (MW)"})
            Gen = pd.merge(Gen,Cap, on = 'gen_name')
            
            if self.prop == 'Date Range':
                print("Plotting specific date range:")
                print(self.start_date)
                print('    to')
                print(self.end_date)
                print('    ')
                
                             
                date_range = pd.date_range(start = start_date, end = end_date)
                date_range_str = [str(date) for date in date_range]
                Gen = Gen.loc[Gen.timestamp == start_date : Gen.timestamp == end_date]
           
            #Calculate CF individually for each plant, since we need to take out all zero rows.
            tech_names = Gen['tech'].unique()
            CF = pd.DataFrame(columns = tech_names,index = [scenario])
            for tech_name in tech_names:
                stt = Gen.loc[Gen['tech'] == tech_name]
                if all(stt['Output (MWh)'] == 0):
                      print('No ' + tech_name + ' generators produced energy.')
                      
                else: 
                    
                    gen_names = stt['gen_name'].unique()
                    
                    cfs = []
                    caps = []
                    for gen in gen_names:
                        sgt = stt.loc[stt['gen_name'] == gen]
                        if all(sgt['Output (MWh)'] == 0):
                            print(gen + ' was off the entire time.')
                            
                        else: 
                            time_delta = sgt.timestamp.iloc[1] - sgt.timestamp.iloc[0]  # Calculates interval step to correct for MWh of generation.
                            sgt = sgt[sgt['Output (MWh)'] !=0] #Remove time intervals when output is zero.
                            duration_hours = (len(sgt) * time_delta + time_delta)/np.timedelta64(1,'h')     #Get length of time series in hours for CF calculation
                            total_gen = sgt['Output (MWh)'].sum()
                            cap = sgt['Installed Capacity (MW)'].mean()
                                              
                            #Calculate CF
                            cf = total_gen/(cap * duration_hours) 
                            cfs.append(cf)
                            caps.append(cap)
                    
                    #Find average "CF" (average output when committed) for this technology, weighted by capacity.
                    cf = np.average(cfs,weights = caps)
                    CF[tech_name] = cf
                
            CF_all_scenarios = CF_all_scenarios.append(CF) 
            
        fig2 = CF_all_scenarios.T.plot.bar(stacked = False, figsize=(9,6), rot=0, 
                             color = self.color_list,edgecolor='black', linewidth='0.1')
        
        fig2.spines['right'].set_visible(False)
        fig2.spines['top'].set_visible(False)
        fig2.set_ylabel('Average Output When Committed',  color='black', rotation='vertical')
        #adds % to y axis data 
        fig2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        fig2.tick_params(axis='y', which='major', length=5, width=1)
        fig2.tick_params(axis='x', which='major', length=5, width=1)
                          
        return {'fig': fig2, 'data_table': CF_all_scenarios.T}


def time_at_min_gen(self):

        # Create Dictionary to hold Datframes for each scenario 
        Gen_Collection = {} 
        Cap_Collection = {}
        
        for scenario in Multi_Scenario:
            Gen_Collection[scenario] = pd.read_hdf(os.path.join(PLEXOS_Scenarios, scenario,"Processed_HDF5_folder", scenario + "_formatted.h5"),"generator_Generation")
            Cap_Collection[scenario] = pd.read_hdf(os.path.join(PLEXOS_Scenarios, scenario,"Processed_HDF5_folder", scenario + "_formatted.h5"),"generator_Installed_Capacity")

        
        CF_all_scenarios = pd.DataFrame()
        
        print("Processing zone " + zone_input)
            
        for scenario in self.Multi_Scenario:
            print(scenario)
            Gen = Gen_Collection.get(scenario)
            Gen = Gen.xs(zone_input,level = AGG_BY)
      
            Gen = Gen.reset_index()
            Gen.tech = Gen.tech.astype("category")
            Gen.tech.cat.set_categories(ordered_gen, inplace=True)
            Gen = Gen.drop(columns = ['region'])
            Gen = Gen.rename(columns = {0:"Output (MWh)"})
    
            
            Cap = Cap_Collection.get(scenario)
            Cap = Cap.xs(zone_input,level =AGG_BY)
            Cap = Cap.reset_index()
            Cap = Cap.drop(columns = ['timestamp','region','tech'])
            Cap = Cap.rename(columns = {0:"Installed Capacity (MW)"})
            Gen = pd.merge(Gen,Cap, on = 'gen_name')
            
            if self.prop == 'Date Range':
                print("Plotting specific date range:")
                print(self.start_date)
                print('    to')
                print(self.end_date)
                print('    ')
                
                             
                date_range = pd.date_range(start = start_date, end = end_date)
                date_range_str = [str(date) for date in date_range]
                Gen = Gen.loc[Gen.timestamp == start_date : Gen.timestamp == end_date]
           
            #Calculate CF individually for each plant, since we need to take out all zero rows.
            tech_names = Gen['tech'].unique()
            CF = pd.DataFrame(columns = tech_names,index = [scenario])
            for tech_name in tech_names:
                stt = Gen.loc[Gen['tech'] == tech_name]
                if all(stt['Output (MWh)'] == 0):
                      print('No ' + tech_name + ' generators produced energy.')
                      
                else: 
                    
                    gen_names = stt['gen_name'].unique()
                    
                    cfs = []
                    caps = []
                    for gen in gen_names:
                        sgt = stt.loc[stt['gen_name'] == gen]
                        if all(sgt['Output (MWh)'] == 0):
                            print(gen + ' was off the entire time.')
                            
                        else: 
                            time_delta = sgt.timestamp.iloc[1] - sgt.timestamp.iloc[0]  # Calculates interval step to correct for MWh of generation.
                            sgt = sgt[sgt['Output (MWh)'] !=0] #Remove time intervals when output is zero.
                            duration_hours = (len(sgt) * time_delta + time_delta)/np.timedelta64(1,'h')     #Get length of time series in hours for CF calculation
                            total_gen = sgt['Output (MWh)'].sum()
                            cap = sgt['Installed Capacity (MW)'].mean()
                                              
                            #Calculate CF
                            cf = total_gen/(cap * duration_hours) 
                            cfs.append(cf)
                            caps.append(cap)
                    
                    #Find average "CF" (average output when committed) for this technology, weighted by capacity.
                    cf = np.average(cfs,weights = caps)
                    CF[tech_name] = cf
                
            CF_all_scenarios = CF_all_scenarios.append(CF) 
            
        fig2 = CF_all_scenarios.T.plot.bar(stacked = False, figsize=(9,6), rot=0, 
                             color = self.color_list,edgecolor='black', linewidth='0.1')
        
        fig2.spines['right'].set_visible(False)
        fig2.spines['top'].set_visible(False)
        fig2.set_ylabel('Average Output When Committed',  color='black', rotation='vertical')
        #adds % to y axis data 
        fig2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        fig2.tick_params(axis='y', which='major', length=5, width=1)
        fig2.tick_params(axis='x', which='major', length=5, width=1)
                          
        return {'fig': fig2, 'data_table': CF_all_scenarios.T}
                
     
        