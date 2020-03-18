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
        
    def capacity_started(self):
        # Create Dictionary to hold Datframes for each scenario 
        Gen_Collection = {}
        Cap_Collection = {}
        
        for scenario in self.Multi_Scenario:
            Gen_Collection[scenario] = pd.read_hdf(os.path.join(self.PLEXOS_Scenarios, scenario,"Processed_HDF5_folder", scenario+ "_formatted.h5"),"generator_Generation")
            Cap_Collection[scenario] = pd.read_hdf(os.path.join(self.PLEXOS_Scenarios, scenario,"Processed_HDF5_folder", scenario+ "_formatted.h5"),"generator_Installed_Capacity")


        print("Processing " + self.zone_input)
        
        cap_started_all_scenarios = pd.DataFrame()
        
        for scenario in self.Multi_Scenario:
            
            Gen = Gen_Collection.get(scenario)
            Gen = Gen.xs(self.zone_input,level = self.AGG_BY)
      
            Gen = Gen.reset_index()
            Gen.tech = Gen.tech.astype("category")
            Gen.tech.cat.set_categories(self.ordered_gen, inplace=True)
            Gen = Gen.drop(columns = ['region'])
            Gen = Gen.rename(columns = {0:"Output (MW)"})
            Gen = Gen[~Gen['tech'].isin(['PV','Wind','Hydro','CSP'])]   #We are only interested in thermal starts/stops.
            
            Cap = Cap_Collection.get(scenario)
            Cap = Cap.xs(self.zone_input,level = self.AGG_BY)
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
                
                Gen = Gen[self.start_date : self.end_date]
            
            tech_names = Gen['tech'].unique()
            Cap_started = pd.DataFrame(columns = tech_names,index = [scenario])
            
            for tech_name in tech_names:
                stt = Gen.loc[Gen['tech'] == tech_name]
            
                gen_names = stt['gen_name'].unique()
                
                cap_started = 0
                      
   
                for gen in gen_names:
                    sgt = stt.loc[stt['gen_name'] == gen]
                    if any(sgt["Output (MW)"] == 0) and not all(sgt["Output (MW)"] == 0):   #Check that this generator has some, but not all, uncommited hours.
                        #print('Couting starts for: ' + gen)
                        for idx in range(len(sgt['timestamp']) - 1):
                                if sgt["Output (MW)"].iloc[idx] == 0 and not sgt["Output (MW)"].iloc[idx + 1] == 0:
                                    cap_started = cap_started + sgt["Installed Capacity (MW)"].iloc[idx] 
                                  # print('started on '+ timestamp)
                                # if sgt[0].iloc[idx] == 0 and not idx == 0 and not sgt[0].iloc[idx - 1] == 0:
                                #     stops = stops + 1
                                      
                Cap_started[tech_name] = cap_started

            cap_started_all_scenarios = cap_started_all_scenarios.append(Cap_started)
        
                # import time
                # start = time.time()
                # for gen in gen_names:
                #     sgt = stt.loc[stt['gen_name'] == gen]
                    
                #     if any(sgt[0] == 0) and not all(sgt[0] == 0):   #Check that this generator has some, but not all, uncommited hours.
                #         zeros = sgt.loc[sgt[0] == 0]
                                       
                #         print('Couting starts and stops for: ' + gen)
                #         for idx in range(len(zeros['timestamp']) - 1):
                #                if not zeros['timestamp'].iloc[idx + 1] == pd.Timedelta(1,'h'):
                #                    starts = starts + 1 
                #                   # print('started on '+ timestamp)
                #                if not zeros['timestamp'].iloc[idx - 1] == pd.Timedelta(1,'h'):
                #                    stops = stops + 1
                                      
                # starts_and_stops = [starts,stops]
                # counts[tech_name] = starts_and_stops
            
            
            # end = time.time()
            # elapsed = end - start
            # print('Method 2 (first making a data frame with only 0s, then checking if timestamps > 1 hour) took ' + str(elapsed) + ' seconds')
                    
                
        fig1 = cap_started_all_scenarios.plot.bar(stacked = False, figsize=(9,6), rot=0, 
                             color = self.color_list,edgecolor='black', linewidth='0.1')
        
        fig1.spines['right'].set_visible(False)
        fig1.spines['top'].set_visible(False)
        fig1.set_ylabel('Capacity Started (MW-starts)',  color='black', rotation='vertical')
        fig1.tick_params(axis='y', which='major', length=5, width=1)
        fig1.tick_params(axis='x', which='major', length=5, width=1)
                           
        return {'fig': fig1, 'data_table': cap_started_all_scenarios}
    

    
        
        