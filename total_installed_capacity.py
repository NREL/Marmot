# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 08:51:15 2019

@author: dlevie
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


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
        self.hdf_out_folder = argument_list[6]
        self.zone_input = argument_list[7]
        self.AGG_BY = argument_list[8]
        self.ordered_gen = argument_list[9]
        self.PLEXOS_color_dict = argument_list[10]
        self.Multi_Scenario = argument_list[11]
        self.PLEXOS_Scenarios = argument_list[13]
        self.gen_names_dict = argument_list[18]
        
    def total_cap(self):
        # Create Dictionary to hold Datframes for each scenario 
        Installed_Capacity_Collection = {} 
        
        for scenario in self.Multi_Scenario:
            Installed_Capacity_Collection[scenario] = pd.read_hdf(self.PLEXOS_Scenarios + r"\\" + scenario + r"\Processed_HDF5_folder" + "/" + scenario+"_formatted.h5",   "generator_Installed_Capacity")
            
            
        Total_Installed_Capacity_Out = pd.DataFrame()
        Data_Table_Out = pd.DataFrame()
        print("     " + self.zone_input)
        
        for scenario in self.Multi_Scenario:
            
            Total_Installed_Capacity = Installed_Capacity_Collection.get(scenario)
            Total_Installed_Capacity = Total_Installed_Capacity.xs(self.zone_input,level=self.AGG_BY)
            Total_Installed_Capacity = df_process_gen_inputs(Total_Installed_Capacity, self)
            Total_Installed_Capacity.reset_index(drop=True, inplace=True)
            Total_Installed_Capacity.rename(index={0:scenario}, inplace=True)
            Total_Installed_Capacity_Out = pd.concat([Total_Installed_Capacity_Out, Total_Installed_Capacity], axis=0, sort=False).fillna(0)
        

        Total_Installed_Capacity_Out = Total_Installed_Capacity_Out/1000
        Total_Installed_Capacity_Out = Total_Installed_Capacity_Out.loc[:, (Total_Installed_Capacity_Out != 0).any(axis=0)]
        
        # Data table of values to return to main program
        Data_Table_Out = pd.concat([Data_Table_Out, Total_Installed_Capacity_Out],  axis=1, sort=False)
        
        Total_Installed_Capacity_Out.index = Total_Installed_Capacity_Out.index.str.replace('_',' ')
        Total_Installed_Capacity_Out.index = Total_Installed_Capacity_Out.index.str.wrap(10, break_long_words=False)
        
        
        fig1 = Total_Installed_Capacity_Out.plot.bar(stacked=True, figsize=(9,6), rot=0, 
                             color=[self.PLEXOS_color_dict.get(x, '#333333') for x in Total_Installed_Capacity_Out.columns], edgecolor='black', linewidth='0.1')
       
        fig1.spines['right'].set_visible(False) 
        fig1.spines['top'].set_visible(False)
        fig1.set_ylabel('Total Installed Capacity (GW)',  color='black', rotation='vertical')
        #adds comma to y axis data 
        fig1.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
        fig1.tick_params(axis='y', which='major', length=5, width=1) 
        fig1.tick_params(axis='x', which='major', length=5, width=1)
    
        handles, labels = fig1.get_legend_handles_labels()
        leg1 = fig1.legend(reversed(handles), reversed(labels), loc='lower left',bbox_to_anchor=(1,0), 
                      facecolor='inherit', frameon=True)
        
    
        return {'fig': fig1, 'data_table': Data_Table_Out}