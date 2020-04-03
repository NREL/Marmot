# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 10:34:48 2019

This code creates generation stack plots and is called from Marmot_plot_main.py

@author: dlevie
"""

import os
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates



#===============================================================================

class mplot(object):
    def __init__(self,argument_list):
        
        self.prop = argument_list[0]
        self.start = argument_list[1]     
        self.end = argument_list[2]
        self.timezone = argument_list[3]
        self.start_date = argument_list[4]
        self.end_date = argument_list[5]
        self.hdf_out_folder = argument_list[6]
        self.zone_input =argument_list[7]
        self.AGG_BY = argument_list[8]
        self.ordered_gen = argument_list[9]
        self.PLEXOS_color_dict = argument_list[10]
        self.Multi_Scenario = argument_list[11]
        self.Scenario_Diff = argument_list[12]
        self.PLEXOS_Scenarios = argument_list[13]
        self.ylabels = argument_list[14]
        self.xlabels = argument_list[15]
        self.color_list = argument_list[16]
        self.gen_names_dict = argument_list[18]
        self.re_gen_cat = argument_list[20]
        
    
    def net_export(self):
        
        print("Zone = " + self.zone_input)
        
        Net_Export_all_scenarios = pd.DataFrame()

        for scenario in self.Multi_Scenario:
            
            print("Scenario = " + str(scenario))
    
            Net_Export_read = pd.read_hdf(os.path.join(self.PLEXOS_Scenarios,scenario, 'Processed_HDF5_folder', scenario + '_formatted.h5'),'region_Net_Interchange')
            Net_Export = Net_Export_read.xs(self.zone_input, level = self.AGG_BY)
            Net_Export = Net_Export.reset_index()
            Net_Export = Net_Export.groupby(["timestamp"]).sum()
            Net_Export.columns = [scenario]
            
            if self.prop == 'Date Range':
                print("Plotting specific date range:")
                print(str(self.start_date) + '  to  ' + str(self.end_date))
                
                Net_Export = Net_Export[self.start_date : self.end_date]

            Net_Export_all_scenarios = pd.concat([Net_Export_all_scenarios,Net_Export], axis = 1) 
            
        # Data table of values to return to main program
        Data_Table_Out = Net_Export_all_scenarios
        
        #Make scenario/color dictionary.
        scenario_color_dict = {}
        for idx,column in enumerate(Net_Export_all_scenarios.columns):
            dictionary = {column : self.color_list[idx]}
            scenario_color_dict.update(dictionary)
            
        fig1, ax = plt.subplots(figsize=(9,6))
        for idx,column in enumerate(Net_Export_all_scenarios.columns):
            ax.plot(Net_Export_all_scenarios.index.values,Net_Export_all_scenarios[column], linewidth=2, color = scenario_color_dict.get(column,'#333333'),label=column)

        
        ax.set_ylabel('Net exports (MW)',  color='black', rotation='vertical')
        ax.set_xlabel('Date ' + '(' + self.timezone + ')',  color='black', rotation='horizontal')
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
        leg1 = ax.legend(reversed(handles), reversed(labels), loc='best',facecolor='inherit', frameon=True)  
        
        # Manually add the first legend back
        ax.add_artist(leg1)

        return {'fig': fig1, 'data_table': Data_Table_Out}
    
  
            