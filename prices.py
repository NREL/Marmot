# -*- coding: utf-8 -*-
"""

price analysis

@author: adyreson
"""

import os
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
import numpy as np 



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
        
       
  
    def price_region(self):          #Duration curve of individual region prices 
        
        Price_Collection = {}        # Create Dictionary to hold Datframes for each scenario 
        
        for scenario in self.Multi_Scenario:
            Price_Collection[scenario] = pd.read_hdf(os.path.join(self.PLEXOS_Scenarios, scenario,"Processed_HDF5_folder", scenario+ "_formatted.h5"),"region_Price")

              
        print("Price analysis done only once (includes all regions).")
        
        fig3, ax3 = plt.subplots(len(self.Multi_Scenario),figsize=(9,6)) # Set up subplots for all scenarios
     
        n=0 #Counter for scenario subplots
        
        for scenario in self.Multi_Scenario:
            
            print("Scenario = " + str(scenario))
            
            Price = Price_Collection.get(scenario)

            for region in Price.index.get_level_values(level='region').unique() :
                duration_curve = Price.xs(region,level="region").sort_values(by=0,ascending=False).reset_index()
                        
                if len(self.Multi_Scenario)>1:                  #Multi scenario
                    if duration_curve[0].max()>1e6:             #Only label curves with high prices
                        ax3[n].plot(duration_curve[0],label=region)
                        ax3[n].legend(loc='upper right')
                    else:
                        ax3[n].plot(duration_curve[0])

                    ax3[n].set_ylabel(scenario+'\n'+'Region Price $/MWh ',  color='black', rotation='vertical')
                    ax3[n].set_xlabel('Intervals',  color='black', rotation='horizontal')
                    ax3[n].spines['right'].set_visible(False)
                    ax3[n].spines['top'].set_visible(False)                         
                   
                    if (self.prop!=self.prop)==False: # This checks for a nan in string. If no limit selected, do nothing
                        ax3[n].set_ylim((0,int(self.prop)))           


                else: #Single scenario
                    if duration_curve[0].max()>1e6: #only label regions with high prices
                        ax3.plot(duration_curve[0],label=region)
                        ax3.legend(loc='upper right')
                    else:    
                        ax3.plot(duration_curve[0])
                    
                    ax3.set_ylabel(scenario+'\n'+'Region Price $/MWh ',  color='black', rotation='vertical')
                    ax3.set_xlabel('Intervals',  color='black', rotation='horizontal')
                    ax3.spines['right'].set_visible(False)
                    ax3.spines['top'].set_visible(False)   

                   
                    if (self.prop!=self.prop)==False: # This checks for a nan in string. If no limit selected, do nothing
                        plt.ylim((0,int(self.prop)))   
                       

                
                del duration_curve
            del Price 
            
               
            n=n+1
        #end scenario loop
                              
        return {'fig': fig3}
    
    def price_region_chron(self):          #Timeseries of individual region prices 
        
        Price_Collection = {}        # Create Dictionary to hold Datframes for each scenario 
        
        for scenario in self.Multi_Scenario:
            Price_Collection[scenario] = pd.read_hdf(os.path.join(self.PLEXOS_Scenarios, scenario,"Processed_HDF5_folder", scenario+ "_formatted.h5"),"region_Price")

              
        print("Price analysis done only once (includes all regions).")
        
        fig3, ax3 = plt.subplots(len(self.Multi_Scenario),figsize=(9,6)) # Set up subplots for all scenarios
     
        n=0 #Counter for scenario subplots
        
        for scenario in self.Multi_Scenario:
            
            print("Scenario = " + str(scenario))
            
            Price = Price_Collection.get(scenario)

            for region in Price.index.get_level_values(level='region').unique() :
                timeseries = Price.xs(region,level="region").reset_index().set_index('timestamp')
                        
                if len(self.Multi_Scenario)>1:
                    ax3[n].plot(timeseries[0])
                    ax3[n].set_ylabel(scenario+'\n'+'Region Price $/MWh ',  color='black', rotation='vertical')
                    ax3[n].spines['right'].set_visible(False)
                    ax3[n].spines['top'].set_visible(False)                         
                   
                    locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
                    formatter = mdates.ConciseDateFormatter(locator)
                    formatter.formats[2] = '%d\n %b'
                    formatter.zero_formats[1] = '%b\n %Y'
                    formatter.zero_formats[2] = '%d\n %b'
                    formatter.zero_formats[3] = '%H:%M\n %d-%b'
                    formatter.offset_formats[3] = '%b %Y'
                    formatter.show_offset = False
                    ax3[n].xaxis.set_major_locator(locator)
                    ax3[n].xaxis.set_major_formatter(formatter)
                    
                    if (self.prop!=self.prop)==False: # This checks for a nan in string. If no limit selected, do nothing
                        ax3[n].set_ylim((0,int(self.prop)))           


                else:

                    ax3.plot(timeseries[0])
                    ax3.set_ylabel(scenario+'\n'+'Region Price $/MWh ',  color='black', rotation='vertical')
                    ax3.spines['right'].set_visible(False)
                    ax3.spines['top'].set_visible(False)   
                    locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
                    formatter = mdates.ConciseDateFormatter(locator)
                    formatter.formats[2] = '%d\n %b'
                    formatter.zero_formats[1] = '%b\n %Y'
                    formatter.zero_formats[2] = '%d\n %b'
                    formatter.zero_formats[3] = '%H:%M\n %d-%b'
                    formatter.offset_formats[3] = '%b %Y'
                    formatter.show_offset = False
                    ax3.xaxis.set_major_locator(locator)
                    ax3.xaxis.set_major_formatter(formatter)
                    
                    if (self.prop!=self.prop)==False: # This checks for a nan in string. If no limit selected, do nothing
                        plt.ylim((0,int(self.prop)))   
                
                del timeseries
            del Price 
            
               
            n=n+1
        #end scenario loop
                              
        return {'fig': fig3}
    
    