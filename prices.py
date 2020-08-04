# -*- coding: utf-8 -*-
"""

price analysis

@author: adyreson
"""

import os
import pandas as pd
#import datetime as dt
import matplotlib.pyplot as plt
#import matplotlib as mpl
import matplotlib.dates as mdates
#import numpy as np 



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
        self.Zones =argument_list[7]
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
            Price_Collection[scenario] = pd.read_hdf(os.path.join(self.PLEXOS_Scenarios, scenario,"Processed_HDF5_folder", scenario+ "_formatted.h5"),self.AGG_BY + "_Price")

        outputs = {}
        for zone_input in self.Zones:              
            print(self.AGG_BY + " = " + zone_input)
            
            fig3, ax3 = plt.subplots(len(self.Multi_Scenario),figsize=(4,4*len(self.Multi_Scenario)),sharey=True) # Set up subplots for all scenarios
         
            n=0 #Counter for scenario subplots
            
            Data_Out=pd.DataFrame()
            
            for scenario in self.Multi_Scenario:
                
                print("Scenario = " + str(scenario))
                
                Price = Price_Collection.get(scenario)
                Price = Price.xs(zone_input,level=self.AGG_BY,drop_level=False) #Filter to the AGGBY level and keep all levels
                
                
                for region in Price.index.get_level_values(level=self.AGG_BY).unique() :
                    duration_curve = Price.xs(region,level=self.AGG_BY).sort_values(by=0,ascending=False).reset_index()
                            
                    if len(self.Multi_Scenario)>1:                  #Multi scenario
                        ax3[n].plot(duration_curve[0],label=region)
                        ax3[n].set_ylabel(scenario,  color='black', rotation='vertical')
                        ax3[n].set_xlabel('Intervals',  color='black', rotation='horizontal')
                        ax3[n].spines['right'].set_visible(False)
                        ax3[n].spines['top'].set_visible(False)                         
                       
                        if (self.prop!=self.prop)==False: # This checks for a nan in string. If no limit selected, do nothing
                            ax3[n].set_ylim(top=int(self.prop))           
                    else: #Single scenario
                        ax3.plot(duration_curve[0],label=region)
                     
                        ax3.set_ylabel(scenario,  color='black', rotation='vertical')
                        ax3.set_xlabel('Intervals',  color='black', rotation='horizontal')
                        ax3.spines['right'].set_visible(False)
                        ax3.spines['top'].set_visible(False)   
                       
                        if (self.prop!=self.prop)==False: # This checks for a nan in string. If no limit selected, do nothing
                            plt.ylim(top=int(self.prop))   
                                          
                    del duration_curve
                   
                if len(Price.index.get_level_values(level=self.AGG_BY).unique()) <10:# Add legend if legible
                    if len(self.Multi_Scenario)>1:
                        ax3[n].legend(loc='upper right')
                    else:
                        ax3.legend(loc='upper right')
                
                Price=Price.reset_index(['timestamp',self.AGG_BY]).set_index(['timestamp'])
                Price.rename(columns={0:scenario},inplace=True)
                Data_Out=pd.concat([Data_Out,Price],axis=1)
    
                del Price 
                   
                n=n+1
            #end scenario loop
            fig3.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            plt.ylabel(self.AGG_BY + ' Price $/MWh ',  color='black', rotation='vertical', labelpad=60)                      
            outputs[zone_input] = {'fig': fig3, 'data_table':Data_Out}
        return outputs
    
    def price_region_chron(self):          #Timeseries of individual region prices 
        Price_Collection = {}        # Create Dictionary to hold Datframes for each scenario 
        
        for scenario in self.Multi_Scenario:
            Price_Collection[scenario] = pd.read_hdf(os.path.join(self.PLEXOS_Scenarios, scenario,"Processed_HDF5_folder", scenario+ "_formatted.h5"), self.AGG_BY + "_Price")
        
        outputs = {}
        for zone_input in self.Zones:              
            print(self.AGG_BY + " = " + zone_input)
            
            fig3, ax3 = plt.subplots(len(self.Multi_Scenario),figsize=(4,4*len(self.Multi_Scenario)),sharey=True) # Set up subplots for all scenarios
         
            n=0 #Counter for scenario subplots
            
            Data_Out=pd.DataFrame()
            
            for scenario in self.Multi_Scenario:
                
                print("Scenario = " + str(scenario))
                
                Price = Price_Collection.get(scenario)
                Price = Price.xs(zone_input,level=self.AGG_BY,drop_level=False) #Filter to the AGGBY level and keep all levels
    
                for region in Price.index.get_level_values(level=self.AGG_BY).unique() :
                    timeseries = Price.xs(region,level=self.AGG_BY).reset_index().set_index('timestamp')
                            
                    if len(self.Multi_Scenario)>1:
                        ax3[n].plot(timeseries[0],label=region)
                        ax3[n].set_ylabel(scenario,  color='black', rotation='vertical')
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
                            ax3[n].set_ylim(top=int(self.prop))           
    
    
                    else:
    
                        ax3.plot(timeseries[0],label=region)
                        ax3.set_ylabel(scenario,  color='black', rotation='vertical')
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
                            plt.ylim(top=int(self.prop))   
                    
                    del timeseries
                    
                if len(Price.index.get_level_values(level=self.AGG_BY).unique()) <10:# Add legend if legible
                        if len(self.Multi_Scenario)>1:
                            ax3[n].legend()
                        else:
                            ax3.legend()
                                
                Price=Price.reset_index(['timestamp',self.AGG_BY]).set_index(['timestamp'])
                Price.rename(columns={0:scenario},inplace=True)
                Data_Out=pd.concat([Data_Out,Price],axis=1)
                
                del Price 
                
                               
                n=n+1
            #end scenario loop
            fig3.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            plt.ylabel(self.AGG_BY + ' Price $/MWh ',  color='black', rotation='vertical', labelpad=60)          
                          
            outputs[zone_input] = {'fig': fig3, 'data_table':Data_Out}
        return outputs
    