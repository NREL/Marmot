# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 13:20:56 2019

This code creates plots of generator utilization factor (similiar to capacity factor but based on Available Capacity instead of Installed Capacity) and is called from Marmot_plot_main.py


@author: adyreson
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mtick
import numpy as np 
import os


#===============================================================================

def df_process_gen_inputs(df,self):
    df = df.reset_index(['timestamp','tech'])
    df['tech'].replace(self.gen_names_dict, inplace=True)
    df = df[df['tech'].isin(self.thermal_gen_cat)]  #Optional, select which technologies to show. 
    df = df.groupby(["timestamp", "tech"], as_index=False).sum()
    df.tech = df.tech.astype("category")
    df.tech.cat.set_categories(self.ordered_gen, inplace=True)
    df = df.sort_values(["tech"]) 
    df.set_index(['timestamp','tech'],inplace=True)
    return df  

def df_process_gen_ind_inputs(df,self):
    df = df.reset_index(['timestamp','tech','gen_name'])
    df['tech'].replace(self.gen_names_dict, inplace=True)
    df = df[df['tech'].isin(self.thermal_gen_cat)]  #Optional, select which technologies to show. 
    df.tech = df.tech.astype("category")
    df.tech.cat.set_categories(self.ordered_gen, inplace=True)
    df = df.sort_values(["tech"]) 
    df.set_index(['timestamp','tech','gen_name'],inplace=True)
    df=df[0]

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
        self.thermal_gen_cat = argument_list[23]
        
    def uf_fleet(self):
        # Create Dictionary to hold Datframes for each scenario 
        Gen_Collection = {} 
        Ava_Collection = {}
        
        for scenario in self.Multi_Scenario:
            Gen_Collection[scenario] = pd.read_hdf(os.path.join(self.PLEXOS_Scenarios, scenario,"Processed_HDF5_folder", scenario+ "_formatted.h5"),"generator_Generation")
            Ava_Collection[scenario] = pd.read_hdf(os.path.join(self.PLEXOS_Scenarios, scenario,"Processed_HDF5_folder", scenario+ "_formatted.h5"),"generator_Available_Capacity")

        CF_all_scenarios = pd.DataFrame()
        print("Zone = " + self.zone_input)
        
        fig3, ax3 = plt.subplots(len(self.Multi_Scenario),figsize=(4,4*len(self.Multi_Scenario)),sharey=True) # Set up subplots for all scenarios
     
        n=0 #Counter for scenario subplots
        
        for scenario in self.Multi_Scenario:
            
            print("Scenario = " + str(scenario))
            Gen = Gen_Collection.get(scenario)
            try:
                Gen = Gen.xs(self.zone_input,level = self.AGG_BY)
            except KeyError:
                print("No generation in "+self.zone_input+".")
                break
            
            Gen = df_process_gen_inputs(Gen,self)
         
            Ava = Ava_Collection.get(scenario)
            Ava = Ava.xs(self.zone_input,level = self.AGG_BY)
            Ava = df_process_gen_inputs(Ava,self)

            
            #Gen = Gen/interval_count
            Total_Gen = Gen.groupby(["tech"],as_index=True).sum(axis=0)
#            Total_Gen.rename(scenario, inplace = True)
            
            Total_Ava= Ava.groupby(["tech"],as_index=True).sum(axis=0)
#            Total_Ava.rename(scenario,inplace=True)
            
            Gen=pd.merge(Gen,Ava,on=['tech','timestamp'])
            Gen['Type CF']=Gen['0_x']/Gen['0_y'] #Calculation of fleet wide capacity factor by hour and type
                
            for i in sorted(Gen.reset_index()['tech'].unique()):
                duration_curve = Gen.xs(i,level="tech").sort_values(by='Type CF',ascending=False).reset_index()
                        
                if len(self.Multi_Scenario)>1:
                    ax3[n].plot(duration_curve['Type CF'],color=self.PLEXOS_color_dict.get(i, '#333333'),label=i)
                    ax3[n].legend()
                    ax3[n].set_ylabel('CF \n'+scenario,  color='black', rotation='vertical')
                    ax3[n].set_xlabel('Intervals',  color='black', rotation='horizontal')
                    ax3[n].spines['right'].set_visible(False)
                    ax3[n].spines['top'].set_visible(False)                         
                
                else:
                    ax3.plot(duration_curve['Type CF'],color=self.PLEXOS_color_dict.get(i, '#333333'),label=i)
                    ax3.legend()
                    ax3.set_ylabel('CF \n'+scenario,  color='black', rotation='vertical')
                    ax3.set_xlabel('Intervals',  color='black', rotation='horizontal')
                    ax3.spines['right'].set_visible(False)
                    ax3.spines['top'].set_visible(False)   
                               
                del duration_curve
                  
            n=n+1
            
            #Calculate CF
            CF = Total_Gen/Total_Ava
            CF.rename(columns={"0":scenario}, inplace = True)
            CF_all_scenarios = pd.concat([CF_all_scenarios, CF], axis=1, sort=False)
            CF_all_scenarios = CF_all_scenarios.dropna(axis = 0)
            
            del CF, Gen, Ava, Total_Gen, Total_Ava
            #end scenario loop

        CF_all_scenarios.index = CF_all_scenarios.index.str.wrap(10, break_long_words = False)
        
                           
        return {'fig': fig3, 'data_table': CF_all_scenarios}
    
    def uf_gen(self):
        # Create Dictionary to hold Datframes for each scenario 
        Gen_Collection = {} 
        Ava_Collection = {}
        
        for scenario in self.Multi_Scenario:
            Gen_Collection[scenario] = pd.read_hdf(os.path.join(self.PLEXOS_Scenarios, scenario,"Processed_HDF5_folder", scenario+ "_formatted.h5"),"generator_Generation")
            Ava_Collection[scenario] = pd.read_hdf(os.path.join(self.PLEXOS_Scenarios, scenario,"Processed_HDF5_folder", scenario+ "_formatted.h5"),"generator_Available_Capacity")

        print("Zone = " + self.zone_input)
        
        fig2, ax2 = plt.subplots(len(self.Multi_Scenario),len(self.thermal_gen_cat),figsize=(len(self.thermal_gen_cat)*4,len(self.Multi_Scenario)*4),sharey=True)# Set up subplots for all scenarios & techs
        CF_all_scenarios=pd.DataFrame()
        n=0 #Counter for scenario subplots
        
        for scenario in self.Multi_Scenario:
            
            print("Scenario = " + str(scenario))
            Gen = Gen_Collection.get(scenario)
            try:
                Gen = Gen.xs(self.zone_input,level = self.AGG_BY)
            except KeyError:
                print("No generation in "+self.zone_input+".")
                break
            Gen=df_process_gen_ind_inputs(Gen,self)
 
            Ava = Ava_Collection.get(scenario)
            Ava = Ava.xs(self.zone_input,level = self.AGG_BY)
            Ava = df_process_gen_ind_inputs(Ava,self)
            
               
            Gen=pd.merge(Gen,Ava,on=['tech','timestamp','gen_name'])
            del Ava
            #Ava.index.get_level_values(level='gen_name').unique()                                      #Count number of gens as a check
            Gen['Interval CF']= Gen['0_x']/Gen['0_y']                                                       #Hourly CF individual generators
#            Gen=Gen.reset_index().set_index(["gen_name","timestamp","tech"])
            thermal_generator_cf=Gen.groupby(["gen_name","tech"]).mean()                                #Calculate annual average of generator's hourly CF
            thermal_generator_cf=thermal_generator_cf[(thermal_generator_cf['Interval CF'].isna())==False]  #Remove na's for categories that don't match gens Add check that the same number of entries is found?
            
            m=0
            for i in sorted(thermal_generator_cf.reset_index()['tech'].unique()):
                cfs = thermal_generator_cf['Interval CF'].xs(i,level="tech")
                
                if len(self.Multi_Scenario)>1:
                    ax2[n][m].hist(cfs.replace([np.inf,np.nan]),bins=20,range=(0,1),color=self.PLEXOS_color_dict.get(i, '#333333'),label=scenario+"_"+i)
                    ax2[len(self.Multi_Scenario)-1][m].set_xlabel('Annual CF',  color='black', rotation='horizontal')
                    ax2[n][m].set_ylabel(('n='+str(len(cfs))),  color='black', rotation='vertical') 
                    ax2[n][m].legend()
             #Plot histograms of individual generator annual CF's on a subplot containing all combinations

                else:
                    ax2[m].hist(cfs.replace([np.inf,np.nan]),bins=20,range=(0,1),color=self.PLEXOS_color_dict.get(i, '#333333'),label=scenario+"_"+i)
                    ax2[m].legend()
                    ax2[m].set_xlabel('Annual CF',  color='black', rotation='horizontal')
                    ax2[m].set_ylabel(('n='+str(len(cfs))),  color='black', rotation='vertical')              #Plot histograms of individual generator annual CF's on a subplot containing all combinations

                m=m+1
                del cfs
            #End tech loop
            
            n=n+1
            thermal_generator_cf.rename(columns={"Interval CF":scenario}, inplace = True)
            thermal_generator_cf=thermal_generator_cf[scenario]
            thermal_generator_cf=pd.DataFrame(thermal_generator_cf.groupby("tech").mean())
            CF_all_scenarios=pd.concat([CF_all_scenarios,thermal_generator_cf])
            
            del Gen, thermal_generator_cf
        #End scenario loop

        fig2.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.ylabel('Generators',  color='black', rotation='vertical', labelpad=60)
            

                          
        return {'fig': fig2, 'data_table': CF_all_scenarios}

    def uf_fleet_by_type(self):
        # Create Dictionary to hold Datframes for each scenario 
        Gen_Collection = {} 
        Ava_Collection = {}
        
        for scenario in self.Multi_Scenario:
            Gen_Collection[scenario] = pd.read_hdf(os.path.join(self.PLEXOS_Scenarios, scenario,"Processed_HDF5_folder", scenario+ "_formatted.h5"),"generator_Generation")
            Ava_Collection[scenario] = pd.read_hdf(os.path.join(self.PLEXOS_Scenarios, scenario,"Processed_HDF5_folder", scenario+ "_formatted.h5"),"generator_Available_Capacity")

        CF_all_scenarios = pd.DataFrame()
        print("Zone = " + self.zone_input)
        
        fig3, ax3 = plt.subplots(len(self.thermal_gen_cat),figsize=(4,4*len(self.thermal_gen_cat)),sharey=True) # Set up subplots for all scenarios
     
        
        for scenario in self.Multi_Scenario:
            
            print("Scenario = " + str(scenario))
            Gen = Gen_Collection.get(scenario)
            Gen = Gen.xs(self.zone_input,level = self.AGG_BY)
            Gen = df_process_gen_inputs(Gen,self)
         
            Ava = Ava_Collection.get(scenario)
            Ava = Ava.xs(self.zone_input,level = self.AGG_BY)
            Ava = df_process_gen_inputs(Ava,self)

            
            #Gen = Gen/interval_count
            Total_Gen = Gen.groupby(["tech"],as_index=True).sum(axis=0)
#            Total_Gen.rename(scenario, inplace = True)
            
            Total_Ava= Ava.groupby(["tech"],as_index=True).sum(axis=0)
#            Total_Ava.rename(scenario,inplace=True)
            
            Gen=pd.merge(Gen,Ava,on=['tech','timestamp'])
            Gen['Type CF']=Gen['0_x']/Gen['0_y'] #Calculation of fleet wide capacity factor by hour and type
            
            n=0 #Counter for type subplots
    
            for i in self.thermal_gen_cat: #Gen.reset_index()['tech'].unique():
                duration_curve = Gen.xs(i,level="tech").sort_values(by='Type CF',ascending=False).reset_index()
                        
                ax3[n].plot(duration_curve['Type CF'],label=scenario)
                ax3[n].legend()
                ax3[n].set_ylabel('CF \n'+i,  color='black', rotation='vertical')
                ax3[n].set_xlabel('Intervals',  color='black', rotation='horizontal')
                ax3[n].spines['right'].set_visible(False)
                ax3[n].spines['top'].set_visible(False)                         
                                                                
                del duration_curve
                  
                n=n+1
            
            #Calculate CF
            CF = Total_Gen/Total_Ava
            CF.rename(columns={"0":scenario}, inplace = True)
            CF_all_scenarios = pd.concat([CF_all_scenarios, CF], axis=1, sort=False)
            CF_all_scenarios = CF_all_scenarios.dropna(axis = 0)
            
            del CF, Gen, Ava, Total_Gen, Total_Ava
            #end scenario loop

        CF_all_scenarios.index = CF_all_scenarios.index.str.wrap(10, break_long_words = False)
        
                           
        return {'fig': fig3, 'data_table': CF_all_scenarios}             

    def GW_fleet(self):
        # Create Dictionary to hold Datframes for each scenario 
        Gen_Collection = {} 
        
        for scenario in self.Multi_Scenario:
            Gen_Collection[scenario] = pd.read_hdf(os.path.join(self.PLEXOS_Scenarios, scenario,"Processed_HDF5_folder", scenario+ "_formatted.h5"),"generator_Generation")

        GW_all_scenarios = pd.DataFrame()
        print("Zone = " + self.zone_input)
        
        fig3, ax3 = plt.subplots(len(self.Multi_Scenario),figsize=(4,4*len(self.Multi_Scenario)),sharey=True) # Set up subplots for all scenarios
     
        n=0 #Counter for scenario subplots
        
        for scenario in self.Multi_Scenario:
            
            print("Scenario = " + str(scenario))
            Gen = Gen_Collection.get(scenario)
            try:
                Gen = Gen.xs(self.zone_input,level = self.AGG_BY)
            except KeyError:
                print("No generation in "+self.zone_input+".")
                break
            
            Gen = df_process_gen_inputs(Gen,self)
        
                       
            Total_Gen = Gen.groupby(["tech"],as_index=True).sum(axis=0)
            

            for i in sorted(Gen.reset_index()['tech'].unique()):
                duration_curve = Gen.xs(i,level="tech").sort_values(by=0,ascending=False).reset_index()
                        
                if len(self.Multi_Scenario)>1:
                    ax3[n].plot(duration_curve[0]/1000,color=self.PLEXOS_color_dict.get(i, '#333333'),label=i)
                    ax3[n].legend()
                    ax3[n].set_ylabel('GW \n'+scenario,  color='black', rotation='vertical')
                    ax3[n].set_xlabel('Intervals',  color='black', rotation='horizontal')
                    ax3[n].spines['right'].set_visible(False)
                    ax3[n].spines['top'].set_visible(False)                         
                
                else:
                    ax3.plot(duration_curve[0]/1000,color=self.PLEXOS_color_dict.get(i, '#333333'),label=i)
                    ax3.legend()
                    ax3.set_ylabel('GW \n'+scenario,  color='black', rotation='vertical')
                    ax3.set_xlabel('Intervals',  color='black', rotation='horizontal')
                    ax3.spines['right'].set_visible(False)
                    ax3.spines['top'].set_visible(False)   
                               
                del duration_curve
                  
            n=n+1
            
            #Calculate CF
            Total_Gen.rename(columns={0:scenario}, inplace = True)
            GW_all_scenarios = pd.concat([GW_all_scenarios, Total_Gen], axis=1, sort=False)
            GW_all_scenarios = GW_all_scenarios.dropna(axis = 0)
            
            del Gen,Total_Gen 
            #end scenario loop

        GW_all_scenarios.index = GW_all_scenarios.index.str.wrap(10, break_long_words = False)
        
                           
        return {'fig': fig3, 'data_table': GW_all_scenarios}