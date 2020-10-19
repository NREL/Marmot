# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 13:20:56 2019

This code creates plots of generator utilization factor 
(similiar to capacity factor but based on Available Capacity instead of Installed Capacity) 
and is called from Marmot_plot_main.py


@author: adyreson
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import marmot_plot_functions as mfunc
import logging

#===============================================================================

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

    def __init__(self, argument_dict):
        # iterate over items in argument_dict and set as properties of class
        # see key_list in Marmot_plot_main for list of properties
        for prop in argument_dict:
            self.__setattr__(prop, argument_dict[prop])
        self.logger = logging.getLogger('marmot_plot.'+__name__)

    def uf_fleet(self):
        outputs = {}
        generation_collection = {}
        gen_available_capacity_collection = {}
        check_input_data = []
        
        check_input_data.extend([mfunc.get_data(generation_collection,"generator_Generation", self.Marmot_Solutions_folder, self.Multi_Scenario)])
        check_input_data.extend([mfunc.get_data(gen_available_capacity_collection,"generator_Available_Capacity", self.Marmot_Solutions_folder, self.Multi_Scenario)])
        
        # Checks if all data required by plot is available, if 1 in list required data is missing
        if 1 in check_input_data:
            outputs = None
            return outputs
        
        for zone_input in self.Zones:
            CF_all_scenarios = pd.DataFrame()
            self.logger.info(self.AGG_BY + " = " + zone_input)

            fig3, ax3 = plt.subplots(len(self.Multi_Scenario),figsize=(4,4*len(self.Multi_Scenario)),sharey=True) # Set up subplots for all scenarios

            n=0 #Counter for scenario subplots

            cf_chunk = []
            for scenario in self.Multi_Scenario:

                self.logger.info("Scenario = " + str(scenario))
                Gen = generation_collection.get(scenario)
                try:
                    Gen = Gen.xs(zone_input,level = self.AGG_BY)
                except KeyError:
                    self.logger.warning("No generation in "+zone_input+".")
                    break

                Gen = df_process_gen_ind_inputs(Gen,self)

                Ava = gen_available_capacity_collection.get(scenario)
                Ava = Ava.xs(zone_input,level = self.AGG_BY)
                Ava = df_process_gen_ind_inputs(Ava,self)

                #Gen = Gen/interval_count
                Total_Gen = Gen.groupby(["tech"],as_index=True).sum() #axis=0)
    #            Total_Gen.rename(scenario, inplace = True)

                Total_Ava= Ava.groupby(["tech"],as_index=True).sum() #axis=0)
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
                CF.rename(scenario, inplace = True)
                cf_chunk.append(CF)

                del CF, Gen, Ava, Total_Gen, Total_Ava
                #end scenario loop
            
            CF_all_scenarios = pd.concat(cf_chunk, axis=1, sort=False)
            CF_all_scenarios = CF_all_scenarios.dropna(axis = 0)
            CF_all_scenarios.index = CF_all_scenarios.index.str.wrap(10, break_long_words = False)

            # If CF_all_scenarios df is empty returns a empty dataframe and does not return plot
            if CF_all_scenarios.empty:
                df = pd.DataFrame()
                outputs[zone_input] = df
                continue

            outputs[zone_input] = {'fig': fig3, 'data_table': CF_all_scenarios}
        return outputs

    def uf_gen(self):
        outputs = {}
        generation_collection = {}
        gen_available_capacity_collection = {}
        check_input_data = []
        
        check_input_data.extend([mfunc.get_data(generation_collection,"generator_Generation", self.Marmot_Solutions_folder, self.Multi_Scenario)])
        check_input_data.extend([mfunc.get_data(gen_available_capacity_collection,"generator_Available_Capacity", self.Marmot_Solutions_folder, self.Multi_Scenario)])
        
        # Checks if all data required by plot is available, if 1 in list required data is missing
        if 1 in check_input_data:
            outputs = None
            return outputs
        
        for zone_input in self.Zones:
            self.logger.info(self.AGG_BY + " = " + zone_input)

            fig2, ax2 = plt.subplots(len(self.Multi_Scenario),len(self.thermal_gen_cat),figsize=(len(self.thermal_gen_cat)*4,len(self.Multi_Scenario)*4),sharey=True)# Set up subplots for all scenarios & techs
            CF_all_scenarios=pd.DataFrame()
            n=0 #Counter for scenario subplots
            
            th_gen_chunk = []
            for scenario in self.Multi_Scenario:

                self.logger.info("Scenario = " + str(scenario))
                Gen = generation_collection.get(scenario)
                try:
                    Gen = Gen.xs(zone_input,level = self.AGG_BY)
                except KeyError:
                    self.logger.warning("No generation in "+ zone_input+".")
                    break
                Gen=df_process_gen_ind_inputs(Gen,self)

                Ava = gen_available_capacity_collection.get(scenario)
                Ava = Ava.xs(zone_input,level = self.AGG_BY)
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
                th_gen_chunk.append(thermal_generator_cf)

                del Gen, thermal_generator_cf
            #End scenario loop
            CF_all_scenarios=pd.concat(th_gen_chunk, axis=1, sort=False)
            
            fig2.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            plt.ylabel('Generators',  color='black', rotation='vertical', labelpad=60)

            # If GW_all_scenarios df is empty returns a empty dataframe and does not return plot
            if CF_all_scenarios.empty:
                df = pd.DataFrame()
                outputs[zone_input] = df
                continue

            outputs[zone_input] = {'fig': fig2, 'data_table': CF_all_scenarios}
        return outputs

    def uf_fleet_by_type(self):
        outputs = {}
        generation_collection = {}
        gen_available_capacity_collection = {}
        check_input_data = []
        
        check_input_data.extend([mfunc.get_data(generation_collection,"generator_Generation", self.Marmot_Solutions_folder, self.Multi_Scenario)])
        check_input_data.extend([mfunc.get_data(gen_available_capacity_collection,"generator_Available_Capacity", self.Marmot_Solutions_folder, self.Multi_Scenario)])
        
        # Checks if all data required by plot is available, if 1 in list required data is missing
        if 1 in check_input_data:
            outputs = None
            return outputs
        
        for zone_input in self.Zones:
            CF_all_scenarios = pd.DataFrame()
            self.logger.info(self.AGG_BY + " = " + zone_input)

            fig3, ax3 = plt.subplots(len(self.thermal_gen_cat),figsize=(4,4*len(self.thermal_gen_cat)),sharey=True) # Set up subplots for all scenarios

            cf_chunk = []
            for scenario in self.Multi_Scenario:

                self.logger.info("Scenario = " + str(scenario))
                Gen = generation_collection.get(scenario)
                try:
                    Gen = Gen.xs(zone_input,level = self.AGG_BY)
                except KeyError:
                    self.logger.warning("No generation in "+zone_input+".")
                    break
                Gen = df_process_gen_ind_inputs(Gen,self)

                Ava = gen_available_capacity_collection.get(scenario)
                Ava = Ava.xs(zone_input,level = self.AGG_BY)
                Ava = df_process_gen_ind_inputs(Ava,self)


                #Gen = Gen/interval_count
                Total_Gen = Gen.groupby(["tech"],as_index=True).sum() #axis=0)
    #            Total_Gen.rename(scenario, inplace = True)

                Total_Ava= Ava.groupby(["tech"],as_index=True).sum() #axis=0)
    #            Total_Ava.rename(scenario,inplace=True)

                Gen=pd.merge(Gen,Ava,on=['tech','timestamp'])
                Gen['Type CF']=Gen['0_x']/Gen['0_y'] #Calculation of fleet wide capacity factor by hour and type
                n=0 #Counter for type subplots

                for i in self.thermal_gen_cat: #Gen.reset_index()['tech'].unique():
                    try:
                        duration_curve = Gen.xs(i,level="tech").sort_values(by='Type CF',ascending=False).reset_index()
                    except KeyError:
                        self.logger.info("{} not in {}, skipping technology".format(i, zone_input))
                        continue

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
                CF.rename(scenario, inplace = True)
                cf_chunk.append(CF)
                

                del CF, Gen, Ava, Total_Gen, Total_Ava
                #end scenario loop
            
            CF_all_scenarios = pd.concat(cf_chunk, axis=1, sort=False)
            CF_all_scenarios = CF_all_scenarios.dropna(axis = 0)
            CF_all_scenarios.index = CF_all_scenarios.index.str.wrap(10, break_long_words = False)

            # If GW_all_scenarios df is empty returns a empty dataframe and does not return plot
            if CF_all_scenarios.empty:
                df = pd.DataFrame()
                outputs[zone_input] = df
                continue

            outputs[zone_input] = {'fig': fig3, 'data_table': CF_all_scenarios}
        return outputs

    def GW_fleet(self):
        outputs = {}
        generation_collection = {}
        check_input_data = []
        
        check_input_data.extend([mfunc.get_data(generation_collection,"generator_Generation", self.Marmot_Solutions_folder, self.Multi_Scenario)])
        
        # Checks if all data required by plot is available, if 1 in list required data is missing
        if 1 in check_input_data:
            outputs = None
            return outputs
        
        for zone_input in self.Zones:
            GW_all_scenarios = pd.DataFrame()
            self.logger.info(self.AGG_BY + " = " + zone_input)

            fig3, ax3 = plt.subplots(len(self.Multi_Scenario),figsize=(4,4*len(self.Multi_Scenario)),sharey=True) # Set up subplots for all scenarios

            n=0 #Counter for scenario subplots
            
            total_gen_chunks = []
            for scenario in self.Multi_Scenario:

                self.logger.info("Scenario = " + str(scenario))
                Gen = generation_collection.get(scenario)
                try:
                    Gen = Gen.xs(zone_input,level = self.AGG_BY)
                except KeyError:
                    self.logger.warning("No generation in "+zone_input+".")
                    break

                Gen = df_process_gen_ind_inputs(Gen,self)


                Total_Gen = Gen.groupby(["tech"],as_index=True).sum() #axis=0)
                Total_Gen.rename(scenario, inplace = True)
                total_gen_chunks.append(Total_Gen)

                for i in sorted(Gen.reset_index()['tech'].unique()):
                    duration_curve = Gen.xs(i,level="tech").sort_values(ascending=False).reset_index()

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

                del Gen,Total_Gen
                #end scenario loop
            
            GW_all_scenarios = pd.concat(total_gen_chunks, axis=1, sort=False)
            GW_all_scenarios = GW_all_scenarios.dropna(axis = 0)
            GW_all_scenarios.index = GW_all_scenarios.index.str.wrap(10, break_long_words = False)

            # If GW_all_scenarios df is empty returns a empty dataframe and does not return plot
            if GW_all_scenarios.empty:
                df = pd.DataFrame()
                outputs[zone_input] = df
                continue

            outputs[zone_input] = {'fig': fig3, 'data_table': GW_all_scenarios}
        return outputs
