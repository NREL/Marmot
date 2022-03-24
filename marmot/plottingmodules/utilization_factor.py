# -*- coding: utf-8 -*-
"""Note! Module is currently not production ready.

Note! Module is currently not production ready and 
methods are not available to use.

This code creates plots of generator utilization factor 
(similar to capacity factor but based on Available Capacity instead of Installed Capacity) 
and is called from Marmot_plot_main.py  

@author: adyreson
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import marmot.utils.mconfig as mconfig

from marmot.plottingmodules.plotutils.plot_data_helper import MPlotDataHelper
from marmot.plottingmodules.plotutils.plot_exceptions import (MissingInputData,
            UnderDevelopment, MissingZoneData)

plot_data_settings = mconfig.parser("plot_data")

def df_process_gen_ind_inputs(df, self):
    df = df.reset_index(['timestamp','tech','gen_name'])
    df['tech'].replace(self.gen_names_dict, inplace=True)
    df = df[df['tech'].isin(self.thermal_gen_cat)]  #Optional, select which technologies to show.
    df.tech = df.tech.astype("category")
    df.tech = df.tech.cat.set_categories(self.ordered_gen)
    df = df.sort_values(["tech"])
    df.set_index(['timestamp','tech','gen_name'],inplace=True)
    df=df[0]

    return df

class UtilizationFactor(MPlotDataHelper):
    """Device utilization plots.

    The utilization.py module contains methods that are
    related to the utilization of generators and other devices. 
    
    UtilizationFactor inherits from the MPlotDataHelper class to assist 
    in creating figures.
    """

    def __init__(self, **kwargs):
        # Instantiation of MPlotHelperFunctions
        super().__init__(**kwargs)

        self.logger = logging.getLogger('plotter.'+__name__)
        

    def uf_fleet(self, **_):
        """Plot under development

        Returns:
            UnderDevelopment(): Exception class, plot is not functional. 
        """
        return UnderDevelopment() #TODO: fix bugs/improve performance, get back to working stage 
        outputs = {}
        
        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True,"generator_Generation",self.Scenarios),
                      (True,"generator_Available_Capacity",self.Scenarios)]
        
        # Runs get_formatted_data within MPlotDataHelper to populate MPlotDataHelper dictionary  
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)

        if 1 in check_input_data:
            return MissingInputData()
        
        for zone_input in self.Zones:
            CF_all_scenarios = pd.DataFrame()
            self.logger.info(self.AGG_BY + " = " + zone_input)

            fig3, ax3 = plt.subplots(len(self.Scenarios),figsize=(4,4*len(self.Scenarios)),sharey=True) # Set up subplots for all scenarios

            n=0 #Counter for scenario subplots

            cf_chunk = []
            for scenario in self.Scenarios:

                self.logger.info("Scenario = " + str(scenario))
                Gen = self["generator_Generation"].get(scenario)
                try:
                    Gen = Gen.xs(zone_input,level = self.AGG_BY)
                except KeyError:
                    self.logger.warning("No generation in %s",zone_input)
                    continue

                Gen = df_process_gen_ind_inputs(Gen,self)

                Ava = self["generator_Available_Capacity"].get(scenario)
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

                    if len(self.Scenarios)>1:
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

                n+=1

                #Calculate CF
                CF = Total_Gen/Total_Ava
                CF.rename(scenario, inplace = True)
                cf_chunk.append(CF)

                del CF, Gen, Ava, Total_Gen, Total_Ava
                #end scenario loop
            
            if not cf_chunk:
                self.logger.warning("No generation in %s",zone_input)
                outputs[zone_input] = MissingZoneData()
                continue
            
            CF_all_scenarios = pd.concat(cf_chunk, axis=1, sort=False)
            CF_all_scenarios = CF_all_scenarios.dropna(axis = 0)
            CF_all_scenarios.index = CF_all_scenarios.index.str.wrap(10, break_long_words = False)

            # If CF_all_scenarios df is empty returns a empty dataframe and does not return plot
            if CF_all_scenarios.empty:
                out = MissingZoneData()
                outputs[zone_input] = out
                continue

            if plot_data_settings["plot_title_as_region"]:
            	ax3.set_title(zone_input)          
            outputs[zone_input] = {'fig': fig3, 'data_table': CF_all_scenarios}
        return outputs

    def uf_gen(self, **_):
        """Plot under development

        Returns:
            UnderDevelopment(): Exception class, plot is not functional. 
        """
        return UnderDevelopment() #TODO: fix bugs/improve performance, get back to working stage 
        outputs = {}
        
        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True,"generator_Generation",self.Scenarios),
                      (True,"generator_Available_Capacity",self.Scenarios)]
        
        # Runs get_formatted_data within MPlotDataHelper to populate MPlotDataHelper dictionary  
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)

        if 1 in check_input_data:
            return MissingInputData()
        
        for zone_input in self.Zones:
            self.logger.info(self.AGG_BY + " = " + zone_input)

            fig2, ax2 = plt.subplots(len(self.Scenarios),len(self.thermal_gen_cat),figsize=(len(self.thermal_gen_cat)*4,len(self.Scenarios)*4),sharey=True)# Set up subplots for all scenarios & techs
            CF_all_scenarios=pd.DataFrame()
            n=0 #Counter for scenario subplots
            
            th_gen_chunk = []
            for scenario in self.Scenarios:

                self.logger.info("Scenario = " + str(scenario))
                Gen = self["generator_Generation"].get(scenario)
                try:
                    Gen = Gen.xs(zone_input,level = self.AGG_BY)
                except KeyError:
                    self.logger.warning("No generation in "+ zone_input+".")
                    continue
                Gen=df_process_gen_ind_inputs(Gen,self)

                Ava = self["generator_Available_Capacity"].get(scenario)
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

                    if len(self.Scenarios)>1:
                        ax2[n][m].hist(cfs.replace([np.inf,np.nan]),bins=20,range=(0,1),color=self.PLEXOS_color_dict.get(i, '#333333'),label=scenario+"_"+i)
                        ax2[len(self.Scenarios)-1][m].set_xlabel('Annual CF',  color='black', rotation='horizontal')
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
                
            if not th_gen_chunk:
                self.logger.warning("No generation in %s",zone_input)
                outputs[zone_input] = MissingZoneData()
                continue
            
            CF_all_scenarios=pd.concat(th_gen_chunk, axis=1, sort=False)
            if plot_data_settings["plot_title_as_region"]:
                ax2.set_title(zone_input)
            fig2.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            plt.ylabel('Generators',  color='black', rotation='vertical', labelpad=60)

            # If GW_all_scenarios df is empty returns a empty dataframe and does not return plot
            if CF_all_scenarios.empty:
                out = MissingZoneData()
                outputs[zone_input] = out
                continue

            outputs[zone_input] = {'fig': fig2, 'data_table': CF_all_scenarios}
        return outputs

    def uf_fleet_by_type(self, **_):
        """Plot under development

        Returns:
            UnderDevelopment(): Exception class, plot is not functional. 
        """
        return UnderDevelopment() #TODO: fix bugs/improve performance, get back to working stage 
        outputs = {}
        
        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True,"generator_Generation",self.Scenarios),
                      (True,"generator_Available_Capacity",self.Scenarios)]
        
        # Runs get_formatted_data within MPlotDataHelper to populate MPlotDataHelper dictionary  
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)

        if 1 in check_input_data:
            return MissingInputData()
        
        for zone_input in self.Zones:
            CF_all_scenarios = pd.DataFrame()
            self.logger.info(self.AGG_BY + " = " + zone_input)

            fig3, ax3 = plt.subplots(len(self.thermal_gen_cat),figsize=(4,4*len(self.thermal_gen_cat)),sharey=True) # Set up subplots for all scenarios

            cf_chunk = []
            for scenario in self.Scenarios:

                self.logger.info("Scenario = " + str(scenario))
                Gen = self["generator_Generation"].get(scenario)
                try:
                    Gen = Gen.xs(zone_input,level = self.AGG_BY)
                except KeyError:
                    self.logger.warning("No generation in "+zone_input+".")
                    continue
                Gen = df_process_gen_ind_inputs(Gen,self)

                Ava = self["generator_Available_Capacity"].get(scenario)
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
            
            if not cf_chunk:
                self.logger.warning("No generation in %s",zone_input)
                outputs[zone_input] = MissingZoneData()
                continue
            if plot_data_settings["plot_title_as_region"]:
                ax3.set_title(zone_input)
            
            CF_all_scenarios = pd.concat(cf_chunk, axis=1, sort=False)
            CF_all_scenarios = CF_all_scenarios.dropna(axis = 0)
            CF_all_scenarios.index = CF_all_scenarios.index.str.wrap(10, break_long_words = False)

            # If GW_all_scenarios df is empty returns a empty dataframe and does not return plot
            if CF_all_scenarios.empty:
                out = MissingZoneData()
                outputs[zone_input] = out
                continue

            outputs[zone_input] = {'fig': fig3, 'data_table': CF_all_scenarios}
        return outputs

    def GW_fleet(self, **_):
        """Plot under development

        Returns:
            UnderDevelopment(): Exception class, plot is not functional. 
        """
        return UnderDevelopment() #TODO: fix bugs/improve performance, get back to working stage 
        outputs = {}
        
        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True,"generator_Generation",self.Scenarios)]
        
        # Runs get_formatted_data within MPlotDataHelper to populate MPlotDataHelper dictionary  
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)

        if 1 in check_input_data:
            return MissingInputData()
        
        for zone_input in self.Zones:
            GW_all_scenarios = pd.DataFrame()
            self.logger.info(self.AGG_BY + " = " + zone_input)

            fig3, ax3 = plt.subplots(len(self.Scenarios),figsize=(4,4*len(self.Scenarios)),sharey=True) # Set up subplots for all scenarios

            n=0 #Counter for scenario subplots
            
            total_gen_chunks = []
            for scenario in self.Scenarios:

                self.logger.info("Scenario = " + str(scenario))
                Gen = self["generator_Generation"].get(scenario)
                try:
                    Gen = Gen.xs(zone_input,level = self.AGG_BY)
                except KeyError:
                    continue

                Gen = df_process_gen_ind_inputs(Gen,self)


                Total_Gen = Gen.groupby(["tech"],as_index=True).sum() #axis=0)
                Total_Gen.rename(scenario, inplace = True)
                total_gen_chunks.append(Total_Gen)

                for i in sorted(Gen.reset_index()['tech'].unique()):
                    duration_curve = Gen.xs(i,level="tech").sort_values(ascending=False).reset_index()

                    if len(self.Scenarios)>1:
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
            
            if not total_gen_chunks:
                self.logger.warning("No generation in %s",zone_input)
                outputs[zone_input] = MissingZoneData()
                continue
            if plot_data_settings["plot_title_as_region"]:
                ax3.set_title(zone_input)

            GW_all_scenarios = pd.concat(total_gen_chunks, axis=1, sort=False)
            GW_all_scenarios = GW_all_scenarios.dropna(axis = 0)
            GW_all_scenarios.index = GW_all_scenarios.index.str.wrap(10, break_long_words = False)

            # If GW_all_scenarios df is empty returns a empty dataframe and does not return plot

            outputs[zone_input] = {'fig': fig3, 'data_table': GW_all_scenarios}
        return outputs
