# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 13:20:56 2019

This module contain methods that are
related to the capacity factor of generators and average output plots 
@author: Daniel Levie 
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt

import marmot.config.mconfig as mconfig
from marmot.plottingmodules.plotutils.plot_data_helper import PlotDataHelper
from marmot.plottingmodules.plotutils.plot_exceptions import (MissingInputData, MissingZoneData)


class MPlot(PlotDataHelper):
    """Marmot MPlot class, common across all plotting modules.

    All the plotting modules use this same class name.
    This class contains plotting methods that are grouped based on the
    current module name.
    
    The capacity_factor.py module contain methods that are
    related to the capacity factor of generators. 

    MPlot inherits from the PlotDataHelper class to assist in creating figures.
    """

    def __init__(self, argument_dict: dict):
        """MPlot init method

        Args:
            argument_dict (dict): Dictionary containing all
                arguments passed from MarmotPlot.
        """
        # iterate over items in argument_dict and set as properties of class
        # see key_list in Marmot_plot_main for list of properties
        for prop in argument_dict:
            self.__setattr__(prop, argument_dict[prop])

        # Instantiation of MPlotHelperFunctions
        super().__init__(self.Marmot_Solutions_folder, self.AGG_BY, self.ordered_gen, 
                    self.PLEXOS_color_dict, self.Scenarios, self.ylabels, 
                    self.xlabels, self.gen_names_dict, Region_Mapping=self.Region_Mapping) 
        self.logger = logging.getLogger('marmot_plot.'+__name__)
        self.x = mconfig.parser("figure_size","xdimension")
        self.y = mconfig.parser("figure_size","ydimension")
        
        
    def avg_output_when_committed(self,
                                  start_date_range: str = None, 
                                  end_date_range: str = None, **_):
        """Creates barplots of the percentage average generation output when committed by technology type. 

        Each scenario is plotted by a different colored grouped bar. 

        Args:
            start_date_range (str, optional): Defines a start date at which to represent data from. 
                Defaults to None.
            end_date_range (str, optional): Defines a end date at which to represent data from.
                Defaults to None.

        Returns:
            dict: dictionary containing the created plot and its data table.
        """
        outputs = {}
        
        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True,"generator_Generation",self.Scenarios),
                      (True,"generator_Installed_Capacity",self.Scenarios)]
        
        # Runs get_formatted_data within PlotDataHelper to populate PlotDataHelper dictionary  
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)

        if 1 in check_input_data:
            return MissingInputData()
        
        for zone_input in self.Zones:
            CF_all_scenarios = pd.DataFrame()
            self.logger.info(f"{self.AGG_BY} = {zone_input}")

            for scenario in self.Scenarios:
                self.logger.info(f"Scenario = {str(scenario)}")
                Gen = self["generator_Generation"].get(scenario)
                try: #Check for regions missing all generation.
                    Gen = Gen.xs(zone_input,level = self.AGG_BY)
                except KeyError:
                        self.logger.warning(f'No data in {zone_input}')
                        continue
                Gen = Gen.reset_index()
                Gen.tech = Gen.tech.astype("category")
                Gen.tech.cat.set_categories(self.ordered_gen, inplace=True)
                Gen = Gen.rename(columns = {0:"Output (MWh)"})
                # techs = list(Gen['tech'].unique())
                Gen = Gen[Gen['tech'].isin(self.thermal_gen_cat)]
                Cap = self["generator_Installed_Capacity"].get(scenario)
                Cap = Cap.xs(zone_input,level = self.AGG_BY)
                Cap = Cap.reset_index()
                Cap = Cap.drop(columns = ['timestamp','tech'])
                Cap = Cap.rename(columns = {0:"Installed Capacity (MW)"})
                Gen = pd.merge(Gen,Cap, on = 'gen_name')
                Gen.set_index('timestamp',inplace=True)
                
                if pd.notna(start_date_range):
                    self.logger.info(f"Plotting specific date range: \
                    {str(start_date_range)} to {str(end_date_range)}")
                    # sort_index added see https://github.com/pandas-dev/pandas/issues/35509
                    Gen = Gen.sort_index()[start_date_range : end_date_range]
                    if Gen.empty is True:
                        self.logger.warning('No data in selected Date Range')
                        continue

                #Calculate CF individually for each plant, since we need to take out all zero rows.
                tech_names = Gen['tech'].unique()
                CF = pd.DataFrame(columns = tech_names,index = [scenario])
                for tech_name in tech_names:
                    stt = Gen.loc[Gen['tech'] == tech_name]
                    if not all(stt['Output (MWh)'] == 0):

                        gen_names = stt['gen_name'].unique()
                        cfs = []
                        caps = []
                        for gen in gen_names:
                            sgt = stt.loc[stt['gen_name'] == gen]
                            if not all(sgt['Output (MWh)'] == 0):
                                
                                # Calculates interval step to correct for MWh of generation
                                time_delta = sgt.index[1] - sgt.index[0]
                                duration = sgt.index[len(sgt)-1] - sgt.index[0]
                                duration = duration + time_delta #Account for last timestep.
                                # Finds intervals in 60 minute period
                                interval_count = 60/(time_delta/np.timedelta64(1, 'm'))
                                duration_hours = duration/np.timedelta64(1,'h')     #Get length of time series in hours for CF calculation.
                                                     
                                sgt = sgt[sgt['Output (MWh)'] !=0] #Remove time intervals when output is zero.
                                total_gen = sgt['Output (MWh)'].sum()/interval_count
                                cap = sgt['Installed Capacity (MW)'].mean()

                                #Calculate CF
                                cf = total_gen/(cap * duration_hours)
                                cfs.append(cf)
                                caps.append(cap)

                        #Find average "CF" (average output when committed) for this technology, weighted by capacity.
                        cf = np.average(cfs,weights = caps)
                        CF[tech_name] = cf

                CF_all_scenarios = CF_all_scenarios.append(CF)
            
            CF_all_scenarios.index = CF_all_scenarios.index.str.replace('_',' ')
            
            if CF_all_scenarios.empty == True:
                outputs[zone_input] = MissingZoneData()
                continue
            
            Data_Table_Out = CF_all_scenarios.T
            fig2, ax = plt.subplots(figsize=(self.x,self.y))
            CF_all_scenarios.T.plot.bar(stacked = False,
                                  color = self.color_list,edgecolor='black', linewidth='0.1',ax=ax)
            
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_ylabel('Average Output When Committed',  color='black', rotation='vertical')
            #adds % to y axis data
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            
            # Set x-tick labels 
            tick_labels = CF_all_scenarios.columns
            PlotDataHelper.set_barplot_xticklabels(tick_labels, ax=ax)

            ax.tick_params(axis='y', which='major', length=5, width=1)
            ax.tick_params(axis='x', which='major', length=5, width=1)
            if mconfig.parser("plot_title_as_region"):
                ax.set_title(zone_input)

            ax.legend(loc='lower left',bbox_to_anchor=(1,0),
                          facecolor='inherit', frameon=True)
            
            outputs[zone_input] = {'fig': fig2, 'data_table': Data_Table_Out}
        return outputs


    def cf(self, start_date_range: str = None, 
           end_date_range: str = None, **_):
        """Creates barplots of generator capacity factors by technology type. 

        Each scenario is plotted by a different colored grouped bar. 

        Args:
            start_date_range (str, optional): Defines a start date at which to represent data from. 
                Defaults to None.
            end_date_range (str, optional): Defines a end date at which to represent data from.
                Defaults to None.

        Returns:
            dict: dictionary containing the created plot and its data table.
        """
        
        outputs = {}
        
        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True,"generator_Generation",self.Scenarios),
                      (True,"generator_Installed_Capacity",self.Scenarios)]
        
        # Runs get_formatted_data within PlotDataHelper to populate PlotDataHelper dictionary  
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)

        if 1 in check_input_data:
            return MissingInputData()
        
        for zone_input in self.Zones:
            CF_all_scenarios = pd.DataFrame()
            self.logger.info(f"{self.AGG_BY} = {zone_input}")

            for scenario in self.Scenarios:

                self.logger.info(f"Scenario = {str(scenario)}")
                Gen = self["generator_Generation"].get(scenario)
                try: #Check for regions missing all generation.
                    Gen = Gen.xs(zone_input,level = self.AGG_BY)
                except KeyError:
                        self.logger.warning(f'No data in {zone_input}')
                        continue
                Gen = self.df_process_gen_inputs(Gen)
                
                if pd.notna(start_date_range):
                    self.logger.info(f"Plotting specific date range: \
                    {str(start_date_range)} to {str(end_date_range)}")
                    Gen = Gen[start_date_range : end_date_range]
                    if Gen.empty is True:
                        self.logger.warning('No data in selected Date Range')
                        continue
                        
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
                
                Cap = self["generator_Installed_Capacity"].get(scenario)
                Cap = Cap.xs(zone_input,level = self.AGG_BY)
                Cap = self.df_process_gen_inputs(Cap)
                Cap = Cap.T.sum(axis = 1)  #Rotate and force capacity to a series.
                Cap.rename(scenario, inplace = True)

                #Calculate CF
                CF = Total_Gen/(Cap * duration_hours)
                CF.rename(scenario, inplace = True)
                CF_all_scenarios = pd.concat([CF_all_scenarios, CF], axis=1, sort=False)
                CF_all_scenarios = CF_all_scenarios.fillna(0, axis = 0)

            CF_all_scenarios.columns = CF_all_scenarios.columns.str.replace('_',' ')

            if CF_all_scenarios.empty == True:
                outputs[zone_input] = MissingZoneData()
                continue
            
            Data_Table_Out = CF_all_scenarios.T

            fig1,ax = plt.subplots(figsize=(self.x*1.5,self.y*1.5))
            #TODO: rewrite with mfunc functions.

            CF_all_scenarios.plot.bar(stacked = False, 
                                  color = self.color_list,edgecolor='black', linewidth='0.1',ax = ax)
            
            # This code would be used to create the bar plot using plotlib.create_bar_plot()
            # fig1, axs = plotlib.setup_plot()
            # #flatten object
            # ax=axs[0]
            # plotlib.create_bar_plot(CF_all_scenarios, ax, self.color_list, angle)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_ylabel('Capacity Factor',  color='black', rotation='vertical')
            #adds % to y axis data
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            
            # Set x-tick labels 
            tick_labels = CF_all_scenarios.index
            PlotDataHelper.set_barplot_xticklabels(tick_labels, ax=ax)

            ax.tick_params(axis='y', which='major', length=5, width=1)
            ax.tick_params(axis='x', which='major', length=5, width=1)
            ax.legend(loc='lower left',bbox_to_anchor=(1,0),
                          facecolor='inherit', frameon=True)

            handles, labels = ax.get_legend_handles_labels()

            #Legend 1
            ax.legend(handles, labels, loc='lower left',bbox_to_anchor=(1,0),
                          facecolor='inherit', frameon=True)
            if mconfig.parser("plot_title_as_region"):
                ax.set_title(zone_input)
            outputs[zone_input] = {'fig': fig1, 'data_table': Data_Table_Out}

        return outputs


    def time_at_min_gen(self, start_date_range: str = None, 
                        end_date_range: str = None, **_):
        """Creates barplots of generator percentage time at min-gen by technology type. 

        Each scenario is plotted by a different colored grouped bar. 

        Args:
            start_date_range (str, optional): Defines a start date at which to represent data from. 
                Defaults to None.
            end_date_range (str, optional): Defines a end date at which to represent data from.
                Defaults to None.

        Returns:
            dict: dictionary containing the created plot and its data table.
        """
        
        outputs = {}
        
        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True,"generator_Generation",self.Scenarios),
                      (True,"generator_Installed_Capacity",self.Scenarios),
                      (True,"generator_Hours_at_Minimum",self.Scenarios)]
        
        # Runs get_formatted_data within PlotDataHelper to populate PlotDataHelper dictionary  
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)

        if 1 in check_input_data:
            return MissingInputData()
        
        for zone_input in self.Zones:
            self.logger.info(f"{self.AGG_BY} = {zone_input}")

            time_at_min = pd.DataFrame()

            for scenario in self.Scenarios:
                self.logger.info(f"Scenario = {str(scenario)}")

                Min = self["generator_Hours_at_Minimum"].get(scenario)
                try:
                    Min = Min.xs(zone_input,level = self.AGG_BY)
                except KeyError:
                    continue
                
                Min = Min.reset_index()
                Min = Min.set_index('gen_name')
                Min = Min.rename(columns = {0:"Hours at Minimum"})

                Gen = self["generator_Generation"].get(scenario)
                try: #Check for regions missing all generation.
                    Gen = Gen.xs(zone_input,level = self.AGG_BY)
                except KeyError:
                        self.logger.warning(f'No data in {zone_input}')
                        continue
                Gen = Gen.reset_index()
                Gen.tech = Gen.tech.astype("category")
                Gen.tech.cat.set_categories(self.ordered_gen, inplace=True)


                Gen = Gen.rename(columns = {0:"Output (MWh)"})
                Gen = Gen[~Gen['tech'].isin(self.vre_gen_cat)]
                Gen.index = Gen.timestamp

                Cap = self["generator_Installed_Capacity"].get(scenario)
                Cap = Cap.xs(zone_input,level = self.AGG_BY)
                Caps = Cap.groupby('gen_name').mean()
                Caps.reset_index()
                Caps = Caps.rename(columns = {0: 'Installed Capacity (MW)'})
                Min = pd.merge(Min,Caps, on = 'gen_name')

                #Find how many hours each generator was operating, for the denominator of the % time at min gen.
                #So remove all zero rows.
                Gen = Gen.loc[Gen['Output (MWh)'] != 0]
                online_gens = Gen.gen_name.unique()
                Min = Min.loc[online_gens]
                Min['hours_online'] = Gen.groupby('gen_name')['Output (MWh)'].count()
                Min['fraction_at_min'] = Min['Hours at Minimum'] / Min.hours_online

                tech_names = Min.tech.unique()
                time_at_min_individ = pd.DataFrame(columns = tech_names, index = [scenario])
                for tech_name in tech_names:
                    stt = Min.loc[Min['tech'] == tech_name]
                    wgts = stt['Installed Capacity (MW)']
                    if wgts.sum() == 0:
                        wgts = pd.Series([1] * len(stt))
                    output = np.average(stt.fraction_at_min,weights = wgts)
                    time_at_min_individ[tech_name] = output

                time_at_min = time_at_min.append(time_at_min_individ)

            if time_at_min.empty == True:
                outputs[zone_input] = MissingZoneData()
                continue
            
            Data_Table_Out = time_at_min.T
            
            fig3, ax = plt.subplots(figsize=(self.x*1.5,self.y*1.5))
            time_at_min.T.plot.bar(stacked = False, 
                                  color = self.color_list,edgecolor='black', linewidth='0.1',ax=ax)
            
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_ylabel('Percentage of time online at minimum generation',  color='black', rotation='vertical')
            #adds % to y axis data
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            
            # Set x-tick labels 
            tick_labels = time_at_min.columns
            PlotDataHelper.set_barplot_xticklabels(tick_labels, ax=ax)

            ax.tick_params(axis='y', which='major', length=5, width=1)
            ax.tick_params(axis='x', which='major', length=5, width=1)

            if mconfig.parser("plot_title_as_region"):
                ax.set_title(zone_input)

            ax.legend(loc='lower left',bbox_to_anchor=(1,0),
                          facecolor='inherit', frameon=True)
            outputs[zone_input] = {'fig': fig3, 'data_table': Data_Table_Out}
        return outputs
