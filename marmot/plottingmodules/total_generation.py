# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 13:20:56 2019

This code creates total generation stacked bar plots and is called from Marmot_plot_main.py

@author: dlevie
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
import matplotlib.ticker as mtick
import logging
import marmot.plottingmodules.marmot_plot_functions as mfunc
import marmot.config.mconfig as mconfig
import textwrap

#===============================================================================

custom_legend_elements = Patch(facecolor='#DD0200',
                            alpha=0.5, edgecolor='#DD0200',
                         label='Unserved Energy')

custom_legend_elements_month = Patch(facecolor='#DD0200',alpha=0.7,edgecolor='#DD0200',
                                     label='Unserved_Energy')

class MPlot(object):

    MONTHS = {  1 : "January",
                2 : "February",
                3 : "March",
                4 : "April",
                5 : "May",
                6 : "June",
                7 : "July",
                8 : "August",
                9 : "September",
                10 : "October",
                11 : "November",
                12 : "December"
            }


    def __init__(self, argument_dict):
        # iterate over items in argument_dict and set as properties of class
        # see key_list in Marmot_plot_main for list of properties
        for prop in argument_dict:
            self.__setattr__(prop, argument_dict[prop])
        self.logger = logging.getLogger('marmot_plot.'+__name__)

        self.x = mconfig.parser("figure_size","xdimension")
        self.y = mconfig.parser("figure_size","ydimension")
        self.y_axes_decimalpt = mconfig.parser("axes_options","y_axes_decimalpt")
        self.curtailment_prop = mconfig.parser("plot_data","curtailment_property")

        self.mplot_data_dict = {}

    def total_gen(self, figure_name=None, prop=None, start=None, end=None,
                  timezone="", start_date_range=None, end_date_range=None):
        # Create Dictionary to hold Datframes for each scenario
        outputs = {}

        if self.AGG_BY == 'zone':
            agg = 'zone'
        else:
            agg = 'region'
        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True,"generator_Generation",self.Scenarios),
                      (False,f"generator_{self.curtailment_prop}",self.Scenarios),
                      (False,"generator_Pump_Load",self.Scenarios),
                      (True,f"{agg}_Load",self.Scenarios),
                      (False,f"{agg}_Unserved_Energy",self.Scenarios)]

        # Runs get_data to populate mplot_data_dict with all required properties, returns a 1 if required data is missing
        check_input_data = mfunc.get_data(self.mplot_data_dict, properties,self.Marmot_Solutions_folder)

        if 1 in check_input_data:
            outputs = mfunc.MissingInputData()
            return outputs

        for zone_input in self.Zones:

            # Will hold retrieved data for each scenario
            gen_chunks = []
            load_chunk = []
            pumped_load_chunk = []
            total_demand_chunk = []
            unserved_energy_chunk = []

            self.logger.info(f"Zone = {zone_input}")

            for scenario in self.Scenarios:

                self.logger.info(f"Scenario = {scenario}")
                Total_Gen_Stack = self.mplot_data_dict['generator_Generation'].get(scenario)

                #Check if zone has generation, if not skips
                try:
                    Total_Gen_Stack = Total_Gen_Stack.xs(zone_input,level=self.AGG_BY)
                except KeyError:
                    self.logger.warning(f"No installed capacity in: {zone_input}")
                    continue

                Total_Gen_Stack = mfunc.df_process_gen_inputs(Total_Gen_Stack, self.ordered_gen)

                # Calculates interval step to correct for MWh of generation
                interval_count = mfunc.get_sub_hour_interval_count(Total_Gen_Stack)

                curtailment_name = self.gen_names_dict.get('Curtailment','Curtailment')

                # Insert Curtailmnet into gen stack if it exhists in database
                if self.mplot_data_dict[f"generator_{self.curtailment_prop}"]:
                    Stacked_Curt = self.mplot_data_dict[f"generator_{self.curtailment_prop}"].get(scenario)
                    if zone_input in Stacked_Curt.index.get_level_values(self.AGG_BY).unique():
                        Stacked_Curt = Stacked_Curt.xs(zone_input,level=self.AGG_BY)
                        Stacked_Curt = mfunc.df_process_gen_inputs(Stacked_Curt, self.ordered_gen)
                        Stacked_Curt = Stacked_Curt.sum(axis=1)
                        Total_Gen_Stack.insert(len(Total_Gen_Stack.columns),column=curtailment_name,value=Stacked_Curt) #Insert curtailment into
                        Total_Gen_Stack = Total_Gen_Stack.loc[:, (Total_Gen_Stack != 0).any(axis=0)]

                Total_Gen_Stack = Total_Gen_Stack/interval_count

                if not pd.isnull(start_date_range):
                    self.logger.info(f"Plotting specific date range: \
                                      {str(start_date_range)} to {str(end_date_range)}")
                    Total_Gen_Stack = Total_Gen_Stack[start_date_range:end_date_range]

                Total_Gen_Stack = Total_Gen_Stack.sum(axis=0)
                Total_Gen_Stack.rename(scenario, inplace=True)
                
                Total_Load = self.mplot_data_dict[f"{agg}_Load"].get(scenario)
                Total_Load = Total_Load.xs(zone_input,level=self.AGG_BY)
                Total_Load = Total_Load.groupby(["timestamp"]).sum()

                if not pd.isnull(start_date_range):
                    Total_Load = Total_Load[start_date_range:end_date_range]

                Total_Load = Total_Load.rename(columns={0:scenario}).sum(axis=0)
                Total_Load = Total_Load/interval_count

                if self.mplot_data_dict[f"{agg}_Unserved_Energy"] == {}:
                    Unserved_Energy = self.mplot_data_dict[f"{agg}_Load"][scenario].copy()
                    Unserved_Energy.iloc[:,0] = 0
                else:
                    Unserved_Energy = self.mplot_data_dict[f"{agg}_Unserved_Energy"][scenario]
                Unserved_Energy = Unserved_Energy.xs(zone_input,level=self.AGG_BY)
                Unserved_Energy = Unserved_Energy.groupby(["timestamp"]).sum()
                
                if not pd.isnull(start_date_range):
                    Unserved_Energy = Unserved_Energy[start_date_range:end_date_range]
                Unserved_Energy = Unserved_Energy.rename(columns={0:scenario}).sum(axis=0)
                Unserved_Energy = Unserved_Energy/interval_count

                # subtract unserved energy from load for graphing (not sure this is actually used)
                if (Unserved_Energy == 0).all() == False:
                    Unserved_Energy = Total_Load - Unserved_Energy

                if self.mplot_data_dict["generator_Pump_Load"] == {} or not mconfig.parser("plot_data","include_total_pumped_load_line"):
                    Pump_Load = self.mplot_data_dict['generator_Generation'][scenario].copy()
                    Pump_Load.iloc[:,0] = 0
                else:
                    Pump_Load = self.mplot_data_dict["generator_Pump_Load"][scenario]
                Pump_Load = Pump_Load.xs(zone_input,level=self.AGG_BY)
                Pump_Load = Pump_Load.groupby(["timestamp"]).sum()
                if not pd.isnull(start_date_range):
                    Pump_Load = Pump_Load[start_date_range:end_date_range]
                
                Pump_Load = Pump_Load.rename(columns={0:scenario}).sum(axis=0)
                Pump_Load = Pump_Load/interval_count
                if (Pump_Load == 0).all() == False:
                    Total_Demand = Total_Load - Pump_Load
                else:
                    Total_Demand = Total_Load

                gen_chunks.append(Total_Gen_Stack)
                load_chunk.append(Total_Load)
                pumped_load_chunk.append(Pump_Load)
                total_demand_chunk.append(Total_Demand)
                unserved_energy_chunk.append(Unserved_Energy)
            
            Total_Generation_Stack_Out = pd.concat(gen_chunks, axis=1, sort=False).fillna(0)
            Total_Load_Out = pd.concat(load_chunk, axis=0, sort=False)
            Pump_Load_Out = pd.concat(pumped_load_chunk, axis=0, sort=False)
            Total_Demand_Out = pd.concat(total_demand_chunk, axis=0, sort=False)
            Unserved_Energy_Out = pd.concat(unserved_energy_chunk, axis=0, sort=False)

            Total_Load_Out = Total_Load_Out.rename('Total Load (Demand + \n Storage Charging)')
            Total_Demand_Out = Total_Demand_Out.rename('Total Demand')
            Unserved_Energy_Out = Unserved_Energy_Out.rename('Unserved Energy')

            Total_Generation_Stack_Out = mfunc.df_process_categorical_index(Total_Generation_Stack_Out, self.ordered_gen)
            Total_Generation_Stack_Out = Total_Generation_Stack_Out.T
            Total_Generation_Stack_Out = Total_Generation_Stack_Out.loc[:, (Total_Generation_Stack_Out != 0).any(axis=0)]

            if Total_Generation_Stack_Out.empty:
                out = mfunc.MissingZoneData()
                outputs[zone_input] = out
                continue
            
            # unit conversion return divisor and energy units
            unitconversion = mfunc.capacity_energy_unitconversion(max(Total_Generation_Stack_Out.sum(axis=1)))

            Total_Generation_Stack_Out = Total_Generation_Stack_Out/unitconversion['divisor']
            Total_Load_Out = Total_Load_Out.T/unitconversion['divisor']
            Pump_Load_Out = Pump_Load_Out.T/unitconversion['divisor']
            Total_Demand_Out = Total_Demand_Out.T/unitconversion['divisor']
            Unserved_Energy_Out = Unserved_Energy_Out.T/unitconversion['divisor']

            # Data table of values to return to main program
            Data_Table_Out = pd.concat([Total_Load_Out.T,
                                        Total_Demand_Out.T,
                                        Unserved_Energy_Out.T,
                                        Total_Generation_Stack_Out],  axis=1, sort=False)
            Data_Table_Out = Data_Table_Out.add_suffix(f" ({unitconversion['units']}h)")

            Total_Generation_Stack_Out.index = Total_Generation_Stack_Out.index.str.replace('_',' ')
            
            Total_Generation_Stack_Out, angle = mfunc.check_label_angle(Total_Generation_Stack_Out, False)
            fig1, ax = plt.subplots(figsize=(self.x,self.y))

            Total_Generation_Stack_Out.plot.bar(stacked=True, rot=angle, ax=ax,
                             color=[self.PLEXOS_color_dict.get(x, '#333333') for x in Total_Generation_Stack_Out.columns], edgecolor='black', linewidth='0.1')
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_ylabel(f"Total Genertaion ({unitconversion['units']}h)",  color='black', rotation='vertical')
            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(x, f',.{self.y_axes_decimalpt}f')))
            if angle > 0:
                ax.set_xticklabels(Total_Generation_Stack_Out.index, ha="right")
                tick_length = 8
            else:
                tick_length = 5
            ax.tick_params(axis='y', which='major', length=tick_length, width=1)
            ax.tick_params(axis='x', which='major', length=tick_length, width=1)
            if mconfig.parser("plot_title_as_region"):
                ax.set_title(zone_input)
            
            # replace x-axis with custom labels if present 
            if len(self.ticklabels) > 1:
                ticklabels = [textwrap.fill(x.replace('_', ' '), 8) for x in self.ticklabels]
                ax.set_xticklabels(ticklabels)
            
            
            for n, scenario in enumerate(self.Scenarios):

                x = [ax.patches[n].get_x(), ax.patches[n].get_x() + ax.patches[n].get_width()]
                height1 = [float(Total_Load_Out[scenario].sum())]*2
                lp1 = plt.plot(x,height1, c='black', linewidth=3)
                if Pump_Load_Out[scenario] > 0:
                    height2 = [float(Total_Demand_Out[scenario])]*2
                    lp2 = plt.plot(x,height2, 'r--', c='black', linewidth=1.5)

                if Unserved_Energy_Out[scenario] > 0:
                    height3 = [float(Unserved_Energy_Out[scenario])]*2
                    plt.plot(x,height3, c='#DD0200', linewidth=1.5)
                    ax.fill_between(x, height3, height1,
                                facecolor = '#DD0200',
                                alpha=0.5)

            handles, labels = ax.get_legend_handles_labels()

            #Combine all legends into one.
            if Pump_Load_Out.values.sum() > 0:
                handles.append(lp2[0])
                handles.append(lp1[0])
                labels += ['Demand','Demand + \n Storage Charging']
            else:
                handles.append(lp1[0])
                labels += ['Demand']

            if Unserved_Energy_Out.values.sum() > 0:
                handles.append(custom_legend_elements)
                labels += ['Unserved Energy']

            ax.legend(reversed(handles),reversed(labels),loc = 'lower left',bbox_to_anchor=(1.05,0),facecolor='inherit', frameon=True)

            outputs[zone_input] = {'fig': fig1, 'data_table': Data_Table_Out}

        return outputs

    def total_gen_diff(self, figure_name=None, prop=None, start=None, end=None,
                       timezone="", start_date_range=None, end_date_range=None):
        # Create Dictionary to hold Datframes for each scenario
        outputs = {}

        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True,"generator_Generation",self.Scenarios),
                      (False,f"generator_{self.curtailment_prop}",self.Scenarios)]

        # Runs get_data to populate mplot_data_dict with all required properties, returns a 1 if required data is missing
        check_input_data = mfunc.get_data(self.mplot_data_dict, properties,self.Marmot_Solutions_folder)

        if 1 in check_input_data:
            outputs = mfunc.MissingInputData()
            return outputs

        for zone_input in self.Zones:

            self.logger.info(f"Zone = {zone_input}")

            gen_chunks =[]
            for scenario in self.Scenarios:

                self.logger.info(f"Scenario = {scenario}")

                Total_Gen_Stack = self.mplot_data_dict['generator_Generation'].get(scenario)

                #Check if zone has generation, if not skips and breaks out of Multi_Scenario loop
                try:
                    Total_Gen_Stack = Total_Gen_Stack.xs(zone_input,level=self.AGG_BY)
                except KeyError:
                    self.logger.warning(f"No installed capacity in : {zone_input}")
                    break

                Total_Gen_Stack = mfunc.df_process_gen_inputs(Total_Gen_Stack, self.ordered_gen)

                # Calculates interval step to correct for MWh of generation
                interval_count = mfunc.get_sub_hour_interval_count(Total_Gen_Stack)

                curtailment_name = self.gen_names_dict.get('Curtailment','Curtailment')

                # Insert Curtailmnet into gen stack if it exhists in database
                if self.mplot_data_dict[f"generator_{self.curtailment_prop}"]:
                    Stacked_Curt = self.mplot_data_dict[f"generator_{self.curtailment_prop}"].get(scenario)
                    if zone_input in Stacked_Curt.index.get_level_values(self.AGG_BY).unique():
                        Stacked_Curt = Stacked_Curt.xs(zone_input,level=self.AGG_BY)
                        Stacked_Curt = mfunc.df_process_gen_inputs(Stacked_Curt, self.ordered_gen)
                        Stacked_Curt = Stacked_Curt.sum(axis=1)
                        Total_Gen_Stack.insert(len(Total_Gen_Stack.columns),column=curtailment_name,value=Stacked_Curt) #Insert curtailment into
                        Total_Gen_Stack = Total_Gen_Stack.loc[:, (Total_Gen_Stack != 0).any(axis=0)]

                Total_Gen_Stack = Total_Gen_Stack/interval_count
                if not pd.isnull(start_date_range):
                    self.logger.info(f"Plotting specific date range: \
                                     {str(start_date_range)} to {str(end_date_range)}")
                    Total_Gen_Stack = Total_Gen_Stack[start_date_range:end_date_range]
                
                Total_Gen_Stack = Total_Gen_Stack.sum(axis=0)
                Total_Gen_Stack.rename(scenario, inplace=True)
                
                gen_chunks.append(Total_Gen_Stack)
            
            Total_Generation_Stack_Out = pd.concat(gen_chunks, axis=1, sort=False).fillna(0)

            Total_Generation_Stack_Out = mfunc.df_process_categorical_index(Total_Generation_Stack_Out, self.ordered_gen)
            Total_Generation_Stack_Out = Total_Generation_Stack_Out.T
            Total_Generation_Stack_Out = Total_Generation_Stack_Out.loc[:, (Total_Generation_Stack_Out != 0).any(axis=0)]

            #Ensures region has generation, else skips
            try:
                Total_Generation_Stack_Out = Total_Generation_Stack_Out-Total_Generation_Stack_Out.xs(self.Scenarios[0]) #Change to a diff on first scenario
            except KeyError:
                out = mfunc.MissingZoneData()
                outputs[zone_input] = out
                continue

            Total_Generation_Stack_Out.drop(self.Scenarios[0],inplace=True) #Drop base entry

            if Total_Generation_Stack_Out.empty == True:
                out = mfunc.MissingZoneData()
                outputs[zone_input] = out
                continue
            
            unitconversion = mfunc.capacity_energy_unitconversion(max(abs(Total_Generation_Stack_Out.sum(axis=1))))
            Total_Generation_Stack_Out = Total_Generation_Stack_Out/unitconversion['divisor']

            # Data table of values to return to main program
            Data_Table_Out = Total_Generation_Stack_Out.add_suffix(f" ({unitconversion['units']}h)")

            net_diff = Total_Generation_Stack_Out
            try:
                net_diff.drop(columns = curtailment_name,inplace=True)
            except KeyError:
                pass
            net_diff = net_diff.sum(axis = 1)

            Total_Generation_Stack_Out.index = Total_Generation_Stack_Out.index.str.replace('_',' ')

            Total_Generation_Stack_Out, angle = mfunc.check_label_angle(Total_Generation_Stack_Out, False)
            
            fig1, ax = plt.subplots(figsize=(self.x,self.y))
            Total_Generation_Stack_Out.plot.bar(stacked=True, rot=angle,
                             color=[self.PLEXOS_color_dict.get(x, '#333333') for x in Total_Generation_Stack_Out.columns], edgecolor='black', linewidth='0.1',ax=ax)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(x, f',.{self.y_axes_decimalpt}f')))
            if angle > 0:
                ax.set_xticklabels(Total_Generation_Stack_Out.index, ha="right")
                tick_length = 8
            else:
                tick_length = 5
            ax.tick_params(axis='y', which='major', length=tick_length, width=1)
            ax.tick_params(axis='x', which='major', length=tick_length, width=1)

            #Add net gen difference line.
            for n, scenario in enumerate(self.Scenarios[1:]):
                x = [ax.patches[n].get_x(), ax.patches[n].get_x() + ax.patches[n].get_width()]
                y_net = [net_diff.loc[scenario]] * 2
                net_line = plt.plot(x,y_net, c='black', linewidth=1.5)

            locs,labels=plt.xticks()

            ax.set_ylabel(f"Generation Change ({format(unitconversion['units'])}h) \n relative to {self.Scenarios[0].replace('_',' ')}",  color='black', rotation='vertical')
            
            # xlabels = [textwrap.fill(x.replace('_',' '),10) for x in self.xlabels]

            # plt.xticks(ticks=locs,labels=xlabels[1:])
            # ax.margins(x=0.01)

            plt.axhline(linewidth=0.5,linestyle='--',color='grey')

            handles, labels = ax.get_legend_handles_labels()

            handles.append(net_line[0])
            labels += ['Net Gen Change']

            #Main legend
            ax.legend(reversed(handles), reversed(labels), loc='lower left',bbox_to_anchor=(1,0),facecolor='inherit', frameon=True)
            if mconfig.parser("plot_title_as_region"):
                ax.set_title(zone_input)
            outputs[zone_input] = {'fig': fig1, 'data_table': Data_Table_Out}
        return outputs

        
    def total_gen_monthly(self, **kwargs):
        """ Total generation by Month"""

        outputs = self._monthly_gen(**kwargs)
        return outputs

    def monthly_vre_generation_percentage(self, **kwargs):
        """Monthly Total Variable Renewable Generation by technology percentage,
           Each vre technology is plotted as a bar, the total of all bars add to 100%
           Each sceanrio is plotted on a seperate facet plot 
           Technologies that belong to VRE can be set in the vre_gen_cat.csv file 
           in the Mapping folder
        """
        outputs = self._monthly_gen(vre_only=True, plot_as_percnt=True, **kwargs)
        return outputs
    
    def monthly_vre_generation(self, **kwargs):
        """Monthly Total Variable Renewable Generation
            Each vre technology is plotted as a bar
            Each sceanrio is plotted on a seperate facet plot 
           Technologies that belong to VRE can be set in the vre_gen_cat.csv file 
           in the Mapping folder
        """
        outputs = self._monthly_gen(vre_only=True, **kwargs)
        return outputs

    def _monthly_gen(self, vre_only=False, plot_as_percnt=False, figure_name=None, prop=None, start=None, end=None,
                  timezone=None, start_date_range=None, end_date_range=None):
        """ Creates monthly generation plot, internal method called from 
            monthly_vre_percentage_generation or monthly_vre_generation
        """
        
        # Create Dictionary to hold Datframes for each scenario
        outputs = {}
        
        if self.AGG_BY == 'zone':
            agg = 'zone'
        else:
            agg = 'region'

        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True,"generator_Generation",self.Scenarios),
                      (False,f"generator_{self.curtailment_prop}",self.Scenarios),
                      (False,"generator_Pump_Load",self.Scenarios),
                      (True,f"{agg}_Load",self.Scenarios)]

        # Runs get_data to populate mplot_data_dict with all required properties, returns a 1 if required data is missing
        check_input_data = mfunc.get_data(self.mplot_data_dict, properties, self.Marmot_Solutions_folder)

        if 1 in check_input_data:
            outputs = mfunc.MissingInputData()
            return outputs
        
        xdimension, ydimension = mfunc.setup_facet_xy_dimensions(self.xlabels, self.ylabels, 
                                                                    multi_scenario=self.Scenarios)
        grid_size = xdimension*ydimension

        # Used to calculate any excess axis to delete
        plot_number = len(self.Scenarios)
        excess_axs = grid_size - plot_number
        
        if xdimension > 1:
            font_scaling_ratio = 1 + ((xdimension-1)*0.09)
            plt.rcParams['xtick.labelsize'] = plt.rcParams['xtick.labelsize']*font_scaling_ratio
            plt.rcParams['ytick.labelsize'] = plt.rcParams['ytick.labelsize']*font_scaling_ratio
            plt.rcParams['legend.fontsize'] = plt.rcParams['legend.fontsize']*font_scaling_ratio
            plt.rcParams['axes.labelsize'] = plt.rcParams['axes.labelsize']*font_scaling_ratio
            plt.rcParams['axes.titlesize'] =  plt.rcParams['axes.titlesize']*font_scaling_ratio
         
        for zone_input in self.Zones:
            
            self.logger.info(f"Zone = {zone_input}")
            
            # Will hold retrieved data for each scenario
            gen_chunks = []
            load_chunk = []
            pumped_load_chunk = []
            total_demand_chunk = []

            # Loop gets all data by scenario
            for i, scenario in enumerate(self.Scenarios):
                
                self.logger.info(f"Scenario = {scenario}")
                Total_Gen_Stack = self.mplot_data_dict['generator_Generation'].get(scenario)
    
                #Check if zone has generation, if not skips
                try:
                    Total_Gen_Stack = Total_Gen_Stack.xs(zone_input,level=self.AGG_BY)
                except KeyError:
                    self.logger.warning(f"No installed capacity in: {zone_input}")
                    continue
                
                if vre_only:
                    Total_Gen_Stack = (Total_Gen_Stack.loc[(slice(None), self.vre_gen_cat),:])
                
                Total_Gen_Stack = mfunc.df_process_gen_inputs(Total_Gen_Stack, self.ordered_gen)
                Total_Gen_Stack.columns = Total_Gen_Stack.columns.add_categories('timestamp')
    
                # Calculates interval step to correct for MWh of generation if data is subhourly
                interval_count = mfunc.get_sub_hour_interval_count(Total_Gen_Stack)
    
                #Insert Curtailment into gen stack if it exhists in database
                if self.mplot_data_dict[f"generator_{self.curtailment_prop}"]:
                    curtailment_name = self.gen_names_dict.get('Curtailment','Curtailment')
                    Stacked_Curt = self.mplot_data_dict[f"generator_{self.curtailment_prop}"].get(scenario)
                    if zone_input in Stacked_Curt.index.get_level_values(self.AGG_BY).unique():
                        Stacked_Curt = Stacked_Curt.xs(zone_input, level=self.AGG_BY)
                        Stacked_Curt = mfunc.df_process_gen_inputs(Stacked_Curt, self.ordered_gen)
                        Stacked_Curt = Stacked_Curt.sum(axis=1)
                        Total_Gen_Stack.insert(len(Total_Gen_Stack.columns), column=curtailment_name, value=Stacked_Curt) 
                        Total_Gen_Stack = Total_Gen_Stack.loc[:, (Total_Gen_Stack != 0).any(axis=0)]

                # Get Total Load 
                Total_Load = self.mplot_data_dict[f"{agg}_Load"].get(scenario)
                Total_Load = Total_Load.xs(zone_input, level=self.AGG_BY)
                Total_Load = Total_Load.groupby(["timestamp"]).sum()

                # Get Pumped Load 
                if self.mplot_data_dict["generator_Pump_Load"] == {} or not mconfig.parser("plot_data","include_total_pumped_load_line"):
                    Pump_Load = self.mplot_data_dict['generator_Generation'][scenario].copy()
                    Pump_Load.iloc[:,0] = 0
                else:
                    Pump_Load = self.mplot_data_dict["generator_Pump_Load"][scenario]
                Pump_Load = Pump_Load.xs(zone_input, level=self.AGG_BY)
                Pump_Load = Pump_Load.groupby(["timestamp"]).sum()

                def _group_monthly_data(input_df, interval_count):
                    '''Groups data into months'''

                    monthly_df = input_df/interval_count
                    monthly_df = monthly_df.groupby(pd.Grouper(freq='M')).sum()
                    if len(monthly_df.index) > 12:
                        monthly_df = monthly_df[:-1]
                    monthly_df.reset_index(drop=False, inplace=True)
                    monthly_df['timestamp'] = monthly_df['timestamp'].dt.month.apply(lambda x: self.MONTHS[x])
                    monthly_df.set_index('timestamp', inplace=True)
                    return monthly_df

                # Group data into months
                monthly_gen_stack = _group_monthly_data(Total_Gen_Stack, interval_count)  
                monthly_total_load = _group_monthly_data(Total_Load, interval_count)  
                monthly_pumped_load = _group_monthly_data(Pump_Load, interval_count)         
                # Calculate Total Demand 
                monthly_total_demand = monthly_total_load - monthly_pumped_load
                
                # If plotting percentage data convert to percentages 
                if plot_as_percnt:
                    monthly_total_gen = pd.DataFrame(monthly_gen_stack.T.sum(), columns=['Total Generation'])
                    for vre_col in monthly_gen_stack.columns:
                        monthly_gen_stack[vre_col] = (monthly_gen_stack[vre_col] / monthly_total_gen['Total Generation'])

                # Add scenario index 
                scenario_names = pd.Series([scenario] * len(monthly_gen_stack), name='Scenario')
                monthly_gen_stack = monthly_gen_stack.set_index([scenario_names], append = True)
                monthly_total_load = monthly_total_load.set_index([scenario_names], append = True)
                monthly_pumped_load = monthly_pumped_load.set_index([scenario_names], append = True)
                monthly_total_demand = monthly_total_demand.set_index([scenario_names], append = True)

                # Add all data to lists
                gen_chunks.append(monthly_gen_stack)
                load_chunk.append(monthly_total_load)
                pumped_load_chunk.append(monthly_pumped_load)
                total_demand_chunk.append(monthly_total_demand)

            # Concat all data into single data-frames
            Gen_Out = pd.concat(gen_chunks,axis=0, sort=False)
            Pump_Load_Out = pd.concat(pumped_load_chunk, axis=0, sort=False)
            Total_Demand_Out = pd.concat(total_demand_chunk, axis=0, sort=False)
            Total_Load_Out = pd.concat(load_chunk, axis=0, sort=False)
            
            # Rename load columns
            Pump_Load_Out.rename(columns={0:'Pump Load'}, inplace=True)
            Total_Demand_Out.rename(columns={0:'Total Demand'}, inplace=True)
            Total_Load_Out.rename(columns={0:'Total Load (Demand + \n Storage Charging)'}, inplace=True)
            
            # Drop any technologies with 0 Gen
            Gen_Out = Gen_Out.loc[:, (Gen_Out != 0).any(axis=0)]

            if Gen_Out.empty:
                out = mfunc.MissingZoneData()
                outputs[zone_input] = out
                continue
            
            # Determine max value of data-frame 
            if vre_only:
                max_value = Gen_Out.to_numpy().max()
            else:
                max_value = max(Gen_Out.sum())
            
            if not plot_as_percnt:
                # unit conversion return divisor and energy units
                unitconversion = mfunc.capacity_energy_unitconversion(max_value)
                Gen_Out = Gen_Out/unitconversion['divisor']
                Total_Demand_Out = Total_Demand_Out/unitconversion['divisor']
                Total_Load_Out = Total_Load_Out/unitconversion['divisor']
                Pump_Load_Out = Pump_Load_Out/unitconversion['divisor']
                #Data table of values to return to main program
                Data_Table_Out = pd.concat([Total_Load_Out, Total_Demand_Out, Pump_Load_Out, 
                                            Gen_Out], axis=1).add_suffix(f" ({unitconversion['units']}h)")
            else:
                Data_Table_Out = Gen_Out.add_suffix(f" (%-Gen)") * 100

            fig, axs = mfunc.setup_plot(xdimension, ydimension, sharey=True)
            plt.subplots_adjust(wspace=0.05, hspace=0.5)

            unique_tech_names = []
            for i, scenario in enumerate(self.Scenarios):

                month_gen = Gen_Out.xs(scenario, level="Scenario")
                month_total_load = Total_Load_Out.xs(scenario, level="Scenario")
                month_pumped_load = Pump_Load_Out.xs(scenario, level="Scenario")
                month_total_demand = Total_Demand_Out.xs(scenario, level="Scenario")

                month_gen, angle = mfunc.check_label_angle(month_gen, False)
                
                if vre_only:
                    stack = False
                else:
                    stack = True

                month_gen.plot.bar(stacked=stack, rot=angle, ax=axs[i],
                                  color=[self.PLEXOS_color_dict.get(x, '#333333') for x in month_gen.columns], 
                                  edgecolor='black', linewidth='0.1', legend=False)
                axs[i].spines['right'].set_visible(False)
                axs[i].spines['top'].set_visible(False)
                axs[i].margins(x=0.01)
                axs[i].set_xlabel("")
                
                if plot_as_percnt:
                    axs[i].yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))  
                else:
                    axs[i].yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(x, f',.{self.y_axes_decimalpt}f')))
                
                if angle > 0:
                    axs[i].set_xticklabels(month_gen.index, ha="right")

                axs[i].tick_params(axis='y', which='major', length=5, width=1)
                axs[i].tick_params(axis='x', which='major', length=5, width=1)

                # create list of gen technologies
                l1 = month_gen.columns.tolist()
                unique_tech_names.extend(l1)

                if not vre_only:
                    for n, _m in enumerate(month_total_load.index):
                        x = [axs[i].patches[n].get_x(), axs[i].patches[n].get_x() + axs[i].patches[n].get_width()]
                        height1 = [float(month_total_load.loc[_m])]*2
                        lp1 = axs[i].plot(x, height1, c='black', linewidth=3)
                        if month_pumped_load.loc[_m].values.sum() > 0:
                            height2 = [float(month_total_demand.loc[_m])]*2
                            lp2 = axs[i].plot(x,height2, 'r--', c='black', linewidth=1.5)

            # create labels list of unique tech names then order
            labels = np.unique(np.array(unique_tech_names)).tolist()
            labels.sort(key = lambda i:self.ordered_gen.index(i))

            handles = []
            # create custom gen_tech legend
            for tech in labels:
                gen_legend_patches = Patch(facecolor=self.PLEXOS_color_dict[tech],
                            alpha=1.0)
                handles.append(gen_legend_patches)

            if not vre_only:
                #Combine all legends into one.
                if Pump_Load_Out.values.sum() > 0:
                    handles.append(lp2[0])
                    handles.append(lp1[0])
                    labels += ['Demand','Demand + \n Storage Charging']
                else:
                    handles.append(lp1[0])
                    labels += ['Demand']

            axs[grid_size-1].legend(reversed(handles),reversed(labels),
                                    loc = 'lower left',bbox_to_anchor=(1.05,0),
                                    facecolor='inherit', frameon=True)

            # add facet labels
            mfunc.add_facet_labels(fig, self.xlabels, self.ylabels)
            
            fig.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            
            #Y-label should change if there are facet labels, leave at 40 for now, works for all values in spacing
            labelpad = 40
            if plot_as_percnt:
                plt.ylabel(f"% of Generation",  color='black', rotation='vertical', labelpad=labelpad)
            else:
                plt.ylabel(f"Total Genertaion ({unitconversion['units']}h)",  color='black', rotation='vertical', labelpad=labelpad)

            if mconfig.parser('plot_title_as_region'):
                plt.title(zone_input)

            #Remove extra axes
            if excess_axs != 0:
                mfunc.remove_excess_axs(axs, excess_axs, grid_size)
            
            outputs[zone_input] = {'fig': fig, 'data_table': Data_Table_Out}

        return outputs
    

    def total_gen_pie(self, figure_name=None, prop=None, start=None, end=None,
                  timezone=None, start_date_range=None, end_date_range=None):
        """Total Generation Pie Chart """
        
        # Create Dictionary to hold Datframes for each scenario
        
        outputs = {}
                    
        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True,"generator_Generation",self.Scenarios), 
                      (False,f"generator_{self.curtailment_prop}",self.Scenarios)]


        # Runs get_data to populate mplot_data_dict with all required properties, returns a 1 if required data is missing
        check_input_data = mfunc.get_data(self.mplot_data_dict, properties,self.Marmot_Solutions_folder)

        if 1 in check_input_data:
            outputs = mfunc.MissingInputData()
            return outputs
        
        xdimension, ydimension = mfunc.setup_facet_xy_dimensions(self.xlabels,self.ylabels,multi_scenario=self.Scenarios)
        grid_size = xdimension*ydimension

        # Used to calculate any excess axis to delete
        plot_number = len(self.Scenarios)
        excess_axs = grid_size - plot_number
        
        for zone_input in self.Zones:
            Total_Gen_Out = pd.DataFrame()
            self.logger.info(f"Zone = {zone_input}")

            fig, axs = mfunc.setup_plot(xdimension, ydimension)
            plt.subplots_adjust(wspace=0.05, hspace=0.5)
            axs = axs.ravel()
            
            gen_chunks = []
            for i, scenario in enumerate(self.Scenarios):
                
                self.logger.info(f"Scenario = {scenario}")
                Total_Gen_Stack = self.mplot_data_dict['generator_Generation'].get(scenario)
    
                #Check if zone has generation, if not skips
                try:
                    Total_Gen_Stack = Total_Gen_Stack.xs(zone_input,level=self.AGG_BY)
                except KeyError:
                    self.logger.warning(f"No installed capacity in: {zone_input}")
                    continue
    
                Total_Gen_Stack = mfunc.df_process_gen_inputs(Total_Gen_Stack, self.ordered_gen)
                
                curtailment_name = self.gen_names_dict.get('Curtailment','Curtailment')
    
                #Insert Curtailmnet into gen stack if it exhists in database
                if self.mplot_data_dict[f"generator_{self.curtailment_prop}"]:
                    Stacked_Curt = self.mplot_data_dict[f"generator_{self.curtailment_prop}"].get(scenario)
                    if zone_input in Stacked_Curt.index.get_level_values(self.AGG_BY).unique():
                        Stacked_Curt = Stacked_Curt.xs(zone_input,level=self.AGG_BY)
                        Stacked_Curt = mfunc.df_process_gen_inputs(Stacked_Curt, self.ordered_gen)
                        Stacked_Curt = Stacked_Curt.sum(axis=1)
                        Total_Gen_Stack.insert(len(Total_Gen_Stack.columns),column=curtailment_name,value=Stacked_Curt) #Insert curtailment into
                        Total_Gen_Stack = Total_Gen_Stack.loc[:, (Total_Gen_Stack != 0).any(axis=0)]
                
                Total_Gen_Stack = Total_Gen_Stack.sum(axis=0)
                Total_Gen_Stack.rename(scenario, inplace=True)
                Total_Gen_Stack = (Total_Gen_Stack/sum(Total_Gen_Stack))*100
                gen_chunks.append(Total_Gen_Stack)
                
            Total_Gen_Out = pd.concat(gen_chunks, axis=1, sort=False).fillna(0)
                
            Total_Gen_Out = Total_Gen_Out.loc[:, (Total_Gen_Out != 0).any(axis=0)]
            
            if Total_Gen_Out.empty:
                out = mfunc.MissingZoneData()
                outputs[zone_input] = out
                continue
            
            unique_tech_names = []

            for i, scenario in enumerate(self.Scenarios):
                
                scenario_data = Total_Gen_Out[scenario]
               
                axs[i].pie(scenario_data, labels=scenario_data.index, 
                                       shadow=True, startangle=90, labeldistance=None,
                                       colors=[self.PLEXOS_color_dict.get(x, '#333333') for x in scenario_data.index])
                
                # create list of gen technologies
                l1 = scenario_data.index.tolist()
                unique_tech_names.extend(l1)

                axs[i].legend().set_visible(False)
            
             # create labels list of unique tech names then order
            labels = np.unique(np.array(unique_tech_names)).tolist()
            labels.sort(key = lambda i:self.ordered_gen.index(i))

            handles = []
            # create custom gen_tech legend
            for tech in labels:
                gen_legend_patches = Patch(facecolor=self.PLEXOS_color_dict[tech],
                            alpha=1.0)
                handles.append(gen_legend_patches)

            axs[grid_size-1].legend(reversed(handles),reversed(labels),
                                    loc = 'lower left',bbox_to_anchor=(1.05,0),
                                    facecolor='inherit', frameon=True)

            # add facet labels
            mfunc.add_facet_labels(fig, self.xlabels, self.ylabels)
            
            fig.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            plt.ylabel(f"Total Genertaion (%)",  color='black', rotation='vertical')
            if mconfig.parser('plot_title_as_region'):
                plt.title(zone_input)

            #Remove extra axes
            if excess_axs != 0:
                mfunc.remove_excess_axs(axs, excess_axs, grid_size)
            
            outputs[zone_input] = {'fig': fig, 'data_table': Total_Gen_Out}

        return outputs


    #===============================================================================
    ## Total Gen Facet Plots removed for now, code not stable and needs testing
    #===============================================================================

    def total_gen_facet(self, figure_name=None, prop=None, start=None, end=None,
                        timezone="", start_date_range=None, end_date_range=None):
        outputs = mfunc.UnderDevelopment()
        self.logger.warning('total_gen_facet is under development')
        return outputs

    #     self.mplot_data_dict['generator_Generation'] = {}
    #     self.mplot_data_dictf"{self.AGG_BY}_Load"] = {}
    #     self.mplot_data_dict[f"generator_{self.curtailment_prop}"] = {}

    #     for scenario in self.Scenarios:
    #         try:
    #             self.mplot_data_dict['generator_Generation'][scenario] = pd.read_hdf(os.path.join(self.Marmot_Solutions_folder, scenario,"Processed_HDF5_folder", scenario+ "_formatted.h5"), "generator_Generation")
    #             self.mplot_data_dict[f"generator_{self.curtailment_prop}"][scenario] = pd.read_hdf(os.path.join(self.Marmot_Solutions_folder, scenario,"Processed_HDF5_folder", scenario+ "_formatted.h5"),  f"generator_{self.curtailment_prop}")
    #             # If data is to be agreagted by zone, then zone properties are loaded, else region properties are loaded
    #             if self.AGG_BY == "zone":
    #                 self.mplot_data_dictf"{self.AGG_BY}_Load"][scenario] = pd.read_hdf(os.path.join(self.Marmot_Solutions_folder, scenario,"Processed_HDF5_folder", scenario+ "_formatted.h5"), "zone_Load")
    #             else:
    #                 self.mplot_data_dictf"{self.AGG_BY}_Load"][scenario] = pd.read_hdf(os.path.join(self.Marmot_Solutions_folder, scenario,"Processed_HDF5_folder", scenario+ "_formatted.h5"),  "region_Load")

    #         except Exception:
    #             pass


    #     Total_Generation_Stack_Out = pd.DataFrame()
    #     Total_Load_Out = pd.DataFrame()
    #     self.logger.info("Zone = " + self.zone_input)


    #     for scenario in self.Scenarios:
    #         self.logger.info("Scenario = " + scenario)
    #         try:
    #             Total_Gen_Stack = self.mplot_data_dict['generator_Generation'].get(scenario)
    #             Total_Gen_Stack = Total_Gen_Stack.xs(self.zone_input,level=self.AGG_BY)
    #             Total_Gen_Stack = df_process_gen_inputs(Total_Gen_Stack, self)
    #             Stacked_Curt = self.mplot_data_dict[f"generator_{self.curtailment_prop}"].get(scenario)
    #             Stacked_Curt = Stacked_Curt.xs(self.zone_input,level=self.AGG_BY)
    #             Stacked_Curt = df_process_gen_inputs(Stacked_Curt, self)
    #             Stacked_Curt = Stacked_Curt.sum(axis=1)
    #             Total_Gen_Stack.insert(len(Total_Gen_Stack.columns),column='Curtailment',value=Stacked_Curt) #Insert curtailment into
    #             Total_Gen_Stack = Total_Gen_Stack.loc[:, (Total_Gen_Stack != 0).any(axis=0)]

    #             Total_Gen_Stack = Total_Gen_Stack.sum(axis=0)
    #             Total_Gen_Stack.rename(scenario, inplace=True)

    #             Total_Generation_Stack_Out = pd.concat([Total_Generation_Stack_Out, Total_Gen_Stack], axis=1, sort=False).fillna(0)

    #             Total_Load = self.mplot_data_dictf"{self.AGG_BY}_Load"].get(scenario)
    #             Total_Load = Total_Load.xs(self.zone_input,level=self.AGG_BY)
    #             Total_Load = Total_Load.groupby(["timestamp"]).sum()
    #             Total_Load = Total_Load.rename(columns={0:scenario}).sum(axis=0)
    #             Total_Load_Out = pd.concat([Total_Load_Out, Total_Load], axis=0, sort=False)
    #         except Exception:
    #             self.logger.warning("Error: Skipping " + scenario)
    #             pass

    #     Total_Load_Out = Total_Load_Out.rename(columns={0:'Total Load'})

    #     Total_Generation_Stack_Out = df_process_categorical_index(Total_Generation_Stack_Out, self)
    #     Total_Generation_Stack_Out = Total_Generation_Stack_Out.T/1000
    #     Total_Generation_Stack_Out = Total_Generation_Stack_Out.loc[:, (Total_Generation_Stack_Out != 0).any(axis=0)]

    #     # Data table of values to return to main program
    #     Data_Table_Out = pd.concat([Total_Load_Out/1000, Total_Generation_Stack_Out],  axis=1, sort=False)

    #     Total_Generation_Stack_Out.index = Total_Generation_Stack_Out.index.str.replace('_',' ')
    
    #     Total_Generation_Stack_Out.index = mfunc.check_label_angle(Total_Generation_Stack_Out,False)

    #     Total_Load_Out.index = Total_Load_Out.index.str.replace('_',' ')
    #     Total_Load_Out.index = Total_Load_Out.index.str.wrap(10, break_long_words=False)
    #     
    #     Total_Load_Out = Total_Load_Out.T/1000

    #     xdimension=len(self.xlabels)
    #     ydimension=len(self.ylabels)
    #     grid_size = xdimension*ydimension

    #     fig2, axs = plt.subplots(ydimension,xdimension, figsize=((2*xdimension), (4*ydimension)), sharey=True)
    #     axs = axs.ravel()
    #     plt.subplots_adjust(wspace=0, hspace=0.01)

    #     i=0
    #     for index in Total_Generation_Stack_Out.index:

    #         sb = Total_Generation_Stack_Out.iloc[i:i+1].plot.bar(stacked=True, rot=angle,
    #         color=[self.PLEXOS_color_dict.get(x, '#333333') for x in Total_Generation_Stack_Out.columns], edgecolor='black', linewidth='0.1',
    #                                      ax=axs[i])

    #         axs[i].get_legend().remove()
    #         axs[i].spines['right'].set_visible(False)
    #         axs[i].spines['top'].set_visible(False)
    #         axs[i].xaxis.set_ticklabels([])
    #         if angle > 0:
    #             ax.set_xticklabels(Total_Generation_Stack_Out.iloc[i:i+1].index, ha="right")
    #             tick_length = 8
    #         else:
    #             tick_length = 5
    #         ax.tick_params(axis='y', which='major', length=tick_length, width=1)
    #         ax.tick_params(axis='x', which='major', length=tick_length, width=1)
    #         axs[i].yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    #         axs[i].margins(x=0.01)

    #         height = [int(Total_Load_Out[index])]
    #         axs[i].axhline(y=height,xmin=0.25,xmax=0.75, linestyle ='--', c="black",linewidth=1.5)

    #         handles, labels = axs[1].get_legend_handles_labels()

    #         leg1 = axs[grid_size-1].legend(reversed(handles), reversed(labels), loc='lower left',bbox_to_anchor=(1,0),
    #                   facecolor='inherit', frameon=True)

    #         #Legend 2
    #         leg2 = axs[grid_size-1].legend(['Load'], loc='upper left',bbox_to_anchor=(1, 0.95),
    #                       facecolor='inherit', frameon=True)

    #         fig2.add_artist(leg1)

    #         i=i+1

    #     all_axes = fig2.get_axes()

    #     self.xlabels = pd.Series(self.xlabels).str.replace('_',' ').str.wrap(10, break_long_words=False)

    #     j=0
    #     k=0
    #     for ax in all_axes:
    #         if ax.is_last_row():
    #             ax.set_xlabel(xlabel=(self.xlabels[j]),  color='black')
    #             j=j+1
    #         if ax.is_first_col():
    #             ax.set_ylabel(ylabel=(self.ylabels[k]),  color='black', rotation='vertical')
    #             k=k+1

    #     fig2.add_subplot(111, frameon=False)
    #     plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    #     plt.ylabel('Total Genertaion (GWh)',  color='black', rotation='vertical', labelpad=60)

    #     return {'fig': fig2, 'data_table': Data_Table_Out}
    

     