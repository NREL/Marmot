# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 08:51:15 2019

@author: dlevie
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
import pdb
import logging
import marmot.plottingmodules.total_generation as gen
import marmot.plottingmodules.marmot_plot_functions as mfunc
import marmot.config.mconfig as mconfig

#===============================================================================

custom_legend_elements = Patch(facecolor='#DD0200',
                            alpha=0.5, edgecolor='#DD0200')

class mplot(object):
    def __init__(self, argument_dict):
        # iterate over items in argument_dict and set as properties of class
        # see key_list in Marmot_plot_main for list of properties
        for prop in argument_dict:
            self.__setattr__(prop, argument_dict[prop])

        # used for combined cap/gen plot
        self.argument_dict = argument_dict
        self.logger = logging.getLogger('marmot_plot.'+__name__)
        
        self.x = mconfig.parser("figure_size","xdimension")
        self.y = mconfig.parser("figure_size","ydimension")
        self.y_axes_decimalpt = mconfig.parser("axes_options","y_axes_decimalpt")
        self.set_title = mconfig.parser("plot_title_as_region")

    def total_cap(self):
        outputs = {}
        installed_capacity_collection = {}
        check_input_data = []
        
        check_input_data.extend([mfunc.get_data(installed_capacity_collection,"generator_Installed_Capacity", self.Marmot_Solutions_folder, self.Scenarios)])
        
        # Checks if all data required by plot is available, if 1 in list required data is missing
        if 1 in check_input_data:
            outputs = mfunc.MissingInputData()
            return outputs
        
        for zone_input in self.Zones:
            Total_Installed_Capacity_Out = pd.DataFrame()
            Data_Table_Out = pd.DataFrame()
            self.logger.info(self.AGG_BY + " = " + zone_input)

            for scenario in self.Scenarios:

                self.logger.info("Scenario = " + scenario)

                Total_Installed_Capacity = installed_capacity_collection.get(scenario)
                
                zones_with_cap = Total_Installed_Capacity.index.get_level_values(self.AGG_BY).unique()
                if scenario == 'ADS':
                    zone_input_adj = zone_input.split('_WI')[0]
                else:
                    zone_input_adj = zone_input
                if zone_input_adj in zones_with_cap:
                    Total_Installed_Capacity = Total_Installed_Capacity.xs(zone_input_adj,level=self.AGG_BY)
                else:
                    self.logger.warning("No installed capacity in %s",zone_input)
                    outputs[zone_input] = mfunc.MissingZoneData()
                    continue

                Total_Installed_Capacity = mfunc.df_process_gen_inputs(Total_Installed_Capacity, self.ordered_gen)
                Total_Installed_Capacity.reset_index(drop=True, inplace=True)
                Total_Installed_Capacity.rename(index={0:scenario}, inplace=True)

                Total_Installed_Capacity_Out = pd.concat([Total_Installed_Capacity_Out, Total_Installed_Capacity], axis=0, sort=False).fillna(0)

            Total_Installed_Capacity_Out = Total_Installed_Capacity_Out.loc[:, (Total_Installed_Capacity_Out != 0).any(axis=0)]

            # If Total_Installed_Capacity_Out df is empty returns a empty dataframe and does not plot
            if Total_Installed_Capacity_Out.empty:
                self.logger.warning("No installed capacity in %s",zone_input)
                out = mfunc.MissingZoneData()
                outputs[zone_input] = out
                continue

            # Data table of values to return to main program
            Data_Table_Out = pd.concat([Data_Table_Out, Total_Installed_Capacity_Out],  axis=1, sort=False)
            
            unitconversion = mfunc.capacity_energy_unitconversion(max(Total_Installed_Capacity_Out.sum()))
            Total_Installed_Capacity_Out = Total_Installed_Capacity_Out/unitconversion['divisor'] 
            
            Total_Installed_Capacity_Out.index = Total_Installed_Capacity_Out.index.str.replace('_',' ')
            Total_Installed_Capacity_Out.index = Total_Installed_Capacity_Out.index.str.wrap(5, break_long_words=False)

			fig1, ax = plt.subplots(figsize=(self.x,self.y))
            Total_Installed_Capacity_Out.plot.bar(stacked=True, figsize=(self.x,self.y), rot=0,
                                 color=[self.PLEXOS_color_dict.get(x, '#333333') for x in Total_Installed_Capacity_Out.columns], edgecolor='black', linewidth='0.1',ax=ax)

            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_ylabel('Total Installed Capacity ({})'.format(unitconversion['units']),  color='black', rotation='vertical')
            if mconfig.parser("plot_title_as_region"):
                ax.set_title(zone_input)

            #adds comma to y axis data
            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(x, f',.{self.y_axes_decimalpt}f')))
            ax.tick_params(axis='y', which='major', length=5, width=1)
            ax.tick_params(axis='x', which='major', length=5, width=1)

            handles, labels = fig1.get_legend_handles_labels()
            ax.legend(reversed(handles), reversed(labels), loc='lower left',bbox_to_anchor=(1,0),
                          facecolor='inherit', frameon=True)


            outputs[zone_input] = {'fig': fig1, 'data_table': Data_Table_Out}
        return outputs

    def total_cap_diff(self):
        outputs = {}
        installed_capacity_collection = {}
        check_input_data = []
        
        check_input_data.extend([mfunc.get_data(installed_capacity_collection,"generator_Installed_Capacity", self.Marmot_Solutions_folder, self.Scenarios)])
        
        # Checks if all data required by plot is available, if 1 in list required data is missing
        if 1 in check_input_data:
            outputs = mfunc.MissingInputData()
            return outputs
        
        for zone_input in self.Zones:
            Total_Installed_Capacity_Out = pd.DataFrame()
            Data_Table_Out = pd.DataFrame()
            self.logger.info(self.AGG_BY + " = " + zone_input)

            for scenario in self.Scenarios:

                self.logger.info("Scenario = " + scenario)

                Total_Installed_Capacity = installed_capacity_collection.get(scenario)
                zones_with_cap = Total_Installed_Capacity.index.get_level_values(self.AGG_BY).unique()
                if scenario == 'ADS':
                    zone_input_adj = zone_input.split('_WI')[0]
                    Total_Installed_Capacity.index = pd.MultiIndex.from_frame(Total_Installed_Capacity.index.to_frame().fillna('All')) #Fix NaN values from formatter
                    zones_with_cap = Total_Installed_Capacity.index.get_level_values(self.AGG_BY).unique()
                else:
                    zone_input_adj = zone_input
                if zone_input_adj in zones_with_cap:
                    Total_Installed_Capacity = Total_Installed_Capacity.xs(zone_input_adj,level=self.AGG_BY)
                else:
                    self.logger.warning("No installed capacity in %s",zone_input)
                    outputs[zone_input] = mfunc.MissingZoneData()
                    continue

                #print(Total_Installed_Capacity.index.get_level_values('tech').unique())
                fn = os.path.join(self.Marmot_Solutions_folder, self.Scenario_name, 'Figures_Output',self.AGG_BY + '_total_installed_capacity','Individual_Gen_Cap_' + scenario + '.csv')  
                Total_Installed_Capacity.reset_index().to_csv(fn)

                Total_Installed_Capacity = mfunc.df_process_gen_inputs(Total_Installed_Capacity, self.ordered_gen)
                Total_Installed_Capacity.reset_index(drop=True, inplace=True)
                Total_Installed_Capacity.rename(index={0:scenario}, inplace=True)
                Total_Installed_Capacity_Out = pd.concat([Total_Installed_Capacity_Out, Total_Installed_Capacity], axis=0, sort=False).fillna(0)
                
            Total_Installed_Capacity_Out = Total_Installed_Capacity_Out.loc[:, (Total_Installed_Capacity_Out != 0).any(axis=0)]

            try:
                Total_Installed_Capacity_Out = Total_Installed_Capacity_Out-Total_Installed_Capacity_Out.xs(self.Scenarios[0]) #Change to a diff on first scenario
            except KeyError:
                out = mfunc.MissingZoneData()
                outputs[zone_input] = out
                continue
            Total_Installed_Capacity_Out.drop(self.Scenarios[0],inplace=True) #Drop base entry

            # If Total_Installed_Capacity_Out df is empty returns a empty dataframe and does not plot
            if Total_Installed_Capacity_Out.empty:
                self.logger.warning("No installed capacity in %s",zone_input)
                out = mfunc.MissingZoneData()
                outputs[zone_input] = out
                continue

            # Data table of values to return to main program
            Data_Table_Out = pd.concat([Data_Table_Out, Total_Installed_Capacity_Out],  axis=1, sort=False)
            
            unitconversion = mfunc.capacity_energy_unitconversion(max(Total_Installed_Capacity_Out.sum()))
            Total_Installed_Capacity_Out = Total_Installed_Capacity_Out/unitconversion['divisor'] 
            
            Total_Installed_Capacity_Out.index = Total_Installed_Capacity_Out.index.str.replace('_',' ')
            Total_Installed_Capacity_Out.index = Total_Installed_Capacity_Out.index.str.wrap(5, break_long_words=False)

			fig1, ax = plt.subplots(figsize=(self.x,self.y))
            Total_Installed_Capacity_Out.plot.bar(stacked=True, figsize=(self.x,self.y), rot=0,ax = ax,
                                 color=[self.PLEXOS_color_dict.get(x, '#333333') for x in Total_Installed_Capacity_Out.columns], edgecolor='black', linewidth='0.1')

            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_ylabel('Capacity Change ({}) \n relative to '.format(unitconversion['units']) + self.Multi_Scenario[0],  color='black', rotation='vertical')

            #adds comma to y axis data
            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(x, f',.{self.y_axes_decimalpt}f')))
            ax.tick_params(axis='y', which='major', length=5, width=1)
            ax.tick_params(axis='x', which='major', length=5, width=1)

            handles, labels = ax.get_legend_handles_labels()
            ax.legend(reversed(handles), reversed(labels), loc='lower left',bbox_to_anchor=(1,0),
                          facecolor='inherit', frameon=True)
            if mconfig.parser("plot_title_as_region"):
                ax.set_title(zone_input)
            outputs[zone_input] = {'fig': fig1, 'data_table': Data_Table_Out}
        return outputs


    def total_cap_and_gen_facet(self):
        # generation figure
        self.logger.info("Generation data")
        gen_obj = gen.mplot(self.argument_dict)
        gen_outputs = gen_obj.total_gen()

        self.logger.info("Installed capacity data")
        cap_outputs = self.total_cap()

        outputs = {}
        for zone_input in self.Zones:

            fig, axs = plt.subplots(1, 2, figsize=(10, 4))

            plt.subplots_adjust(wspace=0.35, hspace=0.2)
            axs = axs.ravel()

            # left panel: installed capacity
            try:
                Total_Installed_Capacity_Out = cap_outputs[zone_input]["data_table"]
            except TypeError:
                outputs[zone_input] = mfunc.MissingZoneData()
                continue
            
            Total_Installed_Capacity_Out.index = Total_Installed_Capacity_Out.index.str.replace('_',' ')
            Total_Installed_Capacity_Out.index = Total_Installed_Capacity_Out.index.str.wrap(5, break_long_words=False)
            
            unitconversion = mfunc.capacity_energy_unitconversion(max(Total_Installed_Capacity_Out.sum()))
            Total_Installed_Capacity_Out = Total_Installed_Capacity_Out/unitconversion['divisor'] 


            Total_Installed_Capacity_Out.plot.bar(stacked=True, rot=0, ax=axs[0],
                                 color=[self.PLEXOS_color_dict.get(x, '#333333') for x in Total_Installed_Capacity_Out.columns],
                                 edgecolor='black', linewidth='0.1')

            axs[0].spines['right'].set_visible(False)
            axs[0].spines['top'].set_visible(False)
            axs[0].set_ylabel('Total Installed Capacity ({})'.format(unitconversion['units']),  color='black', rotation='vertical')
            #adds comma to y axis data
            axs[0].yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(x, f',.{self.y_axes_decimalpt}f')))
            axs[0].tick_params(axis='y', which='major', length=5, width=1)
            axs[0].tick_params(axis='x', which='major', length=5, width=1)
            axs[0].get_legend().remove()

            # replace x-axis with custom labels
            if len(self.ticklabels) > 1:
                self.ticklabels = pd.Series(self.ticklabels).str.replace('-','- ').str.wrap(8, break_long_words=True)
                axs[0].set_xticklabels(self.ticklabels)

            # right panel: annual generation
            Total_Gen_Results = gen_outputs[zone_input]["data_table"]

            unitconversion = mfunc.capacity_energy_unitconversion(max(Total_Gen_Results.sum()))
            Total_Gen_Results = Total_Gen_Results/unitconversion['divisor'] 

            Total_Load_Out = Total_Gen_Results.loc[:, "Total Load (Demand + \n Storage Charging)"]
            Total_Demand_Out = Total_Gen_Results.loc[:, "Total Demand"]
            Unserved_Energy_Out = Total_Gen_Results.loc[:, "Unserved Energy"]
            Total_Generation_Stack_Out = Total_Gen_Results.drop(["Total Load (Demand + \n Storage Charging)", "Total Demand", "Unserved Energy"], axis=1)
            Pump_Load_Out = Total_Load_Out - Total_Demand_Out

            Total_Generation_Stack_Out.index = Total_Generation_Stack_Out.index.str.replace('_',' ')
            Total_Generation_Stack_Out.index = Total_Generation_Stack_Out.index.str.wrap(5, break_long_words=False)
            
            Total_Generation_Stack_Out.plot.bar(stacked=True, rot=0, ax=axs[1],
                             color=[self.PLEXOS_color_dict.get(x, '#333333') for x in Total_Generation_Stack_Out.columns], edgecolor='black', linewidth='0.1')

            axs[1].spines['right'].set_visible(False)
            axs[1].spines['top'].set_visible(False)
            axs[1].set_ylabel('Total Generation ({}h)'.format(unitconversion['units']),  color='black', rotation='vertical')
            #adds comma to y axis data
            axs[1].yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(x, f',.{self.y_axes_decimalpt}f')))
            axs[1].tick_params(axis='y', which='major', length=5, width=1)
            axs[1].tick_params(axis='x', which='major', length=5, width=1)

            n=0
            
            data_tables = {}
            if not self.facet:
                self.Scenarios = [self.Scenarios[0]]
            for scenario in self.Scenarios:

                x = [axs[1].patches[n].get_x(), axs[1].patches[n].get_x() + axs[1].patches[n].get_width()]
                height1 = [int(Total_Load_Out[scenario])]*2
                lp1 = plt.plot(x,height1, c='black', linewidth=1.5)
                height2 = [int(Total_Demand_Out[scenario])]*2
                lp2 = plt.plot(x,height2, 'r--', c='black', linewidth=1.5)
                if Unserved_Energy_Out[scenario].sum() > 0:
                    height3 = [int(Unserved_Energy_Out[scenario])]*2
                    plt.plot(x,height3, c='#DD0200', linewidth=1.5)
                    axs[1].fill_between(x, height3, height1,
                                facecolor = '#DD0200',
                                alpha=0.5)
                    
                data_tables[scenario] = pd.DataFrame() 
                n=n+1

            # replace x-axis with custom labels
            if len(self.ticklabels) > 1:
                self.ticklabels = pd.Series(self.ticklabels).str.replace('-','- ').str.wrap(8, break_long_words=True)
                axs[1].set_xticklabels(self.ticklabels)


            # get names of generator to create custom legend
            l1 = Total_Installed_Capacity_Out.columns.tolist()
            l2 = Total_Generation_Stack_Out.columns.tolist()
            l1.extend(l2)

            labels = np.unique(np.array(l1)).tolist()
            labels.sort(key = lambda i:self.ordered_gen.index(i))

            # create custom gen_tech legend
            handles = []
            for tech in labels:
                gen_tech_legend = Patch(facecolor=self.PLEXOS_color_dict[tech],
                            alpha=1.0)
                handles.append(gen_tech_legend)


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
            
            axs[1].legend(reversed(handles),reversed(labels),
                                    loc = 'lower left',bbox_to_anchor=(1.05,0),
                                    facecolor='inherit', frameon=True)
            

            # add labels to panels
            axs[0].set_title("A.", fontdict={"weight":"bold"}, loc='left')
            axs[1].set_title("B.", fontdict={"weight":"bold"}, loc='left')

            if mconfig.parser("plot_title_as_region"):
                fig.set_title(zone_input)

            # output figure
            outputs[zone_input] = {'fig': fig, 'data_table': data_tables}


        return outputs
