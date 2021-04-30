# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 16:24:40 2019

@author: dlevie
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import marmot.plottingmodules.marmot_plot_functions as mfunc
import logging
import marmot.config.mconfig as mconfig

#===============================================================================

class mplot(object):
    def __init__(self, argument_dict):
        # iterate over items in argument_dict and set as properties of class
        # see key_list in Marmot_plot_main for list of properties
        for prop in argument_dict:
            self.__setattr__(prop, argument_dict[prop])
        self.logger = logging.getLogger('marmot_plot.'+__name__)
        
        self.x = mconfig.parser("figure_size","xdimension")
        self.y = mconfig.parser("figure_size","ydimension")
        self.y_axes_decimalpt = mconfig.parser("axes_options","y_axes_decimalpt")
        
        self.mplot_data_dict = {}

    def prod_cost(self, figure_name=None, prop=None, start=None, end=None, 
                  timezone=None, start_date_range=None, end_date_range=None):
        
        outputs = {}
        
        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True,"generator_Total_Generation_Cost",self.Scenarios),
                      (True,"generator_Pool_Revenue",self.Scenarios),
                      (True,"generator_Reserves_Revenue",self.Scenarios),
                      (True,"generator_Installed_Capacity",self.Scenarios)]
        
        # Runs get_data to populate mplot_data_dict with all required properties, returns a 1 if required data is missing
        check_input_data = mfunc.get_data(self.mplot_data_dict, properties,self.Marmot_Solutions_folder)
        
        # Checks if all data required by plot is available, if 1 in list required data is missing
        if 1 in check_input_data:
            return mfunc.MissingInputData()
        
        for zone_input in self.Zones:
            total_cost_chunk = []
            self.logger.info(f"{self.AGG_BY} = {zone_input}")
            for scenario in self.Scenarios:
                self.logger.info(f"Scenario = {scenario}")
                Total_Systems_Cost = pd.DataFrame()

                Total_Installed_Capacity = self.mplot_data_dict["generator_Installed_Capacity"].get(scenario)
                #Check if zone has installed generation, if not skips
                try:
                    Total_Installed_Capacity = Total_Installed_Capacity.xs(zone_input,level=self.AGG_BY)
                except KeyError:
                    self.logger.warning(f"No installed capacity in : {zone_input}")
                    continue
                Total_Installed_Capacity = mfunc.df_process_gen_inputs(Total_Installed_Capacity, self.ordered_gen)
                Total_Installed_Capacity.reset_index(drop=True, inplace=True)
                Total_Installed_Capacity = Total_Installed_Capacity.iloc[0]

                Total_Gen_Cost = self.mplot_data_dict["generator_Total_Generation_Cost"].get(scenario)
                Total_Gen_Cost = Total_Gen_Cost.xs(zone_input,level=self.AGG_BY)
                Total_Gen_Cost = mfunc.df_process_gen_inputs(Total_Gen_Cost, self.ordered_gen)
                Total_Gen_Cost = Total_Gen_Cost.sum(axis=0)*-1
                Total_Gen_Cost = Total_Gen_Cost/Total_Installed_Capacity #Change to $/MW-year
                Total_Gen_Cost.rename("Total_Gen_Cost", inplace=True)

                Pool_Revenues = self.mplot_data_dict["generator_Pool_Revenue"].get(scenario)
                Pool_Revenues = Pool_Revenues.xs(zone_input,level=self.AGG_BY)
                Pool_Revenues = mfunc.df_process_gen_inputs(Pool_Revenues, self.ordered_gen)
                Pool_Revenues = Pool_Revenues.sum(axis=0)
                Pool_Revenues = Pool_Revenues/Total_Installed_Capacity #Change to $/MW-year
                Pool_Revenues.rename("Energy_Revenues", inplace=True)

                ### Might cvhnage to Net Reserve Revenue at later date
                Reserve_Revenues = self.mplot_data_dict["generator_Reserves_Revenue"].get(scenario)
                Reserve_Revenues = Reserve_Revenues.xs(zone_input,level=self.AGG_BY)
                Reserve_Revenues = mfunc.df_process_gen_inputs(Reserve_Revenues, self.ordered_gen)
                Reserve_Revenues = Reserve_Revenues.sum(axis=0)
                Reserve_Revenues = Reserve_Revenues/Total_Installed_Capacity #Change to $/MW-year
                Reserve_Revenues.rename("Reserve_Revenues", inplace=True)

                Total_Systems_Cost = pd.concat([Total_Systems_Cost, Total_Gen_Cost, Pool_Revenues, Reserve_Revenues], axis=1, sort=False)

                Total_Systems_Cost.columns = Total_Systems_Cost.columns.str.replace('_',' ')
                Total_Systems_Cost = Total_Systems_Cost.sum(axis=0)
                Total_Systems_Cost = Total_Systems_Cost.rename(scenario)
                
                total_cost_chunk.append(Total_Systems_Cost)

            Total_Systems_Cost_Out = pd.concat(total_cost_chunk, axis=1, sort=False)

            Total_Systems_Cost_Out = Total_Systems_Cost_Out.T
            Total_Systems_Cost_Out.index = Total_Systems_Cost_Out.index.str.replace('_',' ')
            Total_Systems_Cost_Out.index = Total_Systems_Cost_Out.index.str.wrap(10, break_long_words=False)

            Total_Systems_Cost_Out = Total_Systems_Cost_Out/1000 #Change to $/kW-year
            Net_Revenue = Total_Systems_Cost_Out.sum(axis=1)

            #Checks if Net_Revenue contains data, if not skips zone and does not return a plot
            if Net_Revenue.empty:
                out = mfunc.MissingZoneData()
                outputs[zone_input] = out
                continue

            # Data table of values to return to main program
            Data_Table_Out = Total_Systems_Cost_Out.add_suffix(" ($/KW-yr)")

            fig1, ax = plt.subplots(figsize=(self.x,self.y))

            net_rev = plt.plot(Net_Revenue.index, Net_Revenue.values, color='black', linestyle='None', marker='o')
            Total_Systems_Cost_Out.plot.bar(stacked=True, rot=0, edgecolor='black', linewidth='0.1', ax=ax)

            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_ylabel('Total System Net Rev, Rev, & Cost ($/KW-yr)',  color='black', rotation='vertical')
            # ax.set_xticklabels(rotation='vertical')
            ax.tick_params(axis='y', which='major', length=5, width=1)
            ax.tick_params(axis='x', which='major', length=5, width=1)
            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(x, f',.{self.y_axes_decimalpt}f')))
            ax.margins(x=0.01)
            plt.xticks(rotation=90)

            handles, labels = ax.get_legend_handles_labels()
            ax.legend(reversed(handles), reversed(labels), loc='upper center',bbox_to_anchor=(0.5,-0.15),
                         facecolor='inherit', frameon=True, ncol=3)

            #Legend 1
            leg1 = ax.legend(reversed(handles), reversed(labels), loc='lower left',bbox_to_anchor=(1,0),
                          facecolor='inherit', frameon=True)
            #Legend 2
            ax.legend(net_rev, ['Net Revenue'], loc='center left',bbox_to_anchor=(1, 0.9),
                          facecolor='inherit', frameon=True)

            # Manually add the first legend back
            ax.add_artist(leg1)
            if mconfig.parser("plot_title_as_region"):
                fig1.set_title(zone_input)

            outputs[zone_input] = {'fig': fig1, 'data_table': Data_Table_Out}
        return outputs


    def sys_cost(self, figure_name=None, prop=None, start=None, end=None, 
                  timezone=None, start_date_range=None, end_date_range=None):
        
        outputs = {}
        
        if self.AGG_BY == 'zone':
            agg = 'zone'
        else:
            agg = 'region'
            
        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True,"generator_Total_Generation_Cost",self.Scenarios),
                      (False,f"{agg}_Cost_Unserved_Energy",self.Scenarios)]
        
        # Runs get_data to populate mplot_data_dict with all required properties, returns a 1 if required data is missing
        check_input_data = mfunc.get_data(self.mplot_data_dict, properties,self.Marmot_Solutions_folder)
        
        # Checks if all data required by plot is available, if 1 in list required data is missing
        if 1 in check_input_data:
            return mfunc.MissingInputData()
        
        for zone_input in self.Zones:
            total_cost_chunk = []
            self.logger.info(f"{self.AGG_BY} = {zone_input}")

            for scenario in self.Scenarios:
                self.logger.info(f"Scenario = {scenario}")
                Total_Systems_Cost = pd.DataFrame()

                Total_Gen_Cost = self.mplot_data_dict["generator_Total_Generation_Cost"].get(scenario)

                try:
                    Total_Gen_Cost = Total_Gen_Cost.xs(zone_input,level=self.AGG_BY)
                except KeyError:
                    self.logger.warning(f"No Generators found for : {zone_input}")
                    continue

                Total_Gen_Cost = Total_Gen_Cost.sum(axis=0)
                Total_Gen_Cost.rename("Total_Gen_Cost", inplace=True)
                
                if self.mplot_data_dict[f"{agg}_Cost_Unserved_Energy"] == {}:
                    Cost_Unserved_Energy = self.mplot_data_dict["generator_Total_Generation_Cost"][scenario].copy()
                    Cost_Unserved_Energy.iloc[:,0] = 0
                else:
                    Cost_Unserved_Energy = self.mplot_data_dict[f"{agg}_Cost_Unserved_Energy"][scenario]
                Cost_Unserved_Energy = Cost_Unserved_Energy.xs(zone_input,level=self.AGG_BY)
                Cost_Unserved_Energy = Cost_Unserved_Energy.sum(axis=0)
                Cost_Unserved_Energy.rename("Cost_Unserved_Energy", inplace=True)

                Total_Systems_Cost = pd.concat([Total_Systems_Cost, Total_Gen_Cost, Cost_Unserved_Energy], axis=1, sort=False)

                Total_Systems_Cost.columns = Total_Systems_Cost.columns.str.replace('_',' ')
                Total_Systems_Cost.rename({0:scenario}, axis='index', inplace=True)
                
                total_cost_chunk.append(Total_Systems_Cost)
            
            # Checks if gen_cost_out_chunks contains data, if not skips zone and does not return a plot
            if not total_cost_chunk:
                outputs[zone_input] = mfunc.MissingZoneData()
                continue
            
            Total_Systems_Cost_Out = pd.concat(total_cost_chunk, axis=0, sort=False)
            Total_Systems_Cost_Out = Total_Systems_Cost_Out/1000000 #Convert cost to millions

            Total_Systems_Cost_Out.index = Total_Systems_Cost_Out.index.str.replace('_',' ')
            Total_Systems_Cost_Out.index = Total_Systems_Cost_Out.index.str.wrap(5, break_long_words=False)

             #Checks if Total_Systems_Cost_Out contains data, if not skips zone and does not return a plot
            if Total_Systems_Cost_Out.empty:
                outputs[zone_input] = mfunc.MissingZoneData()
                continue

            # Data table of values to return to main program
            Data_Table_Out = Total_Systems_Cost_Out.add_suffix(" (Million $)")

            fig2, ax = plt.subplots(figsize=(self.x,self.y))

            Total_Systems_Cost_Out.plot.bar(stacked=True, rot=0, edgecolor='black', linewidth='0.1', ax=ax)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_ylabel('Total System Cost (Million $)',  color='black', rotation='vertical')
            ax.tick_params(axis='y', which='major', length=5, width=1)
            ax.tick_params(axis='x', which='major', length=5, width=1)
            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(x, f',.{self.y_axes_decimalpt}f')))
            ax.margins(x=0.01)

            handles, labels = ax.get_legend_handles_labels()
            ax.legend(reversed(handles), reversed(labels), loc='lower left',bbox_to_anchor=(1,0),
                          facecolor='inherit', frameon=True)
            
            if mconfig.parser("plot_title_as_region"):
                ax.set_title(zone_input)

            cost_totals = Total_Systems_Cost_Out.sum(axis=1) #holds total of each bar

            #inserts values into bar stacks
            for patch in ax.patches:
               width, height = patch.get_width(), patch.get_height()
               if height<=1:
                   continue
               x, y = patch.get_xy()
               ax.text(x+width/2,
                    y+height/2,
                    '{:,.0f}'.format(height),
                    horizontalalignment='center',
                    verticalalignment='center', fontsize=12)

            #inserts total bar value above each bar
            for k, patch in enumerate(ax.patches):
                height = cost_totals[k]
                width = patch.get_width()
                x, y = patch.get_xy()
                ax.text(x+width/2,
                    y+height + 0.05*max(ax.get_ylim()),
                    '{:,.0f}'.format(height),
                    horizontalalignment='center',
                    verticalalignment='center', fontsize=15, color='red')
                if k>=len(cost_totals)-1:
                    break
            

            outputs[zone_input] = {'fig': fig2, 'data_table': Data_Table_Out}
        return outputs


    def detailed_gen_cost(self, figure_name=None, prop=None, start=None, end=None, 
                  timezone=None, start_date_range=None, end_date_range=None):
        
        outputs = {}
        
        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True,"generator_Total_Generation_Cost",self.Scenarios),
                      (True,"generator_Fuel_Cost",self.Scenarios),
                      (True,"generator_VO&M_Cost",self.Scenarios),
                      (True,"generator_Start_&_Shutdown_Cost",self.Scenarios),
                      (False,"generator_Emissions_Cost",self.Scenarios)]
        
        # Runs get_data to populate mplot_data_dict with all required properties, returns a 1 if required data is missing
        check_input_data = mfunc.get_data(self.mplot_data_dict, properties,self.Marmot_Solutions_folder)
        
        # Checks if all data required by plot is available, if 1 in list required data is missing
        if 1 in check_input_data:
            return mfunc.MissingInputData()
        
        for zone_input in self.Zones:
            self.logger.info(f"Zone = {zone_input}")
            gen_cost_out_chunks = []

            for scenario in self.Scenarios:
                self.logger.info(f"Scenario = {scenario}")

                Fuel_Cost = self.mplot_data_dict["generator_Fuel_Cost"].get(scenario)
                # Check if Fuel_cost contains zone_input, skips if not
                try:
                    Fuel_Cost = Fuel_Cost.xs(zone_input,level=self.AGG_BY)
                except KeyError:
                    self.logger.warning(f"No Generators found for: {zone_input}")
                    continue

                Fuel_Cost = Fuel_Cost.sum(axis=0)
                Fuel_Cost.rename("Fuel_Cost", inplace=True)

                VOM_Cost = self.mplot_data_dict["generator_VO&M_Cost"].get(scenario)
                VOM_Cost = VOM_Cost.xs(zone_input,level=self.AGG_BY)
                VOM_Cost[0].values[VOM_Cost[0].values < 0] = 0
                VOM_Cost = VOM_Cost.sum(axis=0)
                VOM_Cost.rename("VO&M_Cost", inplace=True)
                
                Start_Shutdown_Cost = self.mplot_data_dict["generator_Start_&_Shutdown_Cost"].get(scenario)
                Start_Shutdown_Cost = Start_Shutdown_Cost.xs(zone_input,level=self.AGG_BY)
                Start_Shutdown_Cost = Start_Shutdown_Cost.sum(axis=0)
                Start_Shutdown_Cost.rename("Start_&_Shutdown_Cost", inplace=True)
                
                if self.mplot_data_dict["generator_Emissions_Cost"] == {}:
                    self.logger.warning(f"generator_Emissions_Cost not included in {scenario} results, Emissions_Cost will not be included in plot")
                    Emissions_Cost = self.mplot_data_dict["generator_Start_&_Shutdown_Cost"][scenario].copy()
                    Emissions_Cost.iloc[:,0] = 0
                else:
                    Emissions_Cost = self.mplot_data_dict["generator_Emissions_Cost"][scenario]
                Emissions_Cost = Emissions_Cost.xs(zone_input,level=self.AGG_BY)
                Emissions_Cost = Emissions_Cost.sum(axis=0)
                Emissions_Cost.rename("Emissions_Cost", inplace=True)

                Detailed_Gen_Cost = pd.concat([Fuel_Cost, VOM_Cost, Start_Shutdown_Cost, Emissions_Cost], axis=1, sort=False)

                Detailed_Gen_Cost.columns = Detailed_Gen_Cost.columns.str.replace('_',' ')
                Detailed_Gen_Cost = Detailed_Gen_Cost.sum(axis=0)
                Detailed_Gen_Cost = Detailed_Gen_Cost.rename(scenario)
                
                gen_cost_out_chunks.append(Detailed_Gen_Cost)
            
            # Checks if gen_cost_out_chunks contains data, if not skips zone and does not return a plot
            if not gen_cost_out_chunks:
                outputs[zone_input] = mfunc.MissingZoneData()
                continue
            
            Detailed_Gen_Cost_Out = pd.concat(gen_cost_out_chunks, axis=1, sort=False)
            Detailed_Gen_Cost_Out = Detailed_Gen_Cost_Out.T/1000000 #Convert cost to millions
            
            Detailed_Gen_Cost_Out.index = Detailed_Gen_Cost_Out.index.str.replace('_',' ')
            Detailed_Gen_Cost_Out.index = Detailed_Gen_Cost_Out.index.str.wrap(5, break_long_words=False)
            # Deletes columns that are all 0
            Detailed_Gen_Cost_Out = Detailed_Gen_Cost_Out.loc[:, (Detailed_Gen_Cost_Out != 0).any(axis=0)]
            
            # Checks if Detailed_Gen_Cost_Out contains data, if not skips zone and does not return a plot
            if Detailed_Gen_Cost_Out.empty:
                outputs[zone_input] = mfunc.MissingZoneData()
                continue
            
            # Data table of values to return to main program
            Data_Table_Out = Detailed_Gen_Cost_Out.add_suffix(" (Million $)")

            fig3, ax = plt.subplots(figsize=(self.x,self.y))

            Detailed_Gen_Cost_Out.plot.bar(stacked=True, rot=0, edgecolor='black', linewidth='0.1', ax=ax)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.axhline(y = 0)
            ax.set_ylabel('Total Generation Cost (Million $)',  color='black', rotation='vertical')
            ax.tick_params(axis='y', which='major', length=5, width=1)
            ax.tick_params(axis='x', which='major', length=5, width=1)
            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(x, f',.{self.y_axes_decimalpt}f')))
            ax.margins(x=0.01)

            handles, labels = ax.get_legend_handles_labels()
            ax.legend(reversed(handles), reversed(labels), loc='lower left',bbox_to_anchor=(1,0),
                          facecolor='inherit', frameon=True)
            
            if mconfig.parser("plot_title_as_region"):
                ax.set_title(zone_input)
                
            cost_totals = Detailed_Gen_Cost_Out.sum(axis=1) #holds total of each bar

            #inserts values into bar stacks
            for patch in ax.patches:
                width, height = patch.get_width(), patch.get_height()
                if height<=2:
                   continue
                x, y = patch.get_xy()
                ax.text(x+width/2,
                    y+height/2,
                    '{:,.0f}'.format(height),
                    horizontalalignment='center',
                    verticalalignment='center', fontsize=12)

            #inserts total bar value above each bar
            for k, patch in enumerate(ax.patches):
                height = cost_totals[k]
                width = patch.get_width()
                x, y = patch.get_xy()
                ax.text(x+width/2,
                    y+height + 0.05*max(ax.get_ylim()),
                    '{:,.0f}'.format(height),
                    horizontalalignment='center',
                    verticalalignment='center', fontsize=15, color='red')
                if k>=len(cost_totals)-1:
                    break
            
            outputs[zone_input] = {'fig': fig3, 'data_table': Data_Table_Out}
        return outputs


    def sys_cost_type(self, figure_name=None, prop=None, start=None, end=None, 
                  timezone=None, start_date_range=None, end_date_range=None):
        
        # Create Dictionary to hold Datframes for each scenario
        outputs = {}
        
        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True,"generator_Total_Generation_Cost",self.Scenarios)]
        
        # Runs get_data to populate mplot_data_dict with all required properties, returns a 1 if required data is missing
        check_input_data = mfunc.get_data(self.mplot_data_dict, properties,self.Marmot_Solutions_folder)
        
        # Checks if all data required by plot is available, if 1 in list required data is missing
        if 1 in check_input_data:
            return mfunc.MissingInputData()
        
        for zone_input in self.Zones:
            gen_cost_out_chunks = []
            self.logger.info(f"Zone = {zone_input}")

            for scenario in self.Scenarios:
                self.logger.info(f"Scenario = {scenario}")

                Total_Gen_Stack = self.mplot_data_dict["generator_Total_Generation_Cost"].get(scenario)
                # Check if Total_Gen_Stack contains zone_input, skips if not
                try:
                    Total_Gen_Stack = Total_Gen_Stack.xs(zone_input,level=self.AGG_BY)
                except KeyError:
                    self.logger.warning(f"No Generators found for : {zone_input}")
                    continue
                Total_Gen_Stack = mfunc.df_process_gen_inputs(Total_Gen_Stack, self.ordered_gen)

                Total_Gen_Stack = Total_Gen_Stack.sum(axis=0)
                Total_Gen_Stack.rename(scenario, inplace=True)
                gen_cost_out_chunks.append(Total_Gen_Stack)
            
            # Checks if gen_cost_out_chunks contains data, if not skips zone and does not return a plot
            if not gen_cost_out_chunks:
                outputs[zone_input] = mfunc.MissingZoneData()
                continue
            
            Total_Generation_Stack_Out = pd.concat(gen_cost_out_chunks, axis=1, sort=False).fillna(0)
            Total_Generation_Stack_Out = mfunc.df_process_categorical_index(Total_Generation_Stack_Out, self.ordered_gen)
            Total_Generation_Stack_Out = Total_Generation_Stack_Out.T/1000000 #Convert to millions
            Total_Generation_Stack_Out = Total_Generation_Stack_Out.loc[:, (Total_Generation_Stack_Out != 0).any(axis=0)]

            # Checks if Total_Generation_Stack_Out contains data, if not skips zone and does not return a plot
            if Total_Generation_Stack_Out.empty:
                outputs[zone_input] = mfunc.MissingZoneData()
                continue

            # Data table of values to return to main program
            Data_Table_Out = Total_Generation_Stack_Out.add_suffix(" (Million $)")

            Total_Generation_Stack_Out.index = Total_Generation_Stack_Out.index.str.replace('_',' ')
            Total_Generation_Stack_Out.index = Total_Generation_Stack_Out.index.str.wrap(10, break_long_words=False)

            fig1, ax = plt.subplots(figsize=(self.x,self.y))

            Total_Generation_Stack_Out.plot.bar(stacked=True, figsize=(self.x,self.y), rot=0,
                             color=[self.PLEXOS_color_dict.get(x, '#333333') for x in Total_Generation_Stack_Out.columns], edgecolor='black', linewidth='0.1',ax=ax)

            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

            ax.set_ylabel('Total System Cost (Million $)',  color='black', rotation='vertical')
            ax.tick_params(axis='y', which='major', length=5, width=1)
            ax.tick_params(axis='x', which='major', length=5, width=1)
            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(x, f',.{self.y_axes_decimalpt}f')))

            ax.margins(x=0.01)

            handles, labels = ax.get_legend_handles_labels()
            ax.legend(reversed(handles), reversed(labels), loc='lower left',bbox_to_anchor=(1,0),
                          facecolor='inherit', frameon=True)

            if mconfig.parser("plot_title_as_region"):
                ax.set_title(zone_input)

            outputs[zone_input] = {'fig': fig1, 'data_table': Data_Table_Out}
        return outputs


    def sys_cost_diff(self, figure_name=None, prop=None, start=None, end=None, 
                  timezone=None, start_date_range=None, end_date_range=None):
        
        if self.AGG_BY == 'zone':
            agg = 'zone'
        else:
            agg = 'region'
            
        outputs = {}
        
        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True, "generator_Total_Generation_Cost", self.Scenarios),
                      (False, f"{agg}_Cost_Unserved_Energy", self.Scenarios)]
        
        # Runs get_data to populate mplot_data_dict with all required properties, returns a 1 if required data is missing
        check_input_data = mfunc.get_data(self.mplot_data_dict, properties,self.Marmot_Solutions_folder)
        
        # Checks if all data required by plot is available, if 1 in list required data is missing
        if 1 in check_input_data:
            return mfunc.MissingInputData()
        
        for zone_input in self.Zones:
            total_cost_chunk = []
            self.logger.info(f"Zone = {zone_input}")

            for scenario in self.Scenarios:
                self.logger.info(f"Scenario = {scenario}")
                Total_Systems_Cost = pd.DataFrame()
                
                Total_Gen_Cost = self.mplot_data_dict["generator_Total_Generation_Cost"].get(scenario)

                try:
                    Total_Gen_Cost = Total_Gen_Cost.xs(zone_input,level=self.AGG_BY)
                except KeyError:
                    self.logger.warning(f"No Generators found for : {zone_input}")
                    continue

                Total_Gen_Cost = Total_Gen_Cost.sum(axis=0)
                Total_Gen_Cost.rename("Total_Gen_Cost", inplace=True)
                
                if self.mplot_data_dict[f"{agg}_Cost_Unserved_Energy"] == {}:
                    Cost_Unserved_Energy = self.mplot_data_dict["generator_Total_Generation_Cost"][scenario].copy()
                    Cost_Unserved_Energy.iloc[:,0] = 0
                else:
                    Cost_Unserved_Energy = self.mplot_data_dict[f"{agg}_Cost_Unserved_Energy"][scenario]
                Cost_Unserved_Energy = Cost_Unserved_Energy.xs(zone_input,level=self.AGG_BY)
                Cost_Unserved_Energy = Cost_Unserved_Energy.sum(axis=0)
                Cost_Unserved_Energy.rename("Cost_Unserved_Energy", inplace=True)

                Total_Systems_Cost = pd.concat([Total_Systems_Cost, Total_Gen_Cost, Cost_Unserved_Energy], axis=1, sort=False)

                Total_Systems_Cost.columns = Total_Systems_Cost.columns.str.replace('_',' ')
                Total_Systems_Cost.rename({0:scenario}, axis='index', inplace=True)
                total_cost_chunk.append(Total_Systems_Cost)
            
            # Checks if total_cost_chunk contains data, if not skips zone and does not return a plot
            if not total_cost_chunk:
                outputs[zone_input] = mfunc.MissingZoneData()
                continue
            
            Total_Systems_Cost_Out = pd.concat(total_cost_chunk, axis=0, sort=False)
            Total_Systems_Cost_Out = Total_Systems_Cost_Out/1000000 #Convert cost to millions
            #Ensures region has generation, else skips
            try:
                Total_Systems_Cost_Out = Total_Systems_Cost_Out-Total_Systems_Cost_Out.xs(self.Scenarios[0]) #Change to a diff on first scenario
            except KeyError:
                outputs[zone_input] = mfunc.MissingZoneData()
                continue
            Total_Systems_Cost_Out.drop(self.Scenarios[0],inplace=True) #Drop base entry

            # Checks if Total_Systems_Cost_Out contains data, if not skips zone and does not return a plot
            if Total_Systems_Cost_Out.empty:
                outputs[zone_input] = mfunc.MissingZoneData()
                continue
            
            # Data table of values to return to main program
            Data_Table_Out = Total_Systems_Cost_Out
            Data_Table_Out = Data_Table_Out.add_suffix(" (Million $)")

            fig2, ax = plt.subplots(figsize=(self.x,self.y))

            Total_Systems_Cost_Out.plot.bar(stacked=True, rot=0, edgecolor='black', linewidth='0.1', ax=ax)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

            ax.tick_params(axis='y', which='major', length=5, width=1)
            ax.tick_params(axis='x', which='major', length=5, width=1)
            locs,labels=plt.xticks()
            ax.axhline(y = 0, color = 'black')
            ax.set_ylabel('Generation Cost Change (Million $) \n relative to '+ self.Scenarios[0],  color='black', rotation='vertical')
            xlabels = pd.Series(self.Scenarios).str.replace('_',' ').str.wrap(10, break_long_words=False)
            plt.xticks(ticks=locs,labels=xlabels[1:])

            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(x, f',.{self.y_axes_decimalpt}f')))
            ax.margins(x=0.01)
            # plt.ylim((0,600))
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(reversed(handles), reversed(labels), loc='lower left',bbox_to_anchor=(1,0),
                          facecolor='inherit', frameon=True)

            if mconfig.parser("plot_title_as_region"):
                ax.set_title(zone_input)

            outputs[zone_input] = {'fig': fig2, 'data_table': Data_Table_Out}
        return outputs


    def sys_cost_type_diff(self, figure_name=None, prop=None, start=None, end=None, 
                  timezone=None, start_date_range=None, end_date_range=None):
        
        # Create Dictionary to hold Datframes for each scenario
        outputs = {}
        
        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True, "generator_Total_Generation_Cost" ,self.Scenarios)]
        
        # Runs get_data to populate mplot_data_dict with all required properties, returns a 1 if required data is missing
        check_input_data = mfunc.get_data(self.mplot_data_dict, properties,self.Marmot_Solutions_folder)
        
        # Checks if all data required by plot is available, if 1 in list required data is missing
        if 1 in check_input_data:
            return mfunc.MissingInputData()
        
        for zone_input in self.Zones:
            gen_cost_out_chunks = []
            self.logger.info(f"Zone = {zone_input}")

            for scenario in self.Scenarios:
                self.logger.info(f"Scenario = {scenario}")

                Total_Gen_Stack = self.mplot_data_dict["generator_Total_Generation_Cost"].get(scenario)

                try:
                    Total_Gen_Stack = Total_Gen_Stack.xs(zone_input,level=self.AGG_BY)
                except KeyError:
                    self.logger.warning(f"No Generators found for : {zone_input}")
                    continue
                
                Total_Gen_Stack = mfunc.df_process_gen_inputs(Total_Gen_Stack, self.ordered_gen)
                Total_Gen_Stack = Total_Gen_Stack.sum(axis=0)
                Total_Gen_Stack.rename(scenario, inplace=True)
                gen_cost_out_chunks.append(Total_Gen_Stack)
            
            # Checks if gen_cost_out_chunks contains data, if not skips zone and does not return a plot
            if not gen_cost_out_chunks:
                outputs[zone_input] = mfunc.MissingZoneData()
                continue
            
            Total_Generation_Stack_Out = pd.concat(gen_cost_out_chunks, axis=1, sort=False).fillna(0)
            Total_Generation_Stack_Out = mfunc.df_process_categorical_index(Total_Generation_Stack_Out, self.ordered_gen)
            Total_Generation_Stack_Out = Total_Generation_Stack_Out.T/1000000 #Convert to millions
            Total_Generation_Stack_Out = Total_Generation_Stack_Out.loc[:, (Total_Generation_Stack_Out != 0).any(axis=0)]
            #Ensures region has generation, else skips
            try:
                Total_Generation_Stack_Out = Total_Generation_Stack_Out-Total_Generation_Stack_Out.xs(self.Scenarios[0]) #Change to a diff on first scenario
            except KeyError:
                outputs[zone_input] = mfunc.MissingZoneData()
                continue
            Total_Generation_Stack_Out.drop(self.Scenarios[0],inplace=True) #Drop base entry

            # Checks if Total_Generation_Stack_Out contains data, if not skips zone and does not return a plot
            if Total_Generation_Stack_Out.empty == True:
                outputs[zone_input] = mfunc.MissingZoneData()
                continue

            # Data table of values to return to main program
            Data_Table_Out = Total_Generation_Stack_Out.add_suffix(" (Million $)")

            Total_Generation_Stack_Out.index = Total_Generation_Stack_Out.index.str.replace('_',' ')
            Total_Generation_Stack_Out.index = Total_Generation_Stack_Out.index.str.wrap(10, break_long_words=False)

            fig1, ax = plt.subplots(figsize=(self.x,self.y))

            Total_Generation_Stack_Out.plot.bar(stacked=True, figsize=(9,6), rot=0,
                             color=[self.PLEXOS_color_dict.get(x, '#333333') for x in Total_Generation_Stack_Out.columns], edgecolor='black', linewidth='0.1',ax=ax)

            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

            ax.tick_params(axis='y', which='major', length=5, width=1)
            ax.tick_params(axis='x', which='major', length=5, width=1)
            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(x, f',.{self.y_axes_decimalpt}f')))

            locs,labels=plt.xticks()
            ax.axhline(y = 0)
            ax.set_ylabel('Generation Cost Change (Million $) \n relative to '+ self.Scenarios[0],  color='black', rotation='vertical')
            xlabels = pd.Series(self.Scenarios).str.replace('_',' ').str.wrap(10, break_long_words=False)
            plt.xticks(ticks=locs,labels=xlabels[1:])

            ax.margins(x=0.01)
            # plt.ylim((0,600))

            handles, labels = ax.get_legend_handles_labels()

            ax.legend(reversed(handles), reversed(labels), loc='lower left',bbox_to_anchor=(1,0),
                          facecolor='inherit', frameon=True)
            
            if mconfig.parser("plot_title_as_region"):
                ax.set_title(zone_input)

            outputs[zone_input] = {'fig': fig1, 'data_table': Data_Table_Out}
        return outputs


    def detailed_gen_cost_diff(self, figure_name=None, prop=None, start=None, end=None, 
                  timezone=None, start_date_range=None, end_date_range=None):
        
        outputs = {}
        
        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True,"generator_Total_Generation_Cost",self.Scenarios),
                      (True,"generator_Fuel_Cost",self.Scenarios),
                      (True,"generator_VO&M_Cost",self.Scenarios),
                      (True,"generator_Start_&_Shutdown_Cost",self.Scenarios),
                      (False,"generator_Emissions_Cost",self.Scenarios)]
        
        # Runs get_data to populate mplot_data_dict with all required properties, returns a 1 if required data is missing
        check_input_data = mfunc.get_data(self.mplot_data_dict, properties,self.Marmot_Solutions_folder)
        
        # Checks if all data required by plot is available, if 1 in list required data is missing
        if 1 in check_input_data:
            return mfunc.MissingInputData()
        
        for zone_input in self.Zones:
            self.logger.info(f"Zone = {zone_input}")
            gen_cost_out_chunks = []

            for scenario in self.Scenarios:
                self.logger.info(f"Scenario = {scenario}")

                Fuel_Cost = self.mplot_data_dict["generator_Fuel_Cost"].get(scenario)
                try:
                    Fuel_Cost = Fuel_Cost.xs(zone_input,level=self.AGG_BY)
                except KeyError:
                    self.logger.warning(f"No Generators found for : {zone_input}")
                    continue
                Fuel_Cost = Fuel_Cost.sum(axis=0)
                Fuel_Cost.rename("Fuel_Cost", inplace=True)

                VOM_Cost = self.mplot_data_dict["generator_VO&M_Cost"].get(scenario)
                VOM_Cost = VOM_Cost.xs(zone_input,level=self.AGG_BY)
                VOM_Cost = VOM_Cost.sum(axis=0)
                VOM_Cost.rename("VO&M_Cost", inplace=True)

                Start_Shutdown_Cost = self.mplot_data_dict["generator_Start_&_Shutdown_Cost"].get(scenario)
                Start_Shutdown_Cost = Start_Shutdown_Cost.xs(zone_input,level=self.AGG_BY)
                Start_Shutdown_Cost = Start_Shutdown_Cost.sum(axis=0)
                Start_Shutdown_Cost.rename("Start_&_Shutdown_Cost", inplace=True)
                
                if self.mplot_data_dict["generator_Emissions_Cost"] == {}:
                    self.logger.warning(f"generator_Emissions_Cost not included in {scenario} results, Emissions_Cost will not be included in plot")
                    Emissions_Cost = self.mplot_data_dict["generator_Start_&_Shutdown_Cost"][scenario].copy()
                    Emissions_Cost.iloc[:,0] = 0
                else:
                    Emissions_Cost = self.mplot_data_dict["generator_Emissions_Cost"][scenario]
                Emissions_Cost = Emissions_Cost.xs(zone_input,level=self.AGG_BY)
                Emissions_Cost = Emissions_Cost.sum(axis=0)
                Emissions_Cost.rename("Emissions_Cost", inplace=True)

                Detailed_Gen_Cost = pd.concat([Fuel_Cost, VOM_Cost, Start_Shutdown_Cost, Emissions_Cost], axis=1, sort=False)

                Detailed_Gen_Cost.columns = Detailed_Gen_Cost.columns.str.replace('_',' ')
                Detailed_Gen_Cost = Detailed_Gen_Cost.sum(axis=0)
                Detailed_Gen_Cost = Detailed_Gen_Cost.rename(scenario)

                gen_cost_out_chunks.append(Detailed_Gen_Cost)
            
            # Checks if gen_cost_out_chunks contains data, if not skips zone and does not return a plot
            if not gen_cost_out_chunks:
                outputs[zone_input] = mfunc.MissingZoneData()
                continue
            
            Detailed_Gen_Cost_Out = pd.concat(gen_cost_out_chunks, axis=1, sort=False)
            Detailed_Gen_Cost_Out = Detailed_Gen_Cost_Out.T/1000000 #Convert cost to millions
            #TODO: Add $ unit conversion.

            #Ensures region has generation, else skips
            try:
                Detailed_Gen_Cost_Out = Detailed_Gen_Cost_Out-Detailed_Gen_Cost_Out.xs(self.Scenarios[0]) #Change to a diff on first scenario

            except KeyError:
                outputs[zone_input] = mfunc.MissingZoneData()
                continue

            Detailed_Gen_Cost_Out.drop(self.Scenarios[0],inplace=True) #Drop base entry

            net_cost = Detailed_Gen_Cost_Out.sum(axis = 1)

            Detailed_Gen_Cost_Out.index = Detailed_Gen_Cost_Out.index.str.replace('_',' ')
            Detailed_Gen_Cost_Out.index = Detailed_Gen_Cost_Out.index.str.wrap(10, break_long_words=False)
            # Deletes columns that are all 0
            Detailed_Gen_Cost_Out = Detailed_Gen_Cost_Out.loc[:, (Detailed_Gen_Cost_Out != 0).any(axis=0)]

            # Checks if Detailed_Gen_Cost_Out contains data, if not skips zone and does not return a plot
            if Detailed_Gen_Cost_Out.empty == True:
                outputs[zone_input] = mfunc.MissingZoneData()
                continue

            # Data table of values to return to main program
            Data_Table_Out = Detailed_Gen_Cost_Out.add_suffix(" (Million $)")

            fig3, ax = plt.subplots(figsize=(self.x,self.y))

            Detailed_Gen_Cost_Out.plot.bar(stacked=True, rot=0, edgecolor='black', linewidth='0.1', ax=ax)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.axhline(y= 0 ,linewidth=0.5,linestyle='--',color='grey')
            ax.axhline(y = 65.4, linewidth = 1, linestyle = ':',color = 'orange',label = 'Avg 2032 LCOE')
            ax.tick_params(axis='y', which='major', length=5, width=1)
            ax.tick_params(axis='x', which='major', length=5, width=1)
            locs,labels=plt.xticks()
            ax.set_ylabel('Generation Cost Change \n relative to '+ self.Scenarios[0] + ' (Million $)',  color='black', rotation='vertical') #TODO: Add $ unit conversion.
            xlabels = pd.Series(self.Scenarios).str.replace('_',' ').str.wrap(10, break_long_words=False)
            plt.xticks(ticks=locs,labels=xlabels[1:])

            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(x, f',.{self.y_axes_decimalpt}f')))
            ax.margins(x=0.01)

            #Add net cost line.
            for n, scenario in enumerate(self.Scenarios[1:]):
                x = [ax.patches[n].get_x(), ax.patches[n].get_x() + ax.patches[n].get_width()]
                y_net = [net_cost.loc[scenario]] * 2
                net_line = plt.plot(x,y_net, c='black', linewidth=1.5)

            handles, labels = ax.get_legend_handles_labels()

            #Main Legend
            leg_main = ax.legend(reversed(handles), reversed(labels), loc='lower left',bbox_to_anchor=(1,0),
                          facecolor='inherit', frameon=True)

            #Net cost legend
            leg_net = ax.legend(net_line,['Net Cost Change'],loc='center left',bbox_to_anchor=(1, -0.05),facecolor='inherit', frameon=True)
            ax.add_artist(leg_main)
            ax.add_artist(leg_net)

            if mconfig.parser("plot_title_as_region"):
                ax.set_title(zone_input)

            outputs[zone_input] = {'fig': fig3, 'data_table': Data_Table_Out}
        return outputs
