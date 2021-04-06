# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 16:24:40 2019

@author: dlevie
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
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
        
    def prod_cost(self):
        
        outputs = {}
        total_gen_cost_collection = {}
        pool_revenues_collection = {}
        reserve_revenues_collection = {}
        installed_capacity_collection = {}
        check_input_data = []
        
        check_input_data.extend([mfunc.get_data(total_gen_cost_collection,"generator_Total_Generation_Cost", self.Marmot_Solutions_folder, self.Scenarios)])
        check_input_data.extend([mfunc.get_data(pool_revenues_collection,"generator_Pool_Revenue", self.Marmot_Solutions_folder, self.Scenarios)])
        check_input_data.extend([mfunc.get_data(reserve_revenues_collection,"generator_Reserves_Revenue", self.Marmot_Solutions_folder, self.Scenarios)])
        check_input_data.extend([mfunc.get_data(installed_capacity_collection,"generator_Installed_Capacity", self.Marmot_Solutions_folder, self.Scenarios)])
        
        # Checks if all data required by plot is available, if 1 in list required data is missing
        if 1 in check_input_data:
            outputs = mfunc.MissingInputData()
            return outputs
        
        for zone_input in self.Zones:
            Total_Systems_Cost_Out = pd.DataFrame()
            self.logger.info(self.AGG_BY + " = "+ zone_input)
            for scenario in self.Scenarios:
                self.logger.info("Scenario = " + scenario)
                Total_Systems_Cost = pd.DataFrame()

                Total_Installed_Capacity = installed_capacity_collection.get(scenario)
                #Check if zone has installed generation, if not skips
                try:
                    Total_Installed_Capacity = Total_Installed_Capacity.xs(zone_input,level=self.AGG_BY)
                except KeyError:
                    self.logger.warning("No installed capacity in : "+zone_input)
                    continue
                Total_Installed_Capacity = mfunc.df_process_gen_inputs(Total_Installed_Capacity, self.ordered_gen)
                Total_Installed_Capacity.reset_index(drop=True, inplace=True)
                Total_Installed_Capacity = Total_Installed_Capacity.iloc[0]

                Total_Gen_Cost = total_gen_cost_collection.get(scenario)
                Total_Gen_Cost = Total_Gen_Cost.xs(zone_input,level=self.AGG_BY)
                Total_Gen_Cost = mfunc.df_process_gen_inputs(Total_Gen_Cost, self.ordered_gen)
                Total_Gen_Cost = Total_Gen_Cost.sum(axis=0)*-1
                Total_Gen_Cost = Total_Gen_Cost/Total_Installed_Capacity #Change to $/kW-year
                Total_Gen_Cost.rename("Total_Gen_Cost", inplace=True)

                Pool_Revenues = pool_revenues_collection.get(scenario)
                Pool_Revenues = Pool_Revenues.xs(zone_input,level=self.AGG_BY)
                Pool_Revenues = mfunc.df_process_gen_inputs(Pool_Revenues, self.ordered_gen)
                Pool_Revenues = Pool_Revenues.sum(axis=0)
                Pool_Revenues = Pool_Revenues/Total_Installed_Capacity #Change to $/kW-year
                Pool_Revenues.rename("Energy_Revenues", inplace=True)

                ### Might cvhnage to Net Reserve Revenue at later date
                Reserve_Revenues = reserve_revenues_collection.get(scenario)
                Reserve_Revenues = Reserve_Revenues.xs(zone_input,level=self.AGG_BY)
                Reserve_Revenues = mfunc.df_process_gen_inputs(Reserve_Revenues, self.ordered_gen)
                Reserve_Revenues = Reserve_Revenues.sum(axis=0)
                Reserve_Revenues = Reserve_Revenues/Total_Installed_Capacity #Change to $/kW-year
                Reserve_Revenues.rename("Reserve_Revenues", inplace=True)

                Total_Systems_Cost = pd.concat([Total_Systems_Cost, Total_Gen_Cost, Pool_Revenues, Reserve_Revenues], axis=1, sort=False)

                Total_Systems_Cost.columns = Total_Systems_Cost.columns.str.replace('_',' ')
                Total_Systems_Cost = Total_Systems_Cost.sum(axis=0)
                Total_Systems_Cost = Total_Systems_Cost.rename(scenario)

                Total_Systems_Cost_Out = pd.concat([Total_Systems_Cost_Out, Total_Systems_Cost], axis=1, sort=False)

            Total_Systems_Cost_Out = Total_Systems_Cost_Out.T
            Total_Systems_Cost_Out.index = Total_Systems_Cost_Out.index.str.replace('_',' ')
            Total_Systems_Cost_Out.index = Total_Systems_Cost_Out.index.str.wrap(10, break_long_words=False)

            Total_Systems_Cost_Out = Total_Systems_Cost_Out/1000
            Net_Revenue = Total_Systems_Cost_Out.sum(axis=1)

            #Checks if Net_Revenue contains data, if not skips zone and does not return a plot
            if Net_Revenue.empty:
                out = mfunc.MissingZoneData()
                outputs[zone_input] = out
                continue

            # Data table of values to return to main program
            Data_Table_Out = Total_Systems_Cost_Out

            # names = list(Net_Revenue.index)
            # values = list(Net_Revenue.values)

            fig1, ax = plt.subplots(figsize=(self.x,self.y))

            net_rev = plt.plot(Net_Revenue.index, Net_Revenue.values, color='black', linestyle='None', marker='o')
            sb = Total_Systems_Cost_Out.plot.bar(stacked=True, rot=0, edgecolor='black', linewidth='0.1', ax=ax)


            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_ylabel('Total System Net Rev, Rev, & Cost ($/KW-yr)',  color='black', rotation='vertical')
        #    ax.set_xticklabels(rotation='vertical')
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

            outputs[zone_input] = {'fig': fig1, 'data_table': Data_Table_Out}
        return outputs

    def sys_cost(self):
        outputs = {}
        total_gen_cost_collection = {}
        cost_unserved_energy_collection = {}
        check_input_data = []
        
        check_input_data.extend([mfunc.get_data(total_gen_cost_collection,"generator_Total_Generation_Cost", self.Marmot_Solutions_folder, self.Scenarios)])
        if self.AGG_BY == "zone":
            mfunc.get_data(cost_unserved_energy_collection,"zone_Cost_Unserved_Energy", self.Marmot_Solutions_folder, self.Scenarios)
        else:
            mfunc.get_data(cost_unserved_energy_collection,"region_Cost_Unserved_Energy", self.Marmot_Solutions_folder, self.Scenarios)
        
        # Checks if all data required by plot is available, if 1 in list required data is missing
        if 1 in check_input_data:
            outputs = mfunc.MissingInputData()
            return outputs
        
        for zone_input in self.Zones:
            Total_Systems_Cost_Out = pd.DataFrame()
            self.logger.info(self.AGG_BY + " = "+ zone_input)

            for scenario in self.Scenarios:
                self.logger.info("Scenario = " + scenario)
                Total_Systems_Cost = pd.DataFrame()


                Total_Gen_Cost = total_gen_cost_collection.get(scenario)

                try:
                    Total_Gen_Cost = Total_Gen_Cost.xs(zone_input,level=self.AGG_BY)
                except KeyError:
                    self.logger.warning("No Generators found for : "+zone_input)
                    continue

                Total_Gen_Cost = Total_Gen_Cost.sum(axis=0)
                Total_Gen_Cost.rename("Total_Gen_Cost", inplace=True)
                
                try:
                    cost_unserved_energy_collection[scenario]
                except KeyError:
                    cost_unserved_energy_collection[scenario] = total_gen_cost_collection[scenario].copy()
                    cost_unserved_energy_collection[scenario].iloc[:,0] = 0
                Cost_Unserved_Energy = cost_unserved_energy_collection.get(scenario)
                Cost_Unserved_Energy = Cost_Unserved_Energy.xs(zone_input,level=self.AGG_BY)
                Cost_Unserved_Energy = Cost_Unserved_Energy.sum(axis=0)
                Cost_Unserved_Energy.rename("Cost_Unserved_Energy", inplace=True)

                Total_Systems_Cost = pd.concat([Total_Systems_Cost, Total_Gen_Cost, Cost_Unserved_Energy], axis=1, sort=False)

                Total_Systems_Cost.columns = Total_Systems_Cost.columns.str.replace('_',' ')
                Total_Systems_Cost.rename({0:scenario}, axis='index', inplace=True)


                Total_Systems_Cost_Out = pd.concat([Total_Systems_Cost_Out, Total_Systems_Cost], axis=0, sort=False)

            Total_Systems_Cost_Out = Total_Systems_Cost_Out/1000000 #Convert cost to millions

            Total_Systems_Cost_Out.index = Total_Systems_Cost_Out.index.str.replace('_',' ')
            Total_Systems_Cost_Out.index = Total_Systems_Cost_Out.index.str.wrap(5, break_long_words=False)

             #Checks if Total_Systems_Cost_Out contains data, if not skips zone and does not return a plot
            if Total_Systems_Cost_Out.empty:
                out = mfunc.MissingZoneData()
                outputs[zone_input] = out
                continue

            # Data table of values to return to main program
            Data_Table_Out = Total_Systems_Cost_Out

            fig2, ax = plt.subplots(figsize=(self.x,self.y))

            sb = Total_Systems_Cost_Out.plot.bar(stacked=True, rot=0, edgecolor='black', linewidth='0.1', ax=ax)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_ylabel('Total System Cost (Million $)',  color='black', rotation='vertical')
            ax.tick_params(axis='y', which='major', length=5, width=1)
            ax.tick_params(axis='x', which='major', length=5, width=1)
            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(x, f',.{self.y_axes_decimalpt}f')))
            ax.margins(x=0.01)

            handles, labels = ax.get_legend_handles_labels()
            ax.legend(reversed(handles), reversed(labels), loc='upper center',bbox_to_anchor=(0.5,-0.15),
                         facecolor='inherit', frameon=True, ncol=2)


            """adds annotations to bar plots"""

            cost_totals = Total_Systems_Cost_Out.sum(axis=1) #holds total of each bar

            #inserts values into bar stacks
            for i in ax.patches:
               width, height = i.get_width(), i.get_height()
               if height<=1:
                   continue
               x, y = i.get_xy()
               ax.text(x+width/2,
                    y+height/2,
                    '{:,.0f}'.format(height),
                    horizontalalignment='center',
                    verticalalignment='center', fontsize=12)

            #inserts total bar value above each bar
            k=0
            for i in ax.patches:
                height = cost_totals[k]
                width = i.get_width()
                x, y = i.get_xy()
                ax.text(x+width/2,
                    y+height + 0.05*max(ax.get_ylim()),
                    '{:,.0f}'.format(height),
                    horizontalalignment='center',
                    verticalalignment='center', fontsize=15, color='red')
                k=k+1
                if k>=len(cost_totals):
                    break

            outputs[zone_input] = {'fig': fig2, 'data_table': Data_Table_Out}
        return outputs

    def detailed_gen_cost(self):
        outputs = {}
        total_gen_cost_collection = {}
        fuel_cost_collection = {}
        vom_cost_collection = {}
        start_shutdown_cost_collection = {}
        emissions_cost_collection = {}
        check_input_data = []
        
        check_input_data.extend([mfunc.get_data(total_gen_cost_collection,"generator_Total_Generation_Cost", self.Marmot_Solutions_folder, self.Scenarios)])
        check_input_data.extend([mfunc.get_data(fuel_cost_collection,"generator_Fuel_Cost", self.Marmot_Solutions_folder, self.Scenarios)])
        check_input_data.extend([mfunc.get_data(vom_cost_collection,"generator_VO&M_Cost", self.Marmot_Solutions_folder, self.Scenarios)])
        check_input_data.extend([mfunc.get_data(start_shutdown_cost_collection,"generator_Start_&_Shutdown_Cost", self.Marmot_Solutions_folder, self.Scenarios)])
        mfunc.get_data(emissions_cost_collection,"generator_Emissions_Cost", self.Marmot_Solutions_folder, self.Scenarios)
        
        # Checks if all data required by plot is available, if 1 in list required data is missing
        if 1 in check_input_data:
            outputs = mfunc.MissingInputData()
            return outputs
        
        for zone_input in self.Zones:
            self.logger.info("Zone = "+ zone_input)

            Detailed_Gen_Cost_Out = pd.DataFrame()

            for scenario in self.Scenarios:
                self.logger.info("Scenario = " + scenario)

                Fuel_Cost = fuel_cost_collection.get(scenario)
                # Check if Fuel_cost contains zone_input, skips if not
                try:
                    Fuel_Cost = Fuel_Cost.xs(zone_input,level=self.AGG_BY)
                except KeyError:
                    self.logger.warning("No Generators found for : "+zone_input)
                    continue

                Fuel_Cost = Fuel_Cost.sum(axis=0)
                Fuel_Cost.rename("Fuel_Cost", inplace=True)

                VOM_Cost = vom_cost_collection.get(scenario)
                VOM_Cost = VOM_Cost.xs(zone_input,level=self.AGG_BY)
                VOM_Cost[VOM_Cost<0]=0
                VOM_Cost = VOM_Cost.sum(axis=0)
                VOM_Cost.rename("VO&M_Cost", inplace=True)

                Start_Shutdown_Cost = start_shutdown_cost_collection.get(scenario)
                Start_Shutdown_Cost = Start_Shutdown_Cost.xs(zone_input,level=self.AGG_BY)
                Start_Shutdown_Cost = Start_Shutdown_Cost.sum(axis=0)
                Start_Shutdown_Cost.rename("Start_&_Shutdown_Cost", inplace=True)

                try:
                    emissions_cost_collection[scenario]
                except KeyError:
                    self.logger.warning("generator_Emissions_Cost not included in %s results, Emissions_Cost will not be included in plot",scenario)
                    emissions_cost_collection[scenario] = start_shutdown_cost_collection[scenario].copy()
                    emissions_cost_collection[scenario].iloc[:,0] = 0
                Emissions_Cost = emissions_cost_collection.get(scenario)
                Emissions_Cost = Emissions_Cost.xs(zone_input,level=self.AGG_BY)
                Emissions_Cost = Emissions_Cost.sum(axis=0)
                Emissions_Cost.rename("Emissions_Cost", inplace=True)

                Detailed_Gen_Cost = pd.concat([Fuel_Cost, VOM_Cost, Start_Shutdown_Cost, Emissions_Cost], axis=1, sort=False)

                Detailed_Gen_Cost.columns = Detailed_Gen_Cost.columns.str.replace('_',' ')
                Detailed_Gen_Cost = Detailed_Gen_Cost.sum(axis=0)
                Detailed_Gen_Cost = Detailed_Gen_Cost.rename(scenario)

                Detailed_Gen_Cost_Out = pd.concat([Detailed_Gen_Cost_Out, Detailed_Gen_Cost], axis=1, sort=False)

            Detailed_Gen_Cost_Out = Detailed_Gen_Cost_Out.T/1000000 #Convert cost to millions
            
            Detailed_Gen_Cost_Out.index = Detailed_Gen_Cost_Out.index.str.replace('_',' ')
            Detailed_Gen_Cost_Out.index = Detailed_Gen_Cost_Out.index.str.wrap(5, break_long_words=False)
            # Deletes columns that are all 0
            Detailed_Gen_Cost_Out = Detailed_Gen_Cost_Out.loc[:, (Detailed_Gen_Cost_Out != 0).any(axis=0)]

            # Checks if Detailed_Gen_Cost_Out contains data, if not skips zone and does not return a plot
            if Detailed_Gen_Cost_Out.empty:
                out = mfunc.MissingZoneData()
                outputs[zone_input] = out
                continue

            # Data table of values to return to main program
            Data_Table_Out = Detailed_Gen_Cost_Out

            fig3, ax = plt.subplots(figsize=(self.x,self.y))

            sb = Detailed_Gen_Cost_Out.plot.bar(stacked=True, rot=0, edgecolor='black', linewidth='0.1', ax=ax)
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


            """adds annotations to bar plots"""

            cost_totals = Detailed_Gen_Cost_Out.sum(axis=1) #holds total of each bar

            #inserts values into bar stacks
            for i in ax.patches:
                width, height = i.get_width(), i.get_height()
                if height<=2:
                   continue
                x, y = i.get_xy()
                ax.text(x+width/2,
                    y+height/2,
                    '{:,.0f}'.format(height),
                    horizontalalignment='center',
                    verticalalignment='center', fontsize=12)

            #inserts total bar value above each bar
            k=0
            for i in ax.patches:
                height = cost_totals[k]
                width = i.get_width()
                x, y = i.get_xy()
                ax.text(x+width/2,
                    y+height + 0.05*max(ax.get_ylim()),
                    '{:,.0f}'.format(height),
                    horizontalalignment='center',
                    verticalalignment='center', fontsize=15, color='red')
                k=k+1
                if k>=len(cost_totals):
                    break

            outputs[zone_input] = {'fig': fig3, 'data_table': Data_Table_Out}
        return outputs

    def sys_cost_type(self):
        # Create Dictionary to hold Datframes for each scenario
        outputs = {}
        stacked_gen_cost_collection = {}
        check_input_data = []
        
        check_input_data.extend([mfunc.get_data(stacked_gen_cost_collection,"generator_Total_Generation_Cost", self.Marmot_Solutions_folder, self.Scenarios)])
        
        # Checks if all data required by plot is available, if 1 in list required data is missing
        if 1 in check_input_data:
            outputs = mfunc.MissingInputData()
            return outputs
        
        for zone_input in self.Zones:
            Total_Generation_Stack_Out = pd.DataFrame()
            self.logger.info("Zone = " + zone_input)

            for scenario in self.Scenarios:

                self.logger.info("Scenario = " + scenario)

                Total_Gen_Stack = stacked_gen_cost_collection.get(scenario)
                # Check if Total_Gen_Stack contains zone_input, skips if not
                try:
                    Total_Gen_Stack = Total_Gen_Stack.xs(zone_input,level=self.AGG_BY)
                except KeyError:
                    self.logger.warning("No Generators found for : "+zone_input)
                    continue
                Total_Gen_Stack = mfunc.df_process_gen_inputs(Total_Gen_Stack, self.ordered_gen)

                Total_Gen_Stack = Total_Gen_Stack.sum(axis=0)
                Total_Gen_Stack.rename(scenario, inplace=True)
                Total_Generation_Stack_Out = pd.concat([Total_Generation_Stack_Out, Total_Gen_Stack], axis=1, sort=False).fillna(0)

            Total_Generation_Stack_Out = mfunc.df_process_categorical_index(Total_Generation_Stack_Out, self.ordered_gen)
            Total_Generation_Stack_Out = Total_Generation_Stack_Out.T/1000000 #Convert to millions
            Total_Generation_Stack_Out = Total_Generation_Stack_Out.loc[:, (Total_Generation_Stack_Out != 0).any(axis=0)]

            # Checks if Total_Generation_Stack_Out contains data, if not skips zone and does not return a plot
            if Total_Generation_Stack_Out.empty:
                out = mfunc.MissingZoneData()
                outputs[zone_input] = out
                continue

            # Data table of values to return to main program
            Data_Table_Out = pd.concat([Total_Generation_Stack_Out],  axis=1, sort=False)

            Total_Generation_Stack_Out.index = Total_Generation_Stack_Out.index.str.replace('_',' ')
            Total_Generation_Stack_Out.index = Total_Generation_Stack_Out.index.str.wrap(10, break_long_words=False)

            fig1, ax = plt.subplots(figsize=(self.x,self.y))

            bp = Total_Generation_Stack_Out.plot.bar(stacked=True, figsize=(self.x,self.y), rot=0,
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

    #        handles, labels = fig1.get_legend_handles_labels()

            #Legend 1
    #        leg1 = fig1.legend(reversed(handles), reversed(labels), loc='lower left',bbox_to_anchor=(1,0),
    #                      facecolor='inherit', frameon=True)


            # Manually add the first legend back
    #        fig1.add_artist(leg1)



            outputs[zone_input] = {'fig': fig1, 'data_table': Data_Table_Out}
        return outputs

    def sys_cost_diff(self):
        outputs = {}
        total_gen_cost_collection = {}
        cost_unserved_energy_collection = {}
        check_input_data = []
        
        check_input_data.extend([mfunc.get_data(total_gen_cost_collection,"generator_Total_Generation_Cost", self.Marmot_Solutions_folder, self.Scenarios)])
        if self.AGG_BY == "zone":
            mfunc.get_data(cost_unserved_energy_collection,"zone_Cost_Unserved_Energy", self.Marmot_Solutions_folder, self.Scenarios)
        else:
            mfunc.get_data(cost_unserved_energy_collection,"region_Cost_Unserved_Energy", self.Marmot_Solutions_folder, self.Scenarios)
        
        # Checks if all data required by plot is available, if 1 in list required data is missing
        if 1 in check_input_data:
            outputs = mfunc.MissingInputData()
            return outputs
        
        for zone_input in self.Zones:
            Total_Systems_Cost_Out = pd.DataFrame()
            self.logger.info("Zone = "+ zone_input)

            for scenario in self.Scenarios:
                self.logger.info("Scenario = " + scenario)
                Total_Systems_Cost = pd.DataFrame()


                Total_Gen_Cost = total_gen_cost_collection.get(scenario)

                try:
                    Total_Gen_Cost = Total_Gen_Cost.xs(zone_input,level=self.AGG_BY)
                except KeyError:
                    self.logger.warning("No Generators found for : "+ zone_input)
                    continue

                Total_Gen_Cost = Total_Gen_Cost.sum(axis=0)
                Total_Gen_Cost.rename("Total_Gen_Cost", inplace=True)
                
                try:
                    cost_unserved_energy_collection[scenario]
                except KeyError:
                    cost_unserved_energy_collection[scenario] = total_gen_cost_collection[scenario].copy()
                    cost_unserved_energy_collection[scenario].iloc[:,0] = 0
                Cost_Unserved_Energy = cost_unserved_energy_collection.get(scenario)
                Cost_Unserved_Energy = Cost_Unserved_Energy.xs(zone_input,level=self.AGG_BY)
                Cost_Unserved_Energy = Cost_Unserved_Energy.sum(axis=0)
                Cost_Unserved_Energy.rename("Cost_Unserved_Energy", inplace=True)

                Total_Systems_Cost = pd.concat([Total_Systems_Cost, Total_Gen_Cost, Cost_Unserved_Energy], axis=1, sort=False)

                Total_Systems_Cost.columns = Total_Systems_Cost.columns.str.replace('_',' ')
                Total_Systems_Cost.rename({0:scenario}, axis='index', inplace=True)


                Total_Systems_Cost_Out = pd.concat([Total_Systems_Cost_Out, Total_Systems_Cost], axis=0, sort=False)

            Total_Systems_Cost_Out = Total_Systems_Cost_Out/1000000 #Convert cost to millions
            #Ensures region has generation, else skips
            try:
                Total_Systems_Cost_Out = Total_Systems_Cost_Out-Total_Systems_Cost_Out.xs(self.Scenarios[0]) #Change to a diff on first scenario
            except KeyError:
                out = mfunc.MissingZoneData()
                outputs[zone_input] = out
                continue
            Total_Systems_Cost_Out.drop(self.Scenarios[0],inplace=True) #Drop base entry

    #        Total_Systems_Cost_Out.index = Total_Systems_Cost_Out.index.str.replace('_',' ')
    #        Total_Systems_Cost_Out.index = Total_Systems_Cost_Out.index.str.wrap(10, break_long_words=False)

            # Checks if Total_Systems_Cost_Out contains data, if not skips zone and does not return a plot
            if Total_Systems_Cost_Out.empty:
                out = mfunc.MissingZoneData()
                outputs[zone_input] = out
                continue
            # Data table of values to return to main program
            Data_Table_Out = Total_Systems_Cost_Out

            fig2, ax = plt.subplots(figsize=(self.x,self.y))

            sb = Total_Systems_Cost_Out.plot.bar(stacked=True, rot=0, edgecolor='black', linewidth='0.1', ax=ax)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

            ax.tick_params(axis='y', which='major', length=5, width=1)
            ax.tick_params(axis='x', which='major', length=5, width=1)
            locs,labels=plt.xticks()
            ax.axhline(y = 0, color = 'black')
            ax.set_ylabel('Generation Cost Change (Million $) \n relative to '+ self.Scenarios[0],  color='black', rotation='vertical')
            self.xlabels = pd.Series(self.Scenarios).str.replace('_',' ').str.wrap(10, break_long_words=False)
            plt.xticks(ticks=locs,labels=self.xlabels[1:])

            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(x, f',.{self.y_axes_decimalpt}f')))
            ax.margins(x=0.01)
    #        plt.ylim((0,600))
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(reversed(handles), reversed(labels), loc='upper center',bbox_to_anchor=(0.5,-0.15),
                         facecolor='inherit', frameon=True, ncol=2)

            outputs[zone_input] = {'fig': fig2, 'data_table': Data_Table_Out}
        return outputs

    def sys_cost_type_diff(self):
        # Create Dictionary to hold Datframes for each scenario
        outputs = {}
        stacked_gen_cost_collection = {}
        check_input_data = []
        
        check_input_data.extend([mfunc.get_data(stacked_gen_cost_collection,"generator_Total_Generation_Cost", self.Marmot_Solutions_folder, self.Scenarios)])
        
        # Checks if all data required by plot is available, if 1 in list required data is missing
        if 1 in check_input_data:
            outputs = mfunc.MissingInputData()
            return outputs
        
        for zone_input in self.Zones:
            Total_Generation_Stack_Out = pd.DataFrame()
            self.logger.info("Zone = " + zone_input)

            for scenario in self.Scenarios:

                self.logger.info("Scenario = " + scenario)

                Total_Gen_Stack = stacked_gen_cost_collection.get(scenario)

                try:
                    Total_Gen_Stack = Total_Gen_Stack.xs(zone_input,level=self.AGG_BY)
                except KeyError:
                    self.logger.warning("No Generators found for : "+zone_input)
                    continue
                Total_Gen_Stack = mfunc.df_process_gen_inputs(Total_Gen_Stack, self.ordered_gen)
                Total_Gen_Stack = Total_Gen_Stack.sum(axis=0)
                Total_Gen_Stack.rename(scenario, inplace=True)
                Total_Generation_Stack_Out = pd.concat([Total_Generation_Stack_Out, Total_Gen_Stack], axis=1, sort=False).fillna(0)

            Total_Generation_Stack_Out = mfunc.df_process_categorical_index(Total_Generation_Stack_Out, self.ordered_gen)
            Total_Generation_Stack_Out = Total_Generation_Stack_Out.T/1000000 #Convert to millions
            Total_Generation_Stack_Out = Total_Generation_Stack_Out.loc[:, (Total_Generation_Stack_Out != 0).any(axis=0)]
            #Ensures region has generation, else skips
            try:
                Total_Generation_Stack_Out = Total_Generation_Stack_Out-Total_Generation_Stack_Out.xs(self.Scenarios[0]) #Change to a diff on first scenario
            except KeyError:
                out = mfunc.MissingZoneData()
                outputs[zone_input] = out
                continue
            Total_Generation_Stack_Out.drop(self.Scenarios[0],inplace=True) #Drop base entry

            # Checks if Total_Generation_Stack_Out contains data, if not skips zone and does not return a plot
            if Total_Generation_Stack_Out.empty == True:
                out = mfunc.MissingZoneData()
                outputs[zone_input] = out
                continue

            # Data table of values to return to main program
            Data_Table_Out = pd.concat([Total_Generation_Stack_Out],  axis=1, sort=False)

            Total_Generation_Stack_Out.index = Total_Generation_Stack_Out.index.str.replace('_',' ')
            Total_Generation_Stack_Out.index = Total_Generation_Stack_Out.index.str.wrap(10, break_long_words=False)

            fig1, ax = plt.subplots(figsize=(self.x,self.y))

            bp = Total_Generation_Stack_Out.plot.bar(stacked=True, figsize=(9,6), rot=0,
                             color=[self.PLEXOS_color_dict.get(x, '#333333') for x in Total_Generation_Stack_Out.columns], edgecolor='black', linewidth='0.1',ax=ax)


            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

            ax.tick_params(axis='y', which='major', length=5, width=1)
            ax.tick_params(axis='x', which='major', length=5, width=1)
            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(x, f',.{self.y_axes_decimalpt}f')))

            locs,labels=plt.xticks()
            ax.axhline(y = 0)
            ax.set_ylabel('Generation Cost Change (Million $) \n relative to '+ self.Scenarios[0],  color='black', rotation='vertical')
            self.xlabels = pd.Series(self.Scenarios).str.replace('_',' ').str.wrap(10, break_long_words=False)
            plt.xticks(ticks=locs,labels=self.xlabels[1:])

            ax.margins(x=0.01)
    #        plt.ylim((0,600))

            handles, labels = ax.get_legend_handles_labels()
            ax.legend(reversed(handles), reversed(labels), loc='upper center',bbox_to_anchor=(0.5,-0.15),
                         facecolor='inherit', frameon=True, ncol=2)

    #        handles, labels = fig1.get_legend_handles_labels()

            #Legend 1
    #        leg1 = fig1.legend(reversed(handles), reversed(labels), loc='lower left',bbox_to_anchor=(1,0),
    #                      facecolor='inherit', frameon=True)


            # Manually add the first legend back
    #        fig1.add_artist(leg1)



            outputs[zone_input] = {'fig': fig1, 'data_table': Data_Table_Out}
        return outputs

    def detailed_gen_cost_diff(self):
        outputs = {}
        total_gen_cost_collection = {}
        fuel_cost_collection = {}
        vom_cost_collection = {}
        start_shutdown_cost_collection = {}
        emissions_cost_collection = {}
        check_input_data = []
        
        check_input_data.extend([mfunc.get_data(total_gen_cost_collection,"generator_Total_Generation_Cost", self.Marmot_Solutions_folder, self.Scenarios)])
        check_input_data.extend([mfunc.get_data(fuel_cost_collection,"generator_Fuel_Cost", self.Marmot_Solutions_folder, self.Scenarios)])
        check_input_data.extend([mfunc.get_data(vom_cost_collection,"generator_VO&M_Cost", self.Marmot_Solutions_folder, self.Scenarios)])
        check_input_data.extend([mfunc.get_data(start_shutdown_cost_collection,"generator_Start_&_Shutdown_Cost", self.Marmot_Solutions_folder, self.Scenarios)])
        mfunc.get_data(emissions_cost_collection,"generator_Emissions_Cost", self.Marmot_Solutions_folder, self.Scenarios)
        
        # Checks if all data required by plot is available, if 1 in list required data is missing
        if 1 in check_input_data:
            outputs = mfunc.MissingInputData()
            return outputs
        
        for zone_input in self.Zones:
            self.logger.info("Zone = "+ zone_input)

            Detailed_Gen_Cost_Out = pd.DataFrame()

            for scenario in self.Scenarios:
                self.logger.info("Scenario = " + scenario)

                Fuel_Cost = fuel_cost_collection.get(scenario)
                try:
                    Fuel_Cost = Fuel_Cost.xs(zone_input,level=self.AGG_BY)
                except KeyError:
                    self.logger.warning("No Generators found for: "+zone_input)
                    continue
                Fuel_Cost = Fuel_Cost.sum(axis=0)
                Fuel_Cost.rename("Fuel_Cost", inplace=True)

                VOM_Cost = vom_cost_collection.get(scenario)
                VOM_Cost = VOM_Cost.xs(zone_input,level=self.AGG_BY)
                VOM_Cost = VOM_Cost.sum(axis=0)
                VOM_Cost.rename("VO&M_Cost", inplace=True)

                Start_Shutdown_Cost = start_shutdown_cost_collection.get(scenario)
                Start_Shutdown_Cost = Start_Shutdown_Cost.xs(zone_input,level=self.AGG_BY)
                Start_Shutdown_Cost = Start_Shutdown_Cost.sum(axis=0)
                Start_Shutdown_Cost.rename("Start_&_Shutdown_Cost", inplace=True)
                
                try:
                    emissions_cost_collection[scenario]
                except KeyError:
                    self.logger.warning("generator_Emissions_Cost not included in %s results, Emissions_Cost will not be included in plot",scenario)
                    emissions_cost_collection[scenario] = start_shutdown_cost_collection[scenario].copy()
                    emissions_cost_collection[scenario].iloc[:,0] = 0
                    
                Emissions_Cost = emissions_cost_collection.get(scenario)
                Emissions_Cost = Emissions_Cost.xs(zone_input,level=self.AGG_BY)
                Emissions_Cost = Emissions_Cost.sum(axis=0)
                Emissions_Cost.rename("Emissions_Cost", inplace=True)

                Detailed_Gen_Cost = pd.concat([Fuel_Cost, VOM_Cost, Start_Shutdown_Cost, Emissions_Cost], axis=1, sort=False)

                Detailed_Gen_Cost.columns = Detailed_Gen_Cost.columns.str.replace('_',' ')
                Detailed_Gen_Cost = Detailed_Gen_Cost.sum(axis=0)
                Detailed_Gen_Cost = Detailed_Gen_Cost.rename(scenario)

                Detailed_Gen_Cost_Out = pd.concat([Detailed_Gen_Cost_Out, Detailed_Gen_Cost], axis=1, sort=False)

            Detailed_Gen_Cost_Out = Detailed_Gen_Cost_Out.T/1000000 #Convert cost to millions
            #Ensures region has generation, else skips
            try:
                Detailed_Gen_Cost_Out = Detailed_Gen_Cost_Out-Detailed_Gen_Cost_Out.xs(self.Scenarios[0]) #Change to a diff on first scenario
            except KeyError:
                out = mfunc.MissingZoneData()
                outputs[zone_input] = out
                continue
            Detailed_Gen_Cost_Out.drop(self.Scenarios[0],inplace=True) #Drop base entry
            net_cost = Detailed_Gen_Cost_Out.sum(axis = 1)
            Detailed_Gen_Cost_Out.index = Detailed_Gen_Cost_Out.index.str.replace('_',' ')
            Detailed_Gen_Cost_Out.index = Detailed_Gen_Cost_Out.index.str.wrap(10, break_long_words=False)
            # Deletes columns that are all 0
            Detailed_Gen_Cost_Out = Detailed_Gen_Cost_Out.loc[:, (Detailed_Gen_Cost_Out != 0).any(axis=0)]

            # Checks if Detailed_Gen_Cost_Out contains data, if not skips zone and does not return a plot
            if Detailed_Gen_Cost_Out.empty == True:
                out = mfunc.MissingZoneData()
                outputs[zone_input] = out
                continue

            # Data table of values to return to main program
            Data_Table_Out = Detailed_Gen_Cost_Out

            fig3, ax = plt.subplots(figsize=(self.x,self.y))

            sb = Detailed_Gen_Cost_Out.plot.bar(stacked=True, rot=0, edgecolor='black', linewidth='0.1', ax=ax)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.axhline(y= 0 ,linewidth=0.5,linestyle='--',color='grey')
            ax.tick_params(axis='y', which='major', length=5, width=1)
            ax.tick_params(axis='x', which='major', length=5, width=1)
            locs,labels=plt.xticks()
            ax.set_ylabel('Generation Cost Change (Million $) \n relative to '+ self.Scenarios[0],  color='black', rotation='vertical')
            self.xlabels = pd.Series(self.Scenarios).str.replace('_',' ').str.wrap(10, break_long_words=False)
            plt.xticks(ticks=locs,labels=self.xlabels[1:])

            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(x, f',.{self.y_axes_decimalpt}f')))
            ax.margins(x=0.01)
    #        plt.ylim((0,600))

            #Add net cost line.
            n=0
            for scenario in self.Scenarios[1:]:
                x = [ax.patches[n].get_x(), ax.patches[n].get_x() + ax.patches[n].get_width()]
                y_net = [net_cost.loc[scenario]] * 2
                net_line = plt.plot(x,y_net, c='black', linewidth=1.5)
                n += 1

            handles, labels = ax.get_legend_handles_labels()

            #Main Legend
            leg_main = ax.legend(reversed(handles), reversed(labels), loc='lower left',bbox_to_anchor=(1,0),
                          facecolor='inherit', frameon=True)

            #Net cost legend
            leg_net = ax.legend(net_line,['Net Cost Change'],loc='center left',bbox_to_anchor=(1, -0.05),facecolor='inherit', frameon=True)
            ax.add_artist(leg_main)
            ax.add_artist(leg_net)

            outputs[zone_input] = {'fig': fig3, 'data_table': Data_Table_Out}
        return outputs
