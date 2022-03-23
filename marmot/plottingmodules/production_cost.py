# -*- coding: utf-8 -*-
"""System operating cost plots.

This module plots figures related to the cost of operating the power system.
Plots can be broken down by cost categories, generator types etc. 

@author: Daniel Levie
"""

import logging
import pandas as pd
from pathlib import Path

import marmot.utils.mconfig as mconfig
from marmot.plottingmodules.plotutils.plot_library import PlotLibrary
from marmot.plottingmodules.plotutils.plot_data_helper import MPlotDataHelper
from marmot.plottingmodules.plotutils.plot_exceptions import (MissingInputData, MissingZoneData)

plot_data_settings = mconfig.parser("plot_data")
logger = logging.getLogger('plotter.'+__name__)

class SystemCosts(MPlotDataHelper):
    """System operating cost plots.

    The production_cost.py module contains methods that are
    related related to the cost of operating the power system. 
    
    SystemCosts inherits from the MPlotDataHelper class to assist 
    in creating figures.
    """

    def __init__(self, **kwargs):
        # Instantiation of MPlotHelperFunctions
        super().__init__(**kwargs)
        
                
    def prod_cost(self, start_date_range: str = None, 
                  end_date_range: str = None, 
                  custom_data_file_path: Path= None,
                  barplot_groupby: str = 'Scenario', **_):
        """Plots total system net revenue and cost.

        Total revenue is made up of reserve and energy revenues which are displayed in a stacked
        bar plot with total generation cost. Net revensue is represented by a dot.
        Each sceanrio is plotted as a separate bar.

        Args:
            start_date_range (str, optional): Defines a start date at which to represent data from. 
                Defaults to None.
            end_date_range (str, optional): Defines a end date at which to represent data to.
                Defaults to None.
            custom_data_file_path (Path, optional): Path to custom data file to concat extra 
                data. Index and column format should be consistent with output data csv.

        Returns:
            dict: Dictionary containing the created plot and its data table.
        """
        outputs : dict = {}
        
        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True, "generator_Total_Generation_Cost", self.Scenarios),
                      (True, "generator_Pool_Revenue", self.Scenarios),
                      (True, "generator_Reserves_Revenue", self.Scenarios),
                      (True, "generator_Installed_Capacity", self.Scenarios)]
        
        # Runs get_formatted_data within MPlotDataHelper to populate MPlotDataHelper dictionary  
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)
        
        # Checks if all data required by plot is available, if 1 in list required data is missing
        if 1 in check_input_data:
            return MissingInputData()
        
        for zone_input in self.Zones:
            total_cost_chunk = []
            logger.info(f"{self.AGG_BY} = {zone_input}")
            for scenario in self.Scenarios:

                logger.info(f"Scenario = {scenario}")
                Total_Installed_Capacity = self["generator_Installed_Capacity"].get(scenario)
                #Check if zone has installed generation, if not skips
                try:
                    Total_Installed_Capacity = Total_Installed_Capacity.xs(zone_input,level=self.AGG_BY)
                except KeyError:
                    logger.warning(f"No installed capacity in : {zone_input}")
                    continue
                Total_Installed_Capacity = self.df_process_gen_inputs(Total_Installed_Capacity)
                Total_Installed_Capacity.reset_index(drop=True, inplace=True)
                Total_Installed_Capacity = Total_Installed_Capacity.iloc[0]

                gen_cost = self["generator_Total_Generation_Cost"].get(scenario)
                gen_cost = gen_cost.xs(zone_input,level=self.AGG_BY)
                gen_cost = self.df_process_gen_inputs(gen_cost)
                gen_cost = gen_cost.sum(axis=0)*-1
                # gen_cost = gen_cost/Total_Installed_Capacity #Change to $/MW-year
                gen_cost.rename("Generation Cost", inplace=True)

                Pool_Revenues = self["generator_Pool_Revenue"].get(scenario)
                Pool_Revenues = Pool_Revenues.xs(zone_input,level=self.AGG_BY)
                Pool_Revenues = self.df_process_gen_inputs(Pool_Revenues)
                Pool_Revenues = Pool_Revenues.sum(axis=0)
                # Pool_Revenues = Pool_Revenues/Total_Installed_Capacity #Change to $/MW-year
                Pool_Revenues.rename("Energy Revenues", inplace=True)

                ### Might change to Net Reserve Revenue at later date
                Reserve_Revenues = self["generator_Reserves_Revenue"].get(scenario)
                Reserve_Revenues = Reserve_Revenues.xs(zone_input,level=self.AGG_BY)
                Reserve_Revenues = self.df_process_gen_inputs(Reserve_Revenues)
                Reserve_Revenues = Reserve_Revenues.sum(axis=0)
                # Reserve_Revenues = Reserve_Revenues/Total_Installed_Capacity #Change to $/MW-year
                Reserve_Revenues.rename("Reserve Revenues", inplace=True)

                Total_Systems_Cost = pd.concat([gen_cost, Pool_Revenues, 
                                                Reserve_Revenues], 
                                               axis=1, sort=False)

                Total_Systems_Cost = Total_Systems_Cost.sum(axis=0)
                Total_Systems_Cost = Total_Systems_Cost.rename(scenario)
                
                total_cost_chunk.append(Total_Systems_Cost)

            total_systems_cost_out = pd.concat(total_cost_chunk, axis=1, sort=False)

            total_systems_cost_out = total_systems_cost_out.T
                        
            # total_systems_cost_out = total_systems_cost_out/1000 #Change to $/kW-year
            total_systems_cost_out = total_systems_cost_out/1e6 #Convert cost to millions
            
            if pd.notna(custom_data_file_path):
                total_systems_cost_out = self.insert_custom_data_columns(
                                                        total_systems_cost_out, 
                                                        custom_data_file_path)

            Net_Revenue = total_systems_cost_out.sum(axis=1)

            #Checks if Net_Revenue contains data, if not skips zone and does not return a plot
            if Net_Revenue.empty:
                out = MissingZoneData()
                outputs[zone_input] = out
                continue

            # Data table of values to return to main program
            Data_Table_Out = total_systems_cost_out.add_suffix(" (Million $)")

            mplt = PlotLibrary()
            fig, ax = mplt.get_figure()

            # Set x-tick labels
            if self.custom_xticklabels:
                tick_labels = self.custom_xticklabels
            else:
                tick_labels = total_systems_cost_out.index

            mplt.barplot(total_systems_cost_out, stacked=True, 
                         custom_tick_labels=tick_labels)
            ax.plot(Net_Revenue.index, Net_Revenue.values, 
                    color='black', linestyle='None', marker='o',
                    label='Net Revenue')
            
            ax.set_ylabel('Total System Net Rev, Rev, & Cost (Million $)',
                        color='black', rotation='vertical')
            ax.margins(x=0.01)

            mplt.add_legend(reverse_legend=True)
            if plot_data_settings["plot_title_as_region"]:
                mplt.add_main_title(zone_input)

            outputs[zone_input] = {'fig': fig, 'data_table': Data_Table_Out}
        return outputs

    def sys_cost(self, start_date_range: str = None, 
                 end_date_range: str = None, 
                 custom_data_file_path: Path= None,
                 barplot_groupby: str = 'Scenario', **_):
        """Creates a stacked bar plot of Total Generation Cost and Cost of Unserved Energy.

        Plot only shows totals and is NOT broken down into technology or cost type 
        specific values.
        Each sceanrio is plotted as a separate bar.

        Args:
            start_date_range (str, optional): Defines a start date at which to represent data from. 
                Defaults to None.
            end_date_range (str, optional): Defines a end date at which to represent data to.
                Defaults to None.
            custom_data_file_path (Path, optional): Path to custom data file to concat extra 
                data. Index and column format should be consistent with output data csv.

        Returns:
            dict: Dictionary containing the created plot and its data table.
        """
        outputs : dict = {}
        
        if self.AGG_BY == 'zone':
            agg = 'zone'
        else:
            agg = 'region'
            
        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True, "generator_Total_Generation_Cost", self.Scenarios),
                      (False, f"{agg}_Cost_Unserved_Energy", self.Scenarios)]
        
        # Runs get_formatted_data within MPlotDataHelper to populate MPlotDataHelper dictionary  
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)
        
        # Checks if all data required by plot is available, if 1 in list required data is missing
        if 1 in check_input_data:
            return MissingInputData()
        
        for zone_input in self.Zones:
            system_cost_chunk = []
            logger.info(f"{self.AGG_BY} = {zone_input}")

            for scenario in self.Scenarios:
                logger.info(f"Scenario = {scenario}")

                gen_cost : pd.DataFrame = self["generator_Total_Generation_Cost"].get(scenario)
                try:
                    gen_cost = gen_cost.xs(zone_input,level=self.AGG_BY)
                except KeyError:
                    logger.warning(f"No Generators found in : {zone_input}")
                    continue
                gen_cost = gen_cost.rename(columns={0: "Total Generation Cost"})

                cost_unserved_energy : pd.DataFrame = self[f"{agg}_Cost_Unserved_Energy"][scenario]
                if cost_unserved_energy.empty:
                    cost_unserved_energy = self["generator_Total_Generation_Cost"][scenario].copy()
                    cost_unserved_energy.iloc[:,0] = 0
                cost_unserved_energy = cost_unserved_energy.xs(zone_input, level=self.AGG_BY)
                cost_unserved_energy = cost_unserved_energy.rename(columns={0: "Cost Unserved Energy"})

                if pd.notna(start_date_range):
                    gen_cost, cost_unserved_energy = \
                        self.set_timestamp_date_range([gen_cost, cost_unserved_energy],
                                    start_date_range, end_date_range)
                    if gen_cost.empty is True:
                        logger.warning('No generation in selected Date Range')
                        continue
                
                gen_cost = self.year_scenario_grouper(gen_cost, 
                                            scenario, groupby=barplot_groupby).sum()
                cost_unserved_energy = self.year_scenario_grouper(cost_unserved_energy, 
                                            scenario, groupby=barplot_groupby).sum()

                system_cost_chunk.append(pd.concat([gen_cost, cost_unserved_energy], axis=1))

            # Checks if gen_cost_out_chunks contains data, if not skips zone and does not return a plot
            if not system_cost_chunk:
                outputs[zone_input] = MissingZoneData()
                continue
            
            total_systems_cost_out = pd.concat(system_cost_chunk, axis=0, sort=False)
            total_systems_cost_out = total_systems_cost_out/1000000 #Convert cost to millions
            
             #Checks if total_systems_cost_out contains data, if not skips zone and does not return a plot
            if total_systems_cost_out.empty:
                outputs[zone_input] = MissingZoneData()
                continue
            
            if pd.notna(custom_data_file_path):
                total_systems_cost_out = self.insert_custom_data_columns(
                                                        total_systems_cost_out, 
                                                        custom_data_file_path)

            # Data table of values to return to main program
            Data_Table_Out = total_systems_cost_out.add_suffix(" (Million $)")
            
            mplt = PlotLibrary()
            fig, ax = mplt.get_figure()

            # Set x-tick labels
            if self.custom_xticklabels:
                tick_labels = self.custom_xticklabels
            else:
                tick_labels = total_systems_cost_out.index

            mplt.barplot(total_systems_cost_out, stacked=True, 
                         custom_tick_labels=tick_labels)
            ax.set_ylabel('Total System Cost (Million $)', 
                            color='black', rotation='vertical')        
            ax.margins(x=0.01)

            mplt.add_legend(reverse_legend=True)
            if plot_data_settings["plot_title_as_region"]:
                mplt.add_main_title(zone_input)

            cost_totals = total_systems_cost_out.sum(axis=1) #holds total of each bar

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
            
            outputs[zone_input] = {'fig': fig, 'data_table': Data_Table_Out}
        return outputs

    def detailed_gen_cost(self, start_date_range: str = None, 
                          end_date_range: str = None, 
                          custom_data_file_path: Path= None,
                          barplot_groupby: str = 'Scenario', **_):
        """Creates stacked bar plot of total generation cost by cost type (fuel, emission, start cost etc.)

        Creates a more deatiled system cost plot.
        Each sceanrio is plotted as a separate bar.

        Args:
            start_date_range (str, optional): Defines a start date at which to represent data from. 
                Defaults to None.
            end_date_range (str, optional): Defines a end date at which to represent data to.
                Defaults to None.
            custom_data_file_path (Path, optional): Path to custom data file to concat extra 
                data. Index and column format should be consistent with output data csv.

        Returns:
            dict: Dictionary containing the created plot and its data table.
        """
        outputs : dict = {}
        
        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(False, "generator_FO&M_Cost", self.Scenarios),
                      (False, "generator_VO&M_Cost", self.Scenarios),
                      (False, "generator_Fuel_Cost", self.Scenarios),
                      (False, "generator_Start_&_Shutdown_Cost", self.Scenarios),
                      (False, "generator_Reserves_VO&M_Cost", self.Scenarios),
                      (False, "generator_Emissions_Cost", self.Scenarios)]
        
        # Runs get_formatted_data within MPlotDataHelper to populate MPlotDataHelper dictionary  
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)
    
        # Checks if all data required by plot is available, if 1 in list required data is missing
        if 1 in check_input_data:
            return MissingInputData()
        
        for zone_input in self.Zones:
            logger.info(f"Zone = {zone_input}")
            gen_cost_out_chunks = []

            for scenario in self.Scenarios:
                logger.info(f"Scenario = {scenario}")

                data_frames_lst = []
                for prop_name in properties:
                    df : pd.DataFrame = self[prop_name[1]].get(scenario)
                    if df.empty:
                        date_index = pd.date_range(start="2010-01-01", 
                                        periods=1,
                                        freq='H', name="timestamp")
                        df = pd.DataFrame(data=[0], index=date_index)
                    else:
                        try:
                            df = df.xs(zone_input, level=self.AGG_BY)
                            df = df.groupby(["timestamp"]).sum()
                        except KeyError:
                            logger.warning(f"No Generators found in: {zone_input}")
                            break

                    if prop_name[1] == "generator_VO&M_Cost":
                        df[0].to_numpy()[df[0].to_numpy() < 0] = 0
                    df = df.rename(columns={0: prop_name[1]})
                    data_frames_lst.append(df)

                detailed_gen_cost = pd.concat(data_frames_lst, axis=1).fillna(0)
                detailed_gen_cost = detailed_gen_cost.rename(columns=
                                            {"generator_FO&M_Cost": "FO&M Cost",
                                             "generator_VO&M_Cost": "VO&M Cost",
                                             "generator_Fuel_Cost": "Fuel Cost",
                                             "generator_Start_&_Shutdown_Cost": "Start & Shutdown Cost",
                                             "generator_Reserves_VO&M_Cost": "Reserves VO&M Cost",
                                             "generator_Emissions_Cost": "Emissions Cost"})

                if pd.notna(start_date_range):
                    detailed_gen_cost = self.set_timestamp_date_range(
                                        detailed_gen_cost,
                                        start_date_range, end_date_range)
                    if detailed_gen_cost.empty is True:
                        logger.warning('No Generation in selected Date Range')
                        continue
                
                gen_cost_out_chunks.append(self.year_scenario_grouper(detailed_gen_cost, scenario, 
                                                        groupby=barplot_groupby).sum())
            
            # Checks if gen_cost_out_chunks contains data, 
            # if not skips zone and does not return a plot
            if not gen_cost_out_chunks:
                outputs[zone_input] = MissingZoneData()
                continue
            
            detailed_gen_cost_out = pd.concat(gen_cost_out_chunks, axis=0, sort=False)
            detailed_gen_cost_out = detailed_gen_cost_out/1000000 #Convert cost to millions
            
            # Deletes columns that are all 0
            detailed_gen_cost_out = detailed_gen_cost_out.loc[:, (detailed_gen_cost_out != 0).any(axis=0)]
            
            # Checks if detailed_gen_cost_out contains data, if not skips zone and does not return a plot
            if detailed_gen_cost_out.empty:
                outputs[zone_input] = MissingZoneData()
                continue
            
            if pd.notna(custom_data_file_path):
                total_systems_cost_out = self.insert_custom_data_columns(
                                                        total_systems_cost_out, 
                                                        custom_data_file_path)
                                                        
            # Data table of values to return to main program
            Data_Table_Out = detailed_gen_cost_out.add_suffix(" (Million $)")
            
            # Set x-tick labels
            if self.custom_xticklabels:
                tick_labels = self.custom_xticklabels
            else:
                tick_labels = detailed_gen_cost_out.index

            mplt = PlotLibrary()
            fig, ax = mplt.get_figure()

            mplt.barplot(detailed_gen_cost_out, stacked=True, 
                         custom_tick_labels=tick_labels)
            ax.axhline(y=0)
            ax.set_ylabel('Total Generation Cost (Million $)', 
                          color='black', rotation='vertical')
            ax.margins(x=0.01)
            mplt.add_legend(reverse_legend=True)
            if plot_data_settings["plot_title_as_region"]:
                mplt.add_main_title(zone_input)
                
            cost_totals = detailed_gen_cost_out.sum(axis=1) #holds total of each bar

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
            
            outputs[zone_input] = {'fig': fig, 'data_table': Data_Table_Out}
        return outputs


    def sys_cost_type(self, start_date_range: str = None, 
                      end_date_range: str = None, 
                      custom_data_file_path: Path = None,
                      barplot_groupby: str = 'Scenario', **_):
        """Creates stacked bar plot of total generation cost by generator technology type.

        Another way to represent total generation cost, this time by tech type,
        i.e Coal, Gas, Hydro etc.
        Each sceanrio is plotted as a separate bar.

        Args:
            start_date_range (str, optional): Defines a start date at which to represent data from. 
                Defaults to None.
            end_date_range (str, optional): Defines a end date at which to represent data to.
                Defaults to None.
            custom_data_file_path (Path, optional): Path to custom data file to concat extra 
                data. Index and column format should be consistent with output data csv.

        Returns:
            dict: Dictionary containing the created plot and its data table.
        """
        # Create Dictionary to hold Datframes for each scenario
        outputs : dict = {}
        
        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True,"generator_Total_Generation_Cost",self.Scenarios)]
        
        # Runs get_formatted_data within MPlotDataHelper to populate MPlotDataHelper dictionary  
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)
        
        # Checks if all data required by plot is available, if 1 in list required data is missing
        if 1 in check_input_data:
            return MissingInputData()
        
        for zone_input in self.Zones:
            gen_cost_out_chunks = []
            logger.info(f"Zone = {zone_input}")

            for scenario in self.Scenarios:
                logger.info(f"Scenario = {scenario}")

                gen_cost : pd.DataFrame = self["generator_Total_Generation_Cost"].get(scenario)
                # Check if gen_cost contains zone_input, skips if not
                try:
                    gen_cost = gen_cost.xs(zone_input,level=self.AGG_BY)
                except KeyError:
                    logger.warning(f"No Generators found for : {zone_input}")
                    continue
                gen_cost = self.df_process_gen_inputs(gen_cost)

                if pd.notna(start_date_range):
                    gen_cost = self.set_timestamp_date_range(gen_cost,
                                    start_date_range, end_date_range)
                    if gen_cost.empty is True:
                        logger.warning('No generation in selected Date Range')
                        continue
                
                gen_cost_out_chunks.append(self.year_scenario_grouper(gen_cost, 
                                            scenario, groupby=barplot_groupby).sum())
            
            # Checks if gen_cost_out_chunks contains data, if not skips zone and does not return a plot
            if not gen_cost_out_chunks:
                outputs[zone_input] = MissingZoneData()
                continue
            
            total_systems_cost_out = pd.concat(gen_cost_out_chunks, axis=0, sort=False).fillna(0)
            total_systems_cost_out = total_systems_cost_out/1000000 #Convert to millions
            total_systems_cost_out = total_systems_cost_out.loc[:, (total_systems_cost_out != 0).any(axis=0)]

            # Checks if total_systems_cost_out contains data, if not skips zone and does not return a plot
            if total_systems_cost_out.empty:
                outputs[zone_input] = MissingZoneData()
                continue
            
            if pd.notna(custom_data_file_path):
                total_systems_cost_out = self.insert_custom_data_columns(
                                                    total_systems_cost_out,
                                                    custom_data_file_path)
            # Data table of values to return to main program
            Data_Table_Out = total_systems_cost_out.add_suffix(" (Million $)")
            
            mplt = PlotLibrary()
            fig, ax = mplt.get_figure()

            # Set x-tick labels
            if self.custom_xticklabels:
                tick_labels = self.custom_xticklabels
            else:
                tick_labels = total_systems_cost_out.index

            mplt.barplot(total_systems_cost_out, 
                        color=self.PLEXOS_color_dict, stacked=True, 
                        custom_tick_labels=tick_labels)

            ax.set_ylabel('Total System Cost (Million $)',  color='black', rotation='vertical')            
            ax.margins(x=0.01)

            mplt.add_legend(reverse_legend=True)
            if plot_data_settings["plot_title_as_region"]:
                mplt.add_main_title(zone_input)

            outputs[zone_input] = {'fig': fig, 'data_table': Data_Table_Out}
        return outputs


    def sys_cost_diff(self, start_date_range: str = None, 
                      end_date_range: str = None,
                      barplot_groupby: str = 'Scenario', **_):
        """Creates stacked barplots of Total Generation Cost and Cost of Unserved Energy relative to a base scenario.

        Barplots show the change in total total generation cost relative to a base scenario.
        The default is to comapre against the first scenario provided in the inputs list.
        Plot only shows totals and is NOT broken down into technology or cost type specific values.
        Each sceanrio is plotted as a separate bar.

        Args:
            start_date_range (str, optional): Defines a start date at which to represent data from. 
                Defaults to None.
            end_date_range (str, optional): Defines a end date at which to represent data to.
                Defaults to None.

        Returns:
            dict: Dictionary containing the created plot and its data table.
        """
        if self.AGG_BY == 'zone':
            agg = 'zone'
        else:
            agg = 'region'
            
        outputs : dict = {}
        
        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True, "generator_Total_Generation_Cost", self.Scenarios),
                      (False, f"{agg}_Cost_Unserved_Energy", self.Scenarios)]
        
        # Runs get_formatted_data within MPlotDataHelper to populate MPlotDataHelper dictionary  
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)

        # Checks if all data required by plot is available, if 1 in list required data is missing
        if 1 in check_input_data:
            return MissingInputData()
        
        for zone_input in self.Zones:
            system_cost_chunk = []
            logger.info(f"Zone = {zone_input}")

            for scenario in self.Scenarios:
                logger.info(f"Scenario = {scenario}")
                
                gen_cost : pd.DataFrame = self["generator_Total_Generation_Cost"].get(scenario)
                try:
                    gen_cost = gen_cost.xs(zone_input,level=self.AGG_BY)
                except KeyError:
                    logger.warning(f"No Generators found in : {zone_input}")
                    continue
                gen_cost = gen_cost.rename(columns={0: "Total Generation Cost"})

                cost_unserved_energy : pd.DataFrame = self[f"{agg}_Cost_Unserved_Energy"][scenario]
                if cost_unserved_energy.empty:
                    cost_unserved_energy = self["generator_Total_Generation_Cost"][scenario].copy()
                    cost_unserved_energy.iloc[:,0] = 0
                cost_unserved_energy = cost_unserved_energy.xs(zone_input, level=self.AGG_BY)
                cost_unserved_energy = cost_unserved_energy.rename(columns={0: "Cost Unserved Energy"})

                if pd.notna(start_date_range):
                    gen_cost, cost_unserved_energy = \
                        self.set_timestamp_date_range([gen_cost, cost_unserved_energy],
                                    start_date_range, end_date_range)
                    if gen_cost.empty is True:
                        logger.warning('No generation in selected Date Range')
                        continue
                
                gen_cost = self.year_scenario_grouper(gen_cost, 
                                            scenario, groupby=barplot_groupby).sum()
                cost_unserved_energy = self.year_scenario_grouper(cost_unserved_energy, 
                                            scenario, groupby=barplot_groupby).sum()
                system_cost_chunk.append(pd.concat([gen_cost, cost_unserved_energy], axis=1))
            
            # Checks if total_cost_chunk contains data, if not skips zone and does not return a plot
            if not system_cost_chunk:
                outputs[zone_input] = MissingZoneData()
                continue
            
            total_systems_cost_out = pd.concat(system_cost_chunk, axis=0, sort=False)
            total_systems_cost_out = total_systems_cost_out/1000000 #Convert cost to millions
            #Ensures region has generation, else skips
            try:
                #Change to a diff on first scenario
                scen_base = total_systems_cost_out.index[0]
                total_systems_cost_out = total_systems_cost_out - \
                    total_systems_cost_out.xs(scen_base)
            except KeyError:
                outputs[zone_input] = MissingZoneData()
                continue
            total_systems_cost_out.drop(scen_base, inplace=True) #Drop base entry

            # Checks if total_systems_cost_out contains data, if not skips zone and does not return a plot
            if total_systems_cost_out.empty:
                outputs[zone_input] = MissingZoneData()
                continue
                        
            # Data table of values to return to main program
            Data_Table_Out = total_systems_cost_out
            Data_Table_Out = Data_Table_Out.add_suffix(" (Million $)")
            
            mplt = PlotLibrary()
            fig, ax = mplt.get_figure()

            mplt.barplot(total_systems_cost_out, stacked=True)

            ax.axhline(y=0, color='black')
            ax.set_ylabel(f'Generation Cost Change (Million $) \n relative to {scen_base}', 
                            color='black', rotation='vertical')
            ax.margins(x=0.01)
            # plt.ylim((0,600))
            mplt.add_legend()

            if plot_data_settings["plot_title_as_region"]:
                mplt.add_main_title(zone_input)

            outputs[zone_input] = {'fig': fig, 'data_table': Data_Table_Out}
        return outputs


    def sys_cost_type_diff(self, start_date_range: str = None, 
                           end_date_range: str = None,
                            barplot_groupby: str = 'Scenario', **_):
        """Creates stacked barplots of Total Generation Cost by generator technology type relative to a base scenario.

        Barplots show the change in total total generation cost relative to a base scenario.
        The default is to comapre against the first scenario provided in the inputs list.
        Each sceanrio is plotted as a separate bar.

        Args:
            start_date_range (str, optional): Defines a start date at which to represent data from. 
                Defaults to None.
            end_date_range (str, optional): Defines a end date at which to represent data to.
                Defaults to None.

        Returns:
            dict: Dictionary containing the created plot and its data table.
        """
        # Create Dictionary to hold Datframes for each scenario
        outputs : dict = {}
        
        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True, "generator_Total_Generation_Cost" ,self.Scenarios)]
        
        # Runs get_formatted_data within MPlotDataHelper to populate MPlotDataHelper dictionary  
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)
        # Checks if all data required by plot is available, if 1 in list required data is missing
        if 1 in check_input_data:
            return MissingInputData()
        
        for zone_input in self.Zones:
            gen_cost_out_chunks = []
            logger.info(f"Zone = {zone_input}")

            for scenario in self.Scenarios:
                logger.info(f"Scenario = {scenario}")

                gen_cost = self["generator_Total_Generation_Cost"].get(scenario)

                try:
                    gen_cost = gen_cost.xs(zone_input, level=self.AGG_BY)
                except KeyError:
                    logger.warning(f"No Generators found for : {zone_input}")
                    continue
                
                gen_cost = self.df_process_gen_inputs(gen_cost)
                
                if pd.notna(start_date_range):
                    gen_cost = self.set_timestamp_date_range(gen_cost,
                                    start_date_range, end_date_range)
                    if gen_cost.empty is True:
                        logger.warning('No generation in selected Date Range')
                        continue
                
                gen_cost_out_chunks.append(self.year_scenario_grouper(gen_cost, 
                                            scenario, groupby=barplot_groupby).sum())
            
            # Checks if gen_cost_out_chunks contains data, if not skips zone and does not return a plot
            if not gen_cost_out_chunks:
                outputs[zone_input] = MissingZoneData()
                continue
            
            total_systems_cost_out = pd.concat(gen_cost_out_chunks, axis=0, sort=False).fillna(0)
            total_systems_cost_out = total_systems_cost_out/1000000 #Convert to millions
            total_systems_cost_out = total_systems_cost_out.loc[:, (total_systems_cost_out != 0).any(axis=0)]
            #Ensures region has generation, else skips
            try:
                #Change to a diff on first scenario
                scen_base = total_systems_cost_out.index[0]
                total_systems_cost_out = total_systems_cost_out - \
                    total_systems_cost_out.xs(scen_base) 
            except KeyError:
                outputs[zone_input] = MissingZoneData()
                continue
            total_systems_cost_out.drop(scen_base, inplace=True) #Drop base entry

            # Checks if total_systems_cost_out contains data, if not skips zone and does not return a plot
            if total_systems_cost_out.empty == True:
                outputs[zone_input] = MissingZoneData()
                continue
            
            # Data table of values to return to main program
            Data_Table_Out = total_systems_cost_out.add_suffix(" (Million $)")
            
            mplt = PlotLibrary()
            fig, ax = mplt.get_figure()

            mplt.barplot(total_systems_cost_out, 
                        color=self.PLEXOS_color_dict, stacked=True)

            ax.axhline(y=0)
            ax.set_ylabel(f'Generation Cost Change (Million $) \n relative to {scen_base}', 
                            color='black', rotation='vertical')
            ax.margins(x=0.01)
            # plt.ylim((0,600))

            mplt.add_legend()
            if plot_data_settings["plot_title_as_region"]:
                mplt.add_main_title(zone_input)

            outputs[zone_input] = {'fig': fig, 'data_table': Data_Table_Out}
        return outputs


    def detailed_gen_cost_diff(self, start_date_range: str = None, 
                               end_date_range: str = None,
                               barplot_groupby: str = 'Scenario', **_):
        """Creates stacked barplots of Total Generation Cost by by cost type (fuel, emission, start cost etc.)
        relative to a base scenario.

        Barplots show the change in total total generation cost relative to a base scenario.
        The default is to comapre against the first scenario provided in the inputs list.
        Each sceanrio is plotted as a separate bar.

        Args:
            start_date_range (str, optional): Defines a start date at which to represent data from. 
                Defaults to None.
            end_date_range (str, optional): Defines a end date at which to represent data to.
                Defaults to None.

        Returns:
            dict: Dictionary containing the created plot and its data table.
        """
        outputs : dict = {}
        
        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(False, "generator_FO&M_Cost", self.Scenarios),
                      (False, "generator_VO&M_Cost", self.Scenarios),
                      (False, "generator_Fuel_Cost", self.Scenarios),
                      (False, "generator_Start_&_Shutdown_Cost", self.Scenarios),
                      (False, "generator_Reserves_VO&M_Cost", self.Scenarios),
                      (False, "generator_Emissions_Cost", self.Scenarios)]
        
        # Runs get_formatted_data within MPlotDataHelper to populate MPlotDataHelper dictionary  
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)

        
        # Checks if all data required by plot is available, if 1 in list required data is missing
        if 1 in check_input_data:
            return MissingInputData()
        
        for zone_input in self.Zones:
            logger.info(f"Zone = {zone_input}")
            gen_cost_out_chunks = []

            for scenario in self.Scenarios:
                logger.info(f"Scenario = {scenario}")

                data_frames_lst = []
                for prop_name in properties:
                    df : pd.DataFrame = self[prop_name[1]].get(scenario)
                    if df.empty:
                        date_index = pd.date_range(start="2010-01-01", 
                                        periods=1,
                                        freq='H', name="timestamp")
                        df = pd.DataFrame(data=[0], index=date_index)
                    else:
                        try:
                            df = df.xs(zone_input, level=self.AGG_BY)
                            df = df.groupby("timestamp").sum()
                        except KeyError:
                            logger.warning(f"No Generators found in: {zone_input}")
                            break

                    if prop_name[1] == "generator_VO&M_Cost":
                        df[0].to_numpy()[df[0].to_numpy() < 0] = 0
                    df = df.rename(columns={0: prop_name[1]})
                    data_frames_lst.append(df)

                detailed_gen_cost = pd.concat(data_frames_lst, axis=1).fillna(0)
                detailed_gen_cost = detailed_gen_cost.rename(columns=
                                            {"generator_FO&M_Cost": "FO&M Cost",
                                             "generator_VO&M_Cost": "VO&M Cost",
                                             "generator_Fuel_Cost": "Fuel Cost",
                                             "generator_Start_&_Shutdown_Cost": "Start & Shutdown Cost",
                                             "generator_Reserves_VO&M_Cost": "Reserves VO&M Cost",
                                             "generator_Emissions_Cost": "Emissions Cost"})

                if pd.notna(start_date_range):
                    detailed_gen_cost = self.set_timestamp_date_range(
                                        detailed_gen_cost,
                                        start_date_range, end_date_range)
                    if detailed_gen_cost.empty is True:
                        logger.warning('No Generation in selected Date Range')
                        continue
                
                gen_cost_out_chunks.append(self.year_scenario_grouper(detailed_gen_cost, scenario, 
                                                        groupby=barplot_groupby).sum())
            
            # Checks if gen_cost_out_chunks contains data, if not skips zone and does not return a plot
            if not gen_cost_out_chunks:
                outputs[zone_input] = MissingZoneData()
                continue
            
            detailed_gen_cost_out = pd.concat(gen_cost_out_chunks, axis=0, sort=False)
            detailed_gen_cost_out = detailed_gen_cost_out/1000000 #Convert cost to millions
            #TODO: Add $ unit conversion.

            #Ensures region has generation, else skips
            try:
                #Change to a diff on first scenario
                scen_base = detailed_gen_cost_out.index[0]
                detailed_gen_cost_out = detailed_gen_cost_out - \
                    detailed_gen_cost_out.xs(scen_base) #Change to a diff on first scenario

            except KeyError:
                outputs[zone_input] = MissingZoneData()
                continue
            #Drop base entry
            detailed_gen_cost_out.drop(scen_base, inplace=True) 

            net_cost = detailed_gen_cost_out.sum(axis=1)

            # Deletes columns that are all 0
            detailed_gen_cost_out = detailed_gen_cost_out.loc[:, (detailed_gen_cost_out != 0).any(axis=0)]

            # Checks if detailed_gen_cost_out contains data, if not skips zone and does not return a plot
            if detailed_gen_cost_out.empty == True:
                outputs[zone_input] = MissingZoneData()
                continue
            
            # Data table of values to return to main program
            Data_Table_Out = detailed_gen_cost_out.add_suffix(" (Million $)")

            mplt = PlotLibrary()
            fig, ax = mplt.get_figure()

            mplt.barplot(detailed_gen_cost_out,
                         stacked=True)

            ax.axhline(y=0, linewidth=0.5, linestyle='--', color='grey')
            
            ax.set_ylabel(f'Generation Cost Change \n relative to {scen_base} (Million $)', 
                            color='black', rotation='vertical') #TODO: Add $ unit conversion.
            ax.margins(x=0.01)

            #Add net cost line.
            for n, scenario in enumerate(detailed_gen_cost_out.index.unique()):
                x = [ax.patches[n].get_x(), ax.patches[n].get_x() + ax.patches[n].get_width()]
                y_net = [net_cost.loc[scenario]] * 2
                ax.plot(x, y_net, c='black', linewidth=1.5,
                        label='Net Cost Change')

            mplt.add_legend()

            if plot_data_settings["plot_title_as_region"]:
                mplt.add_main_title(zone_input)

            outputs[zone_input] = {'fig': fig, 'data_table': Data_Table_Out}
        return outputs
