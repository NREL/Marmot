# -*- coding: utf-8 -*-
"""Generator reserve plots.

This module creates plots of reserve provision and shortage at the generation 
and region level.

@author: Daniel Levie
"""

import logging
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

import marmot.utils.mconfig as mconfig

from marmot.plottingmodules.plotutils.plot_library import SetupSubplot, PlotLibrary
from marmot.plottingmodules.plotutils.plot_data_helper import MPlotDataHelper
from marmot.plottingmodules.plotutils.plot_exceptions import (MissingInputData, MissingZoneData)

logger = logging.getLogger('plotter.'+__name__)
plot_data_settings = mconfig.parser("plot_data")

class Reserves(MPlotDataHelper):
    """Generator and system reserve plots.

    The reserves.py module contains methods that are
    related to reserve provision and shortage. 

    MPlot inherits from the MPlotDataHelper class to assist 
    in creating figures.
    """

    def __init__(self, **kwargs):
        # Instantiation of MPlotHelperFunctions
        super().__init__(**kwargs)
        
    def reserve_gen_timeseries(self, figure_name: str = None, prop: str = None,
                               start: float = None, end: float= None,
                               timezone: str = "", start_date_range: str = None,
                               end_date_range: str = None, 
                               data_resolution: str = "", **_):
        """Creates a generation timeseries stackplot of total cumulative reserve provision by tech type.
        
        The code will create either a facet plot or a single plot depending on 
        if the Facet argument is active.
        If a facet plot is created, each scenario is plotted on a separate facet, 
        otherwise all scenarios are plotted on a single plot.
        To make a facet plot, ensure the work 'Facet' is found in the figure_name.
        Generation order is determined by the ordered_gen_categories.csv.

        Args:
            figure_name (str, optional): User defined figure output name. Used here 
                to determine if a Facet plot should be created.
                Defaults to None.
            prop (str, optional): Special argument used to adjust specific 
                plot settings. Controlled through the plot_select.csv.
                Opinions available are:

                - Peak Demand
                
                Defaults to None.
            start (float, optional): Used in conjunction with the prop argument.
                Will define the number of days to plot before a certain event in 
                a timeseries plot, e.g Peak Demand.
                Defaults to None.
            end (float, optional): Used in conjunction with the prop argument.
                Will define the number of days to plot after a certain event in 
                a timeseries plot, e.g Peak Demand.
                Defaults to None.
            timezone (str, optional): The timezone to display on the x-axes.
                Defaults to "".
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
        properties = [(True,f"reserves_generators_Provision{data_resolution}",self.Scenarios)]
        
        # Runs get_formatted_data within MPlotDataHelper to populate MPlotDataHelper dictionary  
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)
        
        # Checks if all data required by plot is available, if 1 in list required data is missing
        if 1 in check_input_data:
            return MissingInputData()

        for region in self.Zones:
            logger.info(f"Zone = {region}")

            ncols, nrows = self.set_facet_col_row_dimensions(multi_scenario=self.Scenarios)
            grid_size = ncols*nrows
            excess_axs = grid_size - len(self.Scenarios)
            
            mplt = PlotLibrary(nrows, ncols, sharey=True, 
                                squeeze=False, ravel_axs=True)
            fig, axs = mplt.get_figure()
            plt.subplots_adjust(wspace=0.05, hspace=0.2)

            data_tables = []
            for n, scenario in enumerate(self.Scenarios):
                logger.info(f"Scenario = {scenario}")

                reserve_provision_timeseries = self[f"reserves_generators_Provision{data_resolution}"].get(scenario)
                
                #Check if zone has reserves, if not skips
                try:
                    reserve_provision_timeseries = reserve_provision_timeseries.xs(region,level=self.AGG_BY)
                except KeyError:
                    logger.info(f"No reserves deployed in: {scenario}")
                    continue
                reserve_provision_timeseries = self.df_process_gen_inputs(reserve_provision_timeseries)

                if pd.notna(start_date_range):
                    reserve_provision_timeseries = self.set_timestamp_date_range(
                                    reserve_provision_timeseries,
                                    start_date_range, end_date_range)
                    if reserve_provision_timeseries.empty is True:
                        logger.warning('No reserves in selected Date Range')
                        continue

                # unitconversion based off peak generation hour, only checked once 
                if n == 0:
                    unitconversion = self.capacity_energy_unitconversion(reserve_provision_timeseries,
                                                                            sum_values=True)
                reserve_provision_timeseries = reserve_provision_timeseries/unitconversion['divisor']

                # Adds property annotation
                if pd.notna(prop):
                    x_time_value = mplt.add_property_annotation(
                                            reserve_provision_timeseries, 
                                            prop,
                                            sub_pos=n,
                                            energy_unit=unitconversion['units'])

                    if x_time_value is not None and len(reserve_provision_timeseries) > 1:
                        # if timestamps are larger than hours time_delta will 
                        # be the length of the interval in days, else time_delta == 1 day
                        timestamps = reserve_provision_timeseries.index.unique()
                        time_delta = max(1, (timestamps[1] - timestamps[0])/np.timedelta64(1, 'D'))
                        end_date = x_time_value + dt.timedelta(days=end*time_delta)
                        start_date = x_time_value - dt.timedelta(days=start*time_delta)
                        reserve_provision_timeseries = reserve_provision_timeseries.loc[start_date : end_date]
                
                scenario_names = pd.Series([scenario] * len(reserve_provision_timeseries),name = 'Scenario')
                data_table = reserve_provision_timeseries.add_suffix(f" ({unitconversion['units']})")
                data_table = data_table.set_index([scenario_names],append = True)
                data_tables.append(data_table)
                
                mplt.stackplot(reserve_provision_timeseries, 
                               color_dict=self.PLEXOS_color_dict, 
                               labels=reserve_provision_timeseries.columns,
                               sub_pos=n)
                mplt.set_subplot_timeseries_format(sub_pos=n)

            if not data_tables:
                logger.warning(f'No reserves in {region}')
                out = MissingZoneData()
                outputs[region] = out
                continue
            
            # Add facet labels
            mplt.add_facet_labels(xlabels=self.xlabels,
                                  ylabels = self.ylabels)
            # Add legend
            mplt.add_legend(reverse_legend=True, sort_by=self.ordered_gen)
            #Remove extra axes
            mplt.remove_excess_axs(excess_axs,grid_size)
            if plot_data_settings["plot_title_as_region"]:
                mplt.add_main_title(region)
            plt.ylabel(f"Reserve Provision ({unitconversion['units']})", 
                        color='black', rotation='vertical', labelpad=40)

            data_table_out = pd.concat(data_tables)

            outputs[region] = {'fig': fig, 'data_table': data_table_out}
        return outputs

    def total_reserves_by_gen(self, start_date_range: str = None, 
                              end_date_range: str = None, 
                              barplot_groupby: str = 'Scenario', **_):
        """Creates a generation stacked barplot of total reserve provision by generator tech type.

        A separate bar is created for each scenario.

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
        properties = [(True,"reserves_generators_Provision",self.Scenarios)]
        
        # Runs get_formatted_data within MPlotDataHelper to populate MPlotDataHelper dictionary  
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)
        
        # Checks if all data required by plot is available, if 1 in list required data is missing
        if 1 in check_input_data:
            return MissingInputData()

        for region in self.Zones:
            logger.info(f"Zone = {region}")

            reserve_chunks = []
            for scenario in self.Scenarios:
                logger.info(f"Scenario = {scenario}")

                reserve_provision_timeseries = self["reserves_generators_Provision"].get(scenario)
                #Check if zone has reserves, if not skips
                try:
                    reserve_provision_timeseries = reserve_provision_timeseries.xs(region,level=self.AGG_BY)
                except KeyError:
                    logger.info(f"No reserves deployed in {scenario}")
                    continue
                reserve_provision_timeseries = self.df_process_gen_inputs(reserve_provision_timeseries)

                if pd.notna(start_date_range):
                    reserve_provision_timeseries = self.set_timestamp_date_range(
                                    reserve_provision_timeseries,
                                    start_date_range, end_date_range)
                    if reserve_provision_timeseries.empty is True:
                        logger.warning('No data in selected Date Range')
                        continue

                # Calculates interval step to correct for MWh of generation
                interval_count = self.get_sub_hour_interval_count(reserve_provision_timeseries)
                reserve_provision_timeseries = reserve_provision_timeseries/interval_count
                
                reserve_provision = self.year_scenario_grouper(
                                                    reserve_provision_timeseries, 
                                                    scenario, 
                                                    groupby=barplot_groupby).sum()
                
                reserve_chunks.append(reserve_provision)

            total_reserves_out = pd.concat(reserve_chunks, axis=0, sort=False).fillna(0)

            total_reserves_out = total_reserves_out.loc[:, (total_reserves_out != 0).any(axis=0)]
            
            if total_reserves_out.empty:
                out = MissingZoneData()
                outputs[region] = out
                continue
            
            # Convert units
            unitconversion = self.capacity_energy_unitconversion(total_reserves_out,
                                                                    sum_values=True)
            total_reserves_out = total_reserves_out/unitconversion['divisor']
            data_table_out = total_reserves_out.add_suffix(f" ({unitconversion['units']}h)")
            
            mplt = PlotLibrary()
            fig, ax = mplt.get_figure()
            # Set x-tick labels
            if self.custom_xticklabels:
                tick_labels = self.custom_xticklabels
            else:
                tick_labels = total_reserves_out.index

            mplt.barplot(total_reserves_out, color=self.PLEXOS_color_dict,
                         stacked=True,
                         custom_tick_labels=tick_labels)

            ax.set_ylabel(f"Total Reserve Provision ({unitconversion['units']}h)", 
                            color='black', rotation='vertical')
            # Add legend
            mplt.add_legend(reverse_legend=True, sort_by=self.ordered_gen)
            if plot_data_settings["plot_title_as_region"]:
                mplt.add_main_title(region)

            outputs[region] = {'fig': fig, 'data_table': data_table_out}
        return outputs

    def reg_reserve_shortage(self, **kwargs):
        """Creates a bar plot of reserve shortage for each region in MWh.
        
        Bars are grouped by reserve type, each scenario is plotted as a differnet color.

        The 'Shortage' argument is passed to the _reserve_bar_plots() method to 
        create this plot.

        Returns:
            dict: Dictionary containing the created plot and its data table.
        """
        outputs = self._reserve_bar_plots("Shortage", **kwargs)
        return outputs

    def reg_reserve_provision(self, **kwargs):
        """Creates a bar plot of reserve provision for each region in MWh.
        
        Bars are grouped by reserve type, each scenario is plotted as a differnet color.

        The 'Provision' argument is passed to the _reserve_bar_plots() method to 
        create this plot.

        Returns:
            dict: Dictionary containing the created plot and its data table.
        """
        outputs = self._reserve_bar_plots("Provision", **kwargs)
        return outputs

    def reg_reserve_shortage_hrs(self, **kwargs):
        """creates a bar plot of reserve shortage for each region in hrs.
        
        Bars are grouped by reserve type, each scenario is plotted as a differnet color.
        
        The 'Shortage' argument and count_hours=True is passed to the _reserve_bar_plots() method to 
        create this plot.

        Returns:
            dict: Dictionary containing the created plot and its data table.
        """
        outputs = self._reserve_bar_plots("Shortage", count_hours=True)
        return outputs

    def _reserve_bar_plots(self, data_set: str, count_hours: bool = False, 
                           start_date_range: str = None, 
                           end_date_range: str = None,
                           barplot_groupby: str = 'Scenario', **_):
        """internal _reserve_bar_plots method, creates 'Shortage', 'Provision' and 'Shortage' bar 
        plots

        Bars are grouped by reserve type, each scenario is plotted as a differnet color.

        Args:
            data_set (str): Identifies the reserve data set to use and pull
                from the formatted h5 file.
            count_hours (bool, optional): if True creates a 'Shortage' hours plot.
                Defaults to False.
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
        properties = [(True, f"reserve_{data_set}", self.Scenarios)]
        
        # Runs get_formatted_data within MPlotDataHelper to populate MPlotDataHelper dictionary  
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)
        
        # Checks if all data required by plot is available, if 1 in list required data is missing
        if 1 in check_input_data:
            return MissingInputData()

        for region in self.Zones:
            logger.info(f"Zone = {region}")

            Data_Table_Out=pd.DataFrame()
            reserve_total_chunk = []
            for scenario in self.Scenarios:

                logger.info(f'Scenario = {scenario}')

                reserve_timeseries = self[f"reserve_{data_set}"].get(scenario)
                # Check if zone has reserves, if not skips
                try:
                    reserve_timeseries = reserve_timeseries.xs(region,level=self.AGG_BY)
                except KeyError:
                    logger.info(f"No reserves deployed in {scenario}")
                    continue

                if pd.notna(start_date_range):
                    reserve_timeseries = self.set_timestamp_date_range(reserve_timeseries,
                                    start_date_range, end_date_range)
                    if reserve_timeseries.empty is True:
                        logger.warning('No data in selected Date Range')
                        continue

                reserve_timeseries = reserve_timeseries.reset_index(["timestamp","Type","parent"],drop=False)
                # Drop duplicates to remove double counting
                reserve_timeseries.drop_duplicates(inplace=True)
                # Set Type equal to parent value if Type equals '-'
                reserve_timeseries['Type'] = reserve_timeseries['Type'].mask(reserve_timeseries['Type'] == '-', 
                                                                        reserve_timeseries['parent'])
                reserve_timeseries.set_index(["timestamp","Type","parent"], append=True, inplace=True)

                interval_count = self.get_sub_hour_interval_count(reserve_timeseries)
                # Groupby Type
                if count_hours == False:
                    reserve_total = self.year_scenario_grouper(
                                        reserve_timeseries, scenario, 
                                        groupby=barplot_groupby,
                                        additional_groups=["Type"]).sum()/interval_count

                elif count_hours == True:
                    reserve_total = reserve_timeseries[reserve_timeseries[0]>0] #Filter for non zero values
                    reserve_total = self.year_scenario_grouper(
                                        reserve_timeseries, scenario, 
                                        groupby=barplot_groupby,
                                        additional_groups=["Type"]).count()/interval_count

                reserve_total_chunk.append(reserve_total)
            
            if reserve_total_chunk:
                reserve_out = pd.concat(reserve_total_chunk, axis=0, sort=False)
            else:
                reserve_out=pd.DataFrame()
            # If no reserves return nothing
            if reserve_out.empty:
                out = MissingZoneData()
                outputs[region] = out
                continue

            reserve_out = reserve_out.reset_index().pivot(index="Type", columns='Scenario', values=0)
            if count_hours == False:
                # Convert units
                unitconversion = self.capacity_energy_unitconversion(reserve_out)
                reserve_out = reserve_out/unitconversion['divisor'] 
                Data_Table_Out = reserve_out.add_suffix(f" ({unitconversion['units']}h)")
            else:
                Data_Table_Out = reserve_out.add_suffix(" (hrs)")
            
            # create color dictionary
            color_dict = dict(zip(reserve_out.columns,self.color_list))
            
            mplt = PlotLibrary()
            fig, ax = mplt.get_figure()

            mplt.barplot(reserve_out, color=color_dict,
                         stacked=False)

            if count_hours == False:
                ax.set_ylabel(f"Reserve {data_set} [{unitconversion['units']}h]", 
                               color='black', rotation='vertical')
            elif count_hours == True:
                mplt.set_yaxis_major_tick_format(decimal_accuracy=0)
                ax.set_ylabel(f"Reserve {data_set} Hours", 
                              color='black', rotation='vertical')
            mplt.add_legend()
            if plot_data_settings["plot_title_as_region"]:
                mplt.add_main_title(region)
            outputs[region] = {'fig': fig,'data_table': Data_Table_Out}
        return outputs

    def reg_reserve_shortage_timeseries(self, figure_name: str = None,
                                        timezone: str = "", start_date_range: str = None, 
                                        end_date_range: str = None, **_):
        """Creates a timeseries line plot of reserve shortage.

        A line is plotted for each reserve type shortage.

        The code will create either a facet plot or a single plot depending on 
        if the Facet argument is active.
        If a facet plot is created, each scenario is plotted on a separate facet, 
        otherwise all scenarios are plotted on a single plot.
        To make a facet plot, ensure the work 'Facet' is found in the figure_name.

        Args:
            figure_name (str, optional): User defined figure output name. Used here 
                to determine if a Facet plot should be created.
                Defaults to None.
            timezone (str, optional): The timezone to display on the x-axes.
                Defaults to "".
            start_date_range (str, optional): Defines a start date at which to represent data from. 
                Defaults to None.
            end_date_range (str, optional): Defines a end date at which to represent data to.
                Defaults to None.

        Returns:
            dict: Dictionary containing the created plot and its data table.
        """
        facet=False
        if 'Facet' in figure_name:
            facet = True
        
        # If not facet plot, only plot first scenario
        if not facet:
            Scenarios = [self.Scenarios[0]]
        else:
            Scenarios = self.Scenarios

        outputs : dict = {}
        
        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True, "reserve_Shortage", Scenarios)]
        
        # Runs get_formatted_data within MPlotDataHelper to populate MPlotDataHelper dictionary  
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)
        
        # Checks if all data required by plot is available, if 1 in list required data is missing
        if 1 in check_input_data:
            return MissingInputData()
        
        for region in self.Zones:
            logger.info(f"Zone = {region}")

            ncols, nrows = self.set_facet_col_row_dimensions(facet, multi_scenario=Scenarios)
            grid_size = ncols*nrows
            excess_axs = grid_size - len(Scenarios)

            mplt = SetupSubplot(nrows, ncols, sharey=True, 
                                squeeze=False, ravel_axs=True)
            fig, axs = mplt.get_figure()
            plt.subplots_adjust(wspace=0.05, hspace=0.2)

            data_tables = []

            for n, scenario in enumerate(Scenarios):

                logger.info(f'Scenario = {scenario}')

                reserve_timeseries = self["reserve_Shortage"].get(scenario)
                # Check if zone has reserves, if not skips
                try:
                    reserve_timeseries = reserve_timeseries.xs(region,level=self.AGG_BY)
                except KeyError:
                    logger.info(f"No reserves deployed in {scenario}")
                    continue
                
                reserve_timeseries.reset_index(["timestamp","Type","parent"],drop=False,inplace=True)
                reserve_timeseries = reserve_timeseries.drop_duplicates()
                # Set Type equal to parent value if Type equals '-'
                reserve_timeseries['Type'] = reserve_timeseries['Type'].mask(reserve_timeseries['Type'] == '-', 
                                                                             reserve_timeseries['parent'])
                reserve_timeseries = reserve_timeseries.pivot(index='timestamp', columns='Type', values=0)

                if pd.notna(start_date_range):
                    reserve_timeseries = self.set_timestamp_date_range(reserve_timeseries,
                                    start_date_range, end_date_range)
                    if reserve_timeseries.empty is True:
                        logger.warning('No data in selected Date Range')
                        continue

                # create color dictionary
                color_dict = dict(zip(reserve_timeseries.columns,self.color_list))

                scenario_names = pd.Series([scenario] * len(reserve_timeseries),name = 'Scenario')
                data_table = reserve_timeseries.add_suffix(" (MW)")
                data_table = data_table.set_index([scenario_names],append = True)
                data_tables.append(data_table)

                for column in reserve_timeseries:
                    axs[n].plot(reserve_timeseries.index.values, reserve_timeseries[column], 
                                linewidth=2,
                                color=color_dict[column],
                                label=column)

                mplt.set_yaxis_major_tick_format(sub_pos=n)
                axs[n].margins(x=0.01)
                mplt.set_subplot_timeseries_format(sub_pos=n)

            if not data_tables:
                out = MissingZoneData()
                outputs[region] = out
                continue
            
            # add facet labels
            mplt.add_facet_labels(xlabels=self.xlabels,
                                  ylabels = self.ylabels)
            mplt.add_legend()
            #Remove extra axes
            mplt.remove_excess_axs(excess_axs,grid_size)
            plt.ylabel('Reserve Shortage [MW]',  color='black', 
                       rotation='vertical',labelpad = 40)
            if plot_data_settings["plot_title_as_region"]:
               mplt.add_main_title(region)
            
            data_table_out = pd.concat(data_tables)
            
            outputs[region] =  {'fig': fig, 'data_table': data_table_out}

        return outputs