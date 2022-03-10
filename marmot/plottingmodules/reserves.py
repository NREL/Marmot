# -*- coding: utf-8 -*-
"""Generator reserve plots.

This module creates plots of reserve provision and shortage at the generation 
and region level.

@author: Daniel Levie
"""

import logging
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

import marmot.utils.mconfig as mconfig

from marmot.plottingmodules.plotutils.plot_library import SetupSubplot, PlotLibrary
from marmot.plottingmodules.plotutils.plot_data_helper import PlotDataHelper
from marmot.plottingmodules.plotutils.plot_exceptions import (MissingInputData, MissingZoneData)


class MPlot(PlotDataHelper):
    """reserves MPlot class.

    All the plotting modules use this same class name.
    This class contains plotting methods that are grouped based on the
    current module name.
    
    The reserves.py module contains methods that are
    related to reserve provision and shortage. 

    MPlot inherits from the PlotDataHelper class to assist in creating figures.
    """

    def __init__(self, argument_dict: dict):
        """
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
                    self.xlabels, self.gen_names_dict, self.TECH_SUBSET, 
                    Region_Mapping=self.Region_Mapping) 

        self.logger = logging.getLogger('marmot_plot.'+__name__)
        
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
                - Date Range
                
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
        # If not facet plot, only plot first scenario
        facet=False
        if 'Facet' in figure_name:
            facet = True
            
        if not facet:
            Scenarios = [self.Scenarios[0]]
        else:
            Scenarios = self.Scenarios
            
        outputs = {}
        
        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True,f"reserves_generators_Provision{data_resolution}",self.Scenarios)]
        
        # Runs get_formatted_data within PlotDataHelper to populate PlotDataHelper dictionary  
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)
        
        # Checks if all data required by plot is available, if 1 in list required data is missing
        if 1 in check_input_data:
            return MissingInputData()

        for region in self.Zones:
            self.logger.info(f"Zone = {region}")

            ncols, nrows = self.set_facet_col_row_dimensions(facet,multi_scenario=Scenarios)
            grid_size = ncols*nrows
            excess_axs = grid_size - len(Scenarios)
            
            mplt = PlotLibrary(nrows, ncols, sharey=True, 
                                squeeze=False, ravel_axs=True)
            fig, axs = mplt.get_figure()
            plt.subplots_adjust(wspace=0.05, hspace=0.2)

            data_tables = []
            for n, scenario in enumerate(Scenarios):
                self.logger.info(f"Scenario = {scenario}")

                reserve_provision_timeseries = self[f"reserves_generators_Provision{data_resolution}"].get(scenario)
                
                #Check if zone has reserves, if not skips
                try:
                    reserve_provision_timeseries = reserve_provision_timeseries.xs(region,level=self.AGG_BY)
                except KeyError:
                    self.logger.info(f"No reserves deployed in: {scenario}")
                    continue
                reserve_provision_timeseries = self.df_process_gen_inputs(reserve_provision_timeseries)

                if reserve_provision_timeseries.empty is True:
                    self.logger.info(f"No reserves deployed in: {scenario}")
                    continue
                # unitconversion based off peak generation hour, only checked once 
                if n == 0:
                    unitconversion = self.capacity_energy_unitconversion(reserve_provision_timeseries,
                                                                            sum_values=True)

                if prop == "Peak Demand":
                    self.logger.info("Plotting Peak Demand period")

                    total_reserve = reserve_provision_timeseries.sum(axis=1)/unitconversion['divisor']
                    peak_reserve_t =  total_reserve.idxmax()
                    start_date = peak_reserve_t - dt.timedelta(days=start)
                    end_date = peak_reserve_t + dt.timedelta(days=end)
                    reserve_provision_timeseries = reserve_provision_timeseries[start_date : end_date]
                    Peak_Reserve = total_reserve[peak_reserve_t]

                elif prop == 'Date Range':
                    self.logger.info(f"Plotting specific date range: \
                        {str(start_date_range)} to {str(end_date_range)}")
                    reserve_provision_timeseries = reserve_provision_timeseries[start_date_range : end_date_range]
                else:
                    self.logger.info("Plotting graph for entire timeperiod")
                
                reserve_provision_timeseries = reserve_provision_timeseries/unitconversion['divisor']
                
                scenario_names = pd.Series([scenario] * len(reserve_provision_timeseries),name = 'Scenario')
                data_table = reserve_provision_timeseries.add_suffix(f" ({unitconversion['units']})")
                data_table = data_table.set_index([scenario_names],append = True)
                data_tables.append(data_table)
                
                mplt.stackplot(reserve_provision_timeseries, 
                               color_dict=self.PLEXOS_color_dict, 
                               labels=reserve_provision_timeseries.columns,
                               sub_pos=n)
                mplt.set_subplot_timeseries_format(sub_pos=n)

                if prop == "Peak Demand":
                    axs[n].annotate('Peak Reserve: \n' + str(format(int(Peak_Reserve), '.2f')) + ' {}'.format(unitconversion['units']), 
                                    xy=(peak_reserve_t, Peak_Reserve),
                            xytext=((peak_reserve_t + dt.timedelta(days=0.25)), (Peak_Reserve + Peak_Reserve*0.05)),
                            fontsize=13, arrowprops=dict(facecolor='black', width=3, shrink=0.1))

            if not data_tables:
                self.logger.warning(f'No reserves in {region}')
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
            if mconfig.parser("plot_title_as_region"):
                mplt.add_main_title(region)
            plt.ylabel(f"Reserve Provision ({unitconversion['units']})", 
                        color='black', rotation='vertical', labelpad=40)

            data_table_out = pd.concat(data_tables)

            outputs[region] = {'fig': fig, 'data_table': data_table_out}
        return outputs

    def total_reserves_by_gen(self, start_date_range: str = None, 
                              end_date_range: str = None, **_):
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
        outputs = {}
        
        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True,"reserves_generators_Provision",self.Scenarios)]
        
        # Runs get_formatted_data within PlotDataHelper to populate PlotDataHelper dictionary  
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)
        
        # Checks if all data required by plot is available, if 1 in list required data is missing
        if 1 in check_input_data:
            return MissingInputData()

        for region in self.Zones:
            self.logger.info(f"Zone = {region}")

            Total_Reserves_Out = pd.DataFrame()
            for scenario in self.Scenarios:
                self.logger.info(f"Scenario = {scenario}")

                reserve_provision_timeseries = self["reserves_generators_Provision"].get(scenario)
                #Check if zone has reserves, if not skips
                try:
                    reserve_provision_timeseries = reserve_provision_timeseries.xs(region,level=self.AGG_BY)
                except KeyError:
                    self.logger.info(f"No reserves deployed in {scenario}")
                    continue
                reserve_provision_timeseries = self.df_process_gen_inputs(reserve_provision_timeseries)

                if reserve_provision_timeseries.empty is True:
                    self.logger.info(f"No reserves deployed in: {scenario}")
                    continue

                # Calculates interval step to correct for MWh of generation
                interval_count = self.get_sub_hour_interval_count(reserve_provision_timeseries)

                # sum totals by fuel types
                reserve_provision_timeseries = reserve_provision_timeseries/interval_count
                reserve_provision = reserve_provision_timeseries.sum(axis=0)
                reserve_provision.rename(scenario, inplace=True)
                Total_Reserves_Out = pd.concat([Total_Reserves_Out, reserve_provision], axis=1, sort=False).fillna(0)


            Total_Reserves_Out = self.create_categorical_tech_index(Total_Reserves_Out)
            Total_Reserves_Out = Total_Reserves_Out.T
            Total_Reserves_Out = Total_Reserves_Out.loc[:, (Total_Reserves_Out != 0).any(axis=0)]
            
            if Total_Reserves_Out.empty:
                out = MissingZoneData()
                outputs[region] = out
                continue
            
            Total_Reserves_Out.index = Total_Reserves_Out.index.str.replace('_',' ')
            Total_Reserves_Out.index = Total_Reserves_Out.index.str.wrap(5, break_long_words=False)
            
            # Convert units
            unitconversion = self.capacity_energy_unitconversion(Total_Reserves_Out,
                                                                    sum_values=True)
            Total_Reserves_Out = Total_Reserves_Out/unitconversion['divisor']
            
            data_table_out = Total_Reserves_Out.add_suffix(f" ({unitconversion['units']}h)")
            
            mplt = PlotLibrary()
            fig, ax = mplt.get_figure()

            # Set x-tick labels
            if self.custom_xticklabels:
                tick_labels = self.custom_xticklabels
            else:
                tick_labels = Total_Reserves_Out.index

            mplt.barplot(Total_Reserves_Out, color=self.PLEXOS_color_dict,
                         stacked=True,
                         custom_tick_labels=tick_labels)

            ax.set_ylabel(f"Total Reserve Provision ({unitconversion['units']}h)", 
                            color='black', rotation='vertical')
            # Add legend
            mplt.add_legend(reverse_legend=True, sort_by=self.ordered_gen)
            if mconfig.parser("plot_title_as_region"):
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
                           end_date_range: str = None, **_):
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
        outputs = {}
        
        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True, f"reserve_{data_set}", self.Scenarios)]
        
        # Runs get_formatted_data within PlotDataHelper to populate PlotDataHelper dictionary  
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)
        
        # Checks if all data required by plot is available, if 1 in list required data is missing
        if 1 in check_input_data:
            return MissingInputData()

        for region in self.Zones:
            self.logger.info(f"Zone = {region}")

            Data_Table_Out=pd.DataFrame()
            reserve_total_chunk = []
            for scenario in self.Scenarios:

                self.logger.info(f'Scenario = {scenario}')

                reserve_timeseries = self[f"reserve_{data_set}"].get(scenario)
                # Check if zone has reserves, if not skips
                try:
                    reserve_timeseries = reserve_timeseries.xs(region,level=self.AGG_BY)
                except KeyError:
                    self.logger.info(f"No reserves deployed in {scenario}")
                    continue

                interval_count = self.get_sub_hour_interval_count(reserve_timeseries)

                reserve_timeseries = reserve_timeseries.reset_index(["timestamp","Type","parent"],drop=False)
                # Drop duplicates to remove double counting
                reserve_timeseries.drop_duplicates(inplace=True)
                # Set Type equal to parent value if Type equals '-'
                reserve_timeseries['Type'] = reserve_timeseries['Type'].mask(reserve_timeseries['Type'] == '-', reserve_timeseries['parent'])
                reserve_timeseries.set_index(["timestamp","Type","parent"],append=True,inplace=True)

                # Groupby Type
                if count_hours == False:
                    reserve_total = reserve_timeseries.groupby(["Type"]).sum()/interval_count
                elif count_hours == True:
                    reserve_total = reserve_timeseries[reserve_timeseries[0]>0] #Filter for non zero values
                    reserve_total = reserve_total.groupby("Type").count()/interval_count

                reserve_total.rename(columns={0:scenario},inplace=True)

                reserve_total_chunk.append(reserve_total)
            
            if reserve_total_chunk:
                reserve_out = pd.concat(reserve_total_chunk,axis=1, sort='False')
                reserve_out.columns = reserve_out.columns.str.replace('_',' ')
            else:
                reserve_out=pd.DataFrame()
            # If no reserves return nothing
            if reserve_out.empty:
                out = MissingZoneData()
                outputs[region] = out
                continue
            
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
            if mconfig.parser("plot_title_as_region"):
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

        outputs = {}
        
        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True, "reserve_Shortage", Scenarios)]
        
        # Runs get_formatted_data within PlotDataHelper to populate PlotDataHelper dictionary  
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)
        
        # Checks if all data required by plot is available, if 1 in list required data is missing
        if 1 in check_input_data:
            return MissingInputData()
        
        for region in self.Zones:
            self.logger.info(f"Zone = {region}")

            ncols, nrows = self.set_facet_col_row_dimensions(facet, multi_scenario=Scenarios)
            grid_size = ncols*nrows
            excess_axs = grid_size - len(Scenarios)

            mplt = SetupSubplot(nrows, ncols, sharey=True, 
                                squeeze=False, ravel_axs=True)
            fig, axs = mplt.get_figure()
            plt.subplots_adjust(wspace=0.05, hspace=0.2)

            data_tables = []

            for n, scenario in enumerate(Scenarios):

                self.logger.info(f'Scenario = {scenario}')

                reserve_timeseries = self["reserve_Shortage"].get(scenario)
                # Check if zone has reserves, if not skips
                try:
                    reserve_timeseries = reserve_timeseries.xs(region,level=self.AGG_BY)
                except KeyError:
                    self.logger.info(f"No reserves deployed in {scenario}")
                    continue
                
                reserve_timeseries.reset_index(["timestamp","Type","parent"],drop=False,inplace=True)
                reserve_timeseries = reserve_timeseries.drop_duplicates()
                # Set Type equal to parent value if Type equals '-'
                reserve_timeseries['Type'] = reserve_timeseries['Type'].mask(reserve_timeseries['Type'] == '-', 
                                                                             reserve_timeseries['parent'])
                reserve_timeseries = reserve_timeseries.pivot(index='timestamp', columns='Type', values=0)

                if pd.notna(start_date_range):
                    self.logger.info(f"Plotting specific date range: \
                    {str(start_date_range)} to {str(end_date_range)}")
                    reserve_timeseries = reserve_timeseries[start_date_range : end_date_range]
                else:
                    self.logger.info("Plotting graph for entire timeperiod")

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
            if mconfig.parser("plot_title_as_region"):
               mplt.add_main_title(region)
            
            data_table_out = pd.concat(data_tables)
            
            outputs[region] =  {'fig': fig, 'data_table': data_table_out}

        return outputs