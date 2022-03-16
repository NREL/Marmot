# -*- coding: utf-8 -*-
"""System transmission plots.

This code creates transmission line and interface plots.

@author: Daniel Levie, Marty Schwarz
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.dates as mdates

import marmot.utils.mconfig as mconfig

from marmot.plottingmodules.plotutils.plot_library import PlotLibrary
from marmot.plottingmodules.plotutils.plot_data_helper import PlotDataHelper
from marmot.plottingmodules.plotutils.plot_exceptions import (MissingInputData, DataSavedInModule,
            UnderDevelopment, InputSheetError, MissingMetaData, UnsupportedAggregation, MissingZoneData)

plot_data_settings = mconfig.parser("plot_data")

class MPlot(PlotDataHelper):
    """transmission MPlot class.

    All the plotting modules use this same class name.
    This class contains plotting methods that are grouped based on the
    current module name.
    
    The transmission.py module contains methods that are
    related to the transmission network. 
    
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

        self.logger = logging.getLogger('plotter.'+__name__)
        self.font_defaults = mconfig.parser("font_settings")
        

    def line_util(self, **kwargs):
        """Creates a timeseries line plot of transmission lineflow utilization for each region.

        Utilization is plotted between 0 and 1 on the y-axis.
        The plot will default to showing the 10 highest utilized lines. A Line category 
        can also be passed instead, using the property field in the Marmot_plot_select.csv
        Each scenarios is plotted on a separate Facet plot.

        This methods calls _util() to create the figure.

        Returns:
            dict: Dictionary containing the created plot and its data table.
        """
        outputs = self._util(**kwargs)
        return outputs

    def line_hist(self, **kwargs):
        """Creates a histogram of transmission lineflow utilization for each region.

        Utilization is plotted between 0 and 1 on the x-axis, with # lines on the y-axis.
        Each bar is equal to a 0.05 utilization rate
        The plot will default to showing all lines. A Line category can also be passed
        instead using the property field in the Marmot_plot_select.csv
        Each scenarios is plotted on a separate Facet plot.

        This methods calls _util() and passes the hist=True argument to create the figure.

        Returns:
            dict: Dictionary containing the created plot and its data table.
        """
        outputs = self._util(hist=True, **kwargs)
        return outputs

    def _util(self, hist: bool = False, prop: str = None, 
              start_date_range: str = None, 
              end_date_range: str = None, **_):
        """Creates utilization plots, line plot and histograms

        This methods is called from line_util() and line_hist()

        Args:
            hist (bool, optional): If True creates a histogram of utilization. 
                Defaults to False.
            prop (str, optional): Optional PLEXOS line category to display.
                Defaults to None.
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
        properties = [(True,"line_Flow",self.Scenarios),
                      (True,"line_Import_Limit",self.Scenarios)]
        
        # Runs get_formatted_data within PlotDataHelper to populate PlotDataHelper dictionary  
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)

        if 1 in check_input_data:
            return MissingInputData()

        # sets up x, y dimensions of plot
        ncols, nrows = self.set_facet_col_row_dimensions(facet=True, 
                                                                multi_scenario=self.Scenarios)
        grid_size = ncols*nrows

        # Used to calculate any excess axis to delete
        plot_number = len(self.Scenarios)
        excess_axs = grid_size - plot_number

        for zone_input in self.Zones:
            self.logger.info(f"For all lines touching Zone = {zone_input}")

            mplt = PlotLibrary(nrows, ncols, sharey=True, 
                                squeeze=False, ravel_axs=True)
            fig, axs = mplt.get_figure()
            plt.subplots_adjust(wspace=0.1, hspace=0.25)

            data_table=[]

            for n, scenario in enumerate(self.Scenarios):
                self.logger.info(f"Scenario = {str(scenario)}")
                # gets correct metadata based on area aggregation
                if self.AGG_BY=='zone':
                    zone_lines = self.meta.zone_lines(scenario)
                else:
                    zone_lines = self.meta.region_lines(scenario)
                try:
                    zone_lines = zone_lines.set_index([self.AGG_BY])
                except:
                    self.logger.warning("Column to Aggregate by is missing")
                    continue

                try:
                    zone_lines = zone_lines.xs(zone_input)
                    zone_lines=zone_lines['line_name'].unique()
                except KeyError:
                    self.logger.warning('No data to plot for scenario')
                    outputs[zone_input] = MissingZoneData()
                    continue

                flow = self["line_Flow"].get(scenario).copy()
                #Limit to only lines touching to this zone
                flow = flow[flow.index.get_level_values('line_name').isin(zone_lines)] 

                if self.shift_leapday == True:
                    flow = self.adjust_for_leapday(flow)
                limits = self["line_Import_Limit"].get(scenario).copy()
                limits = limits.droplevel('timestamp').drop_duplicates()

                limits.mask(limits[0]==0.0,other=0.01,inplace=True) #if limit is zero set to small value

                # This checks for a nan in string. If no scenario selected, do nothing.
                if pd.notna(prop):
                    self.logger.info(f"Line category = {str(prop)}")
                    line_relations = self.meta.lines(scenario).rename(columns={"name":"line_name"}).set_index(["line_name"])
                    flow=pd.merge(flow,line_relations, left_index=True, 
                                  right_index=True)
                    flow=flow[flow["category"] == prop]
                    flow=flow.drop('category',axis=1)

                flow = pd.merge(flow,limits[0].abs(),on = 'line_name',how='left')
                flow['Util']=(flow['0_x'].abs()/flow['0_y']).fillna(0)
                #If greater than 1 because exceeds flow limit, report as 1
                flow['Util'][flow['Util'] > 1] = 1
                annual_util=flow['Util'].groupby(["line_name"]).mean().rename(scenario)
                # top annual utilized lines
                top_utilization = annual_util.nlargest(10, keep='first')

                color_dict = dict(zip(self.Scenarios,self.color_list))
                if hist == True:
                    mplt.histogram(annual_util, color_dict,label=scenario, sub_pos=n)
                else:
                    for line in top_utilization.index.get_level_values(level='line_name').unique():
                        duration_curve = flow.loc[line].sort_values(by='Util', 
                                                                    ascending=False).reset_index(drop=True)
                        mplt.lineplot(duration_curve, 'Util' ,label=line, sub_pos=n)
                        axs[n].set_ylim((0,1.1))
                data_table.append(annual_util)

            mplt.add_legend()
            #Remove extra axes
            mplt.remove_excess_axs(excess_axs,grid_size)
            # add facet labels
            mplt.add_facet_labels(xlabels=self.xlabels,
                                  ylabels = self.ylabels)
            if hist == True:
                if pd.notna(prop):
                    prop_name = 'All Lines'
                else:
                    prop_name = prop
                plt.ylabel('Number of lines',  color='black', 
                           rotation='vertical', labelpad=30)
                plt.xlabel(f'Line Utilization: {prop_name}',  color='black', 
                           rotation='horizontal', labelpad=30)
            else:
                if pd.notna(prop):
                    prop_name ='Top 10 Lines'
                else:
                    prop_name = prop
                plt.ylabel(f'Line Utilization: {prop_name}', color='black', 
                           rotation='vertical', labelpad=60)
                plt.xlabel('Intervals',  color='black', 
                           rotation='horizontal', labelpad=20)
            if plot_data_settings["plot_title_as_region"]:
                mplt.add_main_title(zone_input)
            try:
                del annual_util, 
            except:
                continue

            Data_Out = pd.concat(data_table)

            outputs[zone_input] = {'fig': fig,'data_table':Data_Out}
        return outputs

    def int_flow_ind(self, figure_name: str = None, prop: str = None, 
                     start_date_range: str = None, 
                     end_date_range: str = None, **_):
        """Creates a line plot of interchange flows and their import and export limits.

        Each interchange is potted on a separate facet plot.
        The plot includes every interchange that originates or ends in the aggregation zone.
        This can be adjusted by passing a comma separated string of interchanges to the property input.

        The code will create either a timeseries or duration curve depending on 
        if the word 'duration_curve' is in the figure_name.
        To make a duration curve, ensure the word 'duration_curve' is found in the figure_name.

        Args:
            figure_name (str, optional): User defined figure output name.
                Defaults to None.
            prop (str, optional): Comma separated string of interchanges. 
                Defaults to None.
            start_date_range (str, optional): Defines a start date at which to represent data from. 
                Defaults to None.
            end_date_range (str, optional): Defines a end date at which to represent data to.
                Defaults to None.

        Returns:
            dict: dictionary containing the created plot and its data table
        """
        duration_curve=False
        if 'duration_curve' in figure_name:
            duration_curve = True
        
        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True,"interface_Flow",self.Scenarios),
                      (True,"interface_Import_Limit",self.Scenarios),
                      (True,"interface_Export_Limit",self.Scenarios)]
        
        # Runs get_formatted_data within PlotDataHelper to populate PlotDataHelper dictionary  
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)

        if 1 in check_input_data:
            return MissingInputData()

        scenario = self.Scenarios[0]

        outputs = {}

        if pd.notna(start_date_range):
            self.logger.info(f"Plotting specific date range: \
                {str(start_date_range)} to {str(end_date_range)}")

        for zone_input in self.Zones:
            self.logger.info(f"For all interfaces touching Zone = {zone_input}")

            Data_Table_Out = pd.DataFrame()

            # gets correct metadata based on area aggregation
            if self.AGG_BY=='zone':
                zone_lines = self.meta.zone_lines(scenario)
            else:
                zone_lines = self.meta.region_lines(scenario)
            try:
                zone_lines = zone_lines.set_index([self.AGG_BY])
            except:
                self.logger.info("Column to Aggregate by is missing")
                continue

            zone_lines = zone_lines.xs(zone_input)
            zone_lines = zone_lines['line_name'].unique()

            #Map lines to interfaces
            all_ints = self.meta.interface_lines(scenario) #Map lines to interfaces
            all_ints.index = all_ints.line
            ints = all_ints.loc[all_ints.index.intersection(zone_lines)]

            #flow = flow[flow.index.get_level_values('interface_name').isin(ints.interface)] #Limit to only interfaces touching to this zone
            #flow = flow.droplevel('interface_category')

            export_limits = self["interface_Export_Limit"].get(scenario).copy().droplevel('timestamp')
            export_limits.mask(export_limits[0]==0.0,other=0.01,inplace=True) #if limit is zero set to small value
            export_limits = export_limits[export_limits.index.get_level_values('interface_name').isin(ints.interface)]
            export_limits = export_limits[export_limits[0].abs() < 99998] #Filter out unenforced interfaces.

            #Drop unnecessary columns.
            export_limits.reset_index(inplace = True)
            export_limits.drop(columns=['interface_category', 'units'], inplace=True)
            export_limits.set_index('interface_name',inplace = True)

            import_limits = self["interface_Import_Limit"].get(scenario).copy().droplevel('timestamp')
            import_limits.mask(import_limits[0]==0.0,other=0.01,inplace=True) #if limit is zero set to small value
            import_limits = import_limits[import_limits.index.get_level_values('interface_name').isin(ints.interface)]
            import_limits = import_limits[import_limits[0].abs() < 99998] #Filter out unenforced interfaces.
            reported_ints = import_limits.index.get_level_values('interface_name').unique()

            #Drop unnecessary columns.
            import_limits.reset_index(inplace = True)
            import_limits.drop(columns=['interface_category', 'units'], inplace=True)
            import_limits.set_index('interface_name',inplace = True)

            #Extract time index
            ti = self["interface_Flow"][self.Scenarios[0]].index.get_level_values('timestamp').unique()

            if pd.notna(prop):
                interf_list = prop.split(',')
                self.logger.info('Plotting only interfaces specified in Marmot_plot_select.csv')
                self.logger.info(interf_list)
            else:
                interf_list = reported_ints.copy()

            self.logger.info('Plotting full time series results.')
            xdim,ydim = self.set_x_y_dimension(len(interf_list))
            
            mplt = PlotLibrary(ydim, xdim, squeeze=False,
                               ravel_axs=True)
            fig, axs = mplt.get_figure()

            grid_size = xdim * ydim
            excess_axs = grid_size - len(interf_list)
            plt.subplots_adjust(wspace=0.05, hspace=0.2)
            missing_ints = 0
            chunks = []
            n = -1
            for interf in interf_list:
                n += 1

                #Remove leading spaces
                if interf[0] == ' ':
                    interf = interf[1:]
                if interf in reported_ints:
                    chunks_interf = []
                    single_exp_lim = export_limits.loc[interf] / 1000 #TODO: Use auto unit converter 
                    single_imp_lim = import_limits.loc[interf] / 1000

                    #Check if all hours have the same limit.
                    check = single_exp_lim.to_numpy()
                    identical = check[0] == check.all()

                    limits = pd.concat([single_exp_lim,single_imp_lim],axis = 1)
                    limits.columns = ['export limit','import limit']

                    limits.index = ti

                    for scenario in self.Scenarios:
                        flow = self["interface_Flow"].get(scenario)
                        single_int = flow.xs(interf, level='interface_name') / 1000
                        single_int.index = single_int.index.droplevel(['interface_category','units'])
                        single_int.columns = [interf]
                        single_int = single_int.reset_index().set_index('timestamp')
                        limits = limits.reset_index().set_index('timestamp')

                        if self.shift_leapday == True:
                            single_int = self.adjust_for_leapday(single_int)
                        if pd.notna(start_date_range):
                            single_int = single_int[start_date_range : end_date_range]
                            limits = limits[start_date_range : end_date_range]
                        if duration_curve:
                            single_int = self.sort_duration(single_int,interf)

                        mplt.lineplot(single_int, interf, 
                                                label=f"{scenario}\n interface flow",
                                                sub_pos=n)

                        # Only print limits if it doesn't change monthly or if you are plotting a time series. 
                        # Otherwise the limit lines could be misleading.
                        if not duration_curve or identical[0]:
                            if scenario == self.Scenarios[-1]:
                                #Only plot limits for last scenario.
                                limits_color_dict = {'export limit': 'red', 'import limit': 'green'}
                                mplt.lineplot(limits, 'export limit',
                                                         label='export limit', color=limits_color_dict,
                                                         linestyle='--', sub_pos=n)
                                mplt.lineplot(limits, 'import limit', 
                                                         label='import limit', color=limits_color_dict,
                                                         linestyle='--', sub_pos=n)

                        #For output time series .csv
                        scenario_names = pd.Series([scenario] * len(single_int), name='Scenario')
                        single_int_out = single_int.set_index([scenario_names], append=True)
                        chunks_interf.append(single_int_out)

                    Data_out_line = pd.concat(chunks_interf,axis = 0)
                    Data_out_line.columns = [interf]
                    chunks.append(Data_out_line)

                else:
                    self.logger.warning(f"{interf} not found in results. Have you tagged "
                                        "it with the 'Must Report' property in PLEXOS?")
                    excess_axs += 1
                    missing_ints += 1
                    continue

                axs[n].set_title(interf)
                if not duration_curve:
                    mplt.set_subplot_timeseries_format(sub_pos=n)
                if missing_ints == len(interf_list):
                    outputs = MissingInputData()
                    return outputs

            Data_Table_Out = pd.concat(chunks,axis = 1)
            Data_Table_Out = Data_Table_Out.reset_index()
            index_name = 'level_0' if duration_curve else 'timestamp'
            Data_Table_Out = Data_Table_Out.pivot(index = index_name,columns = 'Scenario')
            #Limits_Out = pd.concat(limits_chunks,axis = 1)
            #Limits_Out.index = ['Export Limit','Import Limit']

            # Data_Table_Out = Data_Table_Out.reset_index()
            # Data_Table_Out = Data_Table_Out.groupby(Data_Table_Out.index // 24).mean()
            # Data_Table_Out.index = pd.date_range(start = '1/1/2024',end = '12/31/2024',freq = 'D')
            mplt.add_legend()
            plt.ylabel('Flow (GW)',  color='black', rotation='vertical', 
                       labelpad=30)
            if duration_curve:
                plt.xlabel('Sorted hour of the year', color='black', labelpad=30)
            plt.tight_layout(rect=[0, 0.03, 1, 0.97])
            if plot_data_settings["plot_title_as_region"]:
                mplt.add_main_title(zone_input)
            outputs[zone_input] = {'fig': fig, 'data_table': Data_Table_Out}
            #Limits_Out.to_csv(os.path.join(self.Marmot_Solutions_folder, 'Figures_Output',self.AGG_BY + '_transmission','Individual_Interface_Limits.csv'))
        return outputs 

    def int_flow_ind_seasonal(self, figure_name: str = None, prop: str = None, 
                              start_date_range: str = None, 
                              end_date_range: str = None, **_):
        """#TODO: Finish Docstring 

        Args:
            figure_name (str, optional): User defined figure output name.
                Defaults to None.
            prop (str, optional): Comma separated string of interchanges. 
                Defaults to None.
            start_date_range (str, optional): Defines a start date at which to represent data from. 
                Defaults to None.
            end_date_range (str, optional): Defines a end date at which to represent data to.
                Defaults to None.

        Returns:
            dict: dictionary containing the created plot and its data table
        """
        #TODO: Use auto unit converter in method
        
        duration_curve=False
        if 'duration_curve' in figure_name:
            duration_curve = True
        
        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True,"interface_Flow",self.Scenarios),
                      (True,"interface_Import_Limit",self.Scenarios),
                      (True,"interface_Export_Limit",self.Scenarios)]
        
        # Runs get_formatted_data within PlotDataHelper to populate PlotDataHelper dictionary  
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)
        
        if 1 in check_input_data:
            return MissingInputData()

        scenario = self.Scenarios[0]

        outputs = {}
        for zone_input in self.Zones:
            self.logger.info("For all interfaces touching Zone = "+zone_input)

            Data_Table_Out = pd.DataFrame()

            # gets correct metadata based on area aggregation
            if self.AGG_BY=='zone':
                zone_lines = self.meta.zone_lines(scenario)
            else:
                zone_lines = self.meta.region_lines(scenario)
            try:
                zone_lines = zone_lines.set_index([self.AGG_BY])
            except:
                self.logger.info("Column to Aggregate by is missing")
                continue

            zone_lines = zone_lines.xs(zone_input)
            zone_lines = zone_lines['line_name'].unique()

            #Map lines to interfaces
            all_ints = self.meta.interface_lines(scenario) #Map lines to interfaces
            all_ints.index = all_ints.line
            ints = all_ints.loc[all_ints.index.intersection(zone_lines)]

            #flow = flow[flow.index.get_level_values('interface_name').isin(ints.interface)] #Limit to only interfaces touching to this zone
            #flow = flow.droplevel('interface_category')

            export_limits = self["interface_Export_Limit"].get(scenario).droplevel('timestamp')
            export_limits.mask(export_limits[0]==0.0,other=0.01,inplace=True) #if limit is zero set to small value
            export_limits = export_limits[export_limits.index.get_level_values('interface_name').isin(ints.interface)]
            export_limits = export_limits[export_limits[0].abs() < 99998] #Filter out unenforced interfaces.

            #Drop unnecessary columns.
            export_limits.reset_index(inplace = True)
            export_limits.drop(columns = 'interface_category',inplace = True)
            export_limits.set_index('interface_name',inplace = True)

            import_limits = self["interface_Import_Limit"].get(scenario).droplevel('timestamp')
            import_limits.mask(import_limits[0]==0.0,other=0.01,inplace=True) #if limit is zero set to small value
            import_limits = import_limits[import_limits.index.get_level_values('interface_name').isin(ints.interface)]
            import_limits = import_limits[import_limits[0].abs() < 99998] #Filter out unenforced interfaces.
            reported_ints = import_limits.index.get_level_values('interface_name').unique()

            #Drop unnecessary columns.
            import_limits.reset_index(inplace = True)
            import_limits.drop(columns = 'interface_category',inplace = True)
            import_limits.set_index('interface_name',inplace = True)

            #Extract time index
            ti = self["interface_Flow"][self.Scenarios[0]].index.get_level_values('timestamp').unique()

            if prop != '':
                interf_list = prop.split(',')
                self.logger.info('Plotting only interfaces specified in Marmot_plot_select.csv')
                self.logger.info(interf_list)
            else:
                interf_list = reported_ints.copy()

            self.logger.info('Carving out season from ' + start_date_range + ' to ' + end_date_range)

            #Remove missing interfaces from the list.
            for interf in interf_list:
                #Remove leading spaces
                if interf[0] == ' ':
                    interf = interf[1:]
                if interf not in reported_ints:
                    self.logger.warning(interf + ' not found in results.')
                    interf_list.remove(interf)
            if not interf_list:
                outputs = MissingInputData()
                return outputs

            xdim = 2
            ydim = len(interf_list)

            mplt = PlotLibrary(ydim, xdim, squeeze=False)
            fig, axs = mplt.get_figure()

            grid_size = xdim * ydim
            excess_axs = grid_size - len(interf_list)
            plt.subplots_adjust(wspace=0.05, hspace=0.2)
            missing_ints = 0
            chunks = []
            limits_chunks = []
            n = -1
            for interf in interf_list:
                n += 1

                #Remove leading spaces
                if interf[0] == ' ':
                    interf = interf[1:]

                chunks_interf = []
                single_exp_lim = export_limits.loc[interf] / 1000
                single_imp_lim = import_limits.loc[interf] / 1000

                #Check if all hours have the same limit.
                check = single_exp_lim.to_numpy()
                identical = check[0] == check.all()

                limits = pd.concat([single_exp_lim,single_imp_lim],axis = 1)
                limits.columns = ['export limit','import limit']
                limits.index = ti

                for scenario in self.Scenarios:
                    flow = self["interface_Flow"].get(scenario)
                    single_int = flow.xs(interf,level = 'interface_name') / 1000
                    single_int.index = single_int.index.droplevel('interface_category')
                    single_int.columns = [interf]
                    if self.shift_leapday == True:
                        single_int = self.adjust_for_leapday(single_int)
                    summer = single_int[start_date_range:end_date_range]
                    winter = single_int.drop(summer.index)
                    summer_lim = limits[start_date_range:end_date_range]
                    winter_lim = limits.drop(summer.index)

                    if duration_curve:
                        summer = self.sort_duration(summer,interf)
                        winter = self.sort_duration(winter,interf)
                        summer_lim = self.sort_duration(summer_lim,'export limit')
                        winter_lim = self.sort_duration(winter_lim,'export limit')

                    axs[n,0].plot(summer[interf],linewidth = 1,label = scenario + '\n interface flow')
                    axs[n,1].plot(winter[interf],linewidth = 1,label = scenario + '\n interface flow')
                    if scenario == self.Scenarios[-1]:
                        for col in summer_lim:
                            limits_color_dict = {'export limit': 'red', 'import limit': 'green'}
                            axs[n,0].plot(summer_lim[col], linewidth=1, linestyle='--',
                                          color=limits_color_dict[col], label=col)
                            axs[n,1].plot(winter_lim[col], linewidth=1, linestyle='--', 
                                          color=limits_color_dict[col], label=col)

                    #For output time series .csv
                    scenario_names = pd.Series([scenario] * len(single_int), name='Scenario')
                    single_int_out = single_int.set_index([scenario_names], append=True)
                    chunks_interf.append(single_int_out)

                Data_out_line = pd.concat(chunks_interf,axis = 0)
                Data_out_line.columns = [interf]
                chunks.append(Data_out_line)


                axs[n,0].set_title(interf)
                axs[n,1].set_title(interf)
                if not duration_curve:
                    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
                    formatter = mdates.ConciseDateFormatter(locator)
                    formatter.formats[2] = '%d\n %b'
                    formatter.zero_formats[1] = '%b\n %Y'
                    formatter.zero_formats[2] = '%d\n %b'
                    formatter.zero_formats[3] = '%H:%M\n %d-%b'
                    formatter.offset_formats[3] = '%b %Y'
                    formatter.show_offset = False
                    axs[n,0].xaxis.set_major_locator(locator)
                    axs[n,0].xaxis.set_major_formatter(formatter)
                    axs[n,1].xaxis.set_major_locator(locator)
                    axs[n,1].xaxis.set_major_formatter(formatter)

            mplt.add_legend()

            Data_Table_Out = pd.concat(chunks,axis = 1)
            #Limits_Out = pd.concat(limits_chunks,axis = 1)
            #Limits_Out.index = ['Export Limit','Import Limit']

            plt.ylabel('Flow (GW)',  color='black', rotation='vertical', labelpad=30)
            if duration_curve:
                plt.xlabel('Sorted hour of the year', color = 'black', labelpad = 30)

            fig.text(0.15,0.98,'Summer (' + start_date_range + ' to ' + end_date_range + ')',fontsize = 16)
            fig.text(0.58,0.98,'Winter',fontsize = 16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.97])
            if plot_data_settings["plot_title_as_region"]:
                mplt.add_main_title(zone_input)            
            outputs[zone_input] = {'fig': fig, 'data_table': Data_Table_Out}
            #Limits_Out.to_csv(os.path.join(self.Marmot_Solutions_folder, 'Figures_Output',self.AGG_BY + '_transmission','Individual_Interface_Limits.csv'))

        return outputs

    #TODO: re-organize parameters (self vs. not self)
    def int_flow_ind_diff(self, figure_name: str = None, **_):
        """Plot under development

        This method plots the hourly difference in interface flow between two scenarios for 
        individual interfaces, with a facet for each interface.
        The two scenarios are defined in the "Scenario_Diff" row of Marmot_user_defined_inputs.
        The interfaces are specified in the plot properties field of Marmot_plot_select.csv (column 4).
        The figure and data tables are saved within the module.

        Returns:
            UnderDevelopment(): Exception class, plot is not functional. 
        """
        return UnderDevelopment() # TODO: add new get_data method 
        
        duration_curve=False
        if 'duration_curve' in figure_name:
            duration_curve = True
        
        check_input_data = []
        Flow_Collection = {}
        Import_Limit_Collection = {}
        Export_Limit_Collection = {}

        check_input_data.extend([get_data(Flow_Collection,"interface_Flow",self.Marmot_Solutions_folder, self.Scenarios)])
        check_input_data.extend([get_data(Import_Limit_Collection,"interface_Import_Limit",self.Marmot_Solutions_folder, self.Scenarios)])
        check_input_data.extend([get_data(Export_Limit_Collection,"interface_Export_Limit",self.Marmot_Solutions_folder, self.Scenarios)])

        if 1 in check_input_data:
            outputs = MissingInputData()
            return outputs

        scenario = self.Scenarios[0]

        outputs = {}
        
        if not pd.isnull(self.start_date):
            self.logger.info("Plotting specific date range: \
            {} to {}".format(str(self.start_date),str(self.end_date)))
        
        for zone_input in self.Zones:
            self.logger.info("For all interfaces touching Zone = "+zone_input)

            Data_Table_Out = pd.DataFrame()

            # gets correct metadata based on area aggregation
            if self.AGG_BY=='zone':
                zone_lines = self.meta.zone_lines(scenario)
            else:
                zone_lines = self.meta.region_lines(scenario)
            try:
                zone_lines = zone_lines.set_index([self.AGG_BY])
            except:
                self.logger.info("Column to Aggregate by is missing")
                continue

            zone_lines = zone_lines.xs(zone_input)
            zone_lines = zone_lines['line_name'].unique()

            #Map lines to interfaces
            all_ints = self.meta.interface_lines(scenario) #Map lines to interfaces
            all_ints.index = all_ints.line
            ints = all_ints.loc[all_ints.index.intersection(zone_lines)]

            #flow = flow[flow.index.get_level_values('interface_name').isin(ints.interface)] #Limit to only interfaces touching to this zone
            #flow = flow.droplevel('interface_category')

            export_limits = Export_Limit_Collection.get(scenario).droplevel('timestamp')
            export_limits.mask(export_limits[0]==0.0,other=0.01,inplace=True) #if limit is zero set to small value
            export_limits = export_limits[export_limits.index.get_level_values('interface_name').isin(ints.interface)]
            export_limits = export_limits[export_limits[0].abs() < 99998] #Filter out unenforced interfaces.

            #Drop unnecessary columns.
            export_limits.reset_index(inplace = True)
            export_limits.drop(columns = 'interface_category',inplace = True)
            export_limits.set_index('interface_name',inplace = True)

            import_limits = Import_Limit_Collection.get(scenario).droplevel('timestamp')
            import_limits.mask(import_limits[0]==0.0,other=0.01,inplace=True) #if limit is zero set to small value
            import_limits = import_limits[import_limits.index.get_level_values('interface_name').isin(ints.interface)]
            import_limits = import_limits[import_limits[0].abs() < 99998] #Filter out unenforced interfaces.
            reported_ints = import_limits.index.get_level_values('interface_name').unique()

            #Drop unnecessary columns.
            import_limits.reset_index(inplace = True)
            import_limits.drop(columns = 'interface_category',inplace = True)
            import_limits.set_index('interface_name',inplace = True)
            
            #Extract time index
            ti = Flow_Collection[self.Scenarios[0]].index.get_level_values('timestamp').unique()

            if self.prop != '':
                interf_list = self.prop.split(',')
                self.logger.info('Plotting only interfaces specified in Marmot_plot_select.csv')
                self.logger.info(interf_list) 
            else:
                interf_list = reported_ints.copy()
                
            self.logger.info('Plotting full time series results.')
            xdim,ydim = self.set_x_y_dimension(len(interf_list))

            mplt = PlotLibrary(nrows, ncols,
                              squeeze=False, ravel_axs=True)
            fig, axs = mplt.get_figure()

            grid_size = xdim * ydim
            excess_axs = grid_size - len(interf_list)
            plt.subplots_adjust(wspace=0.05, hspace=0.2)
            missing_ints = 0
            chunks = []
            limits_chunks = []
            n = -1
            for interf in interf_list:
                n += 1

                #Remove leading spaces
                if interf[0] == ' ':
                    interf = interf[1:]
                if interf in reported_ints:
                    chunks_interf = []
                    single_exp_lim = export_limits.loc[interf] / 1000         #TODO: Use auto unit converter in method
                    single_imp_lim = import_limits.loc[interf] / 1000

                    #Check if all hours have the same limit.
                    check = single_exp_lim.to_numpy()
                    identical = check[0] == check.all()

                    limits = pd.concat([single_exp_lim,single_imp_lim],axis = 1)
                    limits.columns = ['export limit','import limit']

                    limits.index = ti

                    for scenario in self.Scenarios:
                        flow = Flow_Collection.get(scenario)
                        single_int = flow.xs(interf,level = 'interface_name') / 1000
                        single_int.index = single_int.index.droplevel('interface_category')
                        single_int.columns = [interf]

                        if self.shift_leapday == True:
                            single_int = self.adjust_for_leapday(single_int)

                        single_int = single_int.reset_index().set_index('timestamp')
                        limits = limits.reset_index().set_index('timestamp')
                        if not pd.isnull(self.start_date):
        
                            single_int = single_int[self.start_date : self.end_date]
                            limits = limits[self.start_date : self.end_date]

                        if duration_curve:
                            single_int = self.sort_duration(single_int,interf)
                            

                        mplt.lineplot(single_int,interf,label = scenario + '\n interface flow', sub_pos = n)
                        
                        #Only print limits if it doesn't change monthly or if you are plotting a time series. Otherwise the limit lines could be misleading.
                        if not duration_curve or identical[0]: 
                            if scenario == self.Scenarios[-1]:
                                #Only plot limits for last scenario.
                                limits_color_dict = {'export limit': 'red', 'import limit': 'green'}
                                mplt.lineplot(limits,'export limit',label = 'export limit',color = limits_color_dict,linestyle = '--', sub_pos = n)
                                mplt.lineplot(limits,'import limit',label = 'import limit',color = limits_color_dict,linestyle = '--', sub_pos = n)

                        #For output time series .csv
                        scenario_names = pd.Series([scenario] * len(single_int),name = 'Scenario')
                        single_int_out = single_int.set_index([scenario_names],append = True)
                        chunks_interf.append(single_int_out)

                    Data_out_line = pd.concat(chunks_interf,axis = 0)
                    Data_out_line.columns = [interf]
                    chunks.append(Data_out_line)

                else:
                    self.logger.warning(interf + ' not found in results. Have you tagged it with the "Must Report" property in PLEXOS?')
                    excess_axs += 1
                    missing_ints += 1
                    continue

                axs[n].set_title(interf)
                handles, labels = axs[n].get_legend_handles_labels()
                if not duration_curve:
                    self.set_subplot_timeseries_format(axs, sub_pos=n)
                if n == len(interf_list) - 1:
                    axs[n].legend(loc='lower left',bbox_to_anchor=(1.05,-0.2))

                if missing_ints == len(interf_list):
                    outputs = MissingInputData()
                    return outputs

            Data_Table_Out = pd.concat(chunks,axis = 1)
            Data_Table_Out = Data_Table_Out.reset_index()
            index_name = 'level_0' if duration_curve else 'timestamp'
            Data_Table_Out = Data_Table_Out.pivot(index = index_name,columns = 'Scenario')
            #Limits_Out = pd.concat(limits_chunks,axis = 1)
            #Limits_Out.index = ['Export Limit','Import Limit']

            # Data_Table_Out = Data_Table_Out.reset_index()
            # Data_Table_Out = Data_Table_Out.groupby(Data_Table_Out.index // 24).mean()
            # Data_Table_Out.index = pd.date_range(start = '1/1/2024',end = '12/31/2024',freq = 'D')

            plt.ylabel('Flow (GW)',  color='black', rotation='vertical', labelpad=30)
            if duration_curve:
                plt.xlabel('Sorted hour of the year', color = 'black', labelpad = 30)
            plt.tight_layout(rect=[0, 0.03, 1, 0.97])
            if plot_data_settings["plot_title_as_region"]:
                mplt.add_main_title(zone_input)            
            outputs[zone_input] = {'fig': fig, 'data_table': Data_Table_Out}
            #Limits_Out.to_csv(os.path.join(self.Marmot_Solutions_folder, 'Figures_Output',self.AGG_BY + '_transmission','Individual_Interface_Limits.csv'))
        return outputs


    def line_flow_ind(self, figure_name: str = None, prop: str = None, **_):
        """
        #TODO: Finish Docstring 


        This method plots flow, import and export limit, for individual transmission lines, 
        with a facet for each line.
        The lines are specified in the plot properties field of Marmot_plot_select.csv (column 4).
        The plot includes every interchange that originates or ends in the aggregation zone.
        Figures and data tables are returned to plot_main

        Args:
            figure_name (str, optional): [description]. Defaults to None.
            prop (str, optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        #TODO: Use auto unit converter in method
        duration_curve=False
        if 'duration_curve' in figure_name:
            duration_curve = True
            
        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True,"line_Flow",self.Scenarios),
                      (True,"line_Import_Limit",self.Scenarios),
                      (True,"line_Export_Limit",self.Scenarios)]
        
        # Runs get_formatted_data within PlotDataHelper to populate PlotDataHelper dictionary  
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)
        
        if 1 in check_input_data:
            return MissingInputData()

        #Select only lines specified in Marmot_plot_select.csv.
        select_lines = prop.split(",")
        if select_lines == None:
            return InputSheetError()

        self.logger.info('Plotting only lines specified in Marmot_plot_select.csv')
        self.logger.info(select_lines)

        scenario = self.Scenarios[0] #Select single scenario for purpose of extracting limits.

        export_limits = self["line_Export_Limit"].get(scenario).droplevel('timestamp')
        export_limits.mask(export_limits[0]==0.0,other=0.01,inplace=True) #if limit is zero set to small value
        export_limits = export_limits[export_limits[0].abs() < 99998] #Filter out unenforced lines.

        import_limits = self["line_Import_Limit"].get(scenario).droplevel('timestamp')
        import_limits.mask(import_limits[0]==0.0,other=0.01,inplace=True) #if limit is zero set to small value
        import_limits = import_limits[import_limits[0].abs() < 99998] #Filter out unenforced lines.


        flows = self["line_Flow"][scenario]

        # limited_lines = []
        # i = 0
        # all_lines = flows.index.get_level_values('line_name').unique()

        # for line in all_lines:
        #     i += 1
        #     print(line)
        #     print(i / len(all_lines))
        #     exp = export_limits.loc[line].squeeze()[0]
        #     imp = import_limits.loc[line].squeeze()[0]
        #     flow = flows.xs(line,level = 'line_name')[0].tolist()
        #     if exp in flow or imp in flow:
        #         limited_lines.append(line)

        # print(limited_lines)
        # pd.DataFrame(limited_lines).to_csv('/Users/mschwarz/OR OSW local/Solutions/Figures_Output/limited_lines.csv')

        xdim,ydim = self.set_x_y_dimension(len(select_lines))
        grid_size = xdim * ydim
        excess_axs = grid_size - len(select_lines)

        mplt = PlotLibrary(ydim, xdim, squeeze=False,
                            ravel_axs=True)
        fig, axs = mplt.get_figure()

        reported_lines = self["line_Flow"][self.Scenarios[0]].index.get_level_values('line_name').unique()
        n = -1
        missing_lines = 0
        chunks = []
        limits_chunks = []
        for line in select_lines:
            n += 1
            #Remove leading spaces
            if line[0] == ' ':
                line = line[1:]
            if line in reported_lines:
                chunks_line = []

                single_exp_lim = export_limits.loc[line]
                single_imp_lim = import_limits.loc[line]
                limits = pd.concat([single_exp_lim,single_imp_lim])
                limits_chunks.append(limits)
                single_exp_lim = single_exp_lim.squeeze()
                single_imp_lim = single_imp_lim.squeeze()

                # If export/import limits were pulled as an interval property, take the average.
                if len(single_exp_lim) > 1:
                    single_exp_lim = single_exp_lim.mean()
                    single_imp_lim = single_imp_lim.mean()

                limits = pd.Series([single_exp_lim,single_imp_lim],name = line)
                limits_chunks.append(limits)

                for scenario in self.Scenarios:
                    flow = self["line_Flow"][scenario]
                    single_line = flow.xs(line,level = 'line_name')
                    single_line = single_line.droplevel('units')
                    single_line.columns = [line]

                    if self.shift_leapday == True:
                        single_line = self.adjust_for_leapday(single_line)

                    single_line_out = single_line.copy()
                    if duration_curve:
                        single_line = self.sort_duration(single_line,line)

                    mplt.lineplot(single_line, line, label = scenario + '\n line flow', sub_pos=n)

                    #Add %congested number to plot.
                    if scenario == self.Scenarios[0]:

                        viol_exp = single_line[single_line[line] > single_exp_lim].count()
                        viol_imp = single_line[single_line[line] < single_imp_lim].count()
                        viol_perc = 100 * (viol_exp + viol_imp) / len(single_line)
                        viol_perc = round(viol_perc.squeeze(),3)
                        axs[n].annotate('Violation = ' + str(viol_perc) + '% of hours', xy = (0.1,0.15),xycoords='axes fraction')

                        cong_exp = single_line[single_line[line] == single_exp_lim].count()
                        cong_imp = single_line[single_line[line] == single_imp_lim].count()
                        cong_perc = 100 * (cong_exp + cong_imp) / len(single_line)
                        cong_perc = round(cong_perc.squeeze(),0)
                        axs[n].annotate('Congestion = ' + str(cong_perc) + '% of hours', xy = (0.1,0.1),xycoords='axes fraction')

                    #For output time series .csv
                    scenario_names = pd.Series([scenario] * len(single_line_out),name = 'Scenario')
                    single_line_out = single_line_out.set_index([scenario_names],append = True)
                    chunks_line.append(single_line_out)

                Data_out_line = pd.concat(chunks_line,axis = 0)
                chunks.append(Data_out_line)
            else:
                self.logger.warning(line + ' not found in results. Have you tagged it with the "Must Report" property in PLEXOS?')
                excess_axs += 1
                missing_lines += 1
                continue

            mplt.remove_excess_axs(excess_axs,grid_size)
            axs[n].axhline(y = single_exp_lim, ls = '--',label = 'Export Limit',color = 'red')
            axs[n].axhline(y = single_imp_lim, ls = '--',label = 'Import Limit', color = 'green')

            axs[n].set_title(line)
            if not duration_curve:
                mplt.set_subplot_timeseries_format(sub_pos=n)

        if missing_lines == len(select_lines):
            outputs = MissingInputData()
            return outputs

        Data_Table_Out = pd.concat(chunks,axis = 1)
        #Limits_Out = pd.concat(limits_chunks,axis = 1)
        #Limits_Out.index = ['Export Limit','Import Limit']

        mplt.add_legend()
        plt.ylabel('Flow (MW)',  color='black', rotation='vertical', labelpad=30)
        #plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.tight_layout()

        fn_suffix = '_duration_curve' if duration_curve else ''

        fig.savefig(os.path.join(self.Marmot_Solutions_folder, 'Figures_Output',self.AGG_BY + '_transmission',figure_name + fn_suffix + '.svg'), dpi=600, bbox_inches='tight')
        Data_Table_Out.to_csv(os.path.join(self.Marmot_Solutions_folder, 'Figures_Output',self.AGG_BY + '_transmission',figure_name + fn_suffix + '.csv'))
       # Limits_Out.to_csv(os.path.join(self.Marmot_Solutions_folder, 'Figures_Output',self.AGG_BY + '_transmission',figure_name + 'limits.csv'))

        outputs = DataSavedInModule()
        return outputs

    def line_flow_ind_diff(self, figure_name: str = None, 
                           prop: str = None, **_):
        """
        #TODO: Finish Docstring 

        This method plots the flow difference for individual transmission lines, with a facet for each line.
        The scenarios are specified in the "Scenario_Diff_plot" field of Marmot_user_defined_inputs.csv.
        The lines are specified in the plot properties field of Marmot_plot_select.csv (column 4).
        Figures and data tables are saved in the module.

        Args:
            figure_name (str, optional): [description]. Defaults to None.
            prop (str, optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        #TODO: Use auto unit converter in method
        
        duration_curve=False
        if 'duration_curve' in figure_name:
            duration_curve = True
            
        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True,"line_Flow",self.Scenarios)]
        
        # Runs get_formatted_data within PlotDataHelper to populate PlotDataHelper dictionary  
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)

        if 1 in check_input_data:
            outputs = MissingInputData()
            return outputs

        #Select only lines specified in Marmot_plot_select.csv.
        select_lines = prop.split(",")
        if select_lines == None:
            outputs = InputSheetError()
            return outputs

        self.logger.info('Plotting only lines specified in Marmot_plot_select.csv')
        self.logger.info(select_lines) 
        flow_diff = self["line_Flow"].get(self.Scenario_Diff[1]) - self["line_Flow"].get(self.Scenario_Diff[0])

        xdim,ydim = self.set_x_y_dimension(len(select_lines))
        grid_size = xdim * ydim
        excess_axs = grid_size - len(select_lines)

        mplt = PlotLibrary(ydim, xdim, squeeze=False,
                            ravel_axs=True)
        fig, axs = mplt.get_figure()
        plt.subplots_adjust(wspace=0.05, hspace=0.2)

        reported_lines = self["line_Flow"].get(self.Scenarios[0]).index.get_level_values('line_name').unique()
        n = -1
        missing_lines = 0
        chunks = []
        for line in select_lines:
            n += 1
            #Remove leading spaces
            if line[0] == ' ':
                line = line[1:]
            if line in reported_lines:


                single_line = flow_diff.xs(line,level = 'line_name')
                single_line.columns = [line]
                if self.shift_leapday == True:
                    single_line = self.adjust_for_leapday(single_line)

                single_line_out = single_line.copy()
                if duration_curve:
                    single_line = self.sort_duration(single_line,line)
                                        
                #mplt.lineplot(single_line,line, label = self.Scenario_Diff[1] + ' - \n' + self.Scenario_Diff[0] + '\n line flow', sub_pos = n)
                mplt.lineplot(single_line,line, label = 'BESS - no BESS \n line flow', sub_pos=n)


            else:
                self.logger.warning(line + ' not found in results. Have you tagged it with the "Must Report" property in PLEXOS?')
                excess_axs += 1
                missing_lines += 1
                continue

            mplt.remove_excess_axs(excess_axs,grid_size)     
            axs[n].set_title(line)
            if not duration_curve:
                mplt.set_subplot_timeseries_format(sub_pos=n)

            chunks.append(single_line_out)

        if missing_lines == len(select_lines):
            outputs = MissingInputData()
            return outputs

        Data_Table_Out = pd.concat(chunks,axis = 1)

        mplt.add_legend()
        plt.ylabel('Flow difference (MW)',  color='black', rotation='vertical', labelpad=30)
        plt.tight_layout()

        fn_suffix = '_duration_curve' if duration_curve else ''

        fig.savefig(os.path.join(self.Marmot_Solutions_folder, 'Figures_Output',self.AGG_BY + '_transmission',figure_name + fn_suffix + '.svg'), dpi=600, bbox_inches='tight')
        Data_Table_Out.to_csv(os.path.join(self.Marmot_Solutions_folder, 'Figures_Output',self.AGG_BY + '_transmission',figure_name + fn_suffix + '.csv'))

        outputs = DataSavedInModule()
        return outputs

    def line_flow_ind_seasonal(self, figure_name: str = None, prop: str = None, 
                               start_date_range: str = None, 
                               end_date_range: str = None, **_):
        """TODO: Finish Docstring.

        This method differs from the previous method, in that it plots seasonal line limits.
        To use this method, line import/export must be an "interval" property, not a "year" property.
        This can be selected in  "plexos_properties.csv".
        Re-run the formatter if necessary, it will overwrite the existing properties in "*_formatted.h5"

        This method plots flow, import and export limit, for individual transmission lines, with a facet for each line.
        The lines are specified in the plot properties field of Marmot_plot_select.csv (column 4).
        The plot includes every interchange that originates or ends in the aggregation zone. 
        Figures and data tables saved in the module.

        Args:
            figure_name (str, optional): [description]. Defaults to None.
            prop (str, optional): [description]. Defaults to None.
            start_date_range (str, optional): [description]. Defaults to None.
            end_date_range (str, optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        #TODO: Use auto unit converter in method

        if pd.isna(start_date_range):
            self.logger.warning('You are attempting to plot a time series facetted by two seasons,\n\
            but you are missing a value in the "Start Date" column of "Marmot_plot_select.csv" \
            Please enter dates in "Start Date" and "End Date". These will define the bounds of \
            one of your two seasons. The other season will be comprised of the rest of the year.')
            return MissingInputData()
        
        duration_curve=False
        if 'duration_curve' in figure_name:
            duration_curve = True
            
        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True,"line_Flow",self.Scenarios),
                      (True,"line_Import_Limit",self.Scenarios),
                      (True,"line_Export_Limit",self.Scenarios)]
        
        # Runs get_formatted_data within PlotDataHelper to populate PlotDataHelper dictionary  
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)

        if 1 in check_input_data:
            return MissingInputData()

        #Select only lines specified in Marmot_plot_select.csv.
        select_lines = prop.split(",")
        if select_lines == None:
            return InputSheetError()


        self.logger.info('Plotting only lines specified in Marmot_plot_select.csv')
        self.logger.info(select_lines)

        scenario = self.Scenarios[0]

        #Line limits are seasonal.
        export_limits = self["line_Export_Limit"].get(scenario).droplevel('timestamp')
        export_limits.mask(export_limits[0]==0.0,other=0.01,inplace=True) #if limit is zero set to small value
        export_limits = export_limits[export_limits[0].abs() < 99998] #Filter out unenforced lines.

        import_limits = self["line_Import_Limit"].get(scenario).droplevel('timestamp')
        import_limits.mask(import_limits[0]==0.0,other=0.01,inplace=True) #if limit is zero set to small value
        import_limits = import_limits[import_limits[0].abs() < 99998] #Filter out unenforced lines.

        #Extract time index
        ti = self["line_Flow"][self.Scenarios[0]].index.get_level_values('timestamp').unique()
        reported_lines = self["line_Flow"][self.Scenarios[0]].index.get_level_values('line_name').unique()

        self.logger.info('Carving out season from ' + start_date_range + ' to ' + end_date_range)

        #Remove missing interfaces from the list.
        for line in select_lines:
            #Remove leading spaces
            if line[0] == ' ':
                line = line[1:]
            if line not in reported_lines:
                self.logger.warning(line + ' not found in results.')
                select_lines.remove(line)
        if not select_lines:
            outputs = MissingInputData()
            return outputs

        xdim = 2
        ydim = len(select_lines)
        grid_size = xdim * ydim
        excess_axs = grid_size - len(select_lines)

        mplt = PlotLibrary(ydim, xdim, squeeze=False)
        fig, axs = mplt.get_figure()

        i = -1
        missing_lines = 0
        chunks = []
        limits_chunks = []
        for line in select_lines:
            i += 1
            #Remove leading spaces
            if line[0] == ' ':
                line = line[1:]

            chunks_line = []

            single_exp_lim = export_limits.loc[line]
            single_exp_lim.index = ti
            single_imp_lim = import_limits.loc[line]
            single_imp_lim.index = ti

            limits = pd.concat([single_exp_lim,single_imp_lim],axis = 1)
            limits.columns = ['export limit','import limit']
            limits.index = ti
            limits_chunks.append(limits)

            for scenario in self.Scenarios:

                flow = self["line_Flow"][scenario]
                single_line = flow.xs(line,level = 'line_name')
                single_line = single_line.droplevel('units')
                single_line_out = single_line.copy()
                single_line.columns = [line]
                if self.shift_leapday == True:
                    single_line = self.adjust_for_leapday(single_line)

                #Split into seasons.
                summer = single_line[start_date_range : end_date_range]
                winter = single_line.drop(summer.index)
                summer_lim = limits[start_date_range:end_date_range]
                winter_lim = limits.drop(summer.index)

                if duration_curve:
                    summer = self.sort_duration(summer,line)
                    winter = self.sort_duration(winter,line)
                    summer_lim = self.sort_duration(summer_lim,'export limit')
                    winter_lim = self.sort_duration(winter_lim,'export limit')

                axs[i,0].plot(summer[line],linewidth = 1,label = scenario + '\n line flow')
                axs[i,1].plot(winter[line],linewidth = 1,label = scenario + '\n line flow')
                if scenario == self.Scenarios[-1]:
                    for col in summer_lim:
                        limits_color_dict = {'export limit': 'red', 'import limit': 'green'}
                        axs[i,0].plot(summer_lim[col],linewidth = 1,linestyle = '--',color = limits_color_dict[col],label = col)
                        axs[i,1].plot(winter_lim[col],linewidth = 1,linestyle = '--',color = limits_color_dict[col],label = col)

                for j in [0,1]:
                    axs[i,j].spines['right'].set_visible(False)
                    axs[i,j].spines['top'].set_visible(False)
                    axs[i,j].tick_params(axis='y', which='major', length=5, width=1)
                    axs[i,j].tick_params(axis='x', which='major', length=5, width=1)
                    axs[i,j].set_title(line)
                    if i == len(select_lines) - 1:
                        axs[i,j].legend(loc = 'lower left',bbox_to_anchor=(1.05,0),facecolor='inherit', frameon=True)

                #For output time series .csv
                scenario_names = pd.Series([scenario] * len(single_line_out),name = 'Scenario')
                single_line_out.columns = [line]
                single_line_out = single_line_out.set_index([scenario_names],append = True)
                chunks_line.append(single_line_out)

            Data_out_line = pd.concat(chunks_line,axis = 0)
            chunks.append(Data_out_line)


        if missing_lines == len(select_lines):
            outputs = MissingInputData()
            return outputs

        Data_Table_Out = pd.concat(chunks,axis = 1)
        #Limits_Out = pd.concat(limits_chunks,axis = 1)
        #Limits_Out.index = ['Export Limit','Import Limit']

        fig.text(0.3,1,'Summer (Jun - Sep)')
        fig.text(0.6,1,'Winter (Jan - Mar,Oct - Dec)')
        plt.ylabel('Flow (MW)',  color='black', rotation='vertical', labelpad=30)
        plt.tight_layout()

        fn_suffix = '_duration_curve' if duration_curve else ''


        fig.savefig(os.path.join(self.Marmot_Solutions_folder, 'Figures_Output',self.AGG_BY + '_transmission','Individual_Line_Flow' + fn_suffix + '_seasonal.svg'), dpi=600, bbox_inches='tight')
        Data_Table_Out.to_csv(os.path.join(self.Marmot_Solutions_folder, 'Figures_Output',self.AGG_BY + '_transmission','Individual_Line_Flow' + fn_suffix + '_seasonal.csv'))
        #Limits_Out.to_csv(os.path.join(self.Marmot_Solutions_folder, 'Figures_Output',self.AGG_BY + '_transmission','Individual_Line_Limits.csv'))
        outputs = DataSavedInModule()
        return outputs

    def extract_tx_cap(self, **_):
        """Plot under development

        Returns:
            UnderDevelopment(): Exception class, plot is not functional. 
        """
        return UnderDevelopment() #TODO: Needs finishing
        
        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True,"interface_Import_Limit",self.Scenarios),
                      (True,"interface_Export_Limit",self.Scenarios),
                      (True,"line_Import_Limit",self.Scenarios),
                      (True,"line_Export_Limit",self.Scenarios)]
        
        # Runs get_formatted_data within PlotDataHelper to populate PlotDataHelper dictionary  
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)
        
        if 1 in check_input_data:
            return MissingInputData()
        
        for scenario in self.Scenarios:
            self.logger.info(scenario)
            for zone_input in self.Zones:

                #Lines
                # lines = self.meta.region_interregionallines(scenario)
                # if scenario == 'ADS':
                #     zone_input = zone_input.split('_WI')[0]
                #     lines = self.meta_ADS.region_interregionallines()

                # lines = lines[lines['region'] == zone_input]
                # import_lim = self["line_Import_Limit"][scenario].reset_index()
                # export_lim = self["line_Export_Limit"][scenario].reset_index()
                # lines = lines.merge(import_lim,how = 'inner',on = 'line_name')
                # lines = lines[['line_name',0]]
                # lines.columns = ['line_name','import_limit']
                # lines = lines.merge(export_lim, how = 'inner',on = 'line_name')
                # lines = lines[['line_name','import_limit',0]]
                # lines.columns = ['line_name','import_limit','export_limit']

                # fn = os.path.join(self.Marmot_Solutions_folder, 'NARIS', 'Figures_Output',self.AGG_BY + '_transmission','Individual_Interregional_Line_Limits_' + scenario + '.csv')
                # lines.to_csv(fn)

                # lines = self.meta.region_intraregionallines(scenario)
                # if scenario == 'ADS':
                #     lines = self.meta_ADS.region_intraregionallines()

                # lines = lines[lines['region'] == zone_input]
                # import_lim = self["line_Import_Limit"][scenario].reset_index()
                # export_lim = self["line_Export_Limit"][scenario].reset_index()
                # lines = lines.merge(import_lim,how = 'inner',on = 'line_name')
                # lines = lines[['line_name',0]]
                # lines.columns = ['line_name','import_limit']
                # lines = lines.merge(export_lim, how = 'inner',on = 'line_name')
                # lines = lines[['line_name','import_limit',0]]
                # lines.columns = ['line_name','import_limit','export_limit']

                # fn = os.path.join(self.Marmot_Solutions_folder, 'NARIS', 'Figures_Output',self.AGG_BY + '_transmission','Individual_Intraregional_Line_Limits_' + scenario + '.csv')
                # lines.to_csv(fn)


                #Interfaces
                PSCo_ints = ['P39 TOT 5_WI','P40 TOT 7_WI']

                int_import_lim = self["interface_Import_Limit"][scenario].reset_index()
                int_export_lim = self["interface_Export_Limit"][scenario].reset_index()
                if scenario == 'NARIS':
                    last_timestamp = int_import_lim['timestamp'].unique()[-1] #Last because ADS uses the last timestamp.
                    int_import_lim = int_import_lim[int_import_lim['timestamp'] == last_timestamp]
                    int_export_lim = int_export_lim[int_export_lim['timestamp'] == last_timestamp]
                    lines2ints = self.meta_ADS.interface_lines()
                else:
                    lines2ints = self.meta.interface_lines(scenario)

                fn = os.path.join(self.Marmot_Solutions_folder, 'NARIS', 'Figures_Output',self.AGG_BY + '_transmission','test_meta_' + scenario + '.csv')
                lines2ints.to_csv(fn)


                ints = pd.merge(int_import_lim,int_export_lim,how = 'inner', on = 'interface_name')
                ints.rename(columns = {'0_x':'import_limit','0_y': 'export_limit'},inplace = True)
                all_lines_in_ints = lines2ints['line'].unique()
                test = [line for line in lines['line_name'].unique() if line in all_lines_in_ints]
                ints = ints.merge(lines2ints, how = 'inner', left_on = 'interface_name',right_on = 'interface')

    def region_region_interchange_all_scenarios(self, **kwargs):
        """
        #TODO: Finish Docstring 

        This method creates a timeseries line plot of interchange flows between the selected region
        to each connecting region.
        If there are more than 4 total interchanges, all other interchanges are aggregated into an 'other' grouping
        Each scenarios is plotted on a separate Facet plot.
        Figures and data tables are returned to plot_main
        """
        outputs = self._region_region_interchange(self.Scenarios, **kwargs)
        return outputs

    def region_region_interchange_all_regions(self, **kwargs):
        """
        #TODO: Finish Docstring 

        This method creates a timeseries line plot of interchange flows between the selected region
        to each connecting region. All regions are plotted on a single figure with each focus region placed on a separate
        facet plot
        If there are more than 4 total interchanges, all other interchanges are aggregated into an 'other' grouping
        This figure only plots a single scenario that is defined by Main_scenario_plot in user_defined_inputs.csv.
        Figures and data tables are saved within method
        """
        outputs = self._region_region_interchange([self.Scenarios[0]],plot_scenario=False, **kwargs)
        return outputs

    def _region_region_interchange(self, scenario_type: str, plot_scenario: bool = True, 
                                    timezone: str = "", **_):
        """#TODO: Finish Docstring 

        Args:
            scenario_type (str): [description]
            plot_scenario (bool, optional): [description]. Defaults to True.
            timezone (str, optional): [description]. Defaults to "".

        Returns:
            [type]: [description]
        """
        outputs = {}
        
        if self.AGG_BY == 'zone':
            agg = 'zone'
        else:
            agg = 'region'
            
        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True,f"{agg}_{agg}s_Net_Interchange",scenario_type)]
        
        # Runs get_formatted_data within PlotDataHelper to populate PlotDataHelper dictionary  
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)
        
        if 1 in check_input_data:
            return MissingInputData()

        for zone_input in self.Zones:
            self.logger.info(f"Zone = {zone_input}")

            ncols, nrows = self.set_facet_col_row_dimensions(multi_scenario=scenario_type)

            mplt = PlotLibrary(nrows, ncols, sharey=True,
                              squeeze=False, ravel_axs=True)
            fig, axs = mplt.get_figure()            
            plt.subplots_adjust(wspace=0.6, hspace=0.3)

            data_table_chunks=[]
            n=0
            for scenario in scenario_type:

                rr_int = self[f"{agg}_{agg}s_Net_Interchange"].get(scenario)
                if self.shift_leapday == True:
                    rr_int = self.adjust_for_leapday(rr_int)

                # For plot_main handeling - need to find better solution
                if plot_scenario == False:
                    outputs={}
                    for zone_input in self.Zones:
                        outputs[zone_input] = pd.DataFrame()

                if self.AGG_BY != 'region' and self.AGG_BY != 'zone':
                    agg_region_mapping = self.Region_Mapping[['region',self.AGG_BY]].set_index('region').to_dict()[self.AGG_BY]
                    # Checks if keys all aggregate to a single value, this plot requires multiple values to work 
                    if len(set(agg_region_mapping.values())) == 1:
                        return UnsupportedAggregation()
                    rr_int = rr_int.reset_index()
                    rr_int['parent'] = rr_int['parent'].map(agg_region_mapping)
                    rr_int['child']  = rr_int['child'].map(agg_region_mapping)

                rr_int_agg = rr_int.groupby(['timestamp','parent','child'],as_index=True).sum()
                rr_int_agg.rename(columns = {0:'flow (MW)'}, inplace = True)
                rr_int_agg = rr_int_agg.reset_index()

                # If plotting all regions update plot setup
                if plot_scenario == False:
                    #Make a facet plot, one panel for each parent zone.
                    parent_region = rr_int_agg['parent'].unique()
                    plot_number = len(parent_region)
                    ncols, nrows =  self.set_x_y_dimension(plot_number)
                    mplt = PlotLibrary(nrows, ncols,
                                        squeeze=False, ravel_axs=True)
                    fig, axs = mplt.get_figure()
                    plt.subplots_adjust(wspace=0.6, hspace=0.7)

                else:
                    parent_region = [zone_input]
                    plot_number = len(scenario_type)

                grid_size = ncols*nrows
                excess_axs = grid_size - plot_number

                for parent in parent_region:
                    single_parent = rr_int_agg[rr_int_agg['parent'] == parent]
                    single_parent = single_parent.pivot(index = 'timestamp',columns = 'child',values = 'flow (MW)')
                    single_parent = single_parent.loc[:,(single_parent != 0).any(axis = 0)] #Remove all 0 columns (uninteresting).
                    if (parent in single_parent.columns):
                        single_parent = single_parent.drop(columns = [parent]) #Remove columns if parent = child

                    #Neaten up lines: if more than 4 total interchanges, aggregated all but the highest 3.
                    if len(single_parent.columns) > 4:
                        # Set the "three highest zonal interchanges" for all three scenarios.
                        cols_dontagg = single_parent.max().abs().sort_values(ascending = False)[0:3].index
                        df_dontagg = single_parent[cols_dontagg]
                        df_toagg = single_parent.drop(columns = cols_dontagg)
                        agged = df_toagg.sum(axis = 1)
                        df_dontagg.insert(len(df_dontagg.columns),'Other',agged)
                        single_parent = df_dontagg.copy()

                    #Convert units
                    if n == 0:
                        unitconversion = self.capacity_energy_unitconversion(single_parent)
                    single_parent = single_parent / unitconversion['divisor']

                    for column in single_parent.columns:

                        mplt.lineplot(single_parent, column, label=column, sub_pos=n)
                        axs[n].set_title(parent)
                        axs[n].margins(x=0.01)
                        mplt.set_subplot_timeseries_format(sub_pos=n)
                        axs[n].hlines(y = 0, xmin = axs[n].get_xlim()[0], xmax = axs[n].get_xlim()[1], linestyle = ':') #Add horizontal line at 0.
                        axs[n].legend(loc='lower left',bbox_to_anchor=(1,0))

                    n+=1
                # Create data table for each scenario
                scenario_names = pd.Series([scenario]*len(single_parent),name='Scenario')
                data_table = single_parent.add_suffix(f" ({unitconversion['units']})")
                data_table = data_table.set_index([scenario_names],append=True)
                data_table_chunks.append(data_table)

            # if plotting all scenarios add facet labels
            if plot_scenario == True:
                mplt.add_facet_labels(xlabels=self.xlabels,
                                      ylabels = self.ylabels)

            #Remove extra axes
            mplt.remove_excess_axs(excess_axs, grid_size)
            plt.xlabel(timezone,  color='black', rotation='horizontal',labelpad = 30)
            plt.ylabel(f"Net Interchange ({unitconversion['units']})",  color='black', rotation='vertical', labelpad = 40)

            # If plotting all regions save output and return none plot_main
            if plot_scenario == False:
                # Location to save to
                Data_Table_Out = rr_int_agg
                save_figures = os.path.join(self.figure_folder, self.AGG_BY + '_transmission')
                fig.savefig(os.path.join(save_figures, "Region_Region_Interchange_{}.svg".format(self.Scenarios[0])), dpi=600, bbox_inches='tight')
                Data_Table_Out.to_csv(os.path.join(save_figures, "Region_Region_Interchange_{}.csv".format(self.Scenarios[0])))
                outputs = DataSavedInModule()
                return outputs

            Data_Out = pd.concat(data_table_chunks, copy=False, axis=0)

            # if plotting all scenarios return figures to plot_main
            outputs[zone_input] = {'fig': fig,'data_table':Data_Out}
        return outputs

    def region_region_checkerboard(self, **_):
        """Creates a checkerboard/heatmap figure showing total interchanges between regions/zones.

        Each scenario is plotted on its own facet plot.
        Plots and Data are saved within the module.

        Returns:
            DataSavedInModule: DataSavedInModule exception.
        """
        outputs = {}
        
        if self.AGG_BY == 'zone':
            agg = 'zone'
        else:
            agg = 'region'
            
        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True,f"{agg}_{agg}s_Net_Interchange",self.Scenarios)]
        
        # Runs get_formatted_data within PlotDataHelper to populate PlotDataHelper dictionary  
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)
        
        if 1 in check_input_data:
            return MissingInputData()

        ncols, nrows = self.set_x_y_dimension(len(self.Scenarios))
        grid_size = ncols*nrows
        excess_axs = grid_size - len(self.Scenarios)

        mplt = PlotLibrary(nrows, ncols,
                              squeeze=False, ravel_axs=True)
        fig, axs = mplt.get_figure()
        plt.subplots_adjust(wspace=0.02, hspace=0.4)
        max_flow_group = []
        Data_Out = []
        n=0
        for scenario in self.Scenarios:
            rr_int = self[f"{agg}_{agg}s_Net_Interchange"].get(scenario)
            if self.shift_leapday == True:
                rr_int = self.adjust_for_leapday(rr_int)

            if self.AGG_BY != 'region' and self.AGG_BY != 'zone':
                    agg_region_mapping = self.Region_Mapping[['region',self.AGG_BY]].set_index('region').to_dict()[self.AGG_BY]
                    # Checks if keys all aggregate to a single value, this plot requires multiple values to work 
                    if len(set(agg_region_mapping.values())) == 1:
                        return UnsupportedAggregation()
                    rr_int = rr_int.reset_index()
                    rr_int['parent'] = rr_int['parent'].map(agg_region_mapping)
                    rr_int['child']  = rr_int['child'].map(agg_region_mapping)
            rr_int_agg = rr_int.groupby(['parent','child'],as_index=True).sum()
            rr_int_agg.rename(columns = {0:'flow (MW)'}, inplace = True)
            rr_int_agg=rr_int_agg.loc[rr_int_agg['flow (MW)']>0.01] # Keep only positive flows
            rr_int_agg.sort_values(ascending=False,by='flow (MW)')
            rr_int_agg = rr_int_agg/1000 # MWh -> GWh

            data_out = rr_int_agg.copy()
            data_out.rename(columns={'flow (MW)':'{} flow (GWh)'.format(scenario)},inplace=True)

            max_flow = max(rr_int_agg['flow (MW)'])
            rr_int_agg = rr_int_agg.unstack('child')
            rr_int_agg = rr_int_agg.droplevel(level = 0, axis = 1)

            current_cmap = plt.cm.get_cmap()
            current_cmap.set_bad(color='grey')

            axs[n].imshow(rr_int_agg)
            axs[n].set_xticks(np.arange(rr_int_agg.shape[1]))
            axs[n].set_yticks(np.arange(rr_int_agg.shape[0]))
            axs[n].set_xticklabels(rr_int_agg.columns)
            axs[n].set_yticklabels(rr_int_agg.index)
            axs[n].set_title(scenario.replace('_',' '),fontweight='bold')

            # Rotate the tick labels and set their alignment.
            plt.setp(axs[n].get_xticklabels(), rotation=30, ha="right",
                 rotation_mode="anchor")

            #Delineate the boxes and make room at top and bottom
            axs[n].set_xticks(np.arange(rr_int_agg.shape[1]+1)-.5, minor=True) 
            axs[n].set_yticks(np.arange(rr_int_agg.shape[0]+1)-.5, minor=True)
            axs[n].grid(which="minor", color="k", linestyle='-', linewidth=1)
            axs[n].tick_params(which="minor", bottom=False, left=False)

            max_flow_group.append(max_flow)
            Data_Out.append(data_out)
            n+=1

        #Remove extra axes
        mplt.remove_excess_axs(excess_axs,grid_size)

        cmap = cm.inferno
        norm = mcolors.Normalize(vmin=0, vmax=max(max_flow_group))
        cax = plt.axes([0.90, 0.1, 0.035, 0.8])
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cax, label='Total Net Interchange [GWh]')
        plt.xlabel('To Region', color='black', rotation='horizontal',
                   labelpad=40)
        plt.ylabel('From Region', color='black', rotation='vertical', 
                   labelpad=40)

        Data_Table_Out = pd.concat(Data_Out,axis=1)
        save_figures = os.path.join(self.figure_folder, f"{self.AGG_BY}_transmission")
        fig.savefig(os.path.join(save_figures, "region_region_checkerboard.svg"), 
                     dpi=600, bbox_inches='tight')
        Data_Table_Out.to_csv(os.path.join(save_figures, "region_region_checkerboard.csv"))

        outputs = DataSavedInModule()
        return outputs

    def line_violations_timeseries(self, **kwargs):
        """Creates a timeseries line plot of lineflow violations for each region.

        The magnitude of each violation is plotted on the y-axis
        Each sceanrio is plotted as a separate line.

        This methods calls _violations() to create the figure.

        Returns:
            dict: Dictionary containing the created plot and its data table.
        """
        outputs = self._violations(**kwargs)
        return outputs

    def line_violations_totals(self, **kwargs):
        """Creates a barplot of total lineflow violations for each region.

        Each sceanrio is plotted as a separate bar.

        This methods calls _violations() and passes the total_violations=True argument 
        to create the figure.

        Returns:
            dict: Dictionary containing the created plot and its data table.
        """
        outputs = self._violations(total_violations=True, **kwargs)
        return outputs

    def _violations(self, total_violations: bool = False, 
                    timezone: str = "", 
                    start_date_range: str = None,
                    end_date_range: str = None, **_):
        """Creates line violation plots, line plot and barplots

        This methods is called from line_violations_timeseries() and line_violations_totals()

        Args:
            total_violations (bool, optional): If True finds the sum of violations. 
                Used to create barplots. Defaults to False.
            timezone (str, optional): The timezone to display on the x-axes.
                Defaults to "".
            start_date_range (str, optional): Defines a start date at which to represent data from. 
                Defaults to None.
            end_date_range (str, optional): Defines a end date at which to represent data to.
                Defaults to None.

        Returns:
            dict: dictionary containing the created plot and its data table
        """
        outputs = {}
        
        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True,"line_Violation",self.Scenarios)]
        
        # Runs get_formatted_data within PlotDataHelper to populate PlotDataHelper dictionary  
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)
        
        if 1 in check_input_data:
            return MissingInputData()

        for zone_input in self.Zones:
            self.logger.info(f'Zone = {zone_input}')
            all_scenarios = pd.DataFrame()

            for scenario in self.Scenarios:
                self.logger.info(f"Scenario = {str(scenario)}")

                if self.AGG_BY == 'zone':
                    lines = self.meta.zone_lines(scenario)
                else:
                    lines = self.meta.region_lines(scenario)

                line_v = self["line_Violation"].get(scenario)
                line_v = line_v.reset_index()

                viol = line_v.merge(lines,on = 'line_name',how = 'left')

                if self.AGG_BY == 'zone':
                    viol = viol.groupby(["timestamp", "zone"]).sum()
                else:
                    viol = viol.groupby(["timestamp", self.AGG_BY]).sum()

                one_zone = viol.xs(zone_input, level = self.AGG_BY)
                one_zone = one_zone.rename(columns = {0 : scenario})
                one_zone = one_zone.abs() #We don't care the direction of the violation
                all_scenarios = pd.concat([all_scenarios,one_zone], axis = 1)

            all_scenarios.columns = all_scenarios.columns.str.replace('_',' ')
            #remove columns that are all equal to 0
            all_scenarios = all_scenarios.loc[:, (all_scenarios != 0).any(axis=0)]
            
            if all_scenarios.empty:
                outputs[zone_input] = MissingZoneData()
                continue
            
            unitconversion = self.capacity_energy_unitconversion(all_scenarios)
            all_scenarios = all_scenarios/unitconversion['divisor']

            Data_Table_Out = all_scenarios.add_suffix(f" ({unitconversion['units']})")

            #Make scenario/color dictionary.
            color_dict = dict(zip(all_scenarios.columns,self.color_list))

            mplt = PlotLibrary()
            fig, ax = mplt.get_figure()

            if total_violations==True:
                all_scenarios_tot = all_scenarios.sum()
                all_scenarios_tot.plot.bar(stacked=False, rot=0,
                                            color=[color_dict.get(x, '#333333') for x in all_scenarios_tot.index],
                                            linewidth='0.1', width=0.35, ax=ax)
            else:
                for column in all_scenarios:
                    mplt.lineplot(all_scenarios,column,color=color_dict,label=column)
                ax.margins(x=0.01)
                mplt.set_subplot_timeseries_format(minticks=6,maxticks=12)
                ax.set_xlabel(timezone,  color='black', rotation='horizontal')
                mplt.add_legend()

            if plot_data_settings["plot_title_as_region"]:
                fig.set_title(zone_input)
            ax.set_ylabel(f"Line violations ({unitconversion['units']})",  color='black', rotation='vertical')

            outputs[zone_input] = {'fig': fig,'data_table':Data_Table_Out}

        return outputs

    def net_export(self, timezone: str = "", 
                   start_date_range: str = None,
                   end_date_range: str = None, **_):
        """creates a timeseries net export line graph.

        Scenarios are plotted as separate lines.

        Args:
            timezone (str, optional): The timezone to display on the x-axes.
                Defaults to "".
            start_date_range (str, optional): Defines a start date at which to represent data from. 
                Defaults to None.
            end_date_range (str, optional): Defines a end date at which to represent data to.
                Defaults to None.

        Returns:
            dict: dictionary containing the created plot and its data table
        """
        if self.AGG_BY == 'zone':
            agg = 'zone'
        else:
            agg = 'region'
            
        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True,f"{agg}_Net_Interchange",self.Scenarios)]
        
        # Runs get_formatted_data within PlotDataHelper to populate PlotDataHelper dictionary  
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)
        
        if 1 in check_input_data:
            return MissingInputData()

        outputs = {}
        for zone_input in self.Zones:
            self.logger.info(f"{self.AGG_BY} = {zone_input}")

            net_export_all_scenarios = pd.DataFrame()

            for scenario in self.Scenarios:


                self.logger.info(f"Scenario = {scenario}")
                net_export_read = self[f"{agg}_Net_Interchange"].get(scenario)
                if self.shift_leapday == True:
                    net_export_read = self.adjust_for_leapday(net_export_read)                

                net_export = net_export_read.xs(zone_input, level = self.AGG_BY)
                net_export = net_export.groupby("timestamp").sum()
                net_export.columns = [scenario]

                if pd.notna(start_date_range):
                    self.logger.info(f"Plotting specific date range: \
                    {str(start_date_range)} to {str(end_date_range)}")
                    net_export = net_export[start_date_range : end_date_range]

                net_export_all_scenarios = pd.concat([net_export_all_scenarios,net_export], axis = 1)
                net_export_all_scenarios.columns = net_export_all_scenarios.columns.str.replace('_', ' ')

            unitconversion = self.capacity_energy_unitconversion(net_export_all_scenarios)

            net_export_all_scenarios = net_export_all_scenarios/unitconversion["divisor"]
            # Data table of values to return to main program
            Data_Table_Out = net_export_all_scenarios.add_suffix(f" ({unitconversion['units']})")
            #Make scenario/color dictionary.
            color_dict = dict(zip(net_export_all_scenarios.columns,self.color_list))

            mplt = PlotLibrary()
            fig, ax = mplt.get_figure()
            plt.subplots_adjust(wspace=0.05, hspace=0.2)

            if net_export_all_scenarios.empty:
                out = MissingZoneData()
                outputs[zone_input] = out
                continue

            for column in net_export_all_scenarios:
                mplt.lineplot(net_export_all_scenarios,column,color_dict, label=column)
                ax.set_ylabel(f'Net exports ({unitconversion["units"]})', color='black', 
                              rotation='vertical')
                ax.set_xlabel(timezone, color='black', rotation='horizontal')
                ax.margins(x=0.01)
                ax.hlines(y=0, xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1], 
                          linestyle=':')
                mplt.set_subplot_timeseries_format()

            mplt.add_legend(reverse_legend=True)
            if plot_data_settings["plot_title_as_region"]:
                mplt.add_main_title(zone_input)            
                
            outputs[zone_input] = {'fig': fig, 'data_table': Data_Table_Out}
        return outputs

    def zonal_interchange(self, figure_name: str = None,
                          start_date_range: str = None,
                          end_date_range: str = None, **_):
        """Creates a line plot of the net interchange between each zone, with a facet for each zone.

        The method will only work if agg_by = "zone".

        The code will create either a timeseries or duration curve depending on 
        if the word 'duration_curve' is in the figure_name.
        To make a duration curve, ensure the word 'duration_curve' is found in the figure_name.

        Args:
            figure_name (str, optional): User defined figure output name.
                Defaults to None.
            start_date_range (str, optional): Defines a start date at which to represent data from. 
                Defaults to None.
            end_date_range (str, optional): Defines a end date at which to represent data to.
                Defaults to None.

        Returns:
            dict: dictionary containing the created plot and its data table
        """
        if self.AGG_BY not in ["zone", "zones", "Zone", "Zones"]:
            self.logger.warning("This plot only supports aggregation zone")
            return UnsupportedAggregation()
        
        duration_curve=False
        if 'duration_curve' in figure_name:
            duration_curve = True
            
        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True,"line_Flow",self.Scenarios)]
        
        # Runs get_formatted_data within PlotDataHelper to populate PlotDataHelper dictionary  
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)

        if 1 in check_input_data:
            return MissingInputData()

        outputs = {}

        # sets up x, y dimensions of plot
        ncols, nrows = self.set_facet_col_row_dimensions(multi_scenario=self.Scenarios)
        grid_size = ncols*nrows

        # Used to calculate any excess axis to delete
        plot_number = len(self.Scenarios)
        excess_axs = grid_size - plot_number

        for zone_input in self.Zones:

            self.logger.info(f"{self.AGG_BY} = {zone_input}")

            mplt = PlotLibrary(nrows, ncols, sharey=True,
                              squeeze=False, ravel_axs=True)
            fig, axs = mplt.get_figure()

            plt.subplots_adjust(wspace=0.1, hspace=0.5)

            net_exports_all = []

            for n, scenario in enumerate(self.Scenarios):
                net_exports = []

                exp_lines = self.meta.zone_exporting_lines(scenario)
                imp_lines = self.meta.zone_importing_lines(scenario)

                if exp_lines.empty or imp_lines.empty:
                    return MissingMetaData()

                exp_lines.columns = ['region','line_name']
                imp_lines.columns = ['region','line_name']

                #Find list of lines that connect each region.
                exp_oz = exp_lines[exp_lines['region'] == zone_input]
                imp_oz = imp_lines[imp_lines['region'] == zone_input]

                other_zones = self.meta.zones(scenario).name.tolist()
                try:
                    other_zones.remove(zone_input)
                except:
                    self.logger.warning("Are you sure you set agg_by = zone?")

                self.logger.info(f"Scenario = {str(scenario)}")
                flow = self["line_Flow"][scenario].copy()
                if self.shift_leapday == True:
                    flow = self.adjust_for_leapday(flow)
                flow = flow.reset_index()

                for other_zone in other_zones:
                    exp_other_oz = exp_lines[exp_lines['region'] == other_zone]
                    imp_other_oz = imp_lines[imp_lines['region'] == other_zone]

                    exp_pair = pd.merge(exp_oz, imp_other_oz, left_on='line_name',
                                        right_on='line_name')
                    imp_pair = pd.merge(imp_oz, exp_other_oz, left_on='line_name',
                                        right_on='line_name')

                    #Swap columns for importing lines
                    imp_pair = imp_pair.reindex(columns=['region_from', 'line_name', 'region_to'])

                    export = flow[flow['line_name'].isin(exp_pair['line_name'])]
                    imports = flow[flow['line_name'].isin(imp_pair['line_name'])]

                    export = export.groupby(['timestamp']).sum()
                    imports = imports.groupby(['timestamp']).sum()

                    #Check for situations where there are only exporting or importing lines for this zonal pair.
                    if imports.empty:
                        net_export = export
                    elif export.empty:
                        net_export = -imports
                    else:
                        net_export = export - imports
                    net_export.columns = [other_zone]

                    if pd.notna(start_date_range):
                        if other_zone == [other_zones[0]]:
                            self.logger.info(f"Plotting specific date range: \
                            {str(start_date_range)} to {str(end_date_range)}")

                        net_export = net_export[start_date_range : end_date_range]

                    if duration_curve:
                        net_export = self.sort_duration(net_export,other_zone)

                    net_exports.append(net_export)

                net_exports = pd.concat(net_exports,axis = 1)
                net_exports = net_exports.dropna(axis = 'columns')
                net_exports.index = pd.to_datetime(net_exports.index)
                net_exports['Net export'] = net_exports.sum(axis = 1)

                # unitconversion based off peak export hour, only checked once
                if zone_input == self.Zones[0] and scenario == self.Scenarios[0]:
                    unitconversion = self.capacity_energy_unitconversion(net_exports)

                net_exports = net_exports / unitconversion['divisor']

                if duration_curve:
                    net_exports = net_exports.reset_index().drop(columns = 'index')

                for column in net_exports:
                    linestyle = '--' if column == 'Net export' else 'solid'
                    mplt.lineplot(net_exports, column=column, label=column,
                                    sub_pos=n, linestyle=linestyle)

                axs[n].margins(x=0.01)
                #Add horizontal line at 0.
                axs[n].hlines(y=0, xmin=axs[n].get_xlim()[0], xmax=axs[n].get_xlim()[1],
                              linestyle=':') 
                if not duration_curve:
                    mplt.set_subplot_timeseries_format(sub_pos=n)

                #Add scenario column to output table.
                scenario_names = pd.Series([scenario] * len(net_exports), name='Scenario')
                net_exports = net_exports.add_suffix(f" ({unitconversion['units']})")
                net_exports = net_exports.set_index([scenario_names], append=True)
                net_exports_all.append(net_exports)

            mplt.add_facet_labels(xlabels=self.xlabels,
                                  ylabels = self.ylabels)
            mplt.add_legend()
            #Remove extra axes
            mplt.remove_excess_axs(excess_axs,grid_size)

            if plot_data_settings["plot_title_as_region"]:
                mplt.add_main_title(zone_input)
            plt.ylabel(f"Net export ({unitconversion['units']})", color='black', 
                       rotation='vertical', labelpad=40)
            if duration_curve:
                plt.xlabel('Sorted hour of the year', color='black', labelpad=30)
            
            Data_Table_Out = pd.concat(net_exports_all)
            # if plotting all scenarios return figures to plot_main
            outputs[zone_input] = {'fig': fig,'data_table' : Data_Table_Out}

        return outputs

    def zonal_interchange_total(self, start_date_range: str = None,
                                end_date_range: str = None, **_):
        """Creates a barplot of the net interchange between each zone, separated by positive and negative flows.

        The method will only work if agg_by = "zone".

        Args:
            start_date_range (str, optional): Defines a start date at which to represent data from. 
                Defaults to None.
            end_date_range (str, optional): Defines a end date at which to represent data to.
                Defaults to None.

        Returns:
            dict: dictionary containing the created plot and its data table
        """
        if self.AGG_BY not in ["zone", "zones", "Zone", "Zones"]:
            self.logger.warning("This plot only supports aggregation zone")
            return UnsupportedAggregation()
        
        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True,"line_Flow",self.Scenarios)]
        
        # Runs get_formatted_data within PlotDataHelper to populate PlotDataHelper dictionary  
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)

        if 1 in check_input_data:
            return MissingInputData()

        outputs = {}

        for zone_input in self.Zones:

            self.logger.info(f"{self.AGG_BY} = {zone_input}")
            
            mplt = PlotLibrary()
            fig, ax = mplt.get_figure()
            plt.subplots_adjust(wspace=0.05, hspace=0.2)

            net_exports_all = []
            # Holds each scenario output table
            data_out_chunk = []

            for n, scenario in enumerate(self.Scenarios):

                exp_lines = self.meta.zone_exporting_lines(scenario)
                imp_lines = self.meta.zone_importing_lines(scenario)

                if exp_lines.empty or imp_lines.empty:
                    return MissingMetaData()

                exp_lines.columns = ['region', 'line_name']
                imp_lines.columns = ['region', 'line_name']

                #Find list of lines that connect each region.
                exp_oz = exp_lines[exp_lines['region'] == zone_input]
                imp_oz = imp_lines[imp_lines['region'] == zone_input]

                other_zones = self.meta.zones(scenario).name.tolist()
                other_zones.remove(zone_input)

                net_exports = []
                self.logger.info(f"Scenario = {str(scenario)}")
                flow = self["line_Flow"][scenario]
                flow = flow.reset_index()

                for other_zone in other_zones:
                    exp_other_oz = exp_lines[exp_lines['region'] == other_zone]
                    imp_other_oz = imp_lines[imp_lines['region'] == other_zone]

                    exp_pair = pd.merge(exp_oz, imp_other_oz, left_on='line_name', 
                                        right_on='line_name')
                    imp_pair = pd.merge(imp_oz, exp_other_oz, left_on='line_name', 
                                        right_on='line_name')

                    #Swap columns for importing lines
                    imp_pair = imp_pair.reindex(columns=['region_from', 'line_name', 'region_to'])

                    export = flow[flow['line_name'].isin(exp_pair['line_name'])]
                    imports = flow[flow['line_name'].isin(imp_pair['line_name'])]

                    export = export.groupby(['timestamp']).sum()
                    imports = imports.groupby(['timestamp']).sum()

                    #Check for situations where there are only exporting or importing lines for this zonal pair.
                    if imports.empty:
                        net_export = export
                    elif export.empty:
                        net_export = -imports
                    else:
                        net_export = export - imports
                    net_export.columns = [other_zone]

                    if pd.notna(start_date_range):
                        if other_zone == other_zones[0]:
                            self.logger.info(f"Plotting specific date range: \
                            {str(start_date_range)} to {str(end_date_range)}")

                        net_export = net_export[start_date_range : end_date_range]

                    net_exports.append(net_export)

                net_exports = pd.concat(net_exports, axis=1)
                net_exports = net_exports.dropna(axis='columns')
                net_exports.index = pd.to_datetime(net_exports.index)
                net_exports['Net Export'] = net_exports.sum(axis=1)

                positive = net_exports.agg(lambda x: x[x>0].sum())
                negative = net_exports.agg(lambda x: x[x<0].sum())

                both = pd.concat([positive,negative], axis=1)
                both.columns = ["Total Export", "Total Import"]

                # unitconversion based off peak export hour, only checked once
                if scenario == self.Scenarios[0]:
                    unitconversion = self.capacity_energy_unitconversion(both)

                both = both / unitconversion['divisor']
                net_exports_all.append(both)

                #Add scenario column to output table.
                scenario_names = pd.Series([scenario] * len(both), name='Scenario')
                data_table = both.set_index([scenario_names], append=True)
                data_table = data_table.add_suffix(f" ({unitconversion['units']})")
                data_out_chunk.append(data_table)

            Data_Table_Out = pd.concat(data_out_chunk)
            
            #Make scenario/color dictionary.
            color_dict = dict(zip(self.Scenarios, self.color_list))
            
            mplt.clustered_stacked_barplot(net_exports_all, 
                                            labels=self.Scenarios, 
                                            color_dict=color_dict)
            ax.hlines(y=0, xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1], 
                      linestyle=':')
            ax.set_ylabel(f"Interchange ({unitconversion['units']}h)", color='black', 
                          rotation='vertical')
            if plot_data_settings["plot_title_as_region"]:
                mplt.add_main_title(zone_input)

            outputs[zone_input] = {'fig': fig,'data_table': Data_Table_Out}

        return outputs

    def total_int_flow_ind(self, prop: str = None, 
                           start_date_range: str = None,
                           end_date_range: str = None, **_):
        """Creates a clustered barplot of the total flow for a specific interface, separated by positive and negative flows.

        Specify the interface(s) of interest by providing a comma separated 
        string to the property entry.
        Scenarios are clustered together as different colored bars.
        If multiple interfaces are provided, each will be plotted as a 
        separate group of clustered bar.

        Args:
            prop (str, optional): Comma separated string of interfaces to plot. 
                Defaults to None.
            start_date_range (str, optional): Defines a start date at which to represent data from. 
                Defaults to None.
            end_date_range (str, optional): Defines a end date at which to represent data to.
                Defaults to None.

        Returns:
            DataSavedInModule: DataSavedInModule exception.
        """
        
        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True,"interface_Flow",self.Scenarios)]
        
        # Runs get_formatted_data within PlotDataHelper to populate PlotDataHelper dictionary  
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)

        if 1 in check_input_data:
            return MissingInputData()

        #Select only interfaces specified in Marmot_plot_select.csv.
        select_ints = prop.split(",")
        if select_ints == None:
            return InputSheetError()

        self.logger.info('Plotting only the interfaces specified in Marmot_plot_select.csv')
        self.logger.info(select_ints) 

        mplt = PlotLibrary()
        fig, ax = mplt.get_figure()
        plt.subplots_adjust(wspace=0.05, hspace=0.2)
        
        net_flows_all = []
        # Holds each scenario output table
        data_out_chunk = []
        
        for i, scenario in enumerate(self.Scenarios):
            
            self.logger.info(f"Scenario = {str(scenario)}")
            flow_all = self["interface_Flow"][scenario]
            pos = pd.Series(name='Total Export')
            neg = pd.Series(name='Total Import')
            
            available_inter = select_ints.copy()
            
            for inter in select_ints:
                if inter not in flow_all.index.get_level_values('interface_name'):
                    self.logger.info(f'{inter} Not in Data')
                    available_inter.remove(inter)
                    continue
                
                #Remove leading spaces
                if inter[0] == ' ':
                    inter = inter[1:]

                flow = flow_all.xs(inter, level='interface_name')
                flow = flow.reset_index()
                 
                if pd.notna(start_date_range):
                    self.logger.info("Plotting specific date range: \
                    {} to {}".format(str(start_date_range), str(end_date_range)))
                    flow = flow[start_date_range : end_date_range]
            
                flow = flow[0]

                pos_sing = pd.Series(flow.where(flow > 0).sum())
                pos = pos.append(pos_sing)
                neg_sing = pd.Series(flow.where(flow < 0).sum())
                neg = neg.append(neg_sing)

            both = pd.concat([pos,neg],axis = 1)
            both.columns = ['Total Export','Total Import']

            if scenario == self.Scenarios[0]:
                unitconversion = self.capacity_energy_unitconversion(both)

            both = both / unitconversion['divisor']
            both.index = available_inter
            net_flows_all.append(both)

            #Add scenario column to output table.
            scenario_names = pd.Series([scenario] * len(both),name = 'Scenario')
            data_table = both.set_index([scenario_names],append = True)
            data_table = data_table.add_suffix(f" ({unitconversion['units']})")
            data_out_chunk.append(data_table)
        
        Data_Table_Out = pd.concat(data_out_chunk)
         
        #Make scenario/color dictionary.
        color_dict = dict(zip(self.Scenarios, self.color_list))
        
        mplt.clustered_stacked_barplot(net_flows_all, 
                                        labels=self.Scenarios, 
                                        color_dict=color_dict)
        ax.hlines(y=0, xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1], 
                  linestyle=':')
        ax.set_ylabel('Flow ({}h)'.format(unitconversion['units']), color='black', 
                      rotation='vertical')
        ax.set_xlabel('')
        fig.savefig(os.path.join(self.Marmot_Solutions_folder, "Figures_Output", f"{self.AGG_BY }_transmission",
                                "Individual_Interface_Total_Flow.svg"), 
                    dpi=600, bbox_inches='tight')
        Data_Table_Out.to_csv(os.path.join(self.Marmot_Solutions_folder, "Figures_Output", f"{self.AGG_BY }_transmission",
                                           "Individual_Interface_Total_Flow.csv"))
        outputs = DataSavedInModule()
        return outputs
                
    ### Archived Code ####

    # def line_util_agged(self):

    #     self["line_Flow"] = {}
    #     interface_flow_collection = {}
    #     line_limit_collection = {}
    #     interface_limit_collection = {}

    #     #Load data
    #     self._getdata(self["line_Flow"],"line_Flow")
    #     self._getdata(interface_flow_collection,"interface_Flow")
    #     self._getdata(line_limit_collection,"line_Export_Limit")
    #     self._getdata(interface_limit_collection,"interface_Export_Limit")

    #     outputs = {}

    #     for zone_input in self.Zones:
    #         self.logger.info('Zone = ' + str(zone_input))

    #         all_scenarios = pd.DataFrame()

    #         for scenario in self.Scenarios:
    #             self.logger.info("Scenario = " + str(scenario))

    #             lineflow = self["line_Flow"].get(scenario)
    #             linelim = line_limit_collection.get(scenario)
    #             linelim = linelim.reset_index().drop('timestamp', axis = 1).set_index('line_name')

    #             #Calculate utilization.
    #             alllines = pd.merge(lineflow,linelim,left_index=True,right_index=True)
    #             alllines = alllines.rename(columns = {'0_x':'flow','0_y':'capacity'})
    #             alllines['flow'] = abs(alllines['flow'])

    #             #Merge in region ID and aggregate by region.
    #             if self.AGG_BY == 'region':
    #                 line2region = self.region_lines
    #             elif self.AGG_BY == 'zone':
    #                 line2region = self.zone_lines

    #             alllines = alllines.reset_index()
    #             alllines = alllines.merge(line2region, on = 'line_name')

    #             #Extract ReEDS expansion lines for next section.
    #             reeds_exp = alllines.merge((self.lines.rename(columns={"name":"line_name"})), on = 'line_name')
    #             reeds_exp = reeds_exp[reeds_exp['category'] == 'ReEDS_Expansion']
    #             reeds_agg = reeds_exp.groupby(['timestamp',self.AGG_BY],as_index = False).sum()

    #             #Subset only enforced lines. This will subset to only EI lines.
    #             enforced = pd.read_csv('/projects/continental/pcm/Results/enforced_lines.csv')
    #             enforced.columns = ['line_name']
    #             lineutil = pd.merge(enforced,alllines, on = 'line_name')

    #             lineutil['line util'] = lineutil['flow'] / lineutil['capacity']
    #             lineutil = lineutil[lineutil.capacity < 10000]

    #             #Drop duplicates if AGG_BY == Interconnection.

    #             #Aggregate by region, merge in region mapping.
    #             agg = alllines.groupby(['timestamp',self.AGG_BY],as_index = False).sum()
    #             agg['util'] = agg['flow'] / agg['capacity']
    #             agg = agg.rename(columns = {'util' : scenario})
    #             onezone = agg[agg[self.AGG_BY] == zone_input]
    #             onezone = onezone.set_index('timestamp')[scenario]

    #             #If zone_input is in WI or ERCOT, the dataframe will be empty here. Lines are not enforced here. Instead, use interfaces.
    #             if onezone.empty:

    #                 #Start with interface flow.
    #                 intflow = interface_flow_collection.get(scenario)
    #                 intlim = interface_limit_collection.get(scenario)

    #                 allint = pd.merge(intflow,intlim,left_index=True,right_index=True)
    #                 allint = allint.rename(columns = {'0_x':'flow','0_y':'capacity'})
    #                 allint = allint.reset_index()

    #                 #Merge in interface/line/region mapping.
    #                 line2int = self.interface_lines
    #                 line2int = line2int.rename(columns = {'line' : 'line_name','interface' : 'interface_name'})
    #                 allint = allint.merge(line2int, on = 'interface_name', how = 'inner')
    #                 allint = allint.merge(line2region, on = 'line_name')
    #                 allint = allint.merge(self.Region_Mapping, on = 'region')
    #                 allint = allint.drop(columns = 'line_name')
    #                 allint = allint.drop_duplicates() #Merging in line info duplicated most of the interfaces.

    #                 agg = allint.groupby(['timestamp',self.AGG_BY],as_index = False).sum()
    #                 agg = pd.concat([agg,reeds_agg]) #Add in ReEDS expansion lines, re-aggregate.
    #                 agg = agg.groupby(['timestamp',self.AGG_BY],as_index = False).sum()
    #                 agg['util'] = agg['flow'] / agg['capacity']
    #                 agg = agg.rename(columns = {'util' : scenario})
    #                 onezone = agg[agg[self.AGG_BY] == zone_input]
    #                 onezone = onezone.set_index('timestamp')[scenario]

    #             if (prop != prop) == False: #Show only subset category of lines.
    #                 reeds_agg = reeds_agg.groupby('timestamp',as_index = False).sum()
    #                 reeds_agg['util'] = reeds_agg['flow'] / reeds_agg['capacity']
    #                 onezone = reeds_agg.rename(columns = {'util' : scenario})
    #                 onezone = onezone.set_index('timestamp')[scenario]

    #             all_scenarios = pd.concat([all_scenarios,onezone], axis = 1)

    #         # Data table of values to return to main program
    #         Data_Table_Out = all_scenarios.copy()

    #         #Make scenario/color dictionary.
    #         scenario_color_dict = {}
    #         for idx,column in enumerate(all_scenarios.columns):
    #             dictionary = {column : self.color_list[idx]}
    #             scenario_color_dict.update(dictionary)

    #         all_scenarios.index = pd.to_datetime(all_scenarios.index)

    #         fig, ax = plt.subplots(figsize=(9,6))
    #         for idx,column in enumerate(all_scenarios.columns):
    #             ax.plot(all_scenarios.index.values,all_scenarios[column], linewidth=2, color = scenario_color_dict.get(column,'#333333'),label=column)

    #         ax.set_ylabel('Transmission utilization (%)',  color='black', rotation='vertical')
    #         ax.set_xlabel(timezone,  color='black', rotation='horizontal')
    #         ax.spines['right'].set_visible(False)
    #         ax.spines['top'].set_visible(False)
    #         ax.tick_params(axis='y', which='major', length=5, width=1)
    #         ax.tick_params(axis='x', which='major', length=5, width=1)
    #         ax.margins(x=0.01)

    #         locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
    #         formatter = mdates.ConciseDateFormatter(locator)
    #         formatter.formats[2] = '%d\n %b'
    #         formatter.zero_formats[1] = '%b\n %Y'
    #         formatter.zero_formats[2] = '%d\n %b'
    #         formatter.zero_formats[3] = '%H:%M\n %d-%b'
    #         formatter.offset_formats[3] = '%b %Y'
    #         formatter.show_offset = False
    #         ax.xaxis.set_major_locator(locator)
    #         ax.xaxis.set_major_formatter(formatter)
    #         ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    #         ax.hlines(y = 1, xmin = ax.get_xlim()[0], xmax = ax.get_xlim()[1], linestyle = ':') #Add horizontal line at 100%.

    #         handles, labels = ax.get_legend_handles_labels()

    #         #Legend 1
    #         leg1 = ax.legend(reversed(handles), reversed(labels), loc='best',facecolor='inherit', frameon=True)

    #         # Manually add the first legend back
    #         ax.add_artist(leg1)

    #         outputs[zone_input] = {'fig': fig, 'data_table': Data_Table_Out}
    #     return outputs


#     def region_region_duration(self):

#         rr_int = pd.read_hdf(os.path.join(self.Marmot_Solutions_folder,self.Scenarios[0],"Processed_HDF5_folder", self.Scenarios[0] + "_formatted.h5"),"region_regions_Net_Interchange")
#         agg_region_mapping = self.Region_Mapping[['region',self.AGG_BY]].set_index('region').to_dict()[self.AGG_BY]

#         rr_int = rr_int.reset_index()
#         rr_int['parent'] = rr_int['parent'].map(agg_region_mapping)
#         rr_int['child']  = rr_int['child'].map(agg_region_mapping)

#         rr_int_agg = rr_int.groupby(['parent','child'],as_index=True).sum() # Determine annual net flow between regions.
#         rr_int_agg.rename(columns = {0:'flow (MW)'}, inplace = True)
#         rr_int_agg = rr_int_agg.reset_index(['parent','child'])
#         rr_int_agg=rr_int_agg.loc[rr_int_agg['flow (MW)']>0.01] # Keep only positive flows
# #        rr_int_agg.set_index(['parent','child'],inplace=True)

#         rr_int_agg['path']=rr_int_agg['parent']+"_"+rr_int_agg['child']

#         if (prop!=prop)==False: # This checks for a nan in string. If a number of paths is selected only plot those
#             pathlist=rr_int_agg.sort_values(ascending=False,by='flow (MW)')['path'][1:int(prop)+1] #list of top paths based on number selected
#         else:
#             pathlist=rr_int_agg['path'] #List of paths


#         rr_int_hr = rr_int.groupby(['timestamp','parent','child'],as_index=True).sum() # Hourly flow
#         rr_int_hr.rename(columns = {0:'flow (MW)'}, inplace = True)
#         rr_int_hr.reset_index(['timestamp','parent','child'],inplace=True)
#         rr_int_hr['path']=rr_int_hr['parent']+"_"+rr_int_hr['child']
#         rr_int_hr.set_index(['path'],inplace=True)
#         rr_int_hr['Abs MW']=abs(rr_int_hr['flow (MW)'])
#         rr_int_hr['Abs MW'].sum()
#         rr_int_hr.loc[pathlist]['Abs MW'].sum()*2  # Check that the sum of the absolute value of flows is the same. i.e. only redundant flows were eliminated.
#         rr_int_hr=rr_int_hr.loc[pathlist].drop(['Abs MW'],axis=1)

#         ## Plot duration curves
#         fig, ax3 = plt.subplots(figsize=(9,6))
#         for i in pathlist:
#             duration_curve = rr_int_hr.loc[i].sort_values(ascending=False,by='flow (MW)').reset_index()
#             plt.plot(duration_curve['flow (MW)'],label=i)
#             del duration_curve

#         ax3.set_ylabel('flow MW',  color='black', rotation='vertical')
#         ax3.set_xlabel('Intervals',  color='black', rotation='horizontal')
#         ax3.spines['right'].set_visible(False)
#         ax3.spines['top'].set_visible(False)

# #        if (prop!=prop)==False: # This checks for a nan in string. If no limit selected, do nothing
#         ax3.legend(loc='best')
# #            plt.lim((0,int(prop)))

#         Data_Table_Out = rr_int_hr

        # return {'fig': fig, 'data_table': Data_Table_Out}
