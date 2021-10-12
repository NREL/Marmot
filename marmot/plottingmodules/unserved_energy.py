# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 07:42:06 2020

This module creates unserved energy timeseries line plots and total bar
plots and is called from marmot_plot_main.py
@author: Daniel Levie 
"""

import logging
import pandas as pd
import matplotlib as mpl
import matplotlib.dates as mdates

import marmot.config.mconfig as mconfig
import marmot.plottingmodules.plotutils.plot_library as plotlib
from marmot.plottingmodules.plotutils.plot_data_helper import PlotDataHelper
from marmot.plottingmodules.plotutils.plot_exceptions import (MissingInputData, MissingZoneData)


class MPlot(PlotDataHelper):
    """Marmot MPlot Class, common across all plotting modules.

    All the plotting modules use this same class name.
    This class contains plotting methods that are grouped based on the
    current module name.
    
    The unserved_energy.py module contains methods that are
    related to unserved energy in the power system. 

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
        self.y_axes_decimalpt = mconfig.parser("axes_options","y_axes_decimalpt")
        

    def unserved_energy_timeseries(self, timezone: str = "",
                                   start_date_range: str = None, 
                                   end_date_range: str = None, **_):
        """Creates a timeseries line plot of total unserved energy.

        Each sceanrio is plotted as a separate line.

        Args:
            timezone (str, optional): The timezone to display on the x-axes.
                Defaults to "".
            start_date_range (str, optional): Defines a start date at which to represent data from. 
                Defaults to None.
            end_date_range (str, optional): Defines a end date at which to represent data to.
                Defaults to None.

        Returns:
            dict: Dictionary containing the created plot and its data table.
        """
        outputs = {}
        
        if self.AGG_BY == 'zone':
            agg = 'zone'
        else:
            agg = 'region'
            
        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True, f"{agg}_Unserved_Energy", self.Scenarios)]
        
        # Runs get_formatted_data within PlotDataHelper to populate PlotDataHelper dictionary  
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)

        if 1 in check_input_data:
            return MissingInputData()
        
        for zone_input in self.Zones:
            self.logger.info(f'Zone = {zone_input}')
            Unserved_Energy_Timeseries_Out = pd.DataFrame()

            for scenario in self.Scenarios:
                self.logger.info(f'Scenario = {scenario}')

                unserved_eng_timeseries = self[f"{agg}_Unserved_Energy"].get(scenario)
                unserved_eng_timeseries = unserved_eng_timeseries.xs(zone_input,level=self.AGG_BY)
                unserved_eng_timeseries = unserved_eng_timeseries.groupby(["timestamp"]).sum()

                if pd.notna(start_date_range):
                    self.logger.info(f"Plotting specific date range: \
                                      {str(start_date_range)} to {str(end_date_range)}")
                    unserved_eng_timeseries = unserved_eng_timeseries[start_date_range : end_date_range]

                unserved_eng_timeseries = unserved_eng_timeseries.squeeze() #Convert to Series
                unserved_eng_timeseries.rename(scenario, inplace=True)
                Unserved_Energy_Timeseries_Out = pd.concat([Unserved_Energy_Timeseries_Out, unserved_eng_timeseries], 
                                                           axis=1, sort=False).fillna(0)

            Unserved_Energy_Timeseries_Out.columns = Unserved_Energy_Timeseries_Out.columns.str.replace('_',' ')
            Unserved_Energy_Timeseries_Out = Unserved_Energy_Timeseries_Out.loc[:, (Unserved_Energy_Timeseries_Out >= 1)
                                                                                    .any(axis=0)]

            if Unserved_Energy_Timeseries_Out.empty==True:
                self.logger.info(f'No Unserved Energy in {zone_input}')
                out = MissingZoneData()
                outputs[zone_input] = out
                continue
            
            # Determine auto unit coversion
            unitconversion = PlotDataHelper.capacity_energy_unitconversion(Unserved_Energy_Timeseries_Out.values.max())
            Unserved_Energy_Timeseries_Out = Unserved_Energy_Timeseries_Out/unitconversion['divisor'] 
            
            # Data table of values to return to main program
            Data_Table_Out = Unserved_Energy_Timeseries_Out.add_suffix(f" ({unitconversion['units']})")
            
            fig1, axs = plotlib.setup_plot()
            #flatten object
            ax = axs[0]
            # Converts color_list into an iterable list for use in a loop
            iter_colour = iter(self.color_list)

            for column in Unserved_Energy_Timeseries_Out:
                ax.plot(Unserved_Energy_Timeseries_Out[column], linewidth=3, 
                        antialiased=True, color=next(iter_colour), label=column)
                ax.legend(loc='lower left',bbox_to_anchor=(1,0),
                          facecolor='inherit', frameon=True)
            ax.set_ylabel(f"Unserved Energy ({unitconversion['units']})", 
                          color='black', rotation='vertical')
            ax.set_ylim(bottom=0)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.tick_params(axis='y', which='major', length=5, width=1)
            ax.tick_params(axis='x', which='major', length=5, width=1)
            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(
                                         lambda x, p: format(x, f',.{self.y_axes_decimalpt}f')))
            ax.margins(x=0.01)
            PlotDataHelper.set_plot_timeseries_format(axs)
            if mconfig.parser("plot_title_as_region"):
                ax.set_title(zone_input)
            outputs[zone_input] = {'fig': fig1, 'data_table': Data_Table_Out}

        return outputs

    def tot_unserved_energy(self, start_date_range: str = None, 
                            end_date_range: str = None, **_):
        """Creates a bar plot of total unserved energy.

        Each sceanrio is plotted as a separate bar.

        Args:
            start_date_range (str, optional): Defines a start date at which to represent data from. 
                Defaults to None.
            end_date_range (str, optional): Defines a end date at which to represent data to.
                Defaults to None.

        Returns:
            dict: Dictionary containing the created plot and its data table.
        """
        outputs = {}
        
        if self.AGG_BY == 'zone':
            agg = 'zone'
        else:
            agg = 'region'
            
        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True, f"{agg}_Unserved_Energy", self.Scenarios)]
        
        # Runs get_formatted_data within PlotDataHelper to populate PlotDataHelper dictionary  
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)

        if 1 in check_input_data:
            return MissingInputData()

        for zone_input in self.Zones:
            Unserved_Energy_Timeseries_Out = pd.DataFrame()
            Total_Unserved_Energy_Out = pd.DataFrame()
            self.logger.info(f"{self.AGG_BY} = {zone_input}")
            
            for scenario in self.Scenarios:
                self.logger.info(f'Scenario = {scenario}')

                unserved_eng_timeseries = self[f"{agg}_Unserved_Energy"].get(scenario)
                unserved_eng_timeseries = unserved_eng_timeseries.xs(zone_input,level=self.AGG_BY)
                unserved_eng_timeseries = unserved_eng_timeseries.groupby(["timestamp"]).sum()
                
                # correct sum for non-hourly runs
                interval_count = PlotDataHelper.get_sub_hour_interval_count(unserved_eng_timeseries)

                if pd.notna(start_date_range):
                    self.logger.info(f"Plotting specific date range: \
                                      {str(start_date_range)} to {str(end_date_range)}")
                    unserved_eng_timeseries = unserved_eng_timeseries[start_date_range : end_date_range]
                    
                unserved_eng_timeseries = unserved_eng_timeseries.squeeze() #Convert to Series
                unserved_eng_timeseries.rename(scenario, inplace=True)
                                
                Unserved_Energy_Timeseries_Out = pd.concat([Unserved_Energy_Timeseries_Out, unserved_eng_timeseries],
                                                           axis=1, sort=False).fillna(0)

            Unserved_Energy_Timeseries_Out.columns = Unserved_Energy_Timeseries_Out.columns.str.replace('_',' ')

            
            Unserved_Energy_Timeseries_Out = Unserved_Energy_Timeseries_Out/interval_count

            Total_Unserved_Energy_Out.index = Total_Unserved_Energy_Out.index.str.replace('_',' ')
            Total_Unserved_Energy_Out = pd.DataFrame(Total_Unserved_Energy_Out.T)

            if Total_Unserved_Energy_Out.values.sum() == 0:
                self.logger.info(f'No Unserved Energy in {zone_input}')
                out = MissingZoneData()
                outputs[zone_input] = out
                continue
            
            # Determine auto unit coversion
            unitconversion = PlotDataHelper.capacity_energy_unitconversion(Total_Unserved_Energy_Out.values.max())
            Total_Unserved_Energy_Out = Total_Unserved_Energy_Out/unitconversion['divisor']
            
            # Data table of values to return to main program
            Data_Table_Out = Total_Unserved_Energy_Out.add_suffix(f" ({unitconversion['units']})")
            
            # create color dictionary
            color_dict = dict(zip(Total_Unserved_Energy_Out.index,self.color_list))
            fig2, axs = plotlib.setup_plot()
            #flatten object
            ax=axs[0]
            
            plotlib.create_bar_plot(Total_Unserved_Energy_Out.T, ax, color_dict)
            ax.set_ylabel(f"Total Unserved Energy ({unitconversion['units']}h)", 
                          color='black', rotation='vertical')
            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(
                                         lambda x, p: format(x, f',.{self.y_axes_decimalpt}f')))
            ax.xaxis.set_visible(False)
            ax.margins(x=0.01)
            
            if len(self.custom_xticklabels) > 1:
                tick_labels = self.custom_xticklabels
            else:
                tick_labels = Total_Unserved_Energy_Out.columns
            PlotDataHelper.set_barplot_xticklabels(tick_labels, ax=ax)

            ax.tick_params(axis='y', which='major', length=5, width=1)
            ax.tick_params(axis='x', which='major', length=5, width=1)
            ax.legend(loc='lower left',bbox_to_anchor=(1,0),
                          facecolor='inherit', frameon=True)
            if mconfig.parser("plot_title_as_region"):
                ax.set_title(zone_input)  
            for patch in ax.patches:
                width, height = patch.get_width(), patch.get_height()
                if height<=1:
                    continue
                x, y = patch.get_xy()
                ax.text(x+width/2,
                    y+height/2,
                    '{:,.1f}'.format(height),
                    horizontalalignment='center',
                    verticalalignment='center', fontsize=13)

            outputs[zone_input] = {'fig': fig2, 'data_table': Data_Table_Out}
        return outputs

    def average_diurnal_ue(self, start_date_range: str = None, 
                           end_date_range: str = None, **_):
        """Creates a line plot of average diurnal unserved energy.

        Each scenario is plotted as a separate line and shows the average 
        hourly unserved energy over a 24 hour period averaged across the entire year
        or time period defined.

        Args:
            start_date_range (str, optional): Defines a start date at which to represent data from. 
                Defaults to None.
            end_date_range (str, optional): Defines a end date at which to represent data to.
                Defaults to None.

        Returns:
            dict: Dictionary containing the created plot and its data table.
        """
        outputs = {}
        
        if self.AGG_BY == 'zone':
            agg = 'zone'
        else:
            agg = 'region'
        
        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True,f"{agg}_Unserved_Energy",self.Scenarios)]
        
        # Runs get_formatted_data within PlotDataHelper to populate PlotDataHelper dictionary  
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)

        if 1 in check_input_data:
            return MissingInputData()

        for zone_input in self.Zones:
            self.logger.info(f"{self.AGG_BY} = {zone_input}")
            
            chunks =[]
            for scenario in self.Scenarios:
                self.logger.info(f"Scenario = {scenario}")
                
                unserved_energy = self[f"{agg}_Unserved_Energy"][scenario]
                try:
                    unserved_energy = unserved_energy.xs(zone_input,level=self.AGG_BY)
                except KeyError:
                    self.logger.info(f'No unserved energy in {zone_input}')
                
                unserved_energy = unserved_energy.groupby(["timestamp"]).sum()
                unserved_energy = unserved_energy.squeeze()

                if pd.isna(start_date_range) == False:
                    self.logger.info(f"Plotting specific date range: \
                    {str(start_date_range)} to {str(end_date_range)}")
                    unserved_energy = unserved_energy[start_date_range : end_date_range]
                    
                    if unserved_energy.empty is True: 
                        self.logger.warning('No data in selected Date Range')
                        continue
                
                interval_count = PlotDataHelper.get_sub_hour_interval_count(unserved_energy)
                unserved_energy = unserved_energy/interval_count
                # Group data by hours and find mean across entire range 
                unserved_energy = unserved_energy.groupby([unserved_energy.index.hour]).mean()
                
                # reset index to datetime 
                unserved_energy.index = pd.date_range("2024-01-01", periods=24, freq="H")
                unserved_energy.rename(scenario, inplace=True)
                chunks.append(unserved_energy)

            unserved_energy_out = pd.concat(chunks, axis=1, sort=False)

            # Replace _ with white space
            unserved_energy_out.columns = unserved_energy_out.columns.str.replace('_',' ')

            # Create Dictionary from scenario names and color list
            colour_dict = dict(zip(unserved_energy_out.columns, self.color_list))

            fig, axs = plotlib.setup_plot()
            # flatten object
            ax = axs[0]

            unitconversion = PlotDataHelper.capacity_energy_unitconversion(unserved_energy_out.values.max())
            unserved_energy_out = unserved_energy_out / unitconversion['divisor']
            Data_Table_Out = unserved_energy_out
            Data_Table_Out = Data_Table_Out.add_suffix(f" ({unitconversion['units']})")
            
            for column in unserved_energy_out:
                ax.plot(unserved_energy_out[column], linewidth=3, color=colour_dict[column],
                        label=column)
            ax.legend(loc='lower left',bbox_to_anchor=(1,0),
                    facecolor='inherit', frameon=True)
            ax.set_ylabel(f"Average Diurnal Unserved Energy ({unitconversion['units']})", 
                          color='black', rotation='vertical')
            
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.tick_params(axis='y', which='major', length=5, width=1)
            ax.tick_params(axis='x', which='major', length=5, width=1)
            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(
                                         lambda x, p: format(x, f',.{self.y_axes_decimalpt}f')))
            ax.margins(x=0.01)
            
            # Set time ticks
            locator = mdates.AutoDateLocator(minticks=8, maxticks=12)
            formatter = mdates.ConciseDateFormatter(locator)
            formatter.zero_formats[3] = '%H:%M'
            formatter.show_offset = False
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)
            
            ax.set_ylim(bottom=0)

            if mconfig.parser("plot_title_as_region"):
                ax.set_title(zone_input)

            outputs[zone_input] = {'fig': fig, 'data_table': Data_Table_Out}
        return outputs
