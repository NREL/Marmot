# -*- coding: utf-8 -*-
"""Thermal capacity plots.

This module plots figures which show the amount of thermal capacity 
available but not committed (i.e in reserve)

@author: Daniel Levie and Marty Schwarz
"""

import logging
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

import marmot.config.mconfig as mconfig
from marmot.plottingmodules.plotutils.plot_library import PlotLibrary
from marmot.plottingmodules.plotutils.plot_data_helper import PlotDataHelper
from marmot.plottingmodules.plotutils.plot_exceptions import (MissingInputData, MissingZoneData)


class MPlot(PlotDataHelper):
    """thermal_cap_reserve MPlot class.

    All the plotting modules use this same class name.
    This class contains plotting methods that are grouped based on the
    current module name.
    
    The thermal_cap_reserve module contains methods that
    display the amount of generation in reserve, i.e non committed capacity.
    
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
                    self.xlabels, self.gen_names_dict, Region_Mapping=self.Region_Mapping) 

        self.logger = logging.getLogger('marmot_plot.'+__name__)
        self.y_axes_decimalpt = mconfig.parser("axes_options","y_axes_decimalpt")
        
        
    def thermal_cap_reserves(self, start_date_range: str = None, 
                             end_date_range: str = None, **_):
        """Plots the total thermal generation capacity that is not committed, i.e in reserve.

        If multiple scenarios are included, each one will be plotted on a 
        separate facet subplot.

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
        properties = [(True,"generator_Generation",self.Scenarios),
                      (True,"generator_Available_Capacity",self.Scenarios)]
        
        # Runs get_formatted_data within PlotDataHelper to populate PlotDataHelper dictionary  
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)

        if 1 in check_input_data:
            return MissingInputData()
        
        for zone_input in self.Zones:
            self.logger.info(f"Zone = {zone_input}")
                
            # sets up x, y dimensions of plot
            xdimension, ydimension = self.setup_facet_xy_dimensions(multi_scenario=self.Scenarios)
            
            grid_size = xdimension*ydimension

            # Used to calculate any excess axis to delete
            plot_number = len(self.Scenarios)
            excess_axs = grid_size - plot_number
            
            mplt = PlotLibrary(ydimension, xdimension, sharey=True, 
                                squeeze=False, ravel_axs=True)
            fig, axs = mplt.get_figure()
            plt.subplots_adjust(wspace=0.05, hspace=0.2)
            
            data_table_chunks = []

            for i, scenario in enumerate(self.Scenarios):

                self.logger.info(f"Scenario = {scenario}")

                Gen = self["generator_Generation"].get(scenario).copy()
                if self.shift_leapday == True:
                    Gen = self.adjust_for_leapday(Gen)
                avail_cap = self["generator_Available_Capacity"].get(scenario).copy()
                if self.shift_leapday == True:
                    avail_cap = self.adjust_for_leapday(avail_cap)               
               
                # Check if zone is in avail_cap
                try:
                    avail_cap = avail_cap.xs(zone_input,level = self.AGG_BY)
                except KeyError:
                    self.logger.warning(f"No installed capacity in: {zone_input}")
                    break
                Gen = Gen.xs(zone_input,level = self.AGG_BY)
                avail_cap = self.df_process_gen_inputs(avail_cap)
                Gen = self.df_process_gen_inputs(Gen)
                Gen = Gen.loc[:, (Gen != 0).any(axis=0)]

                thermal_reserve = avail_cap - Gen         
                non_thermal_gen = set(thermal_reserve.columns) - set(self.thermal_gen_cat)
                # filter for only thermal generation 
                thermal_reserve = thermal_reserve.drop(labels = non_thermal_gen, axis=1)
                
                #Convert units
                if i == 0:
                    unitconversion = self.capacity_energy_unitconversion(max(thermal_reserve.sum(axis=1)))
                thermal_reserve = thermal_reserve / unitconversion['divisor']

                # Check if thermal_reserve contains data, if not skips
                if thermal_reserve.empty == True:
                    out = MissingZoneData()
                    outputs[zone_input] = out
                    continue
                   
                if pd.notna(start_date_range):
                    self.logger.info(f"Plotting specific date range: \
                    {str(start_date_range)} to {str(end_date_range)}")
                    thermal_reserve = thermal_reserve[start_date_range : end_date_range]
                
                # Create data table for each scenario
                scenario_names = pd.Series([scenario]*len(thermal_reserve),name='Scenario')
                data_table = thermal_reserve.add_suffix(f" ({unitconversion['units']})")
                data_table = data_table.set_index([scenario_names],append=True)
                data_table_chunks.append(data_table)
                
                mplt.stackplot(thermal_reserve, color_dict=self.PLEXOS_color_dict,
                                labels=thermal_reserve.columns, n=i)

                axs[i].yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(x, f',.{self.y_axes_decimalpt}f')))
                axs[i].margins(x=0.01)
                mplt.set_plot_timeseries_format(n=i)
            
            # add facet labels
            mplt.add_facet_labels(alternative_xlabels=self.xlabels,
                                  alternative_ylabels=self.ylabels)   
            # Add legend
            mplt.add_legend(reverse_legend=True, sort_by=self.ordered_gen)
            # Remove extra axes
            mplt.remove_excess_axs(excess_axs,grid_size)            
            plt.ylabel(f"Thermal capacity reserve ({unitconversion['units']})", 
                        color='black', rotation='vertical', labelpad=40)
            if mconfig.parser("plot_title_as_region"):
                plt.title(zone_input)
            # If data_table_chunks is empty, does not return data or figure
            if not data_table_chunks:
                outputs[zone_input] = MissingZoneData()
                continue
            
            # Concat all data tables together
            Data_Table_Out = pd.concat(data_table_chunks, copy=False, axis=0)
            
            outputs[zone_input] = {'fig': fig, 'data_table': Data_Table_Out}
        return outputs
