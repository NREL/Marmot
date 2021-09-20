# -*- coding: utf-8 -*-
"""
Created on Mon Dec 9 10:34:48 2019
Updated July 26th 16:20:00 2021

This module plots figures which show the amount of thermal capacity 
available but not commited (i.e in reserve)
@author: Daniel Levie and Marty Schwarz
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch

import marmot.config.mconfig as mconfig
import marmot.plottingmodules.plotutils.plot_library as plotlib
from marmot.plottingmodules.plotutils.plot_data_helper import PlotDataHelper
from marmot.plottingmodules.plotutils.plot_exceptions import (MissingInputData, MissingZoneData)


class MPlot(PlotDataHelper):

    def __init__(self, argument_dict):
        # iterate over items in argument_dict and set as properties of class
        # see key_list in Marmot_plot_main for list of properties
        for prop in argument_dict:
            self.__setattr__(prop, argument_dict[prop])
        
        # Instantiation of MPlotHelperFunctions
        super().__init__(self.AGG_BY, self.ordered_gen, self.PLEXOS_color_dict, 
                    self.Scenarios, self.Marmot_Solutions_folder, self.ylabels, 
                    self.xlabels, self.gen_names_dict, self.Region_Mapping) 

        self.logger = logging.getLogger('marmot_plot.'+__name__)
        self.y_axes_decimalpt = mconfig.parser("axes_options","y_axes_decimalpt")
        
        self.mplot_data_dict = {}

    def thermal_cap_reserves(self, figure_name=None, prop=None, start=None, 
                             end=None, timezone="", start_date_range=None, 
                             end_date_range=None):
        """ 
        Plots the total thermal generation capacity that is not commited, 
        i.e in reserve
        
        If multiple scenarios are included, each one will be plotted on a 
        seperate subplot
        
        """
        outputs = {}
        
        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True,"generator_Generation",self.Scenarios),
                      (True,"generator_Available_Capacity",self.Scenarios)]
        
        # Runs get_data to populate mplot_data_dict with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_data(self.mplot_data_dict, properties)

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
            
            fig1, axs = plotlib.setup_plot(xdimension,ydimension)
            plt.subplots_adjust(wspace=0.05, hspace=0.2)
            
            # holds list of unique generation technologies
            unique_tech_names = []
            data_table_chunks = []

            for i, scenario in enumerate(self.Scenarios):

                self.logger.info(f"Scenario = {scenario}")

                Gen = self.mplot_data_dict["generator_Generation"].get(scenario).copy()
                if self.shift_leapday == True:
                    Gen = self.adjust_for_leapday(Gen)
                avail_cap = self.mplot_data_dict["generator_Available_Capacity"].get(scenario).copy()
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
                    unitconversion = PlotDataHelper.capacity_energy_unitconversion(max(thermal_reserve.sum(axis=1)))
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
                
                axs[i].stackplot(thermal_reserve.index.values, thermal_reserve.values.T, labels = thermal_reserve.columns, linewidth=0,
                             colors = [self.PLEXOS_color_dict.get(x, '#333333') for x in thermal_reserve.T.index])

                axs[i].spines['right'].set_visible(False)
                axs[i].spines['top'].set_visible(False)
                axs[i].tick_params(axis='y', which='major', length=5, width=1)
                axs[i].tick_params(axis='x', which='major', length=5, width=1)
                axs[i].yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(x, f',.{self.y_axes_decimalpt}f')))
                axs[i].margins(x=0.01)
                PlotDataHelper.set_plot_timeseries_format(axs,i)
                
                # create list of unique gen technologies
                l1 = thermal_reserve.columns.tolist()
                unique_tech_names.extend(l1)

            # create labels list of unique tech names then order
            labels = np.unique(np.array(unique_tech_names)).tolist()
            labels.sort(key = lambda i:self.ordered_gen.index(i))
            
            handles = []
            # create custom gen_tech legend
            for tech in labels:
                gen_legend_patches = Patch(facecolor=self.PLEXOS_color_dict[tech],
                            alpha=1.0)
                handles.append(gen_legend_patches)
            
            #Place legend on right side of bottom right plot
            axs[grid_size-1].legend(reversed(handles),reversed(labels),
                                    loc = 'lower left',bbox_to_anchor=(1.05,0),
                                    facecolor='inherit', frameon=True)

            # add facet labels
            self.add_facet_labels(fig1)    
            
            # Remove extra axes
            if excess_axs != 0:
                PlotDataHelper.remove_excess_axs(axs,excess_axs,grid_size)
            
            fig1.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            plt.ylabel(f"Thermal capacity reserve ({unitconversion['units']})",  color='black', rotation='vertical', labelpad=40)
            if mconfig.parser("plot_title_as_region"):
                plt.title(zone_input)
            # If data_table_chunks is empty, does not return data or figure
            if not data_table_chunks:
                outputs[zone_input] = MissingZoneData()
                continue
            
            # Concat all data tables together
            Data_Table_Out = pd.concat(data_table_chunks, copy=False, axis=0)
            
            outputs[zone_input] = {'fig': fig1, 'data_table': Data_Table_Out}
        return outputs
