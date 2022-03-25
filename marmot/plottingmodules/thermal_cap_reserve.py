# -*- coding: utf-8 -*-
"""Thermal capacity plots.

This module plots figures which show the amount of thermal capacity 
available but not committed (i.e in reserve)

@author: Daniel Levie and Marty Schwarz
"""

import logging
import pandas as pd
import matplotlib.pyplot as plt

import marmot.utils.mconfig as mconfig

from marmot.plottingmodules.plotutils.plot_library import PlotLibrary
from marmot.plottingmodules.plotutils.plot_data_helper import MPlotDataHelper
from marmot.plottingmodules.plotutils.plot_exceptions import (MissingInputData, MissingZoneData)

logger = logging.getLogger('plotter.'+__name__)   
plot_data_settings : dict = mconfig.parser("plot_data")
shift_leapday : bool = mconfig.parser("shift_leapday")

class ThermalReserve(MPlotDataHelper):
    """Thermal capacity in reserve plots.

    The thermal_cap_reserve module contains methods that
    display the amount of generation in reserve, i.e non committed capacity.
    
    ThermalReserve inherits from the MPlotDataHelper class to assist 
    in creating figures.
    """

    def __init__(self, **kwargs):
        # Instantiation of MPlotHelperFunctions
        super().__init__(**kwargs)     
        
    def thermal_cap_reserves(self, start_date_range: str = None, 
                             end_date_range: str = None,
                             data_resolution: str = "", **_):
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
        outputs : dict = {}
        
        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True, f"generator_Generation{data_resolution}", self.Scenarios),
                      (True, f"generator_Available_Capacity{data_resolution}", self.Scenarios)]
        
        # Runs get_formatted_data within MPlotDataHelper to populate MPlotDataHelper dictionary  
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)

        if 1 in check_input_data:
            return MissingInputData()
        
        for zone_input in self.Zones:
            logger.info(f"Zone = {zone_input}")
                
            # sets up x, y dimensions of plot
            ncols, nrows = self.set_facet_col_row_dimensions(multi_scenario=self.Scenarios)
            
            grid_size = ncols*nrows

            # Used to calculate any excess axis to delete
            plot_number = len(self.Scenarios)
            excess_axs = grid_size - plot_number
            
            mplt = PlotLibrary(nrows, ncols, sharey=True, 
                                squeeze=False, ravel_axs=True)
            fig, axs = mplt.get_figure()
            plt.subplots_adjust(wspace=0.05, hspace=0.2)
            
            data_table_chunks = []

            for i, scenario in enumerate(self.Scenarios):
                logger.info(f"Scenario = {scenario}")

                generation : pd.DataFrame = self["generator_Generation"].get(scenario)
                if shift_leapday:
                    generation = self.adjust_for_leapday(generation)

                avail_cap :pd.DataFrame = self["generator_Available_Capacity"].get(scenario) 
                if shift_leapday:
                    avail_cap = self.adjust_for_leapday(avail_cap)               
               
                # Check if zone is in avail_cap
                try:
                    avail_cap = avail_cap.xs(zone_input, level=self.AGG_BY)
                except KeyError:
                    logger.warning(f"No installed capacity in: {zone_input}")
                    break
                generation = generation.xs(zone_input, level=self.AGG_BY)
                avail_cap = self.df_process_gen_inputs(avail_cap)
                generation = self.df_process_gen_inputs(generation)
                generation = generation.loc[:, (generation != 0).any(axis=0)]

                thermal_reserve = avail_cap - generation         
                non_thermal_gen = set(thermal_reserve.columns) - set(self.thermal_gen_cat)
                # filter for only thermal generation 
                thermal_reserve = thermal_reserve.drop(labels=non_thermal_gen, axis=1)
                
                #Convert units
                if i == 0:
                    unitconversion = self.capacity_energy_unitconversion(thermal_reserve,
                                                                            sum_values=True)
                thermal_reserve = thermal_reserve / unitconversion['divisor']

                # Check if thermal_reserve contains data, if not skips
                if thermal_reserve.empty == True:
                    out = MissingZoneData()
                    outputs[zone_input] = out
                    continue
                   
                if pd.notna(start_date_range):
                    thermal_reserve = self.set_timestamp_date_range(
                                        thermal_reserve,
                                        start_date_range, end_date_range)
                    if thermal_reserve.empty is True:
                        logger.warning('No Generation in selected Date Range')
                        continue
                
                # Create data table for each scenario
                scenario_names = pd.Series([scenario]*len(thermal_reserve), name='Scenario')
                data_table = thermal_reserve.add_suffix(f" ({unitconversion['units']})")
                data_table = data_table.set_index([scenario_names], append=True)
                data_table_chunks.append(data_table)
                
                mplt.stackplot(thermal_reserve, color_dict=self.PLEXOS_color_dict,
                                labels=thermal_reserve.columns, sub_pos=i)

                axs[i].margins(x=0.01)
                mplt.set_subplot_timeseries_format(sub_pos=i)
            
            # add facet labels
            mplt.add_facet_labels(xlabels=self.xlabels,
                                  ylabels=self.ylabels)   
            # Add legend
            mplt.add_legend(reverse_legend=True, sort_by=self.ordered_gen)
            # Remove extra axes
            mplt.remove_excess_axs(excess_axs,grid_size)            
            plt.ylabel(f"Thermal capacity reserve ({unitconversion['units']})", 
                        color='black', rotation='vertical', labelpad=40)
            if plot_data_settings["plot_title_as_region"]:
                mplt.add_main_title(zone_input)
            # If data_table_chunks is empty, does not return data or figure
            if not data_table_chunks:
                outputs[zone_input] = MissingZoneData()
                continue
            
            # Concat all data tables together
            Data_Table_Out = pd.concat(data_table_chunks, copy=False, axis=0)
            
            outputs[zone_input] = {'fig': fig, 'data_table': Data_Table_Out}
        return outputs
