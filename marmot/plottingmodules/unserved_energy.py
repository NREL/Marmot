# -*- coding: utf-8 -*-
"""System unserved energy plots.

This module creates unserved energy timeseries line plots and total bar
plots and is called from marmot_plot_main.py

@author: Daniel Levie 
"""

import logging
import pandas as pd

import marmot.utils.mconfig as mconfig

from marmot.plottingmodules.plotutils.plot_library import PlotLibrary
from marmot.plottingmodules.plotutils.plot_data_helper import MPlotDataHelper
from marmot.plottingmodules.plotutils.plot_exceptions import (MissingInputData, MissingZoneData)

logger = logging.getLogger('plotter.'+__name__)        
plot_data_settings = mconfig.parser("plot_data")

class UnservedEnergy(MPlotDataHelper):
    """System unserved energy plots.

    The unserved_energy.py module contains methods that are
    related to unserved energy in the power system. 

    UnservedEnergy inherits from the MPlotDataHelper class to assist 
    in creating figures.
    """

    def __init__(self, **kwargs):
        # Instantiation of MPlotHelperFunctions
        super().__init__(**kwargs)

    def unserved_energy_timeseries(self, start_date_range: str = None, 
                                   end_date_range: str = None,
                                   data_resolution: str = "", **_):
        """Creates a timeseries line plot of total unserved energy.

        Each sceanrio is plotted as a separate line.

        Args:
            start_date_range (str, optional): Defines a start date at which to represent data from. 
                Defaults to None.
            end_date_range (str, optional): Defines a end date at which to represent data to.
                Defaults to None.
            data_resolution (str, optional): Specifies the data resolution to pull from the formatted 
                data and plot.
                Defaults to "", which will pull interval data.

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
        properties = [(True, f"{agg}_Unserved_Energy{data_resolution}", self.Scenarios)]
        
        # Runs get_formatted_data within MPlotDataHelper to populate MPlotDataHelper dictionary  
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)

        if 1 in check_input_data:
            return MissingInputData()
        
        for zone_input in self.Zones:
            logger.info(f'Zone = {zone_input}')
            unserved_energy_chunks = []

            for scenario in self.Scenarios:
                logger.info(f'Scenario = {scenario}')

                unserved_energy : pd.DataFrame = self[f"{agg}_Unserved_Energy{data_resolution}"].get(scenario)
                unserved_energy = unserved_energy.xs(zone_input,level=self.AGG_BY)
                unserved_energy = unserved_energy.groupby(["timestamp"]).sum()

                if pd.notna(start_date_range):
                    unserved_energy = self.set_timestamp_date_range(
                                        unserved_energy,
                                        start_date_range, end_date_range)
                    if unserved_energy.empty is True:
                        logger.warning('No Unserved Energy in selected Date Range')
                        continue
                
                unserved_energy = unserved_energy.rename(columns={0: scenario})
                unserved_energy_chunks.append(unserved_energy)

            unserved_energy_out :pd.DataFrame = pd.concat(unserved_energy_chunks, 
                                                            axis=1, sort=False).fillna(0)
            unserved_energy_out = unserved_energy_out.loc[:, (unserved_energy_out != 0).any(axis=0)]

            if unserved_energy_out.to_numpy().sum() == 0:
                logger.info(f'No Unserved Energy in {zone_input}')
                out = MissingZoneData()
                outputs[zone_input] = out
                continue
            
            # Determine auto unit coversion
            unitconversion = self.capacity_energy_unitconversion(unserved_energy_out)
            unserved_energy_out = unserved_energy_out/unitconversion['divisor'] 
            
            # Data table of values to return to main program
            Data_Table_Out = unserved_energy_out.add_suffix(f" ({unitconversion['units']})")
            
            mplt = PlotLibrary()
            fig, ax = mplt.get_figure()
            # Converts color_list into an iterable list for use in a loop
            iter_colour = iter(self.color_list)

            for column in unserved_energy_out:
                mplt.lineplot(unserved_energy_out[column],
                              color=next(iter_colour), linewidth=3,
                              label=column)
            mplt.add_legend()
            ax.set_ylabel(f"Unserved Energy ({unitconversion['units']})", 
                          color='black', rotation='vertical')
            ax.set_ylim(bottom=0)
            ax.margins(x=0.01)
            mplt.set_subplot_timeseries_format()
            if plot_data_settings["plot_title_as_region"]:
                mplt.add_main_title(zone_input)
            outputs[zone_input] = {'fig': fig, 'data_table': Data_Table_Out}

        return outputs

    def tot_unserved_energy(self, start_date_range: str = None, 
                            end_date_range: str = None,
                            scenario_groupby: str = 'Scenario', **_):
        """Creates a bar plot of total unserved energy.

        Each sceanrio is plotted as a separate bar.

        Args:
            start_date_range (str, optional): Defines a start date at which to represent data from. 
                Defaults to None.
            end_date_range (str, optional): Defines a end date at which to represent data to.
                Defaults to None.
            scenario_groupby (str, optional): Specifies whether to group data by Scenario 
                or Year-Sceanrio. If grouping by Year-Sceanrio the year will be identified from 
                the timestamp and appeneded to the sceanrio name. This is useful when plotting data 
                which covers multiple years such as ReEDS.
                Defaults to Scenario.

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
        properties = [(True, f"{agg}_Unserved_Energy", self.Scenarios)]
        
        # Runs get_formatted_data within MPlotDataHelper to populate MPlotDataHelper dictionary  
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)

        if 1 in check_input_data:
            return MissingInputData()

        for zone_input in self.Zones:
            unserved_energy_chunks = []
            logger.info(f"{self.AGG_BY} = {zone_input}")
            
            for scenario in self.Scenarios:
                logger.info(f'Scenario = {scenario}')

                unserved_energy :pd.DataFrame = self[f"{agg}_Unserved_Energy"].get(scenario)
                unserved_energy = unserved_energy.xs(zone_input,level=self.AGG_BY)
                unserved_energy = unserved_energy.groupby(["timestamp"]).sum()
                
                # correct sum for non-hourly runs
                interval_count = self.get_sub_hour_interval_count(unserved_energy)
                unserved_energy = unserved_energy/interval_count
                if pd.notna(start_date_range):
                    unserved_energy = self.set_timestamp_date_range(
                                        unserved_energy,
                                        start_date_range, end_date_range)
                    if unserved_energy.empty is True:
                        logger.warning('No Unserved Energy in selected Date Range')
                        continue
                                
                unserved_energy_chunks.append(self.year_scenario_grouper(unserved_energy, 
                                                scenario, groupby=scenario_groupby).sum())

            unserved_energy_out = pd.concat(unserved_energy_chunks, axis=0, sort=False).fillna(0)

            # Set scenarios as columns
            unserved_energy_out = unserved_energy_out.T
            unserved_energy_out = unserved_energy_out.loc[:, (unserved_energy_out >= 1)
                                                                                    .any(axis=0)]

            if unserved_energy_out.empty==True:
                logger.info(f'No Unserved Energy in {zone_input}')
                out = MissingZoneData()
                outputs[zone_input] = out
                continue
            
            # Determine auto unit coversion
            unitconversion = self.capacity_energy_unitconversion(unserved_energy_out)
            unserved_energy_out = unserved_energy_out/unitconversion['divisor']
            
            # Data table of values to return to main program
            Data_Table_Out = unserved_energy_out.add_suffix(f" ({unitconversion['units']})")
            
            # create color dictionary
            color_dict = dict(zip(unserved_energy_out.columns, self.color_list))
            
            # Set scenarios as column names
            mplt = PlotLibrary()
            fig, ax = mplt.get_figure()
            
            if self.custom_xticklabels:
                tick_labels = self.custom_xticklabels
            else:
                tick_labels = unserved_energy_out.columns

            mplt.barplot(unserved_energy_out, color=color_dict,
                        custom_tick_labels=tick_labels)

            ax.set_ylabel(f"Total Unserved Energy ({unitconversion['units']}h)", 
                          color='black', rotation='vertical')
            ax.xaxis.set_visible(False)
            ax.margins(x=0.01)
            
            mplt.add_legend()
            if plot_data_settings["plot_title_as_region"]:
                mplt.add_main_title(zone_input)  
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

            outputs[zone_input] = {'fig': fig, 'data_table': Data_Table_Out}
        return outputs

    def average_diurnal_ue(self, start_date_range: str = None, 
                           end_date_range: str = None,
                           scenario_groupby: str = 'Scenario', **_):
        """Creates a line plot of average diurnal unserved energy.

        Each scenario is plotted as a separate line and shows the average 
        hourly unserved energy over a 24 hour period averaged across the entire year
        or time period defined.

        Args:
            start_date_range (str, optional): Defines a start date at which to represent data from. 
                Defaults to None.
            end_date_range (str, optional): Defines a end date at which to represent data to.
                Defaults to None.
            scenario_groupby (str, optional): Specifies whether to group data by Scenario 
                or Year-Sceanrio. If grouping by Year-Sceanrio the year will be identified from 
                the timestamp and appeneded to the sceanrio name. This is useful when plotting data 
                which covers multiple years such as ReEDS.
                Defaults to Scenario.

        Returns:
            dict: Dictionary containing the created plot and its data table.
        """
        outputs : dict = {}
        
        if self.AGG_BY == 'zone':
            agg = 'zone'
        else:
            agg = 'region'
        
        # List of properties needed by the plot, properties are a set 
        # of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, 
        # scenarios must be a list.
        properties = [(True,f"{agg}_Unserved_Energy",self.Scenarios)]
        
        # Runs get_formatted_data within MPlotDataHelper to populate 
        # MPlotDataHelper dictionary with all required properties, 
        # returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)

        if 1 in check_input_data:
            return MissingInputData()

        for zone_input in self.Zones:
            logger.info(f"{self.AGG_BY} = {zone_input}")
            
            chunks =[]
            for scenario in self.Scenarios:
                logger.info(f"Scenario = {scenario}")
                
                unserved_energy = self[f"{agg}_Unserved_Energy"][scenario]
                try:
                    unserved_energy = unserved_energy.xs(zone_input,level=self.AGG_BY)
                except KeyError:
                    logger.info(f'No unserved energy in {zone_input}')
                
                unserved_energy = unserved_energy.groupby(["timestamp"]).sum()

                if pd.notna(start_date_range):
                    unserved_energy = self.set_timestamp_date_range(
                                        unserved_energy,
                                        start_date_range, end_date_range)
                    if unserved_energy.empty is True:
                        logger.warning('No Unserved Energy in selected Date Range')
                        continue
                
                interval_count = self.get_sub_hour_interval_count(unserved_energy)
                unserved_energy = unserved_energy/interval_count
                # Group data by hours and find mean across entire range 
                unserved_energy = self.year_scenario_grouper(unserved_energy, scenario, groupby=scenario_groupby,
                                                    additional_groups=[unserved_energy.index.hour]).mean()
                
                # reset index to datetime 
                for scen in unserved_energy.index.get_level_values('Scenario').unique():
                    unserved_energy_scen = unserved_energy.xs(scen, level='Scenario')
                    # If hours are missing, fill with 0
                    if len(unserved_energy_scen) < 24:
                        unserved_energy_idx = range(0,24)
                        unserved_energy_scen = \
                            unserved_energy_scen.reindex(unserved_energy_idx, fill_value=0)
                    # reset index to datetime 
                    unserved_energy_scen.index = pd.date_range("2024-01-01", periods=24, freq="H")
                    unserved_energy_scen.rename(columns={0: scen}, inplace=True)
                    chunks.append(unserved_energy_scen)
 
            unserved_energy_out = pd.concat(chunks, axis=1, sort=False)

            # Create Dictionary from scenario names and color list
            colour_dict = dict(zip(unserved_energy_out.columns, self.color_list))

            mplt = PlotLibrary()
            fig, ax = mplt.get_figure()

            unitconversion = self.capacity_energy_unitconversion(unserved_energy_out)
            unserved_energy_out = unserved_energy_out / unitconversion['divisor']
            Data_Table_Out = unserved_energy_out
            Data_Table_Out = Data_Table_Out.add_suffix(f" ({unitconversion['units']})")
            
            for column in unserved_energy_out:
                mplt.lineplot(unserved_energy_out[column], color=colour_dict,
                             linewidth=3, label=column)

            mplt.set_subplot_timeseries_format(zero_formats_3='%H:%M')
            mplt.add_legend()
            ax.set_ylabel(f"Average Diurnal Unserved Energy ({unitconversion['units']})", 
                          color='black', rotation='vertical')            
            ax.margins(x=0.01)            
            ax.set_ylim(bottom=0)

            if plot_data_settings["plot_title_as_region"]:
                mplt.add_main_title(zone_input)

            outputs[zone_input] = {'fig': fig, 'data_table': Data_Table_Out}
        return outputs
