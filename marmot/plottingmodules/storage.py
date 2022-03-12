# -*- coding: utf-8 -*-
"""Energy storage plots.

This module creates energy storage plots.
"""

import logging
import pandas as pd
import matplotlib.pyplot as plt

import marmot.utils.mconfig as mconfig

from marmot.plottingmodules.plotutils.plot_library import SetupSubplot
from marmot.plottingmodules.plotutils.plot_data_helper import PlotDataHelper
from marmot.plottingmodules.plotutils.plot_exceptions import (MissingInputData, MissingZoneData)


class MPlot(PlotDataHelper):
    """storage MPlot class.

    All the plotting modules use this same class name.
    This class contains plotting methods that are grouped based on the
    current module name.
    
    The storage.py module contains methods that are
    related to storage devices. 
   
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

    def storage_volume(self, timezone: str = "", 
                       start_date_range: str = None, 
                       end_date_range: str = None, **_):
        """Creates time series plot of aggregate storage volume for all storage objects in a given region.

        A horizontal line represents full charge of the storage device.
        All scenarios are plotted on a single figure.

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
        if self.AGG_BY == 'zone':
                agg = 'zone'
        else:
            agg = 'region'
            
        outputs = {}
        
        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True, "storage_Initial_Volume", self.Scenarios),
                      (True, f"{agg}_Unserved_Energy", self.Scenarios),
                      (True, "storage_Max_Volume", [self.Scenarios[0]])]
        
        # Runs get_formatted_data within PlotDataHelper to populate PlotDataHelper dictionary  
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)

        if 1 in check_input_data:
            return MissingInputData()
        
        for zone_input in self.Zones:
            self.logger.info(f"{self.AGG_BY} = {zone_input}")

            storage_volume_all_scenarios = pd.DataFrame()
            use_all_scenarios = pd.DataFrame()

            for scenario in self.Multi_Scenario:

                self.logger.info(f"Scenario = {str(scenario)}")

                storage_volume_read = self["storage_Initial_Volume"].get(scenario)
                try:
                    storage_volume = storage_volume_read.xs(zone_input, level = self.AGG_BY)
                except KeyError:
                    self.logger.warning(f'No storage resources in {zone_input}')
                    outputs[zone_input] = MissingZoneData()
                    continue

                #Isolate only head storage objects (not tail).
                storage_gen_units = storage_volume.index.get_level_values('storage_resource')
                head_units = [unit for unit in storage_gen_units if 'head' in unit]
                storage_volume = storage_volume.iloc[storage_volume.index.get_level_values('storage_resource').isin(head_units)]
                storage_volume = storage_volume.groupby("timestamp").sum()
                storage_volume.columns = [scenario]

                max_volume = storage_volume.max().squeeze()
                try:
                    max_volume = self["storage_Max_Volume"].get(scenario)
                    max_volume = max_volume.xs(zone_input, level = self.AGG_BY)
                    max_volume = max_volume.groupby('timestamp').sum()
                    max_volume = max_volume.squeeze()[0]
                except KeyError:
                    self.logger.warning(f'No storage resources in {zone_input}')

                #Pull unserved energy.
                use_read = self[f"{agg}_Unserved_Energy"].get(scenario)
                use = use_read.xs(zone_input, level = self.AGG_BY)
                use = use.groupby("timestamp").sum() / 1000
                use.columns = [scenario]

                # if prop == "Peak Demand":

                #     peak_demand_t = Total_Demand.idxmax()
                #     end_date = peak_demand_t + dt.timedelta(days=end)
                #     start_date = peak_demand_t - dt.timedelta(days=start)
                #     Peak_Demand = Total_Demand[peak_demand_t]
                #     storage_volume = storage_volume[start_date : end_date]
                #     use = use[start_date : end_date]

                # elif prop == "Min Net Load":
                #     min_net_load_t = Net_Load.idxmin()
                #     end_date = min_net_load_t + dt.timedelta(days=end)
                #     start_date = min_net_load_t - dt.timedelta(days=start)
                #     Min_Net_Load = Net_Load[min_net_load_t]
                #     storage_volume = storage_volume[start_date : end_date]
                #     use = use[start_date : end_date]

                if pd.notna(start_date_range):
                    self.logger.info(f"Plotting specific date range: \
                    {str(start_date_range)} to {str(end_date_range)}")

                    storage_volume = storage_volume[start_date_range : end_date_range]
                    use = use[start_date_range : end_date_range]

                storage_volume_all_scenarios = pd.concat([storage_volume_all_scenarios,storage_volume], axis = 1)
                #storage_volume_all_scenarios.columns = storage_volume_all_scenarios.columns.str.replace('_',' ')

                use_all_scenarios = pd.concat([use_all_scenarios,use], axis = 1)
                #use_all_scenarios.columns = use_all_scenarios.columns.str.replace('_',' ')

            # Data table of values to return to main program
            Data_Table_Out = pd.concat([storage_volume_all_scenarios,use_all_scenarios],axis = 1)
            #Make scenario/color dictionary.
            color_dict = dict(zip(storage_volume_all_scenarios.columns,self.color_list))

            mplt = SetupSubplot(nrows=2, squeeze=False, 
                                ravel_axs=True)
            fig, axs = mplt.get_figure()
            plt.subplots_adjust(wspace=0.05, hspace=0.2)
            
            if storage_volume_all_scenarios.empty:
                out = MissingZoneData()
                outputs[zone_input] = out
                continue
            
            for column in storage_volume_all_scenarios:
                axs[0].plot(storage_volume_all_scenarios.index.values, 
                            storage_volume_all_scenarios[column], 
                            linewidth=1,
                            color=color_dict[column],
                            label=column)

                axs[0].set_ylabel('Head Storage Volume (GWh)', 
                                  color='black', rotation='vertical')
                mplt.set_yaxis_major_tick_format(sub_pos=0)
                axs[0].margins(x=0.01)
                mplt.set_subplot_timeseries_format(sub_pos=0)
                axs[0].set_ylim(ymin = 0)
                axs[0].set_title(zone_input)

                axs[1].plot(use_all_scenarios.index.values, 
                            use_all_scenarios[column], 
                            linewidth=1,
                            color=color_dict[column],
                            label= f"{column} Unserved Energy")

                axs[1].set_ylabel('Unserved Energy (GWh)',  
                                  color='black', rotation='vertical')
                axs[1].set_xlabel(timezone,  color='black', 
                                    rotation='horizontal')
                mplt.set_yaxis_major_tick_format(sub_pos=1)
                axs[1].margins(x=0.01)
                mplt.set_subplot_timeseries_format(sub_pos=1)
            
            mplt.set_yaxis_major_tick_format()
            axs[0].axhline(y=max_volume, linestyle=':', label='Max Volume')
            axs[0].legend(loc='lower left', bbox_to_anchor = (1.15,0))
            axs[1].legend(loc='lower left', bbox_to_anchor = (1.15,0.2))
            if mconfig.parser("plot_title_as_region"):
                mplt.add_main_title(zone_input)

            outputs[zone_input] = {'fig': fig, 'data_table': Data_Table_Out}
        return outputs


