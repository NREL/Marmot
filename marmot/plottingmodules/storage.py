# -*- coding: utf-8 -*-
"""
Created April 2020, updated August 2020

This code creates energy storage plots and is called from Marmot_plot_main.py
@author: 
"""

import logging
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

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
        super().__init__(self.Marmot_Solutions_folder, self.AGG_BY, self.ordered_gen, 
                    self.PLEXOS_color_dict, self.Scenarios, self.ylabels, 
                    self.xlabels, self.gen_names_dict, Region_Mapping=self.Region_Mapping) 

        self.logger = logging.getLogger('marmot_plot.'+__name__)
        self.y_axes_decimalpt = mconfig.parser("axes_options","y_axes_decimalpt")
        

    def storage_volume(self, figure_name=None, prop=None, start=None, 
                             end=None, timezone="", start_date_range=None, 
                             end_date_range=None):

        """
        This method creates time series plot of aggregate storage volume for all storage objects in a given region, 
        along with a horizontal line representing full charge.
        All scenarios are plotted on a single figure.
        Figures and data tables are returned to plot_main
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

            fig1, axs = plotlib.setup_plot(ydimension = 2,sharey = False)
            plt.subplots_adjust(wspace=0.05, hspace=0.2)
            
            if storage_volume_all_scenarios.empty:
                out = MissingZoneData()
                outputs[zone_input] = out
                continue
            
            for column in storage_volume_all_scenarios:
                plotlib.create_line_plot(axs,storage_volume_all_scenarios,column,color_dict,label = column,n = 0)      
                axs[0].set_ylabel('Head Storage Volume (GWh)',  color='black', rotation='vertical')
                axs[0].yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(x, f',.{self.y_axes_decimalpt}f')))
                axs[0].margins(x=0.01)

                axs[0].set_ylim(ymin = 0)
                axs[0].set_title(zone_input)
                #axs[0].xaxis.set_visible(False)

                plotlib.create_line_plot(axs,use_all_scenarios,column,color_dict,label = column + ' Unserved Energy', n = 1)
                axs[1].set_ylabel('Unserved Energy (GWh)',  color='black', rotation='vertical')
                axs[1].set_xlabel(timezone,  color='black', rotation='horizontal')
                axs[1].yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(x, f',.{self.y_axes_decimalpt}f')))
                axs[1].margins(x=0.01)

                [PlotDataHelper.set_plot_timeseries_format(axs,n) for n in range(0,2)]
            
            axs[0].axhline(y = max_volume, linestyle = ':',label = 'Max Volume')
            axs[0].legend(loc = 'lower left',bbox_to_anchor = (1.15,0),facecolor = 'inherit',frameon = True)
            axs[1].legend(loc = 'lower left',bbox_to_anchor = (1.15,0.2),facecolor = 'inherit',frameon = True)
            if mconfig.parser("plot_title_as_region"):
                fig1.title(zone_input)

            outputs[zone_input] = {'fig': fig1, 'data_table': Data_Table_Out}
        return outputs


