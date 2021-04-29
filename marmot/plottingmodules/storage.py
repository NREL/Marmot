# -*- coding: utf-8 -*-
"""
Created April 2020, updated August 2020

This code creates transmission line and interface plots and is called from Marmot_plot_main.py

@author: dlevie
"""

import os
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
import numpy as np
import math
import logging
import marmot.marmot_plot_functions as mfunc
import marmot.config.mconfig as mconfig


#===============================================================================

class mplot(object):

    def __init__(self, argument_dict):
        # iterate over items in argument_dict and set as properties of class
        # see key_list in Marmot_plot_main for list of properties
        for prop in argument_dict:
            self.__setattr__(prop, argument_dict[prop])
        self.logger = logging.getLogger('marmot_plot.'+__name__)
        self.y_axes_decimalpt = mconfig.parser("axes_options","y_axes_decimalpt")

    def storage_volume(self):

        """
        This method creates time series plot of aggregate storage volume for all storage objects in a given region, 
        along with a horizontal line representing full charge.
        All scenarios are plotted on a single figure.
        Figures and data tables are returned to plot_main
        """
        initial_volume_collection = {}
        unserved_e_collection = {}
        max_volume_collection = {}
        check_input_data = []
        check_input_data.extend([mfunc.get_data(initial_volume_collection,"storage_Initial_Volume",self.Marmot_Solutions_folder, self.Multi_Scenario)])
        check_input_data.extend([mfunc.get_data(unserved_e_collection,"region_Unserved_Energy",self.Marmot_Solutions_folder, self.Multi_Scenario)])
        check_input_data.extend([mfunc.get_data(max_volume_collection,"storage_Max_Volume",self.Marmot_Solutions_folder, [self.Multi_Scenario[0]])])


        if 1 in check_input_data:
            outputs = mfunc.MissingInputData()
            return outputs

        outputs = {}
        for zone_input in self.Zones:
            self.logger.info(self.AGG_BY + " = " + zone_input)

            storage_volume_all_scenarios = pd.DataFrame()
            use_all_scenarios = pd.DataFrame()

            for scenario in self.Multi_Scenario:

                self.logger.info("Scenario = " + str(scenario))

                storage_volume_read = initial_volume_collection.get(scenario)
                try:
                    storage_volume = storage_volume_read.xs(zone_input, level = self.AGG_BY)
                except KeyError:
                    self.logger.warning('No storage resources in %s',zone_input)
                    outputs[zone_input] = mfunc.MissingZoneData()
                    continue

                #Isolate only head storage objects (not tail).
                storage_gen_units = storage_volume.index.get_level_values('storage_resource')
                head_units = [unit for unit in storage_gen_units if 'head' in unit]
                storage_volume = storage_volume.iloc[storage_volume.index.get_level_values('storage_resource').isin(head_units)]
                storage_volume = storage_volume.groupby("timestamp").sum()
                storage_volume.columns = [scenario]

                max_volume = storage_volume.max().squeeze()
                try:
                    max_volume = max_volume_collection.get(scenario)
                    max_volume = max_volume.xs(zone_input, level = self.AGG_BY)
                    max_volume = max_volume.groupby('timestamp').sum()
                    max_volume = max_volume.squeeze()[0]
                except KeyError:
                    self.logger.warning('No storage resources in %s',zone_input)

                #Pull unserved energy.
                use_read = unserved_e_collection.get(scenario)
                use = use_read.xs(zone_input, level = self.AGG_BY)
                use = use.groupby("timestamp").sum() / 1000
                use.columns = [scenario]

                if self.prop == "Peak Demand":

                    peak_demand_t = Total_Demand.idxmax()
                    end_date = peak_demand_t + dt.timedelta(days=self.end)
                    start_date = peak_demand_t - dt.timedelta(days=self.start)
                    Peak_Demand = Total_Demand[peak_demand_t]
                    storage_volume = storage_volume[start_date : end_date]
                    use = use[start_date : end_date]

                elif self.prop == "Min Net Load":
                    min_net_load_t = Net_Load.idxmin()
                    end_date = min_net_load_t + dt.timedelta(days=self.end)
                    start_date = min_net_load_t - dt.timedelta(days=self.start)
                    Min_Net_Load = Net_Load[min_net_load_t]
                    storage_volume = storage_volume[start_date : end_date]
                    use = use[start_date : end_date]

                elif self.prop == 'Date Range':
                    self.logger.info("Plotting specific date range: \
                    {} to {}".format(str(self.start_date),str(self.end_date)))

                    storage_volume = storage_volume[self.start_date : self.end_date]
                    use = use[self.start_date : self.end_date]

                storage_volume_all_scenarios = pd.concat([storage_volume_all_scenarios,storage_volume], axis = 1)
                #storage_volume_all_scenarios.columns = storage_volume_all_scenarios.columns.str.replace('_',' ')

                use_all_scenarios = pd.concat([use_all_scenarios,use], axis = 1)
                #use_all_scenarios.columns = use_all_scenarios.columns.str.replace('_',' ')

            # Data table of values to return to main program
            Data_Table_Out = pd.concat([storage_volume_all_scenarios,use_all_scenarios],axis = 1)
            #Make scenario/color dictionary.
            color_dict = dict(zip(storage_volume_all_scenarios.columns,self.color_list))

            # if '2008' not in self.Marmot_Solutions_folder and '2012' not in self.Marmot_Solutions_folder and storage_volume_all_scenarios.index[1] > dt.datetime(2024,2,28,0,0):
            #     storage_volume_all_scenarios.index = storage_volume_all_scenarios.index.shift(1,freq = 'D') #TO DEAL WITH LEAP DAYS, SPECIFIC TO MARTY'S PROJECT, REMOVE AFTER.

            fig1, axs = mfunc.setup_plot(ydimension = 2,sharey = False)
            plt.subplots_adjust(wspace=0.05, hspace=0.2)
            
            if storage_volume_all_scenarios.empty:
                out = mfunc.MissingZoneData()
                outputs[zone_input] = out
                continue
            
            for column in storage_volume_all_scenarios:
                mfunc.create_line_plot(axs,storage_volume_all_scenarios,column,color_dict,label = column,n = 0)      
                axs[0].set_ylabel('Head Storage Volume (GWh)',  color='black', rotation='vertical')
                axs[0].yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(x, f',.{self.y_axes_decimalpt}f')))
                axs[0].margins(x=0.01)

                axs[0].set_ylim(ymin = 0)
                axs[0].set_title(zone_input)
                #axs[0].xaxis.set_visible(False)

                mfunc.create_line_plot(axs,use_all_scenarios,column,color_dict,label = column + ' Unserved Energy', n = 1)
                axs[1].set_ylabel('Unserved Energy (GWh)',  color='black', rotation='vertical')
                axs[1].set_xlabel('Date ' + '(' + self.timezone + ')',  color='black', rotation='horizontal')
                axs[1].yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(x, f',.{self.y_axes_decimalpt}f')))
                axs[1].margins(x=0.01)

                [mfunc.set_plot_timeseries_format(axs,n) for n in range(0,2)]
            
            axs[0].axhline(y = max_volume, linestyle = ':',label = 'Max Volume')
            axs[0].legend(loc = 'lower left',bbox_to_anchor = (1.15,0),facecolor = 'inherit',frameon = True)
            axs[1].legend(loc = 'lower left',bbox_to_anchor = (1.15,0.2),facecolor = 'inherit',frameon = True)
            if mconfig.parser("plot_title_as_region"):
                fig1.title(zone_input)

            outputs[zone_input] = {'fig': fig1, 'data_table': Data_Table_Out}
        return outputs


