# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 07:42:06 2020

This code creates unserved energy timeseries line plots and total bat plots and is called from Marmot_plot_main.py

@author: dlevie
"""


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
import os
import logging

#===============================================================================

class mplot(object):
    def __init__(self, argument_dict):
        # iterate over items in argument_dict and set as properties of class
        # see key_list in Marmot_plot_main for list of properties
        for prop in argument_dict:
            self.__setattr__(prop, argument_dict[prop])
        
        self.logger = logging.getLogger('marmot_plot.'+__name__)
        
    def unserved_energy_timeseries(self):

        Unserved_Energy_Collection = {}

        for scenario in self.Multi_Scenario:
            # If data is to be agregated by zone, then zone properties are loaded, else region properties are loaded
            if self.AGG_BY == "zone":
                Unserved_Energy_Collection[scenario] = pd.read_hdf(os.path.join(self.Marmot_Solutions_folder, scenario, "Processed_HDF5_folder", scenario + "_formatted.h5"), "zone_Unserved_Energy")
            else:
                Unserved_Energy_Collection[scenario] = pd.read_hdf(os.path.join(self.Marmot_Solutions_folder,scenario, "Processed_HDF5_folder", scenario + "_formatted.h5"), "region_Unserved_Energy")

        outputs = {}
        for zone_input in self.Zones:
            self.logger.info('Zone = ' + zone_input)
            Unserved_Energy_Timeseries_Out = pd.DataFrame()
            #Total_Unserved_Energy_Out = pd.DataFrame()

            for scenario in self.Multi_Scenario:

                self.logger.info('Scenario = ' + scenario)

                unserved_eng_timeseries = Unserved_Energy_Collection.get(scenario)
                unserved_eng_timeseries = unserved_eng_timeseries.xs(zone_input,level=self.AGG_BY)
                unserved_eng_timeseries = unserved_eng_timeseries.groupby(["timestamp"]).sum()
                unserved_eng_timeseries = unserved_eng_timeseries.squeeze() #Convert to Series
                unserved_eng_timeseries.rename(scenario, inplace=True)
                Unserved_Energy_Timeseries_Out = pd.concat([Unserved_Energy_Timeseries_Out, unserved_eng_timeseries], axis=1, sort=False).fillna(0)

            Unserved_Energy_Timeseries_Out.columns = Unserved_Energy_Timeseries_Out.columns.str.replace('_',' ')
            Unserved_Energy_Timeseries_Out = Unserved_Energy_Timeseries_Out.loc[:, (Unserved_Energy_Timeseries_Out >= 1).any(axis=0)]
            #Total_Unserved_Energy_Out = Unserved_Energy_Timeseries_Out.sum(axis=0)

             # Data table of values to return to main program
            Data_Table_Out = Unserved_Energy_Timeseries_Out

            if Unserved_Energy_Timeseries_Out.empty==True:
                self.logger.warning('No Unserved Energy in {}'.format(zone_input))
                df = pd.DataFrame()
                outputs[zone_input] = df
                continue

            else:
                fig1, ax = plt.subplots(figsize=(9,6))

                # Converts color_list into an iterable list for use in a loop
                iter_colour = iter(self.color_list)

                for column in Unserved_Energy_Timeseries_Out:
                    ax.plot(Unserved_Energy_Timeseries_Out[column], linewidth=3, antialiased=True,
                             color=next(iter_colour), label=column)
                    ax.legend(loc='lower left',bbox_to_anchor=(1,0),
                              facecolor='inherit', frameon=True)
                ax.set_ylabel('Unserved Energy (MW)',  color='black', rotation='vertical')
                ax.set_ylim(bottom=0)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.tick_params(axis='y', which='major', length=5, width=1)
                ax.tick_params(axis='x', which='major', length=5, width=1)
                ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
                ax.margins(x=0.01)

            #    ax.axvline(dt.datetime(2024, 1, 2, 2, 0), color='black', linestyle='--')
            #    ax.axvline(outage_date_from, color='black', linestyle='--')
            #    ax.text(dt.datetime(2024, 1, 1, 5, 15), 0.8*max(ax.get_ylim()), "Outage \nBegins", fontsize=13)
            #    ax.axvline(dt.datetime(2024, 1, 6, 23, 0), color='black', linestyle='--')
            #    ax.text(dt.datetime(2024, 1, 7, 1, 30), 0.8*max(ax.get_ylim()), "Outage \nEnds", fontsize=13)

                locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
                formatter = mdates.ConciseDateFormatter(locator)
                formatter.formats[2] = '%d\n %b'
                formatter.zero_formats[1] = '%b\n %Y'
                formatter.zero_formats[2] = '%d\n %b'
                formatter.zero_formats[3] = '%H:%M\n %d-%b'
                formatter.offset_formats[3] = '%b %Y'
                formatter.show_offset = False
                ax.xaxis.set_major_locator(locator)
                ax.xaxis.set_major_formatter(formatter)

                outputs[zone_input] = {'fig': fig1, 'data_table': Data_Table_Out}
        return outputs

    def tot_unserved_energy(self):
        Unserved_Energy_Collection = {}

        for scenario in self.Multi_Scenario:
            # If data is to be agregated by zone, then zone properties are loaded, else region properties are loaded
            if self.AGG_BY == "zone":
                Unserved_Energy_Collection[scenario] = pd.read_hdf(os.path.join(self.Marmot_Solutions_folder, scenario, "processed_HDF5_folder", scenario + "_formatted.h5"), "zone_Unserved_Energy")
            else:
                Unserved_Energy_Collection[scenario] = pd.read_hdf(os.path.join(self.Marmot_Solutions_folder, scenario, "Processed_HDF5_folder", scenario + "_formatted.h5"), "region_Unserved_Energy")

        # Unserved_Energy_Timeseries_Out = pd.DataFrame()
        # Total_Unserved_Energy_Out = pd.DataFrame()

        outputs = {}
        for zone_input in self.Zones:
            Unserved_Energy_Timeseries_Out = pd.DataFrame()
            Total_Unserved_Energy_Out = pd.DataFrame()

            self.logger.info(self.AGG_BY + ' = ' + zone_input)
            for scenario in self.Multi_Scenario:

                self.logger.info('Scenario = ' + scenario)

                unserved_eng_timeseries = Unserved_Energy_Collection.get(scenario)
                unserved_eng_timeseries = unserved_eng_timeseries.xs(zone_input,level=self.AGG_BY)
                unserved_eng_timeseries = unserved_eng_timeseries.groupby(["timestamp"]).sum()
                unserved_eng_timeseries = unserved_eng_timeseries.squeeze() #Convert to Series
                unserved_eng_timeseries.rename(scenario, inplace=True)
                Unserved_Energy_Timeseries_Out = pd.concat([Unserved_Energy_Timeseries_Out, unserved_eng_timeseries], axis=1, sort=False).fillna(0)

            Unserved_Energy_Timeseries_Out.columns = Unserved_Energy_Timeseries_Out.columns.str.replace('_',' ')

            Total_Unserved_Energy_Out = Unserved_Energy_Timeseries_Out.sum(axis=0)

            Total_Unserved_Energy_Out.index = Total_Unserved_Energy_Out.index.str.replace('_',' ')
            Total_Unserved_Energy_Out.index = Total_Unserved_Energy_Out.index.str.wrap(10, break_long_words=False)

            if Total_Unserved_Energy_Out.values.sum() == 0:
                self.logger.warning('No Unserved Energy in {}'.format(zone_input))
                df = pd.DataFrame()
                outputs[zone_input] = df
                continue

            else:

                # Data table of values to return to main program
                Data_Table_Out = Total_Unserved_Energy_Out

                # Converts color_list into an iterable list for use in a loop
                iter_colour = iter(self.color_list)

                fig2, ax = plt.subplots(figsize=(6,4))

                Total_Unserved_Energy_Out.plot.bar(stacked=False, rot=0, edgecolor='black',
                                                        color=next(iter_colour), linewidth='0.1',
                                                        width=0.35, ax=ax)

                ax.set_ylabel('Total Unserved Energy (MWh)',  color='black', rotation='vertical')
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.tick_params(axis='y', which='major', length=5, width=1)
                ax.tick_params(axis='x', which='major', length=5, width=1)
                ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
                ax.margins(x=0.01)

                for i in ax.patches:
                    width, height = i.get_width(), i.get_height()
                    if height<=1:
                        continue
                    x, y = i.get_xy()
                    ax.text(x+width/2,
                        y+height/2,
                        '{:,.0f}'.format(height),
                        horizontalalignment='center',
                        verticalalignment='center', fontsize=13)

                outputs[zone_input] = {'fig': fig2, 'data_table': Data_Table_Out}
        return outputs
