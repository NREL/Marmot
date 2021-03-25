# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 10:34:48 2019

This code creates generation stack plots and is called from Marmot_plot_main.py

@author: dlevie
"""

import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
from matplotlib.patches import Patch
import logging
import plottingmodules.marmot_plot_functions as mfunc

#===============================================================================

custom_legend_elements = [Patch(facecolor='#DD0200',
                            alpha=0.5, edgecolor='#DD0200',
                         label='Unserved Energy')]

class mplot(object):

    def __init__(self, argument_dict):
        # iterate over items in argument_dict and set as properties of class
        # see key_list in Marmot_plot_main for list of properties
        for prop in argument_dict:
            self.__setattr__(prop, argument_dict[prop])
        self.logger = logging.getLogger('marmot_plot.'+__name__)
        
    def thermal_cap_reserves(self):
        outputs = {}
        generation_collection = {}
        gen_available_capacity_collection = {}
        check_input_data = []
        
        check_input_data.extend([mfunc.get_data(generation_collection,"generator_Generation", self.Marmot_Solutions_folder, self.Scenarios)])
        check_input_data.extend([mfunc.get_data(gen_available_capacity_collection,"generator_Available_Capacity", self.Marmot_Solutions_folder, self.Scenarios)])
        
        # Checks if all data required by plot is available, if 1 in list required data is missing
        if 1 in check_input_data:
            outputs = mfunc.MissingInputData()
            return outputs
        
        for zone_input in self.Zones:
            self.logger.info("Zone = "+ zone_input)

            xdimension=len(self.xlabels)
            if xdimension == 0:
                xdimension = 1
            ydimension=len(self.ylabels)
            if ydimension == 0:
                ydimension = 1

            Data_Table_Out = pd.DataFrame()

            fig1, axs = plt.subplots(ydimension,xdimension, figsize=((8*xdimension),(4*ydimension)), sharey=True, squeeze=False)
            plt.subplots_adjust(wspace=0.05, hspace=0.2)
            # if len(self.Scenarios) > 1:
            axs = axs.ravel()
            i=0

            for scenario in self.Scenarios:

                self.logger.info("Scenario = " + scenario)

                Gen = generation_collection.get(scenario).copy()
                if self.shift_leapday:
                    Gen = mfunc.shift_leapday(Gen,self.Marmot_Solutions_folder)
                avail_cap = gen_available_capacity_collection.get(scenario).copy()
                if self.shift_leapday:
                    avail_cap = mfunc.shift_leapday(avail_cap,self.Marmot_Solutions_folder)               
               
                # Check if zone is in avail_cap
                try:
                    avail_cap = avail_cap.xs(zone_input,level = self.AGG_BY)
                except KeyError:
                    self.logger.warning("No installed capacity in : "+zone_input)
                    break
                Gen = Gen.xs(zone_input,level = self.AGG_BY)
                avail_cap = mfunc.df_process_gen_inputs(avail_cap,self.ordered_gen)
                Gen = mfunc.df_process_gen_inputs(Gen,self.ordered_gen)
                Gen = Gen.loc[:, (Gen != 0).any(axis=0)]

                thermal_reserve = avail_cap - Gen

                #Convert units
                if i == 0:
                    unitconversion = mfunc.capacity_energy_unitconversion(max(thermal_reserve.sum(axis=1)))
                thermal_reserve = thermal_reserve / unitconversion['divisor']

                # Check if thermal_reserve contains data, if not skips
                if thermal_reserve.empty == True:
                    out = mfunc.MissingZoneData()
                    outputs[zone_input] = out
                    continue

                Data_Table_Out = thermal_reserve



                locator = mdates.AutoDateLocator(minticks = self.minticks, maxticks = self.maxticks)
                formatter = mdates.ConciseDateFormatter(locator)
                formatter.formats[2] = '%d\n %b'
                formatter.zero_formats[1] = '%b\n %Y'
                formatter.zero_formats[2] = '%d\n %b'
                formatter.zero_formats[3] = '%H:%M\n %d-%b'
                formatter.offset_formats[3] = '%b %Y'
                formatter.show_offset = False

                if len(self.Scenarios) > 1:
                    sp = axs[i].stackplot(thermal_reserve.index.values, thermal_reserve.values.T, labels = thermal_reserve.columns, linewidth=0,
                                 colors = [self.PLEXOS_color_dict.get(x, '#333333') for x in thermal_reserve.T.index])

                    axs[i].spines['right'].set_visible(False)
                    axs[i].spines['top'].set_visible(False)
                    axs[i].tick_params(axis='y', which='major', length=5, width=1)
                    axs[i].tick_params(axis='x', which='major', length=5, width=1)
                    axs[i].yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
                    axs[i].margins(x=0.01)
                    axs[i].xaxis.set_major_locator(locator)
                    axs[i].xaxis.set_major_formatter(formatter)
                    handles, labels = axs[i].get_legend_handles_labels()
                    #Legend 1
                    leg1 = axs[i].legend(reversed(handles), reversed(labels), loc='lower left',bbox_to_anchor=(1,0),facecolor='inherit', frameon=True)
                    # Manually add the first legend back
                    axs[i].add_artist(leg1)

                else:
                    sp = axs[i].stackplot(thermal_reserve.index.values, thermal_reserve.values.T, labels = thermal_reserve.columns, linewidth=0,
                                 colors = [self.PLEXOS_color_dict.get(x, '#333333') for x in thermal_reserve.T.index])

                    axs[i].spines['right'].set_visible(False)
                    axs[i].spines['top'].set_visible(False)
                    axs[i].tick_params(axis='y', which='major', length=5, width=1)
                    axs[i].tick_params(axis='x', which='major', length=5, width=1)
                    axs[i].yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
                    axs[i].margins(x=0.01)
                    axs[i].xaxis.set_major_locator(locator)
                    axs[i].xaxis.set_major_formatter(formatter)
                    handles, labels = axs[i].get_legend_handles_labels()
                    #Legend 1
                    leg1 = axs[i].legend(reversed(handles), reversed(labels), loc='lower left',bbox_to_anchor=(1,0),facecolor='inherit', frameon=True)
                    # Manually add the first legend back
                    axs[i].add_artist(leg1)

                i=i+1

            all_axes = fig1.get_axes()

            self.xlabels = pd.Series(self.xlabels).str.replace('_',' ').str.wrap(10, break_long_words=False)

            j=0
            k=0
            for ax in all_axes:
                if ax.is_last_row():
                    ax.set_xlabel(xlabel=(self.xlabels[j]),  color='black')
                    j=j+1
                if ax.is_first_col():
                    ax.set_ylabel(ylabel=(self.ylabels[k]),  color='black', rotation='vertical')
                    k=k+1

            fig1.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            plt.ylabel('Thermal capacity reserve ({})'.format(unitconversion['units']),  color='black', rotation='vertical', labelpad=60)

            #fig1.savefig('/home/mschwarz/PLEXOS results analysis/test/SPP_thermal_cap_reserves_test', dpi=600, bbox_inches='tight') #Test

            # If Data_Table_Out is empty, does not return data or figure
            if Data_Table_Out.empty == True:
                out = mfunc.MissingZoneData()
                outputs[zone_input] = out
                continue

            outputs[zone_input] = {'fig': fig1, 'data_table': Data_Table_Out}
        return outputs
