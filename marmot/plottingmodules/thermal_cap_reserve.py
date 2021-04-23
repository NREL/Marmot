# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 10:34:48 2019


@author: dlevie
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
import logging
import marmot.plottingmodules.marmot_plot_functions as mfunc
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
            self.logger.info(f"Zone = {zone_input}")

            xdimension=len(self.xlabels)
            if xdimension == 0:
                xdimension = 1
            ydimension=len(self.ylabels)
            if ydimension == 0:
                ydimension = 1
            
            grid_size = xdimension*ydimension

            # Used to calculate any excess axis to delete
            plot_number = len(self.Scenarios)
            excess_axs = grid_size - plot_number
            
            fig1, axs = mfunc.setup_plot(xdimension,ydimension)
            plt.subplots_adjust(wspace=0.05, hspace=0.2)
            
            # holds list of unique generation technologies
            unique_tech_names = []
            data_table_chunks = []

            for i, scenario in enumerate(self.Scenarios):

                self.logger.info(f"Scenario = {scenario}")

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
                    self.logger.warning(f"No installed capacity in: {zone_input}")
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
                mfunc.set_plot_timeseries_format(axs,i)
                
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

            self.xlabels = pd.Series(self.xlabels).str.replace('_',' ').str.wrap(10, break_long_words=False)
            # add facet labels
            mfunc.add_facet_labels(fig1, self.xlabels, self.ylabels)           
            
            #Remove extra axes
            if excess_axs != 0:
                mfunc.remove_excess_axs(axs,excess_axs,grid_size)
            
            fig1.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            plt.ylabel(f"Thermal capacity reserve ({unitconversion['units']})",  color='black', rotation='vertical', labelpad=60)

            # Concat all data tables together
            Data_Table_Out = pd.concat(data_table_chunks, copy=False, axis=0)
            
            # If Data_Table_Out is empty, does not return data or figure
            if Data_Table_Out.empty == True:
                out = mfunc.MissingZoneData()
                outputs[zone_input] = out
                continue

            outputs[zone_input] = {'fig': fig1, 'data_table': Data_Table_Out}
        return outputs
