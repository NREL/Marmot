# -*- coding: utf-8 -*-
"""
Created on Mon Nov 2 8:41:40 2020

@author: Brian Sergi

TO DO:
    - fix pollutant subsetting (faceted)
    - units formatting
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import marmot_plot_functions as mfunc
import logging

#===============================================================================

class mplot(object):
    def __init__(self, argument_dict):
        # iterate over items in argument_dict and set as properties of class
        # see key_list in Marmot_plot_main for list of properties
        for prop in argument_dict:
            self.__setattr__(prop, argument_dict[prop])
        self.logger = logging.getLogger('marmot_plot.'+__name__)

    # function to collect total emissions by fuel type
    def total_emissions_by_type(self):
        # Create Dictionary to hold Datframes for each scenario
        outputs = {}
        emit_gen_collection = {}
        check_input_data = []
        check_input_data.extend([mfunc.get_data(emit_gen_collection,"emissions_generators_Production", self.Marmot_Solutions_folder, self.Multi_Scenario)])

        # Checks if all data required by plot is available, if 1 in list required data is missing
        if 1 in check_input_data:
            outputs = mfunc.MissingInputData()
            return outputs

        for zone_input in self.Zones:
            emitList = []
            self.logger.info("Zone = " + zone_input)

            # collect data for all scenarios and pollutants
            for scenario in self.Multi_Scenario:

                self.logger.info("Scenario = " + scenario)

                emit = emit_gen_collection.get(scenario)

                # Check if Total_Gen_Stack contains zone_input, skips if not
                try:
                    emit = emit.xs(zone_input,level=self.AGG_BY)
                except KeyError:
                    self.logger.warning("No emissions found for : "+zone_input)
                    continue

                # summarize annual emissions by pollutant and tech
                emit = emit.groupby(['pollutant', 'tech']).sum()

                # rename column based on scenario
                emit.rename(columns={0:scenario}, inplace=True)
                emitList.append(emit)

            # concatenate chunks
            emitOut = pd.concat(emitList, axis=1)

            # format results
            emitOut = emitOut.T/1E6 # Convert from metric tons to million metric tons
            emitOut = emitOut.loc[:, (emitOut != 0).any(axis=0)] # drop any generators with no emissions
            emitOut = emitOut.T  # transpose back (easier for slicing by pollutant later)

            # Checks if emitOut contains data, if not skips zone and does not return a plot
            if emitOut.empty:
                out = mfunc.MissingZoneData()
                outputs[zone_input] = out
                continue

            # subset to relevant pollutant (specified by user as property)
            try:
                emitPlot = emitOut.xs(self.prop, level="pollutant").T
                dataOut = emitPlot.copy()

                # formatting for plot
                emitPlot.index = emitPlot.index.str.replace('_',' ')
                emitPlot.index = emitPlot.index.str.wrap(10, break_long_words=False)

                # single pollutant plot
                fig1 = emitPlot.plot.bar(stacked=True, figsize=(6,4), rot=0,
                             color=[self.PLEXOS_color_dict.get(x, '#333333') for x in emitPlot.columns.values], edgecolor='black', linewidth='0.1')

                # plot formatting
                fig1.spines['right'].set_visible(False)
                fig1.spines['top'].set_visible(False)
                fig1.set_ylabel('Annual ' + self.prop + ' Emissions\n(million metric tons)',  color='black', rotation='vertical')
                #adds comma to y axis data
                fig1.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
                fig1.tick_params(axis='y', which='major', length=5, width=1)
                fig1.tick_params(axis='x', which='major', length=5, width=1)

                # legend formatting
                handles, labels = fig1.get_legend_handles_labels()
                leg1 = fig1.legend(reversed(handles), reversed(labels), loc='lower left',bbox_to_anchor=(1,0),
                              facecolor='inherit', frameon=True)
                fig1.add_artist(leg1)

                # replace x-axis with custom labels
                if len(self.ticklabels) > 1:
                    self.ticklabels = pd.Series(self.ticklabels).str.replace('-','- ').str.wrap(8, break_long_words=True)
                    fig1.set_xticklabels(self.ticklabels)

                outputs[zone_input] = {'fig': fig1, 'data_table': dataOut}

            except KeyError:
                self.logger.warning(self.prop+ " emissions not found")
                outputs = mfunc.InputSheetError()
                return outputs

        return outputs
