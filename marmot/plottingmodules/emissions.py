# -*- coding: utf-8 -*-
"""
Created on Mon Nov 2 8:41:40 2020

@author: Brian Sergi

TO DO:
    - fix pollutant subsetting (faceted)
    - units formatting
"""

import pandas as pd
import textwrap
import matplotlib as mpl
import matplotlib.pyplot as plt
import marmot.plottingmodules.marmot_plot_functions as mfunc
import logging
import marmot.config.mconfig as mconfig

#===============================================================================

class MPlot(object):
    def __init__(self, argument_dict):
        # iterate over items in argument_dict and set as properties of class
        # see key_list in Marmot_plot_main for list of properties
        for prop in argument_dict:
            self.__setattr__(prop, argument_dict[prop])
        self.logger = logging.getLogger('marmot_plot.'+__name__)

        self.x = mconfig.parser("figure_size","xdimension")
        self.y = mconfig.parser("figure_size","ydimension")
        self.y_axes_decimalpt = mconfig.parser("axes_options","y_axes_decimalpt")
        self.mplot_data_dict = {}

    # function to collect total emissions by fuel type
    def total_emissions_by_type(self, figure_name=None, prop=None, start=None,
                             end=None, timezone="", start_date_range=None,
                             end_date_range=None):

        # Create Dictionary to hold Datframes for each scenario
        outputs = {}

        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True,"emissions_generators_Production",self.Scenarios)]

        # Runs get_data to populate mplot_data_dict with all required properties, returns a 1 if required data is missing
        check_input_data = mfunc.get_data(self.mplot_data_dict, properties,self.Marmot_Solutions_folder)

        if 1 in check_input_data:
            return mfunc.MissingInputData()

        for zone_input in self.Zones:
            emitList = []
            self.logger.info(f"Zone = {zone_input}")

            # collect data for all scenarios and pollutants
            for scenario in self.Scenarios:

                self.logger.info(f"Scenario = {scenario}")

                emit = self.mplot_data_dict["emissions_generators_Production"].get(scenario)

                # Check if Total_Gen_Stack contains zone_input, skips if not
                try:
                    emit = emit.xs(zone_input,level=self.AGG_BY)
                except KeyError:
                    self.logger.warning(f"No emissions in Scenario : {scenario}")
                    continue

                # summarize annual emissions by pollutant and tech
                emit = emit.groupby(['pollutant', 'tech']).sum()

                # rename column based on scenario
                emit.rename(columns={0:scenario}, inplace=True)
                emitList.append(emit)

            # concatenate chunks
            try:
                emitOut = pd.concat(emitList, axis=1)
            except ValueError:
                self.logger.warning(f"No emissions found for : {zone_input}")
                out = mfunc.MissingZoneData()
                outputs[zone_input] = out
                continue

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
                emitPlot = emitOut.xs(prop, level="pollutant").T
                dataOut = emitPlot.copy()

                # formatting for plot
                emitPlot.index = emitPlot.index.str.replace('_',' ')
                
                # single pollutant plot
                fig1, ax = plt.subplots(figsize=(self.x,self.y))
                emitPlot.plot.bar(stacked=True,
                             color=[self.PLEXOS_color_dict.get(x, '#333333') for x in emitPlot.columns.values], edgecolor='black', linewidth='0.1',ax=ax)

                # plot formatting
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.set_ylabel('Annual ' + prop + ' Emissions\n(million metric tons)',  color='black', rotation='vertical')
                #adds comma to y axis data
                ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(x, f',.{self.y_axes_decimalpt}f')))
                
                # Set x-tick labels 
                if len(self.custom_xticklabels) > 1:
                    tick_labels = self.custom_xticklabels
                else:
                    tick_labels = emitPlot.index
                mfunc.set_barplot_xticklabels(tick_labels, ax=ax)
                
                ax.tick_params(axis='y', which='major', length=5, width=1)
                ax.tick_params(axis='x', which='major', length=5, width=1)

                # legend formatting
                handles, labels = ax.get_legend_handles_labels()
                leg1 = ax.legend(reversed(handles), reversed(labels), loc='lower left',bbox_to_anchor=(1,0),
                              facecolor='inherit', frameon=True)
                ax.add_artist(leg1)
                if mconfig.parser("plot_title_as_region"):
                    ax.set_title(zone_input)

                outputs[zone_input] = {'fig': fig1, 'data_table': dataOut}

            except KeyError:
                self.logger.warning(self.prop+ " emissions not found")
                outputs = mfunc.InputSheetError()
                return outputs

        return outputs
