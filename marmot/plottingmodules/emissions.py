# -*- coding: utf-8 -*-
"""Generator emissions plots.

This module plots figures related to the fossil fuel emissions of generators. 

@author: Brian Sergi

TO DO:
    - fix pollutant subsetting (faceted)
    - units formatting
"""

import logging
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import marmot.config.mconfig as mconfig
from marmot.plottingmodules.plotutils.plot_data_helper import PlotDataHelper
from marmot.plottingmodules.plotutils.plot_exceptions import (MissingInputData, InputSheetError,
                MissingZoneData)


class MPlot(PlotDataHelper):
    """emissions MPlot class.

    All the plotting modules use this same class name.
    This class contains plotting methods that are grouped based on the
    current module name.
    
    The emissions.py module contains methods that are
    related to the fossil fuel emissions of generators. 
    
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

        self.logger = logging.getLogger('marmot_plot.'+__name__)

        self.x = mconfig.parser("figure_size","xdimension")
        self.y = mconfig.parser("figure_size","ydimension")
        self.y_axes_decimalpt = mconfig.parser("axes_options","y_axes_decimalpt")
        

    # function to collect total emissions by fuel type
    def total_emissions_by_type(self, prop: str = None, start_date_range: str = None,
                                end_date_range: str = None, custom_data_file_path: str = None,
                                **_):
        """Creates a stacked bar plot of emissions by generator tech type.

        The emission type to plot is defined using the prop argument.
        A separate bar is created for each scenario.

        Args:
            prop (str, optional): Controls type of emission to plot.
                Controlled through the plot_select.csv.
                Defaults to None.
            start_date_range (str, optional): Defines a start date at which to represent data from. 
                Defaults to None.
            end_date_range (str, optional): Defines a end date at which to represent data to.
                Defaults to None.
            custom_data_file_path (str, optional): Path to custom data file to concat extra 
                data. Index and column format should be consistent with output data csv.

        Returns:
            dict: dictionary containing the created plot and its data table.
        """
        # Create Dictionary to hold Datframes for each scenario
        outputs = {}

        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True,"emissions_generators_Production",self.Scenarios)]

        # Runs get_formatted_data within PlotDataHelper to populate PlotDataHelper dictionary  
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)

        if 1 in check_input_data:
            return MissingInputData()

        for zone_input in self.Zones:
            emitList = []
            self.logger.info(f"Zone = {zone_input}")

            # collect data for all scenarios and pollutants
            for scenario in self.Scenarios:

                self.logger.info(f"Scenario = {scenario}")

                emit = self["emissions_generators_Production"].get(scenario)

                # Check if Total_Gen_Stack contains zone_input, skips if not
                try:
                    emit = emit.xs(zone_input,level=self.AGG_BY)
                except KeyError:
                    self.logger.warning(f"No emissions in Scenario : {scenario}")
                    continue
                
                # Rename generator technologies
                emit = self.rename_gen_techs(emit)
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
                out = MissingZoneData()
                outputs[zone_input] = out
                continue

            # format results
            emitOut = emitOut/1E9 # Convert from kg to million metric tons
            emitOut = emitOut.loc[(emitOut != 0).any(axis=1), :] # drop any generators with no emissions
            emitOut = emitOut.T  # transpose back (easier for slicing by pollutant later)

            # subset to relevant pollutant (specified by user as property)
            try:
                emitPlot = emitOut.xs(prop, level="pollutant", axis=1)
            except KeyError:
                self.logger.warning(prop+ " emissions not found")
                outputs = InputSheetError()
                return outputs

            if pd.notna(custom_data_file_path):
                emitPlot = self.insert_custom_data_columns(
                                                        emitPlot, 
                                                        custom_data_file_path)

            # Checks if emitOut contains data, if not skips zone and does not return a plot
            if emitPlot.empty:
                out = MissingZoneData()
                outputs[zone_input] = out
                continue
        
            dataOut = emitPlot.copy()
            # single pollutant plot
            fig1, ax = plt.subplots(figsize=(self.x,self.y))
            emitPlot.plot.bar(stacked=True,
                            color=[self.PLEXOS_color_dict.get(x, '#333333') for x in emitPlot.columns.values], 
                            edgecolor='black', linewidth='0.1', ax=ax)

            # plot formatting
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_xlabel("")
            ax.set_ylabel(f"Annual {prop} Emissions\n(million metric tons)", 
                          color='black', rotation='vertical')
            #adds comma to y axis data
            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(x, f',.{self.y_axes_decimalpt}f')))
            
            # Set x-tick labels 
            if len(self.custom_xticklabels) > 1:
                tick_labels = self.custom_xticklabels
            else:
                tick_labels = emitPlot.index
            PlotDataHelper.set_barplot_xticklabels(tick_labels, ax=ax)
            
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

            

        return outputs
