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
from typing import List
from pathlib import Path

import marmot.utils.mconfig as mconfig

from marmot.plottingmodules.plotutils.styles import GeneratorColorDict
from marmot.plottingmodules.plotutils.plot_library import PlotLibrary
from marmot.plottingmodules.plotutils.plot_data_helper import PlotDataStoreAndProcessor, GenCategories
from marmot.plottingmodules.plotutils.timeseries_modifiers import set_timestamp_date_range
from marmot.plottingmodules.plotutils.plot_exceptions import (
    MissingInputData,
    InputSheetError,
    MissingZoneData,
)

logger = logging.getLogger("plotter." + __name__)
plot_data_settings: dict = mconfig.parser("plot_data")


class Emissions(PlotDataStoreAndProcessor):
    """Generator emissions plots.

    The emissions.py module contains methods that are
    related to the fossil fuel emissions of generators.

    Emissions inherits from the PlotDataStoreAndProcessor class to assist
    in creating figures.
    """

    def __init__(self, 
        Zones: List[str], 
        Scenarios: List[str], 
        AGG_BY: str,
        ordered_gen: List[str],
        marmot_solutions_folder: Path,
        marmot_color_dict: dict = None,
        custom_xticklabels: List[str] = None,
        **kwargs):
        """
        Args:
            Zones (List[str]): List of regions/zones to plot.
            Scenarios (List[str]): List of scenarios to plot.
            AGG_BY (str): Informs region type to aggregate by when creating plots.
            ordered_gen (List[str]): Ordered list of generator technologies to plot,
                order defines the generator technology position in stacked bar and area plots.
            marmot_solutions_folder (Path): Directory containing Marmot solution outputs.
            marmot_color_dict (dict, optional): Dictionary of colors to use for 
                generation technologies.
                Defaults to None.
            custom_xticklabels (List[str], optional): List of custom x labels to 
                apply to barplots. Values will overwite existing ones. 
                Defaults to None.
        """
        # Instantiation of PlotDataStoreAndProcessor
        super().__init__(AGG_BY, ordered_gen, marmot_solutions_folder, **kwargs)

        self.Zones = Zones
        self.Scenarios = Scenarios
        if marmot_color_dict is None:
            self.marmot_color_dict = GeneratorColorDict.set_random_colors(self.ordered_gen).color_dict
        else:
            self.marmot_color_dict = marmot_color_dict
        self.custom_xticklabels = custom_xticklabels

    def total_emissions_by_type(
        self,
        prop: str = None,
        start_date_range: str = None,
        end_date_range: str = None,
        custom_data_file_path: Path = None,
        scenario_groupby: str = "Scenario",
        **_,
    ):
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
            custom_data_file_path (Path, optional): Path to custom data file to concat extra
                data. Index and column format should be consistent with output data csv.
            scenario_groupby (str, optional): Specifies whether to group data by Scenario
                or Year-Sceanrio. If grouping by Year-Sceanrio the year will be identified 
                from the timestamp and appeneded to the sceanrio name. This is useful when 
                plotting data which covers multiple years such as ReEDS.
                Defaults to Scenario.

                .. versionadded:: 0.10.0

        Returns:
            dict: dictionary containing the created plot and its data table.
        """
        # Create Dictionary to hold Datframes for each scenario
        outputs: dict = {}

        # List of properties needed by the plot, properties are a set of tuples and 
        # contain 3 parts: required True/False, property name and scenarios required, 
        # scenarios must be a list.
        properties = [(True, "emissions_generators_Production", self.Scenarios)]

        # Runs get_formatted_data within PlotDataStoreAndProcessor to populate PlotDataStoreAndProcessor dictionary
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)

        if 1 in check_input_data:
            return MissingInputData()

        for zone_input in self.Zones:
            emitList = []
            logger.info(f"Zone = {zone_input}")

            # collect data for all scenarios and pollutants
            for scenario in self.Scenarios:

                logger.info(f"Scenario = {scenario}")

                emit: pd.DataFrame = self["emissions_generators_Production"].get(
                    scenario
                )

                # Check if Total_Gen_Stack contains zone_input, skips if not
                try:
                    emit = emit.xs(zone_input, level=self.AGG_BY)
                except KeyError:
                    logger.warning(f"No {prop} emissions in Scenario : {scenario}")
                    continue

                if pd.notna(start_date_range):
                    emit = set_timestamp_date_range(
                        emit, start_date_range, end_date_range
                    )
                    if emit.empty:
                        emit.warning(f"No {prop} emissions in selected Date Range")
                        continue

                # Rename generator technologies
                emit = self.rename_gen_techs(emit)
                # summarize annual emissions by pollutant and tech
                emitList.append(
                    self.year_scenario_grouper(
                        emit,
                        scenario,
                        groupby=scenario_groupby,
                        additional_groups=["pollutant", "tech"],
                    ).sum()
                )
            # concatenate chunks
            try:
                emitOut = pd.concat(emitList, axis=0)
            except ValueError:
                logger.warning(f"No emissions found for : {zone_input}")
                out = MissingZoneData()
                outputs[zone_input] = out
                continue

            # format results
            emitOut = emitOut / 1e9  # Convert from kg to million metric tons
            emitOut = emitOut.loc[
                (emitOut != 0).any(axis=1), :
            ]  # drop any generators with no emissions

            # subset to relevant pollutant (specified by user as property)
            try:
                emitPlot = emitOut.xs(prop, level="pollutant")
            except KeyError:
                logger.warning(f"{prop} emissions not found")
                outputs = InputSheetError()
                return outputs

            emitPlot = emitPlot.reset_index()
            emitPlot = emitPlot.pivot(index="Scenario", columns="tech", values=0)

            if pd.notna(custom_data_file_path):
                emitPlot = self.insert_custom_data_columns(
                    emitPlot, custom_data_file_path
                )

            # Checks if emitOut contains data, if not skips zone and does not return a plot
            if emitPlot.empty:
                out = MissingZoneData()
                outputs[zone_input] = out
                continue

            dataOut = emitPlot

            # single pollutant plot
            mplt = PlotLibrary()
            fig, ax = mplt.get_figure()

            # Set x-tick labels
            if self.custom_xticklabels:
                tick_labels = self.custom_xticklabels
            else:
                tick_labels = emitPlot.index

            mplt.barplot(
                emitPlot,
                color=self.marmot_color_dict,
                stacked=True,
                custom_tick_labels=tick_labels,
            )

            ax.set_ylabel(
                f"Annual {prop} Emissions\n(million metric tons)",
                color="black",
                rotation="vertical",
            )
            # Add legend
            mplt.add_legend(reverse_legend=True)
            # Add title
            if plot_data_settings["plot_title_as_region"]:
                mplt.add_main_title(zone_input)

            outputs[zone_input] = {"fig": fig, "data_table": dataOut}

        return outputs
