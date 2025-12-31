# -*- coding: utf-8 -*-
"""System unserved energy plots.

This module creates unserved energy timeseries line plots and total bar
plots and is called from marmot_plot_main.py

@author: Daniel Levie 
"""

import logging
from pathlib import Path
from typing import List

import pandas as pd
import numpy as np

import marmot.utils.mconfig as mconfig
from marmot.plottingmodules.plotutils.plot_data_helper import PlotDataStoreAndProcessor
from marmot.plottingmodules.plotutils.plot_exceptions import (
    MissingInputData,
    MissingZoneData,
)
from marmot.plottingmodules.plotutils.plot_library import PlotLibrary
from marmot.plottingmodules.plotutils.styles import ColorList
from marmot.plottingmodules.plotutils.timeseries_modifiers import (
    get_sub_hour_interval_count,
    set_timestamp_date_range,
)

logger = logging.getLogger("plotter." + __name__)
plot_data_settings: dict = mconfig.parser("plot_data")


class SRMC(PlotDataStoreAndProcessor):

    def __init__(
        self,
        Zones: List[str],
        Scenarios: List[str],
        AGG_BY: str,
        ordered_gen: List[str],
        marmot_solutions_folder: Path,
        custom_xticklabels: List[str] = None,
        color_list: list = ColorList().colors,
        **kwargs,
    ):
        """
        Args:
            Zones (List[str]): List of regions/zones to plot.
            Scenarios (List[str]): List of scenarios to plot.
            AGG_BY (str): Informs region type to aggregate by when creating plots.
            ordered_gen (List[str]): Ordered list of generator technologies to plot,
                order defines the generator technology position in stacked bar and area plots.
            marmot_solutions_folder (Path): Directory containing Marmot solution outputs.
            custom_xticklabels (List[str], optional): List of custom x labels to
                apply to barplots. Values will overwite existing ones.
                Defaults to None.
            color_list (list, optional): List of colors to apply to non-gen plots.
                Defaults to ColorList().colors.
        """
        # Instantiation of PlotDataStoreAndProcessor
        super().__init__(AGG_BY, ordered_gen, marmot_solutions_folder, **kwargs)

        self.Zones = Zones
        self.Scenarios = Scenarios
        self.custom_xticklabels = custom_xticklabels
        self.color_list = color_list

    def SRMC(
        self,
        start_date_range: str = None,
        end_date_range: str = None,
        data_resolution: str = "",
        **_,
    ):

        outputs: dict = {}

        if self.AGG_BY == "zone":
            agg = "zone"
        else:
            agg = "region"

        # List of properties needed by the plot, properties are a set of tuples and
        # contain 3 parts: required True/False, property name and scenarios required,
        # scenarios must be a list.
        properties = [(True, 'generator_SRMC', self.Scenarios)]

        check_input_data = self.get_formatted_data(properties)

        if 1 in check_input_data:
            return MissingInputData()

        for zone_input in self.Zones:
            logger.info(f"Zone = {zone_input}")
            unserved_energy_chunks = []

            for scenario in self.Scenarios:
                logger.info(f"Scenario = {scenario}")

                SRMC: pd.DataFrame = self[
                    'generator_SRMC'
                ].get(scenario)
                SRMC = SRMC.xs(zone_input, level=self.AGG_BY)
                SRMC.to_csv(f"/projects/midc/Grant/analysis/{scenario}_SRMC.csv")