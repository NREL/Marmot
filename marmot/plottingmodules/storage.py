# -*- coding: utf-8 -*-
"""Energy storage plots.

This module creates energy storage plots.
"""

import logging
import pandas as pd
import matplotlib.pyplot as plt

import marmot.utils.mconfig as mconfig

from marmot.plottingmodules.plotutils.plot_library import SetupSubplot
from marmot.plottingmodules.plotutils.plot_data_helper import MPlotDataHelper
from marmot.plottingmodules.plotutils.plot_exceptions import (
    MissingInputData,
    MissingZoneData,
)

logger = logging.getLogger("plotter." + __name__)
plot_data_settings = mconfig.parser("plot_data")


class Storage(MPlotDataHelper):
    """Energy storage plots.

    The storage.py module contains methods that are
    related to storage devices.

    Storage inherits from the MPlotDataHelper class to assist
    in creating figures.
    """

    def __init__(self, *args, **kwargs):
        """
        Args:
            *args
                Minimum required parameters passed to the MPlotDataHelper 
                class.
            **kwargs
                These parameters will be passed to the MPlotDataHelper 
                class.
        """
        # Instantiation of MPlotHelperFunctions
        super().__init__(*args, **kwargs)

    def storage_volume(
        self,
        timezone: str = "",
        start_date_range: str = None,
        end_date_range: str = None,
        **_,
    ):
        """Creates time series plot of aggregate storage volume for all storage objects in a given region.

        A horizontal line represents full charge of the storage device.
        All scenarios are plotted on a single figure.

        Args:
            timezone (str, optional): The timezone to display on the x-axes.
                Defaults to "".
            start_date_range (str, optional): Defines a start date at which to represent data from.
                Defaults to None.
            end_date_range (str, optional): Defines a end date at which to represent data to.
                Defaults to None.

        Returns:
            dict: Dictionary containing the created plot and its data table.
        """
        if self.AGG_BY == "zone":
            agg = "zone"
        else:
            agg = "region"

        outputs: dict = {}

        # List of properties needed by the plot, properties are a set of tuples and 
        # contain 3 parts: required True/False, property name and scenarios required, 
        # scenarios must be a list.
        properties = [
            (True, "storage_Initial_Volume", self.Scenarios),
            (True, f"{agg}_Unserved_Energy", self.Scenarios),
            (True, "storage_Max_Volume", [self.Scenarios[0]]),
        ]

        # Runs get_formatted_data within MPlotDataHelper to populate MPlotDataHelper dictionary
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)

        if 1 in check_input_data:
            return MissingInputData()

        for zone_input in self.Zones:
            logger.info(f"{self.AGG_BY} = {zone_input}")

            stor_chunk = []
            use_chunk = []

            for scenario in self.Multi_Scenario:

                logger.info(f"Scenario = {str(scenario)}")

                storage_volume_read = self["storage_Initial_Volume"].get(scenario)
                try:
                    storage_volume: pd.DataFrame = storage_volume_read.xs(
                        zone_input, level=self.AGG_BY
                    )
                except KeyError:
                    logger.warning(f"No storage resources in {zone_input}")
                    outputs[zone_input] = MissingZoneData()
                    continue

                # Isolate only head storage objects (not tail).
                storage_gen_units = storage_volume.index.get_level_values(
                    "storage_resource"
                )
                head_units = [unit for unit in storage_gen_units if "head" in unit]
                storage_volume = storage_volume.iloc[
                    storage_volume.index.get_level_values("storage_resource").isin(
                        head_units
                    )
                ]
                storage_volume = storage_volume.groupby("timestamp").sum()
                storage_volume.columns = [scenario]

                max_volume = storage_volume.max().squeeze()
                try:
                    max_volume: pd.DataFrame = self["storage_Max_Volume"].get(scenario)
                    max_volume = max_volume.xs(zone_input, level=self.AGG_BY)
                    max_volume = max_volume.groupby("timestamp").sum()
                    max_volume = max_volume.squeeze()
                except KeyError:
                    logger.warning(f"No storage resources in {zone_input}")

                use_read: pd.DataFrame = self[f"{agg}_Unserved_Energy"].get(scenario)
                use = use_read.xs(zone_input, level=self.AGG_BY)
                use = use.groupby("timestamp").sum() / 1000
                use.columns = [scenario]

                if pd.notna(start_date_range):
                    storage_volume, max_volume, use = self.set_timestamp_date_range(
                        [storage_volume, max_volume, use],
                        start_date_range,
                        end_date_range,
                    )
                    if storage_volume.empty is True:
                        logger.warning("No Storage resources in selected Date Range")
                        continue

                stor_chunk.append(storage_volume)
                use_chunk.append(use)

            storage_volume_all_scenarios = pd.concat(stor_chunk, axis=1)

            use_all_scenarios = pd.concat(use_chunk, axis=1)

            # Data table of values to return to main program
            Data_Table_Out = pd.concat(
                [storage_volume_all_scenarios, use_all_scenarios], axis=1
            )
            # Make scenario/color dictionary.
            color_dict = dict(
                zip(storage_volume_all_scenarios.columns, self.color_list)
            )

            mplt = SetupSubplot(nrows=2, squeeze=False, ravel_axs=True)
            fig, axs = mplt.get_figure()
            plt.subplots_adjust(wspace=0.05, hspace=0.2)

            if storage_volume_all_scenarios.empty:
                out = MissingZoneData()
                outputs[zone_input] = out
                continue

            for column in storage_volume_all_scenarios:
                axs[0].plot(
                    storage_volume_all_scenarios.index.values,
                    storage_volume_all_scenarios[column],
                    linewidth=1,
                    color=color_dict[column],
                    label=column,
                )

                axs[0].set_ylabel(
                    "Head Storage Volume (GWh)", color="black", rotation="vertical"
                )
                mplt.set_yaxis_major_tick_format(sub_pos=0)
                axs[0].margins(x=0.01)
                mplt.set_subplot_timeseries_format(sub_pos=0)
                axs[0].set_ylim(ymin=0)
                axs[0].set_title(zone_input)

                axs[1].plot(
                    use_all_scenarios.index.values,
                    use_all_scenarios[column],
                    linewidth=1,
                    color=color_dict[column],
                    label=f"{column} Unserved Energy",
                )

                axs[1].set_ylabel(
                    "Unserved Energy (GWh)", color="black", rotation="vertical"
                )
                axs[1].set_xlabel(timezone, color="black", rotation="horizontal")
                mplt.set_yaxis_major_tick_format(sub_pos=1)
                axs[1].margins(x=0.01)
                mplt.set_subplot_timeseries_format(sub_pos=1)

            mplt.set_yaxis_major_tick_format()
            axs[0].axhline(y=max_volume, linestyle=":", label="Max Volume")
            axs[0].legend(loc="lower left", bbox_to_anchor=(1.15, 0))
            axs[1].legend(loc="lower left", bbox_to_anchor=(1.15, 0.2))
            if plot_data_settings["plot_title_as_region"]:
                mplt.add_main_title(zone_input)

            outputs[zone_input] = {"fig": fig, "data_table": Data_Table_Out}
        return outputs
