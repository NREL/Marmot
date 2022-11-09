# -*- coding: utf-8 -*-
"""Energy storage plots.

This module creates energy storage plots.
"""

import logging
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from pathlib import Path

import marmot.utils.mconfig as mconfig
from marmot.plottingmodules.plotutils.styles import ColorList
from marmot.plottingmodules.plotutils.plot_library import PlotLibrary
from marmot.plottingmodules.plotutils.plot_data_helper import (
    PlotDataStoreAndProcessor,
    GenCategories,
    set_x_y_dimension,
)
from marmot.plottingmodules.plotutils.timeseries_modifiers import (
    set_timestamp_date_range,
    adjust_for_leapday,
    sort_duration,
)
from marmot.plottingmodules.plotutils.plot_exceptions import (
    MissingInputData,
    DataSavedInModule,
    MissingZoneData,
)
from marmot.plottingmodules.plotutils.plot_library import SetupSubplot

logger = logging.getLogger("plotter." + __name__)
plot_data_settings: dict = mconfig.parser("plot_data")
shift_leapday: bool = mconfig.parser("shift_leapday")


class Storage(PlotDataStoreAndProcessor):
    """Energy storage plots.

    The storage.py module contains methods that are
    related to storage devices.

    Storage inherits from the PlotDataStoreAndProcessor class to assist
    in creating figures.
    """

    def __init__(
        self,
        Zones: List[str],
        Scenarios: List[str],
        AGG_BY: str,
        ordered_gen: List[str],
        marmot_solutions_folder: Path,
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
            color_list (list, optional): List of colors to apply to non-gen plots.
                Defaults to ColorList().colors.
        """
        # Instantiation of PlotDataStoreAndProcessor
        super().__init__(AGG_BY, ordered_gen, marmot_solutions_folder, **kwargs)

        self.Zones = Zones
        self.Scenarios = Scenarios
        self.color_list = color_list

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

        # Runs get_formatted_data within PlotDataStoreAndProcessor to populate PlotDataStoreAndProcessor dictionary
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
                    storage_volume, max_volume, use = set_timestamp_date_range(
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

    def ind_bat(
        self,
        figure_name: str,
        prop: str,
        timezone: str = "",
        start_date_range: str = None,
        end_date_range: str = None,
        **_,
    ):
        """Creates time series or duration curve plot of generation or load for a single battery.

        A horizontal line represents max discharge power of the battery.
        All scenarios are plotted on a single figure.

        Args:
            figure_name (str): Figure output name. Used to control whether plot shows generation or load,
                                as well as duration curve or chronological time series.
            timezone (str, optional): The timezone to display on the x-axes.
                Defaults to "".
            prop (str): Used to pass in battery names.
                Input format should be a comma seperated string.
            start_date_range (str, optional): Defines a start date at which to represent data from.
                Defaults to None.
            end_date_range (str, optional): Defines a end date at which to represent data to.
                Defaults to None.

        Returns:
            dict: Dictionary containing the created plot and its data table.
        """
        duration_curve = False
        if "duration_curve" in figure_name:
            duration_curve = True

        var = "Generation"
        var_name = "Battery discharge"
        if "Load" in figure_name:
            var = "Load"
            var_name = "Battery charging"

        if self.AGG_BY == "zone":
            agg = "zone"
        else:
            agg = "region"

        outputs: dict = {}

        # List of properties needed by the plot, properties are a set of tuples and
        # contain 3 parts: required True/False, property name and scenarios required,
        # scenarios must be a list.
        properties = [
            (True, f"batterie_{var}", self.Scenarios),
        ]
        # Runs get_formatted_data within PlotDataStoreAndProcessor to populate PlotDataStoreAndProcessor dictionary
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)

        if 1 in check_input_data:
            return MissingInputData()

        # Select only lines specified in Marmot_plot_select.csv.
        select_bats = [x.strip() for x in prop.split(",")]
        logger.info(f"Plotting only batteries specified in Marmot_plot_select.csv")
        logger.info(select_bats)

        xdim, ydim = set_x_y_dimension(len(select_bats))
        mplt = PlotLibrary(ydim, xdim, squeeze=False, ravel_axs=True, sharey=True)
        fig, axs = mplt.get_figure()

        data_tables = []
        for n, bat in enumerate(select_bats):
            logger.info(f"Battery = {bat}")
            chunks_bat = []

            for scenario in self.Scenarios:
                logger.info(f"Scenario = {scenario}")
                gen = self[f"batterie_{var}"].get(scenario)
                try:
                    gen = gen.xs(bat, level=f"battery_name")
                except KeyError:
                    logger.warning(f"{bat} not found in results.")
                    return MissingInputData()

                gen = gen.droplevel(["units", "category"])
                gen = gen.rename(columns={"values": scenario})

                if shift_leapday:
                    gen = adjust_for_leapday(gen)

                if pd.notna(start_date_range):
                    gen = set_timestamp_date_range(
                        gen, start_date_range, end_date_range
                    )
                    if gen.empty is True:
                        logger.warning("No data in selected Date Range")
                        continue

                chunks_bat.append(gen)

                # For output time series.csv
                gen_out = gen.rename(columns={scenario: f"battery {var}"})
                scenario_names = pd.Series([scenario] * len(gen_out), name="Scenario")
                gen_name = pd.Series([bat] * len(gen_out), name="battery")
                gen_out = gen_out.set_index([scenario_names, gen_name], append=True)

                # print(gen_out)
                data_tables.append(gen_out)

            if chunks_bat:
                gen_out = pd.concat(chunks_bat, axis=1)
            else:
                return MissingInputData()

            # Only convert on first lines
            if n == 0:
                unitconversion = self.capacity_energy_unitconversion(
                    gen_out, self.Scenarios, sum_values=False
                )
            gen_out = gen_out / unitconversion["divisor"]

            legend_order = []
            # Plot line flow
            for column in gen_out:
                if duration_curve:
                    gen_single = sort_duration(gen_out, column)
                else:
                    gen_single = gen_out
                legend_label = f"{column}"
                mplt.lineplot(gen_single, column, label=legend_label, sub_pos=n)
                legend_order.append(legend_label)

            # Get and process all line limits
            # line_limits = self.get_line_interface_limits(
            #                 [f"{connection}_Export_Limit",
            #                     f"{connection}_Import_Limit",
            #                 ],
            #                 line,
            # ) / unitconversion["divisor"]
            # # Plot line limits
            # self.plot_line_interface_limits(line_limits, mplt, n, duration_curve)

            # axs[n].set_title(line)
            if not duration_curve:
                mplt.set_subplot_timeseries_format(sub_pos=n)

        data_table_out = pd.concat(data_tables, axis=0) / unitconversion["divisor"]
        data_table_out = data_table_out.add_suffix(f" ({unitconversion['units']})")

        # legend_order.extend(["Export Limit", "Import Limit"])
        mplt.add_legend(sort_by=legend_order)
        axs[0].set_ylim(top=83)
        # axs[0].set_xlim(right = 2000)
        plt.ylabel(
            f"{var_name} ({unitconversion['units']})",
            color="black",
            rotation="vertical",
            labelpad=30,
        )
        # plt.tight_layout()
        if duration_curve:
            plt.xlabel(
                "Hour of the year", color="black", rotation="horizontal", labelpad=20
            )
        fn_suffix = "_duration_curve" if duration_curve else ""
        fig.savefig(
            self.figure_folder.joinpath(
                f"{self.AGG_BY}_storage", f"{figure_name}{fn_suffix}.svg"
            ),
            dpi=600,
            bbox_inches="tight",
        )
        data_table_out.to_csv(
            self.figure_folder.joinpath(
                f"{self.AGG_BY}_storage", f"{figure_name}{fn_suffix}.csv"
            )
        )

        outputs = DataSavedInModule()
        return outputs
