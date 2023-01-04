# -*- coding: utf-8 -*-
"""Generato total installed capacity plots.

This module plots figures of the total installed capacity of the system.
This
@author: Daniel Levie
"""

import logging
import re
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd

import marmot.utils.mconfig as mconfig
from marmot.plottingmodules.plotutils.plot_data_helper import (
    PlotDataStoreAndProcessor,
    set_facet_col_row_dimensions,
)
from marmot.plottingmodules.plotutils.plot_exceptions import (
    MissingInputData,
    MissingZoneData,
)
from marmot.plottingmodules.plotutils.plot_library import PlotLibrary
from marmot.plottingmodules.plotutils.styles import GeneratorColorDict
from marmot.plottingmodules.plotutils.timeseries_modifiers import (
    set_timestamp_date_range,
)
from marmot.plottingmodules.total_generation import TotalGeneration

logger = logging.getLogger("plotter." + __name__)
plot_data_settings: dict = mconfig.parser("plot_data")


class InstalledCapacity(PlotDataStoreAndProcessor):
    """Installed capacity plots.

    The total_installed_capacity module contains methods that are
    related to the total installed capacity of generators and other devices.

    InstalledCapacity inherits from the PlotDataStoreAndProcessor class to assist
    in creating figures.
    """

    def __init__(
        self,
        Zones: List[str],
        Scenarios: List[str],
        AGG_BY: str,
        ordered_gen: List[str],
        marmot_solutions_folder: Path,
        marmot_color_dict: dict = None,
        ylabels: List[str] = None,
        xlabels: List[str] = None,
        custom_xticklabels: List[str] = None,
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
            marmot_color_dict (dict, optional): Dictionary of colors to use for
                generation technologies.
                Defaults to None.
            ylabels (List[str], optional): y-axis labels for facet plots.
                Defaults to None.
            xlabels (List[str], optional): x-axis labels for facet plots.
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
            self.marmot_color_dict = GeneratorColorDict.set_random_colors(
                self.ordered_gen
            ).color_dict
        else:
            self.marmot_color_dict = marmot_color_dict
        self.ylabels = ylabels
        self.xlabels = xlabels
        self.custom_xticklabels = custom_xticklabels

        self.argument_dict = kwargs

    def total_cap(
        self,
        start_date_range: str = None,
        end_date_range: str = None,
        scenario_groupby: str = "Scenario",
        **_,
    ):
        """Creates a stacked barplot of total installed capacity.

        Each sceanrio will be plotted as a separate bar.

        Args:
            start_date_range (str, optional): Defines a start date at which to represent data from.
                Defaults to None.
            end_date_range (str, optional): Defines a end date at which to represent data to.
                Defaults to None.
            scenario_groupby (str, optional): Specifies whether to group data by Scenario
                or Year-Sceanrio. If grouping by Year-Sceanrio the year will be identified
                from the timestamp and appeneded to the sceanrio name. This is useful when
                plotting data which covers multiple years such as ReEDS.
                Defaults to Scenario.

                .. versionadded:: 0.10.0

        Returns:
            dict: Dictionary containing the created plot and its data table.
        """
        outputs: dict = {}
        # List of properties needed by the plot, properties are a set of tuples and
        # contain 3 parts: required True/False, property name and scenarios required,
        # scenarios must be a list.
        properties = [(True, "generator_Installed_Capacity", self.Scenarios)]

        # Runs get_data to populate mplot_data_dict with all required properties,
        # returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)

        # Checks if all data required by plot is available, if 1 in list required data is missing
        if 1 in check_input_data:
            outputs = MissingInputData()
            return outputs

        for zone_input in self.Zones:

            capacity_chunks = []
            logger.info(f"{self.AGG_BY} = {zone_input}")

            for scenario in self.Scenarios:

                logger.info(f"Scenario = {scenario}")

                Total_Installed_Capacity = self["generator_Installed_Capacity"].get(
                    scenario
                )

                zones_with_cap = Total_Installed_Capacity.index.get_level_values(
                    self.AGG_BY
                ).unique()
                if scenario == "ADS":
                    zone_input_adj = zone_input.split("_WI")["values"]
                else:
                    zone_input_adj = zone_input
                if zone_input_adj in zones_with_cap:
                    Total_Installed_Capacity = Total_Installed_Capacity.xs(
                        zone_input_adj, level=self.AGG_BY
                    )
                else:
                    logger.warning(f"No installed capacity in {zone_input}")
                    outputs[zone_input] = MissingZoneData()
                    continue
                if 'PLEXOS' in scenario:
                    Total_Installed_Capacity.drop('ReEDS_pvb1',level = 'tech',axis = 0,inplace = True)

                Total_Installed_Capacity = self.df_process_gen_inputs(
                    Total_Installed_Capacity
                )

                if pd.notna(start_date_range):
                    Total_Installed_Capacity = set_timestamp_date_range(
                        Total_Installed_Capacity, start_date_range, end_date_range
                    )
                    if Total_Installed_Capacity.empty is True:
                        logger.warning("No Data in selected Date Range")
                        continue
                capacity_chunks.append(
                    self.year_scenario_grouper(
                        Total_Installed_Capacity, scenario, groupby=scenario_groupby
                    ).sum()
                )

            if capacity_chunks:
                Total_Installed_Capacity_Out = (
                    pd.concat(
                        capacity_chunks,
                        axis=0,
                    )
                    .fillna(0)
                    .sort_index(axis=1)
                )
                Total_Installed_Capacity_Out = Total_Installed_Capacity_Out.loc[
                    :, (Total_Installed_Capacity_Out != 0).any(axis=0)
                ]
            # If Total_Installed_Capacity_Out df is empty returns a empty
            # dataframe and does not plot
            else:
                logger.warning(f"No installed capacity in {zone_input}")
                out = MissingZoneData()
                outputs[zone_input] = out
                continue
            unitconversion = self.capacity_energy_unitconversion(
                Total_Installed_Capacity_Out, self.Scenarios, sum_values=True
            )
            Total_Installed_Capacity_Out = (
                Total_Installed_Capacity_Out / unitconversion["divisor"]
            )

            Data_Table_Out = Total_Installed_Capacity_Out
            Data_Table_Out = Data_Table_Out.add_suffix(f" ({unitconversion['units']})")

            mplt = PlotLibrary()
            fig, ax = mplt.get_figure()

            # Set x-tick labels
            if self.custom_xticklabels:
                tick_labels = self.custom_xticklabels
            else:
                tick_labels = Total_Installed_Capacity_Out.index

            mplt.barplot(
                Total_Installed_Capacity_Out,
                color=self.marmot_color_dict,
                stacked=True,
                custom_tick_labels=tick_labels,
            )

            ax.set_ylabel(
                f"Total Installed Capacity ({unitconversion['units']})",
                color="black",
                rotation="vertical",
            )

            mplt.add_legend(reverse_legend=True)
            if plot_data_settings["plot_title_as_region"]:
                mplt.add_main_title(zone_input)

            outputs[zone_input] = {"fig": fig, "data_table": Data_Table_Out}
        return outputs

    def total_cap_diff(
        self,
        start_date_range: str = None,
        end_date_range: str = None,
        scenario_groupby: str = "Scenario",
        **_,
    ):
        """Creates a stacked barplot of total installed capacity relative to a base scenario.

        Barplots show the change in total installed capacity relative to a base scenario.
        The default is to comapre against the first scenario provided in the inputs list.
        Each sceanrio is plotted as a separate bar.

        Args:
            start_date_range (str, optional): Defines a start date at which to represent data from.
                Defaults to None.
            end_date_range (str, optional): Defines a end date at which to represent data to.
                Defaults to None.
            scenario_groupby (str, optional): Specifies whether to group data by Scenario
                or Year-Sceanrio. If grouping by Year-Sceanrio the year will be identified
                from the timestamp and appeneded to the sceanrio name. This is useful when
                plotting data which covers multiple years such as ReEDS.
                Defaults to Scenario.

                .. versionadded:: 0.10.0

        Returns:
            dict: Dictionary containing the created plot and its data table.
        """
        outputs: dict = {}
        # List of properties needed by the plot, properties are a set of tuples and
        # contain 3 parts: required True/False, property name and scenarios required,
        # scenarios must be a list.
        properties = [(True, "generator_Installed_Capacity", self.Scenarios)]

        # Runs get_formatted_data within PlotDataStoreAndProcessor to populate PlotDataStoreAndProcessor dictionary
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)

        # Checks if all data required by plot is available, if 1 in list required data is missing
        if 1 in check_input_data:
            outputs = MissingInputData()
            return outputs

        for zone_input in self.Zones:
            capacity_chunks = []
            logger.info(f"{self.AGG_BY} = {zone_input}")

            for scenario in self.Scenarios:

                logger.info(f"Scenario = {scenario}")

                Total_Installed_Capacity = self["generator_Installed_Capacity"].get(
                    scenario
                )
                zones_with_cap = Total_Installed_Capacity.index.get_level_values(
                    self.AGG_BY
                ).unique()
                if scenario == "ADS":
                    zone_input_adj = zone_input.split("_WI")[0]
                    Total_Installed_Capacity.index = pd.MultiIndex.from_frame(
                        Total_Installed_Capacity.index.to_frame().fillna("All")
                    )  # Fix NaN values from formatter
                    zones_with_cap = Total_Installed_Capacity.index.get_level_values(
                        self.AGG_BY
                    ).unique()
                else:
                    zone_input_adj = zone_input
                if zone_input_adj in zones_with_cap:
                    Total_Installed_Capacity = Total_Installed_Capacity.xs(
                        zone_input_adj, level=self.AGG_BY
                    )
                else:
                    logger.warning(f"No installed capacity in {zone_input}")
                    outputs[zone_input] = MissingZoneData()
                    continue

                fn = self.figure_folder.joinpath(
                    f"{self.AGG_BY}_total_installed_capacity",
                    "Individual_Gen_Cap_{scenario}.csv",
                )

                Total_Installed_Capacity.reset_index().to_csv(fn)

                Total_Installed_Capacity = self.df_process_gen_inputs(
                    Total_Installed_Capacity
                )

                if pd.notna(start_date_range):
                    Total_Installed_Capacity = set_timestamp_date_range(
                        Total_Installed_Capacity, start_date_range, end_date_range
                    )
                    if Total_Installed_Capacity.empty is True:
                        logger.warning("No Data in selected Date Range")
                        continue

                capacity_chunks.append(
                    self.year_scenario_grouper(
                        Total_Installed_Capacity, scenario, groupby=scenario_groupby
                    ).sum()
                )

            if capacity_chunks:
                Total_Installed_Capacity_Out = pd.concat(
                    capacity_chunks, axis=0, sort=False
                ).fillna(0)
            else:
                out = MissingZoneData()
                outputs[zone_input] = out
                continue
            try:
                # Change to a diff on first scenario
                scen_base = Total_Installed_Capacity_Out.index[0]
                Total_Installed_Capacity_Out = (
                    Total_Installed_Capacity_Out
                    - Total_Installed_Capacity_Out.xs(scen_base)
                )
            except KeyError:
                out = MissingZoneData()
                outputs[zone_input] = out
                continue
            Total_Installed_Capacity_Out.drop(
                scen_base, inplace=True
            )  # Drop base entry

            Total_Installed_Capacity_Out = Total_Installed_Capacity_Out.loc[
                :, (Total_Installed_Capacity_Out != 0).any(axis=0)
            ]

            # If Total_Installed_Capacity_Out df is empty returns a empty dataframe and does not plot
            if Total_Installed_Capacity_Out.empty:
                logger.warning(f"No installed capacity in {zone_input}")
                out = MissingZoneData()
                outputs[zone_input] = out
                continue

            unitconversion = self.capacity_energy_unitconversion(
                Total_Installed_Capacity_Out, self.Scenarios, sum_values=True
            )
            Total_Installed_Capacity_Out = (
                Total_Installed_Capacity_Out / unitconversion["divisor"]
            )

            Data_Table_Out = Total_Installed_Capacity_Out
            Data_Table_Out = Data_Table_Out.add_suffix(f" ({unitconversion['units']})")

            mplt = PlotLibrary()
            fig, ax = mplt.get_figure()

            mplt.barplot(
                Total_Installed_Capacity_Out, color=self.marmot_color_dict, stacked=True
            )

            ax.set_ylabel(
                (
                    f"Capacity Change ({unitconversion['units']}) \n "
                    f"relative to {scen_base}"
                ),
                color="black",
                rotation="vertical",
            )

            mplt.add_legend(reverse_legend=True)
            if plot_data_settings["plot_title_as_region"]:
                mplt.add_main_title(zone_input)

            outputs[zone_input] = {"fig": fig, "data_table": Data_Table_Out}
        return outputs

    def total_cap_and_gen_facet(
        self,
        start_date_range: str = None,
        end_date_range: str = None,
        scenario_groupby: str = "Scenario",
        **_,
    ):
        """Creates a facet plot comparing total generation and installed capacity.

        Creates a plot with 2 facet plots, total installed capacity on the left
        and total generation on the right.
        Each facet contains stacked bar plots, each scenario is plotted as a
        separate bar.

        Args:
            start_date_range (str, optional): Defines a start date at which to represent data from.
                Defaults to None.
            end_date_range (str, optional): Defines a end date at which to represent data to.
                Defaults to None.
            scenario_groupby (str, optional): Specifies whether to group data by Scenario
                or Year-Sceanrio. If grouping by Year-Sceanrio the year will be identified
                from the timestamp and appeneded to the sceanrio name. This is useful when
                plotting data which covers multiple years such as ReEDS.
                Defaults to Scenario.

                .. versionadded:: 0.10.0

        Returns:
            dict: Dictionary containing the created plot and its data table.
        """
        # generation figure
        logger.info("Generation data")

        gen_obj = TotalGeneration(
            self.Zones,
            self.Scenarios,
            self.AGG_BY,
            self.ordered_gen,
            self.marmot_solutions_folder,
            marmot_color_dict=self.marmot_color_dict,
            ylabels=self.ylabels,
            xlabels=self.xlabels,
            custom_xticklabels=self.custom_xticklabels,
            **self.argument_dict,
        )

        gen_outputs = gen_obj.total_gen(
            start_date_range, end_date_range, scenario_groupby
        )

        logger.info("Installed capacity data")
        cap_outputs = self.total_cap(start_date_range, end_date_range, scenario_groupby)

        outputs: dict = {}
        for zone_input in self.Zones:

            mplt = PlotLibrary(1, 2, figsize=(5, 4))
            fig, axs = mplt.get_figure()

            plt.subplots_adjust(wspace=0.35, hspace=0.2)

            # left panel: installed capacity
            try:
                Total_Installed_Capacity_Out: pd.DataFrame = cap_outputs[zone_input][
                    "data_table"
                ]
            except TypeError:
                outputs[zone_input] = MissingZoneData()
                continue

            # right panel: annual generation
            try:
                Total_Gen_Results: pd.DataFrame = gen_outputs[zone_input]["data_table"]
            except TypeError:
                outputs[zone_input] = MissingZoneData()
                continue

            # Check units of data
            capacity_units = [
                re.search("GW|MW|TW|kW", unit)
                for unit in Total_Installed_Capacity_Out.columns
            ]
            capacity_units = [unit for unit in capacity_units if unit is not None][
                0
            ].group()

            # Remove any suffixes from column names
            Total_Installed_Capacity_Out.columns = [
                re.sub("[(]|GW|TW|MW|kW|\)", "", i).strip()
                for i in Total_Installed_Capacity_Out.columns
            ]

            if self.custom_xticklabels:
                tick_labels = self.custom_xticklabels
            else:
                tick_labels = Total_Installed_Capacity_Out.index

            mplt.barplot(
                Total_Installed_Capacity_Out,
                color=self.marmot_color_dict,
                stacked=True,
                sub_pos=0,
                custom_tick_labels=tick_labels,
            )

            axs[0].set_ylabel(
                f"Total Installed Capacity ({capacity_units})",
                color="black",
                rotation="vertical",
            )

            # Check units of data
            energy_units = [
                re.search("GWh|MWh|TWh|kWh", unit) for unit in Total_Gen_Results.columns
            ]
            energy_units = [unit for unit in energy_units if unit is not None][
                0
            ].group()

            # Remove any suffixes from column names
            Total_Gen_Results.columns = [
                re.sub("[(]|GWh|TWh|MWh|kWh|\)", "", i).strip()
                for i in Total_Gen_Results.columns
            ]

            if plot_data_settings["include_barplot_load_lines"]:
                extra_plot_data = pd.DataFrame(Total_Gen_Results.loc[:, "Total Load"])
                extra_plot_data["Total Demand"] = Total_Gen_Results.loc[
                    :, f"Total Demand"
                ]
                extra_plot_data["Unserved Energy"] = Total_Gen_Results.loc[
                    :, f"Unserved Energy"
                ]
                if "Load-Unserved_Energy" in Total_Gen_Results.columns:
                    extra_plot_data["Load-Unserved_Energy"] = Total_Gen_Results[
                        "Load-Unserved_Energy"
                    ]
                    Total_Gen_Results.drop("Load-Unserved_Energy", axis=1, inplace=True)

            Total_Generation_Stack_Out = Total_Gen_Results.drop(
                ["Total Load", f"Total Demand", f"Unserved Energy"], axis=1
            )

            if self.custom_xticklabels:
                tick_labels = self.custom_xticklabels
            else:
                tick_labels = Total_Generation_Stack_Out.index

            mplt.barplot(
                Total_Generation_Stack_Out,
                color=self.marmot_color_dict,
                stacked=True,
                sub_pos=1,
                custom_tick_labels=tick_labels,
            )

            axs[1].set_ylabel(
                f"Total Generation ({energy_units})", color="black", rotation="vertical"
            )

            data_tables = []

            if plot_data_settings["include_barplot_load_lines"]:
                mplt.add_barplot_load_lines_and_use(extra_plot_data, sub_pos=1)

            data_tables = pd.DataFrame()  # TODO pass output data back to plot main

            mplt.add_legend(reverse_legend=True, sort_by=self.ordered_gen)
            # add labels to panels
            axs[0].set_title(
                "A.", fontdict={"weight": "bold", "size": 11}, loc="left", pad=4
            )
            axs[1].set_title(
                "B.", fontdict={"weight": "bold", "size": 11}, loc="left", pad=4
            )

            if plot_data_settings["plot_title_as_region"]:
                mplt.add_main_title(zone_input)

            # output figure
            outputs[zone_input] = {"fig": fig, "data_table": data_tables}

        return outputs

    def total_cap_facet(
        self,
        start_date_range: str = None,
        end_date_range: str = None,
        scenario_groupby: str = "Scenario",
        **_,
    ):
        """Creates a stacked barplot of total installed capacity.
        Each sceanrio will be plotted in a separate bar subplot.
        This plot is particularly useful for plotting ReEDS results or
        other models than span multiple years with changing capacity.
        Ensure scenario_groupby is set to 'Year-Sceanrio' to observe this
        effect.

        Args:
            start_date_range (str, optional): Defines a start date at which to represent data from.
                Defaults to None.
            end_date_range (str, optional): Defines a end date at which to represent data to.
                Defaults to None.
            scenario_groupby (str, optional): Specifies whether to group data by Scenario
                or Year-Sceanrio. If grouping by Year-Sceanrio the year will be identified
                from the timestamp and appeneded to the sceanrio name. This is useful when
                plotting data which covers multiple years such as ReEDS.
                Defaults to Scenario.

                .. versionadded:: 0.10.0

        Returns:
            dict: Dictionary containing the created plot and its data table.
        """
        outputs: dict = {}
        # List of properties needed by the plot, properties are a set of tuples and
        # contain 3 parts: required True/False, property name and scenarios required,
        # scenarios must be a list.
        properties = [(True, "generator_Installed_Capacity", self.Scenarios)]

        # Runs get_data to populate mplot_data_dict with all required properties,
        # returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)

        # Checks if all data required by plot is available, if 1 in list required data is missing
        if 1 in check_input_data:
            outputs = MissingInputData()
            return outputs

        for zone_input in self.Zones:

            logger.info(f"Zone = {zone_input}")

            # sets up x, y dimensions of plot
            ncols, nrows = set_facet_col_row_dimensions(
                self.xlabels, self.ylabels, multi_scenario=self.Scenarios
            )
            grid_size = ncols * nrows
            # Used to calculate any excess axis to delete
            plot_number = len(self.Scenarios)
            excess_axs = grid_size - plot_number

            mplt = PlotLibrary(nrows, ncols, sharey=True, squeeze=False, ravel_axs=True)
            fig, axs = mplt.get_figure()

            plt.subplots_adjust(wspace=0.05, hspace=0.5)

            # If creating a facet plot the font is scaled by 9% for each added x dimesion fact plot
            if ncols > 1:
                font_scaling_ratio = 1 + ((ncols - 1) * 0.09)
                plt.rcParams["xtick.labelsize"] *= font_scaling_ratio
                plt.rcParams["ytick.labelsize"] *= font_scaling_ratio
                plt.rcParams["legend.fontsize"] *= font_scaling_ratio
                plt.rcParams["axes.labelsize"] *= font_scaling_ratio
                plt.rcParams["axes.titlesize"] *= font_scaling_ratio

            data_tables = []

            for i, scenario in enumerate(self.Scenarios):
                logger.info(f"Scenario = {scenario}")

                total_installed_capacity = self["generator_Installed_Capacity"].get(
                    scenario
                )

                try:
                    installed_capacity = total_installed_capacity.xs(
                        zone_input, level=self.AGG_BY
                    )
                except KeyError:
                    logger.warning(f"No installed capacity in {zone_input}")
                    outputs[zone_input] = MissingZoneData()
                    continue

                installed_capacity = self.df_process_gen_inputs(installed_capacity)

                if pd.notna(start_date_range):
                    installed_capacity = set_timestamp_date_range(
                        installed_capacity, start_date_range, end_date_range
                    )
                    if installed_capacity.empty is True:
                        logger.warning("No Data in selected Date Range")
                        continue
                installed_capacity_grouped = self.year_scenario_grouper(
                    installed_capacity, scenario, groupby=scenario_groupby
                ).sum()

                # unitconversion based off peak generation hour, only checked once
                if i == 0:
                    unitconversion = self.capacity_energy_unitconversion(
                        installed_capacity_grouped, self.Scenarios, sum_values=True
                    )
                installed_capacity_grouped = (
                    installed_capacity_grouped / unitconversion["divisor"]
                )

                data_tables.append(installed_capacity_grouped)

                # Set x-tick labels
                if self.custom_xticklabels:
                    tick_labels = self.custom_xticklabels
                elif scenario_groupby == "Year-Scenario":
                    tick_labels = [
                        x.split(":")[0] for x in installed_capacity_grouped.index
                    ]
                else:
                    tick_labels = installed_capacity_grouped.index

                mplt.barplot(
                    installed_capacity_grouped,
                    color=self.marmot_color_dict,
                    stacked=True,
                    custom_tick_labels=tick_labels,
                    sub_pos=i,
                )

                if scenario_groupby == "Year-Scenario":
                    axs[i].set_xlabel(scenario)
                else:
                    axs[i].set_xlabel("")

            if not data_tables:
                outputs[zone_input] = MissingZoneData()
                continue

            # Add facet labels
            if self.xlabels or self.ylabels:
                mplt.add_facet_labels(xlabels=self.xlabels, ylabels=self.ylabels)
            # Add legend
            mplt.add_legend(reverse_legend=True, sort_by=self.ordered_gen)
            # Remove extra axes
            mplt.remove_excess_axs(excess_axs, grid_size)
            # Add title
            if plot_data_settings["plot_title_as_region"]:
                mplt.add_main_title(zone_input)

            # Ylabel should change if there are facet labels, leave at 40 for now,
            # works for all values in spacing
            labelpad = 40
            plt.ylabel(
                f"Total Installed Capacity ({unitconversion['units']})",
                color="black",
                rotation="vertical",
                labelpad=labelpad,
            )

            Data_Table_Out = pd.concat(data_tables)
            Data_Table_Out = Data_Table_Out.add_suffix(f" ({unitconversion['units']})")
            outputs[zone_input] = {"fig": fig, "data_table": Data_Table_Out}
        return outputs
