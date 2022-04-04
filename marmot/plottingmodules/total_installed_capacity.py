# -*- coding: utf-8 -*-
"""Generato total installed capacity plots.

This module plots figures of the total installed capacity of the system.
This
@author: Daniel Levie
"""

import re
import logging
import pandas as pd
import matplotlib.pyplot as plt

import marmot.utils.mconfig as mconfig
from marmot.plottingmodules.total_generation import TotalGeneration
from marmot.plottingmodules.plotutils.plot_library import PlotLibrary
from marmot.plottingmodules.plotutils.plot_data_helper import MPlotDataHelper
from marmot.plottingmodules.plotutils.plot_exceptions import (
    MissingInputData,
    MissingZoneData,
)

logger = logging.getLogger("plotter." + __name__)
plot_data_settings: dict = mconfig.parser("plot_data")
load_legend_names: dict = mconfig.parser("load_legend_names")


class InstalledCapacity(MPlotDataHelper):
    """Installed capacity plots.

    The total_installed_capacity module contains methods that are
    related to the total installed capacity of generators and other devices.

    InstalledCapacity inherits from the MPlotDataHelper class to assist
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
        
        self.argument_list = args
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
                    zone_input_adj = zone_input.split("_WI")[0]
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

                Total_Installed_Capacity = self.df_process_gen_inputs(
                    Total_Installed_Capacity
                )

                if pd.notna(start_date_range):
                    Total_Installed_Capacity = self.set_timestamp_date_range(
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
                    capacity_chunks, axis=0, sort=True
                ).fillna(0)
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
                Total_Installed_Capacity_Out, sum_values=True
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
                color=self.PLEXOS_color_dict,
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

        # Runs get_formatted_data within MPlotDataHelper to populate MPlotDataHelper dictionary
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
                    Total_Installed_Capacity = self.set_timestamp_date_range(
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
                Total_Installed_Capacity_Out, sum_values=True
            )
            Total_Installed_Capacity_Out = (
                Total_Installed_Capacity_Out / unitconversion["divisor"]
            )

            Data_Table_Out = Total_Installed_Capacity_Out
            Data_Table_Out = Data_Table_Out.add_suffix(f" ({unitconversion['units']})")

            mplt = PlotLibrary()
            fig, ax = mplt.get_figure()

            mplt.barplot(
                Total_Installed_Capacity_Out, color=self.PLEXOS_color_dict, stacked=True
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
        gen_obj = TotalGeneration(*self.argument_list, **self.argument_dict)
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
                color=self.PLEXOS_color_dict,
                stacked=True,
                sub_pos=0,
                custom_tick_labels=tick_labels,
            )

            axs[0].set_ylabel(
                f"Total Installed Capacity ({capacity_units})",
                color="black",
                rotation="vertical",
            )

            # right panel: annual generation
            Total_Gen_Results: pd.DataFrame = gen_outputs[zone_input]["data_table"]

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
                    extra_plot_data["Load-Unserved_Energy"] = (
                            Total_Gen_Results["Load-Unserved_Energy"]
                    )
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
                color=self.PLEXOS_color_dict,
                stacked=True,
                sub_pos=1,
                custom_tick_labels=tick_labels,
            )

            axs[1].set_ylabel(
                f"Total Generation ({energy_units})", color="black", rotation="vertical"
            )

            data_tables = []

            if plot_data_settings["include_barplot_load_lines"]:
                for n, scenario in enumerate(Total_Generation_Stack_Out.index.unique()):
                    x = [
                        axs[1].patches[n].get_x(),
                        axs[1].patches[n].get_x() + axs[1].patches[n].get_width(),
                    ]

                    height1 = [
                        float(extra_plot_data.loc[scenario, "Total Load"].sum())
                    ] * 2

                    if (
                        plot_data_settings["include_barplot_load_storage_charging_line"]
                        and extra_plot_data.loc[scenario, "Total Load"].sum()
                        > extra_plot_data.loc[scenario, "Total Demand"].sum()
                    ):
                        axs[1].plot(
                            x,
                            height1,
                            c="black",
                            linewidth=1.5,
                            linestyle="--",
                            label=load_legend_names["load"],
                        )
                        height2 = [
                            float(extra_plot_data.loc[scenario, "Total Demand"])
                        ] * 2
                        axs[1].plot(
                            x,
                            height2,
                            c="black",
                            linewidth=1.5,
                            label=load_legend_names["demand"],
                        )
                    elif extra_plot_data.loc[scenario, "Total Demand"].sum() > 0:
                        axs[1].plot(
                            x,
                            height1,
                            c="black",
                            linewidth=1.5,
                            label=load_legend_names["demand"],
                        )

                    if extra_plot_data.loc[scenario, "Unserved Energy"] > 0:
                        height3 = [
                            float(extra_plot_data.loc[scenario, "Load-Unserved_Energy"])
                        ] * 2
                        axs[1].fill_between(
                            x,
                            height3,
                            height1,
                            facecolor="#DD0200",
                            alpha=0.5,
                            label="Unserved Energy",
                        )

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
