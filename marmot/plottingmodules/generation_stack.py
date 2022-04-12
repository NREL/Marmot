"""Timeseries generation stacked area plots.

This code creates generation stack plots.

@author: Daniel Levie
"""

import logging
import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt

import marmot.utils.mconfig as mconfig

from marmot.plottingmodules.plotutils.plot_library import SetupSubplot, PlotLibrary
from marmot.plottingmodules.plotutils.plot_data_helper import MPlotDataHelper
from marmot.plottingmodules.plotutils.plot_exceptions import (
    MissingInputData,
    UnderDevelopment,
    InputSheetError,
    MissingZoneData,
)

logger = logging.getLogger("plotter." + __name__)
plot_data_settings: dict = mconfig.parser("plot_data")
shift_leapday: bool = mconfig.parser("shift_leapday")
load_legend_names: dict = mconfig.parser("load_legend_names")


class GenerationStack(MPlotDataHelper):
    """Timeseries generation stacked area plots.

    The generation_stack.py contains methods that are
    related to the timeseries generation of generators,
    in a stacked area format.

    GenerationStack inherits from the MPlotDataHelper class to assist
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

        self.curtailment_prop = mconfig.parser("plot_data", "curtailment_property")

    def committed_stack(
        self,
        start_date_range: str = None,
        end_date_range: str = None,
        data_resolution: str = "",
        **_,
    ):
        """Plots the timeseries of committed generation compared to the total available capacity

        The upper line shows the total available cpacity that can be committed
        The area between the lower line and the x-axis plots the total capacity that is
        committed and producing energy.

        Any gap that exists between the upper and lower line is generation that is
        not committed but available to use.

        Data is plotted in a facet plot, each row of the facet plot represents
        separate generation technologies.
        Each bar the facet plot represents separate scenarios.

        Args:
            start_date_range (str, optional): Defines a start date at which to represent data from.
                Defaults to None.
            end_date_range (str, optional): Defines a end date at which to represent data to.
                Defaults to None.
            data_resolution (str, optional): Specifies the data resolution to pull from the formatted
                data and plot.
                Defaults to "", which will pull interval data.

                .. versionadded:: 0.10.0

        Returns:
            dict: dictionary containing the created plot and its data table.
        """
        outputs: dict = {}

        # List of properties needed by the plot, properties are a set of tuples and 
        # contain 3 parts: required True/False, property name and scenarios required, 
        # scenarios must be a list.
        properties = [
            (
                True,
                f"generator_Installed_Capacity{data_resolution}",
                [self.Scenarios[0]],
            ),
            (True, f"generator_Generation{data_resolution}", self.Scenarios),
            (True, f"generator_Units_Generating{data_resolution}", self.Scenarios),
            (True, f"generator_Available_Capacity{data_resolution}", self.Scenarios),
        ]

        # Runs get_formatted_data within MPlotDataHelper to populate MPlotDataHelper dictionary
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)

        # Checks if all data required by plot is available, if 1 in list required data is missing
        if 1 in check_input_data:
            return MissingInputData()

        for zone_input in self.Zones:
            logger.info(f"Zone = {str(zone_input)}")

            # Get technology list.
            gens = self[f"generator_Installed_Capacity{data_resolution}"].get(
                self.Scenarios[0]
            )
            try:
                gens = gens.xs(zone_input, level=self.AGG_BY)
            except KeyError:
                logger.warning(f"No Generation in: {zone_input}")
                outputs[zone_input] = MissingZoneData()
                continue

            gens = self.df_process_gen_inputs(gens)
            tech_list = list(gens.columns)
            tech_list_sort = [
                tech_type
                for tech_type in self.ordered_gen
                if tech_type in tech_list and tech_type in self.gen_categories.thermal
            ]

            if not tech_list_sort:
                logger.info(f"No Thermal Generation in: {zone_input}")
                outputs[zone_input] = MissingZoneData()
                continue

            ncols = len(self.Scenarios)
            nrows = len(tech_list_sort)

            mplt = SetupSubplot(nrows, ncols, sharex=True, sharey="row", squeeze=False)
            fig, axs = mplt.get_figure()
            plt.subplots_adjust(wspace=0.1, hspace=0.2)

            for i, scenario in enumerate(self.Scenarios):
                logger.info(f"Scenario = {scenario}")

                units_gen: pd.DataFrame = self[
                    f"generator_Units_Generating{data_resolution}"
                ].get(scenario)
                avail_cap: pd.DataFrame = self[
                    f"generator_Available_Capacity{data_resolution}"
                ].get(scenario)

                units_gen = units_gen.xs(zone_input, level=self.AGG_BY)
                units_gen = self.df_process_gen_inputs(units_gen)
                avail_cap = avail_cap.xs(zone_input, level=self.AGG_BY)
                avail_cap = self.df_process_gen_inputs(avail_cap)

                # Calculate  committed cap (for thermal only).
                thermal_commit_cap = units_gen * avail_cap
                # Drop all zero columns
                thermal_commit_cap = thermal_commit_cap.loc[
                    :, (thermal_commit_cap != 0).any(axis=0)
                ]

                # unitconversion based off peak generation hour, only checked once
                if i == 0:
                    unitconversion = self.capacity_energy_unitconversion(
                        thermal_commit_cap
                    )
                thermal_commit_cap = thermal_commit_cap / unitconversion["divisor"]

                # Process generation.
                gen: pd.DataFrame = self[f"generator_Generation{data_resolution}"].get(
                    scenario
                )
                gen = gen.xs(zone_input, level=self.AGG_BY)
                gen = self.df_process_gen_inputs(gen)
                gen = gen.loc[:, (gen != 0).any(axis=0)]
                gen = gen / unitconversion["divisor"]

                avail_cap = avail_cap.loc[:, (avail_cap != 0).any(axis=0)]
                avail_cap = avail_cap / unitconversion["divisor"]

                if pd.notna(start_date_range):
                    thermal_commit_cap, gen, avail_cap = self.set_timestamp_date_range(
                        [thermal_commit_cap, gen, avail_cap],
                        start_date_range,
                        end_date_range,
                    )
                    if thermal_commit_cap.empty is True:
                        logger.warning("No Generation in selected Date Range")
                        continue

                for j, tech in enumerate(tech_list_sort):
                    if tech not in gen.columns:
                        gen_one_tech = pd.Series(0, index=gen.index)
                        commit_cap = pd.Series(0, index=gen.index)
                    elif tech in self.gen_categories.thermal:
                        gen_one_tech = gen[tech]
                        commit_cap = thermal_commit_cap[tech]
                    # For all other techs
                    else:
                        gen_one_tech = gen[tech]
                        commit_cap = avail_cap[tech]

                    axs[j, i].plot(
                        gen_one_tech, alpha=0, color=self.PLEXOS_color_dict[tech]
                    )[0]
                    axs[j, i].fill_between(
                        gen_one_tech.index,
                        gen_one_tech,
                        0,
                        color=self.PLEXOS_color_dict[tech],
                        alpha=0.5,
                    )
                    if tech != "Hydro":
                        axs[j, i].plot(commit_cap, color=self.PLEXOS_color_dict[tech])

                    mplt.set_yaxis_major_tick_format(sub_pos=(j, i))
                    axs[j, i].margins(x=0.01)
                    mplt.set_subplot_timeseries_format(sub_pos=(j, i))

            mplt.add_facet_labels(
                xlabels_bottom=False, xlabels=self.Scenarios, ylabels=tech_list_sort
            )

            if plot_data_settings["plot_title_as_region"]:
                mplt.add_main_title(zone_input)
            plt.ylabel(
                f"Generation or Committed Capacity ({unitconversion['units']})",
                color="black",
                rotation="vertical",
                labelpad=60,
            )

            data_table = pd.DataFrame()  # TODO: write actual data out
            outputs[zone_input] = {"fig": fig, "data_table": data_table}
        return outputs

    def gen_stack(
        self,
        prop: str = None,
        start: float = None,
        end: float = None,
        timezone: str = "",
        start_date_range: str = None,
        end_date_range: str = None,
        data_resolution: str = "",
        **_,
    ):
        """Creates a timeseries stacked area plot of generation by technology.

        The stack order of technologies is determined by the ordered_gen_categories.csv

        If multiple scenarios are passed they will be plotted in a facet plot.
        The plot can be further customized by passing specific values to the
        prop argument.

        Args:
            prop (str, optional): Special argument used to adjust specific
                plot settings. Controlled through the plot_select.csv.
                Opinions available are:

                - Peak Demand
                - Min Net Load
                - Peak RE
                - Peak Unserved Energy
                - Peak Curtailment

                Defaults to None.
            start (float, optional): Used in conjunction with the prop argument.
                Will define the number of days to plot before a certain event in
                a timeseries plot, e.g Peak Demand.
                Defaults to None.
            end (float, optional): Used in conjunction with the prop argument.
                Will define the number of days to plot after a certain event in
                a timeseries plot, e.g Peak Demand.
                Defaults to None.
            timezone (str, optional): The timezone to display on the x-axes.
                Defaults to "".
            start_date_range (str, optional): Defines a start date at which to represent data from.
                Defaults to None.
            end_date_range (str, optional): Defines a end date at which to represent data to.
                Defaults to None.
            data_resolution (str, optional): Specifies the data resolution to pull from the formatted
                data and plot.
                Defaults to "", which will pull interval data.

                .. versionadded:: 0.10.0

        Returns:
            dict: dictionary containing the created plot and its data table.

        """
        if self.AGG_BY == "zone":
            agg = "zone"
        else:
            agg = "region"

        # Main loop for gen_stack
        outputs: dict = {}

        # List of properties needed by the plot, properties are a set of tuples and 
        # contain 3 parts: required True/False, property name and scenarios required, 
        # scenarios must be a list.
        properties = [
            (True, f"generator_Generation{data_resolution}", self.Scenarios),
            (
                False,
                f"generator_{self.curtailment_prop}{data_resolution}",
                self.Scenarios,
            ),
            (False, f"{agg}_Load{data_resolution}", self.Scenarios),
            (False, f"{agg}_Demand{data_resolution}", self.Scenarios),
            (False, f"{agg}_Unserved_Energy{data_resolution}", self.Scenarios),
        ]

        # Runs get_formatted_data within MPlotDataHelper to populate MPlotDataHelper dictionary
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)

        # Checks if all data required by plot is available, if 1 in list required data is missing
        if 1 in check_input_data:
            outputs = MissingInputData()
            return outputs

        for zone_input in self.Zones:
            logger.info(f"Zone = {zone_input}")

            # sets up x, y dimensions of plot
            ncols, nrows = self.set_facet_col_row_dimensions(
                multi_scenario=self.Scenarios
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

                try:
                    stacked_gen_df: pd.DataFrame = self[
                        f"generator_Generation{data_resolution}"
                    ].get(scenario)
                    if shift_leapday:
                        stacked_gen_df = self.adjust_for_leapday(stacked_gen_df)
                    stacked_gen_df = stacked_gen_df.xs(zone_input, level=self.AGG_BY)
                except KeyError:
                    logger.warning(f"No generation in {zone_input}")
                    outputs[zone_input] = MissingZoneData()
                    continue
                stacked_gen_df = self.df_process_gen_inputs(stacked_gen_df)

                # Insert Curtailment into gen stack if it exists in database
                stacked_curt_df: pd.DataFrame = self[
                    f"generator_{self.curtailment_prop}{data_resolution}"
                ].get(scenario)
                curtailment_name = self.gen_names_dict.get("Curtailment", "Curtailment")
                if not stacked_curt_df.empty:
                    if shift_leapday:
                        stacked_curt_df = self.adjust_for_leapday(stacked_curt_df)
                    if (
                        zone_input
                        in stacked_curt_df.index.get_level_values(self.AGG_BY).unique()
                    ):
                        stacked_curt_df = stacked_curt_df.xs(
                            zone_input, level=self.AGG_BY
                        )
                        stacked_curt_df = self.df_process_gen_inputs(stacked_curt_df)
                        # If using Marmot's curtailment property
                        if self.curtailment_prop == "Curtailment":
                            stacked_curt_df = self.assign_curtailment_techs(
                                stacked_curt_df
                            )
                        stacked_curt_df = stacked_curt_df.sum(axis=1)
                        stacked_curt_df[
                            stacked_curt_df < 0.05
                        ] = 0  # Remove values less than 0.05 MW
                        # Insert curtailment into
                        stacked_gen_df.insert(
                            len(stacked_gen_df.columns),
                            column=curtailment_name,
                            value=stacked_curt_df,
                        )
                        stacked_gen_df = stacked_gen_df.fillna(0)
                        # Calculates Net Load by removing variable gen + curtailment
                        vre_gen_cat = self.gen_categories.vre + [curtailment_name]
                    else:
                        vre_gen_cat = self.gen_categories.vre
                else:
                    vre_gen_cat = self.gen_categories.vre

                if pd.notna(start_date_range):
                    stacked_gen_df = self.set_timestamp_date_range(
                        stacked_gen_df, start_date_range, end_date_range
                    )
                    if stacked_gen_df.empty is True:
                        logger.warning("No Generation in selected Date Range")
                        continue

                # Adjust list of values to drop depending on if it exists in stacked_gen_df df
                vre_gen_cat = [
                    name for name in vre_gen_cat if name in stacked_gen_df.columns
                ]
                net_load = stacked_gen_df.drop(labels=vre_gen_cat, axis=1)
                net_load = net_load.sum(axis=1)
                net_load = net_load.rename("Net Load")

                # Extra optional properties
                extra_data_frames = []
                extra_property_names = [
                    f"{agg}_Load{data_resolution}",
                    f"{agg}_Demand{data_resolution}",
                    f"{agg}_Unserved_Energy{data_resolution}",
                ]
                # Get and process extra properties
                for ext_prop in extra_property_names:
                    df: pd.DataFrame = self[ext_prop].get(scenario)
                    if (
                        df.empty
                        or not plot_data_settings["include_stackplot_load_lines"]
                    ):
                        date_index = pd.date_range(
                            start="2010-01-01", periods=1, freq="H", name="timestamp"
                        )
                        df = pd.DataFrame(data=[0], index=date_index)
                    else:
                        df = df.xs(zone_input, level=self.AGG_BY)
                        df = df.groupby(["timestamp"]).sum()
                    df = df.rename(columns={0: ext_prop})
                    extra_data_frames.append(df)

                extra_plot_data = pd.concat(extra_data_frames, axis=1).fillna(0)

                if (
                    extra_plot_data[f"{agg}_Unserved_Energy{data_resolution}"] == 0
                ).all() == False:
                    extra_plot_data["Load-Unserved_Energy"] = (
                        extra_plot_data[f"{agg}_Demand{data_resolution}"]
                        - extra_plot_data[f"{agg}_Unserved_Energy{data_resolution}"]
                    )

                extra_plot_data = extra_plot_data.rename(
                    columns={
                        f"{agg}_Load{data_resolution}": "Total Load",
                        f"{agg}_Unserved_Energy{data_resolution}": "Unserved Energy",
                        f"{agg}_Demand{data_resolution}": "Total Demand",
                    }
                )

                # Adjust extra data to generator date range
                extra_plot_data = extra_plot_data.loc[
                    stacked_gen_df.index.min() : stacked_gen_df.index.max()
                ]
                # append net_load load to extra data,
                # fill na with 0 in the event of misaligned timeseries
                extra_plot_data = pd.concat([extra_plot_data, net_load], axis=1).fillna(
                    0
                )

                # unitconversion based off peak generation hour, only checked once
                if i == 0:
                    unitconversion = self.capacity_energy_unitconversion(
                        stacked_gen_df, sum_values=True
                    )
                # Convert units
                stacked_gen_df = stacked_gen_df / unitconversion["divisor"]
                extra_plot_data = extra_plot_data / unitconversion["divisor"]

                # Adds property annotation and
                if pd.notna(prop):
                    x_time_value = mplt.add_property_annotation(
                        pd.concat([stacked_gen_df, extra_plot_data], axis=1),
                        prop,
                        sub_pos=i,
                        curtailment_name=curtailment_name,
                        energy_unit=unitconversion["units"],
                        re_gen_cat=self.gen_categories.re,
                        gen_cols=stacked_gen_df.columns,
                    )

                    if x_time_value is not None and len(stacked_gen_df) > 1:
                        # if timestamps are larger than hours time_delta will
                        # be the length of the interval in days, else time_delta == 1 day
                        timestamps = stacked_gen_df.index.unique()
                        time_delta = max(
                            1, (timestamps[1] - timestamps[0]) / np.timedelta64(1, "D")
                        )
                        end_date = x_time_value + dt.timedelta(days=end * time_delta)
                        start_date = x_time_value - dt.timedelta(
                            days=start * time_delta
                        )
                        stacked_gen_df = stacked_gen_df.loc[start_date:end_date]
                        extra_plot_data = extra_plot_data.loc[start_date:end_date]

                if mconfig.parser("plot_data", "include_stackplot_net_imports"):
                    stacked_gen_df = self.include_net_imports(
                        stacked_gen_df,
                        extra_plot_data["Total Load"],
                        extra_plot_data["Unserved Energy"],
                    )

                # Remove any all 0 columns
                stacked_gen_df = stacked_gen_df.loc[
                    :, (stacked_gen_df != 0).any(axis=0)
                ]

                # Data table of values to return to main program
                single_scen_out = pd.concat(
                    [extra_plot_data, stacked_gen_df], axis=1, sort=False
                )
                scenario_names = pd.Series(
                    [scenario] * len(single_scen_out), name="Scenario"
                )
                single_scen_out = single_scen_out.add_suffix(
                    f" ({unitconversion['units']})"
                )
                single_scen_out = single_scen_out.set_index(
                    [scenario_names], append=True
                )
                single_scen_out = single_scen_out.loc[
                    :, (single_scen_out != 0).any(axis=0)
                ]
                data_tables.append(single_scen_out)

                mplt.stackplot(
                    stacked_gen_df,
                    self.PLEXOS_color_dict,
                    labels=stacked_gen_df.columns,
                    sub_pos=i,
                )
                mplt.set_subplot_timeseries_format(sub_pos=i)

                if (extra_plot_data["Unserved Energy"] == 0).all() == False:
                    axs[i].fill_between(
                        extra_plot_data["Total Demand"].index,
                        extra_plot_data["Total Demand"],
                        extra_plot_data["Load-Unserved_Energy"],
                        label="Unserved Energy",
                        facecolor="#DD0200",
                        alpha=0.5,
                    )

                if plot_data_settings["include_stackplot_load_lines"]:
                    if (
                        plot_data_settings[
                            "include_timeseries_load_storage_charging_line"
                        ]
                        and extra_plot_data["Total Load"].sum()
                        > extra_plot_data["Total Demand"].sum()
                    ):
                        axs[i].plot(
                            extra_plot_data["Total Load"],
                            color="black",
                            linestyle="--",
                            label=load_legend_names["load"],
                        )
                        axs[i].plot(
                            extra_plot_data["Total Demand"],
                            color="black",
                            label=load_legend_names["demand"],
                        )
                    elif extra_plot_data["Total Demand"].sum() > 0:
                        axs[i].plot(
                            extra_plot_data["Total Demand"],
                            color="black",
                            label=load_legend_names["demand"],
                        )

            if not data_tables:
                outputs[zone_input] = MissingZoneData()
                continue

            # Add facet labels
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
            if data_resolution == "_Annual":
                plt.ylabel(
                    f"Annual Generation ({unitconversion['units']}h)",
                    color="black",
                    rotation="vertical",
                    labelpad=labelpad,
                )
            else:
                plt.ylabel(
                    f"Generation ({unitconversion['units']})",
                    color="black",
                    rotation="vertical",
                    labelpad=labelpad,
                )

            Data_Table_Out = pd.concat(data_tables)
            outputs[zone_input] = {"fig": fig, "data_table": Data_Table_Out}
        return outputs

    def gen_diff(
        self,
        timezone: str = "",
        start_date_range: str = None,
        end_date_range: str = None,
        data_resolution: str = "",
        **_,
    ):
        """Plots the difference in generation between two scenarios.

        A line plot is created for each technology representing the difference
        between the scenarios.

        Args:
            timezone (str, optional): The timezone to display on the x-axes.
                Defaults to "".
            start_date_range (str, optional): Defines a start date at which to represent data from.
                Defaults to None.
            end_date_range (str, optional): Defines a end date at which to represent data to.
                Defaults to None.
            data_resolution (str, optional): Specifies the data resolution to pull from the formatted
                data and plot.
                Defaults to "", which will pull interval data.

                .. versionadded:: 0.10.0

        Returns:
            dict: dictionary containing the created plot and its data table.
        """
        outputs: dict = {}

        # List of properties needed by the plot, properties are a set of tuples and 
        # contain 3 parts: required True/False, property name and scenarios required, 
        # scenarios must be a list.
        properties = [(True, f"generator_Generation{data_resolution}", self.Scenarios)]

        # Runs get_formatted_data within MPlotDataHelper to populate MPlotDataHelper dictionary
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)

        if 1 in check_input_data:
            outputs = MissingInputData()
            return outputs

        for zone_input in self.Zones:
            logger.info(f"Zone = {zone_input}")
            # Create Dictionary to hold Datframes for each scenario

            Total_Gen_Stack_1: pd.DataFrame = self[
                f"generator_Generation{data_resolution}"
            ].get(self.Scenario_Diff[0])
            if Total_Gen_Stack_1 is None:
                logger.warning(
                    f"Scenario_Diff '{self.Scenario_Diff[0]}'' is not in data. "
                    "Ensure User Input Sheet is set up correctly!"
                )
                outputs = InputSheetError()
                return outputs

            if (
                zone_input
                not in Total_Gen_Stack_1.index.get_level_values(self.AGG_BY).unique()
            ):
                outputs[zone_input] = MissingZoneData()
                continue

            Total_Gen_Stack_1 = Total_Gen_Stack_1.xs(zone_input, level=self.AGG_BY)
            Total_Gen_Stack_1 = self.df_process_gen_inputs(Total_Gen_Stack_1)
            # Adds in all possible columns from ordered gen to ensure the two dataframes have same column names
            Total_Gen_Stack_1 = pd.DataFrame(
                Total_Gen_Stack_1, columns=self.ordered_gen
            ).fillna(0)

            Total_Gen_Stack_2: pd.DataFrame = self[
                f"generator_Generation{data_resolution}"
            ].get(self.Scenario_Diff[1])
            if Total_Gen_Stack_2 is None:
                logger.warning(
                    f"Scenario_Diff '{self.Scenario_Diff[1]}' is not in data. "
                    "Ensure User Input Sheet is set up correctly!"
                )
                outputs = InputSheetError()
                return outputs

            Total_Gen_Stack_2 = Total_Gen_Stack_2.xs(zone_input, level=self.AGG_BY)
            Total_Gen_Stack_2 = self.df_process_gen_inputs(Total_Gen_Stack_2)
            # Adds in all possible columns from ordered gen to ensure the two dataframes have same column names
            Total_Gen_Stack_2 = pd.DataFrame(
                Total_Gen_Stack_2, columns=self.ordered_gen
            ).fillna(0)

            logger.info(f"Scenario 1 = {self.Scenario_Diff[0]}")
            logger.info(f"Scenario 2 = {self.Scenario_Diff[1]}")
            Gen_Stack_Out = Total_Gen_Stack_1 - Total_Gen_Stack_2

            if pd.notna(start_date_range):
                Gen_Stack_Out = self.set_timestamp_date_range(
                    Gen_Stack_Out, start_date_range, end_date_range
                )
                if Gen_Stack_Out.empty is True:
                    logger.warning("No Generation in selected Date Range")
                    continue

            # Removes columns that only equal 0
            Gen_Stack_Out.dropna(inplace=True)
            Gen_Stack_Out = Gen_Stack_Out.loc[:, (Gen_Stack_Out != 0).any(axis=0)]

            if Gen_Stack_Out.empty == True:
                outputs[zone_input] = MissingZoneData()
                continue

            # Reverses order of columns
            Gen_Stack_Out = Gen_Stack_Out.iloc[:, ::-1]

            unitconversion = self.capacity_energy_unitconversion(Gen_Stack_Out)
            Gen_Stack_Out = Gen_Stack_Out / unitconversion["divisor"]

            # Data table of values to return to main program
            Data_Table_Out = Gen_Stack_Out.add_suffix(f" ({unitconversion['units']})")

            mplt = SetupSubplot()
            fig, ax = mplt.get_figure()

            for column in Gen_Stack_Out:
                ax.plot(
                    Gen_Stack_Out[column],
                    linewidth=3,
                    color=self.PLEXOS_color_dict[column],
                    label=column,
                )

            mplt.add_main_title(f"{self.Scenario_Diff[0]} vs. {self.Scenario_Diff[1]}")
            ax.set_ylabel(
                f"Generation Difference ({unitconversion['units']})",
                color="black",
                rotation="vertical",
            )
            ax.set_xlabel(timezone, color="black", rotation="horizontal")
            mplt.set_yaxis_major_tick_format()
            ax.margins(x=0.01)
            mplt.set_subplot_timeseries_format()
            mplt.add_legend()

            outputs[zone_input] = {"fig": fig, "data_table": Data_Table_Out}
        return outputs
