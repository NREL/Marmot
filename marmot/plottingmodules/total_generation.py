# -*- coding: utf-8 -*-
"""Total generation plots.

This module plots figures of total generation for a year, month etc.

@author: Daniel Levie 
"""

import logging
import pandas as pd
import matplotlib.pyplot as plt

import marmot.utils.mconfig as mconfig

from marmot.plottingmodules.plotutils.plot_library import PlotLibrary, SetupSubplot
from marmot.plottingmodules.plotutils.plot_data_helper import MPlotDataHelper
from marmot.plottingmodules.plotutils.plot_exceptions import (
    MissingInputData,
    MissingZoneData,
)

logger = logging.getLogger("plotter." + __name__)
shift_leapday: bool = mconfig.parser("shift_leapday")
plot_data_settings: dict = mconfig.parser("plot_data")
load_legend_names: dict = mconfig.parser("load_legend_names")


class TotalGeneration(MPlotDataHelper):
    """Total generation plots.

    The total_genertion.py module contains methods that are
    display the total amount of generation over a given time period.

    TotalGeneration inherits from the MPlotDataHelper class to assist
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

    def total_gen(
        self,
        start_date_range: str = None,
        end_date_range: str = None,
        scenario_groupby: str = "Scenario",
        **_,
    ):
        """Creates a stacked bar plot of total generation by technology type.

        A separate bar is created for each scenario.

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
        # Create Dictionary to hold Datframes for each scenario
        outputs: dict = {}

        if self.AGG_BY == "zone":
            agg = "zone"
        else:
            agg = "region"
        # List of properties needed by the plot, properties are a set of tuples and 
        # contain 3 parts: required True/False, property name and scenarios required, 
        # scenarios must be a list.
        properties = [
            (True, "generator_Generation", self.Scenarios),
            (False, f"generator_{self.curtailment_prop}", self.Scenarios),
            (False, f"{agg}_Load", self.Scenarios),
            (False, f"{agg}_Demand", self.Scenarios),
            (False, f"{agg}_Unserved_Energy", self.Scenarios),
            (False,"batterie_Generation", self.Scenarios)
        ]

        # Runs get_formatted_data within MPlotDataHelper to populate MPlotDataHelper dictionary
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)

        if 1 in check_input_data:
            outputs = MissingInputData()
            return outputs

        for zone_input in self.Zones:

            # Will hold retrieved data for each scenario
            gen_chunks = []
            extra_data_chunks = []

            logger.info(f"Zone = {zone_input}")

            for scenario in self.Scenarios:

                logger.info(f"Scenario = {scenario}")
                Total_Gen_Stack: pd.DataFrame = self["generator_Generation"].get(
                    scenario
                )

                # Check if zone has generation, if not skips
                try:
                    Total_Gen_Stack = Total_Gen_Stack.xs(zone_input, level=self.AGG_BY)
                except KeyError:
                    logger.warning(f"No generation in: {zone_input}")
                    continue
                Total_Gen_Stack = self.df_process_gen_inputs(Total_Gen_Stack)

                if pd.notna(start_date_range):
                    Total_Gen_Stack = self.set_timestamp_date_range(
                        Total_Gen_Stack, start_date_range, end_date_range
                    )
                    if Total_Gen_Stack.empty is True:
                        logger.warning("No Generation in selected Date Range")
                        continue

                # Calculates interval step to correct for MWh of generation
                interval_count = self.get_sub_hour_interval_count(Total_Gen_Stack)

                curtailment_name = self.gen_names_dict.get("Curtailment", "Curtailment")
                # Insert Curtailment into gen stack if it exists in database
                Stacked_Curt = self[f"generator_{self.curtailment_prop}"].get(scenario)
                if not Stacked_Curt.empty:
                    if (
                        zone_input
                        in Stacked_Curt.index.get_level_values(self.AGG_BY).unique()
                    ):
                        Stacked_Curt = Stacked_Curt.xs(zone_input, level=self.AGG_BY)
                        Stacked_Curt = self.df_process_gen_inputs(Stacked_Curt)
                        # If using Marmot's curtailment property
                        if self.curtailment_prop == "Curtailment":
                            Stacked_Curt = self.assign_curtailment_techs(Stacked_Curt)
                        Stacked_Curt = Stacked_Curt.sum(axis=1)
                        Total_Gen_Stack.insert(
                            len(Total_Gen_Stack.columns),
                            column=curtailment_name,
                            value=Stacked_Curt,
                        )  # Insert curtailment into
                        Total_Gen_Stack = Total_Gen_Stack.loc[
                            :, (Total_Gen_Stack != 0).any(axis=0)
                        ]

                #Insert battery generation.
                stacked_bat_gen : pd.DataFrame = self[
                    f"batterie_Generation"
                ].get(scenario)
                battery_discharge_name = self.gen_names_dict.get("battery", "Storage")
                if stacked_bat_gen.empty is True:
                    logger.info("No Battery generation in selected Date Range")
                else:
                    if shift_leapday:
                        stacked_bat_gen = self.adjust_for_leapday(stacked_bat_gen)

                    stacked_bat_gen = stacked_bat_gen.xs(
                        zone_input, level=self.AGG_BY
                    )

                    stacked_bat_gen.index = stacked_bat_gen.index.droplevel(['category','units'])
                    Total_Gen_Stack.insert(
                        len(Total_Gen_Stack.columns),
                        column=battery_discharge_name,
                        value=stacked_bat_gen,
                    )
                    
                Total_Gen_Stack = Total_Gen_Stack / interval_count

                # Extra optional properties
                extra_data_frames = []
                extra_property_names = [
                    f"{agg}_Load",
                    f"{agg}_Demand",
                    f"{agg}_Unserved_Energy",
                ]
                # Get and process extra properties
                for ext_prop in extra_property_names:
                    df: pd.DataFrame = self[ext_prop].get(scenario)
                    if df.empty or not plot_data_settings["include_barplot_load_lines"]:
                        date_index = pd.date_range(
                            start="2010-01-01", periods=1, freq="H", name="timestamp"
                        )
                        df = pd.DataFrame(data=[0], index=date_index, columns=["values"])
                    else:
                        df = df.xs(zone_input, level=self.AGG_BY)
                        df = df.groupby(["timestamp"]).sum()
                    df = df.rename(columns={"values" : ext_prop})
                    extra_data_frames.append(df)

                extra_plot_data = pd.concat(extra_data_frames, axis=1).fillna(0)

                if (extra_plot_data[f"{agg}_Unserved_Energy"] == 0).all() == False:
                    extra_plot_data["Load-Unserved_Energy"] = (
                        extra_plot_data[f"{agg}_Demand"]
                        - extra_plot_data[f"{agg}_Unserved_Energy"]
                    )

                extra_plot_data = extra_plot_data.rename(
                    columns={
                        f"{agg}_Load": "Total Load",
                        f"{agg}_Unserved_Energy": "Unserved Energy",
                        f"{agg}_Demand": "Total Demand",
                    }
                )
                extra_plot_data = extra_plot_data / interval_count               
                extra_plot_data = extra_plot_data.loc[
                    Total_Gen_Stack.index.min() : Total_Gen_Stack.index.max()
                ]
                gen_chunks.append(
                    self.year_scenario_grouper(
                        Total_Gen_Stack, scenario, groupby=scenario_groupby
                    ).sum()
                )
                extra_data_chunks.append(
                    self.year_scenario_grouper(
                        extra_plot_data, scenario, groupby=scenario_groupby
                    ).sum()
                )

            if not gen_chunks:
                outputs[zone_input] = MissingZoneData()
                continue

            total_generation_stack_out = pd.concat(
                gen_chunks, axis=0
            ).fillna(0).sort_index(axis=1)
            extra_data_out = pd.concat(extra_data_chunks, axis=0, sort=False)

            # Add Net Imports if desired
            if plot_data_settings["include_barplot_net_imports"]:
                total_generation_stack_out = self.include_net_imports(
                    total_generation_stack_out,
                    extra_data_out["Total Load"],
                    extra_data_out["Unserved Energy"],
                )

            total_generation_stack_out = total_generation_stack_out.loc[
                :, (total_generation_stack_out != 0).any(axis=0)
            ]

            # unit conversion return divisor and energy units
            unitconversion = self.capacity_energy_unitconversion(
                total_generation_stack_out, sum_values=True
            )

            total_generation_stack_out = (
                total_generation_stack_out / unitconversion["divisor"]
            )
            extra_data_out = extra_data_out / unitconversion["divisor"]

            # Data table of values to return to main program
            if plot_data_settings["include_barplot_load_lines"]:
                Data_Table_Out = pd.concat(
                    [extra_data_out, total_generation_stack_out], axis=1, sort=False
                )
            else:
                Data_Table_Out = total_generation_stack_out
            Data_Table_Out = Data_Table_Out.add_suffix(f" ({unitconversion['units']}h)")

            mplt = PlotLibrary()
            fig, ax = mplt.get_figure()

            # Set x-tick labels
            if self.custom_xticklabels:
                tick_labels = self.custom_xticklabels
            else:
                tick_labels = total_generation_stack_out.index

            mplt.barplot(
                total_generation_stack_out,
                color=self.PLEXOS_color_dict,
                stacked=True,
                custom_tick_labels=tick_labels,
            )

            ax.set_ylabel(
                f"Total Generation ({unitconversion['units']}h)",
                color="black",
                rotation="vertical",
            )

            if plot_data_settings["include_barplot_load_lines"]:
                for n, scenario in enumerate(total_generation_stack_out.index.unique()):
                    x = [
                        ax.patches[n].get_x(),
                        ax.patches[n].get_x() + ax.patches[n].get_width(),
                    ]

                    height1 = [
                        float(extra_data_out.loc[scenario, "Total Load"].sum())
                    ] * 2

                    if (
                        plot_data_settings["include_barplot_load_storage_charging_line"]
                        and extra_plot_data["Total Load"].sum()
                        > extra_plot_data["Total Demand"].sum()
                    ):
                        ax.plot(
                            x,
                            height1,
                            c="black",
                            linewidth=1.5,
                            linestyle="--",
                            label=load_legend_names["load"],
                        )
                        height2 = [
                            float(extra_data_out.loc[scenario, "Total Demand"])
                        ] * 2
                        ax.plot(
                            x,
                            height2,
                            c="black",
                            linewidth=1.5,
                            label=load_legend_names["demand"],
                        )
                    elif extra_plot_data["Total Demand"].sum() > 0:
                        ax.plot(
                            x,
                            height1,
                            c="black",
                            linewidth=1.5,
                            label=load_legend_names["demand"],
                        )

                    if extra_data_out.loc[scenario, "Unserved Energy"] > 0:
                        height3 = [
                            float(extra_data_out.loc[scenario, "Load-Unserved_Energy"])
                        ] * 2
                        ax.fill_between(
                            x,
                            height3,
                            height1,
                            facecolor="#DD0200",
                            alpha=0.5,
                            label="Unserved Energy",
                        )
            # Add legend
            mplt.add_legend(reverse_legend=True, sort_by=self.ordered_gen)
            # Add title
            if plot_data_settings["plot_title_as_region"]:
                mplt.add_main_title(zone_input)

            outputs[zone_input] = {"fig": fig, "data_table": Data_Table_Out}

        return outputs

    def total_gen_diff(
        self,
        start_date_range: str = None,
        end_date_range: str = None,
        scenario_groupby: str = "Scenario",
        **_,
    ):
        """Creates a stacked bar plot of total generation by technology type, relative to a base scenario.

        Barplots show the change in total generation relative to a base scenario.
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
        # Create Dictionary to hold Datframes for each scenario
        outputs: dict = {}

        # List of properties needed by the plot, properties are a set of tuples and 
        # contain 3 parts: required True/False, property name and scenarios required, 
        # scenarios must be a list.
        properties = [
            (True, "generator_Generation", self.Scenarios),
            (False, f"generator_{self.curtailment_prop}", self.Scenarios),
        ]

        # Runs get_formatted_data within MPlotDataHelper to populate MPlotDataHelper dictionary
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)

        if 1 in check_input_data:
            outputs = MissingInputData()
            return outputs

        for zone_input in self.Zones:

            logger.info(f"Zone = {zone_input}")

            gen_chunks = []
            for scenario in self.Scenarios:

                logger.info(f"Scenario = {scenario}")

                Total_Gen_Stack = self["generator_Generation"].get(scenario)

                # Check if zone has generation, if not skips and breaks out of Multi_Scenario loop
                try:
                    Total_Gen_Stack = Total_Gen_Stack.xs(zone_input, level=self.AGG_BY)
                except KeyError:
                    logger.warning(f"No installed capacity in : {zone_input}")
                    break

                Total_Gen_Stack = self.df_process_gen_inputs(Total_Gen_Stack)

                # Calculates interval step to correct for MWh of generation
                interval_count = self.get_sub_hour_interval_count(Total_Gen_Stack)

                # Insert Curtailment into gen stack if it exists in database
                Stacked_Curt = self[f"generator_{self.curtailment_prop}"].get(scenario)
                if not Stacked_Curt.empty:
                    curtailment_name = self.gen_names_dict.get(
                        "Curtailment", "Curtailment"
                    )
                    if (
                        zone_input
                        in Stacked_Curt.index.get_level_values(self.AGG_BY).unique()
                    ):
                        Stacked_Curt = Stacked_Curt.xs(zone_input, level=self.AGG_BY)
                        Stacked_Curt = self.df_process_gen_inputs(Stacked_Curt)
                        # If using Marmot's curtailment property
                        if self.curtailment_prop == "Curtailment":
                            Stacked_Curt = self.assign_curtailment_techs(Stacked_Curt)
                        Stacked_Curt = Stacked_Curt.sum(axis=1)
                        Total_Gen_Stack.insert(
                            len(Total_Gen_Stack.columns),
                            column=curtailment_name,
                            value=Stacked_Curt,
                        )  # Insert curtailment into
                        Total_Gen_Stack = Total_Gen_Stack.loc[
                            :, (Total_Gen_Stack != 0).any(axis=0)
                        ]

                Total_Gen_Stack = Total_Gen_Stack / interval_count

                if pd.notna(start_date_range):
                    Total_Gen_Stack = self.set_timestamp_date_range(
                        Total_Gen_Stack, start_date_range, end_date_range
                    )

                gen_chunks.append(
                    self.year_scenario_grouper(
                        Total_Gen_Stack, scenario, groupby=scenario_groupby
                    ).sum()
                )

            if not gen_chunks:
                outputs[zone_input] = MissingZoneData()
                continue

            total_generation_stack_out = pd.concat(
                gen_chunks, axis=0, sort=False
            ).fillna(0)
            total_generation_stack_out = total_generation_stack_out.loc[
                :, (total_generation_stack_out != 0).any(axis=0)
            ]

            # Ensures region has generation, else skips
            try:
                # Change to a diff on first scenario
                scen_base = total_generation_stack_out.index[0]
                total_generation_stack_out = (
                    total_generation_stack_out
                    - total_generation_stack_out.xs(scen_base)
                )
            except KeyError:
                outputs[zone_input] = MissingZoneData()
                continue

            total_generation_stack_out.drop(scen_base, inplace=True)  # Drop base entry

            if total_generation_stack_out.empty:
                outputs[zone_input] = MissingZoneData()
                continue

            unitconversion = self.capacity_energy_unitconversion(
                total_generation_stack_out, sum_values=True
            )
            total_generation_stack_out = (
                total_generation_stack_out / unitconversion["divisor"]
            )

            # Data table of values to return to main program
            Data_Table_Out = total_generation_stack_out.add_suffix(
                f" ({unitconversion['units']}h)"
            )

            net_diff = total_generation_stack_out
            try:
                net_diff.drop(columns=curtailment_name, inplace=True)
            except KeyError:
                pass
            net_diff = net_diff.sum(axis=1)

            mplt = PlotLibrary()
            fig, ax = mplt.get_figure()

            mplt.barplot(
                total_generation_stack_out, stacked=True, color=self.PLEXOS_color_dict
            )

            # Set x-tick labels
            tick_labels = total_generation_stack_out.index
            mplt.set_barplot_xticklabels(tick_labels)

            # Add net gen difference line.
            for n, scenario in enumerate(total_generation_stack_out.index.unique()):
                x = [
                    ax.patches[n].get_x(),
                    ax.patches[n].get_x() + ax.patches[n].get_width(),
                ]
                y_net = [net_diff.loc[scenario]] * 2
                ax.plot(x, y_net, c="black", linewidth=1.5, label="Net Gen Change")

            ax.set_ylabel(
                (
                    f"Generation Change ({format(unitconversion['units'])}h) \n "
                    f"relative to {scen_base}"
                ),
                color="black",
                rotation="vertical",
            )

            ax.axhline(linewidth=0.5, linestyle="--", color="grey")
            # Add legend
            mplt.add_legend(reverse_legend=True, sort_by=self.ordered_gen)

            if plot_data_settings["plot_title_as_region"]:
                mplt.add_main_title(zone_input)
            outputs[zone_input] = {"fig": fig, "data_table": Data_Table_Out}
        return outputs

    def total_gen_monthly(self, **kwargs):
        """Creates stacked bar plot of total generation by technology by month.

        A separate bar is created for each scenario.

        This methods calls _monthly_gen() to create the figure.

        Returns:
            dict: Dictionary containing the created plot and its data table.
        """

        outputs = self._monthly_gen(**kwargs)
        return outputs

    def monthly_vre_generation_percentage(self, **kwargs):
        """Creates clustered barplot of total monthly percentage variable renewable generation by technology.

           Each vre technology + curtailment if present is plotted as a separate clustered bar,
           the total of all bars add to 100%.
           Each scenario is plotted on a separate facet plot.
           Technologies that belong to VRE can be set in the ordered_gen_catagories.csv file
           in the Mapping folder.

           This methods calls _monthly_gen() and passes the vre_only=True and
           plot_as_percnt=True arguments to create the figure.

        Returns:
            dict: Dictionary containing the created plot and its data table.
        """

        outputs = self._monthly_gen(vre_only=True, plot_as_percnt=True, **kwargs)
        return outputs

    def monthly_vre_generation(self, **kwargs):
        """Creates clustered barplot of total monthly variable renewable generation by technology.

           Each vre technology + curtailment if present is plotted as a separate clustered bar
           Each scenario is plotted on a separate facet plot.
           Technologies that belong to VRE can be set in the ordered_gen_catagories.csv file
           in the Mapping folder.

           This methods calls _monthly_gen() and passes the vre_only=True arguments to
           create the figure.

        Returns:
            dict: Dictionary containing the created plot and its data table.
        """

        outputs = self._monthly_gen(vre_only=True, **kwargs)
        return outputs

    def _monthly_gen(
        self,
        vre_only: bool = False,
        plot_as_percnt: bool = False,
        start_date_range: str = None,
        end_date_range: str = None,
        **_,
    ):
        """Creates monthly generation plot, internal method called from
            monthly_vre_percentage_generation or monthly_vre_generation

        Args:
            vre_only (bool, optional): If True only plots vre technologies.
                Defaults to False.
            plot_as_percnt (bool, optional): If True only plots data as a percentage.
                Defaults to False.

        Returns:
            dict: Dictionary containing the created plot and its data table.
        """
        # Create Dictionary to hold Datframes for each scenario
        outputs: dict = {}

        if self.AGG_BY == "zone":
            agg = "zone"
        else:
            agg = "region"

        # List of properties needed by the plot, properties are a set of tuples and 
        # contain 3 parts: required True/False, property name and scenarios required, 
        # scenarios must be a list.
        properties = [
            (True, "generator_Generation", self.Scenarios),
            (False, f"generator_{self.curtailment_prop}", self.Scenarios),
            (False, f"{agg}_Load", self.Scenarios),
            (False, f"{agg}_Demand", self.Scenarios),
        ]

        # Runs get_formatted_data within MPlotDataHelper to populate MPlotDataHelper dictionary
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)

        if 1 in check_input_data:
            outputs = MissingInputData()
            return outputs

        ncols, nrows = self.set_facet_col_row_dimensions(multi_scenario=self.Scenarios)
        grid_size = ncols * nrows

        # Used to calculate any excess axis to delete
        plot_number = len(self.Scenarios)
        excess_axs = grid_size - plot_number

        for zone_input in self.Zones:

            logger.info(f"Zone = {zone_input}")

            # Will hold retrieved data for each scenario
            gen_chunks = []
            extra_data_chunks = []

            # Loop gets all data by scenario
            for i, scenario in enumerate(self.Scenarios):

                logger.info(f"Scenario = {scenario}")
                Total_Gen_Stack = self["generator_Generation"].get(scenario)
                # Check if zone has generation, if not skips
                try:
                    Total_Gen_Stack = Total_Gen_Stack.xs(zone_input, level=self.AGG_BY)
                except KeyError:
                    logger.warning(f"No installed capacity in: {zone_input}")
                    continue

                Total_Gen_Stack = self.df_process_gen_inputs(Total_Gen_Stack)
                if vre_only:
                    Total_Gen_Stack = Total_Gen_Stack[
                        Total_Gen_Stack.columns.intersection(self.gen_categories.vre)
                    ]

                if Total_Gen_Stack.empty:
                    if vre_only:
                        logger.warning(f"No vre in: {zone_input}")
                    gen_chunks.append(pd.DataFrame())
                    continue

                Total_Gen_Stack.columns = Total_Gen_Stack.columns.add_categories(
                    "timestamp"
                )

                # Calculates interval step to correct for MWh of generation if data is subhourly
                interval_count = self.get_sub_hour_interval_count(Total_Gen_Stack)

                # Insert Curtailment into gen stack if it exists in database
                Stacked_Curt = self[f"generator_{self.curtailment_prop}"].get(scenario)
                if not Stacked_Curt.empty:
                    curtailment_name = self.gen_names_dict.get(
                        "Curtailment", "Curtailment"
                    )
                    if (
                        zone_input
                        in Stacked_Curt.index.get_level_values(self.AGG_BY).unique()
                    ):
                        Stacked_Curt = Stacked_Curt.xs(zone_input, level=self.AGG_BY)
                        Stacked_Curt = self.df_process_gen_inputs(Stacked_Curt)
                        # If using Marmot's curtailment property
                        if self.curtailment_prop == "Curtailment":
                            Stacked_Curt = self.assign_curtailment_techs(Stacked_Curt)
                        Stacked_Curt = Stacked_Curt.sum(axis=1)
                        Total_Gen_Stack.insert(
                            len(Total_Gen_Stack.columns),
                            column=curtailment_name,
                            value=Stacked_Curt,
                        )
                        Total_Gen_Stack = Total_Gen_Stack.loc[
                            :, (Total_Gen_Stack != 0).any(axis=0)
                        ]

                extra_data_frames = []
                extra_property_names = [f"{agg}_Load", f"{agg}_Demand"]
                # Get and process extra properties
                for ext_prop in extra_property_names:
                    df: pd.DataFrame = self[ext_prop].get(scenario)
                    if df.empty or not plot_data_settings["include_barplot_load_lines"]:
                        date_index = pd.date_range(
                            start="2010-01-01", periods=1, freq="H", name="timestamp"
                        )
                        df = pd.DataFrame(data=[0], index=date_index, columns=["values"])
                    else:
                        df = df.xs(zone_input, level=self.AGG_BY)
                        df = df.groupby(["timestamp"]).sum()
                    df = df.rename(columns={"values": ext_prop})
                    extra_data_frames.append(df)

                extra_plot_data = pd.concat(extra_data_frames, axis=1).fillna(0)

                extra_plot_data = extra_plot_data.rename(
                    columns={
                        f"{agg}_Load": "Total Load",
                        f"{agg}_Demand": "Total Demand",
                    }
                )

                if pd.notna(start_date_range):
                    Total_Gen_Stack = self.set_timestamp_date_range(
                        Total_Gen_Stack, start_date_range, end_date_range
                    )
                    if Total_Gen_Stack.empty is True:
                        logger.warning("No Generation in selected Date Range")
                        continue
                # Adjust extra data to generator date range
                extra_plot_data = extra_plot_data.loc[
                    Total_Gen_Stack.index.min() : Total_Gen_Stack.index.max()
                ]

                def _group_monthly_data(
                    input_df: pd.DataFrame, interval_count: int
                ) -> pd.DataFrame:
                    """Groups data into months"""

                    monthly_df = input_df / interval_count
                    monthly_df = monthly_df.groupby(pd.Grouper(freq="M")).sum()
                    monthly_df.reset_index(drop=False, inplace=True)
                    monthly_df.set_index("timestamp", inplace=True)
                    if len(monthly_df.index.year.unique()) > 1:
                        monthly_df.index = monthly_df.index.strftime("%B-%Y")
                    else:
                        monthly_df.index = monthly_df.index.strftime("%B")
                    return monthly_df

                # Group data into months
                monthly_gen_stack = _group_monthly_data(Total_Gen_Stack, interval_count)
                monthly_extra_plot_data = _group_monthly_data(
                    extra_plot_data, interval_count
                )

                # If plotting percentage data convert to percentages
                if plot_as_percnt:
                    monthly_total_gen = pd.DataFrame(
                        monthly_gen_stack.T.sum(), columns=["Total Generation"]
                    )
                    for vre_col in monthly_gen_stack.columns:
                        monthly_gen_stack[vre_col] = (
                            monthly_gen_stack[vre_col]
                            / monthly_total_gen["Total Generation"]
                        )

                # Add scenario index
                scenario_names = pd.Series(
                    [scenario] * len(monthly_gen_stack), name="Scenario"
                )
                monthly_gen_stack = monthly_gen_stack.set_index(
                    [scenario_names], append=True
                )
                monthly_extra_plot_data = monthly_extra_plot_data.set_index(
                    pd.Series(
                        [scenario] * len(monthly_extra_plot_data), name="Scenario"
                    ),
                    append=True,
                )
                # Add all data to lists
                gen_chunks.append(monthly_gen_stack)
                extra_data_chunks.append(monthly_extra_plot_data)

            if not gen_chunks:
                # If no generation in select zone/region
                outputs[zone_input] = MissingZoneData()
                continue

            # Concat all data into single data-frames
            Gen_Out = pd.concat(gen_chunks, axis=0).sort_index(axis=1)
            # Drop any technologies with 0 Gen
            Gen_Out = Gen_Out.loc[:, (Gen_Out != 0).any(axis=0)]

            if Gen_Out.empty:
                outputs[zone_input] = MissingZoneData()
                continue

            extra_data_out = pd.concat(extra_data_chunks, axis=0, sort=False)

            # Add Net Imports if desired
            if plot_data_settings["include_barplot_net_imports"] and not vre_only:
                Gen_Out = self.include_net_imports(
                    Gen_Out, extra_data_out["Total Load"]
                )

            if not plot_as_percnt:
                # unit conversion return divisor and energy units
                if vre_only:
                    unitconversion = self.capacity_energy_unitconversion(Gen_Out)
                else:
                    unitconversion = self.capacity_energy_unitconversion(
                        Gen_Out, sum_values=True
                    )
                Gen_Out = Gen_Out / unitconversion["divisor"]
                extra_data_out = extra_data_out / unitconversion["divisor"]

                # Data table of values to return to main program
                Data_Table_Out = pd.concat(
                    [extra_data_out, Gen_Out], axis=1
                ).add_suffix(f" ({unitconversion['units']}h)")
            else:
                Data_Table_Out = Gen_Out.add_suffix(f" (%-Gen)") * 100

            mplt = PlotLibrary(nrows, ncols, sharey=True, squeeze=False, ravel_axs=True)
            fig, axs = mplt.get_figure()
            plt.subplots_adjust(wspace=0.05, hspace=0.5)

            if ncols > 1:
                font_scaling_ratio = 1 + ((ncols - 1) * 0.09)
                plt.rcParams["xtick.labelsize"] *= font_scaling_ratio
                plt.rcParams["ytick.labelsize"] *= font_scaling_ratio
                plt.rcParams["legend.fontsize"] *= font_scaling_ratio
                plt.rcParams["axes.labelsize"] *= font_scaling_ratio
                plt.rcParams["axes.titlesize"] *= font_scaling_ratio

            for i, scenario in enumerate(
                Gen_Out.index.get_level_values("Scenario").unique()
            ):

                month_gen = Gen_Out.xs(scenario, level="Scenario")
                # Drop 0 generation techs
                month_gen = month_gen.loc[:, (month_gen != 0).any(axis=0)]
                # Drop months with no data
                month_gen = month_gen.loc[(month_gen != 0).any(axis=1)]

                if vre_only:
                    stack = False
                else:
                    stack = True

                mplt.barplot(
                    month_gen, color=self.PLEXOS_color_dict, stacked=stack, sub_pos=i
                )

                axs[i].margins(x=0.01)
                axs[i].set_xlabel("")

                if plot_as_percnt:
                    mplt.set_yaxis_major_tick_format(tick_format="percent", sub_pos=i)

                if not vre_only and plot_data_settings["include_barplot_load_lines"]:
                    month_extra = extra_data_out.xs(scenario, level="Scenario")
                    for n, _m in enumerate(month_extra.index):
                        x = [
                            axs[i].patches[n].get_x(),
                            axs[i].patches[n].get_x() + axs[i].patches[n].get_width(),
                        ]
                        height1 = [float(month_extra.loc[_m, "Total Load"])] * 2

                        if (
                            plot_data_settings["include_barplot_load_storage_charging_line"]
                            and month_extra.loc[_m, "Total Load"].sum()
                            > month_extra.loc[_m, "Total Demand"].sum()
                        ):

                            axs[i].plot(
                                x,
                                height1,
                                c="black",
                                linewidth=2,
                                linestyle="--",
                                label=load_legend_names["load"],
                            )
                            height2 = [float(month_extra.loc[_m, "Total Demand"])] * 2
                            axs[i].plot(
                                x,
                                height2,
                                c="black",
                                linewidth=1.5,
                                label=load_legend_names["demand"],
                            )
                        elif month_extra.loc[_m, "Total Demand"].sum() > 0:
                            axs[i].plot(
                                x,
                                height1,
                                c="black",
                                linewidth=2,
                                label=load_legend_names["demand"],
                            )

            # add facet labels
            mplt.add_facet_labels(xlabels=self.xlabels, ylabels=self.ylabels)
            # Add legend
            mplt.add_legend(reverse_legend=True, sort_by=self.ordered_gen)
            # Remove extra axes
            mplt.remove_excess_axs(excess_axs, grid_size)

            # Y-label should change if there are facet labels, leave at 40 for now, 
            # works for all values in spacing
            labelpad = 40
            if plot_as_percnt:
                plt.ylabel(
                    f"% of Generation",
                    color="black",
                    rotation="vertical",
                    labelpad=labelpad,
                )
            else:
                plt.ylabel(
                    f"Total Generation ({unitconversion['units']}h)",
                    color="black",
                    rotation="vertical",
                    labelpad=labelpad,
                )

            if plot_data_settings["plot_title_as_region"]:
                mplt.add_main_title(zone_input)

            outputs[zone_input] = {"fig": fig, "data_table": Data_Table_Out}

        return outputs

    def total_gen_pie(
        self, start_date_range: str = None, end_date_range: str = None, **_
    ):
        """Creates a pie chart of total generation and curtailment.

        Each sceanrio is plotted as a separate pie chart.

        Args:
            start_date_range (str, optional): Defines a start date at which to represent data from.
                Defaults to None.
            end_date_range (str, optional): Defines a end date at which to represent data to.
                Defaults to None.

        Returns:
            dict: Dictionary containing the created plot and its data table.
        """
        # Create Dictionary to hold Datframes for each scenario
        outputs: dict = {}

        # List of properties needed by the plot, properties are a set of tuples and 
        # contain 3 parts: required True/False, property name and scenarios required, 
        # scenarios must be a list.
        properties = [
            (True, "generator_Generation", self.Scenarios),
            (False, f"generator_{self.curtailment_prop}", self.Scenarios),
        ]

        # Runs get_formatted_data within MPlotDataHelper to populate MPlotDataHelper dictionary
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)

        if 1 in check_input_data:
            outputs = MissingInputData()
            return outputs

        ncols, nrows = self.set_facet_col_row_dimensions(multi_scenario=self.Scenarios)
        grid_size = ncols * nrows

        # Used to calculate any excess axis to delete
        plot_number = len(self.Scenarios)
        excess_axs = grid_size - plot_number

        for zone_input in self.Zones:
            Total_Gen_Out = pd.DataFrame()
            logger.info(f"Zone = {zone_input}")

            mplt = SetupSubplot(
                nrows, ncols, sharey=True, squeeze=False, ravel_axs=True
            )
            fig, axs = mplt.get_figure()

            gen_chunks = []
            for i, scenario in enumerate(self.Scenarios):

                logger.info(f"Scenario = {scenario}")
                Total_Gen_Stack = self["generator_Generation"].get(scenario)

                # Check if zone has generation, if not skips
                try:
                    Total_Gen_Stack = Total_Gen_Stack.xs(zone_input, level=self.AGG_BY)
                except KeyError:
                    logger.warning(f"No installed capacity in: {zone_input}")
                    continue

                Total_Gen_Stack = self.df_process_gen_inputs(Total_Gen_Stack)

                if pd.notna(start_date_range):
                    Total_Gen_Stack = self.set_timestamp_date_range(
                        Total_Gen_Stack, start_date_range, end_date_range
                    )
                    if Total_Gen_Stack.empty is True:
                        logger.warning("No Generation in selected Date Range")
                        continue

                # Insert Curtailment into gen stack if it exists in database
                Stacked_Curt = self[f"generator_{self.curtailment_prop}"].get(scenario)
                if not Stacked_Curt.empty:
                    curtailment_name = self.gen_names_dict.get(
                        "Curtailment", "Curtailment"
                    )
                    if (
                        zone_input
                        in Stacked_Curt.index.get_level_values(self.AGG_BY).unique()
                    ):
                        Stacked_Curt = Stacked_Curt.xs(zone_input, level=self.AGG_BY)
                        Stacked_Curt = self.df_process_gen_inputs(Stacked_Curt)
                        # If using Marmot's curtailment property
                        if self.curtailment_prop == "Curtailment":
                            Stacked_Curt = self.assign_curtailment_techs(Stacked_Curt)
                        Stacked_Curt = Stacked_Curt.sum(axis=1)
                        # Insert curtailment into
                        Total_Gen_Stack.insert(
                            len(Total_Gen_Stack.columns),
                            column=curtailment_name,
                            value=Stacked_Curt,
                        )
                        Total_Gen_Stack = Total_Gen_Stack.loc[
                            :, (Total_Gen_Stack != 0).any(axis=0)
                        ]

                Total_Gen_Stack = self.year_scenario_grouper(
                    Total_Gen_Stack, scenario
                ).sum()

                Total_Gen_Stack = (
                    Total_Gen_Stack / Total_Gen_Stack.to_numpy().sum()
                ) * 100
                gen_chunks.append(Total_Gen_Stack)

            if not gen_chunks:
                outputs[zone_input] = MissingZoneData()
                continue

            Total_Gen_Out = pd.concat(gen_chunks, axis=0, sort=False).fillna(0)
            # Pie charts can't have negative values
            Total_Gen_Out[Total_Gen_Out < 0] = 0
            Total_Gen_Out = Total_Gen_Out.loc[:, (Total_Gen_Out != 0).any(axis=0)]

            if Total_Gen_Out.empty:
                outputs[zone_input] = MissingZoneData()
                continue

            for i, scenario in enumerate(Total_Gen_Out.index):

                scenario_data = Total_Gen_Out.loc[scenario]
                axs[i].pie(
                    scenario_data,
                    labels=scenario_data.index,
                    shadow=True,
                    startangle=90,
                    labeldistance=None,
                    colors=[
                        self.PLEXOS_color_dict.get(x, "#333333")
                        for x in scenario_data.index
                    ],
                )
                axs[i].legend().set_visible(False)

            # add facet labels
            mplt.add_facet_labels(xlabels=self.xlabels, ylabels=self.ylabels)
            # Add legend
            mplt.add_legend(reverse_legend=True, sort_by=self.ordered_gen)
            # Remove extra axes
            mplt.remove_excess_axs(excess_axs, grid_size)

            plt.tick_params(
                labelcolor="none", top=False, bottom=False, left=False, right=False
            )
            plt.ylabel(f"Total Generation (%)", color="black", rotation="vertical")
            if plot_data_settings["plot_title_as_region"]:
                mplt.add_main_title(zone_input)

            outputs[zone_input] = {"fig": fig, "data_table": Total_Gen_Out}

        return outputs

    def total_gen_facet(
        self,
        start_date_range: str = None,
        end_date_range: str = None,
        scenario_groupby: str = "Scenario",
        **_,
    ):
        """Creates a stacked barplot of total generation.
        Each scenario will be plotted in a separate bar subplot.
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
        properties = [(True, "generator_Generation", self.Scenarios)]

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


                total_generation = self["generator_Generation"].get(
                    scenario
                )

                try:
                    generation = total_generation.xs(zone_input, level=self.AGG_BY)
                except KeyError:
                    logger.warning(f"No installed capacity in {zone_input}")
                    outputs[zone_input] = MissingZoneData()
                    continue

                generation = self.df_process_gen_inputs(
                    generation
                )

                if pd.notna(start_date_range):
                    generation = self.set_timestamp_date_range(
                        generation, start_date_range, end_date_range
                    )
                    if generation.empty is True:
                        logger.warning("No Data in selected Date Range")
                        continue
                generation_grouped = self.year_scenario_grouper(
                        generation, scenario, groupby=scenario_groupby
                    ).sum()
                

                # unitconversion based off peak generation hour, only checked once
                if i == 0:
                    unitconversion = self.capacity_energy_unitconversion(
                    generation_grouped, sum_values=True
                    )
                generation_grouped = (
                    generation_grouped / unitconversion["divisor"]
                )

                data_tables.append(generation_grouped)

                # Set x-tick labels
                if self.custom_xticklabels:
                    tick_labels = self.custom_xticklabels
                elif scenario_groupby == "Year-Scenario":
                    tick_labels = [x.split(":")[0] for x in generation_grouped.index]
                else:
                    tick_labels = generation_grouped.index

                mplt.barplot(
                    generation_grouped,
                    color=self.PLEXOS_color_dict,
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
                    f"Total Generation ({unitconversion['units']})",
                    color="black",
                    rotation="vertical",
                    labelpad=labelpad,
            )

            Data_Table_Out = pd.concat(data_tables)
            Data_Table_Out = Data_Table_Out.add_suffix(f" ({unitconversion['units']})")
            outputs[zone_input] = {"fig": fig, "data_table": Data_Table_Out}
        return outputs
