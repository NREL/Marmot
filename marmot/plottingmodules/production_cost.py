# -*- coding: utf-8 -*-
"""System operating cost plots.

This module plots figures related to the cost of operating the power system.
Plots can be broken down by cost categories, generator types etc. 

@author: Daniel Levie
"""

import logging
from pathlib import Path
from typing import List

import pandas as pd
import matplotlib.pyplot as plt

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

plot_data_settings: dict = mconfig.parser("plot_data")
logger = logging.getLogger("plotter." + __name__)
include_batteries: bool = mconfig.parser("plot_data","include_explicit_battery_objects")


# gen_names_dict = pd.read_csv('/Users/mschwarz/Marmot_local/Marmot/input_files/mapping_folder/gen_names.csv')
# gen_names_dict = gen_names_dict.set_index(gen_names_dict.columns[0]).squeeze().to_dict()

# self = SystemCosts(
#     Zones = ['USA'],
#     AGG_BY = 'Country',
#     Scenarios = ['ACDH90by35_2035','LimDH90by35_2035','LCCDH90by35_2035','VSCDH90by35_2035'],
#     ordered_gen = ['Nuclear', 'Coal', 'Gas-CC', 'Gas-CC CCS', 'Gas-CT', 'Gas', 'Gas-Steam', 'Dual Fuel', 'DualFuel', 'Oil-Gas-Steam', 'Oil', 'Hydro', 'Ocean', 'Geothermal', 'Biomass', 'Biopower', 'Other', 'VRE', 'Wind', 'Offshore Wind', 'OffshoreWind', 'Solar', 'PV', 'dPV', 'CSP', 'PV-Battery', 'Battery', 'OSW-Battery', 'PHS', 'Storage', 'Net Imports', 'Curtailment', 'curtailment', 'Demand', 'Deamand + Storage Charging'],
#     marmot_solutions_folder = '/Users/mschwarz/NTPS_local',
#     gen_names_dict = gen_names_dict
# )


class SystemCosts(PlotDataStoreAndProcessor):
    """System total cost plots.

    The cost.py module contains methods that are
    related to the capital and operating cost of the power system.

    SystemCosts inherits from the PlotDataStoreAndProcessor class to assist
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

    def prod_cost(
        self,
        start_date_range: str = None,
        end_date_range: str = None,
        custom_data_file_path: Path = None,
        scenario_groupby: str = "Scenario",
        **_,
    ):
        """Plots total system net revenue and cost.

        Total revenue is made up of reserve and energy revenues which are displayed in a stacked
        bar plot with total generation cost. Net revensue is represented by a dot.
        Each sceanrio is plotted as a separate bar.

        Args:
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
            dict: Dictionary containing the created plot and its data table.
        """
        outputs: dict = {}

        # List of properties needed by the plot, properties are a set of tuples and
        # contain 3 parts: required True/False, property name and scenarios required,
        # scenarios must be a list.
        properties = [
            (True, "generator_Total_Generation_Cost", self.Scenarios),
            (True, "generator_Pool_Revenue", self.Scenarios),
            (True, "generator_Reserves_Revenue", self.Scenarios),
            (True, "generator_Installed_Capacity", self.Scenarios),
        ]

        # Runs get_formatted_data within PlotDataStoreAndProcessor to populate PlotDataStoreAndProcessor dictionary
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)

        # Checks if all data required by plot is available, if 1 in list required data is missing
        if 1 in check_input_data:
            return MissingInputData()

        for zone_input in self.Zones:
            total_cost_chunk = []
            logger.info(f"{self.AGG_BY} = {zone_input}")
            for scenario in self.Scenarios:

                logger.info(f"Scenario = {scenario}")
                Total_Installed_Capacity = self["generator_Installed_Capacity"].get(
                    scenario
                )
                # Check if zone has installed generation, if not skips
                try:
                    Total_Installed_Capacity = Total_Installed_Capacity.xs(
                        zone_input, level=self.AGG_BY
                    )
                except KeyError:
                    logger.warning(f"No installed capacity in : {zone_input}")
                    continue
                Total_Installed_Capacity = self.df_process_gen_inputs(
                    Total_Installed_Capacity
                )
                Total_Installed_Capacity.reset_index(drop=True, inplace=True)
                Total_Installed_Capacity = Total_Installed_Capacity.iloc[0]

                gen_cost = self["generator_Total_Generation_Cost"].get(scenario)
                gen_cost = gen_cost.xs(zone_input, level=self.AGG_BY)
                gen_cost = self.df_process_gen_inputs(gen_cost)
                gen_cost = gen_cost.sum(axis=0) * -1
                # gen_cost = gen_cost/Total_Installed_Capacity #Change to $/MW-year
                gen_cost.rename("Generation Cost", inplace=True)

                Pool_Revenues = self["generator_Pool_Revenue"].get(scenario)
                Pool_Revenues = Pool_Revenues.xs(zone_input, level=self.AGG_BY)
                Pool_Revenues = self.df_process_gen_inputs(Pool_Revenues)
                Pool_Revenues = Pool_Revenues.sum(axis=0)
                # Pool_Revenues = Pool_Revenues/Total_Installed_Capacity #Change to $/MW-year
                Pool_Revenues.rename("Energy Revenues", inplace=True)

                ### Might change to Net Reserve Revenue at later date
                Reserve_Revenues = self["generator_Reserves_Revenue"].get(scenario)
                Reserve_Revenues = Reserve_Revenues.xs(zone_input, level=self.AGG_BY)
                Reserve_Revenues = self.df_process_gen_inputs(Reserve_Revenues)
                Reserve_Revenues = Reserve_Revenues.sum(axis=0)
                # Reserve_Revenues = Reserve_Revenues/Total_Installed_Capacity #Change to $/MW-year
                Reserve_Revenues.rename("Reserve Revenues", inplace=True)

                Total_Systems_Cost = pd.concat(
                    [gen_cost, Pool_Revenues, Reserve_Revenues], axis=1, sort=False
                )

                Total_Systems_Cost = Total_Systems_Cost.sum(axis=0)
                Total_Systems_Cost = Total_Systems_Cost.rename(scenario)

                total_cost_chunk.append(Total_Systems_Cost)

            total_systems_cost_out = pd.concat(total_cost_chunk, axis=1, sort=False)

            total_systems_cost_out = total_systems_cost_out.T

            # total_systems_cost_out = total_systems_cost_out/1000 #Change to $/kW-year
            total_systems_cost_out = (
                total_systems_cost_out / 1e6
            )  # Convert cost to millions

            if pd.notna(custom_data_file_path):
                total_systems_cost_out = self.insert_custom_data_columns(
                    total_systems_cost_out, custom_data_file_path
                )

            Net_Revenue = total_systems_cost_out.sum(axis=1)

            # Checks if Net_Revenue contains data, if not skips zone and does not return a plot
            if Net_Revenue.empty:
                out = MissingZoneData()
                outputs[zone_input] = out
                continue

            # Data table of values to return to main program
            Data_Table_Out = total_systems_cost_out.add_suffix(" (Million $)")

            mplt = PlotLibrary()
            fig, ax = mplt.get_figure()

            # Set x-tick labels
            if self.custom_xticklabels:
                tick_labels = self.custom_xticklabels
            else:
                tick_labels = total_systems_cost_out.index

            mplt.barplot(
                total_systems_cost_out, stacked=True, custom_tick_labels=tick_labels
            )
            ax.plot(
                Net_Revenue.index,
                Net_Revenue.values,
                color="black",
                linestyle="None",
                marker="o",
                label="Net Revenue",
            )

            ax.set_ylabel(
                "Total System Net Rev, Rev, & Cost (Million $)",
                color="black",
                rotation="vertical",
            )
            ax.margins(x=0.01)

            mplt.add_legend(reverse_legend=True)
            if plot_data_settings["plot_title_as_region"]:
                mplt.add_main_title(zone_input)

            outputs[zone_input] = {"fig": fig, "data_table": Data_Table_Out}
        return outputs

    def sys_cost(
        self,
        start_date_range: str = None,
        end_date_range: str = None,
        custom_data_file_path: Path = None,
        scenario_groupby: str = "Scenario",
        **_,
    ):
        """Creates a stacked bar plot of Total Operational Cost, Capital Cost, and Cost of Unserved Energy.

        Plot only shows totals and is NOT broken down into technology or cost type beyond operational/capital 
        specific values.
        Each sceanrio is plotted as a separate bar.

        Args:
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
            dict: Dictionary containing the created plot and its data table.
        """
        outputs: dict = {}

        if self.AGG_BY == "zone":
            agg = "zone"
        else:
            agg = "region"

        # List of properties needed by the plot, properties are a set of tuples and
        # contain 3 parts: required True/False, property name and scenarios required,
        # scenarios must be a list.
        properties = [
            (True, "generator_Total_Generation_Cost", self.Scenarios),
            (True, "generator_Build_Cost", self.Scenarios),
            (False, f"{agg}_Cost_Unserved_Energy", self.Scenarios),
        ]

        # Runs get_formatted_data within PlotDataStoreAndProcessor to populate PlotDataStoreAndProcessor dictionary
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)

        # Checks if all data required by plot is available, if 1 in list required data is missing
        if 1 in check_input_data:
            return MissingInputData()

        for zone_input in self.Zones:
            system_cost_chunk = []
            logger.info(f"{self.AGG_BY} = {zone_input}")

            for scenario in self.Scenarios:
                logger.info(f"Scenario = {scenario}")

                gen_cost: pd.DataFrame = self["generator_Total_Generation_Cost"].get(
                    scenario
                )
                try:
                    gen_cost = gen_cost.xs(zone_input, level=self.AGG_BY)
                except KeyError:
                    logger.warning(f"No Generators found in : {zone_input}")
                    continue
                gen_cost = gen_cost.rename(columns={"values": "Total Operational Cost"})

                cap_cost: pd.DataFrame = self["generator_Build_Cost"].get(
                    scenario
                )
                try:
                    cap_cost = cap_cost.xs(zone_input, level=self.AGG_BY)
                except KeyError:
                    logger.warning(f"No Generators found in : {zone_input}")
                    continue
                cap_cost = cap_cost.rename(columns={"values": "Total Capital Cost"})

                cost_unserved_energy: pd.DataFrame = self[
                    f"{agg}_Cost_Unserved_Energy"
                ][scenario]
                if cost_unserved_energy.empty:
                    cost_unserved_energy = self["generator_Total_Generation_Cost"][
                        scenario
                    ].copy()
                    cost_unserved_energy.iloc[:, 0] = 0
                cost_unserved_energy = cost_unserved_energy.xs(
                    zone_input, level=self.AGG_BY
                )
                cost_unserved_energy = cost_unserved_energy.rename(
                    columns={"values": "Cost Unserved Energy"}
                )

                if pd.notna(start_date_range):
                    gen_cost, cap_cost, cost_unserved_energy = set_timestamp_date_range(
                        [gen_cost, cap_cost, cost_unserved_energy],
                        start_date_range,
                        end_date_range,
                    )
                    if gen_cost.empty is True:
                        logger.warning("No generation in selected Date Range")
                        continue
                    if cap_cost.empty is True:
                        logger.warning("No generation capital costs in selected Date Range")
                        continue

                gen_cost = self.year_scenario_grouper(
                    gen_cost, scenario, groupby=scenario_groupby
                ).sum()
                cap_cost = self.year_scenario_grouper(
                    cap_cost, scenario, groupby=scenario_groupby
                ).sum()
                cost_unserved_energy = self.year_scenario_grouper(
                    cost_unserved_energy, scenario, groupby=scenario_groupby
                ).sum()

                system_cost_chunk.append(
                    pd.concat([cap_cost, gen_cost, cost_unserved_energy], axis=1)
                )

            # Checks if gen_cost_out_chunks contains data, if not skips zone and does not return a plot
            if not system_cost_chunk:
                outputs[zone_input] = MissingZoneData()
                continue

            total_systems_cost_out = pd.concat(system_cost_chunk, axis=0, sort=False)
            total_systems_cost_out = (
                total_systems_cost_out / 1000000
            )  # Convert cost to millions

            # Checks if total_systems_cost_out contains data, if not skips zone and does not return a plot
            if total_systems_cost_out.empty:
                outputs[zone_input] = MissingZoneData()
                continue

            if pd.notna(custom_data_file_path):
                total_systems_cost_out = self.insert_custom_data_columns(
                    total_systems_cost_out, custom_data_file_path
                )

            # Data table of values to return to main program
            Data_Table_Out = total_systems_cost_out.add_suffix(" (Million $)")

            mplt = PlotLibrary()
            fig, ax = mplt.get_figure()

            # Set x-tick labels
            if self.custom_xticklabels:
                tick_labels = self.custom_xticklabels
            else:
                tick_labels = total_systems_cost_out.index

            mplt.barplot(
                total_systems_cost_out, stacked=True, custom_tick_labels=tick_labels
            )
            ax.set_ylabel(
                "Total System Cost (Million $)", color="black", rotation="vertical"
            )
            ax.margins(x=0.01)

            mplt.add_legend(reverse_legend=True)
            if plot_data_settings["plot_title_as_region"]:
                mplt.add_main_title(zone_input)

            cost_totals = total_systems_cost_out.sum(axis=1)  # holds total of each bar

            # inserts values into bar stacks
            for patch in ax.patches:
                width, height = patch.get_width(), patch.get_height()
                if height <= 1:
                    continue
                x, y = patch.get_xy()
                ax.text(
                    x + width / 2,
                    y + height / 2,
                    "{:,.0f}".format(height),
                    horizontalalignment="center",
                    verticalalignment="center",
                    fontsize=12,
                )

            # inserts total bar value above each bar
            for k, patch in enumerate(ax.patches):
                height = cost_totals[k]
                width = patch.get_width()
                x, y = patch.get_xy()
                ax.text(
                    x + width / 2,
                    y + height + 0.05 * max(ax.get_ylim()),
                    "{:,.0f}".format(height),
                    horizontalalignment="center",
                    verticalalignment="center",
                    fontsize=15,
                    color="red",
                )
                if k >= len(cost_totals) - 1:
                    break

            outputs[zone_input] = {"fig": fig, "data_table": Data_Table_Out}
        return outputs
    def detailed_gen_cost_facet(
        self,
        start_date_range: str = None,
        end_date_range: str = None,
        custom_data_file_path: Path = None,
        scenario_groupby: str = "Scenario",
        **_,
    ):
        """Creates stacked bar plot of total generation cost by cost type (fuel, emission, start cost etc.)

        Creates a more detailed system cost plot.
        Each scenario is plotted as a separate bar subplot.

        Args:
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
            dict: Dictionary containing the created plot and its data table.
        """
        outputs: dict = {}

        # List of properties needed by the plot, properties are a set of tuples and
        # contain 3 parts: required True/False, property name and scenarios required,
        # scenarios must be a list.
        properties = [
            (False, "generator_Fuel_Cost", self.Scenarios),
            (False, "generator_FOM_Cost", self.Scenarios),
            (False, "generator_VOM_Cost", self.Scenarios),
            (False, "generator_Start_and_Shutdown_Cost", self.Scenarios),
            (False, "generator_Reserves_VOM_Cost", self.Scenarios),
            (False, "generator_Emissions_Cost", self.Scenarios),
            (False, "generator_Fossil_Annualized_Build_Cost", self.Scenarios),
            (False, "generator_RE_Annualized_Build_Cost", self.Scenarios),
            (False, "generator_UoS_Cost", self.Scenarios),
            (False, "generator_Annualized_One_Time_Cost", self.Scenarios),
            (False, "batterie_Annualized_Build_Cost", self.Scenarios),
        ]

        column_dict = {
                        "generator_Fuel_Cost": "Fuel",
                        "generator_FOM_Cost": "FO&M",
                        "generator_VOM_Cost": "VO&M",
                        "generator_Start_and_Shutdown_Cost": "Start & Shutdown",
                        "generator_Reserves_VOM_Cost": "Reserves VO&M",
                        "generator_Emissions_Cost": "Emissions",
                        "generator_Fossil_Annualized_Build_Cost":"Non-Renewable Capacity",
                        "generator_RE_Annualized_Build_Cost":"Renewable Purchases",
                        "generator_UoS_Cost":"Production Tax Credit",
                        "generator_Annualized_One_Time_Cost":"Spur Line",
                        "batterie_Annualized_Build_Cost":"Annualized Storage Build",
        }

        if scenario_groupby == "Scenario":
            for i in range(len(properties)):
                column_dict[properties[i][1] + "_NPV"] = column_dict[properties[i][1]]
                del column_dict[properties[i][1]]
                properties[i] = (properties[i][0], properties[i][1] + "_NPV", properties[i][2])
                
        # Runs get_formatted_data within PlotDataStoreAndProcessor to populate PlotDataStoreAndProcessor dictionary
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)

        # Checks if all data required by plot is available, if 1 in list required data is missing
        if 1 in check_input_data:
            return MissingInputData()

        for zone_input in self.Zones:
            logger.info(f"Zone = {zone_input}")
            gen_cost_out_chunks = []

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

            plt.subplots_adjust(wspace = 0.05, hspace = 0.5)

            # If creating a facet plot the font is scaled by 9% for each added x dimesion fact plot
            if ncols > 1:
                font_scaling_ratio = 1 + ((ncols - 1) * 0.09)
                plt.rcParams["xtick.labelsize"] *= font_scaling_ratio
                plt.rcParams["ytick.labelsize"] *= font_scaling_ratio
                plt.rcParams["legend.fontsize"] *= font_scaling_ratio
                plt.rcParams["axes.labelsize"] *= font_scaling_ratio
                plt.rcParams["axes.titlesize"] *= font_scaling_ratio

            for i, scenario in enumerate(self.Scenarios):
                logger.info(f"Scenario = {scenario}")

                data_frames_lst = []
                for prop_name in properties:
                    df: pd.DataFrame = self[prop_name[1]].get(scenario)
                    if df.empty:
                        continue
                    else:
                        try:
                            df = df.xs(zone_input, level=self.AGG_BY)
                            df = df.groupby(["timestamp"]).sum()
                        except KeyError:
                            logger.warning(f"No Generators found in: {zone_input}")
                            break

                    if (prop_name[1] == "generator_VOM_Cost" or prop_name[1] == "generator_VOM_Cost_NPV"):
                        try:
                            df["values"].to_numpy()[df["values"].to_numpy() < 0] = 0
                        except:
                            df[0].to_numpy()[df[0].to_numpy() < 0] = 0
                    df = df.rename(columns={"values": prop_name[1],0: prop_name[1]})

                    data_frames_lst.append(df)

                detailed_gen_cost = pd.concat(data_frames_lst, axis=1).fillna(0)
                detailed_gen_cost = detailed_gen_cost.rename(
                    columns=column_dict
                )

                if pd.notna(start_date_range):
                    detailed_gen_cost = set_timestamp_date_range(
                        detailed_gen_cost, start_date_range, end_date_range
                    )
                    if detailed_gen_cost.empty is True:
                        logger.warning("No Generation in selected Date Range")
                        continue
                detailed_gen_cost = self.year_scenario_grouper(
                        detailed_gen_cost, scenario, groupby=scenario_groupby
                ).sum()
                # Convert costs to millions
                detailed_gen_cost = detailed_gen_cost / 1000000

                detailed_gen_cost["Non-Renewable Capacity"] = detailed_gen_cost["Non-Renewable Capacity"] + detailed_gen_cost["Annualized Storage Build"]
                detailed_gen_cost["Renewable Purchases"] = detailed_gen_cost["Renewable Purchases"] + detailed_gen_cost["Production Tax Credit"]
                detailed_gen_cost.drop(["Annualized Storage Build", "Production Tax Credit"], axis= "columns", inplace = True)

                # Delete columns that are all 0
                detailed_gen_cost = detailed_gen_cost.loc[
                    :, (detailed_gen_cost != 0).any(axis=0)
                ]

                gen_cost_out_chunks.append(detailed_gen_cost)
                net_cost = detailed_gen_cost.sum(axis=1)

                # Set x-tick labels
                if self.custom_xticklabels:
                    tick_labels = self.custom_xticklabels
                elif scenario_groupby == "Year-Scenario":
                    tick_labels = [
                        x.split(":")[0] for x in detailed_gen_cost.index
                    ]
                else:
                    tick_labels = detailed_gen_cost.index
                
                mplt.barplot(
                    detailed_gen_cost,
                    stacked = True,
                    custom_tick_labels=tick_labels, 
                    sub_pos=i
                )
                axs[i].axhline(y=0, color = "grey", linewidth = 0.5, linestyle = "--")

                # Add net cost line
                for k, idx in enumerate(detailed_gen_cost.index.unique()):
                    x = [
                        axs[i].patches[k].get_x(),
                        axs[i].patches[k].get_x() + axs[i].patches[k].get_width(),
                    ]
                    y_net = [net_cost.loc[idx]] * 2
                    axs[i].plot(x, y_net, c="black", linewidth = 1.5, label = "Net Cost")
                
                if scenario_groupby == "Year-Scenario":
                    axs[i].set_xlabel(scenario)
                    ylabel = "Annual Cost (Million $)"
                else:
                    axs[i].set_xlabel("")
                    ylabel = "Cumulative NPV Cost (Million $)"

                #ax[i].margins(x=0.01)

            # Checks if gen_cost_out_chunks contains data,
            # if not skips zone and does not return a plot
            if not gen_cost_out_chunks:
                outputs[zone_input] = MissingZoneData()
                continue


            # Add facet labels
            if self.xlabels or self.ylabels:
                mplt.add_facet_labels(xlabels=self.xlabels, ylabels= self.ylabels)
            # Add legend
            mplt.add_legend(reverse_legend = True)
            # Remove extra axes
            mplt.remove_excess_axs(excess_axs, grid_size)
            # Add title
            if plot_data_settings["plot_title_as_region"]:
                mplt.add_main_title(zone_input)

            # Ylabel should change if there are facet labels, leave at 40 for now,
            # works for all values in spacing
            labelpad = 20
            plt.ylabel(
                ylabel, 
                color = "black",
                rotation = "vertical",
                labelpad=labelpad,
            )
            # Data table of values to return to main program
            detailed_gen_cost_out = pd.concat(gen_cost_out_chunks, axis=0, sort=False)
            Data_Table_Out = detailed_gen_cost_out.add_suffix(" (Million $)")
            outputs[zone_input] = {"fig": fig, "data_table": Data_Table_Out}
        return outputs
    
    def detailed_gen_cost(
        self,
        start_date_range: str = None,
        end_date_range: str = None,
        custom_data_file_path: Path = None,
        scenario_groupby: str = "Scenario",
        **_,
    ):
        """Creates stacked bar plot of total generation cost by cost type (fuel, emission, start cost etc.)

        Creates a more deatiled system cost plot.
        Each sceanrio is plotted as a separate bar.

        Args:
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
            dict: Dictionary containing the created plot and its data table.
        """
        outputs: dict = {}

        # List of properties needed by the plot, properties are a set of tuples and
        # contain 3 parts: required True/False, property name and scenarios required,
        # scenarios must be a list.
        properties = [
            (False, "generator_Fuel_Cost", self.Scenarios),
            (False, "generator_FOM_Cost", self.Scenarios),
            (False, "generator_VOM_Cost", self.Scenarios),
            (False, "generator_Start_and_Shutdown_Cost", self.Scenarios),
            (False, "generator_Reserves_VOM_Cost", self.Scenarios),
            (False, "generator_Emissions_Cost", self.Scenarios),
            (False, "generator_Fossil_Annualized_Build_Cost", self.Scenarios),
            (False, "generator_RE_Annualized_Build_Cost", self.Scenarios),
            (False, "generator_UoS_Cost", self.Scenarios),
            (False, "generator_Annualized_One_Time_Cost", self.Scenarios),
            (False, "batterie_Annualized_Build_Cost", self.Scenarios),
        ]

        column_dict = {
                        "generator_Fuel_Cost": "Fuel",
                        "generator_FOM_Cost": "FO&M",
                        "generator_VOM_Cost": "VO&M",
                        "generator_Start_and_Shutdown_Cost": "Start & Shutdown",
                        "generator_Reserves_VOM_Cost": "Reserves VO&M",
                        "generator_Emissions_Cost": "Emissions",
                        "generator_Fossil_Annualized_Build_Cost":"Non-Renewable Capacity",
                        "generator_RE_Annualized_Build_Cost":"Renewable Purchases",
                        "generator_UoS_Cost":"Production Tax Credit",
                        "generator_Annualized_One_Time_Cost":"Spur Line",
                        "batterie_Annualized_Build_Cost":"Annualized Storage Build",
        }

        if scenario_groupby == "Scenario":
            for i in range(len(properties)):
                column_dict[properties[i][1] + "_NPV"] = column_dict[properties[i][1]]
                del column_dict[properties[i][1]]
                properties[i] = (properties[i][0], properties[i][1] + "_NPV", properties[i][2])
                
        # Runs get_formatted_data within PlotDataStoreAndProcessor to populate PlotDataStoreAndProcessor dictionary
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)

        # Checks if all data required by plot is available, if 1 in list required data is missing
        if 1 in check_input_data:
            return MissingInputData()

        for zone_input in self.Zones:
            logger.info(f"Zone = {zone_input}")
            gen_cost_out_chunks = []

            for scenario in self.Scenarios:
                logger.info(f"Scenario = {scenario}")

                data_frames_lst = []
                for prop_name in properties:
                    df: pd.DataFrame = self[prop_name[1]].get(scenario)
                    if df.empty:
                        continue
                    else:
                        try:
                            df = df.xs(zone_input, level=self.AGG_BY)
                            df = df.groupby(["timestamp"]).sum()
                        except KeyError:
                            logger.warning(f"No Generators found in: {zone_input}")
                            break

                    if (prop_name[1] == "generator_VOM_Cost" or prop_name[1] == "generator_VOM_Cost_NPV"):
                        try:
                            df["values"].to_numpy()[df["values"].to_numpy() < 0] = 0
                        except:
                            df[0].to_numpy()[df[0].to_numpy() < 0] = 0
                    df = df.rename(columns={"values": prop_name[1],0: prop_name[1]})

                    data_frames_lst.append(df)

                detailed_gen_cost = pd.concat(data_frames_lst, axis=1).fillna(0)
                detailed_gen_cost = detailed_gen_cost.rename(
                    columns=column_dict
                )

                if pd.notna(start_date_range):
                    detailed_gen_cost = set_timestamp_date_range(
                        detailed_gen_cost, start_date_range, end_date_range
                    )
                    if detailed_gen_cost.empty is True:
                        logger.warning("No Generation in selected Date Range")
                        continue

                gen_cost_out_chunks.append(
                    self.year_scenario_grouper(
                        detailed_gen_cost, scenario, groupby=scenario_groupby
                    ).sum()
                )

            # Checks if gen_cost_out_chunks contains data,
            # if not skips zone and does not return a plot
            if not gen_cost_out_chunks:
                outputs[zone_input] = MissingZoneData()
                continue

            detailed_gen_cost_out = pd.concat(gen_cost_out_chunks, axis=0, sort=False)
            detailed_gen_cost_out = (
                detailed_gen_cost_out / 1000000
            )  # Convert cost to millions

            # Checks if detailed_gen_cost_out contains data, if not skips zone and does not return a plot
            if detailed_gen_cost_out.empty:
                outputs[zone_input] = MissingZoneData()
                continue

            if pd.notna(custom_data_file_path):
                detailed_gen_cost_out = self.insert_custom_data_columns(
                    detailed_gen_cost_out, custom_data_file_path
                )
            detailed_gen_cost_out["Non-Renewable Capacity"] = detailed_gen_cost_out["Non-Renewable Capacity"] + detailed_gen_cost_out["Annualized Storage Build"]
            detailed_gen_cost_out["Renewable Purchases"] = detailed_gen_cost_out["Renewable Purchases"] + detailed_gen_cost_out["Production Tax Credit"]
            detailed_gen_cost_out.drop(["Annualized Storage Build", "Production Tax Credit"], axis= "columns", inplace = True)
        
            # Deletes columns that are all 0
            detailed_gen_cost_out = detailed_gen_cost_out.loc[
                :, (detailed_gen_cost_out != 0).any(axis=0)
            ]

            #if 'FO&M Cost' in detailed_gen_cost_out.columns: detailed_gen_cost_out.drop(columns =['FO&M Cost'],inplace = True)
            # Data table of values to return to main program
            Data_Table_Out = detailed_gen_cost_out.add_suffix(" (Million $)")
            
            #new_order = ["NoNewRE", "Base", "RPS"]
            if scenario_groupby == "Scenario": 
                # Reorder scenarios for plotting
                #detailed_gen_cost_out = reorder_scenarios(detailed_gen_cost_out, new_order, scenario_groupby=scenario_groupby)
                net_cost = [detailed_gen_cost_out.copy().sum(axis=1)]
                detailed_gen_cost_out = [detailed_gen_cost_out.copy()]

                mplt = PlotLibrary(squeeze = False, ravel_axs=True)
 
            
            else: #scenario groupby = Year-Scenario
                # Create a facet plot per scenario
                mplt = PlotLibrary(ncols = len(self.Scenarios), sharey=True)
                mplt.add_facet_labels(xlabels = self.Scenarios)
                temp = detailed_gen_cost_out.copy()
                detailed_gen_cost_out = []
                net_cost = []
                for scen in self.Scenarios:
                    sub_df = temp[temp.index.str.endswith(scen)]
                    sub_df.index = sub_df.index.str.split(":").str[0]
                    detailed_gen_cost_out.append(sub_df)
                    net_cost.append(sub_df.sum(axis=1))


            fig, ax = mplt.get_figure()

            n=0
            while n < len(self.Scenarios):
                # Set x-tick labels
                if self.custom_xticklabels:
                    tick_labels = self.custom_xticklabels
                else:
                    tick_labels = detailed_gen_cost_out[n].index

                mplt.barplot(
                    detailed_gen_cost_out[n], sub_pos = n, stacked=True, custom_tick_labels=tick_labels
                )
                ax[n].axhline(y=0, color = "grey", linewidth = 0.5, linestyle = "--")


                # Add net cost line
                for i, scenario in enumerate(detailed_gen_cost_out[n].index.unique()):
                    x = [
                        ax[n].patches[i].get_x(),
                        ax[n].patches[i].get_x() + ax[n].patches[i].get_width(),
                    ]
                    y_net = [net_cost[n].loc[scenario]] * 2
                    ax[n].plot(x, y_net, c="black", linewidth = 1.5, label = "Net Cost")

                ax[n].margins(x=0.01)
                if scenario_groupby == "Scenario":
                    ax[n].set_ylabel(
                        "Cumulative NPV Cost (Million $)", color="black", rotation="vertical"
                    ) 
                    n = len(self.Scenarios)
                else:
                    ax[n].set_xlabel(self.Scenarios[n])
                    ax[n].set_ylabel(
                        "Annual Cost (Million $)", color="black", rotation="vertical"
                    )
                    n += 1
                    

            mplt.add_legend(reverse_legend=True)
            if plot_data_settings["plot_title_as_region"]:
                mplt.add_main_title(zone_input)


            #cost_totals = detailed_gen_cost_out.sum(axis=1)  # holds total of each bar

            # inserts values into bar stacks
            #for patch in ax.patches:
            #    width, height = patch.get_width(), patch.get_height()
            #    if height <= 2:
            #        continue
            #    x, y = patch.get_xy()
            #    ax.text(
            #        x + width / 2,
            #        y + height / 2,
            #        "{:,.0f}".format(height),
            #        horizontalalignment="center",
            #        verticalalignment="center",
            #        fontsize=12,
            #    )

            # inserts total bar value above each bar
            #for k, patch in enumerate(ax.patches):
            #    height = cost_totals[k]
            #    width = patch.get_width()
            #    x, y = patch.get_xy()
            #    ax.text(
            #        x + width / 2,
            #        y + height + 0.05 * max(ax.get_ylim()),
            #        "{:,.0f}".format(height),
            #        horizontalalignment="center",
            #        verticalalignment="center",
            #        fontsize=15,
            #        color="red",
            #    )
            #   if k >= len(cost_totals) - 1:
            #        break

            outputs[zone_input] = {"fig": fig, "data_table": Data_Table_Out}
        return outputs

    def sys_cost_type(
        self,
        start_date_range: str = None,
        end_date_range: str = None,
        custom_data_file_path: Path = None,
        scenario_groupby: str = "Scenario",
        **_,
    ):
        """Creates stacked bar plot of total generation cost by generator technology type.

        Another way to represent total generation cost, this time by tech type,
        i.e Coal, Gas, Hydro etc.
        Each sceanrio is plotted as a separate bar.

        Args:
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
            dict: Dictionary containing the created plot and its data table.
        """
        # Create Dictionary to hold Datframes for each scenario
        outputs: dict = {}

        # List of properties needed by the plot, properties are a set of tuples and
        # contain 3 parts: required True/False, property name and scenarios required,
        # scenarios must be a list.
        properties = [(True, "generator_Total_Generation_Cost", self.Scenarios)]

        # Runs get_formatted_data within PlotDataStoreAndProcessor to populate PlotDataStoreAndProcessor dictionary
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)

        # Checks if all data required by plot is available, if 1 in list required data is missing
        if 1 in check_input_data:
            return MissingInputData()

        for zone_input in self.Zones:
            gen_cost_out_chunks = []
            logger.info(f"Zone = {zone_input}")

            for scenario in self.Scenarios:
                logger.info(f"Scenario = {scenario}")

                gen_cost: pd.DataFrame = self["generator_Total_Generation_Cost"].get(
                    scenario
                )
                # Check if gen_cost contains zone_input, skips if not
                try:
                    gen_cost = gen_cost.xs(zone_input, level=self.AGG_BY)
                except KeyError:
                    logger.warning(f"No Generators found for : {zone_input}")
                    continue
                gen_cost = self.df_process_gen_inputs(gen_cost)

                if pd.notna(start_date_range):
                    gen_cost = set_timestamp_date_range(
                        gen_cost, start_date_range, end_date_range
                    )
                    if gen_cost.empty is True:
                        logger.warning("No generation in selected Date Range")
                        continue

                gen_cost_out_chunks.append(
                    self.year_scenario_grouper(
                        gen_cost, scenario, groupby=scenario_groupby
                    ).sum()
                )

            # Checks if gen_cost_out_chunks contains data,
            # if not skips zone and does not return a plot
            if not gen_cost_out_chunks:
                outputs[zone_input] = MissingZoneData()
                continue

            total_systems_cost_out = pd.concat(
                gen_cost_out_chunks, axis=0, sort=False
            ).fillna(0)
            total_systems_cost_out = (
                total_systems_cost_out / 1000000
            )  # Convert to millions
            total_systems_cost_out = total_systems_cost_out.loc[
                :, (total_systems_cost_out != 0).any(axis=0)
            ]

            # Checks if total_systems_cost_out contains data,
            # if not skips zone and does not return a plot
            if total_systems_cost_out.empty:
                outputs[zone_input] = MissingZoneData()
                continue

            if pd.notna(custom_data_file_path):
                total_systems_cost_out = self.insert_custom_data_columns(
                    total_systems_cost_out, custom_data_file_path
                )
            # Data table of values to return to main program
            Data_Table_Out = total_systems_cost_out.add_suffix(" (Million $)")

            mplt = PlotLibrary()
            fig, ax = mplt.get_figure()

            # Set x-tick labels
            if self.custom_xticklabels:
                tick_labels = self.custom_xticklabels
            else:
                tick_labels = total_systems_cost_out.index

            mplt.barplot(
                total_systems_cost_out,
                color=self.marmot_color_dict,
                stacked=True,
                custom_tick_labels=tick_labels,
            )

            ax.set_ylabel(
                "Total System Cost (Million $)", color="black", rotation="vertical"
            )
            ax.margins(x=0.01)

            mplt.add_legend(reverse_legend=True)
            if plot_data_settings["plot_title_as_region"]:
                mplt.add_main_title(zone_input)

            outputs[zone_input] = {"fig": fig, "data_table": Data_Table_Out}
        return outputs

    def sys_cost_diff(
        self,
        start_date_range: str = None,
        end_date_range: str = None,
        scenario_groupby: str = "Scenario",
        **_,
    ):
        """Creates stacked barplots of Total Generation Cost, Total Capital Cost, and Cost of Unserved Energy 
        relative to a base scenario.

        Barplots show the change in total generation cost relative to a base scenario.
        The default is to compare against the first scenario provided in the inputs list.
        Plot only shows totals and is NOT broken down into technology or cost type specific values beyond capital vs operational.
        Each scenario is plotted as a separate bar.

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
        if self.AGG_BY == "zone":
            agg = "zone"
        else:
            agg = "region"

        outputs: dict = {}

        # List of properties needed by the plot, properties are a set of tuples and
        # contain 3 parts: required True/False, property name and scenarios required,
        # scenarios must be a list.
        # make sure to start this list with a required property
        properties = [ 
            (False, "generator_Total_Generation_Cost", self.Scenarios), 
            (False, "generator_Build_Cost", self.Scenarios), 
            (False, f"{agg}_Cost_Unserved_Energy", self.Scenarios),
        ]

        # Runs get_formatted_data within PlotDataStoreAndProcessor to populate PlotDataStoreAndProcessor dictionary
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)

        # Checks if all data required by plot is available, if 1 in list required data is missing
        if 1 in check_input_data:
            return MissingInputData()

        for zone_input in self.Zones:
            system_cost_chunk = []
            logger.info(f"Zone = {zone_input}")

            for scenario in self.Scenarios:
                logger.info(f"Scenario = {scenario}")

                data_frames_lst = []

                for prop_name in properties:
                    df: pd.DataFrame = self[prop_name[1]].get(scenario)
                    if df.empty:
                        continue
                    else:
                        try:
                            df = df.xs(zone_input, level=self.AGG_BY)
                            df = df.groupby("timestamp").sum()
                        except:
                            logger.warning(f"No Generators found in: {zone_input} {prop_name[1]}")
                            break
                    
                    if prop_name[1] == "generator_VOM_Cost":
                        try:
                            df["values"].to_numpy()[df["values"].to_numpy() < 0] = 0
                        except:
                            df[0].to_numpy()[df[0].to_numpy() < 0] = 0
                    df = df.rename(columns={"values": prop_name[1], 0: prop_name[1]})
                    data_frames_lst.append(df)

                sys_gen_cost = pd.concat(data_frames_lst, axis=1).fillna(0)
                sys_gen_cost = sys_gen_cost.rename(
                    columns={
                        "generator_Total_Generation_Cost":"Total Operational Cost",
                        "generator_Build_Cost":"Generator Build Cost",
                        f"{agg}_Cost_Unserved_Energy":"Cost of Unserved Energy",
                    }
                )
                

                if pd.notna(start_date_range):
                    sys_gen_cost = set_timestamp_date_range(
                        sys_gen_cost,
                        start_date_range,
                        end_date_range,
                    )
                    
                    if sys_gen_cost.empty is True:
                        logger.warning("No generation in selected Date Range")
                        continue
                
                system_cost_chunk.append(
                    self.year_scenario_grouper(
                        sys_gen_cost, scenario, groupby=scenario_groupby
                    ).sum()
                )

            # Checks if total_cost_chunk contains data, if not skips zone and does not return a plot
            if not system_cost_chunk:
                outputs[zone_input] = MissingZoneData()
                continue

            try:
                # Change to a diff on first scenario
                # Convert cost to millions
                #scen_base = total_systems_cost_out.index[0]
                scen_base = system_cost_chunk[0] / 1000000
                diff_system_cost_chunk = []
                for scen in system_cost_chunk[1:]:
                    scen = scen/1000000
                    diff_scen = scen.sub(scen_base.values, axis='columns')
                    diff_system_cost_chunk.append(diff_scen)
                total_systems_cost_out = pd.concat(diff_system_cost_chunk, axis=0, sort=False)

            except KeyError:
                outputs[zone_input] = MissingZoneData()
                continue
            #total_systems_cost_out.drop(scen_base, inplace=True)  # Drop base entry

            # Checks if total_systems_cost_out contains data, if not skips zone and does not return a plot
            if total_systems_cost_out.empty:
                outputs[zone_input] = MissingZoneData()
                continue

            # Data table of values to return to main program
            Data_Table_Out = total_systems_cost_out
            Data_Table_Out = Data_Table_Out.add_suffix(" (Million $)")

            mplt = PlotLibrary()
            fig, ax = mplt.get_figure()

            mplt.barplot(total_systems_cost_out, stacked=True)

            ax.axhline(y=0, color="black")
            ax.set_ylabel(
                f"Generation Cost Change (Million $) \n relative to {self.Scenarios[0]} Scenario",
                color="black",
                rotation="vertical",
            )
            ax.margins(x=0.01)
            # plt.ylim((0,600))
            mplt.add_legend()

            if plot_data_settings["plot_title_as_region"]:
                mplt.add_main_title(zone_input)
            fig.tight_layout()

            outputs[zone_input] = {"fig": fig, "data_table": Data_Table_Out}
        return outputs

    def sys_cost_type_diff(
        self,
        start_date_range: str = None,
        end_date_range: str = None,
        scenario_groupby: str = "Scenario",
        **_,
    ):
        """Creates stacked barplots of Total Generation Cost by generator technology type relative to a base scenario.

        Barplots show the change in total total generation cost relative to a base scenario.
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
        properties = [(True, "generator_Total_Generation_Cost", self.Scenarios)]

        # Runs get_formatted_data within PlotDataStoreAndProcessor to populate PlotDataStoreAndProcessor dictionary
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)
        # Checks if all data required by plot is available, if 1 in list required data is missing
        if 1 in check_input_data:
            return MissingInputData()

        for zone_input in self.Zones:
            gen_cost_out_chunks = []
            logger.info(f"Zone = {zone_input}")

            for scenario in self.Scenarios:
                logger.info(f"Scenario = {scenario}")

                gen_cost = self["generator_Total_Generation_Cost"].get(scenario)

                try:
                    gen_cost = gen_cost.xs(zone_input, level=self.AGG_BY)
                except KeyError:
                    logger.warning(f"No Generators found for : {zone_input}")
                    continue

                gen_cost = self.df_process_gen_inputs(gen_cost)

                if pd.notna(start_date_range):
                    gen_cost = set_timestamp_date_range(
                        gen_cost, start_date_range, end_date_range
                    )
                    if gen_cost.empty is True:
                        logger.warning("No generation in selected Date Range")
                        continue

                gen_cost_out_chunks.append(
                    self.year_scenario_grouper(
                        gen_cost, scenario, groupby=scenario_groupby
                    ).sum()
                )

            # Checks if gen_cost_out_chunks contains data,
            # if not skips zone and does not return a plot
            if not gen_cost_out_chunks:
                outputs[zone_input] = MissingZoneData()
                continue

            total_systems_cost_out = pd.concat(
                gen_cost_out_chunks, axis=0, sort=False
            ).fillna(0)
            total_systems_cost_out = (
                total_systems_cost_out / 1000000
            )  # Convert to millions
            total_systems_cost_out = total_systems_cost_out.loc[
                :, (total_systems_cost_out != 0).any(axis=0)
            ]
            # Ensures region has generation, else skips
            try:
                # Change to a diff on first scenario
                scen_base = total_systems_cost_out.index[0]
                total_systems_cost_out = (
                    total_systems_cost_out - total_systems_cost_out.xs(scen_base)
                )
            except KeyError:
                outputs[zone_input] = MissingZoneData()
                continue
            total_systems_cost_out.drop(scen_base, inplace=True)  # Drop base entry

            # Checks if total_systems_cost_out contains data,
            # if not skips zone and does not return a plot
            if total_systems_cost_out.empty == True:
                outputs[zone_input] = MissingZoneData()
                continue

            # Data table of values to return to main program
            Data_Table_Out = total_systems_cost_out.add_suffix(" (Million $)")

            mplt = PlotLibrary()
            fig, ax = mplt.get_figure()

            mplt.barplot(
                total_systems_cost_out, color=self.marmot_color_dict, stacked=True
            )

            ax.axhline(y=0)
            ax.set_ylabel(
                f"Generation Cost Change (Million $) \n relative to {scen_base}",
                color="black",
                rotation="vertical",
            )
            ax.margins(x=0.01)
            # plt.ylim((0,600))

            mplt.add_legend()
            if plot_data_settings["plot_title_as_region"]:
                mplt.add_main_title(zone_input)

            outputs[zone_input] = {"fig": fig, "data_table": Data_Table_Out}
        return outputs

    def detailed_gen_cost_diff(
        self,
        start_date_range: str = None,
        end_date_range: str = None,
        scenario_groupby: str = "Scenario",
        **_,
    ):
        """Creates stacked barplots of Total Generation Cost by by cost type
        (fuel, emission, start cost etc.) relative to a base scenario each year.

        Barplots show the change in total total generation cost relative to a base scenario.
        The default is to comapre against the first scenario provided in the inputs list.
        Each sceanrio is plotted as a separate bar if scenario_groupby = Scenario.
        Each scenario is plotted in its own facet plot if scenario_groupby = Year-Scenario.

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
        properties = [
            (False, "generator_Fuel_Cost", self.Scenarios),
            (False, "generator_FOM_Cost", self.Scenarios),
            (False, "generator_VOM_Cost", self.Scenarios),
            (False, "generator_Start_and_Shutdown_Cost", self.Scenarios),
            (False, "generator_Reserves_VOM_Cost", self.Scenarios),
            (False, "generator_Emissions_Cost", self.Scenarios),
            (False, "generator_Fossil_Annualized_Build_Cost", self.Scenarios),
            (False, "generator_RE_Annualized_Build_Cost", self.Scenarios),
            (False, "generator_UoS_Cost", self.Scenarios),
            (False, "generator_Annualized_One_Time_Cost", self.Scenarios),
            (False, "batterie_Annualized_Build_Cost", self.Scenarios),
        ]

        column_dict = {
                        "generator_Fuel_Cost": "Fuel",
                        "generator_FOM_Cost": "FO&M",
                        "generator_VOM_Cost": "VO&M",
                        "generator_Start_and_Shutdown_Cost": "Start & Shutdown",
                        "generator_Reserves_VOM_Cost": "Reserves VO&M",
                        "generator_Emissions_Cost": "Emissions",
                        "generator_Fossil_Annualized_Build_Cost":"Non-Renewable Capacity",
                        "generator_RE_Annualized_Build_Cost":"Renewable Purchases",
                        "generator_UoS_Cost":"Production Tax Credit",
                        "generator_Annualized_One_Time_Cost":"Spur Line",
                        "batterie_Annualized_Build_Cost":"Annualized Storage Build",
        }

        if scenario_groupby == "Scenario":
            for i in range(len(properties)):
                column_dict[properties[i][1] + "_NPV"] = column_dict[properties[i][1]]
                del column_dict[properties[i][1]]
                properties[i] = (properties[i][0], properties[i][1] + "_NPV", properties[i][2])

        # Runs get_formatted_data within PlotDataStoreAndProcessor to populate PlotDataStoreAndProcessor dictionary
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)

        # Checks if all data required by plot is available, if 1 in list required data is missing
        if 1 in check_input_data:
            return MissingInputData()

        for zone_input in self.Zones:
            logger.info(f"Zone = {zone_input}")
            gen_cost_out_chunks = []

            for scenario in self.Scenarios:
                logger.info(f"Scenario = {scenario}")

                data_frames_lst = []
                for prop_name in properties:
                    df: pd.DataFrame = self[prop_name[1]].get(scenario)
                    if df.empty:
                        continue
                    else:
                        try:
                            df = df.xs(zone_input, level=self.AGG_BY)
                            df = df.groupby("timestamp").sum()
                        except KeyError:
                            logger.warning(f"No Generators found in: {zone_input}")
                            break

                    if (prop_name[1] == "generator_VOM_Cost" or prop_name[1] == "generator_VOM_Cost_NPV"):
                        try:
                            df["values"].to_numpy()[df["values"].to_numpy() < 0] = 0
                        except:
                            df[0].to_numpy()[df[0].to_numpy() < 0] = 0
                    df = df.rename(columns={"values": prop_name[1], 0: prop_name[1]})

                    data_frames_lst.append(df)

                detailed_gen_cost = pd.concat(data_frames_lst, axis=1).fillna(0)
                detailed_gen_cost = detailed_gen_cost.rename(
                    columns=column_dict
                )

                if pd.notna(start_date_range):
                    detailed_gen_cost = set_timestamp_date_range(
                        detailed_gen_cost, start_date_range, end_date_range
                    )
                    if detailed_gen_cost.empty is True:
                        logger.warning("No Generation in selected Date Range")
                        continue

                gen_cost_out_chunks.append(
                    self.year_scenario_grouper(
                        detailed_gen_cost, scenario, groupby=scenario_groupby
                    ).sum()
                )

            # Checks if gen_cost_out_chunks contains data, if not skips zone and does
            # not return a plot
            if not gen_cost_out_chunks:
                outputs[zone_input] = MissingZoneData()
                continue
            
            try:
                # Change to a diff on first scenario
                # Convert cost to millions
                #scen_base = total_systems_cost_out.index[0]
                scen_base = gen_cost_out_chunks[0] / 1000000
                diff_gen_cost_out_chunks = []
                for scen in gen_cost_out_chunks[1:]:
                    scen = scen/1000000
                    diff_scen = scen.sub(scen_base.values, axis='columns')
                    diff_scen = -1*diff_scen
                    diff_gen_cost_out_chunks.append(diff_scen)
                detailed_gen_cost_out = pd.concat(diff_gen_cost_out_chunks, axis=0, sort=False)

            except KeyError:
                outputs[zone_input] = MissingZoneData()
                continue

            # TODO: Add $ unit conversion.

            # Ensures region has generation, else skips
            # Drop base entry
            # detailed_gen_cost_out.drop(scen_base, inplace=True)



            # Checks if detailed_gen_cost_out contains data,
            # if not skips zone and does not return a plot
            if detailed_gen_cost_out.empty == True:
                outputs[zone_input] = MissingZoneData()
                continue

            detailed_gen_cost_out["Non-Renewable Capacity"] = detailed_gen_cost_out["Non-Renewable Capacity"] + detailed_gen_cost_out["Annualized Storage Build"]
            detailed_gen_cost_out["Renewable Purchases"] = detailed_gen_cost_out["Renewable Purchases"] + detailed_gen_cost_out["Production Tax Credit"]
            detailed_gen_cost_out.drop(["Annualized Storage Build", "Production Tax Credit"], axis= "columns", inplace = True)

            # Deletes columns that are all 0
            detailed_gen_cost_out = detailed_gen_cost_out.loc[
                :, (detailed_gen_cost_out != 0).any(axis=0)
            ]

            # Data table of values to return to main program
            Data_Table_Out = detailed_gen_cost_out.add_suffix(" (Million $)")

            #new_order = ["NoNewRE","RPS"]
            if scenario_groupby == "Scenario":
                # Reorder scenarios for plotting
                #detailed_gen_cost_out = reorder_scenarios(detailed_gen_cost_out, new_order=new_order, scenario_groupby=scenario_groupby)
                net_cost = [detailed_gen_cost_out.copy().sum(axis=1)]
                detailed_gen_cost_out = [detailed_gen_cost_out.copy()]
                mplt = PlotLibrary(squeeze = False, ravel_axs = True)
            else: #scenario_groupby == "Year-Scenario"
                # Create a facet plot per scenario instead
                mplt = PlotLibrary(ncols = len(self.Scenarios)-1, sharey = True, squeeze=False, ravel_axs = True)
                temp = detailed_gen_cost_out.copy()
                detailed_gen_cost_out = []
                net_cost = []
                for scen in self.Scenarios[1:]:
                    sub_df = temp[temp.index.str.endswith(scen)]
                    sub_df.index = sub_df.index.str.split(":").str[0]
                    detailed_gen_cost_out.append(sub_df)
                    net_cost.append(sub_df.sum(axis=1))

            fig, ax = mplt.get_figure()
            
            n = 0
            while n < len(self.Scenarios)-1:

                mplt.barplot(detailed_gen_cost_out[n], sub_pos = n, stacked=True)

                ax[n].axhline(y=0, linewidth=0.5, linestyle="--", color="grey")

                ax[n].margins(x=0.01)

                # Add net cost line.
                for i, scenario in enumerate(detailed_gen_cost_out[n].index.unique()):
                    x = [
                        ax[n].patches[i].get_x(),
                        ax[n].patches[i].get_x() + ax[n].patches[i].get_width(),
                    ]
                    y_net = [net_cost[n].loc[scenario]] * 2
                    ax[n].plot(x, y_net, c="black", linewidth=1.5, label="Net Cost Change")
                
                if scenario_groupby == "Scenario":
                    #n = len(self.Scenarios)-1
                    ax[n].set_ylabel(
                        f"Cumulative NPV Savings relative to\n {self.Scenarios[0]} Scenario (Million $)", color="black", rotation="vertical"
                    ) 
                    n = len(self.Scenarios)
                else:
                    ax[n].set_xlabel(self.Scenarios[n+1])
                    ax[n].set_ylabel(
                        f"Annual Savings relative to \n {self.Scenarios[0]} Scenario (Million $)", color="black", rotation="vertical"
                    )
                    n += 1

            mplt.add_legend()
                

            if plot_data_settings["plot_title_as_region"]:
                mplt.add_main_title(zone_input)

            outputs[zone_input] = {"fig": fig, "data_table": Data_Table_Out}
        return outputs

    def npv_lineplot(
        self,
        start_date_range: str = None,
        end_date_range: str = None,
        custom_data_file_path: Path = None,
        scenario_groupby: str = "Scenario",
        **_,
    ):
        """Creates line plot of cumulative NPV cost with one line per scenario

        Args:
            start_date_range (str, optional): Defines a start date at which to represent data from.
                Defaults to None.
            end_date_range (str, optional): Defines a end date at which to represent data to.
                Defaults to None.
            custom_data_file_path (Path, optional): Path to custom data file to concat extra
                data. Index and column format should be consistent with output data csv.
            scenario_groupby (str, optional): Specifies whether to group data by Scenario
                or Year-Sceanrio. Works best when grouping by Year-Scenario

                .. versionadded:: 0.10.0

        Returns:
            dict: Dictionary containing the created plot and its data table.
        """
        outputs: dict = {}

        # List of properties needed by the plot, properties are a set of tuples and
        # contain 3 parts: required True/False, property name and scenarios required,
        # scenarios must be a list.
        properties = [
            (False, "generator_Fuel_Cost_NPV", self.Scenarios),
            (False, "generator_FOM_Cost_NPV", self.Scenarios),
            (False, "generator_VOM_Cost_NPV", self.Scenarios),
            (False, "generator_Start_and_Shutdown_Cost_NPV", self.Scenarios),
            (False, "generator_Reserves_VOM_Cost_NPV", self.Scenarios),
            (False, "generator_Emissions_Cost_NPV", self.Scenarios),
            (False, "generator_Annualized_Build_Cost_NPV", self.Scenarios),
            (False, "generator_UoS_Cost_NPV", self.Scenarios),
            (False, "generator_Annualized_One_Time_Cost_NPV", self.Scenarios),
            (False, "batterie_Annualized_Build_Cost_NPV", self.Scenarios),
        ]
                
        # Runs get_formatted_data within PlotDataStoreAndProcessor to populate PlotDataStoreAndProcessor dictionary
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)

        # Checks if all data required by plot is available, if 1 in list required data is missing
        if 1 in check_input_data:
            return MissingInputData()

        for zone_input in self.Zones:
            logger.info(f"Zone = {zone_input}")
            gen_cost_out_chunks = []

            for scenario in self.Scenarios:
                logger.info(f"Scenario = {scenario}")

                data_frames_lst = []
                for prop_name in properties:
                    df: pd.DataFrame = self[prop_name[1]].get(scenario)
                    if df.empty:
                        continue
                    else:
                        try:
                            df = df.xs(zone_input, level=self.AGG_BY)
                            df = df.groupby(["timestamp"]).sum()
                        except KeyError:
                            logger.warning(f"No Generators found in: {zone_input}")
                            break

                    if (prop_name[1] == "generator_VOM_Cost" or prop_name[1] == "generator_VOM_Cost_NPV"):
                        try:
                            df["values"].to_numpy()[df["values"].to_numpy() < 0] = 0
                        except:
                            df[0].to_numpy()[df[0].to_numpy() < 0] = 0
                    df = df.rename(columns={"values": prop_name[1],0: prop_name[1]})

                    data_frames_lst.append(df)

                detailed_gen_cost = pd.concat(data_frames_lst, axis=1).fillna(0)

                if pd.notna(start_date_range):
                    detailed_gen_cost = set_timestamp_date_range(
                        detailed_gen_cost, start_date_range, end_date_range
                    )
                    if detailed_gen_cost.empty is True:
                        logger.warning("No Generation in selected Date Range")
                        continue
                
                grouped = self.year_scenario_grouper(
                            detailed_gen_cost, scenario, groupby=scenario_groupby
                        ).sum()
                if scenario_groupby == "Year-Scenario":
                    grouped.index = grouped.index.str.split(":").str[0]
                    grouped.index.names = ["Year"]
                
                grouped[scenario] = grouped[list(grouped.columns)].sum(axis=1)
                grouped = grouped[[scenario]]

                gen_cost_out_chunks.append(grouped)

            # Checks if gen_cost_out_chunks contains data,
            # if not skips zone and does not return a plot
            if not gen_cost_out_chunks:
                outputs[zone_input] = MissingZoneData()
                continue

            # FIX INDEX TO JUST YEAR

            detailed_gen_cost_out = pd.concat(gen_cost_out_chunks, axis=1, sort=False)
            detailed_gen_cost_out = (
                detailed_gen_cost_out / 1000000
            )  # Convert cost to millions

            # Checks if detailed_gen_cost_out contains data, if not skips zone and does not return a plot
            if detailed_gen_cost_out.empty:
                outputs[zone_input] = MissingZoneData()
                continue

            # Find the cumulative sum along each column
            detailed_gen_cost_out = detailed_gen_cost_out.cumsum()

            if pd.notna(custom_data_file_path):
                detailed_gen_cost_out = self.insert_custom_data_columns(
                    detailed_gen_cost_out, custom_data_file_path
                )

            # Data table of values to return to main program
            Data_Table_Out = detailed_gen_cost_out.add_suffix(" (Million $)")

            mplt = PlotLibrary()
            fig, ax = mplt.get_figure()

            # Set x-tick labels
            if self.custom_xticklabels:
                tick_labels = self.custom_xticklabels
            else:
                tick_labels =detailed_gen_cost_out.index
        
            mplt.multilineplot(
                detailed_gen_cost_out,
                custom_tick_labels=tick_labels,
            )
            # Add legend
            mplt.add_legend()
            # Add title
            if plot_data_settings["plot_title_as_region"]:
                mplt.add_main_title(zone_input)

            # Ylabel should change if there are facet labels, leave at 40 for now,
            # works for all values in spacing
            ax.set_ylabel(
                f"Cumulative NPV (Million $)",
                color="black",
                rotation="vertical",
            )

            outputs[zone_input] = {"fig": fig, "data_table": Data_Table_Out}
        return outputs