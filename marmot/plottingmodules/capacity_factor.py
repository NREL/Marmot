# -*- coding: utf-8 -*-
"""Generator capacity factor plots .

This module contain methods that are related to the capacity factor 
of generators and average output plots 
"""

import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

import marmot.utils.mconfig as mconfig
from marmot.plottingmodules.plotutils.plot_data_helper import (
    GenCategories,
    PlotDataStoreAndProcessor,
)
from marmot.plottingmodules.plotutils.plot_exceptions import (
    MissingInputData,
    MissingZoneData,
)
from marmot.plottingmodules.plotutils.plot_library import PlotLibrary
from marmot.plottingmodules.plotutils.styles import ColorList
from marmot.plottingmodules.plotutils.timeseries_modifiers import (
    set_timestamp_date_range,
)

logger = logging.getLogger("plotter." + __name__)
plot_data_settings: dict = mconfig.parser("plot_data")
xdimension: int = mconfig.parser("figure_size", "xdimension")
ydimension: int = mconfig.parser("figure_size", "ydimension")


class CapacityFactor(PlotDataStoreAndProcessor):
    """Generator capacity factor plots.

    The capacity_factor.py module contain methods that are
    related to the capacity factor of generators.

    CapacityFactor inherits from the PlotDataStoreAndProcessor class to
    assist in creating figures.
    """

    def __init__(
        self,
        Zones: List[str],
        Scenarios: List[str],
        AGG_BY: str,
        ordered_gen: List[str],
        marmot_solutions_folder: Path,
        gen_categories: GenCategories = GenCategories(),
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
            gen_categories (GenCategories): Instance of GenCategories class, groups generator technologies
                into defined categories.
                Deafults to GenCategories.
            color_list (list, optional): List of colors to apply to non-gen plots.
                Defaults to ColorList().colors.
        """
        # Instantiation of PlotDataStoreAndProcessor
        super().__init__(AGG_BY, ordered_gen, marmot_solutions_folder, **kwargs)

        self.Zones = Zones
        self.Scenarios = Scenarios
        self.gen_categories = gen_categories
        self.color_list = color_list

    def avg_output_when_committed(
        self,
        start_date_range: str = None,
        end_date_range: str = None,
        scenario_groupby: str = "Scenario",
        **_,
    ):
        """Creates barplots of the percentage average generation output when committed
        by technology type.

        Each scenario is plotted by a different colored grouped bar.

        Args:
            start_date_range (str, optional): Defines a start date at which to represent
                data from.
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
            dict: dictionary containing the created plot and its data table.
        """
        outputs: dict = {}

        # List of properties needed by the plot, properties are a set of tuples and
        # contain 3 parts: required True/False, property name and scenarios required,
        # scenarios must be a list.
        properties = [
            (True, "generator_Generation", self.Scenarios),
            (True, "generator_Installed_Capacity", self.Scenarios),
        ]

        # Runs get_formatted_data within PlotDataStoreAndProcessor to populate PlotDataStoreAndProcessor
        # dictionary with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)

        if 1 in check_input_data:
            return MissingInputData()

        for zone_input in self.Zones:
            cf_chunks = []
            logger.info(f"{self.AGG_BY} = {zone_input}")

            for scenario in self.Scenarios:
                logger.info(f"Scenario = {str(scenario)}")

                Gen: pd.DataFrame = self["generator_Generation"].get(scenario)
                try:
                    Gen = Gen.xs(zone_input, level=self.AGG_BY)
                except KeyError:
                    logger.warning(f"No data in {zone_input}")
                    continue
                Gen = Gen.reset_index()
                Gen = self.rename_gen_techs(Gen)
                Gen.tech = Gen.tech.astype("category")
                Gen.tech = Gen.tech.cat.set_categories(self.ordered_gen)
                Gen = Gen[Gen["tech"].isin(self.thermal_gen_cat)]
                Gen.set_index("timestamp", inplace=True)
                Gen = Gen.rename(columns={0: "Output (MWh)"})

                Cap: pd.DataFrame = self["generator_Installed_Capacity"].get(scenario)
                Cap = Cap.xs(zone_input, level=self.AGG_BY)
                Cap = Cap.rename(columns={0: "Installed Capacity (MW)"})

                #Totable = gen.merga(cap,on['gen_name','region','zone','State','country'],how='left')

                if pd.notna(start_date_range):
                    Cap, Gen = set_timestamp_date_range(
                        [Cap, Gen], start_date_range, end_date_range
                    )
                    if Gen.empty is True:
                        logger.warning("No data in selected Date Range")
                        continue

                Gen["year"] = Gen.index.year.astype(str)
                Cap["year"] = Cap.index.get_level_values("timestamp").year.astype(str)
                Gen = Gen.reset_index()
                Gen = pd.merge(Gen, Cap, on=["gen_name", "year"])
                Gen.set_index("timestamp", inplace=True)

                if scenario_groupby == "Year-Scenario":
                    Gen["Scenario"] = Gen.index.year.astype(str) + f"_{scenario}"
                else:
                    Gen["Scenario"] = scenario

                year_scen = Gen["Scenario"].unique()
                for scen in year_scen:
                    Gen_scen = Gen.loc[Gen["Scenario"] == scen]
                    # Calculate CF individually for each plant,
                    # since we need to take out all zero rows.
                    tech_names = Gen_scen.sort_values(["tech"])["tech"].unique()
                    CF = pd.DataFrame(columns=tech_names, index=[scen])
                    for tech_name in tech_names:
                        stt = Gen_scen.loc[Gen_scen["tech"] == tech_name]
                        if not all(stt["Output (MWh)"] == 0):

                            gen_names = stt["gen_name"].unique()
                            cfs = []
                            caps = []
                            for gen in gen_names:
                                sgt = stt.loc[stt["gen_name"] == gen]
                                if not all(sgt["Output (MWh)"] == 0):
                                    # Calculates interval step to correct for MWh of generation
                                    time_delta = sgt.index[1] - sgt.index[0]
                                    duration = sgt.index[len(sgt) - 1] - sgt.index[0]
                                    duration = (
                                        duration + time_delta
                                    )  # Account for last timestep.
                                    # Finds intervals in 60 minute period
                                    interval_count = 60 / (
                                        time_delta / np.timedelta64(1, "m")
                                    )
                                    # Get length of time series in hours for CF calculation.
                                    duration_hours = min(
                                        8760, duration / np.timedelta64(1, "h")
                                    )
                                    # Remove time intervals when output is zero.
                                    sgt = sgt[sgt["Output (MWh)"] != 0]
                                    total_gen = (
                                        sgt["Output (MWh)"].sum() / interval_count
                                    )
                                    cap = sgt["Installed Capacity (MW)"].mean()
                                    # Calculate CF
                                    #ww changed from cf = total_gen / (cap * duration_hours) to cf = total_gen / (cap)
                                    cf = total_gen / (cap)
                                    cfs.append(cf)
                                    caps.append(cap)

                            # Find average "CF" (average output when committed)
                            # for this technology, weighted by capacity.
                            #ww changed from cf = np.average(cfs, weights=caps)
                            cf = np.average(cfs, weights=caps)
                            CF[tech_name] = cf
                            #ww added
                            ##PCC
                            #CF["CCS off"] = CF["CCS off"] - CF["CCS on"]
                            ##storage
                            CF["Gas-CC CCS-flex-storage off"] = CF["Gas-CC CCS-flex-storage off"] - CF["Gas-CC CCS-flex-storage on"]
                            ##DAC
                            #CF["DAC&CCS-off"] = CF["DAC&CCS-off"] - CF["DAC&CCS-on"]
                    cf_chunks.append(CF)

            if cf_chunks:
                CF_all_scenarios = pd.concat(cf_chunks)
            else:
                outputs[zone_input] = MissingZoneData()
                continue

            Data_Table_Out = CF_all_scenarios.T

            mplt = PlotLibrary()
            fig, ax = mplt.get_figure()

            mplt.barplot(
                CF_all_scenarios.T,
                color=self.color_list,
                custom_tick_labels=list(CF_all_scenarios.columns),
                #ytick_major_fmt="percent",
            )

            ax.set_ylabel(
                #ww changed:"Average Output When Committed", color="black", rotation="vertical" 
                "MIT Annual Operational Hours(h)", color="black", rotation="vertical"
            )

            if plot_data_settings["plot_title_as_region"]:
                mplt.add_main_title(zone_input)
            # Add legend
            mplt.add_legend()

            outputs[zone_input] = {"fig": fig, "data_table": Data_Table_Out}
        return outputs

    def cf(
        self,
        start_date_range: str = None,
        end_date_range: str = None,
        scenario_groupby: str = "Scenario",
        **_,
    ):
        """Creates barplots of generator capacity factors by technology type.

        Each scenario is plotted by a different colored grouped bar.

        Args:
            start_date_range (str, optional): Defines a start date at which to represent
                data from.
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
            dict: dictionary containing the created plot and its data table.
        """

        outputs: dict = {}

        # List of properties needed by the plot, properties are a set of tuples and
        # contain 3 parts: required True/False, property name and scenarios required,
        # scenarios must be a list.
        properties = [
            (True, "generator_Generation", self.Scenarios),
            (True, "generator_Installed_Capacity", self.Scenarios),
        ]

        # Runs get_formatted_data within PlotDataStoreAndProcessor to populate PlotDataStoreAndProcessor
        # dictionary with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)

        if 1 in check_input_data:
            return MissingInputData()

        for zone_input in self.Zones:
            cf_scen_chunks = []
            logger.info(f"{self.AGG_BY} = {zone_input}")

            for scenario in self.Scenarios:

                logger.info(f"Scenario = {str(scenario)}")
                Gen = self["generator_Generation"].get(scenario)
                try:  # Check for regions missing all generation.
                    Gen = Gen.xs(zone_input, level=self.AGG_BY)
                except KeyError:
                    logger.warning(f"No data in {zone_input}")
                    continue
                Gen = self.df_process_gen_inputs(Gen)

                Cap = self["generator_Installed_Capacity"].get(scenario)
                Cap = Cap.xs(zone_input, level=self.AGG_BY)
                Cap = self.df_process_gen_inputs(Cap)

                if pd.notna(start_date_range):
                    Cap, Gen = set_timestamp_date_range(
                        [Cap, Gen], start_date_range, end_date_range
                    )
                    if Gen.empty is True:
                        logger.warning("No data in selected Date Range")
                        continue

                # Calculates interval step to correct for MWh of generation
                time_delta = Gen.index[1] - Gen.index[0]
                duration = Gen.index[len(Gen) - 1] - Gen.index[0]
                duration = duration + time_delta  # Account for last timestep.
                # Finds intervals in 60 minute period
                interval_count: int = 60 / (time_delta / np.timedelta64(1, "m"))
                # Get length of time series in hours for CF calculation.
                duration_hours: int = min(8760, duration / np.timedelta64(1, "h"))

                Gen = Gen / interval_count

                Total_Gen = self.year_scenario_grouper(
                    Gen, scenario, groupby=scenario_groupby
                ).sum()
                Cap = self.year_scenario_grouper(
                    Cap, scenario, groupby=scenario_groupby
                ).sum()
                # Calculate CF
                #ww changed from  CF = Total_Gen / (Cap * duration_hours)
                CF = duration_hours
                cf_scen_chunks.append(CF)

            if cf_scen_chunks:
                CF_all_scenarios = pd.concat(cf_scen_chunks, axis=0, sort=False).T
                CF_all_scenarios = CF_all_scenarios.fillna(0, axis=0)
            else:
                outputs[zone_input] = MissingZoneData()
                continue

            Data_Table_Out = CF_all_scenarios.T

            mplt = PlotLibrary(figsize=(xdimension * 1.5, ydimension * 1.5))
            fig, ax = mplt.get_figure()

            mplt.barplot(
                CF_all_scenarios, color=self.color_list, ytick_major_fmt="percent"
            )

            ax.set_ylabel("Capacity Factor", color="black", rotation="vertical")
            # Add legend
            mplt.add_legend()
            # Add title
            if plot_data_settings["plot_title_as_region"]:
                mplt.add_main_title(zone_input)
            outputs[zone_input] = {"fig": fig, "data_table": Data_Table_Out}

        return outputs

    def time_at_min_gen(
        self,
        start_date_range: str = None,
        end_date_range: str = None,
        scenario_groupby: str = "Scenario",
        **_,
    ):
        """Creates barplots of generator percentage time at min-gen by technology type.

        Each scenario is plotted by a different colored grouped bar.

        Args:
            start_date_range (str, optional): Defines a start date at which to represent
                data from.
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
            dict: dictionary containing the created plot and its data table.
        """

        outputs: dict = {}

        # List of properties needed by the plot, properties are a set of tuples and
        # contain 3 parts: required True/False, property name and scenarios required,
        # scenarios must be a list.
        properties = [
            (True, "generator_Generation", self.Scenarios),
            (True, "generator_Installed_Capacity", self.Scenarios),
            (True, "generator_Hours_at_Minimum", self.Scenarios),
        ]

        # Runs get_formatted_data within PlotDataStoreAndProcessor to populate PlotDataStoreAndProcessor dictionary
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)

        if 1 in check_input_data:
            return MissingInputData()

        for zone_input in self.Zones:
            logger.info(f"{self.AGG_BY} = {zone_input}")

            time_at_min = pd.DataFrame()

            for scenario in self.Scenarios:
                logger.info(f"Scenario = {str(scenario)}")

                Min = self["generator_Hours_at_Minimum"].get(scenario)
                try:
                    Min = Min.xs(zone_input, level=self.AGG_BY)
                except KeyError:
                    continue
                Gen = self["generator_Generation"].get(scenario)
                try:  # Check for regions missing all generation.
                    Gen = Gen.xs(zone_input, level=self.AGG_BY)
                except KeyError:
                    logger.warning(f"No data in {zone_input}")
                    continue
                Cap = self["generator_Installed_Capacity"].get(scenario)
                Cap = Cap.xs(zone_input, level=self.AGG_BY)

                if pd.notna(start_date_range):
                    Min, Gen, Cap = set_timestamp_date_range(
                        [Min, Gen, Cap], start_date_range, end_date_range
                    )
                    if Gen.empty is True:
                        logger.warning("No data in selected Date Range")
                        continue

                Min = Min.reset_index()
                Min = Min.set_index("gen_name")
                Min = Min.rename(columns={0: "Hours at Minimum"})

                Gen = Gen.reset_index()
                Gen.tech = Gen.tech.astype("category")
                Gen.tech = Gen.tech.cat.set_categories(self.ordered_gen)
                Gen = Gen.rename(columns={0: "Output (MWh)"})
                Gen = Gen[~Gen["tech"].isin(self.vre_gen_cat)]
                Gen.index = Gen.timestamp

                Caps = Cap.groupby("gen_name").mean()
                Caps.reset_index()
                Caps = Caps.rename(columns={0: "Installed Capacity (MW)"})
                Min = pd.merge(Min, Caps, on="gen_name")

                # Find how many hours each generator was operating, for the denominator of the % time at min gen.
                # So remove all zero rows.
                Gen = Gen.loc[Gen["Output (MWh)"] != 0]
                online_gens = Gen.gen_name.unique()
                Min = Min.loc[online_gens]
                Min["hours_online"] = Gen.groupby("gen_name")["Output (MWh)"].count()
                Min["fraction_at_min"] = Min["Hours at Minimum"] / Min.hours_online

                tech_names = Min.tech.unique()
                time_at_min_individ = pd.DataFrame(columns=tech_names, index=[scenario])
                for tech_name in tech_names:
                    stt = Min.loc[Min["tech"] == tech_name]
                    wgts = stt["Installed Capacity (MW)"]
                    if wgts.sum() == 0:
                        wgts = pd.Series([1] * len(stt))
                    output = np.average(stt.fraction_at_min, weights=wgts)
                    time_at_min_individ[tech_name] = output

                time_at_min = time_at_min.append(time_at_min_individ)

            if time_at_min.empty == True:
                outputs[zone_input] = MissingZoneData()
                continue

            Data_Table_Out = time_at_min.T

            mplt = PlotLibrary(figsize=(xdimension * 1.5, ydimension * 1.5))
            fig, ax = mplt.get_figure()

            mplt.barplot(
                time_at_min.T,
                color=self.color_list,
                custom_tick_labels=list(time_at_min.columns),
                ytick_major_fmt="percent",
            )

            ax.set_ylabel(
                "Percentage of time online at minimum generation",
                color="black",
                rotation="vertical",
            )
            # Add legend
            mplt.add_legend()
            # Add title
            if plot_data_settings["plot_title_as_region"]:
                mplt.add_main_title(zone_input)

            outputs[zone_input] = {"fig": fig, "data_table": Data_Table_Out}
        return outputs
