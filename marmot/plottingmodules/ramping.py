# -*- coding: utf-8 -*-
"""Generator start and ramping plots.

This module creates bar plot of the total volume of generator starts in MW,GW,etc.

@author: Marty Schwarz
"""

import logging
import pandas as pd

import marmot.utils.mconfig as mconfig

from marmot.plottingmodules.plotutils.plot_library import PlotLibrary
from marmot.plottingmodules.plotutils.plot_data_helper import MPlotDataHelper
from marmot.plottingmodules.plotutils.plot_exceptions import (
    MissingInputData,
    MissingZoneData,
    UnderDevelopment,
)

logger = logging.getLogger("plotter." + __name__)
plot_data_settings = mconfig.parser("plot_data")


class Ramping(MPlotDataHelper):
    """Generator start and ramping plots.

    The ramping.py module contains methods that are
    related to the ramp periods of generators.

    Ramping inherits from the MPlotDataHelper class to assist
    in creating figures.
    """

    def __init__(self, **kwargs):
        """
        Args:
            **kwargs
                These parameters will be passed to the MPlotDataHelper 
                class.
        """

    def capacity_started(
        self,
        start_date_range: str = None,
        end_date_range: str = None,
        scenario_groupby: str = "Scenario",
        **_,
    ):
        """Creates bar plots of total thermal capacity started by technology type.

        Each sceanrio is plotted as a separate color grouped bar.

        Args:
            start_date_range (str, optional): Defines a start date at which to
                represent data from.
                Defaults to None.
            end_date_range (str, optional): Defines a end date at which to
                represent data to.
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

        # List of properties needed by the plot, properties are a set of
        # tuples and contain 3 parts:
        # required True/False, property name and scenarios required,
        # scenarios must be a list.
        properties = [
            (True, "generator_Generation", self.Scenarios),
            (True, "generator_Installed_Capacity", self.Scenarios),
        ]

        # Runs get_formatted_data within MPlotDataHelper to populate
        # MPlotDataHelper dictionary with all required properties,
        # returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)

        if 1 in check_input_data:
            return MissingInputData()

        for zone_input in self.Zones:
            logger.info(f"{self.AGG_BY} = {zone_input}")
            cap_stated_chunks = []

            for scenario in self.Scenarios:
                logger.info(f"Scenario = {str(scenario)}")

                Gen: pd.DataFrame = self["generator_Generation"].get(scenario)
                try:
                    Gen = Gen.xs(zone_input, level=self.AGG_BY)
                except KeyError:
                    logger.warning(f"No installed capacity in : {zone_input}")
                    break

                Cap: pd.DataFrame = self["generator_Installed_Capacity"].get(scenario)
                Cap = Cap.xs(zone_input, level=self.AGG_BY)

                if pd.notna(start_date_range):
                    Gen = self.set_timestamp_date_range(
                        Gen, start_date_range, end_date_range
                    )
                    if Gen.empty is True:
                        logger.warning("No Generation in selected Date Range")
                        continue

                Gen = Gen.reset_index()
                Gen["year"] = Gen.timestamp.dt.year
                Gen = self.rename_gen_techs(Gen)
                Gen.tech = Gen.tech.astype("category")
                Gen.tech = Gen.tech.cat.set_categories(self.ordered_gen)
                Gen = Gen.rename(columns={0: "Output (MWh)"})
                # We are only interested in thermal starts/stops.
                Gen = Gen[Gen["tech"].isin(self.thermal_gen_cat)]

                Cap = Cap.reset_index()
                Cap["year"] = Cap.timestamp.dt.year
                Cap = self.rename_gen_techs(Cap)
                Cap = Cap.drop(columns=["timestamp", "units"])
                Cap = Cap.rename(columns={0: "Installed Capacity (MW)"})
                gen_cap = Gen.merge(Cap, on=["year", "tech", "gen_name"])
                gen_cap = gen_cap.set_index("timestamp")

                gen_cap = self.year_scenario_grouper(
                    gen_cap,
                    scenario,
                    groupby=scenario_groupby,
                    additional_groups=["timestamp", "tech", "gen_name"],
                ).sum()
                unique_idx = list(gen_cap.index.get_level_values("Scenario").unique())
                cap_started_df = pd.DataFrame(
                    columns=gen_cap.index.get_level_values("tech").unique(),
                    index=unique_idx,
                )

                # If split on Year-Scenario we want to loop over individual years
                for scen in cap_started_df.index:
                    df = gen_cap.xs(scen, level="Scenario")
                    # Get list of unique techs by Scenario-year
                    tech_names = df.index.get_level_values("tech").unique()
                    for tech_name in tech_names:
                        stt = df.xs(tech_name, level="tech", drop_level=False)
                        gen_names = stt.index.get_level_values("gen_name").unique()
                        cap_started = 0
                        for gen in gen_names:
                            sgt = stt.xs(gen, level="gen_name").fillna(0)
                            # Check that this generator has some, but not all,
                            # uncommitted hours.
                            if any(sgt["Output (MWh)"] == 0) and not all(
                                sgt["Output (MWh)"] == 0
                            ):
                                for idx in range(len(sgt["Output (MWh)"]) - 1):
                                    if (
                                        sgt["Output (MWh)"].iloc[idx] == 0
                                        and not sgt["Output (MWh)"].iloc[idx + 1] == 0
                                    ):
                                        cap_started = (
                                            cap_started
                                            + sgt["Installed Capacity (MW)"].iloc[idx]
                                        )

                        cap_started_df.loc[scen, tech_name] = cap_started

                cap_stated_chunks.append(cap_started_df)

            cap_started_all_scenarios = pd.concat(cap_stated_chunks).dropna(axis=1)

            if cap_started_all_scenarios.empty == True:
                out = MissingZoneData()
                outputs[zone_input] = out
                continue

            cap_started_all_scenarios = cap_started_all_scenarios.fillna(0)
            unitconversion = self.capacity_energy_unitconversion(
                cap_started_all_scenarios
            )

            cap_started_all_scenarios = (
                cap_started_all_scenarios / unitconversion["divisor"]
            )
            Data_Table_Out = cap_started_all_scenarios.T.add_suffix(
                f" ({unitconversion['units']}-starts)"
            )

            # transpose, sets scenarios as columns
            cap_started_all_scenarios = cap_started_all_scenarios.T

            mplt = PlotLibrary()
            fig, ax = mplt.get_figure()
            mplt.barplot(cap_started_all_scenarios, color=self.color_list)

            ax.set_ylabel(
                f"Capacity Started ({unitconversion['units']}-starts)",
                color="black",
                rotation="vertical",
            )

            mplt.add_legend()
            if plot_data_settings["plot_title_as_region"]:
                mplt.add_main_title(zone_input)

            outputs[zone_input] = {"fig": fig, "data_table": Data_Table_Out}
        return outputs

    def count_ramps(self, **_):
        """Plot under development

        Returns:
            UnderDevelopment(): Exception class, plot is not functional.
        """

        # Plot currently displays the same as capacity_started, this plot needs looking at

        outputs = UnderDevelopment()
        logger.warning("count_ramps is under development")
        return outputs

        outputs: dict = {}

        # List of properties needed by the plot, properties are a set of tuples and 
        # contain 3 parts: required True/False, property name and scenarios required, 
        # scenarios must be a list.
        properties = [
            (True, "generator_Generation", self.Scenarios),
            (True, "generator_Installed_Capacity", self.Scenarios),
        ]

        # Runs get_formatted_data within MPlotDataHelper to populate MPlotDataHelper dictionary
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)

        if 1 in check_input_data:
            return MissingInputData()

        for zone_input in self.Zones:
            logger.info(f"Zone =  {zone_input}")
            cap_started_chunk = []

            for scenario in self.Scenarios:

                logger.info(f"Scenario = {str(scenario)}")
                Gen = self["generator_Generation"].get(scenario)
                Gen = Gen.xs(zone_input, level=self.AGG_BY)

                Gen = Gen.reset_index()
                Gen.tech = Gen.tech.astype("category")
                Gen.tech = Gen.tech.cat.set_categories(self.ordered_gen)
                Gen = Gen.rename(columns={0: "Output (MWh)"})
                Gen = Gen[["timestamp", "gen_name", "tech", "Output (MWh)"]]
                Gen = Gen[
                    Gen["tech"].isin(self.thermal_gen_cat)
                ]  # We are only interested in thermal starts/stops.tops.

                Cap = self["generator_Installed_Capacity"].get(scenario)
                Cap = Cap.xs(zone_input, level=self.AGG_BY)
                Cap = Cap.reset_index()
                Cap = Cap.rename(columns={0: "Installed Capacity (MW)"})
                Cap = Cap[["gen_name", "Installed Capacity (MW)"]]
                Gen = pd.merge(Gen, Cap, on=["gen_name"])
                Gen.index = Gen.timestamp
                Gen = Gen.drop(columns=["timestamp"])

                # Min = pd.read_hdf(os.path.join(Marmot_Solutions_folder, scenario,"Processed_HDF5_folder", scenario + "_formatted.h5"),"generator_Hours_at_Minimum")
                # Min = Min.xs(zone_input, level = AGG_BY)

                if pd.notna(start_date_range):
                    logger.info(
                        f"Plotting specific date range: \
                    {str(start_date_range)} to {str(end_date_range)}"
                    )
                    Gen = Gen[start_date_range:end_date_range]

                tech_names = Gen["tech"].unique()
                ramp_counts = pd.DataFrame(columns=tech_names, index=[scenario])

                for tech_name in tech_names:
                    stt = Gen.loc[Gen["tech"] == tech_name]

                    gen_names = stt["gen_name"].unique()

                    up_ramps = 0

                    for gen in gen_names:
                        sgt = stt.loc[stt["gen_name"] == gen]
                        if any(sgt["Output (MWh)"] == 0) and not all(
                            sgt["Output (MWh)"] == 0
                        ):  # Check that this generator has some, but not all, uncommitted hours.
                            # print('Counting starts for: ' + gen)
                            for idx in range(len(sgt["Output (MWh)"]) - 1):
                                if (
                                    sgt["Output (MWh)"].iloc[idx] == 0
                                    and not sgt["Output (MWh)"].iloc[idx + 1] == 0
                                ):
                                    up_ramps = (
                                        up_ramps
                                        + sgt["Installed Capacity (MW)"].iloc[idx]
                                    )
                                # print('started on '+ timestamp)
                                # if sgt[0].iloc[idx] == 0 and not idx == 0 and not sgt[0].iloc[idx - 1] == 0:
                                #     stops = stops + 1

                    ramp_counts[tech_name] = up_ramps

                cap_started_chunk.append(ramp_counts)

            cap_started_all_scenarios = pd.concat(cap_started_chunk)

            if cap_started_all_scenarios.empty == True:
                out = MissingZoneData()
                outputs[zone_input] = out
                continue

            cap_started_all_scenarios.index = (
                cap_started_all_scenarios.index.str.replace("_", " ")
            )

            unitconversion = self.capacity_energy_unitconversion(
                cap_started_all_scenarios
            )

            cap_started_all_scenarios = (
                cap_started_all_scenarios / unitconversion["divisor"]
            )
            Data_Table_Out = cap_started_all_scenarios.T.add_suffix(
                f" ({unitconversion['units']}-starts)"
            )

            mplt = PlotLibrary()
            fig2, ax = mplt.get_figure()
            cap_started_all_scenarios.T.plot.bar(
                stacked=False,
                color=self.color_list,
                edgecolor="black",
                linewidth="0.1",
                ax=ax,
            )

            ax.set_ylabel(
                f"Capacity Started ({unitconversion['units']}-starts)",
                color="black",
                rotation="vertical",
            )

            # Set x-tick labels
            tick_labels = cap_started_all_scenarios.columns
            mplt.set_barplot_xticklabels(tick_labels)

            ax.legend(
                loc="lower left",
                bbox_to_anchor=(1, 0),
                facecolor="inherit",
                frameon=True,
            )
            if plot_data_settings["plot_title_as_region"]:
                mplt.add_main_title(zone_input)

            outputs[zone_input] = {"fig": fig2, "data_table": Data_Table_Out}
        return outputs
