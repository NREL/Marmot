"""Timeseries generation line plots. 

This code creates generation non-stacked line plots.

@author: Daniel Levie
"""
import logging
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from pathlib import Path
import marmot.utils.mconfig as mconfig
from typing import List

from marmot.plottingmodules.plotutils.styles import GeneratorColorDict
from marmot.plottingmodules.plotutils.plot_library import SetupSubplot
from marmot.plottingmodules.plotutils.plot_data_helper import (
    PlotDataStoreAndProcessor,
    GenCategories,
    set_facet_col_row_dimensions,
)
from marmot.plottingmodules.plotutils.timeseries_modifiers import (
    set_timestamp_date_range,
    adjust_for_leapday,
)
from marmot.plottingmodules.plotutils.plot_exceptions import (
    MissingInputData,
    MissingZoneData,
)

logger = logging.getLogger("plotter." + __name__)
plot_data_settings: dict = mconfig.parser("plot_data")
shift_leapday: bool = mconfig.parser("shift_leapday")
curtailment_prop: str = mconfig.parser("plot_data", "curtailment_property")


class GenerationUnStack(PlotDataStoreAndProcessor):
    """Timeseries generation line plots.

    The generation_unstack.py module contains methods that are
    related to the timeseries generation of generators,
    displayed in an unstacked line format.

    GenerationUnStack inherits from the PlotDataStoreAndProcessor class to assist
    in creating figures.
    """

    def __init__(
        self,
        Zones: List[str],
        Scenarios: List[str],
        AGG_BY: str,
        ordered_gen: List[str],
        marmot_solutions_folder: Path,
        gen_categories: GenCategories = GenCategories(),
        marmot_color_dict: dict = None,
        ylabels: List[str] = None,
        xlabels: List[str] = None,
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
            marmot_color_dict (dict, optional): Dictionary of colors to use for
                generation technologies.
                Defaults to None.
            ylabels (List[str], optional): y-axis labels for facet plots.
                Defaults to None.
            xlabels (List[str], optional): x-axis labels for facet plots.
                Defaults to None.
        """
        # Instantiation of PlotDataStoreAndProcessor
        super().__init__(AGG_BY, ordered_gen, marmot_solutions_folder, **kwargs)

        self.Zones = Zones
        self.Scenarios = Scenarios
        self.gen_categories = gen_categories
        if marmot_color_dict is None:
            self.marmot_color_dict = GeneratorColorDict.set_random_colors(
                self.ordered_gen
            ).color_dict
        else:
            self.marmot_color_dict = marmot_color_dict
        self.ylabels = ylabels
        self.xlabels = xlabels

    def gen_unstack(
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
        """Creates a timeseries plot of generation by technology each plotted as a line.

        If multiple scenarios are passed they will be plotted in a facet plot.
        The plot can be further customized by passing specific values to the
        prop argument.

        Args:
            prop (str, optional): Special argument used to adjust specific
                plot settings. Controlled through the plot_select.csv.
                Opinions available are:

                - Peak Demand
                - Min Net Load
                - Date Range

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
        outputs: dict = {}

        if self.AGG_BY == "zone":
            agg = "zone"
        else:
            agg = "region"

        def getdata(scenario_list):

            # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
            # required True/False, property name and scenarios required, scenarios must be a list.
            properties = [
                (True, f"generator_Generation{data_resolution}", scenario_list),
                (
                    False,
                    f"generator_{curtailment_prop}{data_resolution}",
                    scenario_list,
                ),
                (False, f"{agg}_Load{data_resolution}", scenario_list),
                (False, f"{agg}_Demand{data_resolution}", scenario_list),
                (False, f"{agg}_Unserved_Energy{data_resolution}", scenario_list),
            ]

            # Runs get_formatted_data within PlotDataStoreAndProcessor to populate PlotDataStoreAndProcessor dictionary
            # with all required properties, returns a 1 if required data is missing
            return self.get_formatted_data(properties)

        check_input_data = getdata(self.Scenarios)
        all_scenarios = self.Scenarios

        # Checks if all data required by plot is available, if 1 in list required data is missing
        if 1 in check_input_data:
            outputs = MissingInputData()
            return outputs

        # sets up x, y dimensions of plot
        ncols, nrows = set_facet_col_row_dimensions(
            self.xlabels, self.ylabels, multi_scenario=all_scenarios
        )

        grid_size = ncols * nrows

        # Used to calculate any excess axis to delete
        plot_number = len(all_scenarios)

        for zone_input in self.Zones:
            logger.info(f"Zone = {zone_input}")

            excess_axs = grid_size - plot_number

            mplt = SetupSubplot(
                nrows, ncols, sharey=True, squeeze=False, ravel_axs=True
            )
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

            for i, scenario in enumerate(all_scenarios):
                logger.info(f"Scenario = {scenario}")

                try:
                    stacked_gen_df = (
                        self[f"generator_Generation{data_resolution}"]
                        .get(scenario)
                        .copy()
                    )
                    if shift_leapday:
                        stacked_gen_df = adjust_for_leapday(stacked_gen_df)
                    stacked_gen_df = stacked_gen_df.xs(zone_input, level=self.AGG_BY)
                except KeyError:
                    logger.warning(f"No generation in {zone_input}")
                    continue

                if stacked_gen_df.empty == True:
                    continue

                stacked_gen_df = self.df_process_gen_inputs(stacked_gen_df)

                # Insert Curtailment into gen stack if it exists in database
                stacked_gen_df = self.add_curtailment_to_df(
                    stacked_gen_df,
                    scenario,
                    zone_input,
                    self.gen_categories.vre,
                    data_resolution=data_resolution,
                )

                curtailment_name = self.gen_names_dict.get("Curtailment", "Curtailment")
                if curtailment_name in stacked_gen_df.columns:
                    vre_gen_cat = self.gen_categories.vre + [curtailment_name]
                else:
                    vre_gen_cat = self.gen_categories.vre

                if pd.notna(start_date_range):
                    stacked_gen_df = set_timestamp_date_range(
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

                # Process extra optional properties
                extra_property_names = [
                    f"{agg}_Load{data_resolution}",
                    f"{agg}_Demand{data_resolution}",
                    f"{agg}_Unserved_Energy{data_resolution}",
                ]
                extra_plot_data = self.process_extra_properties(
                    extra_property_names,
                    scenario,
                    zone_input,
                    agg,
                    data_resolution=data_resolution,
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
                        stacked_gen_df, self.Scenarios, sum_values=True
                    )
                # Convert units
                stacked_gen_df = stacked_gen_df / unitconversion["divisor"]
                extra_plot_data = extra_plot_data / unitconversion["divisor"]

                # Adds property annotation and
                if prop:
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

                # Remove any all 0 columns
                stacked_gen_df = stacked_gen_df.loc[
                    :, (stacked_gen_df != 0).any(axis=0)
                ]

                scenario_names = pd.Series(
                    [scenario] * len(stacked_gen_df), name="Scenario"
                )
                data_table = stacked_gen_df.add_suffix(f" ({unitconversion['units']})")
                data_table = data_table.set_index([scenario_names], append=True)
                data_tables.append(data_table)

                for column in stacked_gen_df.columns:
                    axs[i].plot(
                        stacked_gen_df.index.values,
                        stacked_gen_df[column],
                        linewidth=2,
                        color=self.marmot_color_dict.get(column, "#333333"),
                        label=column,
                    )

                if (extra_plot_data["Unserved Energy"] == 0).all() == False:
                    axs[i].plot(
                        extra_plot_data["Unserved Energy"],
                        color="#DD0200",
                        label="Unserved Energy",
                    )

                mplt.set_yaxis_major_tick_format(sub_pos=i)
                axs[i].margins(x=0.01)
                mplt.set_subplot_timeseries_format(sub_pos=i)

            if not data_tables:
                logger.warning(f"No generation in {zone_input}")
                out = MissingZoneData()
                outputs[zone_input] = out
                continue

            data_table_out = pd.concat(data_tables)

            # add facet labels
            mplt.add_facet_labels(xlabels=self.xlabels, ylabels=self.ylabels)
            # Add legend
            mplt.add_legend(reverse_legend=True, sort_by=self.ordered_gen)
            # Remove extra supl
            mplt.remove_excess_axs(excess_axs, grid_size)
            # Add title
            mplt.add_main_title(zone_input)

            labelpad = 40
            plt.ylabel(
                f"Generation ({unitconversion['units']})",
                color="black",
                rotation="vertical",
                labelpad=labelpad,
            )

            outputs[zone_input] = {"fig": fig, "data_table": data_table_out}
        return outputs
