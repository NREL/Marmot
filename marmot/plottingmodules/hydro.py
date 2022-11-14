# -*- coding: utf-8 -*-
"""Hydro generator plots.

This module creates hydro analysis plots.

DL: Oct 9th 2021, This plot is in need of work. 
It may not produce production ready figures.

@author: adyreson
"""

#####
#Testing
#####
# gen_names = pd.read_csv('/Users/mschwarz/Marmot_local/Marmot/input_files/mapping_folder/gen_names.csv')
# gen_names = gen_names.rename(
# columns={gen_names.columns[0]: "Original", gen_names.columns[1]: "New"}
# )
# gen_names_dict = (
#     gen_names[["Original", "New"]].set_index("Original").to_dict()["New"]
# )

# self = Hydro(Zones = ['BPAT_WI'],
#           Scenarios = ['a_fh', 'a_td'],
#           AGG_BY = 'region',
#           ordered_gen = ['Nuclear', 'Coal', 'Gas-CC', 'Gas-CC CCS', 'Gas-CT', 'Gas', 'Landfill gas', 'Gas-Steam', 'Dual Fuel', 'DualFuel', 'Oil-Gas-Steam', 'Oil/Gas', 'Oil', 'Hydro', 'Hydropower', 'Ocean', 'Geothermal', 'Biomass', 'Biopower', 'Other', 'VRE', 'Wind', 'Offshore Wind', 'OffshoreWind', 'Solar', 'PV', 'dPV', 'CSP', 'PV-Battery', 'Battery', 'OSW-Battery', 'PHS', 'Tidal', 'Storage', 'Storage discharge', 'Battery discharge', 'Net Imports', 'Curtailment', 'curtailment', 'Demand', 'Deamand + Storage Charging'],
#           marmot_solutions_folder = '/Users/mschwarz/WaterRisk local/StageA_2009_results',
#            gen_names_dict = gen_names_dict
# )

import datetime as dt
import logging
import os
from pathlib import Path
from typing import List

import matplotlib.ticker as mtick
import pandas as pd

import marmot.utils.mconfig as mconfig
from marmot.plottingmodules.plotutils.plot_data_helper import (
    GenCategories,
    PlotDataStoreAndProcessor,
)
from marmot.plottingmodules.plotutils.plot_exceptions import (
    DataSavedInModule,
    MissingInputData,
    MissingZoneData,
    UnderDevelopment,
)
from marmot.plottingmodules.plotutils.plot_library import SetupSubplot
from marmot.plottingmodules.plotutils.styles import GeneratorColorDict
from marmot.plottingmodules.plotutils.plot_library import PlotLibrary, SetupSubplot
from marmot.plottingmodules.plotutils.styles import GeneratorColorDict
from marmot.plottingmodules.plotutils.timeseries_modifiers import (
    adjust_for_leapday,
    set_timestamp_date_range,
)


logger = logging.getLogger("plotter." + __name__)
plot_data_settings: dict = mconfig.parser("plot_data")


class Hydro(PlotDataStoreAndProcessor):
    """Hydro generator plots.

    The hydro.py module contains methods that are
    related to hydro generators.

    Hydro inherits from the PlotDataStoreAndProcessor class to
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
        marmot_color_dict: dict = None,
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

    def hydro_timeseries(self, end: int = 7, timezone: str = "", **_):
        """Timeseries Line plot of hydro generation.

        Creates separate plots for each week of the year, or longer depending
        on 'Day After' value passed through plot_select.csv

        Data is saved within this method.

        Args:
            end (float, optional): Determines length of plot period.
                Defaults to 7.
            timezone (str, optional): The timezone to display on the x-axes.
                Defaults to "".

        Returns:
            DataSavedInModule: DataSavedInModule exception
        """
        outputs: dict = {}

        # List of properties needed by the plot, properties are a set of tuples and
        # contain 3 parts: required True/False, property name and scenarios required,
        # scenarios must be a list.
        properties = [(True, "generator_Generation", self.Scenarios)]

        # Runs get_formatted_data within PlotDataStoreAndProcessor to populate PlotDataStoreAndProcessor dictionary
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)

        if 1 in check_input_data:
            return MissingInputData()

        for zone_input in self.Zones:
            logger.info("Zone = " + zone_input)

            # Location to save to
            hydro_figures = os.path.join(self.figure_folder, self.AGG_BY + "_Hydro")

            hydro_gen_all = []
            for scenario in self.Scenarios:
                logger.info(f"Scenario = {scenario}")
                #Create hydro data frame
                Stacked_Gen_read = self["generator_Generation"].get(scenario)

                # The rest of the function won't work if this particular zone can't be found
                # in the solution file (e.g. if it doesn't include Mexico)
                try:
                    Stacked_Gen = Stacked_Gen_read.xs(zone_input, level=self.AGG_BY)
                except KeyError:
                    logger.warning("No Generation in %s", zone_input)
                    continue

                del Stacked_Gen_read
                Stacked_Gen = self.df_process_gen_inputs(Stacked_Gen)

                # Removes columns that only contain 0
                Stacked_Gen = Stacked_Gen.loc[:, (Stacked_Gen != 0).any(axis=0)]
                try:
                    Hydro_Gen = Stacked_Gen["Hydro"]
                except KeyError:
                    logger.warning("No Hydro Generation in %s", zone_input)
                    Hydro_Gen = MissingZoneData()
                    continue

                del Stacked_Gen
                Hydro_Gen.name = scenario
                hydro_gen_all.append(Hydro_Gen)
            Hydro_Gen_Out = pd.concat(hydro_gen_all,axis = 1)

            # Scatter plot by season
            mplt = PlotLibrary(1, 1, sharey=True, squeeze=False, ravel_axs=True)
            fig, axs = mplt.get_figure()

            for col in Hydro_Gen_Out:
                mplt.lineplot(
                    Hydro_Gen_Out,
                    col,
                    # linewidth=2,
                    # color=self.marmot_color_dict.get("Hydro", "#333333"),
                    # label="Hydro",
                )

            axs.set_ylabel("Generation (MW)", color="black", rotation="vertical")
            axs.set_xlabel(timezone, color="black", rotation="horizontal")
            mplt.set_yaxis_major_tick_format()
            ax.margins(x=0.01)

            mplt.set_subplot_timeseries_format()

            # Add title
            if plot_data_settings["plot_title_as_region"]:
                mplt.add_main_title(zone_input)
            # Add legend
            mplt.add_legend(reverse_legend=True)

            fig.savefig(
                os.path.join(
                    hydro_figures,
                    zone_input
                    + f"_Hydro_And_Net_Load_{self.Scenarios[0]}_period_{str(wk)}",
                ),
                dpi=600,
                bbox_inches="tight",
            )
            Data_Table_Out.to_csv(
                os.path.join(
                    hydro_figures,
                    zone_input
                    + f"_Hydro_And_Net_Load_{self.Scenarios[0]}_period_{str(wk)}.csv",
                )
            )
            del fig
            del Data_Table_Out
            # end weekly loop
            # Scatter plot

            mplt = SetupSubplot()
            fig, ax = mplt.get_figure()
            ax.scatter(Net_Load, Hydro_Gen, color="black", s=5)

            ax.set_ylabel(
                "In-Region Hydro Generation (MW)", color="black", rotation="vertical"
            )
            ax.set_xlabel(
                "In-Region Net Load (MW)", color="black", rotation="horizontal"
            )
            mplt.set_yaxis_major_tick_format()
            ax.xaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))
            ax.margins(x=0.01)

            mplt.add_legend(reverse_legend=True)

            if plot_data_settings["plot_title_as_region"]:
                mplt.add_main_title(zone_input)
            fig.savefig(
                os.path.join(
                    hydro_figures,
                    zone_input + f"_Hydro_Versus_Net_Load_{self.Scenarios[0]}",
                ),
                dpi=600,
                bbox_inches="tight",
            )

        outputs = DataSavedInModule()
        return outputs


    def hydro_continent_net_load(
        self, start_date_range: str = None, end_date_range: str = None, **_
    ):
        """Creates a scatter plot of hydro generation vs net load

        Data is saved within this method.

        Args:
            start_date_range (str, optional): Defines a start date at which to represent data from.
                Defaults to None.
            end_date_range (str, optional): Defines a end date at which to represent data to.
                Defaults to None.

        Returns:
            DataSavedInModule: DataSavedInModule exception
        """
        return UnderDevelopment()
        outputs: dict = {}

        # List of properties needed by the plot, properties are a set of tuples and
        # contain 3 parts: required True/False, property name and scenarios required,
        # scenarios must be a list.
        properties = [(True, "generator_Generation", [self.Scenarios[0]])]

        # Runs get_formatted_data within PlotDataStoreAndProcessor to populate PlotDataStoreAndProcessor dictionary
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)

        if 1 in check_input_data:
            return MissingInputData()

        for zone_input in self.Zones:
            # Location to save to
            hydro_figures = os.path.join(self.figure_folder, self.AGG_BY + "_Hydro")

            Stacked_Gen_read = self["generator_Generation"].get(self.Scenarios[0])

            logger.info("Zone = " + zone_input)
            logger.info(
                "Winter is defined as date range: \
            {} to {}".format(
                    str(start_date_range), str(end_date_range)
                )
            )
            Net_Load = self.df_process_gen_inputs(Stacked_Gen_read)

            # Calculates Net Load by removing variable gen
            # Adjust list of values to drop depending on if it exists in Stacked_Gen df
            vre_gen_cat = [
                name for name in self.gen_categories.vre if name in Net_Load.columns
            ]
            Net_Load = Net_Load.drop(labels=vre_gen_cat, axis=1)
            Net_Load = Net_Load.sum(axis=1)  # Continent net load

            try:
                Stacked_Gen = Stacked_Gen_read.xs(zone_input, level=self.AGG_BY)
            except KeyError:
                logger.warning("No Generation in %s", zone_input)
                continue
            del Stacked_Gen_read
            Stacked_Gen = self.df_process_gen_inputs(Stacked_Gen)
            # Removes columns only containing 0
            Stacked_Gen = Stacked_Gen.loc[:, (Stacked_Gen != 0).any(axis=0)]

            # end weekly loop

            try:
                Hydro_Gen = Stacked_Gen["Hydro"]
            except KeyError:
                logger.warning("No Hydro Generation in %s", zone_input)
                Hydro_Gen = MissingZoneData()
                continue

            del Stacked_Gen

            # Scatter plot by season
            mplt = SetupSubplot()
            fig, ax = mplt.get_figure()

            ax.scatter(
                Net_Load[end_date_range:start_date_range],
                Hydro_Gen[end_date_range:start_date_range],
                color="black",
                s=5,
                label="Non-winter",
            )
            ax.scatter(
                Net_Load[start_date_range:],
                Hydro_Gen[start_date_range:],
                color="blue",
                s=5,
                label="Winter",
                alpha=0.5,
            )
            ax.scatter(
                Net_Load[:end_date_range],
                Hydro_Gen[:end_date_range],
                color="blue",
                s=5,
                alpha=0.5,
            )

            ax.set_ylabel(
                "In Region Hydro Generation (MW)", color="black", rotation="vertical"
            )
            ax.set_xlabel(
                "Continent Net Load (MW)", color="black", rotation="horizontal"
            )
            mplt.set_yaxis_major_tick_format()
            ax.xaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))
            ax.margins(x=0.01)
            # Add title
            if plot_data_settings["plot_title_as_region"]:
                mplt.add_main_title(zone_input)
            mplt.add_legend(reverse_legend=True)

            fig.savefig(
                os.path.join(
                    hydro_figures,
                    zone_input
                    + f"_Hydro_Versus_Continent_Net_Load_{self.Scenarios[0]}",
                ),
                dpi=600,
                bbox_inches="tight",
            )

        outputs = DataSavedInModule()
        return outputs

    def hydro_net_load(self, end: int = 7, timezone: str = "", **_):
        """Line plot of hydro generation vs net load.

        Creates separate plots for each week of the year, or longer depending
        on 'Day After' value passed through plot_select.csv

        Data is saved within this method.

        Args:
            end (float, optional): Determines length of plot period.
                Defaults to 7.
            timezone (str, optional): The timezone to display on the x-axes.
                Defaults to "".

        Returns:
            DataSavedInModule: DataSavedInModule exception
        """
        outputs: dict = {}

        # List of properties needed by the plot, properties are a set of tuples and
        # contain 3 parts: required True/False, property name and scenarios required,
        # scenarios must be a list.
        properties = [(True, "generator_Generation", [self.Scenarios[0]])]

        # Runs get_formatted_data within PlotDataStoreAndProcessor to populate PlotDataStoreAndProcessor dictionary
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)

        if 1 in check_input_data:
            return MissingInputData()

        for zone_input in self.Zones:
            logger.info("Zone = " + zone_input)

            # Location to save to
            hydro_figures = os.path.join(self.figure_folder, self.AGG_BY + "_Hydro")

            Stacked_Gen_read = self["generator_Generation"].get(self.Scenarios[0])

            # The rest of the function won't work if this particular zone can't be found
            # in the solution file (e.g. if it doesn't include Mexico)
            try:
                Stacked_Gen = Stacked_Gen_read.xs(zone_input, level=self.AGG_BY)
            except KeyError:
                logger.warning("No Generation in %s", zone_input)
                continue

            del Stacked_Gen_read
            Stacked_Gen = self.df_process_gen_inputs(Stacked_Gen)

            # Calculates Net Load by removing variable gen
            # Adjust list of values to drop depending on if it exists in Stacked_Gen df
            vre_gen_cat = [
                name for name in self.gen_categories.vre if name in Stacked_Gen.columns
            ]
            Net_Load = Stacked_Gen.drop(labels=vre_gen_cat, axis=1)
            Net_Load = Net_Load.sum(axis=1)

            # Removes columns that only contain 0
            Stacked_Gen = Stacked_Gen.loc[:, (Stacked_Gen != 0).any(axis=0)]
            try:
                Hydro_Gen = Stacked_Gen["Hydro"]
            except KeyError:
                logger.warning("No Hydro Generation in %s", zone_input)
                Hydro_Gen = MissingZoneData()
                continue

            del Stacked_Gen

            first_date = Net_Load.index[0]
            # assumes weekly, could be something else if user changes end Marmot_plot_select
            for wk in range(1, 53):

                period_start = first_date + dt.timedelta(days=(wk - 1) * 7)
                period_end = period_start + dt.timedelta(days=end)
                logger.info(str(period_start) + " and next " + str(end) + " days.")
                Hydro_Period = Hydro_Gen[period_start:period_end]
                Net_Load_Period = Net_Load[period_start:period_end]

                # Data table of values to return to main program
                Data_Table_Out = pd.concat(
                    [Net_Load_Period, Hydro_Period], axis=1, sort=False
                )

                # Scatter plot by season
                mplt = SetupSubplot()
                fig, ax = mplt.get_figure()

                ax.plot(
                    Hydro_Period,
                    linewidth=2,
                    color=self.marmot_color_dict.get("Hydro", "#333333"),
                    label="Hydro",
                )

                ax.plot(Net_Load_Period, color="black", label="Load")

                ax.set_ylabel("Generation (MW)", color="black", rotation="vertical")
                ax.set_xlabel(timezone, color="black", rotation="horizontal")
                mplt.set_yaxis_major_tick_format()
                ax.margins(x=0.01)

                mplt.set_subplot_timeseries_format()

                # Add title
                if plot_data_settings["plot_title_as_region"]:
                    mplt.add_main_title(zone_input)
                # Add legend
                mplt.add_legend(reverse_legend=True)

                fig.savefig(
                    os.path.join(
                        hydro_figures,
                        zone_input
                        + f"_Hydro_And_Net_Load_{self.Scenarios[0]}_period_{str(wk)}",
                    ),
                    dpi=600,
                    bbox_inches="tight",
                )
                Data_Table_Out.to_csv(
                    os.path.join(
                        hydro_figures,
                        zone_input
                        + f"_Hydro_And_Net_Load_{self.Scenarios[0]}_period_{str(wk)}.csv",
                    )
                )
                del fig
                del Data_Table_Out
            # end weekly loop
            # Scatter plot

            mplt = SetupSubplot()
            fig, ax = mplt.get_figure()
            ax.scatter(Net_Load, Hydro_Gen, color="black", s=5)

            ax.set_ylabel(
                "In-Region Hydro Generation (MW)", color="black", rotation="vertical"
            )
            ax.set_xlabel(
                "In-Region Net Load (MW)", color="black", rotation="horizontal"
            )
            mplt.set_yaxis_major_tick_format()
            ax.xaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))
            ax.margins(x=0.01)

            mplt.add_legend(reverse_legend=True)

            if plot_data_settings["plot_title_as_region"]:
                mplt.add_main_title(zone_input)
            fig.savefig(
                os.path.join(
                    hydro_figures,
                    zone_input + f"_Hydro_Versus_Net_Load_{self.Scenarios[0]}",
                ),
                dpi=600,
                bbox_inches="tight",
            )

        outputs = DataSavedInModule()
        return outputs
