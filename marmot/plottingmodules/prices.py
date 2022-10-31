# -*- coding: utf-8 -*-
"""Locational price analysis plots.

Price analysis plots, price duration curves and timeseries plots.
Prices plotted in $/MWh

@author: adyreson and Daniel Levie
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import marmot.utils.mconfig as mconfig

from marmot.plottingmodules.plotutils.plot_library import SetupSubplot
from marmot.plottingmodules.plotutils.plot_data_helper import MPlotDataHelper
from marmot.plottingmodules.plotutils.plot_exceptions import (
    MissingInputData,
    DataSavedInModule,
    InputSheetError,
)

logger = logging.getLogger("plotter." + __name__)
plot_data_settings = mconfig.parser("plot_data")


class Prices(MPlotDataHelper):
    """Locational price analysis plots.

    The price.py module contains methods that are
    related to grid prices at regions, zones, nodes etc.

    Prices inherits from the MPlotDataHelper class to assist
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

    def pdc_all_regions(
        self,
        y_axis_max: float = None,
        start_date_range: str = None,
        end_date_range: str = None,
        **_,
    ):
        """Creates a price duration curve for all regions/zones and plots them on a single facet plot.

        Price is in $/MWh.
        The code automatically creates a facet plot based on the number of regions/zones in the input.
        All scenarios are plotted on a single facet for each region/zone

        Args:
            y_axis_max (float, optional): Max y-axis value.
                Defaults to None.
            start_date_range (str, optional): Defines a start date at which to represent data from.
                Defaults to None.
            end_date_range (str, optional): Defines a end date at which to represent data to.
                Defaults to None.

        Returns:
            dict: dictionary containing the created plot and its data table
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
            (True, f"{agg}_Price", self.Scenarios),
            #(True, "generator_Installed_Capacity", self.Scenarios),
            (True, f"generator_Installed_Capacity", self.Scenarios),
            (True, f"generator_Available_Capacity", self.Scenarios),
        ]

        # Runs get_formatted_data within MPlotDataHelper to populate MPlotDataHelper dictionary
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)

        if 1 in check_input_data:
            return MissingInputData()

        # Location to save to
        save_figures: Path = self.figure_folder.joinpath(f"{self.AGG_BY}_prices")

        region_number = len(self.Zones)
        # determine x,y length for plot
        ncols, nrows = self.set_x_y_dimension(region_number)

        grid_size = ncols * nrows
        # Used to calculate any excess axis to delete
        excess_axs = grid_size - region_number

        # setup plot
        mplt = SetupSubplot(nrows, ncols, sharey=True, squeeze=False, ravel_axs=True)
        fig, axs = mplt.get_figure()
        plt.subplots_adjust(wspace=0.1, hspace=0.50)

        data_table = []
        for n, zone_input in enumerate(self.Zones):

            all_prices = []
            for scenario in self.Scenarios:
                price = self._process_data(self[f"{agg}_Price"], scenario, zone_input)
                price = price.groupby(["timestamp"]).sum()
                #ww added
                Cap = self._process_data(self[f"generator_Available_Capacity"], scenario, zone_input)
                #Cap = Cap.xs(zone_input, level=self.AGG_BY)
                #Cap = Cap.rename(columns={0: "Installed Capacity (MW)"})
                Cap = Cap.groupby(["timestamp"]).sum()
                #price = pd.merge(price,Cap,on=["timestamp"])
                #global price
                #price = [price / Cap for price, Cap in zip(price, Cap)]
                #(price.astype(int)* Cap.astype(int)) /
                #(price.astype(int)/ sum(Cap.astype(int)) )*Cap.astype(int)
                price = (price.astype(int)/ (Cap.astype(int).sum()) )*Cap.astype(int)
                if pd.notna(start_date_range):
                    #ww changed
                    price,cap = self.set_timestamp_date_range(
                        [Cap,price], start_date_range, end_date_range
                    )
                #price.sort_values(by=scenario, ascending=False, inplace=True)
                price.reset_index(drop=True, inplace=True)
                all_prices.append(price)
                #all_prices.append(Cap)

                
            #Cap["year"] = Cap.index.get_level_values("timestamp").year.astype(str

            
            duration_curve = pd.concat(all_prices, axis=1)
            duration_curve.columns = duration_curve.columns.str.replace("_", " ")


            data_out = duration_curve.copy()
            data_out.columns = [zone_input + "_" + str(col) for col in data_out.columns]
            #ww added
            #global cap 
            #data_out.columns = sum(data_out*cap)/sum(cap)
            data_table.append(data_out)

            color_dict = dict(zip(duration_curve.columns, self.color_list))

            for column in duration_curve:
                axs[n].plot(
                    duration_curve[column],
                    linewidth=1,
                    color=color_dict[column],
                    label=column,
                    alpha=1,
                )
                if pd.notna(y_axis_max):
                    axs[n].set_ylim(bottom=0, top=float(y_axis_max))
                axs[n].set_xlim(0, len(duration_curve))
                axs[n].set_title(zone_input.replace("_", " "))

        mplt.add_legend()
        # Remove extra axes
        mplt.remove_excess_axs(excess_axs, grid_size)

        plt.ylabel(
            f"{self.AGG_BY} Price ($/MWh)",
            color="black",
            rotation="vertical",
            labelpad=30,
        )
        plt.xlabel("Intervals", color="black", rotation="horizontal", labelpad=20)

        Data_Table_Out = pd.concat(data_table, axis=1)
        Data_Table_Out = Data_Table_Out.add_suffix(" ($/MWh)")

        fig.savefig(
            save_figures.joinpath("Price_weight_average.svg"),
            dpi=600,
            bbox_inches="tight",
        )
        Data_Table_Out.to_csv(
            save_figures.joinpath("Price_weight_average.csv")
        )
        outputs = DataSavedInModule()
        return outputs

    def region_pdc(
        self,
        figure_name: str = None,
        y_axis_max: float = None,
        start_date_range: str = None,
        end_date_range: str = None,
        **_,
    ):
        """Creates a price duration curve for each region. Price in $/MWh

        The code will create either a facet plot or a single plot depending on
        if the Facet argument is active.
        If a facet plot is created, each scenario is plotted on a separate facet,
        otherwise all scenarios are plotted on a single plot.
        To make a facet plot, ensure the word 'Facet' is found in the figure_name.

        Args:
            figure_name (str, optional): User defined figure output name.
                Defaults to None.
            y_axis_max (float, optional): Max y-axis value.
                Defaults to None.
            start_date_range (str, optional): Defines a start date at which to represent data from.
                Defaults to None.
            end_date_range (str, optional): Defines a end date at which to represent data to.
                Defaults to None.

        Returns:
            dict: dictionary containing the created plot and its data table
        """
        outputs: dict = {}

        facet = False
        if "Facet" in figure_name:
            facet = True

        if self.AGG_BY == "zone":
            agg = "zone"
        else:
            agg = "region"

        # List of properties needed by the plot, properties are a set of tuples and 
        # contain 3 parts: required True/False, property name and scenarios required, 
        # scenarios must be a list.
        properties = [(True, f"{agg}_Price", self.Scenarios)]

        # Runs get_formatted_data within MPlotDataHelper to populate MPlotDataHelper dictionary
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)

        if 1 in check_input_data:
            return MissingInputData()

        for zone_input in self.Zones:
            logger.info(f"{self.AGG_BY} = {zone_input}")

            all_prices = []
            for scenario in self.Scenarios:

                price = self._process_data(self[f"{agg}_Price"], scenario, zone_input)
                price = price.groupby(["timestamp"]).sum()
                if pd.notna(start_date_range):
                    price = self.set_timestamp_date_range(
                        price, start_date_range, end_date_range
                    )
                if 'values' in price.columns:
                    price.sort_values(by='values', ascending=False, inplace=True)
                else:
                    price.sort_values(by=scenario, ascending=False, inplace=True)             
                price.reset_index(drop=True, inplace=True)
                price.columns = [scenario]
                all_prices.append(price)

            duration_curve = pd.concat(all_prices, axis=1)
            duration_curve.columns = duration_curve.columns.str.replace("_", " ")

            Data_Out = duration_curve.add_suffix(" ($/MWh)")

            ncols = len(self.xlabels)
            if ncols == 0:
                ncols = 1
            nrows = len(self.ylabels)
            if nrows == 0:
                nrows = 1

            # If the plot is not a facet plot, grid size should be 1x1
            if not facet:
                ncols = 1
                nrows = 1

            color_dict = dict(zip(duration_curve.columns, self.color_list))

            # setup plot
            mplt = SetupSubplot(
                nrows, ncols, sharey=True, squeeze=False, ravel_axs=True
            )
            fig, axs = mplt.get_figure()
            plt.subplots_adjust(wspace=0.05, hspace=0.2)

            n = 0
            for column in duration_curve:
                axs[n].plot(
                    duration_curve[column],
                    linewidth=1,
                    color=color_dict[column],
                    label=column,
                    alpha=1,
                )
                if pd.notna(y_axis_max):
                    axs[n].set_ylim(bottom=0, top=float(y_axis_max))
                axs[n].set_xlim(0, len(duration_curve))
                if facet:
                    n += 1

            mplt.add_legend()
            plt.ylabel(
                f"{self.AGG_BY} Price ($/MWh)",
                color="black",
                rotation="vertical",
                labelpad=20,
            )
            plt.xlabel("Intervals", color="black", rotation="horizontal", labelpad=20)
            if plot_data_settings["plot_title_as_region"]:
                mplt.add_main_title(zone_input)
            outputs[zone_input] = {"fig": fig, "data_table": Data_Out}
        return outputs

    def national_lwavg_price(
        self,
        figure_name: str = None,
        y_axis_max: float = None,
        timezone: str = "",
        start_date_range: str = None,
        end_date_range: str = None,
        **_,
    ):   

        outputs: dict={}

        def weighted_avg(df,values,weights):
            return sum(df[values] * df[values] / df[weights].sum())

        properties = [(True, "region_Price", self.Scenarios),
                      (True, "region_Demand", self.Scenarios)]

        # Runs get_formatted_data within MPlotDataHelper to populate MPlotDataHelper dictionary
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)

        if 1 in check_input_data:
            return MissingInputData()

        df_all_list = []
        for scenario in self.Scenarios:
            logger.info(f"Scenario = {scenario}")
            price = self['region_Price'].get(scenario)
            demand = self['region_Demand'].get(scenario)
            avgs = pd.merge(price,demand,on=['timestamp','region'])
            avgs.columns = ['price','demand']
            avgs = avgs.reset_index()
            avgs.set_index('timestamp',inplace=True)
            df = pd.DataFrame(index = avgs.index.unique(),
                            columns = [scenario])
            for ts in df.index:
                avg_hour = avgs[avgs.index == ts]
                lwavg = np.average(a = avg_hour['price'],weights = avg_hour['demand'])
                df.loc[ts,scenario] = lwavg
        
            df.sort_values(by=scenario, ascending=False, inplace=True)             
            df.reset_index(drop=True, inplace=True)
            df_all_list.append(df)
        
        df_all = pd.concat(df_all_list)

        # setup plot
        nrows = 1
        ncols = 1
        mplt = SetupSubplot(
            nrows, ncols, sharey=True, squeeze=False, ravel_axs=True
        )
        fig, axs = mplt.get_figure()
        plt.subplots_adjust(wspace=0.05, hspace=0.2)

        n = 0
        for column in df_all:
            axs[n].plot(
                df_all[column],
                linewidth=1,
                label=column,
                alpha=1,
            )
            if pd.notna(y_axis_max):
                axs[n].set_ylim(bottom=0, top=float(y_axis_max))
                axs[n].set_xlim(0, len(df_all))

        mplt.add_legend()
        plt.ylabel(
            "Price ($/MWh)",
            color="black",
            rotation="vertical",
            labelpad=20,
        )
        plt.xlabel("Intervals", color="black", rotation="horizontal", labelpad=20)
        mplt.add_main_title('Nation-wide load weighted average')

        fig.savefig(self.figure_folder.joinpath("Country_prices", f"{figure_name}.svg"), 
                    dpi=600, bbox_inches='tight')
        df_all.to_csv(self.figure_folder.joinpath("Country_prices", f"{figure_name}.csv"))

        outputs = DataSavedInModule()
        return outputs

    def region_timeseries_price(
        self,
        figure_name: str = None,
        y_axis_max: float = None,
        timezone: str = "",
        start_date_range: str = None,
        end_date_range: str = None,
        **_,
    ):
        """Creates price timeseries line plot for each region. Price is $/MWh.

        The code will create either a facet plot or a single plot depending on
        if the Facet argument is active.
        If a facet plot is created, each scenario is plotted on a separate facet,
        otherwise all scenarios are plotted on a single plot.
        To make a facet plot, ensure the work 'Facet' is found in the figure_name.

        Args:
            figure_name (str, optional): User defined figure output name.
                Defaults to None.
            y_axis_max (float, optional): Max y-axis value.
                Defaults to None.
            timezone (str, optional): The timezone to display on the x-axes.
                Defaults to "".
            start_date_range (str, optional): Defines a start date at which to represent data from.
                Defaults to None.
            end_date_range (str, optional): Defines a end date at which to represent data to.
                Defaults to None.

        Returns:
            dict: dictionary containing the created plot and its data table
        """
        outputs: dict = {}

        facet = False
        if "Facet" in figure_name:
            facet = True

        if self.AGG_BY == "zone":
            agg = "zone"
        else:
            agg = "region"

        # List of properties needed by the plot, properties are a set of tuples and 
        # contain 3 parts: required True/False, property name and scenarios required, 
        # scenarios must be a list.
        properties = [(True, f"{agg}_Price", self.Scenarios)]

        # Runs get_formatted_data within MPlotDataHelper to populate MPlotDataHelper dictionary
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)

        if 1 in check_input_data:
            return MissingInputData()

        for zone_input in self.Zones:
            logger.info(f"{self.AGG_BY} = {zone_input}")

            all_prices = []
            for scenario in self.Scenarios:
                price = self._process_data(self[f"{agg}_Price"], scenario, zone_input)
                price = price.groupby(["timestamp"]).sum()
                if pd.notna(start_date_range):
                    price = self.set_timestamp_date_range(
                        price, start_date_range, end_date_range
                    )
                all_prices.append(price)

            timeseries = pd.concat(all_prices, axis=1)
            timeseries.columns = timeseries.columns.str.replace("_", " ")

            Data_Out = timeseries.add_suffix(" ($/MWh)")

            ncols = len(self.xlabels)
            if ncols == 0:
                ncols = 1
            nrows = len(self.ylabels)
            if nrows == 0:
                nrows = 1

            # If the plot is not a facet plot, grid size should be 1x1
            if not facet:
                ncols = 1
                nrows = 1

            color_dict = dict(zip(timeseries.columns, self.color_list))

            # setup plot
            mplt = SetupSubplot(
                nrows, ncols, sharey=True, squeeze=False, ravel_axs=True
            )
            fig, axs = mplt.get_figure()
            plt.subplots_adjust(wspace=0.05, hspace=0.2)

            n = 0  # Counter for scenario subplots
            for column in timeseries:
                axs[n].plot(
                    timeseries[column],
                    linewidth=1,
                    color=color_dict[column],
                    label=column,
                    alpha=1,
                )
                if pd.notna(y_axis_max):
                    axs[n].set_ylim(bottom=0, top=float(y_axis_max))
                mplt.set_subplot_timeseries_format(sub_pos=n)
                if facet:
                    n += 1

            # Add legend
            mplt.add_legend()
            # Add title
            if plot_data_settings["plot_title_as_region"]:
                mplt.add_main_title(zone_input)
            plt.ylabel(
                f"{self.AGG_BY} Price ($/MWh)",
                color="black",
                rotation="vertical",
                labelpad=20,
            )
            plt.xlabel(timezone, color="black", rotation="horizontal", labelpad=20)

            outputs[zone_input] = {"fig": fig, "data_table": Data_Out}
        return outputs

    def timeseries_price_all_regions(
        self,
        y_axis_max: float = None,
        timezone: str = "",
        start_date_range: str = None,
        end_date_range: str = None,
        **_,
    ):
        """Creates a price timeseries plot for all regions/zones and plots them on a single facet plot.

        Price in $/MWh.
        The code automatically creates a facet plot based on the number of regions/zones in the input.
        All scenarios are plotted on a single facet for each region/zone.

        Args:
            y_axis_max (float, optional): Max y-axis value.
                Defaults to None.
            timezone (str, optional): The timezone to display on the x-axes.
                Defaults to "".
            start_date_range (str, optional): Defines a start date at which to represent data from.
                Defaults to None.
            end_date_range (str, optional): Defines a end date at which to represent data to.
                Defaults to None.

        Returns:
            dict: dictionary containing the created plot and its data table.
        """
        outputs: dict = {}

        if self.AGG_BY == "zone":
            agg = "zone"
        else:
            agg = "region"

        # List of properties needed by the plot, properties are a set of tuples and 
        # contain 3 parts: required True/False, property name and scenarios required, 
        # scenarios must be a list.
        properties = [(True, f"{agg}_Price", self.Scenarios)]

        # Runs get_formatted_data within MPlotDataHelper to populate MPlotDataHelper dictionary
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)

        if 1 in check_input_data:
            return MissingInputData()

        # Location to save to
        save_figures: Path = self.figure_folder.joinpath(f"{self.AGG_BY}_prices")

        region_number = len(self.Zones)
        ncols, nrows = self.set_x_y_dimension(region_number)

        grid_size = ncols * nrows
        # Used to calculate any excess axis to delete
        excess_axs = grid_size - region_number

        # setup plot
        mplt = SetupSubplot(nrows, ncols, sharey=True, squeeze=False, ravel_axs=True)
        fig, axs = mplt.get_figure()

        plt.subplots_adjust(wspace=0.1, hspace=0.70)

        data_table = []
        for n, zone_input in enumerate(self.Zones):
            logger.info(f"{self.AGG_BY} = {zone_input}")

            all_prices = []
            for scenario in self.Scenarios:
                price = self._process_data(self[f"{agg}_Price"], scenario, zone_input)
                price = price.groupby(["timestamp"]).sum()
                if pd.notna(start_date_range):
                    price = self.set_timestamp_date_range(
                        price, start_date_range, end_date_range
                    )
                all_prices.append(price)

            timeseries = pd.concat(all_prices, axis=1)
            timeseries.columns = timeseries.columns.str.replace("_", " ")

            data_out = timeseries.copy()
            data_out.columns = [zone_input + "_" + str(col) for col in data_out.columns]
            data_table.append(data_out)

            color_dict = dict(zip(timeseries.columns, self.color_list))

            for column in timeseries:
                axs[n].plot(
                    timeseries[column],
                    linewidth=1,
                    color=color_dict[column],
                    label=column,
                    alpha=1,
                )
                axs[n].set_title(zone_input.replace("_", " "))
                if pd.notna(y_axis_max):
                    axs[n].set_ylim(bottom=0, top=float(y_axis_max))
                mplt.set_subplot_timeseries_format(sub_pos=n)

        # Add legend
        mplt.add_legend()
        # Remove extra axes
        mplt.remove_excess_axs(excess_axs, grid_size)

        plt.ylabel(
            f"{self.AGG_BY} Price ($/MWh)",
            color="black",
            rotation="vertical",
            labelpad=30,
        )
        plt.xlabel(timezone, color="black", rotation="horizontal", labelpad=20)

        Data_Table_Out = pd.concat(data_table, axis=1)

        Data_Table_Out = Data_Table_Out.add_suffix(" ($/MWh)")

        fig.savefig(
            save_figures.joinpath("Price_Timeseries_All_Regions.svg"),
            dpi=600,
            bbox_inches="tight",
        )
        Data_Table_Out.to_csv(save_figures.joinpath("Price_Timeseries_All_Regions.csv"))
        outputs = DataSavedInModule()
        return outputs

    def node_pdc(self, **kwargs):
        """Creates a price duration curve for a set of specifc nodes.

        Price in $/MWh.
        The code will create either a facet plot or a single plot depending on
        the number of nodes included in plot_select.csv property entry.

        Returns:
            DataSavedInModule: DataSavedInModule exception.
        """
        outputs = self._node_price(PDC=True, **kwargs)
        return outputs

    def node_timeseries_price(self, **kwargs):
        """Creates a price timeseries plot for a set of specifc nodes.

        Price in $/MWh.
        The code will create either a facet plot or a single plot depending on
        the number of nodes included in plot_select.csv property entry.

        Returns:
            DataSavedInModule: DataSavedInModule exception.
        """
        outputs = self._node_price(**kwargs)
        return outputs

    def _node_price(
        self,
        PDC: bool = False,
        figure_name: str = None,
        prop: str = None,
        y_axis_max: float = None,
        timezone: str = "",
        start_date_range: str = None,
        end_date_range: str = None,
        **_,
    ):
        """Creates a price duration curve or timeseries plot for a set of specifc nodes.

        This method is called from either node_pdc() or node_timeseries_price()

        If PDC == True, a price duration curve plot will be created
        The code will create either a facet plot or a single plot depending on
        the number of nodes included in plot_select.csv property entry.
        Plots and Data are saved within the module

        Args:
            PDC (bool, optional): If True creates a price duration curve.
                Defaults to False.
            figure_name (str, optional): User defined figure output name.
                Defaults to None.
            prop (str, optional): comma seperated string of nodes to display.
                Defaults to None.
            y_axis_max (float, optional): Max y-axis value.
                Defaults to None.
            timezone (str, optional): The timezone to display on the x-axes.
                Defaults to "".
            start_date_range (str, optional): Defines a start date at which to represent data from.
                Defaults to None.
            end_date_range (str, optional): Defines a end date at which to represent data to.
                Defaults to None.

        Returns:
            DataSavedInModule: DataSavedInModule exception.
        """
        # List of properties needed by the plot, properties are a set of tuples and 
        # contain 3 parts: required True/False, property name and scenarios required, 
        # scenarios must be a list.
        properties = [(True, "node_Price", self.Scenarios)]

        # Runs get_formatted_data within MPlotDataHelper to populate MPlotDataHelper dictionary
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)

        if 1 in check_input_data:
            return MissingInputData()

        node_figure_folder: Path = self.figure_folder.joinpath("node_prices")
        node_figure_folder.mkdir(exist_ok=True)

        # Select only node specified in Marmot_plot_select.csv.
        select_nodes = prop.split(",")
        if select_nodes == None:
            return InputSheetError()

        logger.info(f"Plotting Prices for {select_nodes}")

        all_prices = []
        for scenario in self.Scenarios:
            logger.info(f"Scenario = {scenario}")

            price: pd.DataFrame = self["node_Price"][scenario]
            price = price.loc[(slice(None), select_nodes), :]
            price = price.groupby(["timestamp", "node"]).sum()
            price.rename(columns={0: scenario}, inplace=True)
            if pd.notna(start_date_range):
                price = self.set_timestamp_date_range(
                    price, start_date_range, end_date_range
                )
            if PDC:
                price.sort_values(by=["node", scenario], ascending=False, inplace=True)
                price.reset_index("timestamp", drop=True, inplace=True)
            all_prices.append(price)

        pdc = pd.concat(all_prices, axis=1)
        pdc.columns = pdc.columns.str.replace("_", " ")

        Data_Out = pdc.add_suffix(" ($/MWh)")

        ncols, nrows = self.set_x_y_dimension(len(select_nodes))

        # setup plot
        mplt = SetupSubplot(nrows, ncols, sharey=True, squeeze=False, ravel_axs=True)
        fig, axs = mplt.get_figure()
        plt.subplots_adjust(wspace=0.1, hspace=0.70)

        color_dict = dict(zip(pdc.columns, self.color_list))

        for n, node in enumerate(select_nodes):

            if PDC:
                try:
                    node_pdc = pdc.xs(node)
                    node_pdc.reset_index(drop=True, inplace=True)
                except KeyError:
                    logger.info(f"{node} not found")
                    continue
            else:
                try:
                    node_pdc = pdc.xs(node, level="node")
                except KeyError:
                    logger.info(f"{node} not found")
                    continue

            for column in node_pdc:
                axs[n].plot(
                    node_pdc[column],
                    linewidth=1,
                    color=color_dict[column],
                    label=column,
                    alpha=1,
                )
                if pd.notna(y_axis_max):
                    axs[n].set_ylim(bottom=0, top=float(y_axis_max))
                if not PDC:
                    mplt.set_subplot_timeseries_format(sub_pos=n)
                # axs[n].set_xlim(0,len(node_pdc))

        mplt.add_legend()
        plt.ylabel(
            "Node Price ($/MWh)", color="black", rotation="vertical", labelpad=30
        )
        if PDC:
            plt.xlabel("Intervals", color="black", rotation="horizontal", labelpad=20)
        else:
            plt.xlabel(timezone, color="black", rotation="horizontal", labelpad=20)

        fig.savefig(
            node_figure_folder.joinpath(f"{figure_name}.svg"),
            dpi=600,
            bbox_inches="tight",
        )
        Data_Out.to_csv(node_figure_folder.joinpath(f"{figure_name}.csv"))
        outputs = DataSavedInModule()
        return outputs

    def node_price_hist(self, **kwargs):
        """Creates a price histogram for a specifc nodes. Price in $/MWh.

        A facet plot will be created if more than one scenario are included on the
        user input sheet
        Each scenario will be plotted on a separate subplot.
        If a set of nodes are passed at input, each will be saved to a separate
        figure with node name as a suffix.
        Plots and Data are saved within the module

        Returns:
            DataSavedInModule: DataSavedInModule exception.
        """
        outputs = self._node_hist(**kwargs)
        return outputs

    def node_price_hist_diff(self, **kwargs):
        """Creates a difference price histogram for a specifc nodes. Price in $/MWh.

        This plot requires more than one scenario to display correctly.
        A facet plot will be created
        Each scenario will be plotted on a separate subplot, with values displaying
        the relative difference to the first scenario in the list.
        If a set of nodes are passed at input, each will be saved to a separate
        figure with node name as a suffix.
        Plots and Data are saved within the module

        Returns:
            DataSavedInModule: DataSavedInModule exception.
        """
        outputs = self._node_hist(diff_plot=True, **kwargs)
        return outputs

    def _node_hist(
        self,
        diff_plot: bool = False,
        figure_name: str = None,
        prop: str = None,
        start_date_range: str = None,
        end_date_range: str = None,
        **_,
    ):
        """Internal code for hist plots.

        Called from node_price_hist() or node_price_hist_diff().

        Hist range and bin size is currently hardcoded from -100 to +100
        with a bin width of 2.5 $/MWh

        Args:
            diff_plot (bool, optional): If True creates a diff plot.
                Defaults to False.
            figure_name (str, optional): User defined figure output name.
                Defaults to None.
            prop (str, optional): comma seperated string of nodes to display.
                Defaults to None.
            start_date_range (str, optional): Defines a start date at which to represent data from.
                Defaults to None.
            end_date_range (str, optional): Defines a end date at which to represent data to.
                Defaults to None.

        Returns:
            DataSavedInModule: DataSavedInModule exception.
        """
        # List of properties needed by the plot, properties are a set of tuples and 
        # contain 3 parts: required True/False, property name and scenarios required, 
        # scenarios must be a list.
        properties = [(True, "node_Price", self.Scenarios)]

        # Runs get_formatted_data within MPlotDataHelper to populate MPlotDataHelper dictionary
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)

        if 1 in check_input_data:
            return MissingInputData()

        node_figure_folder: Path = self.figure_folder.joinpath("node_prices")
        node_figure_folder.mkdir(exist_ok=True)

        # Select only node specified in Marmot_plot_select.csv.
        select_nodes = prop.split(",")
        if select_nodes == None:
            return InputSheetError()

        for node in select_nodes:
            logger.info(f"Plotting Prices for Node: {node}")

            all_prices = []
            for scenario in self.Scenarios:
                logger.info(f"Scenario = {scenario}")

                price: pd.DataFrame = self["node_Price"][scenario]
                try:
                    price = price.xs(node, level="node")
                except KeyError:
                    logger.info(f"{node} not found")
                    continue

                price = price.groupby(["timestamp"]).sum()
                price.rename(columns={0: scenario}, inplace=True)
                if pd.notna(start_date_range):
                    price = self.set_timestamp_date_range(
                        price, start_date_range, end_date_range
                    )

                price.reset_index("timestamp", drop=True, inplace=True)
                all_prices.append(price)

            if not all_prices:
                logger.info(f"Nodes not found in database, input sheet error likely!")
                return InputSheetError()

            p_hist = pd.concat(all_prices, axis=1)

            if diff_plot:
                p_hist = p_hist.subtract(p_hist[f"{self.Scenarios[0]}"], axis=0)

            p_hist.columns = p_hist.columns.str.replace("_", " ")
            data_out = p_hist.add_suffix(" ($/MWh)")

            ncols, nrows = self.set_facet_col_row_dimensions(
                multi_scenario=self.Scenarios
            )
            grid_size = ncols * nrows
            # Used to calculate any excess axis to delete
            plot_number = len(self.Scenarios)
            excess_axs = grid_size - plot_number

            # setup plot
            mplt = SetupSubplot(
                nrows, ncols, sharey=True, squeeze=False, ravel_axs=True
            )
            fig, axs = mplt.get_figure()
            plt.subplots_adjust(wspace=0.1, hspace=0.25)

            color_dict = dict(zip(p_hist.columns, self.color_list))

            # max, min values in histogram range and bin width
            # TODO: Determine a way to pass the following as an input.
            range_max = 100
            range_min = -100
            bin_width = 2.5

            # no of bines
            bins = int((range_max + abs(range_min)) / bin_width)

            for n, column in enumerate(p_hist):

                # Set plot data equal to 0 if all zero, e.g diff plot
                if sum(p_hist[column]) == 0:
                    data = 0
                else:
                    data = p_hist[column]
                # values above range_max and below range_min are binned together
                axs[n].hist(
                    np.clip(data, range_min, range_max),
                    bins=bins,
                    range=(range_min, range_max),
                    color=color_dict[column],
                    zorder=2,
                    rwidth=0.8,
                )
                # get xlabels and edit them
                xticks = axs[n].get_xticks()
                # min range_min, max range_max
                xticks = np.unique(np.clip(xticks, range_min, range_max))
                xlabels = xticks.astype(int).astype(str)
                # adds a '+' to final xlabel
                xlabels[-1] += "+"
                if min(xticks) < 0:
                    xlabels[0] += "-"
                # sets x_tick spacing
                axs[n].set_xticks(xticks)
                axs[n].set_xticklabels(xlabels)

            # Remove extra axes
            mplt.remove_excess_axs(excess_axs, grid_size)
            # Add Facet Labels
            mplt.add_facet_labels(xlabels=self.xlabels, ylabels=self.ylabels)
            plt.ylabel(
                "Occurrence",
                color="black",
                rotation="vertical",
                labelpad=60,
                fontsize=24,
            )
            mplt.add_main_title(node)
            if diff_plot:
                plt.xlabel(
                    f"Node LMP Change ($/MWh) relative to {self.Scenarios[0].replace('_',' ')}",
                    color="black",
                    labelpad=40,
                )
            else:
                plt.xlabel("Node LMP ($/MWh)", color="black", labelpad=40)

            fig.savefig(
                node_figure_folder.joinpath(f"{figure_name}_{node}.svg"),
                dpi=600,
                bbox_inches="tight",
            )
            data_out.to_csv(node_figure_folder.joinpath(f"{figure_name}_{node}.csv"))

        outputs = DataSavedInModule()
        return outputs

    def _process_data(
        self, data_collection: dict, scenario: str, zone_input: str
    ) -> pd.DataFrame:
        df: pd.DataFrame = data_collection.get(scenario)
        df = df.xs(zone_input, level=self.AGG_BY)
        df = df.rename(columns={0: scenario})
        return df