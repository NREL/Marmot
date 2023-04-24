# -*- coding: utf-8 -*-
"""Energy storage plots.

This module creates energy storage plots.
"""

import datetime as dt
import logging
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import marmot.utils.mconfig as mconfig
from marmot.plottingmodules.plotutils.plot_data_helper import (
    GenCategories,
    PlotDataStoreAndProcessor,
    set_facet_col_row_dimensions,
)
from marmot.plottingmodules.plotutils.plot_exceptions import (
    InputSheetError,
    MissingInputData,
    MissingZoneData,
)
from marmot.plottingmodules.plotutils.plot_library import PlotLibrary, SetupSubplot
from marmot.plottingmodules.plotutils.styles import GeneratorColorDict
from marmot.plottingmodules.plotutils.timeseries_modifiers import (
    adjust_for_leapday,
    set_timestamp_date_range,
)

logger = logging.getLogger("plotter." + __name__)
plot_data_settings: dict = mconfig.parser("plot_data")
shift_leapday: bool = mconfig.parser("shift_leapday")
load_legend_names: dict = mconfig.parser("load_legend_names")
curtailment_prop: str = mconfig.parser("plot_data", "curtailment_property")


#####
#Testing
#####
# import os
# os.chdir('/Users/mschwarz/CCS_local/ReEDS2_PLEXOS_link/Marmot')

# self = FLECCSOperation(
#     Zones = ['p97'],
#     AGG_BY = 'region',
#     Scenarios = ['Envergex_Ref150','Envergex_Ref225','Envergex_Ref300'],
#     ordered_gen = ['Nuclear', 'Coal', 'Gas-CC', 'Gas-CC CCS', 'Gas-CT', 'Gas', 'Gas-Steam', 'Dual Fuel', 'DualFuel', 'Oil-Gas-Steam', 'Oil', 'Hydro', 'Ocean', 'Geothermal', 'Biomass', 'Biopower', 'Other', 'VRE', 'Wind', 'Offshore Wind', 'OffshoreWind', 'Solar', 'PV', 'dPV', 'CSP', 'PV-Battery', 'Battery', 'OSW-Battery', 'PHS', 'Storage', 'Net Imports', 'Curtailment', 'curtailment', 'Demand', 'Deamand + Storage Charging'],
#     marmot_solutions_folder = '/Users/mschwarz/CCS_local'
# )
# start_date_range='5/7/50'
# end_date_range='5/14/50'
#prop = 'Envergex_p58_12'
#prop = 'Envergex_p92_2'

#######

class FLECCSOperation(PlotDataStoreAndProcessor):
    """The FLECCS_operation.py module contains methods that are
    related to NGCC plants fitted with a flexible CCS technology, 
    specifically designed for the ARPA-E FLECCS project.

    FLECCS_operation inherits from the MPlotDataHelper class to assist
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

    def FLECCS_timeseries_singleplant(self,**kwargs):

        return self._FLECCS_timeseries(single_plant=True,**kwargs)

    def _FLECCS_timeseries(
        self,
        timezone: str = "",
        start_date_range: str = None,
        end_date_range: str = None,
        prop: str = None,
        single_plant: bool = False,
        **_,
    ):
        """Creates time series plot of aggregate storage volume for all storage objects in a given region.

        A horizontal line represents full charge of the storage device.
        All scenarios are plotted on a single figure.

        Args:
            timezone (str, optional): The timezone to display on the x-axes.
                Defaults to "".
            start_date_range (str, optional): Defines a start date at which to represent data from.
                Defaults to None.
            end_date_range (str, optional): Defines a end date at which to represent data to.
                Defaults to None.
            prop (str, optional): Specificies FLECCS technology as necessary.

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
        properties = [
            #(False, "storage_Initial_Volume", self.Scenarios),  #Releveant for storage techs
            #(False, "waterway_Flow", self.Scenarios), #Represents CCS load.
            (True, "generator_Generation", self.Scenarios),  #For the base NGCC
            (True, "region_Price", self.Scenarios),
            (True, "generator_Pump_Load", self.Scenarios)
        ]

        # Runs get_formatted_data within MPlotDataHelper to populate MPlotDataHelper dictionary
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)

        if 1 in check_input_data:
            return MissingInputData()

        for zone_input in self.Zones:
            logger.info(f"{self.AGG_BY} = {zone_input}")

            # sets up x, y dimensions of plot
            ncols, nrows = set_facet_col_row_dimensions(
                multi_scenario=self.Scenarios
            )
            #mplt = SetupSubplot(nrows=len(self.Scenarios), squeeze=False, ravel_axs=True)
            mplt = PlotLibrary(nrows, ncols, sharey=True, squeeze=False, ravel_axs=True)
            fig, axs = mplt.get_figure()
            plt.subplots_adjust(wspace=0.05, hspace=0.2)

            for i,scenario in enumerate(self.Scenarios):

                logger.info(f"Scenario = {str(scenario)}")

                #####
                #Pull storage volume.
                #####
                # storage_volume_read = self["storage_Initial_Volume"].get(scenario)
                # try:
                #     storage_volume: pd.DataFrame = storage_volume_read.xs(
                #         zone_input, level=self.AGG_BY
                #     )
                # except KeyError:
                #     logger.warning(f"No storage resources in {zone_input}")
                #     outputs[zone_input] = MissingZoneData()
                #     continue

                # # Isolate only FLECCS head storage objects (no actual storage plants and no tail).
                # storage_gen_units = storage_volume.index.get_level_values(
                #     "storage_resource"
                # )
                # if single_plant:
                #     single_stor = prop.split('_')[0] + '_head_' + prop.split('_')[1] + '_' + prop.split('_')[2]
                #     FLECCS_units = [unit for unit in storage_gen_units if single_stor in unit]
                # else:
                #     FLECCS_units = [unit for unit in storage_gen_units if f"{prop}_head" in unit]

                # storage_volume = storage_volume.iloc[
                #     storage_volume.index.get_level_values("storage_resource").isin(
                #         FLECCS_units
                #     )
                # ]
                # storage_volume = storage_volume.groupby("timestamp").sum()
                # storage_volume.columns = ['Solvent storage volume \n (MWh equivalent)']

                # ######
                # #Pull waterway flow
                # #####

                # waterway_flow = self["waterway_Flow"].get(scenario)
                # if single_plant:
                #     single_ww = prop.split('_')[0] + '_waterway_' + prop.split('_')[1] + '_' + prop.split('_')[2]
                #     waterway_flow = waterway_flow.iloc[
                #         waterway_flow.index.get_level_values("waterway_name").isin([single_ww])
                #     ] #Downselect specific unit  
                # waterway_flow.columns = ['CCS load']

                ######
                #Pull NGCC generation
                #####

                gen = self["generator_Generation"].get(scenario)
                if single_plant:
                    #FLECCS_gens = prop.split('_')[0] + '_NGCC_' + prop.split('_')[1] + '_' + prop.split('_')[2]
                    FLECCS_gens = 'Georgia_Tech_NGCC_p58_12'
                else:
                    all_gens = gen.index.get_level_values('gen_name').unique()
                    FLECCS_gens = [g for g in all_gens if 'Envergex_NGCC' in g]
                NGCCgen = gen.iloc[
                    gen.index.get_level_values("gen_name").isin([FLECCS_gens])
                ] #Downselect specific unit
                NGCCgen.columns = ['Net power to the grid']

                ######
                #Pull region price
                #####

                price = self["region_Price"].get(scenario)
                price = price.iloc[
                    price.index.get_level_values("region").isin([zone_input])
                ] #Downselect specific unit  
                price.columns = ['Energy price ($/MWh)']

                ######
                #Pull pump load
                #####

                pump = self["generator_Pump_Load"].get(scenario)
                FLECCS_pump = prop.split('_')[0] + '_pump_' + prop.split('_')[1] + '_' + prop.split('_')[2]
                FLECCS_pump = 'Georgia_Tech_pump_p58_12'
                pump = pump.iloc[
                    pump.index.get_level_values("gen_name").isin([FLECCS_pump])
                ] #Downselect specific unit  
                pump.columns = ['Storage charging']
                pump.columns = ['DAC load']


                if pd.notna(start_date_range):
                    # storage_volume, waterway_flow, NGCCgen, pump, price = self.set_timestamp_date_range(
                    #     [storage_volume, waterway_flow, NGCCgen, pump, price],
                    #     start_date_range,
                    #     end_date_range,
                    # )
                    NGCCgen, pump, price = set_timestamp_date_range(
                        [NGCCgen, pump, price],
                        start_date_range,
                        end_date_range,
                    )

                    # if storage_volume.empty is True:
                    #     logger.warning("No Storage resources in selected Date Range")
                    #     continue
                    # if waterway_flow.empty is True:
                    #     logger.warning("No waterway flow in selected Date Range")
                    #     continue
                    if NGCCgen.empty is True:
                        logger.warning("No NGCCgen in selected Date Range")
                        continue
                    if pump.empty is True:
                        logger.warning("No pump in selected Date Range")
                        continue

                #df=pd.merge(NGCCgen,waterway_flow,on=['timestamp'])
                df=pd.merge(NGCCgen,pump,on=['timestamp'])
                #Subtract pump load from gen to get net power to the grid.
                #df['Net power to the grid'] = df['Net power to the grid'] - df['Storage charging']
                #df=df.merge(storage_volume,on=['timestamp'])
                df=df.merge(price*10,on=['timestamp'])

                # Data table of values to return to main program
                Data_Table_Out = df.copy()

                if df.empty:
                    out = MissingZoneData()
                    outputs[zone_input] = out
                    continue

                for col in df.columns:
                    mplt.lineplot(
                        df,
                        col,
                        linewidth=1,
                        linestyle='dashed' if col == 'Energy price ($/MWh)' else 'solid',
                        label=col,
                        sub_pos=i
                    )
                axs[0].set_ylabel(
                    "Power (MW)", color="black", rotation="vertical"
                )

                def power2price(x):
                    return x/10
                def price2power(x):
                    return x*10
                if i == len(self.Scenarios)-1:
                    secax = axs[i].secondary_yaxis('right',functions=(power2price,price2power))
                    secax.set_ylabel('Price ($/MWh)')

                mplt.set_yaxis_major_tick_format(sub_pos=i)
                axs[i].margins(x=0.01)
                mplt.set_subplot_timeseries_format(sub_pos=i)
                axs[i].set_ylim(ymin=0,ymax=2000)
                axs[i].set_title(scenario.split('_')[1])

                axs[len(self.Scenarios)-1].legend(loc="lower left", bbox_to_anchor=(1.15, 0))
                if plot_data_settings["plot_title_as_region"]:
                    if single_plant:
                        mplt.add_main_title(prop)
                    else:
                        mplt.add_main_title(zone_input)

            outputs[zone_input] = {"fig": fig, "data_table": Data_Table_Out}
        return outputs
