# -*- coding: utf-8 -*-
"""Methods used to assist with the creation of Marmot plots.

@author: Daniel Levie
"""

import os
import math
import textwrap
import logging
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
import functools
import concurrent.futures
from matplotlib.axes import Axes
from typing import Tuple

import marmot.config.mconfig as mconfig

logger = logging.getLogger('marmot_plot.'+__name__)


class PlotDataHelper(dict):
    """Methods used to assist with the creation of Marmot plots

    Collection of Methods to assist with creation of figures,
    including getting and formatting data, setting up plot sizes and adding 
    elements to plots such as labels.

    PlotDataHelper inherits the python class 'dict' so acts like a dictionary and stores the
    formatted data when retrieved by the get_formatted_data method.
    """

    def __init__(self, Marmot_Solutions_folder: str, AGG_BY: str, ordered_gen: list, 
                 PLEXOS_color_dict: dict, Scenarios: list, ylabels: list, 
                 xlabels: list, gen_names_dict: dict,
                 Region_Mapping: pd.DataFrame = pd.DataFrame()):
        """
        Args:
            Marmot_Solutions_folder (str): Folder to save Marmot solution files.
            AGG_BY (str): Informs region type to aggregate by when creating plots.
            ordered_gen (list): Ordered list of generator technologies to plot, 
                order defines the generator technology position in stacked bar and area plots
            PLEXOS_color_dict (dict): Dictionary of colors to use for generation technologies
            Scenarios (list): Name of scenarios to process.
            ylabels (list): y-axis labels for facet plots.
            xlabels (list): x-axis labels for facet plots.
            gen_names_dict (dict): Mapping dictionary to rename generator technologies.
            Region_Mapping (pd.DataFrame, optional): Mapping file to map custom regions/zones 
                to create custom aggregations. Aggregations are created by grouping PLEXOS regions.
                Defaults to pd.DataFrame().
        """
        self.Marmot_Solutions_folder = Marmot_Solutions_folder
        self.AGG_BY = AGG_BY
        self.ordered_gen = ordered_gen
        self.PLEXOS_color_dict = PLEXOS_color_dict
        self.Scenarios = Scenarios
        self.ylabels = ylabels
        self.xlabels = xlabels
        self.gen_names_dict = gen_names_dict
        self.Region_Mapping = Region_Mapping

    def get_formatted_data(self, properties: list) -> list:
        """Get data from formatted h5 file.
        
        Adds data to dictionary with scenario name as key

        Args:
            properties (list): list of tuples containing required 
                plexos property information

        Returns:
            list: If 1 in list required data is missing .
        """
        check_input_data = []
        
        for prop in properties:
            required, plx_prop_name, scenario_list = prop
            if f"{plx_prop_name}" not in self:
                self[f"{plx_prop_name}"] = {}
            
            # Create new set of scenarios that are not yet in dictionary
            scen_list = set(scenario_list) - set(self[f"{plx_prop_name}"].keys())
            
            # If set is not empty add data to dict
            if scen_list:
                #Read data in with multi threading
                executor_func_setup = functools.partial(self.read_processed_h5file, plx_prop_name)
                with concurrent.futures.ThreadPoolExecutor(max_workers=mconfig.parser("multithreading_workers")) as executor:
                    data_files = executor.map(executor_func_setup, scen_list)
                # Save data to dict
                for scenario, df in zip(scen_list, data_files):
                    self[f"{plx_prop_name}"][scenario] = df

            # If any of the dataframes are empty for given property log warning
            if True in [df.empty for df in self[f"{plx_prop_name}"].values()]:
                logger.warning(f"{plx_prop_name} is MISSING from the Marmot formatted h5 files")
                if required == True:
                    check_input_data.append(1)
        return check_input_data

    def read_processed_h5file(self, plx_prop_name: str, scenario: str) -> pd.DataFrame:
        """Reads Data from processed h5file.

        Args:
            plx_prop_name (str): Name of property, e.g generator_Generation
            scenario (str): Name of scenario.

        Returns:
            pd.DataFrame: Requested dataframe.
        """
        try:
            with pd.HDFStore(os.path.join(self.Marmot_Solutions_folder, "Processed_HDF5_folder", 
                                            f"{scenario}_formatted.h5"), 'r') as file:
                return file[plx_prop_name]
        except KeyError:
            return pd.DataFrame()
        
    def rename_gen_techs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Renames generator technologies based on the gen_names.csv file.

        Args:
            df (pd.DataFrame): Dataframe to process.

        Returns:
            pd.DataFrame: Processed DataFrame with renamed techs.
        """
        # If tech is a column name
        if 'tech' in df.columns:
            original_tech_index = df.tech.unique()
            # Checks if all generator tech categories have been identified and matched. If not, lists categories that need a match
            unmapped_techs = set(original_tech_index) - set(self.gen_names_dict.keys())
            df['tech'] = pd.CategoricalIndex(df.tech.map(lambda x: self.gen_names_dict.get(x, 'Other')))
        
        # If tech is in the index 
        elif 'tech' in df.index.names:
            original_tech_index = df.index.get_level_values(level='tech')
            # Checks if all generator tech categories have been identified and matched. If not, lists categories that need a match
            unmapped_techs = set(original_tech_index) - set(self.gen_names_dict.keys())
        
            tech_index = pd.CategoricalIndex(original_tech_index.map(lambda x: self.gen_names_dict.get(x, 'Other')))
            df.reset_index(level='tech', drop=True, inplace=True)

            idx_map = pd.MultiIndex(levels=df.index.levels + [tech_index.categories],
                                        codes=df.index.codes + [tech_index.codes],
                                        names=df.index.names + tech_index.names)

            df = pd.DataFrame(data=df.values.reshape(-1), index=idx_map)
            # Move tech back to position 1
            index_labels = list(df.index.names)
            index_labels.insert(1, index_labels.pop(index_labels.index("tech")))
            df = df.reorder_levels(index_labels, axis=0)

        if unmapped_techs:
            logger.warning(f"The following Generators could not be re-classified, they wil be renamed 'Other': {unmapped_techs}")
        return df

    def assign_curtailment_techs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign technologies to Marmot's Curtailment property (generator_Curtailment).

        Args:
            df (pd.DataFrame): Dataframe to process.

        Returns:
            pd.DataFrame: Dataframe containing only specified curtailment technologies.
        """

        # Adjust list of values to drop from vre_gen_cat depending on if it exists in processed techs
        adjusted_vre_gen_list = [name for name in self.vre_gen_cat if name in df.columns]

        if not adjusted_vre_gen_list:
            logger.warning("Curtailment techs could not be identified correctly for Marmot's Curtailment property. "
            "This is likely happening as the 'vre' column was not present in the ordered_gen_categories.csv or there "
            "are no vre generators in the selected region")
            return df
        else: 
            # Retrun df with just vre techs
            return df[df.columns.intersection(self.vre_gen_cat)]

    def df_process_gen_inputs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Processes generation data into a pivot table. 

        Also calls rename_gen_techs() to rename technologies 
        Technology names will be columns,
        Timeseries as index

        Args:
            df (pd.DataFrame): Dataframe to process.

        Returns:
            pd.DataFrame: Transformed Dataframe.
        """
        if set(['timestamp','tech']).issubset(df.index.names):
            df = df.reset_index(['timestamp','tech'])
        df = df.groupby(["timestamp", "tech"], as_index=False, observed=True).sum()
        # Rename generator technologies
        df = self.rename_gen_techs(df)
        # If duplicate rows remain, groupby again
        if df[["timestamp", "tech"]].duplicated().any():
            df = df.groupby(["timestamp", "tech"], as_index=False, observed=True).sum()
        # Filter for only data in ordered_gen
        df = df[df.tech.isin(self.ordered_gen)]
        # Check if data is not already categorical
        if df.tech.dtype.name != "category":
            df.tech = df.tech.astype("category")
        df.tech.cat.set_categories(self.ordered_gen, inplace=True)
        df = df.sort_values(["tech"])
        df = df.pivot(index='timestamp', columns='tech', values=0)
        return df

    def create_categorical_tech_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Creates categorical index based on generators.

        Args:
            df (pd.DataFrame): Dataframe to process.

        Returns:
            pd.DataFrame: Processed DataFrame.
        """
        df.index = df.index.astype("category")
        df.index = df.index.set_categories(self.ordered_gen)
        df = df.sort_index()
        return df

    def merge_new_agg(self, df: pd.DataFrame) -> pd.DataFrame:
        #TODO Needs fixing 
        """Adds new region aggregation in the plotting step.

        This allows one to create a new aggregation without re-formatting the .h5 file.
        Args:
            df (pd.DataFrame): Dataframe to process.

        Returns:
            pd.DataFrame: Same dataframe, with new aggregation level added.
        """
        agg_new = self.Region_Mapping[['region', self.AGG_BY]]
        agg_new = agg_new.set_index('region')
        df = df.merge(agg_new,left_on = 'region', right_index = True)
        return(df)

    def adjust_for_leapday(self, df: pd.DataFrame) -> pd.DataFrame:
        """Shifts dataframe ahead by one day, if a non-leap year time series is modeled with a leap year time index.

        Modeled year must be included in the scenario parent directory name.
        Args:
            df (pd.DataFrame): Dataframe to process.

        Returns:
            pd.DataFrame: Same dataframe, with time index shifted.
        """
        if ('2008' not in self.Marmot_Solutions_folder 
            and '2012' not in self.Marmot_Solutions_folder 
            and df.index.get_level_values('timestamp')[0] > dt.datetime(2024,2,28,0,0)):
            
            df.index = df.index.set_levels(
                df.index.levels[df.index.names.index('timestamp')].shift(1,freq = 'D'),
                level = 'timestamp')
            
        # # Special case where timezone shifting may also be necessary.
        #     df.index = df.index.set_levels(
        #         df.index.levels[df.index.names.index('timestamp')].shift(-3,freq = 'H'),
        #         level = 'timestamp')

        return df

    def setup_facet_xy_dimensions(self, facet: bool = True, 
                                  multi_scenario: list = None) -> Tuple[int, int]:
        """Sets facet plot x,y dimensions based on user defined labeles

        Args:
            facet (bool, optional): Trigger for plotting facet plots. 
                Defaults to True.
            multi_scenario (list, optional): List of scenarios.
                Defaults to None.

        Returns:
            Tuple[int, int]: Facet x,y dimensions.
        """
        xdimension=len(self.xlabels)
        if self.xlabels == ['']:
            xdimension = 1
        ydimension=len(self.ylabels)
        if self.ylabels == ['']:
            ydimension = 1
        # If the plot is not a facet plot, grid size should be 1x1
        if not facet:
            xdimension = 1
            ydimension = 1
            return xdimension, ydimension
        # If no labels were provided or dimensions less than len scenarios use Marmot default dimension settings
        if self.xlabels == [''] and self.ylabels == [''] or xdimension*ydimension<len(multi_scenario):
            logger.info("Dimensions could not be determined from x & y labels - Using Marmot default dimensions")
            xdimension, ydimension = self.set_x_y_dimension(len(multi_scenario))
        return xdimension, ydimension

    def set_x_y_dimension(self, region_number: int) -> Tuple[int, int]:
        """Sets X,Y dimension of plots without x,y labels.

        Args:
            region_number (int): # regions/scenarios

        Returns:
            Tuple[int, int]: Facet x,y dimensions.
        """
        if region_number >= 5:
            xdimension = 3
            ydimension = math.ceil(region_number/3)
        if region_number <= 3:
            xdimension = region_number
            ydimension = 1
        if region_number == 4:
            xdimension = 2
            ydimension = 2
        return xdimension,ydimension

    def add_facet_labels(self, fig, 
                         xlabels_bottom: bool = True,
                         alternative_xlabels: list = None,
                         alternative_ylabels: list = None,
                         **kwargs) -> None:
        """Adds labels to outside of Facet plot.

        Args:
            fig (matplotlib.fig): matplotlib figure.
            xlabels_bottom (bool, optional): If True labels are placed under bottom. 
                Defaults to True.
            alternative_xlabels (list, optional): Alteranative xlabels. 
                Defaults to None.
            alternative_ylabels (list, optional): Alteranative ylabels. 
                Defaults to None.
        """
        font_defaults = mconfig.parser("font_settings")

        if alternative_xlabels:
            xlabel = alternative_xlabels
        else:
            xlabel = self.xlabels

        if alternative_ylabels:
            ylabel = alternative_ylabels
        else:
            ylabel = self.ylabels

        all_axes = fig.get_axes()
        j=0
        k=0
        for ax in all_axes:
            if xlabels_bottom:
                if ax.is_last_row():
                    try:
                        ax.set_xlabel(xlabel=(xlabel[j]), color='black', 
                                    fontsize=font_defaults['axes_label_size']-2, **kwargs)
                    except IndexError:
                        logger.warning(f"Warning: xlabel missing for subplot x{j}")
                        continue
                    j=j+1
            else:
                if ax.is_first_row():
                    try:
                        ax.set_xlabel(xlabel=(xlabel[j]), color='black', 
                                    fontsize=font_defaults['axes_label_size']-2, **kwargs)
                        ax.xaxis.set_label_position('top')
                    except IndexError:
                        logger.warning(f"Warning: xlabel missing for subplot x{j}")
                        continue
                    j=j+1
            if ax.is_first_col():
                try:
                    ax.set_ylabel(ylabel=(ylabel[k]), color='black', rotation='vertical', 
                                    fontsize=font_defaults['axes_label_size']-2, **kwargs)
                except IndexError:
                    logger.warning(f"Warning: ylabel missing for subplot y{k}")
                    continue
                k=k+1

    @staticmethod
    def set_legend_position(axs: Axes, handles=None, labels=None, 
                            loc=mconfig.parser("axes_options", "legend_position"),
                            ncol=mconfig.parser("axes_options", "legend_columns"),
                            reverse_legend=True, bbox_to_anchor=None,
                            facecolor='inherit', frameon=True, **kwargs):
        
        loc_anchor = {'lower right': ('lower left', (1.05, 0.0)),
                      'center right': ('center left', (1.05, 0.5)),
                      'upper right': ('upper left', (1.05, 1.0)),
                      'upper center': ('lower center', (0.5, 1.25)),
                      'lower center': ('upper center', (0.5, -0.25)),
                      'lower left': ('lower right', (-0.2, 0.0)),
                      'center left': ('center right', (-0.2, 0.5)),
                      'upper left': ('upper right', (-0.2, 1.0))}

        if handles == None or labels == None:
            handles, labels = axs.get_legend_handles_labels()
        if reverse_legend:
            handles = reversed(handles)
            labels = reversed(labels)

        if loc in loc_anchor:
            bbox_to_anchor = loc_anchor.get(loc, None)[1]
            new_loc = loc_anchor.get(loc, None)[0]
        else:
            bbox_to_anchor = bbox_to_anchor
            new_loc = loc

        axs.legend(handles, labels, loc=new_loc, ncol=ncol,
                    bbox_to_anchor=bbox_to_anchor, facecolor=facecolor, 
                    frameon=frameon, **kwargs)



    @staticmethod
    def set_plot_timeseries_format(axs, n: int = 0,
                                   minticks: int = mconfig.parser("axes_options","x_axes_minticks"),
                                   maxticks: int = mconfig.parser("axes_options","x_axes_maxticks")
                                   ) -> None:
        """Auto sets timeseries format.

        Args:
            axs (matplotlib.axes): matplotlib.axes
            n (int, optional): Counter for facet plot. Defaults to 0.
            minticks (int, optional): Minimum tick marks. 
                Defaults to mconfig.parser("axes_options","x_axes_minticks").
            maxticks (int, optional): Max tick marks. 
                Defaults to mconfig.parser("axes_options","x_axes_maxticks").
        """
        locator = mdates.AutoDateLocator(minticks=minticks, maxticks=maxticks)
        formatter = mdates.ConciseDateFormatter(locator)
        formatter.formats[2] = '%d\n %b'
        formatter.zero_formats[1] = '%b\n %Y'
        formatter.zero_formats[2] = '%d\n %b'
        formatter.zero_formats[3] = '%H:%M\n %d-%b'
        formatter.offset_formats[3] = '%b %Y'
        formatter.show_offset = False
        axs[n].xaxis.set_major_locator(locator)
        axs[n].xaxis.set_major_formatter(formatter)

    @staticmethod
    def set_barplot_xticklabels(labels: list, ax, 
                                rotate: bool = mconfig.parser("axes_label_options", "rotate_x_labels"),
                                num_labels: int = mconfig.parser("axes_label_options", "rotate_at_num_labels"),
                                angle: float = mconfig.parser("axes_label_options", "rotation_angle"),
                                **kwargs) -> None:
        """Set the xticklabels on bar plots and determine whether they will be rotated.

        Wrapper around matplotlib set_xticklabels
        
        Checks to see if the number of labels is greater than or equal to the default
        number set in config.yml. If this is the case, rotate
        specify whether or not to rotate the labels and angle specifies what angle they should 
        be rotated to.

        Args:
            labels (list): Labels to apply to xticks
            ax (matplotlib.axes): matplotlib.axes
            rotate (bool, optional): rotate labels True/False. 
                Defaults to mconfig.parser("axes_label_options", "rotate_x_labels").
            num_labels (int, optional): Number of labels to rotate at. 
                Defaults to mconfig.parser("axes_label_options", "rotate_at_num_labels").
            angle (float, optional): Angle of rotation. 
                Defaults to mconfig.parser("axes_label_options", "rotation_angle").
        """
        if rotate:
            if (len(labels)) >= num_labels:
                ax.set_xticklabels(labels, rotation=angle, ha="right", **kwargs)
            else:
                labels = [textwrap.fill(x, 10, break_long_words=False) for x in labels]
                ax.set_xticklabels(labels, rotation=0, **kwargs)
        else:
            labels = [textwrap.fill(x, 10, break_long_words=False) for x in labels]
            ax.set_xticklabels(labels, rotation=0, **kwargs)

    @staticmethod
    def remove_excess_axs(axs, excess_axs: int, grid_size: int) -> None:
        """Removes excess axes spins + tick marks.

        Args:
            axs (matplotlib.axes): matplotlib.axes
            excess_axs (int): # of excess axes.
            grid_size (int): Size of facet grid.
        """
        while excess_axs > 0:
            axs[(grid_size)-excess_axs].spines['right'].set_visible(False)
            axs[(grid_size)-excess_axs].spines['left'].set_visible(False)
            axs[(grid_size)-excess_axs].spines['bottom'].set_visible(False)
            axs[(grid_size)-excess_axs].spines['top'].set_visible(False)
            axs[(grid_size)-excess_axs].tick_params(axis='both',
                                                    which='both',
                                                    colors='white')
            excess_axs-=1

    @staticmethod
    def get_sub_hour_interval_count(df: pd.DataFrame) -> int:
        """Detects the interval spacing of timeseries data. 
        
        Used to adjust sums of certain variables for sub-hourly data.

        Args:
            df (pd.DataFrame): pandas dataframe with timestamp in index.

        Returns:
            int: Number of intervals per 60 minutes.
        """
        timestamps = df.index.get_level_values('timestamp').unique()
        time_delta = timestamps[1] - timestamps[0]
        # Finds intervals in 60 minute period
        intervals_per_hour = 60/(time_delta/np.timedelta64(1, 'm'))
        # If intervals are greater than 1 hour, returns 1
        return max(1, intervals_per_hour)

    @staticmethod
    def sort_duration(df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Converts a dataframe time series into a duration curve.

        Args:
            df (pd.DataFrame): pandas multiindex dataframe.
            col (str): Column name by which to sort.

        Returns:
            pd.DataFrame: Dataframe with values sorted from largest to smallest.
        """
        sorted_duration = (df.sort_values(by=col, ascending=False)
                        .reset_index()
                        .drop(columns=['timestamp']))
        
        return sorted_duration

    @staticmethod
    def capacity_energy_unitconversion(max_value: float) -> dict:
        """Auto unitconversion for capacity and energy figures.

        Args:
            max_value (float): Value used to determine divisor and units.

        Returns:
            dict: Dictionary containing divisor and units.
        """
        if max_value < 1000 and max_value > 1:
            divisor = 1
            units = 'MW'
        elif max_value < 1:
            divisor = 0.001
            units = 'kW'
        elif max_value > 999999.9:
            divisor = 1000000
            units = 'TW'
        else:
            divisor = 1000
            units = 'GW'
        
        # Disables auto unit conversion, all values in MW
        if mconfig.parser("auto_convert_units") == False:
            divisor = 1
            units = 'MW'
            
        return {'units':units, 'divisor':divisor}


