# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 16:37:06 2020

Functions used to assist with the creation of Marmot plots
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

import marmot.config.mconfig as mconfig


logger = logging.getLogger('marmot_plot.'+__name__)


class PlotDataHelper(dict):
    """ Methods used to assist with the creation of Marmot plots

    Collection of Methods to assist with creation of figures,
    including getting and formatting data, setting up plot sizes and adding 
    elements to plots such as labels.

    PlotDataHelper inherits the python class 'dict' so acts like a dictionary and stores the
    formatted data when retrieved by the get_formatted_data method.


    """

    def __init__(self, Marmot_Solutions_folder, AGG_BY, ordered_gen, PLEXOS_color_dict, 
                    Scenarios, ylabels, xlabels, gen_names_dict,
                    Region_Mapping=pd.DataFrame()):

        """
        Parameters
        ----------

        Marmot_Solutions_folder : string directory
            Folder to save Marmot solution files.
        AGG_BY : string
            Informs region type to aggregate by when creating plots.
        ordered_gen : list 
            Ordered list of generator technologies to plot, order defines 
            the generator technology position in stacked bar and area plots
        PLEXOS_color_dict : Dictionary
            Dictionary of colors to use for generation technologies
        Scenarios : list
            Name of scenarios to process.
        ylabels : list
            y axis labels for facet plots.
        xlabels : list
            x axis labels for facet plots.
        gen_names_dict : Dictionary
            Mapping dic to rename generator technologies.
        Region_Mapping : pd.DataFrame
            Mapping file to map custom regions/zones to create custom aggregations. 
            Aggregations are created by grouping PLEXOS regions.
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


    def get_formatted_data(self, properties:list):
        """
        Get data from formatted h5 file
        Adds data to dictionary with scenario name as key
        Parameters
        ----------

        properties : list
            list of tuples containg required plexos property information

        Returns
        -------
        return_value : list
            If 1 in list required data is missing 
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
                # Read data in with multi threading
                executor_func_setup = functools.partial(self.read_processed_h5file, plx_prop_name)
                with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
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

    def read_processed_h5file(self, plx_prop_name:str, scenario:str):
        """Reads Data from processed h5file"""
        try:
            with pd.HDFStore(os.path.join(self.Marmot_Solutions_folder, "Processed_HDF5_folder", 
                                            f"{scenario}_formatted.h5"), 'r') as file:
                return file[plx_prop_name]
        except KeyError:
            return pd.DataFrame()
        
    def rename_gen_techs(self, df) -> pd.DataFrame:
        """
        Renames generator technologies based on the gen_names.csv file 

        Works for both index and column based values
        Parameters
        ----------
        df : DataFrame
            Dataframe to process.

        Returns
        -------
        df : DataFrame
            Proceessed DataFrame
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
            self.logger.warning(f"The following Generators could not be re-classified, they wil be renamed 'Other': {unmapped_techs}")
        return df

    def assign_curtailment_techs(self, df) -> pd.DataFrame:
        """
        Assign technologies to Marmot's Curtailment property (generator_Curtailment)

        Parameters
        ----------
        df : DataFrame
            Dataframe to process.

        Returns
        -------
        df : DataFrame
            Proceessed DataFrame
        """

        # Adjust list of values to drop from vre_gen_cat depending on if it exhists in processed techs
        adjusted_vre_gen_list = [name for name in self.vre_gen_cat if name in df.columns]

        if not adjusted_vre_gen_list:
            self.logger.warning("Curtailment techs could not be identified correctly for Marmot's Curtailment property. "
            "This is likely happening as the 'vre' column was not present in the ordered_gen_categories.csv.")
            return df
        else: 
            # Retrun df with just vre techs
            return df[df.columns.intersection(self.vre_gen_cat)]

    def df_process_gen_inputs(self, df) -> pd.DataFrame:
        """
        Processes generation data into a pivot
        Technology names as columns,
        Timeseries as index
        Parameters
        ----------
        df : DataFrame
            Dataframe to process.

        Returns
        -------
        df : DataFrame
            Proceessed DataFrame
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

    def create_categorical_tech_index(self, df) -> pd.DataFrame:
        """
        Creates categorical index based on generators
        Parameters
        ----------
        df : DataFrame
            Dataframe to process.

        Returns
        -------
        df : DataFrame
            Processed DataFrame
        """
        df.index = df.index.astype("category")
        df.index = df.index.set_categories(self.ordered_gen)
        df = df.sort_index()
        return df

    def merge_new_agg(self, df) -> pd.DataFrame:
        #TODO Needs fixing 
        """
        Adds new region aggregation in the plotting step. This allows one to create a new aggregation without re-formatting the .h5 file.
        Parameters
        ----------
        df : Pandas multiindex dataframe
            reported parameter (i.e. generator_Generation)
        Returns
        -------
        df: Pandas multiindex dataframe
            same dataframe, with new aggregation level added
        """

        agg_new = self.Region_Mapping[['region', self.AGG_BY]]
        agg_new = agg_new.set_index('region')
        df = df.merge(agg_new,left_on = 'region', right_index = True)
        return(df)

    def adjust_for_leapday(self, df) -> pd.DataFrame:
        """
        Shifts dataframe ahead by one day, if a non-leap year time series is modeled with a leap year time index.
        Modeled year must be included in the scenario parent directory name.
        Parameters
        ----------
        df : Pandas multiindex dataframe
            reported parameter (i.e. generator_Generation)

        Returns
        -------
        df: Pandas multiindex dataframe
            same dataframe, with time index shifted
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


    def setup_facet_xy_dimensions(self, facet:bool=True, 
                                    multi_scenario:list=None) -> tuple:
        """
        Sets facet plot x,y dimensions based on user defined labeles
        ----------

        facet : bool
            Trigger for plotting facet plots.
        multi_scenario : list, optional
            list of scenarios. The default is None.
        Returns
        -------
        xdimension : int
            X axes facet dimension.
        ydimension : int
            Y axes facet dimension.
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

    def set_x_y_dimension(self, region_number:int) -> tuple:
        """
        Sets X,Y dimension of plots without x,y labels
        Parameters
        ----------
        region_number : int
            # regions/scenarios.
        Returns
        -------
        xdimension : int
            X axes facet dimension..
        ydimension : int
            Y axes facet dimension..
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
                        xlabels_bottom:bool=True,
                        alternative_xlabels:list=None ,
                        alternative_ylabels:list=None,
                        **kwargs) -> None:
        """
        Adds labels to outside of Facet plot
        Parameters
        ----------
        fig : matplotlib fig
            matplotlib fig.
        xlabels_bottom: bool, optional
            If True labels are placed under bottom
            row, else top of top row, default True
        alternative_xlabels: list, optional
            alteranative xlabels, default none
        alternative_ylabels: list, optional
            alteranative ylabels, default none
        Returns
        -------
        None.
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
    def set_plot_timeseries_format(axs, n=0,
                               minticks=mconfig.parser("axes_options","x_axes_minticks"),
                               maxticks=mconfig.parser("axes_options","x_axes_maxticks")
                               ) -> None:
        """
        Auto sets timeseries format
        Parameters
        ----------
        axs : matplotlib.axes
            matplotlib.axes.
        n : int, optional
            Counter for facet plot. The default is 0.
        minticks : int, optional
            Minimum tick marks. The default is 6.
        maxticks : int, optional
            Max tick marks. The default is 8.
        Returns
        -------
        None.
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
    def set_barplot_xticklabels(labels, ax, 
                            rotate=mconfig.parser("axes_label_options", "rotate_x_labels"),
                            num_labels=mconfig.parser("axes_label_options", "rotate_at_num_labels"),
                            angle=mconfig.parser("axes_label_options", "rotation_angle"),
                            **kwargs) -> None:
        """
        Set the xticklabels on bar plots and determine whether they will be rotated.
        Wrapper around matplotlib set_xticklabels
        
        Checks to see if the number of labels is greater than or equal to the default
        number set in config.yml.  If this is the case, rotate
        specify whether or not to rotate the labels and angle specifies what angle they should 
        be rotated to.
        ----------

        labels : (list) labels to apply to xticks
        ax : matplotlib.axes
            matplotlib.axes.
        rotate : (bool)
            rotate labels True/False, Optional
            default set in config.yml
        num_labels : (int)
            number of labels to rotate at, Optional
            default set in config.yml
        angle : (int/float)
            angle of rotation, Optional
            default set in config.yml
        **kwargs : set_xticklabels keywords, Optional 
        
        Returns
        -------
        None, sets xticklabels inplace
        
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
    def check_label_angle(data_to_plot, dot_T):
        """
        Will be depreciated in next release
        Checks to see if the number of labels is greater than or equal to the default
        number set in mconfig.py.  If this is the case, other values in mconfig.py
        specify whether or not to rotate the labels and what angle they should 
        be rotated to.
        ----------
        data_to_plot : Pandas dataframe
        
        dot_T : Boolean value for whether of not the label data is saved 
        as columns or rows within data_to_plot.
        
        Returns
        -------
        data_to_plot: Pandas dataframe
            same dataframe, with updated label strings
        
        angle: Integer value of angle to rotate labels, 0 --> no rotation
        """    
        rotate = mconfig.parser("axes_label_options", "rotate_x_labels")
        num_labels = mconfig.parser("axes_label_options", "rotate_at_num_labels")
        angle = mconfig.parser("axes_label_options", "rotation_angle")
            
        if rotate:
            if dot_T:
                if (len(data_to_plot.T)) >= num_labels:
                    return data_to_plot, angle
                else:
                    data_to_plot.columns = data_to_plot.columns.str.wrap(10, break_long_words = False)
                    return data_to_plot, 0
            
            else:
                if (len(data_to_plot)) >= num_labels:
                    return data_to_plot, angle
                else:
                    data_to_plot.index = data_to_plot.index.str.wrap(10, break_long_words = False)
                    return data_to_plot, 0
        
        else:
            if dot_T:
                data_to_plot.columns = data_to_plot.columns.str.wrap(10, break_long_words = False)
            else:
                data_to_plot.index = data_to_plot.index.str.wrap(10, break_long_words = False)
            return data_to_plot, 0

    @staticmethod
    def remove_excess_axs(axs, excess_axs:int, grid_size:int) -> None:
        """
        Removes excess axes spins + tick marks
        Parameters
        ----------
        axs : matplotlib.axes
            matplotlib.axes.
        excess_axs : int
            # of excess axes.
        grid_size : int
            Size of facet grid.
        Returns
        -------
        None.
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
    def get_sub_hour_interval_count(df) -> int:
        """
        Detects the interval spacing; used to adjust sums of certain for variables for sub-hourly runs
        Parameters
        ----------
        df : Pandas multiindex dataframe for some reported parameter (e.g. generator_Generation)
        Returns
        -------
        interval_count : number of intervals per 60 minutes
        """
        time_delta = df.index[1] - df.index[0]
        # Finds intervals in 60 minute period
        intervals_per_hour = 60/(time_delta/np.timedelta64(1, 'm'))
        # If interals are greater than 1 hour, returns 1
        return max(1, intervals_per_hour)

    @staticmethod
    def sort_duration(df, col):
        """
        Converts a dataframe time series into a duration curve.
        
        Parameters
        ----------
        df : Pandas multiindex dataframe for some reported parameter (e.g. line_Flow)
        col : Column name by which to sort.
        Returns
        -------
        df : Sorted time series. 
        """

        sorted_duration = (df.sort_values(by=col, ascending=False)
                        .reset_index()
                        .drop(columns=['timestamp']))
        
        return sorted_duration

    @staticmethod
    def capacity_energy_unitconversion(max_value) -> dict:
        """
        auto unitconversion for capacity and energy figures.
        Parameters
        ----------
        max_value : float
            value used to determine divisor and units.
        Returns
        -------
        dict
            dictionary containing divisor and units.
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


