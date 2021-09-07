# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 16:37:06 2020
Functions required to create Marmot plots
@author: Daniel Levie
"""

import os
import math
import textwrap
import logging
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
import marmot.config.mconfig as mconfig


logger = logging.getLogger('marmot_plot.'+__name__)
#===============================================================================

class MissingInputData:
    """Exception Class for handling return of missing data."""
    
    def __init__(self):
       return

class MissingZoneData:
    """Exception Class for handling return of zones with no data."""
    
    def __init__(self):
        return

class DataSavedInModule:
    """Exception Class for handling data saved within modules."""
    
    def __init__(self):
        return

class UnderDevelopment:
    """Exception Class for handling methods under development."""
    
    def __init__(self):
        return

class InputSheetError:
    """Exception Class for handling user input sheet errors."""
    
    def __init__(self):
        return

class FacetLabelError:
    """Exception Class for incorrect facet labeling."""
    
    def __init__(self):
        return
    
class MissingMetaData:
    """Exception Class for missing meta data."""
    
    def __init__(self):
        return
    
class UnsupportedAggregation:
    """Exception Class for plotting using unsupported AGG_BY attribute."""
    
    def __init__(self):
        return
    
def get_data(mplot_data_dict, properties, Marmot_Solutions_folder):
    """
    Used to get data from formatted h5 file
    Adds data to dictionary with scenario name as key
    Parameters
    ----------
    mplot_data_dict : dictionary
        dictionary of data with scenarios as keys.
    properties : list
        list of tuples containg required plexos property information
    Marmot_Solutions_folder : folder
        Main Mamrmot folder
    scenario_list : List
        List of scenarios to plot.
    Returns
    -------
    return_value : list
        If 1 in list required data is missing 
    """
    
    check_input_data = []
    
    for prop in properties:
        required, plx_prop_name, scenario_list = prop
        if f"{plx_prop_name}" not in mplot_data_dict:
            mplot_data_dict[f"{plx_prop_name}"] = {}
        
        for scenario in scenario_list:
            if scenario not in mplot_data_dict[f"{plx_prop_name}"]:
                try:
                    mplot_data_dict[f"{plx_prop_name}"][scenario] = pd.read_hdf(os.path.join(Marmot_Solutions_folder,"Processed_HDF5_folder", scenario + "_formatted.h5"),plx_prop_name)
                except KeyError:
                    break
        
        if mplot_data_dict[f"{plx_prop_name}"] == {}:
            logger.warning(f"{plx_prop_name} is MISSING from the Marmot formatted h5 files")
            if required == True:
                check_input_data.append(1)
    return check_input_data


def df_process_gen_inputs(df, ordered_gen):
    """
    Processes generation data into a pivot
    Technology names as columns,
    Timeseries as index
    Parameters
    ----------
    df : DataFrame
        Dataframe to process.
    ordered_gen : list
        List of gen tech types ordered.
    Returns
    -------
    df : DataFrame
        Proceessed DataFrame
    """
    if set(['timestamp','tech']).issubset(df.index.names):
        df = df.reset_index(['timestamp','tech'])
    df = df.groupby(["timestamp", "tech"], as_index=False).sum()
    df = df[df.tech.isin(ordered_gen)]
    df.tech = df.tech.astype("category")
    df.tech.cat.set_categories(ordered_gen, inplace=True)
    df = df.sort_values(["tech"])
    df = df.pivot(index='timestamp', columns='tech', values=0)
    return df

def df_process_categorical_index(df, ordered_gen):
    """
    Creates categorical index based on generators
    Parameters
    ----------
    df : DataFrame
        Dataframe to process.
    ordered_gen : list
        List of gen tech types ordered.
    Returns
    -------
    df : DataFrame
        Processed DataFrame
    """
    df=df
    df.index = df.index.astype("category")
    df.index = df.index.set_categories(ordered_gen)
    df = df.sort_index()
    return df

def setup_facet_xy_dimensions(xlabels, ylabels, facet=True, 
                                multi_scenario=None):
    """
    Sets facet plot x,y dimensions based on provided labeles
    Parameters
    ----------
    xlabels : list
        X axes labels.
    ylabels : string
        Y axes labels.
    facet : list
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
    xdimension=len(xlabels)
    if xlabels == ['']:
        xdimension = 1
    ydimension=len(ylabels)
    if ylabels == ['']:
        ydimension = 1
    # If the plot is not a facet plot, grid size should be 1x1
    if not facet:
        xdimension = 1
        ydimension = 1
        return xdimension, ydimension
    # If no labels were provided or dimensions less than len scenarios use Marmot default dimension settings
    if xlabels == [''] and ylabels == [''] or xdimension*ydimension<len(multi_scenario):
        logger.info("Dimensions could not be determined from x & y labels - Using Marmot default dimensions")
        xdimension, ydimension = set_x_y_dimension(len(multi_scenario))
    return xdimension, ydimension


def set_x_y_dimension(region_number):
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


def setup_plot(xdimension=1,ydimension=1, 
                figsize=(mconfig.parser("figure_size","xdimension"), 
                        mconfig.parser("figure_size","ydimension")),
                sharey=True, 
                squeeze=False, **kwargs):
    """
    Setup matplotlib plot
    Parameters
    ----------
    xdimension : int, optional
        facet plot x dimension. The default is 1.
    ydimension : int, optional
        facet plot y dimension. The default is 1.
    figsize : tuple, optional. Default set in config.yml
        The dimensions of each subplot in inches
    sharey : bool, optional
        Share y axes labels. The default is True.
    Returns
    -------
    fig : matplotlib fig
        matplotlib fig.
    axs : matplotlib.axes
        matplotlib axes.
    """

    x=figsize[0]
    y=figsize[1]
    
    fig, axs = plt.subplots(ydimension, xdimension, 
                            figsize=((x*xdimension),(y*ydimension)), 
                            sharey=sharey, squeeze=squeeze, **kwargs)
    axs = axs.ravel()
    return fig,axs


def create_bar_plot(df, axs, colour, angle=0, stacked=False):
    """
    Creates a bar plot
    Parameters
    ----------
    df : DataFrame
        DataFrame of data to plot.
    axs : matplotlib.axes
        matplotlib.axes.
    colour : dictionary
        colour dictionary.
    stacked : Bool
        True/False for stacked bar
    Returns
    -------
    fig : matplotlib fig
        matplotlib fig.
    """
    fig = df.plot.bar(stacked=stacked, rot=angle, edgecolor='white', linewidth='1.5',
                      color=[colour.get(x, '#333333') for x in df.columns], ax=axs)
    
    
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.tick_params(axis='y', which='major', length=5, width=1)
    axs.tick_params(axis='x', which='major', length=5, width=1)
    return fig


def create_grouped_bar_plot(df, colour, angle=0, custom_tick_labels=None):
    """
    Creates a grouped bar plot
    Parameters
    ----------
    df : DataFrame
        DataFrame of data to plot.
    colour : dictionary
        colour dictionary.
    custom_tick_labels : list
        Custom tick labels, default None
    Returns
    -------
    fig : matplotlib fig
        matplotlib fig.
    """
    
    fig, axs = plt.subplots(figsize=tuple(mconfig.parser("figure_size").values()))
    df.plot.bar(rot=angle, edgecolor='white', linewidth='1.5',
                                      color=[colour.get(x, '#333333') for x in df.columns],ax=axs)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    # Set x-tick labels 
    if custom_tick_labels and len(custom_tick_labels) > 1:
        tick_labels = custom_tick_labels
    else:
        tick_labels = df.index
    set_barplot_xticklabels(tick_labels, ax=axs)

    axs.tick_params(axis='y', which='major', length=5, width=1)
    axs.tick_params(axis='x', which='major', length=5, width=1)
    return fig,axs

def create_stacked_bar_plot(df, colour, angle=0, custom_tick_labels=None):
    """
    Creates a stacked bar plot
    Parameters
    ----------
    df : DataFrame
        DataFrame of data to plot.
    colour : dictionary
        colour dictionary.
    custom_tick_labels : list
        Custom tick labels, default None
    Returns
    -------
    fig : matplotlib fig
        matplotlib fig.
    """
    
    y_axes_decimalpt = mconfig.parser("axes_options","y_axes_decimalpt")
    
    fig, axs = plt.subplots(figsize=tuple(mconfig.parser("figure_size").values()))
    df.plot.bar(stacked=True, rot=angle, edgecolor='black', linewidth='0.1',
                                                color=[colour.get(x, '#333333') for x in df.columns],ax=axs)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    #adds comma to y axis data
    axs.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(x, f',.{y_axes_decimalpt}f')))
    
    # Set x-tick labels 
    if custom_tick_labels and len(custom_tick_labels) > 1:
        tick_labels = custom_tick_labels
    else:
        tick_labels = df.index
    set_barplot_xticklabels(tick_labels, ax=axs)
    
    axs.tick_params(axis='y', which='major', length=5, width=1)
    axs.tick_params(axis='x', which='major', length=5, width=1)
    return fig, axs

def create_clustered_stacked_bar_plot(df_list, ax, labels, color_dict, title="",  H="//", **kwargs):
    """Given a lbar plot with both stacked and unstacked bars.
    
    Parameters
    ----------
    df_list: List of Pandas DataFrames.
        The columns within each dataframe will be stacked with different colors. 
        The corresponding columns between each dataframe will be set next to each other and given different hatches.
    ax : matplotlib.axes
        matplotlib.axes.
    labels: A list of strings, usually the sceanrio names
    color_dict: color dictionary, keys should be the same as labels 
    title: Optional plot title.
    H: Sets the hatch pattern to differentiate dataframe bars. Each consecutive bar will have a higher density of the same hatch pattern.
        
    Returns
    -------
    None.
    """

    n_df = len(df_list)
    n_col = len(df_list[0].columns) 
    n_ind = len(df_list[0].index)
    
    column_names = []
    for df, label in zip(df_list, labels) : # for each data frame
        df.plot(kind="bar",
            linewidth=0.5,
            stacked=True,
            ax=ax,
            legend=False,
            grid=False,
            color=[color_dict.get(x, '#333333') for x in [label]],
            **kwargs)  # make bar plots
        
        column_names.append(df.columns)
    
    #Unique Column names
    column_names = np.unique(np.array(column_names)).tolist()
    
    h,l = ax.get_legend_handles_labels() # get the handles we want to modify
    for i in range(0, n_df * n_col, n_col): # len(h) = n_col * n_df
        for j, pa in enumerate(h[i:i+n_col]):

            for rect in pa.patches: # for each index
                rect.set_x((rect.get_x() + 1 / float(n_df + 1) * i / float(n_col))-0.15)
                if rect.get_height() < 0:
                    rect.set_hatch(H) #edited part 
                rect.set_width(1 / float(n_df + 1))
    
    ax.set_xticks((np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.)
    
    x_labels = df.index.get_level_values(0)
    set_barplot_xticklabels(x_labels, ax=ax, **kwargs)
    ax.set_title(title)
    
    def custom_legend_elements(label):
        color = color_dict.get(label, '#333333')
        return Patch(facecolor=color, edgecolor=color)
    
    handles = []
    label_list = labels.copy()
    for label in label_list:
        handles.append(custom_legend_elements(label))
    
    for i, c_name in enumerate(column_names):
        handles.append(Patch(facecolor='gray', hatch=H*i))
        label_list.append(c_name)
        
    ax.legend(handles, label_list, loc='lower left', bbox_to_anchor=(1.05,0), 
                facecolor='inherit', frameon=True)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='y', which='major', length=5, width=1)
    ax.tick_params(axis='x', which='major', length=5, width=1)


def create_line_plot(axs, data, column, color_dict=None,
                        label=None, linestyle='solid',
                        n=0, alpha=1):
    """
    Creates a line plot
    Parameters
    ----------
    axs : matplotlib.axes
        matplotlib.axes.
    data : DataFrame
         DataFrame of data to plot.
    column : DataFrame column
        column from DataFrame.
    color_dict : dictionary, optional
        colour dictionary. The default is None.
    label : list, optional
        list of labels for legend. The default is None.
    n : int, optional
        counter for facet plot. The default is 0.
    Returns
    -------
    None.
    """
    if color_dict==None:
        axs[n].plot(data[column], linewidth=1, linestyle=linestyle, 
                        label=label, alpha=alpha)
    else:
        axs[n].plot(data[column], linewidth=1, linestyle=linestyle,
                         color=color_dict[column],
                         label=label, alpha=alpha)
    axs[n].spines['right'].set_visible(False)
    axs[n].spines['top'].set_visible(False)
    axs[n].tick_params(axis='y', which='major', length=5, width=1)
    axs[n].tick_params(axis='x', which='major', length=5, width=1)


def create_hist_plot(axs, data, color_dict, 
                        label=None, n=0):
    """
    Creates a histogram plot
    Parameters
    ----------
    axs : matplotlib.axes
        matplotlib.axes.
    data : DataFrame
         DataFrame of data to plot.
    color_dict : dictionary
        colour dictionary
    label : list, optional
        list of labels for legend. The default is None.
    n : int, optional
        counter for facet plot. The default is 0.
    Returns
    -------
    None.
    """
    axs[n].hist(data,bins=20, range=(0,1), color=color_dict[label], zorder=2, rwidth=0.8,label=label)
    axs[n].spines['right'].set_visible(False)
    axs[n].spines['top'].set_visible(False)
    axs[n].tick_params(axis='y', which='major', length=5, width=1)


def create_stackplot(axs, data, color_dict, 
                        label=None, n=0):
    """
    Creates a stacked area plot
    Parameters
    ----------
    axs : matplotlib.axes
        matplotlib.axes.
    data : DataFrame
         DataFrame of data to plot.
    color_dict : dictionary
        colour dictionary
    label : list, optional
        list of labels for legend. The default is None.
    n : int, optional
        counter for facet plot. The default is 0.
    Returns
    -------
    None.
    """
    y_axes_decimalpt = mconfig.parser("axes_options","y_axes_decimalpt")
    
    axs[n].stackplot(data.index.values, data.values.T, labels=label, linewidth=0,
                             colors=[color_dict.get(x, '#333333') for x in data.T.index])
    axs[n].spines['right'].set_visible(False)
    axs[n].spines['top'].set_visible(False)
    axs[n].tick_params(axis='y', which='major', length=5, width=1)
    axs[n].tick_params(axis='x', which='major', length=5, width=1)
    axs[n].yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(x, f',.{y_axes_decimalpt}f')))
    axs[n].margins(x=0.01)


def set_plot_timeseries_format(axs, n=0,
                               minticks=mconfig.parser("axes_options","x_axes_minticks"),
                               maxticks=mconfig.parser("axes_options","x_axes_maxticks")
                               ):
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


def remove_excess_axs(axs, excess_axs, grid_size):
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


def add_facet_labels(fig, xlabels, ylabels):
    """
    Adds labels to outside of Facet plot
    Parameters
    ----------
    fig : matplotlib fig
        matplotlib fig.
    xlabels : list
        X axes labels.
    ylabels : list
        Y axes labels.
    Returns
    -------
    None.
    """
    font_defaults = mconfig.parser("font_settings")

    all_axes = fig.get_axes()
    j=0
    k=0
    for ax in all_axes:
        if ax.is_last_row():
            try:
                ax.set_xlabel(xlabel=(xlabels[j]),  color='black', fontsize=font_defaults['axes_label_size']-2)
            except IndexError:
                logger.warning(f"Warning: xlabel missing for subplot x{j}")
                continue
            j=j+1
        if ax.is_first_col():
            try:
                ax.set_ylabel(ylabel=(ylabels[k]),  color='black', rotation='vertical', fontsize=font_defaults['axes_label_size']-2)
            except IndexError:
                logger.warning(f"Warning: ylabel missing for subplot y{k}")
                continue
            k=k+1


def shift_leapday(df, Marmot_Solutions_folder):
    """
    Shifts dataframe ahead by one day, if a non-leap year time series is modeled with a leap year time index.
    Modeled year must be included in the scenario parent directory name.
    Parameters
    ----------
    df : Pandas multiindex dataframe
        reported parameter (i.e. generator_Generation)
    Marmot_Solutions_folder : string
        Parent directory of scenario results
    shift_leap_day : boolean
        Switch to turn on/off leap day shifting.
        Defined in the "shift_leap_day" field of Marmot_user_defined_inputs.csv.
    Returns
    -------
    df: Pandas multiindex dataframe
        same dataframe, with time index shifted
    """
    if '2008' not in Marmot_Solutions_folder and '2012' not in Marmot_Solutions_folder and df.index.get_level_values('timestamp')[0] > dt.datetime(2024,2,28,0,0):
        df.index = df.index.set_levels(
            df.index.levels[df.index.names.index('timestamp')].shift(1,freq = 'D'),
            level = 'timestamp')
        
    # # Special case where timezone shifting may also be necessary.
    #     df.index = df.index.set_levels(
    #         df.index.levels[df.index.names.index('timestamp')].shift(-3,freq = 'H'),
    #         level = 'timestamp')

    return(df)

def check_label_angle(data_to_plot, dot_T):
    """
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


def merge_new_agg(df, Region_Mapping, AGG_BY):

    """
    Adds new region aggregation in the plotting step. This allows one to create a new aggregation without re-formatting the .h5 file.
    Parameters
    ----------
    df : Pandas multiindex dataframe
        reported parameter (i.e. generator_Generation)
    Region_Mapping : Pandas dataframe
        Dataframe that maps regions to user-specified aggregation levels.
    AGG_BY : string
        Name of new aggregation. Needs to match the appropriate column in the user defined Region Mapping file.
    Returns
    -------
    df: Pandas multiindex dataframe
        same dataframe, with new aggregation level added
    """

    agg_new = Region_Mapping[['region',AGG_BY]]
    agg_new = agg_new.set_index('region')
    df = df.merge(agg_new,left_on = 'region', right_index = True)
    return(df)

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


# test = pd.DataFrame({'A':[1,3,2],
#                     'B' :[9,3,10],
#                     'C' : [100,4,2]})
# test.index = pd.Series(['9 am','10 am','11 am'],name = 'timestamp')

def capacity_energy_unitconversion(max_value):
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