# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 16:37:06 2020

Functions required to create Marmot plots

@author: Daniel Levie
"""

import pandas as pd
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import math
#===============================================================================


def get_data(data_collection,data,Marmot_Solutions_folder,scenario_list):
    """
    Used to get data from formatted h5 file
    Adds data to dictionary with scenario name as key

    Parameters
    ----------
    data_collection : dictionary
        dictionary of data with scenarios as keys.
    data : string
        name of data to pull from h5 file.
    Marmot_Solutions_folder : folder
        Main Mamrmot folder
    scenario_list : List
        List of scenarios to plot.

    Returns
    -------
    return_value : int
        1 or 0 for checking data
    """
    for scenario in scenario_list:
        try:
            data_collection[scenario] = pd.read_hdf(os.path.join(Marmot_Solutions_folder, scenario,"Processed_HDF5_folder", scenario+ "_formatted.h5"),data)
            return_value = 0
        except KeyError:
            print("'{}' is MISSING from the Marmot formatted h5 files".format(data))
            return_value = 1
            return return_value
    return return_value

def df_process_gen_inputs(df,ordered_gen):
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
    df = df.reset_index(['timestamp','tech'])
    df = df.groupby(["timestamp", "tech"], as_index=False).sum()
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

def setup_facet_xy_dimensions(xlabels,ylabels,facet,multi_scenario=None):
    """
    Sets facet plot x,y dimensions baded on provided labeles

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
    # If no labels were provided use Marmot default dimension settings
    if xlabels == [''] and ylabels == ['']:
        print("Warning: Facet Labels not provided - Using Marmot default dimensions")
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


def setup_plot(xdimension=1,ydimension=1,sharey=True):
    """
    Setup matplotlib plot

    Parameters
    ----------
    xdimension : int, optional
        plot x dimension. The default is 1.
    ydimension : int, optional
        plot y dimension. The default is 1.
    sharey : bool, optional
        Share y axes labels. The default is True.

    Returns
    -------
    fig : matplotlib fig
        matplotlib fig.
    axs : matplotlib.axes
        matplotlib axes.
    """
    fig, axs = plt.subplots(ydimension,xdimension, figsize=((6*xdimension),(4*ydimension)), sharey=sharey, squeeze=False)
    axs = axs.ravel()
    return fig,axs


def create_grouped_bar_plot(df, colour):
    """
    Creates a grouped bar plot

    Parameters
    ----------
    df : DataFrame
        DataFrame of data to plot.
    colour : dictionary
        colour dictionary.

    Returns
    -------
    fig : matplotlib fig
        matplotlib fig.
    """
    fig = df.plot.bar(figsize=(6,4), rot=0, edgecolor='white', linewidth='1.5',
                                      color=[colour.get(x, '#333333') for x in df.columns])
    fig.spines['right'].set_visible(False)
    fig.spines['top'].set_visible(False)
    fig.tick_params(axis='y', which='major', length=5, width=1)
    fig.tick_params(axis='x', which='major', length=5, width=1)
    return fig

def create_stacked_bar_plot(df, colour):
    """
    Creates a stacked bar plot

    Parameters
    ----------
    df : DataFrame
        DataFrame of data to plot.
    colour : dictionary
        colour dictionary.

    Returns
    -------
    fig : matplotlib fig
        matplotlib fig.
    """

    fig = df.plot.bar(stacked=True, figsize=(6,4), rot=0, edgecolor='black', linewidth='0.1',
                                                color=[colour.get(x, '#333333') for x in df.columns])
    fig.spines['right'].set_visible(False)
    fig.spines['top'].set_visible(False)
    #adds comma to y axis data
    fig.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    fig.tick_params(axis='y', which='major', length=5, width=1)
    fig.tick_params(axis='x', which='major', length=5, width=1)
    return fig

def create_line_plot(axs,data,column,color_dict=None,label=None,n=0):
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
        axs[n].plot(data[column], linewidth=1,label=label)
    else:
        axs[n].plot(data[column], linewidth=1, color=color_dict[column],label=column)
    axs[n].spines['right'].set_visible(False)
    axs[n].spines['top'].set_visible(False)
    axs[n].tick_params(axis='y', which='major', length=5, width=1)
    axs[n].tick_params(axis='x', which='major', length=5, width=1)


def create_hist_plot(axs,data,color_dict,label=None,n=0):
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


def create_stackplot(axs,data,color_dict,label=None,n=0):
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
    axs[n].stackplot(data.index.values, data.values.T, labels=label, linewidth=0,
                             colors=[color_dict.get(x, '#333333') for x in data.T.index])
    axs[n].spines['right'].set_visible(False)
    axs[n].spines['top'].set_visible(False)
    axs[n].tick_params(axis='y', which='major', length=5, width=1)
    axs[n].tick_params(axis='x', which='major', length=5, width=1)
    axs[n].yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    axs[n].margins(x=0.01)


def set_plot_timeseries_format(axs,n=0,minticks=6, maxticks=8):
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
    all_axes = fig.get_axes()
    j=0
    k=0
    for ax in all_axes:
        if ax.is_last_row():
            ax.set_xlabel(xlabel=(xlabels[j]),  color='black', fontsize=16)
            j=j+1
        if ax.is_first_col():
            ax.set_ylabel(ylabel=(ylabels[k]),  color='black', rotation='vertical', fontsize=16)
            k=k+1
