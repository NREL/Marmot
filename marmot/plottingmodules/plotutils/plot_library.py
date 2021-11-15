# -*- coding: utf-8 -*-
"""Plot library for creating regularly used plot types.

@author: Daniel Levie
"""

import logging
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from typing import Tuple, List

import marmot.config.mconfig as mconfig
from marmot.plottingmodules.plotutils.plot_data_helper import PlotDataHelper

logger = logging.getLogger('marmot_plot.'+__name__)


def setup_plot(xdimension: int = 1, 
               ydimension: int = 1, 
               figsize: Tuple[int, int] = (mconfig.parser("figure_size","xdimension"), 
                                            mconfig.parser("figure_size","ydimension")),
               ravel_axs: bool = True,
               sharey: bool = True, 
               squeeze: bool = False, **kwargs) -> Tuple: 
    """Setup matplotlib plot

    Args:
        xdimension (int, optional): Facet plot x dimension. 
            Defaults to 1.
        ydimension (int, optional): Facet plot y dimension. 
            Defaults to 1.
        figsize (Tuple[int, int], optional): The dimensions of each subplot in inches. 
            Defaults to (mconfig.parser("figure_size","xdimension"), mconfig.parser("figure_size","ydimension")).
        ravel_axs (bool, optional): Whether to flatten matplotlib axs array. 
            Defaults to True.
        sharey (bool, optional): Share y-axes labels. 
            Defaults to True.
        squeeze (bool, optional): Share x-axes labels.
            Defaults to False.

    Returns:
        tuple: matplotlib fig, matplotlib axes
    """
    x=figsize[0]
    y=figsize[1]
    
    fig, axs = plt.subplots(ydimension, xdimension, 
                            figsize=((x*xdimension),(y*ydimension)), 
                            sharey=sharey, squeeze=squeeze, **kwargs)
    if ravel_axs:
        axs = axs.ravel()
    return fig, axs


def create_bar_plot(df: pd.DataFrame, axs, colour: dict, angle: int = 0, 
                    stacked: bool = False, **kwargs):
    """Creates a bar plot

    Wrapper around pandas.plot.bar

    Args:
        df (pd.DataFrame): DataFrame of data to plot.
        axs (matplotlib.axes): matplotlib.axes
        colour (dict): dictionary of colours, dict keys should be 
            found in df columns.
        angle (int, optional): angle of rotation of labels. 
            Defaults to 0.
        stacked (bool, optional): Whether to stack bar values. 
            Defaults to False.

    Returns:
        matplotlib.fig: matplotlib fig
    """
    fig = df.plot.bar(stacked=stacked, rot=angle, edgecolor='white', linewidth='1.5',
                      color=[colour.get(x, '#333333') for x in df.columns], ax=axs)
    
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.tick_params(axis='y', which='major', length=5, width=1)
    axs.tick_params(axis='x', which='major', length=5, width=1)
    return fig


def create_grouped_bar_plot(df: pd.DataFrame, colour: dict, angle: int = 0,
                            custom_tick_labels: list = None, **kwargs):
    """Creates a grouped bar plot

    Wrapper around pandas.plot.bar

    Args:
        df (pd.DataFrame): DataFrame of data to plot.
        colour (dict): dictionary of colours, dict keys should be 
            found in df columns.
        angle (int, optional): angle of rotation of labels. 
            Defaults to 0.
        custom_tick_labels (list, optional): list of custom labels to apply to x-axes. 
            Defaults to None.

    Returns:
        matplotlib.fig: matplotlib fig
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
    PlotDataHelper.set_barplot_xticklabels(tick_labels, ax=axs)

    axs.tick_params(axis='y', which='major', length=5, width=1)
    axs.tick_params(axis='x', which='major', length=5, width=1)
    return fig, axs


def create_stacked_bar_plot(df: pd.DataFrame, colour: dict, angle: int = 0,
                            custom_tick_labels: list = None, **kwargs):
    """Creates a stacked bar plot

    Wrapper around pandas.plot.bar

    Args:
        df (pd.DataFrame): DataFrame of data to plot.
        colour (dict): dictionary of colours, dict keys should be 
            found in df columns.
        angle (int, optional): angle of rotation of labels. 
            Defaults to 0.
        custom_tick_labels (list, optional): list of custom labels to apply to x-axes. 
            Defaults to None.

    Returns:
        matplotlib.fig: matplotlib fig
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
    PlotDataHelper.set_barplot_xticklabels(tick_labels, ax=axs)
    
    axs.tick_params(axis='y', which='major', length=5, width=1)
    axs.tick_params(axis='x', which='major', length=5, width=1)
    return fig, axs


def create_clustered_stacked_bar_plot(df_list: List[pd.DataFrame], 
                                      ax, labels: list, color_dict: dict, 
                                      title: str = "",  H: str = "//", **kwargs):
    """Creates a clustered stacked barplot.

    Args:
        df_list (List[pd.DataFrame, pd.DataFrame]): List of Pandas DataFrames
            The columns within each dataframe will be stacked with different colors. 
            The corresponding columns between each dataframe will be set next to each 
            other and given different hatches.
        ax (matplotlib.axes): matplotlib.axes
        labels (list): A list of strings, usually the scenario names
        color_dict (dict): Color dictionary, keys should be the same as labels 
        title (str, optional): Optional plot title. Defaults to "".
        H (str, optional): Sets the hatch pattern to differentiate dataframe bars. 
            Defaults to "//".
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
    PlotDataHelper.set_barplot_xticklabels(x_labels, ax=ax, **kwargs)
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


def create_line_plot(axs, data: pd.DataFrame, column: str, color_dict: dict = None,
                        label: list = None, linestyle: str = 'solid',
                        n: int = 0, alpha:int = 1, **kwargs):
    """Creates a line plot

    Wrapper around matplotlib.plot

    Args:
        axs (matplotlib.axes): matplotlib.axes
        data (pd.DataFrame): DataFrame of data to plot.
        column (str): Column from DataFrame to plot.
        color_dict (dict, optional): Colour dictionary.. 
            Defaults to None.
        label (list, optional): List of labels for legend..
            Defaults to None.
        linestyle (str, optional): Style of line to plot. 
            Defaults to 'solid'.
        n (int, optional): Counter for facet plot. Defaults to 0.
        alpha (int, optional): Line opacity. Defaults to 1.
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


def create_hist_plot(axs, data: pd.DataFrame, color_dict: dict, 
                        label: list = None, n: int = 0, **kwargs):
    """Creates a histogram plot

    Wrapper around matplotlib.hist

    Args:
        axs (matplotlib.axes): matplotlib.axes
        data (pd.DataFrame): DataFrame of data to plot.
        color_dict (dict): Colour dictionary
        label (list, optional): List of labels for legend. 
            Defaults to None.
        n (int, optional): Counter for facet plot. 
            Defaults to 0.
    """
    axs[n].hist(data,bins=20, range=(0,1), color=color_dict[label], zorder=2, 
                rwidth=0.8, label=label)
    axs[n].spines['right'].set_visible(False)
    axs[n].spines['top'].set_visible(False)
    axs[n].tick_params(axis='y', which='major', length=5, width=1)


def create_stackplot(axs, data: pd.DataFrame, color_dict: dict, 
                        labels: list = None, n: int = 0, **kwargs):
    """Creates a stacked area plot

    Wrapper around matplotlib.stackplot.

    Args:
        axs (matplotlib.axes): matplotlib.axes
        data (pd.DataFrame): DataFrame of data to plot.
        color_dict (dict): Colour dictionary, keys should be in data columns.
        label (list, optional): List of labels for legend. 
            Defaults to None.
        n (int, optional): Counter for facet plot. 
            Defaults to 0.
    """
    y_axes_decimalpt = mconfig.parser("axes_options","y_axes_decimalpt")
    
    axs[n].stackplot(data.index.values, data.values.T, labels=labels, linewidth=0,
                             colors=[color_dict.get(x, '#333333') for x in data.T.index], **kwargs)
    axs[n].spines['right'].set_visible(False)
    axs[n].spines['top'].set_visible(False)
    axs[n].tick_params(axis='y', which='major', length=5, width=1)
    axs[n].tick_params(axis='x', which='major', length=5, width=1)
    axs[n].yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(x, f',.{y_axes_decimalpt}f')))
    axs[n].margins(x=0.01)
