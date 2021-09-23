# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 10:34:06 2021

Plot library for creating regularly used plot types
@author: Daniel Levie
"""

import logging
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import marmot.config.mconfig as mconfig
from marmot.plottingmodules.plotutils.plot_data_helper import PlotDataHelper


logger = logging.getLogger('marmot_plot.'+__name__)


def setup_plot(xdimension=1, ydimension=1, 
                figsize=(mconfig.parser("figure_size","xdimension"), 
                        mconfig.parser("figure_size","ydimension")),
                ravel_axs=True,
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
    if ravel_axs:
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
    PlotDataHelper.set_barplot_xticklabels(tick_labels, ax=axs)

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
    PlotDataHelper.set_barplot_xticklabels(tick_labels, ax=axs)
    
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
                        labels=None, n=0, **kwargs):
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
    
    axs[n].stackplot(data.index.values, data.values.T, labels=labels, linewidth=0,
                             colors=[color_dict.get(x, '#333333') for x in data.T.index], **kwargs)
    axs[n].spines['right'].set_visible(False)
    axs[n].spines['top'].set_visible(False)
    axs[n].tick_params(axis='y', which='major', length=5, width=1)
    axs[n].tick_params(axis='x', which='major', length=5, width=1)
    axs[n].yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(x, f',.{y_axes_decimalpt}f')))
    axs[n].margins(x=0.01)
