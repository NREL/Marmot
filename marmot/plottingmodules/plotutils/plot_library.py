# -*- coding: utf-8 -*-
"""Plot library for creating regularly used plot types.

@author: Daniel Levie
"""

import logging
import textwrap
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from typing import Tuple, List, Union

import marmot.config.mconfig as mconfig

logger = logging.getLogger('marmot_plot.'+__name__)

font_settings = mconfig.parser("font_settings")
text_position = mconfig.parser("text_position")
axes_options = mconfig.parser("axes_options")


class SetupSubplot():

    def __init__(self, ydimension=1, xdimension=1, 
                 figsize: Tuple[int, int] = (mconfig.parser("figure_size","xdimension"), 
                                            mconfig.parser("figure_size","ydimension")), 
                 sharey=False, squeeze=True, ravel_axs=False, **kwargs):
        

        # Set Plot defaults
        mpl.rc('xtick', labelsize=font_settings['xtick_size'])
        mpl.rc('ytick', labelsize=font_settings['ytick_size'])
        mpl.rc('legend', fontsize=font_settings['legend_size'],
                         frameon=axes_options['show_legend_frame'])
        mpl.rc('font', family=font_settings['font_family'])
        mpl.rc('figure', max_open_warning = 0)
        mpl.rc('axes', labelsize=font_settings['axes_label_size'],
                    titlesize=font_settings['title_size'], 
                    titlepad=text_position['title_height'])
        mpl.rc('axes.spines', top=not(axes_options['hide_top_spine']),
                            bottom=not(axes_options['hide_bottom_spine']),
                            left=not(axes_options['hide_left_spine']),
                            right=not(axes_options['hide_right_spine']))   
        mpl.rc('xtick.major', size=axes_options['major_x_tick_length'],
                              width=1)
        mpl.rc('ytick.major', size=axes_options['major_y_tick_length'],
                              width=1)

        x=figsize[0]
        y=figsize[1]
        
        fig, axs = plt.subplots(ydimension, xdimension, 
                                figsize=((x*xdimension),(y*ydimension)), 
                                sharey=sharey, squeeze=squeeze, **kwargs)
        if ravel_axs:
            axs = axs.ravel()

        self.fig : Figure = fig
        self.axs : Axes = axs
        self.exax : Axes = fig.add_subplot(111, frameon=False) 
        self.exax.tick_params(labelcolor='none', top=False, bottom=False, 
                            left=False, right=False)


    def get_figure(self) -> Tuple[Figure, Union[Axes, List[Axes]]]: 

        return self.fig, self.axs

    def _check_if_array(self, n):
        if isinstance(self.axs, Axes):
            ax = self.axs
        else:
            ax = self.axs[n]
        return ax


    def add_legend(self, handles=None, labels=None, 
                        loc=mconfig.parser("axes_options", "legend_position"),
                        ncol=mconfig.parser("axes_options", "legend_columns"),
                        reverse_legend=False, sort_by=None, bbox_to_anchor=None, **kwargs):
    
        loc_anchor = {'lower right': ('lower left', (1.05, 0.0)),
                        'center right': ('center left', (1.05, 0.5)),
                        'upper right': ('upper left', (1.05, 1.0)),
                        'upper center': ('lower center', (0.5, 1.25)),
                        'lower center': ('upper center', (0.5, -0.25)),
                        'lower left': ('lower right', (-0.2, 0.0)),
                        'center left': ('center right', (-0.2, 0.5)),
                        'upper left': ('upper right', (-0.2, 1.0))}

        if handles == None or labels == None:
            
            if isinstance(self.axs, Axes):
                handles_list, labels_list = self.axs.get_legend_handles_labels()
            else:
                handles_list = []
                labels_list = []
                for ax in self.axs.ravel():
                    h, l = ax.get_legend_handles_labels()
                    handles_list.extend(h)
                    labels_list.extend(l)

            # Ensure there are unique labels and handle pairs
            labels_handles = dict(zip(labels_list, handles_list))
        
            if sort_by:
                sorted_list = list(sort_by.copy())
                extra_values = list(labels_handles.keys() - sorted_list)
                sorted_list.extend(extra_values)
                labels_handles = dict(sorted(labels_handles.items(), 
                            key=lambda pair: sorted_list.index(pair[0])))

            labels = labels_handles.keys()
            handles = labels_handles.values()
        
        if reverse_legend:
            handles = reversed(handles)
            labels = reversed(labels)

        if loc in loc_anchor:
            new_loc, bbox_to_anchor = loc_anchor.get(loc, None)
        else:
            bbox_to_anchor = bbox_to_anchor
            new_loc = loc

        self.exax.legend(handles, labels, loc=new_loc, ncol=ncol,
                    bbox_to_anchor=bbox_to_anchor,
                    **kwargs)

    def set_plot_timeseries_format(self, n: int = 0,
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
        ax = self._check_if_array(n)

        locator = mdates.AutoDateLocator(minticks=minticks, maxticks=maxticks)
        formatter = mdates.ConciseDateFormatter(locator)
        formatter.formats[2] = '%d\n %b'
        formatter.zero_formats[1] = '%b\n %Y'
        formatter.zero_formats[2] = '%d\n %b'
        formatter.zero_formats[3] = '%H:%M\n %d-%b'
        formatter.offset_formats[3] = '%b %Y'
        formatter.show_offset = False
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

    def remove_excess_axs(self, excess_axs: int, grid_size: int) -> None:
        """Removes excess axes spins + tick marks.

        Args:
            excess_axs (int): # of excess axes.
            grid_size (int): Size of facet grid.
        """
        axs = self.axs.ravel()
        while excess_axs > 0:
            axs[(grid_size)-excess_axs].spines['right'].set_visible(False)
            axs[(grid_size)-excess_axs].spines['left'].set_visible(False)
            axs[(grid_size)-excess_axs].spines['bottom'].set_visible(False)
            axs[(grid_size)-excess_axs].spines['top'].set_visible(False)
            axs[(grid_size)-excess_axs].tick_params(axis='both',
                                                    which='both',
                                                    colors='white')
            excess_axs-=1

    def set_barplot_xticklabels(self, labels: list, n: int = 0, 
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
            rotate (bool, optional): rotate labels True/False. 
                Defaults to mconfig.parser("axes_label_options", "rotate_x_labels").
            num_labels (int, optional): Number of labels to rotate at. 
                Defaults to mconfig.parser("axes_label_options", "rotate_at_num_labels").
            angle (float, optional): Angle of rotation. 
                Defaults to mconfig.parser("axes_label_options", "rotation_angle").
        """
        ax = self._check_if_array(n)
        if rotate:
            if (len(labels)) >= num_labels:
                ax.set_xticklabels(labels, rotation=angle, ha="right", **kwargs)
            else:
                labels = [textwrap.fill(x, 10, break_long_words=False) for x in labels]
                ax.set_xticklabels(labels, rotation=0, **kwargs)
        else:
            labels = [textwrap.fill(x, 10, break_long_words=False) for x in labels]
            ax.set_xticklabels(labels, rotation=0, **kwargs)

    def add_facet_labels(self, 
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
        font_settings = mconfig.parser("font_settings")

        if alternative_xlabels:
            xlabel = alternative_xlabels
        else:
            xlabel = self.xlabels

        if alternative_ylabels:
            ylabel = alternative_ylabels
        else:
            ylabel = self.ylabels

        if isinstance(self.axs, Axes):
            all_axes = [self.axs]
        else:
            all_axes = self.axs.ravel()
        j=0
        k=0
        for ax in all_axes:
            if xlabels_bottom:
                if ax.is_last_row():
                    try:
                        ax.set_xlabel(xlabel=(xlabel[j]), color='black', 
                                    fontsize=font_settings['axes_label_size']-2, **kwargs)
                    except IndexError:
                        logger.warning(f"Warning: xlabel missing for subplot x{j}")
                        continue
                    j=j+1
            else:
                if ax.is_first_row():
                    try:
                        ax.set_xlabel(xlabel=(xlabel[j]), color='black', 
                                    fontsize=font_settings['axes_label_size']-2, **kwargs)
                        ax.xaxis.set_label_position('top')
                    except IndexError:
                        logger.warning(f"Warning: xlabel missing for subplot x{j}")
                        continue
                    j=j+1
            if ax.is_first_col():
                try:
                    ax.set_ylabel(ylabel=(ylabel[k]), color='black', rotation='vertical', 
                                    fontsize=font_settings['axes_label_size']-2, **kwargs)
                except IndexError:
                    logger.warning(f"Warning: ylabel missing for subplot y{k}")
                    continue
                k=k+1


class PlotLibrary(SetupSubplot):
    
    
    def stackplot(self, data: pd.DataFrame, color_dict: dict = None, 
                 n: int = 0, **kwargs):
        """Creates a stacked area plot

        Wrapper around matplotlib.stackplot.

        Args:
            data (pd.DataFrame): DataFrame of data to plot.
            color_dict (dict): Colour dictionary, keys should be in data columns.
                Defaults to None.
            n (int, optional): Counter for facet plot. 
                Defaults to 0.
        """

        ax = self._check_if_array(n)
        y_axes_decimalpt = axes_options["y_axes_decimalpt"]
        
        if color_dict:
            color_list = [color_dict.get(x, '#333333') for x in data.columns]
        else:
            color_list=None

        ax.stackplot(data.index.values, data.values.T, linewidth=0,
                     colors=color_list, **kwargs)

        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(x, f',.{y_axes_decimalpt}f')))
        ax.margins(x=0.01)
    

    def barplot(self, df: pd.DataFrame, color: Union[dict, list] = None,
                        stacked: bool = False, n: int = 0, custom_tick_labels=None,
                        legend=False,
                        **kwargs):
        """Creates a bar plot

        Wrapper around pandas.plot.bar

        Args:
            df (pd.DataFrame): DataFrame of data to plot.
            color (dict): dictionary of colors, dict keys should be 
                found in df columns.
            stacked (bool, optional): Whether to stack bar values. 
                Defaults to False.

        Returns:
            matplotlib.fig: matplotlib fig
        """
        ax = self._check_if_array(n)

        if isinstance(color, dict):
            color_list = [color.get(x, '#333333') for x in df.columns]
        elif isinstance(color, list):
            color_list = color
        else:
            color_list=None

        df.plot.bar(stacked=stacked,
                    color=color_list, 
                    ax=ax, legend=legend,
                    **kwargs)
        
        # Set x-tick labels 
        if custom_tick_labels and len(custom_tick_labels) > 1:
            tick_labels = custom_tick_labels
        else:
            tick_labels = df.index
        self.set_barplot_xticklabels(tick_labels, n)

    def lineplot(self, data: pd.Series, column=None,
                 color: Union[dict, str] = None,
                 linestyle: str = 'solid',
                 n: int = 0, alpha:int = 1, **kwargs):
        """Creates a line plot

        Wrapper around matplotlib.plot

        Args:
            data (pd.Series, pd.DataFrame): Series/ df of data to plot.
            column (str): If passing df as data column is required.
            color (dict, str, optional): Color dict or str,
                if dict color is chosen based on df column 
                Defaults to None.
            linestyle (str, optional): Style of line to plot. 
                Defaults to 'solid'.
            n (int, optional): Counter for facet plot. Defaults to 0.
            alpha (int, optional): Line opacity. Defaults to 1.
        """
        ax = self._check_if_array(n)

        if isinstance(data, pd.DataFrame):
            plot_data = data[column]
        else:
            plot_data = data

        if isinstance(color, (dict)):
            color = color.get(plot_data.name, '#333333')
        elif isinstance(color, str):
            color = color
        else:
            color=None

        ax.plot(plot_data, linewidth=1, 
                linestyle=linestyle,
                color=color,
                alpha=alpha, **kwargs)

    def histogram(self, data: pd.DataFrame, color_dict: dict,
                  label=None,
                  n: int = 0, **kwargs):
        """Creates a histogram plot

        Wrapper around matplotlib.hist

        Args:
            data (pd.DataFrame): DataFrame of data to plot.
            color_dict (dict): Colour dictionary
            label (list, optional): List of labels for legend. 
                Defaults to None.
            n (int, optional): Counter for facet plot. 
                Defaults to 0.
        """
        ax = self._check_if_array(n)

        ax.hist(data,bins=20, range=(0,1), color=color_dict[label], zorder=2, 
                    rwidth=0.8, label=label, **kwargs)


    def clustered_stacked_barplot(self, df_list: List[pd.DataFrame], 
                                          labels: list, color_dict: dict, 
                                          title: str = "",  H: str = "//", n=0,
                                          **kwargs):
        """Creates a clustered stacked barplot.

        Args:
            df_list (List[pd.DataFrame, pd.DataFrame]): List of Pandas DataFrames
                The columns within each dataframe will be stacked with different colors. 
                The corresponding columns between each dataframe will be set next to each 
                other and given different hatches.
            labels (list): A list of strings, usually the scenario names
            color_dict (dict): Color dictionary, keys should be the same as labels 
            title (str, optional): Optional plot title. Defaults to "".
            H (str, optional): Sets the hatch pattern to differentiate dataframe bars. 
                Defaults to "//".
            n (int, optional): Counter for facet plot. 
                Defaults to 0.
        """
        ax = self._check_if_array(n)

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
        self.set_barplot_xticklabels(x_labels, **kwargs)
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
            
        self.add_legend(handles, label_list)

