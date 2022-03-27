# -*- coding: utf-8 -*-
"""Methods for creating plot figure and axs objects
and a library of regularly used plot types.

@author: Daniel Levie
"""

import logging
import textwrap
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from typing import Tuple, List, Union

import marmot.utils.mconfig as mconfig


logger = logging.getLogger('plotter.'+__name__)

font_settings = mconfig.parser("font_settings")
text_position = mconfig.parser("text_position")
axes_options = mconfig.parser("axes_options")


class SetupSubplot():
    """Sets up the main figure and subplots for use in Marmot
    and expands the functionality of matplotlib by adding further
    methods to quickly build plots.
    """

    def __init__(self, nrows: int = 1, 
                 ncols: int = 1, 
                 figsize: Tuple[int, int] = (mconfig.parser("figure_size",
                                                            "xdimension"), 
                                            mconfig.parser("figure_size",
                                                           "ydimension")), 
                 sharey: bool = False, squeeze: bool = True, 
                 ravel_axs: bool = False, **kwargs):
        """Defines the dimensions (nrows, ncols) of a figure and its subplots.
        Builds on top of matplotlib.pyplot.subplots and preserves all 
        functionality of that function.

        All arguments are optional and by default calling SetupSubplot() 
        or PlotLibrary() which inherits SetupSubplot, will create a 
        1x1 figure with a figsize defined in the config.yml file. 
        The following are some common values to pass when creating various 
        plots.

        - 1x1 figure: SetupSubplot()
        - 1xN figure: SetupSubplot(nrows=1, ncols=N, sharey=True)
        - MxN figure: SetupSubplot(nrows=M, ncols=N, sharey=True, 
                                   squeeze=False, ravel_axs=True)

        Plotting defaults are also set in this class, which are defined
        in the config.yml file. 

        Args:
            nrows (int, optional): Number of rows of the subplot grid.
                Defaults to 1.
            ncols (int, optional): Number of columns of the subplot grid.
                Defaults to 1.
            figsize (Tuple[int, int], optional): The x,y dimension of each
                subplot. 
                Defaults set in config.yml
            sharey (bool, optional): share the y-axis across all subplots. 
                Defaults to False.
            squeeze (bool, optional): If True, extra dimensions are squeezed 
                out from the returned axs objects if 1x1 figure or 1xN figure.
                Defaults to True.
            ravel_axs (bool, optional): If True the returned axs object is a 
                1D numpy object array of Axes objects. This can be used to 
                convert MxN figure axs objects to 1D.
                Defaults to False.
        """
        # Set plot defaults
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
        
        fig, axs = plt.subplots(nrows, ncols, 
                                figsize=((x*ncols),(y*nrows)), 
                                sharey=sharey, squeeze=squeeze, **kwargs)
        if ravel_axs:
            axs = axs.ravel()

        self.fig : Figure = fig
        self.axs : Axes = axs
        self.exax : Axes = fig.add_subplot(111, frameon=False) 
        self.exax.tick_params(labelcolor='none', top=False, bottom=False, 
                            left=False, right=False)

    def _check_if_array(self, sub_pos):
        if isinstance(self.axs, Axes):
            ax = self.axs
        else:
            ax = self.axs[sub_pos]
        return ax

    def get_figure(self) -> Tuple[Figure, Union[Axes, List[Axes]]]: 
        """Returns the matplotlib figure and Axes objects.

        axes can be either a single Axes object or an array of 
        Axes objects depending on the figure dimensions defined.

        Returns:
            Tuple[Figure, Union[Axes, List[Axes]]]: 
            matplotlib figure and axes objects
        """
        return self.fig, self.axs

    def add_legend(self, handles=None, 
                    labels: List[str] = None, 
                    loc=mconfig.parser("axes_options", "legend_position"),
                    ncol=mconfig.parser("axes_options", "legend_columns"),
                    reverse_legend: bool = False, sort_by: list = None, 
                    bbox_to_anchor=None, 
                    **kwargs) -> None:
        """Adds a legend to the desired location on the figure.

        Wrapper around matplotlib.axes.Axes.legend

        The location and number of columns to create can be set 
        through the config.yml file.
        The available default options are:

            - 'lower right'
            - 'center right'
            - 'upper right'
            - 'upper center'
            - 'lower center'
            - 'lower left'
            - 'center left'
            - 'upper left'

        The default options will place the legend outside the main figure 
        subplots to avoid any overlaps of elements. Custom placement is still 
        possible through the loc and bbox_to_anchor arguments but is not advised 
        as this will hard code the placement.

        Passing handles and labels is not required as these are obtained from 
        the axs objects. Any duplicate label and their handle will be removed.

        Sorting of legend is advised when plotting multiple subplots as original
        order cannot be guaranteed. This can be done by passing a list to the 
        sort_by argument, values will then be sorted by the order they appear in 
        the list.
        
        Args:
            handles (Artists sequence, optional): A list of Artists (lines, patches) to be 
                added to the legend. Use this together with labels, if you need full 
                control on what is shown in the legend and the automatic mechanism 
                described above is not sufficient.
                The length of handles and labels should be the same in this case. 
                If they are not, they are truncated to the smaller length. 
                Defaults to None.
            labels (List[str], optional): A list of labels to show next to the artists. 
                Use this together with handles, if you need full control on what is shown 
                in the legend and the automatic mechanism described above is not sufficient. 
                Defaults to None.
            loc (str or pair of floats, optional): The location of the legend.
                Defaults set in config.yml.
            ncol (int, optional):The number of columns that the legend has. 
                Defaults set in config.yml.
            reverse_legend (bool, optional): Revereses the legend order. 
                Defaults to False.
            sort_by (list, optional): A list to sort the legend by, list can contain more or 
                less entries than the legend, entries with no matches are not sorted and added 
                to end of legend. 
                Defaults to None.
            bbox_to_anchor (2-tuple, or 4-tuple of floats, optional): Box that is used 
                to position the legend in conjunction with loc. This argument allows arbitrary 
                placement of the legend.
                Defaults to None.
        """
    
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
            handles = reversed(list(handles))
            labels = reversed(list(labels))

        if loc in loc_anchor:
            new_loc, bbox_to_anchor = loc_anchor.get(loc, None)
        else:
            bbox_to_anchor = bbox_to_anchor
            new_loc = loc

        self.exax.legend(handles, labels, loc=new_loc, ncol=ncol,
                        bbox_to_anchor=bbox_to_anchor,
                        **kwargs)

    def add_property_annotation(self, df: pd.DataFrame,
                                prop: str,
                                sub_pos: Union[int, Tuple[int, int]] = 0,
                                curtailment_name: str = 'Curtailment',
                                energy_unit: str = 'MW',
                                re_gen_cat: list = None,
                                gen_cols: list = None) -> pd.Timestamp:
        """Adds a property annotation to the subplot.

        The current supported properties are:

            - Peak Demand
            - Min Net Load
            - Peak RE
            - Peak Unserved Energy
            - Peak Curtailment

        Based on the property selected this method will locate the timestamp 
        and corresponding value of the property. The value and an arrow pointing 
        to the location of the property will be annotated.

        Args:
            df (pd.DataFrame): Dataframe with datetime index
            prop (str): property
            sub_pos (Union[int, Tuple[int, int]], optional): Position of subplot,
                can be either a integer or a tuple of 2 integers depending on 
                how SetupSubplot was instantiated. 
                Defaults to 0
            curtailment_name (str, optional): Name of curtailment column. 
                Defaults to 'Curtailment'.
            energy_unit (str, optional): units to add to prop annotation. 
                Defaults to 'MW'.
            re_gen_cat (list, optional): list of re techs, needed to for 
                'Peak RE' property.
                Defaults to None.
            gen_cols (list, optional): list of gen columns, needed for 
                'Peak RE' and 'Peak Curtailmnet' property. 
                Defaults to None.

        Returns:
            pd.Timestamp: timstamp of property
        """

        ax = self._check_if_array(sub_pos)

        # Ensure values are lists
        if isinstance(re_gen_cat, pd.Index):
                re_gen_cat = list(re_gen_cat)
        if isinstance(gen_cols, pd.Index):
            gen_cols = list(gen_cols)

        if prop == "Peak Demand":
            x_time_value = df["Total Demand"].idxmax()
            y_mw_value = df.loc[x_time_value, "Total Demand"]
            y_point_value = y_mw_value

        elif prop == "Min Net Load":
            x_time_value = df["Net Load"].idxmin()
            y_mw_value = df.loc[x_time_value, "Net Load"]
            y_point_value = y_mw_value
        
        elif prop == 'Peak RE':
            if not re_gen_cat:
                logger.warning(f"To plot a {prop} annotation a "
                                "list of re gen names is required. "
                                "Pass a list to the add_property_annotation " 
                                "re_gen_cat argument.")
                return None
            if not gen_cols:
                logger.warning(f"To plot a {prop} annotation a "
                                "list of all generator names is required. "
                                "Pass a list to the add_property_annotation " 
                                "gen_cols argument.")
                return None
            re_df = df[df.columns.intersection(re_gen_cat)]
            re_total = re_df.sum(axis=1)
            gen_df = df[df.columns.intersection(gen_cols)]
            if curtailment_name in gen_df.columns:
                gen_df = gen_df.drop(curtailment_name, axis=1)
            x_time_value = re_total.idxmax()
            y_mw_value = re_total[x_time_value]
            y_point_value = gen_df.loc[x_time_value].sum()
            
        elif prop == 'Peak Unserved Energy':
            x_time_value = df["Unserved Energy"].idxmax()
            y_mw_value = df.loc[x_time_value, "Unserved Energy"]
            y_point_value = df.loc[x_time_value, "Total Demand"]
 
        elif prop == 'Peak Curtailment':
            if curtailment_name not in df:
                logger.warning("No Curtailment in dataset")
                return None
            if not gen_cols:
                logger.warning(f"To plot a {prop} annotation a "
                                "list of generator names is required. "
                                "Pass a list to the add_property_annotation " 
                                "gen_cols argument.")
                return None
            curtailment = df[curtailment_name]
            gen_df = df[df.columns.intersection(gen_cols)]
            x_time_value = curtailment.idxmax()
            y_mw_value = curtailment[x_time_value]
            y_point_value = gen_df.loc[x_time_value].sum()
        
        elif prop == "Peak Reserve Provision":
            peak_re = df.sum(axis=1)
            x_time_value = peak_re.idxmax()
            y_mw_value = peak_re[x_time_value]
            y_point_value = y_mw_value

        else:
            logger.warning(f"Property: {prop}, Not Supported!")
            return None

        ax.annotate(f"{prop}: \n{str(format(y_mw_value, '.2f'))} {energy_unit}",
                    xy=(x_time_value, y_point_value), 
                    xytext=(0.05, 1.0),
                    textcoords='axes fraction',
                    fontsize=13, 
                    arrowprops=dict(facecolor='black', width=1.5, shrink=0.05))

        return x_time_value

    def add_main_title(self, label: str, **kwargs) -> None:
        """Adds a title centered above the main figure 

        Wrapper around matplotlib.axes.Axes.set_title

        Args:
            label (str): Title of figure.
        """
        self.exax.set_title(label, **kwargs)

    def set_yaxis_major_tick_format(self, tick_format: str = 'standard',
                                    decimal_accuracy: int = mconfig.parser("axes_options", 
                                                                    "y_axes_decimalpt"),
                                    sub_pos: Union[int, Tuple[int, int]] = 0) -> None:
        """Sets the y axis major tick format of numbers.

        The decimal point accuracy of the numbers can be further adjusted 
        using the config file "y_axes_decimalpt" input.

        Args:
            tick_format (str, optional): Format options. 
                Opinions available are:

                    - standard (1000 values seperated by ',')
                    - percent (Adds % symbal to axis values)
                    - log

                Defaults to 'standard'.
            decimal_accuracy (int, optional): Number of decimal 
                points to use. 
                Default set in config file.
            sub_pos (Union[int, Tuple[int, int]], optional): Position of subplot,
                can be either a integer or a tuple of 2 integers depending on 
                how SetupSubplot was instantiated. 
                Defaults to 0
        """
        ax = self._check_if_array(sub_pos)

        if tick_format == 'standard':
            ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: 
                        format(x, f',.{decimal_accuracy}f')))
        elif tick_format == 'percent':
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        elif tick_format == 'log':
            logger.warning("set_yaxis_major_tick_format: 'log' Not yet developed")
            pass

    def set_subplot_timeseries_format(self,
                                   minticks: int = mconfig.parser("axes_options",
                                                                "x_axes_minticks"),
                                   maxticks: int = mconfig.parser("axes_options",
                                                                "x_axes_maxticks"),
                                   sub_pos: Union[int, Tuple[int, int]] = 0,
                                   zero_formats_1: str = '%b\n %Y', 
                                   zero_formats_2: str = '%d\n %b', 
                                   zero_formats_3: str = '%H:%M\n %d-%b') -> None:
        """Auto sets timeseries format of subplot.

        Args:
            minticks (int, optional): Minimum tick marks. 
                Defaults to mconfig.parser("axes_options","x_axes_minticks").
            maxticks (int, optional): Max tick marks. 
                Defaults to mconfig.parser("axes_options","x_axes_maxticks").
            sub_pos (Union[int, Tuple[int, int]], optional): Position of subplot,
                can be either a integer or a tuple of 2 integers depending on 
                how SetupSubplot was instantiated. 
                Defaults to 0.
            zero_formats_1 (str, optional): Sets the zero_fromats[1] format.
                Defaults to '%b\n %Y'.
            zero_formats_2 (str, optional): Sets the zero_formats[2] format.
                Defaults to '%d\n %b'.
            zero_formats_3 (str, optional): Sets the zero_formats[3] format.
                Defaults to '%H:%M\n %d-%b'.
        """
        ax = self._check_if_array(sub_pos)

        locator = mdates.AutoDateLocator(minticks=minticks, maxticks=maxticks)
        formatter = mdates.ConciseDateFormatter(locator)
        formatter.formats[2] = '%d\n %b'
        formatter.zero_formats[1] = zero_formats_1
        formatter.zero_formats[2] = zero_formats_2
        formatter.zero_formats[3] = zero_formats_3
        formatter.offset_formats[3] = '%b %Y'
        formatter.show_offset = False
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

    def remove_excess_axs(self, excess_axs: int, grid_size: int) -> None:
        """Removes excess axes spins and tick marks.

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

    def set_barplot_xticklabels(self, labels: list, 
                                rotate: bool = mconfig.parser("axes_label_options", 
                                                              "rotate_x_labels"),
                                num_labels: int = mconfig.parser("axes_label_options", 
                                                                 "rotate_at_num_labels"),
                                angle: float = mconfig.parser("axes_label_options", 
                                                              "rotation_angle"),
                                sub_pos: Union[int, Tuple[int, int]] = 0, 
                                **kwargs) -> None:
        """Set the xticklabels on bar plots and determine whether they will be rotated.

        Wrapper around matplotlib.axes.Axes.set_xticklabels
        
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
            sub_pos (Union[int, Tuple[int, int]], optional): Position of subplot,
                can be either a integer or a tuple of 2 integers depending on 
                how SetupSubplot was instantiated. 
                Defaults to 0.
        """
        ax = self._check_if_array(sub_pos)
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
                         xlabels: list = None,
                         ylabels: list = None,
                         **kwargs) -> None:
        """Adds labels to outside of facet plot.

        Args:
            xlabels_bottom (bool, optional): 
                If True labels are placed under bottom axis. 
                Defaults to True.
            xlabels (list, optional): list of xlabels. 
                Defaults to None.
            ylabels (list, optional): list of ylabels. 
                Defaults to None.
        """

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
                        ax.set_xlabel(xlabel=(xlabels[j]), color='black', 
                                    fontsize=font_settings['axes_label_size']-2,
                                    **kwargs)
                    except IndexError:
                        logger.warning(f"Warning: xlabel missing for subplot x{j}")
                        continue
                    j=j+1
            else:
                if ax.is_first_row():
                    try:
                        ax.set_xlabel(xlabel=(xlabels[j]), color='black', 
                                    fontsize=font_settings['axes_label_size']-2,
                                    **kwargs)
                        ax.xaxis.set_label_position('top')
                    except IndexError:
                        logger.warning(f"Warning: xlabel missing for subplot x{j}")
                        continue
                    j=j+1
            if ax.is_first_col():
                try:
                    ax.set_ylabel(ylabel=(ylabels[k]), color='black', 
                                  rotation='vertical', 
                                  fontsize=font_settings['axes_label_size']-2,
                                  **kwargs)
                except IndexError:
                    logger.warning(f"Warning: ylabel missing for subplot y{k}")
                    continue
                k=k+1


class PlotLibrary(SetupSubplot):
    """A library of commonly used plotting methods.
    
    Inherits the SetupSubplot class and takes all the 
    same arguments as it.
    """

    def stackplot(self, df: pd.DataFrame, color_dict: dict = None, 
                 sub_pos: Union[int, Tuple[int, int]] = 0, 
                 ytick_major_fmt: str = 'standard', **kwargs):
        """Creates a stacked area plot.

        Wrapper around matplotlib.stackplot.

        Args:
            df (pd.DataFrame): DataFrame of data to plot.
            color_dict (dict): Colour dictionary, keys should be in data 
                columns.
                Defaults to None.
            ytick_major_fmt (str, optional): Sets the ytick major format.
                Value gets passed to the set_yaxis_major_tick_format method
                Defaults to 'standard'
            sub_pos (Union[int, Tuple[int, int]], optional): Position of subplot,
                can be either a integer or a tuple of 2 integers depending on 
                how SetupSubplot was instantiated. 
                Defaults to 0
        """
        ax = self._check_if_array(sub_pos)
        
        if color_dict:
            color_list = [color_dict.get(x, '#333333') for x in df.columns]
        else:
            color_list=None

        ax.stackplot(df.index.values, df.values.T, linewidth=0,
                     colors=color_list, **kwargs)

        self.set_yaxis_major_tick_format(tick_format=ytick_major_fmt, sub_pos=sub_pos)
        ax.margins(x=0.01)
        # ax.grid(which='both', axis='both', linewidth=0.5)

    def barplot(self, df: pd.DataFrame, color: Union[dict, list] = None,
                stacked: bool = False, sub_pos: Union[int, Tuple[int, int]] = 0, 
                custom_tick_labels: list = None,
                ytick_major_fmt: str = 'standard',
                legend=False, edgecolor='black', 
                linewidth='0.1',
                **kwargs):
        """Creates a bar plot.

        Wrapper around pandas.plot.bar

        Args:
            df (pd.DataFrame): DataFrame of data to plot.
            color (dict): dictionary of colors, dict keys should be 
                found in df columns.
            stacked (bool, optional): Whether to stack bar values. 
                Defaults to False.
            sub_pos (Union[int, Tuple[int, int]], optional): Position of subplot,
                can be either a integer or a tuple of 2 integers depending on 
                how SetupSubplot was instantiated. 
                Defaults to 0
            custom_tick_labels (list, optional): List of custom tick 
                labels to use.
                Defaults to None
            ytick_major_fmt (str, optional): Sets the ytick major format.
                Value gets passed to the set_yaxis_major_tick_format method
                Defaults to 'standard'
        """
        ax = self._check_if_array(sub_pos)

        if isinstance(color, dict):
            color_list = [color.get(x, '#333333') for x in df.columns]
        elif isinstance(color, list):
            color_list = color
        else:
            color_list=None

        df.plot.bar(stacked=stacked,
                    color=color_list, 
                    ax=ax, legend=legend,
                    edgecolor=edgecolor,
                    linewidth=linewidth,
                    **kwargs)
        
        self.set_yaxis_major_tick_format(tick_format=ytick_major_fmt, 
                                            sub_pos=sub_pos)

        # ax.grid(which='both', axis='both', linewidth=0.5)
        # Set x-tick labels
        if isinstance(custom_tick_labels, pd.Index):
            custom_tick_labels = list(custom_tick_labels)
        if custom_tick_labels:
            tick_labels = custom_tick_labels
        else:
            tick_labels = df.index
        self.set_barplot_xticklabels(tick_labels, sub_pos=sub_pos)

    def lineplot(self, data: pd.Series, column=None,
                 color: Union[dict, str] = None,
                 linestyle: str = 'solid',
                 sub_pos: Union[int, Tuple[int, int]] = 0,
                 alpha: int = 1, 
                 ytick_major_fmt: str = 'standard', **kwargs):
        """Creates a line plot.

        Wrapper around matplotlib.plot

        Args:
            data (pd.Series, pd.DataFrame): Series/ df of data to plot.
            column (str): If passing df as data column is required.
            color (dict, str, optional): Color dict or str,
                if dict color is chosen based on df column 
                Defaults to None.
            linestyle (str, optional): Style of line to plot. 
                Defaults to 'solid'.
            sub_pos (Union[int, Tuple[int, int]], optional): Position of subplot,
                can be either a integer or a tuple of 2 integers depending on 
                how SetupSubplot was instantiated. 
                Defaults to 0
            alpha (int, optional): Line opacity. 
                Defaults to 1.
            ytick_major_fmt (str, optional): Sets the ytick major format.
                Value gets passed to the set_yaxis_major_tick_format method
                Defaults to 'standard'
        """
        ax = self._check_if_array(sub_pos)

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
        
        ax.plot(plot_data, 
                linestyle=linestyle,
                color=color,
                alpha=alpha, **kwargs)

        self.set_yaxis_major_tick_format(tick_format=ytick_major_fmt, 
                                            sub_pos=sub_pos)

    def histogram(self, df: pd.DataFrame, color_dict: dict,
                  label=None,
                  sub_pos: Union[int, Tuple[int, int]] = 0, **kwargs):
        """Creates a histogram plot

        Wrapper around matplotlib.hist

        Args:
            df (pd.DataFrame): DataFrame of data to plot.
            color_dict (dict): Colour dictionary
            label (list, optional): List of labels for legend. 
                Defaults to None.
            sub_pos (Union[int, Tuple[int, int]], optional): Position of subplot,
                can be either a integer or a tuple of 2 integers depending on 
                how SetupSubplot was instantiated. 
                Defaults to 0
        """
        ax = self._check_if_array(sub_pos)

        ax.hist(df, bins=20, range=(0,1), color=color_dict[label], zorder=2, 
                    rwidth=0.8, label=label, **kwargs)

    def clustered_stacked_barplot(self, df_list: List[pd.DataFrame], 
                                    labels: list, color_dict: dict, 
                                    title: str = "",  H: str = "//", 
                                    sub_pos: Union[int, Tuple[int, int]] = 0,
                                    ytick_major_fmt='standard', **kwargs):
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
            sub_pos (Union[int, Tuple[int, int]], optional): Position of subplot,
                can be either a integer or a tuple of 2 integers depending on 
                how SetupSubplot was instantiated. 
                Defaults to 0
            ytick_major_fmt (str, optional): Sets the ytick major format.
                Value gets passed to the set_yaxis_major_tick_format method
                Defaults to 'standard'
        """
        ax = self._check_if_array(sub_pos)

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
        
        self.set_yaxis_major_tick_format(tick_format=ytick_major_fmt, 
                                            sub_pos=sub_pos)
        self.add_legend(handles, label_list)

