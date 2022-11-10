# -*- coding: utf-8 -*-
"""This module creates the default config.yml file that is used by Marmot.

The parser function is used to parse information from the config file.
The defaults defined here should not be modifed by any user, 
instead edit the values directly in the config.yml file once created. 
"""

from pathlib import Path
from typing import Union

import yaml

from marmot.utils.definitions import ROOT_DIR
from marmot.utils.error_handler import ConfigFileReadError

CONFIGFILE_NAME = "config.yml"
configfile_path = ROOT_DIR.parent.joinpath(CONFIGFILE_NAME)


def createConfig(configfile_path: Path):
    """Creates config.yml file using default values.

    The following are Marmot default config settings that are used to
    create the config.yml file when Marmot is first run.
    Users Should NOT edit these values, instead edit the values directly
    in the config.yml file once created. If the config.yml file is deleted,
    it will be created with these defaults when mconfig.py is called anytime Marmot
    is run.

    The default config settings are as follows:

    - **font_settings:**
        *Settings to adjust font sizes, family, and tick size within figures*

        - xtick_size: 12
        - ytick_size: 12
        - axes_label_size: 16
        - legend_size: 12
        - title_size: 16
        - font_family: serif

    - **text_position:**
        *Adjust the position of text relative to the edge of the plot (matplotlib points)*
        - title_height: 40

    - **figure_size:**
        *Adjust the x and y-axes dimensions of the output figure*
        - xdimension: 6
        - ydimension: 4

    - **axes_options:**
        *Allows adjustment of the minimum and maximum tick marks on datetime x-axes and
        the number of decimal points to include on the y-axes. Also controls visibility of
        figure spines and legend settings and position*

        - x_axes_maxticks: 8
        - x_axes_minticks: 4
        - y_axes_decimalpt: 1
        - major_x_tick_length: 5
        - major_y_tick_length: 5
        - hide_top_spine: true
        - hide_right_spine: true
        - hide_bottom_spine: false
        - hide_left_spine: false
        - legend_position: center right
        - legend_columns: 1
        - show_legend_frame: true

        .. versionadded:: 0.9.0
            legend control settings

    - **axes_label_options:**
        *Controls whether x-axes labels are rotated from their default horizontal (0 degrees),
        and at what number of labels to begin rotating to the specified rotation_angle.
        By default, labels will begin rotating at 7 labels to an angle of 45 degrees from
        0 degrees*

        - rotate_x_labels: true
        - rotate_at_num_labels: 7
        - rotation_angle: 45

    - **plot_data:**
        *Controls certain plot data settings.
        `curtailment_property` source of Curtailment data. The code defaults to
        Marmot's calculated Curtailment property.
        `plot_title_as_region` If True a the region/zone name will be added as a title to the
        figure
        `include_barplot_load_storage_charging_line` specifies whether to include the line
        representing pumped load in total generation bar plots.
        `include_timeseries_load_storage_charging_line` specifies whether to include the line
        representing pumped load in timeseries generation plots
        `*_net_imports` settings controls whether net imports should be included in the figures*

        - curtailment_property: Curtailment
        - plot_title_as_region: true
        - include_barplot_load_lines: true
        - include_stackplot_load_lines: true
        - include_barplot_load_storage_charging_line: true
        - include_timeseries_load_storage_charging_line: true
        - include_barplot_net_imports: true
        - include_stackplot_net_imports: true

    - **load_legend_names:**
        *Sets the legened name of load and demand lines*

        - load: 'Demand + Storage Charging'
        - demand: Demand

        .. versionadded:: 0.10.0

    - **formatter_settings:**
        *Formatter specific settings, VOLL value,
        `skip_existing_properties` Toggles whether existing properties are skipped or
        overwritten if they already contained in a previous processed_h5 file, the default is
        to skip.
        `append_plexos_block_name` Toggles whether to append PLEXOS block name to formatted
        results e.g ST, MT, LT, PASA. Defaults to False.
        `exclude_pumping_from_reeds_storage_gen` toggles whether to exclude pumping
        (negative gen) from ReEDS storage generation. Defaults to True*

        - VoLL: 10000
        - skip_existing_properties: true
        - append_plexos_block_name: false
        - exclude_pumping_from_reeds_storage_gen: true

        .. versionadded:: 0.10.0
            exclude_pumping_from_reeds_storage_gen setting

    - **multithreading_workers:** 1
        *Sets multithread workers when reading data, Defaults to 1

    - **figure_file_format:** svg
        *Adjust the plot image format. The default is **svg**, a vector-based image.
        This field accepts any format that is compatible with matplotlib*

    - **shift_leapday:** false
        *Handles auto shifting of leap day, if required by your model. The default is false*

    - **auto_convert_units:** true
        *If True automatically converts Energy and Capacity units so that no number
        exceeds 1000. All base units are in MW, and units can be converted to GW, TW and kW*

    - **read_csv_properties:** false
        *If True the Marmot plotter will attempt to read the required plot property from a
        csv file if it cannot be found in the formatted h5 file.
        Format of data must adhere to the standard 
        Marmot formats for each data class, e.g generator, line etc.*

        Filename should be of the following pattern:
            - {scenario}_{plx_prop_name}.csv
        
        An example of a line_Net_Import:
            - Base DA_line_Net_Import.csv   
        
        These csv files should be saved in the *csv_properties* folder which will be 
        created in the Marmot_Solutions_folder.

        .. versionadded:: 0.11.0

    - **user_defined_inputs_file:** Marmot_user_defined_inputs.csv
        *Change the default Marmot_user_defined_inputs file, file must be created first*

    - **plot_select_file:** Marmot_plot_select.csv
        *Change the default Marmot_plot_select.csv file, file must be created first*

    - **plexos_properties_file:** plexos_properties.csv
        *Change the default plexos_properties_file.csv file, file must be created first*

    - **reeds_properties_file:** reeds_properties.csv
        *Change the default reeds_properties_file.csv file, file must be created first*

    - **reeds_india_properties_file:** reeds_india_properties.csv
        *Change the default reeds_india_properties_file.csv file, file must be created first*

    - **siip_properties_file:** siip_properties.csv
        *Change the default siip_properties_file.csv file, file must be created first*


    Args:
        configfile_path (Path): Path to config.yml file
    """
    data = dict(
        font_settings=dict(
            xtick_size=12,
            ytick_size=12,
            axes_label_size=16,
            legend_size=12,
            title_size=16,
            font_family="serif",
        ),
        text_position=dict(title_height=40),
        figure_size=dict(xdimension=6, ydimension=4),
        axes_options=dict(
            x_axes_minticks=4,
            x_axes_maxticks=8,
            y_axes_decimalpt=1,
            major_x_tick_length=5,
            major_y_tick_length=5,
            hide_top_spine=True,
            hide_right_spine=True,
            hide_bottom_spine=False,
            hide_left_spine=False,
            legend_position="lower right",
            legend_columns=1,
            show_legend_frame=True,
        ),
        axes_label_options=dict(
            rotate_x_labels=True, rotate_at_num_labels=7, rotation_angle=45
        ),
        plot_data=dict(
            curtailment_property="Curtailment",
            plot_title_as_region=True,
            include_barplot_load_lines=True,
            include_stackplot_load_lines=True,
            include_barplot_load_storage_charging_line=True,
            include_timeseries_load_storage_charging_line=True,
            include_barplot_net_imports=True,
            include_stackplot_net_imports=True,
        ),
        load_legend_names=dict(load="Demand +\nStorage Charging", demand="Demand"),
        formatter_settings=dict(
            VoLL=10000,
            skip_existing_properties=True,
            append_plexos_block_name=False,
            exclude_pumping_from_reeds_storage_gen=True,
        ),
        multithreading_workers=1,
        figure_file_format="svg",
        shift_leapday=False,
        auto_convert_units=True,
        read_csv_properties=False,
        user_defined_inputs_file="Marmot_user_defined_inputs.csv",
        plot_select_file="Marmot_plot_select.csv",
        plexos_properties_file="plexos_properties.csv",
        reeds_properties_file="reeds_properties.csv",
        reeds_india_properties_file="reeds_india_properties.csv",
        siip_properties_file="siip_properties.csv",
    )

    with open(configfile_path, "w") as cfgfile:
        yaml.safe_dump(data, cfgfile, default_flow_style=False, sort_keys=False)


# Check if there is already a configuration file
if not configfile_path.is_file():
    # Create the configuration file as it doesn't exist yet
    createConfig(configfile_path)


def parser(top_level: str, second_level: str = None) -> Union[dict, str, int, bool]:
    """Pull requested value from config.yml file.

    Args:
        top_level (str): Top level of config dictionary,
            will return specified level and any sublevel.
        second_level (str, optional): Second level of config dictionary
            under top_level, will return a single value.
            Defaults to None.

    Returns:
        Union[dict, str, int, bool]: Returns the requested level or value from
        the config file. Return type varies based on levesl accessed.
    """
    with open(configfile_path, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile.read())

    try:
        if not second_level:
            value = cfg[top_level]
        else:
            value = cfg[top_level][second_level]
        return value
    except KeyError as e:
        raise ConfigFileReadError(e.args[0])


def edit_value(new_value: str, top_level: str, second_level: str = None):
    """Edit the config.yml file through code.

    Args:
        new_value (str): New value to apply to config file.
        top_level (str): Top level of config dictionary,
            will return specified level and any sublevel.
        second_level (str, optional): Second level of config dictionary under top_level,
            will return a single value.
            Defaults to None.
    """
    with open(configfile_path, "r") as f:
        cfg = yaml.safe_load(f)

    if not second_level:
        cfg[top_level] = new_value
    else:
        cfg[top_level][second_level] = new_value

    with open(configfile_path, "w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)


def reset_defaults():
    """When called, resets config.yml to default values."""
    createConfig(configfile_path)
