# -*- coding: utf-8 -*-
"""This module creates the default config.yml file that is used by Marmot.

The parser function is used to parse information from the config file.
The defaults defined here should not be modifed by any user, 
instead edit the values directly in the config.yml file once created. 
"""

import yaml
import traceback
import sys
from pathlib import Path
from typing import Union
from marmot.utils.definitions import ROOT_DIR

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

        - xtick_size: 12
        - ytick_size: 12
        - axes_label_size: 16
        - legend_size: 12
        - title_size: 16
        - font_family: serif

    *Settings to adjust font sizes, family, and tick size within figures*

    - **text_position:**

        - title_height: 40

    *Adjust the position of text relative to the edge of the plot (matplotlib points)*

    - **figure_size:**

        - xdimension: 6
        - ydimension: 4

    *Adjust the x and y-axes dimensions of the output figure*

    - **axes_options:**

        - x_axes_maxticks: 8
        - x_axes_minticks: 4
        - y_axes_decimalpt: 1

    *Allows adjustment of the minimum and maximum tick marks on datetime x-axes and the number of decimal
    points to include on the y-axes*

    - **axes_label_options:**

        - rotate_x_labels: true
        - rotate_at_num_labels: 7
        - rotation_angle: 45

    *Controls whether x-axes labels are rotated from their default horizontal (0 degrees),
    and at what number of labels to begin rotating to the specified rotation_angle.
    By default, labels will begin rotating at 7 labels to an angle of 45 degrees from 0 degrees*

    - **plot_data:**

        - curtailment_property: Curtailment
        - include_barplot_load_storage_charging_line: false
        - include_timeseries_load_storage_charging_line: true
        - include_barplot_net_imports: true
        - include_stackplot_net_imports: true

    *Controls certain plot data settings. `curtailment_property` source of Curtailment data.
    The code defaults to Marmot's calculated Curtailment property. `include_barplot_load_storage_charging_line`
    specifies whether to include the line representing pumped load in total generation bar plots.
    `include_timeseries_load_storage_charging_line` specifies whether to include the line representing pumped load
    in timeseries generation plots*

    - **figure_file_format:** svg

    *Adjust the plot image format. The default is **svg**, a vector-based image.
    This field accepts any format that is compatible with matplotlib*

    - **shift_leapday:** false

    *Handles auto shifting of leap day, if required by your model. The default is false*

    - **skip_existing_properties:** true

    *Toggles whether existing properties are skipped or overwritten if they already contained in
    a previous processed_h5 file, the default is to skip*

    - **append_plexos_block_name:** true

    *Toggles whether to append PLEXOS block name to formatted results e.g ST, MT, LT, PASA.
    defaults to append.

    - **auto_convert_units:** true

    *If True automatically converts Energy and Capacity units so that no number exceeds 1000.
    All base units are in MW, and units can be converted to GW, TW and kW*

    - **plot_title_as_region:** true

    *If True a the region/zone name will be added as a title to the figure*

    - **user_defined_inputs_file:** Marmot_user_defined_inputs.csv

    *Change the default Marmot_user_defined_inputs file, file must be created first*

    - **plot_select_file:** Marmot_plot_select.csv

    *Change the default Marmot_plot_select.csv file, file must be created first*

    - **plexos_properties_file:** plexos_properties.csv

    *Change the default plexos_properties_file.csv file, file must be created first*

    - **reeds_properties_file:** reeds_properties.csv

    *Change the default reeds_properties_file.csv file, file must be created first*


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
        multithreading_workers=16,
        figure_file_format="svg",
        shift_leapday=False,
        auto_convert_units=True,
        user_defined_inputs_file="Marmot_user_defined_inputs.csv",
        plot_select_file="Marmot_plot_select.csv",
        plexos_properties_file="plexos_properties.csv",
        reeds_properties_file="reeds_properties.csv",
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
        Union[dict, str, int, bool]: Returns the requested level or value from the config file.
        Return type varies based on level accessed.
    """
    with open(configfile_path, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile.read())

    try:
        if not second_level:
            value = cfg[top_level]
        else:
            value = cfg[top_level][second_level]
        return value
    except KeyError:
        print(traceback.format_exc())
        print(
            "Config file read Error: New config settings have been added which "
            "require the config.yml to be re-created. "
            "To continue delete config.yml located in the top directory level of Marmot"
        )
        sys.exit()


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
