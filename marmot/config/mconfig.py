# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 17:44:11 2021

@author: Daniel Levie

This module creates the default config.yml file that is used by Marmot.
The parser function is used to parse information from the config file.
The defaults defined here should not be modifed by any user, instead edit the values directly in the config.yml file once created. 
"""

import os
import yaml

configfile_name = "config.yml"

path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)
configfile_path = os.path.join(dir_path,configfile_name)


def createConfig(configfile_path):
    """
    The following are Marmot default config settings that are used to create the config.yml file when Marmot is first run.
    Users Should NOT edit these values, instead edit the values directly in the config.yml file once created. 
    If the config.yml file is deleted, it will be created with these defaults when mconfig.py is called.
    """
        
    data = dict(
        
        font_settings = dict(
            xtick_size = 11,
            ytick_size = 12,
            axes_label_size = 16,
            legend_size = 11,
            font_family = 'serif'
            ),
        
        figure_size = dict(
            xdimension = 6,
            ydimension = 4),
        
        axes_options = dict(
            x_axes_minticks = 4,
            x_axes_maxticks = 8,
            y_axes_decimalpt = 1),
        
        figure_file_format = 'svg',
        
        shift_leapday = False,
        skip_existing_properties = True,
        auto_convert_units = True,
        plot_title_as_region = True,
        
        user_defined_inputs_file = 'Marmot_user_defined_inputs.csv',
        
        category_file_names = dict(
            pv_gen_cat = 'pv_gen_cat.csv',
            re_gen_cat = 're_gen_cat.csv',
            vre_gen_cat = 'vre_gen_cat.csv',
            thermal_gen_cat = 'thermal_gen_cat.csv'
            ),
        color_dictionary_file = 'colour_dictionary.csv',
        ordered_gen_file = 'ordered_gen.csv'
        )

    with open(configfile_path, "w") as cfgfile:
        yaml.safe_dump(data, cfgfile,default_flow_style=False, sort_keys=False)


# Check if there is already a configurtion file
if not os.path.isfile(configfile_path):
    # Create the configuration file as it doesn't exist yet
    createConfig(configfile_path)
    
    
def parser(top_level, second_level=None): 
    """
    Pull requested value from config.yml file
    
    Parameters
    ----------
    top_level : string
        top level of config dictionary, will return specified level and any sublevel.
    second_level : string, optional
        second level of config dictionary under top_level, will return a single value

    Returns
    -------
    value : varies(dict,string,int,float)
        returns the requested level or value from the config file.

    """
    with open(configfile_path, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile.read())
    
    if not second_level:
        value = cfg[top_level]
    else:
        value = cfg[top_level][second_level]
    return value 
    

def edit_value(new_value, top_level, second_level=None): 
    """
    Edit the config.yml file through code

    Parameters
    ----------
    new_value : string/int/bool
        New value to apply to config file.
    top_level : string
        top level of config dictionary, will return specified level and any sublevel.
    second_level : string, optional
        second level of config dictionary under top_level, will return a single value

    Returns
    -------
    None.

    """

    with open(configfile_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    if not second_level:
        cfg[top_level] = new_value
    else:
        cfg[top_level][second_level] = new_value  
            
    with open(configfile_path,'w') as f:
            yaml.safe_dump(cfg,f,default_flow_style=False, sort_keys=False)
            

def reset_defaults():   
    '''
    When called, resets config.yml to default values

    Returns
    -------
    None.

    '''
    createConfig(configfile_path)
