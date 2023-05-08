"""
Stateless functions to perform commond data transformations.

1. Calculating Curtailment - Needs Generation and Availability dataframes.

Most functions are also present in the scenariohandlers Classes.


@author: Micah Webb
"""



import pandas as pd
import json
import os


def get_gen_color_map(file):
    """
        Opens a .css file and parses it to get a color mapping.
        Open the css file in VSCode to quickly update mapping with built in Color Wheel.
    """
    color_dict = {}
    with open(file, 'r') as f:

        for line in f:

            temp = line.strip().split(' ')
            gen_type = temp[0].replace("#","")
            color = temp[1].split(':')[1].replace("}","")
            color_dict[gen_type] = color
    return color_dict


def scale_up_down(df):

    max_val = df.max().max() # Get max value of dataframe

    return df


MW_to_TW = 1000000
MW_to_GW = 1000


# minimum requirements
# mapping to standard technology types
# TODO remove this dependence?
curt_tech = ['Wind',"Offshore-Wind", 'PV']


# TODO add labels for each column level. (entity, technology, generator) -> group by entity/technology or just technology.

def get_map_from_file(file_path):
    """
        Gets map from json file or python dictionary. Allows user to pass dictionary objects
        or paths to json files for generating maps required for other data apis.
    """
    map = json.loads(open("tech_map_simple.json", 'r').read())


    return map


def get_raw_generators(scenario_dir):

    gen_path = os.path.normpath(f'{scenario_dir}/generation_actual.pq.gz')
    #option to read from raw h5 files
    #TODO track down negative gas generation
    df = pd.read_parquet(gen_path).applymap(lambda x: x if x >=0.0 else 0.0)
    return df

def get_raw_availability(scenario_dir):

    avail_path = os.path.normpath(f'{scenario_dir}/generation_availability.pq.gz')
    #option to read from raw h5 files
    df = pd.read_parquet(avail_path).applymap(lambda x: x if x >=0.0 else 0.0)

    return df


def get_generators_tech(scenario_dir, tech_map):

    gen_df = get_raw_generators(scenario_dir)

    # TODO replace simple tech map with standard EIA tech map
    gen_df.columns = pd.MultiIndex.from_tuples([(tech_map[col], col) for col in gen_df.columns])

    return gen_df

def get_availability_tech(scenario_dir, tech_map):

    avail_df = get_raw_availability(scenario_dir)

    # TODO replace simple tech map with standard EIA tech map
    avail_df.columns = pd.MultiIndex.from_tuples([(tech_map[col], col) for col in avail_df.columns])

    return avail_df



def calc_curtailment(gen_tech, avail_tech):

    curt_gen = pd.concat([avail_tech[curt_tech], -1*gen_tech[curt_tech]]).groupby(level=0).sum()
    # TODO track down negative curtailment
    return curt_gen.applymap(lambda x: x if x>=0.0 else 0.0)


def get_gen_and_curtailment(scenario_dir, tech_map):

    gen_tech = get_generators_tech(scenario_dir, tech_map)
    avail_tech = get_availability_tech(scenario_dir, tech_map)

    curt_tech = calc_curtailment(gen_tech, avail_tech)

    curt_tech.columns = pd.MultiIndex.from_tuples([("Curtailment", col[1]) for col in curt_tech.columns])

    return pd.merge(gen_tech, curt_tech, left_index=True, right_index=True)


# aggregates across generators
def get_entity_tech_aggregates(scenario_dir, tech_map, entity_map):

    gen_curt_tech = get_gen_and_curtailment(scenario_dir, tech_map)

    gen_curt_tech.columns = pd.MultiIndex.from_tuples([(entity_map[col[1]], col[0],col[1]) for col in gen_curt_tech.columns])

    return gen_curt_tech




def get_regional_load(scenario_dir):

    return pd.read_parquet(f'{scenario_dir}/regional_load.pq.gz')

def get_entity_load(scenario_dir, entity_map):

    df = get_regional_load(scenario_dir)

    df.columns = pd.MultiIndex.from_tuples([
        (entity_map[str(col)], str(col)) if col in entity_map.keys() else ("other", str(col)) for col in df.columns

        ])

    return df


def get_entity_tech_load_aggregates(scenario_dir, tech_map, gen_entity_map, region_entity_map):

    regional_load = get_entity_load(scenario_dir, region_entity_map)
    regional_load_agg = regional_load.groupby(axis=1, level=0).sum()
    regional_load_agg.columns = pd.MultiIndex.from_tuples([(col, "Demand") for col in regional_load_agg.columns])

    gen_tech_load = get_entity_tech_aggregates(scenario_dir, tech_map, gen_entity_map)

    gen_tech_load_agg = gen_tech_load.groupby(axis=1, level=[0,1]).sum()

    gen_tech_load_entity = pd.merge(regional_load_agg, gen_tech_load_agg, left_index=True, right_index=True).sort_index(axis=1)
    gen_tech_load_entity.attrs['units'] = 'MW'

    return gen_tech_load_entity


## Get line flow data


def get_line_flow_data(scenario_dir):
    """
        Get the raw line flow data for a given SIIP scenario
    """

    df = pd.read_parquet(f"{scenario_dir}/power_flow_actual.pq.gz")

    return df



def get_line_load_data(scenario_dir, line_rating_map):
    """
    Gets the raw line flow data and normalizes each value to the corresponding
    Line rating. Requires dictionary/json file of

    """

    flow = get_line_flow_data(scenario_dir)

    loading = pd.DataFrame({col: abs(flow[col]/line_rating_map[col]) for col in flow.columns.values }, index=flow.index)

    #TODO should we have allow for line load to be calculated from peak load instead of dictionary map
    return loading


def get_line_utilization(scenario_dir, line_ratings, threshold=95):
    """
        Gets the U99, U95, U90, etc line load rating. Threshold can be set
        to get desired flags.
        Returns dataframe of bools/ints
    """

    return








