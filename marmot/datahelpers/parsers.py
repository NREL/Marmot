import pandas as pd
import os
from glob import iglob
import h5py


def combine_frames_skip_prev(frames):

    """
    Combines multiple dataframes and skips previously simulated dates

    First: Orders dataframes by the max(TimeStamp)

    Second: Drops/reduces rows for each dataframe and combines into single frame.

    """
    frame_dict = {}
    end_dates = []

    for df in frames:

        last_idx = df.index.max()
        end_dates.append(last_idx)
        frame_dict[last_idx] = df

    # Sort the end dates
    end_dates.sort()

    # Filter each frame to exclude overlapping timestamps
    for i in range(1, len(end_dates)):

        prev_date = end_dates[i-1]
        curr_end = end_dates[i]

        curr_df = frame_dict[curr_end]
        new_df = curr_df[curr_df.index > prev_date]
        frame_dict[curr_end] = new_df

    final_frames = [df for end_date, df in frame_dict.items()]

    agg_df = pd.concat(final_frames)

    return agg_df



def extract_h5_data(file_path, freq, partition, dataset):


    """
        Takes a single h5 file and
        extracts a dataset

        This method acts as a base function that provides the underlying datasets
        without much modification /aggregation
    """

    with h5py.File(file_path) as h5data:

        headers = h5data[f'/metadata/objects/{partition}'][()]
        columns = [r[0].decode() for r in headers]


        data = h5data[f'/data/ST/{freq}/{partition}/{dataset}'][()]
        timestamps = pd.to_datetime([val.decode() for val in h5data[f'/metadata/times/{freq}'][()]])

        attributes = h5data[f'/data/ST/{freq}/{partition}/{dataset}'].attrs
        period_offset = attributes['period_offset']
        units = attributes['units'].decode()

        piv_data = data.squeeze().transpose()
        df = pd.DataFrame(piv_data, columns=columns, index=timestamps[period_offset:period_offset+len(piv_data)])
        #df.columns = pd.MultiIndex.from_tuples( ((col, dataset + f' ({units})') for col in df.columns) )

        return df

# Inputs
    # Directory, List of Filepaths,

# Outputs
    #

def get_plexos_paths(plexos_dir):

    return [os.path.normpath(file) for file in iglob(f"{plexos_dir}/*.h5")]


def agg_plexos_dataset(plexos_dir, freq, partition, dataset):

    """
        Input: directory of h5 files
        Output: dataframe of combined files
    """

    paths = get_plexos_paths(plexos_dir)

    frames = []
    for path in paths:

        df = extract_h5_data(path, freq, partition, dataset)

        frames.append(df)

    agg_df = combine_frames_skip_prev(frames)

    return agg_df





def agg_plexos_generation(plexos_dir):
    """
        Input: directory of h5 files
        Output: dataframe of combined files
    """
    return agg_plexos_dataset(plexos_dir, 'interval', 'generators', 'Generation')


def agg_plexos_availability(plexos_dir):

    return agg_plexos_dataset(plexos_dir, 'interval','generators', 'Available Capacity')


def agg_plexos_load(plexos_dir):

    return agg_plexos_dataset(plexos_dir, 'interval', 'nodes', 'Load')


def parse_h5_map(file_path, metadata_path, reverse=False):

    with h5py.File(file_path) as h5data:

        metadata = h5data[metadata_path][()]
    if reverse:
        return {val[1].decode(): val[0].decode() for val in metadata}
    else:
        return {val[0].decode(): val[1].decode() for val in metadata}


def get_h5_gen_tech_map(file_path):

    return parse_h5_map(file_path, 'metadata/objects/generators')


def get_h5_gen_entity_map(file_path, entity):

    return



def get_h5_gen_zone_map(file_path):

    return parse_h5_map(file_path, 'metadata/relations/zones_generators', reverse=True)

def get_h5_region_zone_map(file_path):

    return parse_h5_map(file_path, 'metadata/relations/nodes_zone')


def get_h5_gen_region_map(file_path):

    return parse_h5_map(file_path, 'metadata/relations/regions_generators', reverse=True)

def get_h5_region_region_map(file_path):

    return parse_h5_map(file_path, 'metadata/relations/nodes_region')








# Plexos Handler from raw h5

# Input = directory

# Cache the maps, base datasets

#