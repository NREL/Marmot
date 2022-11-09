"""Write SIIP metadata to Marmot formatted results file 
"""
import json
import pandas as pd
from pathlib import Path
from marmot.utils.dataio import write_metadata_to_h5

META_KEYS_TO_FUNCTIONS: dict = {
    "Regions": [("format_regions_meta", "objects/regions")],
    "Generator_fuel_mapping": [
        ("format_generator_category_meta", "objects/generators")
    ],
    "Generator_region_mapping": [
        ("format_region_generators_meta", "relations/regions_generators")
    ],
    "Generator_reserve_mapping": [
        ("format_reserve_generators_meta", "relations/reserves_generators")
    ],
    "Lines": [
        ("format_region_intraregionallines", "relations/region_intraregionallines"),
        ("format_region_interregionallines", "relations/region_interregionallines"),
    ],
}
"""json metadata keys to functions and Marmot metadata keys."""


def metadata_to_h5(
    metadata_file: Path, output_file_path: Path, partition: str = "SIIP_metadata"
) -> None:
    """Process and write all SIIP metadata to hdf5 file

    Args:
        metadata_file (Path): Path to SIIP metadata json file
        output_file_path (Path): Path to formatted h5 output file.
        partition (str, optional): Metadata partition.
            Defaults to "SIIP_metadata".
    """
    with open(metadata_file) as f:
        json_data = json.load(f)

    for key in json_data.keys():
        func_key_tup_list = META_KEYS_TO_FUNCTIONS.get(key)
        for func_key in func_key_tup_list:
            meta_func = globals()[func_key[0]]
            df = meta_func(json_data[key])
            write_metadata_to_h5(df, output_file_path, func_key[1], partition)


def format_regions_meta(data: dict) -> pd.DataFrame:
    """Format SIIP regions metadata

    Args:
        data (dict): "Regions" SIIP json metadata entry

    Returns:
        pd.DataFrame: Formatted metadata
    """
    df = pd.DataFrame(data).rename(columns={0: "name"})
    df["category"] = "-"
    return df


def format_generator_category_meta(data: dict) -> pd.DataFrame:
    """Format SIIP generator category metadata

    Args:
        data (dict): "Generator_fuel_mapping" SIIP json metadata entry

    Returns:
        pd.DataFrame: Formatted metadata
    """
    return pd.DataFrame(data.items()).rename(columns={0: "name", 1: "category"})


def format_region_generators_meta(data: dict) -> pd.DataFrame:
    """ "Format SIIP region generator metadata

    Args:
        data (dict): "Generator_region_mapping" SIIP json metadata entry

    Returns:
        pd.DataFrame: Formatted metadata
    """
    return pd.DataFrame(data.items()).rename(columns={0: "child", 1: "parent"})


def format_reserve_generators_meta(data: dict) -> pd.DataFrame:
    """ "Format SIIP reserve generators metadata

    Args:
        data (dict): "Generator_reserve_mapping" SIIP json metadata entry

    Returns:
        pd.DataFrame: Formatted metadata
    """
    df = pd.DataFrame.from_dict(data, orient="index", columns=["child", "parent"])
    df = df.reset_index().rename(columns={"index": "gen_name_reserve"})
    return df


def format_region_intraregionallines(data: dict) -> pd.DataFrame:
    """ "Format SIIP region_intraregionallines metadata

    Args:
        data (dict): "Lines" SIIP json metadata entry

    Returns:
        pd.DataFrame: Formatted metadata
    """
    df = pd.DataFrame.from_dict(
        data, orient="index", columns=["from_region", "to_region"]
    )
    df = df.loc[df.from_region == df.to_region]["from_region"]
    return df.reset_index().rename(columns={"index": "child", "from_region": "parent"})[
        ["parent", "child"]
    ]


def format_region_interregionallines(data: dict) -> pd.DataFrame:
    """ "Format SIIP region_interregionallines metadata

    Args:
        data (dict): "Lines" SIIP json metadata entry

    Returns:
        pd.DataFrame: Formatted metadata
    """
    df = pd.DataFrame.from_dict(
        data, orient="index", columns=["from_region", "to_region"]
    )
    df = df.loc[(df.from_region != df.to_region)]
    df = pd.concat([df.from_region, df.to_region])
    return df.reset_index().rename(columns={"index": "child", 0: "parent"})[
        ["parent", "child"]
    ]
