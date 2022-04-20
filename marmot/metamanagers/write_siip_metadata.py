import logging
import json
import pandas as pd
from pathlib import Path
from marmot.utils.dataio import metadata_to_h5


class WriteSIIPMetaData():

    META_KEYS_TO_METHODS = {
        "Regions": ("write_regions", "objects/regions"),
        "Generator_fuel_mapping": ("write_generator_category", "objects/generators"),
        "Generator_region_mapping": ("write_region_generators", "relations/regions_generators"),
        "Generator_reserve_mapping": ("write_reserve_generators", "relations/reserves_generators"),
    }

    def __init__(self, metadata_file: Path, output_file_path: Path,
                partition: str = "SIIP_metadata") -> None:
        
        self.metadata_file = metadata_file
        self.output_file_path = output_file_path
        self.partition = partition

    @classmethod
    def write_to_h5(cls, metadata_file: Path, output_file_path: Path, 
                partition: str = "SIIP_metadata") -> None:
        
        meta_cls = cls(metadata_file, output_file_path, partition)

        with open(metadata_file) as f:
            json_data = json.load(f)

        for key in json_data.keys():
            method_key_tup = meta_cls.META_KEYS_TO_METHODS.get(key)
            meta_method = getattr(meta_cls, method_key_tup[0])
            df = meta_method(json_data[key])
            metadata_to_h5(df, output_file_path, method_key_tup[1], partition)

    @staticmethod
    def write_regions(data: dict) -> pd.DataFrame:
        
        df = pd.DataFrame(data).rename(columns={0: "name"})
        df["category"] = "-"
        return df

    @staticmethod
    def write_generator_category(data: dict) -> pd.DataFrame:

        df = pd.DataFrame(data.items()).rename(columns={0: "name", 1: "category"})
        return df

    @staticmethod
    def write_region_generators(data: dict) -> pd.DataFrame:

        df = pd.DataFrame(data.items()).rename(columns={0: "child", 1: "parent"})
        return df

    @staticmethod
    def write_reserve_generators(data: dict) -> pd.DataFrame:

        df = pd.DataFrame.from_dict(data, orient='index', columns=["child", "parent"])
        df = df.reset_index().rename(columns={"index": "gen_name_reserve"})
        return df
