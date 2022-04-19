import logging
import json
import pandas as pd
from pathlib import Path



class WriteSIIPMetaData():

    META_KEYS_TO_METHODS = {
        "Regions": "write_regions",
        "Generator_fuel_mapping": "write_generator_category",
        "Generator_region_mapping": "write_region_generators",
        "Generator_reserve_mapping": "write_reserve_generators"
    }

    def __init__(self, metadata_file: Path, output_file_path: Path) -> None:
        
        self.metadata_file = metadata_file
        self.output_file_path = output_file_path

    @classmethod
    def write_to_h5(cls, metadata_file: Path, output_file_path: Path, 
                partition: str = "SIIP_metadata"):
        
        meta_cls = cls(metadata_file, output_file_path)

        with open(metadata_file) as f:
            json_data = json.load(f)

        for key in json_data.keys():
            meta_method = getattr(meta_cls, meta_cls.META_KEYS_TO_METHODS.get(key))
            meta_method(json_data[key], partition)

    def write_regions(self, data: dict, partition: str):
        
        df = pd.DataFrame(data).rename(columns={0: "name"})
        df["category"] = "-"
        df.to_hdf(self.output_file_path, 
                f"metadata/{partition}/objects/regions",
                mode="a")

    def write_generator_category(self, data: dict, partition: str):

        df = pd.DataFrame(data.items()).rename(columns={0: "name", 1: "category"})
        df.to_hdf(self.output_file_path, 
                f"metadata/{partition}/objects/generators",
                mode="a")

    def write_region_generators(self, data: dict, partition: str):

        df = pd.DataFrame(data.items()).rename(columns={0: "child", 1: "parent"})
        df.to_hdf(self.output_file_path, 
                f"metadata/{partition}/relations/regions_generators",
                mode="a")

    def write_reserve_generators(self, data: dict, partition: str):

        df = pd.DataFrame.from_dict(data, orient='index', columns=["child", "parent"])
        df = df.reset_index().rename(columns={"index": "gen_name_reserve"})
        df.to_hdf(self.output_file_path, 
                f"metadata/{partition}/relations/reserves_generators",
                mode="a")
