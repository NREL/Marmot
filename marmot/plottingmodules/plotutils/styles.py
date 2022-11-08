
from dataclasses import dataclass, field, asdict
from typing import List
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


@dataclass
class ColorList():

    colors: List[str] = field(
        default_factory=lambda: [
            "#396AB1",
            "#CC2529",
            "#3E9651",
            "#ff7f00",
            "#6B4C9A",
            "#922428",
            "#cab2d6",
            "#6a3d9a",
            "#fb9a99",
            "#b15928",]
    )

@dataclass
class PlotMarkers():

    markers: List[str] = field(
        default_factory=lambda: [
            "^", 
            "*", 
            "o", 
            "D", 
            "x", 
            "<", 
            "P", 
            "H", 
            "8", 
            "+"]
    )

@dataclass
class GeneratorColorDict():

    color_dict: dict

    @classmethod
    def set_colors_from_df(cls, color_df: pd.DataFrame) -> "GeneratorColorDict":

        color_dict = color_df.rename(
            columns={
                color_df.columns[0]: "Generator",
                color_df.columns[1]: "Colour",
            }
        )
        color_dict["Generator"] = color_dict["Generator"].str.strip()
        color_dict["Colour"] = color_dict["Colour"].str.strip()
        color_dict = (
            color_dict[["Generator", "Colour"]]
            .set_index("Generator")
            .to_dict()["Colour"]
        )
        return cls(color_dict)

    @classmethod
    def set_random_colors(cls, generator_list: list) -> "GeneratorColorDict":

        cmap = plt.cm.get_cmap(lut=len(generator_list))
        colors = []
        for i in range(cmap.N):
            colors.append(mcolors.rgb2hex(cmap(i)))
        color_dict = dict(zip(generator_list, colors))
        return cls(color_dict)
