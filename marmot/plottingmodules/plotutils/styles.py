"""Classes containing color styles and markers for plotting

@author: Daniel Levie
"""

from dataclasses import dataclass, field
from typing import List

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd


@dataclass
class ColorList:
    """List of colors to apply to plots.

    Generally used with non generator plots.
    """

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
            "#b15928",
        ]
    )


@dataclass
class PlotMarkers:
    """List of plot markers to use in certain plots."""

    markers: List[str] = field(
        default_factory=lambda: ["^", "*", "o", "D", "x", "<", "P", "H", "8", "+"]
    )


@dataclass
class GeneratorColorDict:
    """Dictionary of gen names to colors for generation plots.

    The dictionary is usually set with the colour_dictionary.csv using
    the set_colors_from_df method. The file should have the following
    format:

        https://nrel.github.io/Marmot/references/input-files/mapping-folder/colour_dictionary.html

    Random colors can also be set with the set_random_colors method or assigned
    manually directly to the color_dict attribute.

    Args:
        Dictionary of generator names to colors.
    """

    color_dict: dict

    @classmethod
    def set_colors_from_df(cls, color_df: pd.DataFrame) -> "GeneratorColorDict":
        """Sets colors from a dataframe.

        The dataframe should have the following format:

            https://nrel.github.io/Marmot/references/input-files/mapping-folder/colour_dictionary.html

        Args:
            color_df (pd.DataFrame): DataFrame with Generator and Color column

        Returns:
            GeneratorColorDict: Instance of class
        """
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
        """Sets random colors, given a list of names/technologies

        Args:
            generator_list (list): List of generator names.

        Returns:
            GeneratorColorDict: Instance of class
        """
        cmap = plt.cm.get_cmap(lut=len(generator_list))
        colors = []
        for i in range(cmap.N):
            colors.append(mcolors.rgb2hex(cmap(i)))
        color_dict = dict(zip(generator_list, colors))
        return cls(color_dict)
