# -*- coding: utf-8 -*-
"""Main plotting source code, creates output figures and data-tables.

marmot_plot_main.py is the main plotting script within Marmot which calls on 
supporting files to read in data, create the plot, and then return the plot and 
data to marmot_plot_main.py. The supporting modules can be viewed within the repo 
plottingmodules folder and have descriptive names such as total_generation.py, 
generation_stack.py, curtailment.py etc.

@author: Daniel Levie
"""
# ========================================================================================
# Import Python Libraries
# ========================================================================================

import importlib
import sys
import time
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import pandas as pd

try:
    import marmot.utils.mconfig as mconfig
except ModuleNotFoundError:
    from utils.definitions import INCORRECT_ENTRY_POINT

    print(INCORRECT_ENTRY_POINT.format(Path(__file__).name))
    sys.exit()
from marmot.metamanagers.read_metadata import MetaData
from marmot.plottingmodules.plotutils.plot_data_helper import GenCategories
from marmot.plottingmodules.plotutils.plot_exceptions import (
    DataSavedInModule,
    InputSheetError,
    MissingInputData,
    MissingMetaData,
    MissingZoneData,
    UnderDevelopment,
    UnsupportedAggregation,
)
from marmot.plottingmodules.plotutils.styles import (
    ColorList,
    GeneratorColorDict,
    PlotMarkers,
)
from marmot.utils.definitions import INPUT_DIR, Module_CLASS_MAPPING
from marmot.utils.loggersetup import SetupLogger

# A bug in pandas requires this to be included, otherwise df.to_string truncates
# long strings.
# Fix available in Pandas 1.0 but leaving here in case user version not up to date
pd.set_option("display.max_colwidth", 1000)


class MarmotPlot(SetupLogger):
    """Main module class to be instantiated to run the plotter.

    MarmotPlot handles the selection of plotting module to
    create the desired figure and saving of outputs.
    It also handles the area aggregation selection
    """

    def __init__(
        self,
        Scenarios: Union[str, list],
        AGG_BY: str,
        model_solutions_folder: Union[str, Path],
        gen_names_dict: Union[str, Path, pd.DataFrame, dict],
        ordered_gen_categories: Union[str, Path, pd.DataFrame],
        color_dictionary: Union[str, Path, pd.DataFrame, dict],
        marmot_plot_select: Union[str, Path, pd.DataFrame],
        marmot_solutions_folder: Union[str, Path] = None,
        Scenario_Diff: Union[str, list] = None,
        zone_region_sublist: Union[str, list] = None,
        xlabels: Union[str, list] = None,
        ylabels: Union[str, list] = None,
        ticklabels: Union[str, list] = None,
        region_mapping: Union[str, Path, pd.DataFrame] = pd.DataFrame(),
        TECH_SUBSET: Union[str, list] = None,
        **kwargs,
    ):
        """
        Args:
            Scenarios (Union[str, list]): Name of scenarios
                to process.
            AGG_BY (str): Informs region type to aggregate by
                when creating plots.
            model_solutions_folder (Union[str, Path]): Directory containing model simulation
                results subfolders and their files.
            gen_names_dict (Union[str, Path, pd.DataFrame, dict]): Path to, Dataframe or dict
                of generator technologies to rename.
            ordered_gen_categories (Union[str, Path, pd.DataFrame]): Path to or Dataframe
                containing ordered generation and columns to specify technology subsets.
            color_dictionary (Union[str, Path, pd.DataFrame, dict]): Path to, Dataframe or dict
                containing list of colors to assign to each generator category.
            marmot_plot_select (Union[str, Path, pd.DataFrame]): Path to or DataFrame
                containing information on plots to create and certain settings.
            marmot_solutions_folder (Union[str, Path], optional): Directory to save
                Marmot solution files.
                Defaults to None.
            Scenario_Diff (Union[str, list], optional): 2 value string
                or list, used to compare 2 scenarios.
                Defaults to None.
            zone_region_sublist (Union[str, list], optional): Subset of regions
                to plot from AGG_BY.
                Defaults to None.
            xlabels (Union[str, list], optional): x axis labels for facet plots.
                Defaults to None.
            ylabels (Union[str, list], optional): y axis labels for facet plots.
                Defaults to None.
            ticklabels (Union[str, list], optional): custom ticklabels for plots,
                not available for every plot type.
                Defaults to None.
            region_mapping (Union[str, Path, pd.DataFrame], optional): Path to or Dataframe
                to map custom regions/zones to create custom aggregations.
                Aggregations are created by grouping PLEXOS regions.
                Defaults to pd.DataFrame().
            TECH_SUBSET (Union[str, list], optional): Tech subset category to plot.
                The TECH_SUBSET value should be a column in the
                ordered_gen_categories.csv. If left None all techs will be plotted
                Defaults to None.
            **kwargs
                These parameters will be passed to the
                marmot.utils.loggersetup.SetupLogger class.
        """
        super().__init__("plotter", **kwargs)  # Instantiation of SetupLogger

        self.Scenarios = self.convert_str_to_list(Scenarios)
        self.AGG_BY = AGG_BY
        self.model_solutions_folder = Path(model_solutions_folder)
        self.gen_names_dict = gen_names_dict
        self.ordered_gen_categories = ordered_gen_categories
        self.color_dictionary = color_dictionary
        self.marmot_plot_select = marmot_plot_select

        if marmot_solutions_folder is None:
            self.marmot_solutions_folder = self.model_solutions_folder
        else:
            self.marmot_solutions_folder = Path(marmot_solutions_folder)

        self.Scenario_Diff = self.convert_str_to_list(Scenario_Diff)
        self.zone_region_sublist = self.convert_str_to_list(zone_region_sublist)
        self.xlabels = self.convert_str_to_list(xlabels)
        self.ylabels = self.convert_str_to_list(ylabels)
        self.custom_xticklabels = self.convert_str_to_list(ticklabels)
        self.region_mapping = region_mapping
        self.TECH_SUBSET = TECH_SUBSET
        self._ordered_gen_list = None

    @property
    def ordered_gen_list(self) -> list:
        """List of ordered generator technolgies.

        Oder is specified in the ordered_gen_categories input.

        Returns:
            list: Ordered list of generator technolgies
        """

        if self._ordered_gen_list is None:
            # Subset ordered_gen to user desired generation
            if self.TECH_SUBSET:
                if self.TECH_SUBSET not in self.ordered_gen_categories.columns:
                    self.logger.warning(
                        f"{self.TECH_SUBSET} column was not found "
                        "in the ordered_gen_categories.csv. "
                        "All generator technologies will be plotted"
                    )
                    self._ordered_gen_list = (
                        self.ordered_gen_categories["Ordered_Gen"].str.strip().tolist()
                    )
                else:
                    ordered_gen = self.ordered_gen_categories.loc[
                        self.ordered_gen_categories[self.TECH_SUBSET] == True
                    ]
                    self._ordered_gen_list = (
                        ordered_gen["Ordered_Gen"].str.strip().tolist()
                    )
                    self.logger.info(f"Tech Aggregation selected: {self.TECH_SUBSET}")
            else:
                self._ordered_gen_list = (
                    self.ordered_gen_categories["Ordered_Gen"].str.strip().tolist()
                )
            # If Other category does not exist in ordered_gen, create entry
            if "Other" not in self._ordered_gen_list:
                self._ordered_gen_list.append("Other")
        return self._ordered_gen_list

    @property
    def gen_names_dict(self) -> dict:
        """Dictionary of existing gen technology names to new names.

        Used to rename technologies.

        Returns:
            dict: Keys: Existing names, Values: New names
        """
        return self._gen_names_dict

    @gen_names_dict.setter
    def gen_names_dict(self, gen_names_dict) -> None:

        if isinstance(gen_names_dict, (str, Path)):
            try:
                gen_names_dict = pd.read_csv(gen_names_dict)
            except FileNotFoundError:
                msg = (
                    "Could not find specified gen_names_dict csv file; "
                    "check file name and path."
                )
                self.logger.error(msg)
                raise FileNotFoundError(msg)

        if isinstance(gen_names_dict, pd.DataFrame):
            if len(gen_names_dict.axes[1]) == 2:
                self._gen_names_dict = (
                    gen_names_dict.set_index(gen_names_dict.columns[0])
                    .squeeze()
                    .to_dict()
                )
            else:
                msg = (
                    "Expected exactly 2 columns for gen_names_dict input, "
                    f"{len(input.axes[1])} columns were in the DataFrame."
                )
                self.logger.error(msg)
                raise ValueError(msg)
        elif isinstance(gen_names_dict, dict):
            self._gen_names_dict = gen_names_dict
        else:
            msg = (
                "Expected a DataFrame, dict, or a file path to csv for the gen_names_dict input but "
                f"recieved a {type(gen_names_dict)}"
            )
            self.logger.error(msg)
            raise NotImplementedError(msg)

    @property
    def ordered_gen_categories(self) -> pd.DataFrame:
        """DataFrame containing generator order and category information

        Has at least one column named Ordered_Gen.
        Other columns define different generator technology category groupings.

        Returns:
            pd.DataFrame: ordered_gen_categories DataFrame
        """
        return self._ordered_gen_categories

    @ordered_gen_categories.setter
    def ordered_gen_categories(self, ordered_gen_categories) -> None:

        if isinstance(ordered_gen_categories, (str, Path)):
            try:
                ordered_gen_categories = pd.read_csv(ordered_gen_categories)
            except FileNotFoundError:
                msg = (
                    "Could not find specified ordered_gen_categories csv file; "
                    "check file name and path."
                )
                self.logger.error(msg)
                raise FileNotFoundError(msg)
        if isinstance(ordered_gen_categories, pd.DataFrame):
            if "Ordered_Gen" in ordered_gen_categories.columns:
                self._ordered_gen_categories = ordered_gen_categories
            else:
                msg = "Misssing 'Ordered_Gen' column from ordered_gen_categories input."
                self.logger.error(msg)
                raise ValueError(msg)
        else:
            msg = (
                "Expected a DataFrame or a file path to csv for the ordered_gen_categories input but "
                f"recieved a {type(ordered_gen_categories)}"
            )
            self.logger.error(msg)
            raise NotImplementedError(msg)

        # Compare gen_names_dict to ordered_gen_categories
        if (
            set(self.gen_names_dict.values()).issubset(
                self._ordered_gen_categories["Ordered_Gen"].str.strip().tolist()
            )
        ) == False:
            missing_gen = set(self.gen_names_dict.values()) - (
                set(self._ordered_gen_categories["Ordered_Gen"].str.strip().tolist())
            )
            self.logger.warning(
                "The following tech categories from the "
                "gen_names_dict input do not exist in "
                "ordered_gen_categorie input!: "
                f"{missing_gen}"
            )

    @property
    def color_dictionary(self) -> dict:
        """Dictionary of gen technology names to plotting colors.

        Returns:
            dict: Keys gen technologies, Values colors
        """
        return self._color_dictionary

    @color_dictionary.setter
    def color_dictionary(self, color_dictionary) -> None:

        if isinstance(color_dictionary, (str, Path)):
            try:
                color_dictionary = pd.read_csv(color_dictionary)
            except FileNotFoundError:
                msg = (
                    "Could not find specified color dictionary csv file; "
                    "check file name and path."
                )
                self.logger.error(msg)
                raise FileNotFoundError(msg)

        if isinstance(color_dictionary, pd.DataFrame):
            if len(color_dictionary.axes[1]) == 2:
                self._color_dictionary = GeneratorColorDict.set_colors_from_df(
                    color_dictionary
                ).color_dict
            else:
                msg = (
                    "Expected exactly 2 columns for color_dictionary input, "
                    f"{len(color_dictionary.axes[1])} columns were in the DataFrame."
                )
                self.logger.error(msg)
                raise ValueError(msg)
        elif isinstance(color_dictionary, dict):
            self._color_dictionary = GeneratorColorDict(color_dictionary).color_dict
        else:
            msg = (
                "Expected a DataFrame, dict, or file path to csv for the color_dictionary input but "
                f"recieved a {type(color_dictionary)}"
            )
            self.logger.error(msg)
            raise NotImplementedError(msg)

    @property
    def marmot_plot_select(self) -> pd.DataFrame:
        """DataFrame containing information on plots to create and certain settings.

        Returns:
            pd.DataFrame:
        """
        return self._marmot_plot_select

    @marmot_plot_select.setter
    def marmot_plot_select(self, marmot_plot_select) -> None:

        if isinstance(marmot_plot_select, (str, Path)):
            try:
                self._marmot_plot_select = pd.read_csv(marmot_plot_select)
            except FileNotFoundError:
                msg = (
                    "Could not find specified marmot_plot_select csv file; "
                    "check file name and path."
                )
                self.logger.error(msg)
                raise FileNotFoundError(msg)

        elif isinstance(marmot_plot_select, pd.DataFrame):
            self._marmot_plot_select = marmot_plot_select
        else:
            msg = (
                "Expected a DataFrame or a file path to csv for the marmot_plot_select input but "
                f"recieved a {type(marmot_plot_select)}"
            )
            self.logger.error(msg)
            raise NotImplementedError(msg)

    @property
    def region_mapping(self) -> pd.DataFrame:
        """Region mapping Dataframe to map custom aggregations.

        Returns:
            pd.DataFrame:
        """
        return self._region_mapping

    @region_mapping.setter
    def region_mapping(self, region_mapping) -> None:
        if isinstance(region_mapping, (str, Path)):
            try:
                region_mapping = pd.read_csv(region_mapping)
            except FileNotFoundError:
                msg = (
                    "Could not find specified region_mapping csv file; "
                    "check file name and path."
                )
                self.logger.error(msg)
                raise FileNotFoundError(msg)

        if isinstance(region_mapping, pd.DataFrame):
            self._region_mapping = region_mapping.astype(str)
            if "category" in region_mapping.columns:
                # delete category columns if exists
                self._region_mapping = self._region_mapping.drop(["category"], axis=1)
        else:
            msg = (
                "Expected a DataFrame or a file path to csv for the region_mapping input but "
                f"recieved a {type(region_mapping)}"
            )
            self.logger.error(msg)
            raise NotImplementedError(msg)

    @staticmethod
    def convert_str_to_list(string_object: str) -> list:
        """Converts a comma separated string to a list.

        Args:
            string_object (str): A comma separated string

        Returns:
            list: list of strings.
        """
        if isinstance(string_object, str):
            list_obj = [x.strip() for x in string_object.split(",")]
        else:
            list_obj = string_object
        return list_obj

    def get_geographic_regions(self, meta: MetaData) -> list:
        """Gets the geographic regions to plot based on the geographic aggregation.

        The aggregation is determined with the AGG_BY attribute.

        region and zone (PLEXOS) will pull model defined aggregations.
        Other aggregations will use values from the region mapping file.

        The zone_region_sublist attribute can be used to reduce the set of geographic regions
        to plot.

        Args:
            meta (MetaData): instance of MetaData class

        Returns:
            list: List of geographic regions to plot
        """

        if self.AGG_BY in {"zone", "zones", "Zone", "Zones"}:
            self.AGG_BY = "zone"
            zones = pd.concat([meta.zones(scenario) for scenario in self.Scenarios])
            if zones.empty == True:
                self.logger.warning(
                    "Input Sheet Data Incorrect! Your model does "
                    "not contain Zones, enter a different aggregation"
                )
                sys.exit()
            Zones = zones["name"].unique()

            if self.zone_region_sublist:
                zsub = []
                for zone in self.zone_region_sublist:
                    if zone in Zones:
                        zsub.append(zone)
                    else:
                        self.logger.info(
                            "metadata does not contain zone: " f"{zone}, SKIPPING ZONE"
                        )
                if zsub:
                    Zones = zsub
                else:
                    self.logger.warning(
                        f"None of: {self.zone_region_sublist} "
                        "in model Zones. Plotting all Zones"
                    )

        elif self.AGG_BY in {"region", "regions", "Region", "Regions"}:
            self.AGG_BY = "region"
            regions = pd.concat([meta.regions(scenario) for scenario in self.Scenarios])
            if regions.empty == True:
                self.logger.warning(
                    "Input Sheet Data Incorrect! Your model does "
                    "not contain Regions, enter a different aggregation"
                )
                sys.exit()

            Zones = regions["region"].unique()
            if self.zone_region_sublist:
                zsub = []
                for region in self.zone_region_sublist:
                    if region in Zones:
                        zsub.append(region)
                    else:
                        self.logger.info(
                            "metadata does not contain region: "
                            f"{region}, SKIPPING REGION"
                        )
                if zsub:
                    Zones = zsub
                else:
                    self.logger.warning(
                        f"None of: {self.zone_region_sublist} "
                        "in model Regions. Plotting all Regions"
                    )

        elif not self.region_mapping.empty:
            self.logger.info(
                "Plotting Custom region aggregation from " "region_mapping File"
            )
            regions = pd.concat([meta.regions(scenario) for scenario in self.Scenarios])
            self.region_mapping = regions.merge(
                self.region_mapping, how="left", on="region"
            )
            self.region_mapping.dropna(axis=1, how="all", inplace=True)

            try:
                Zones = self.region_mapping[self.AGG_BY].unique()
            except KeyError:
                self.logger.warning(
                    f"AGG_BY = '{self.AGG_BY}' is not in the "
                    "region_mapping File, enter a different aggregation"
                )
                sys.exit()

            # remove any nan that might end  up in list
            Zones = [x for x in Zones if str(x) != "nan"]

            if self.zone_region_sublist:
                zsub = []
                for region in self.zone_region_sublist:
                    if region in Zones:
                        zsub.append(region)
                    else:
                        self.logger.info(
                            "region_mapping File does not contain region: "
                            f"{region}, SKIPPING REGION"
                        )
                if zsub:
                    Zones = zsub
                else:
                    self.logger.warning(
                        f"None of: {self.zone_region_sublist} "
                        "in region_mapping File. Plotting all "
                        f"Regions of aggregation '{self.AGG_BY}'"
                    )
        else:
            self.logger.warning(
                "AGG_BY is not defined correctly, aggregation "
                "specified was not found, system will now exit"
            )
            sys.exit()
        return Zones

    def run_plotter(self):
        """Main method to call to begin plotting figures.

        This method takes no input variables, all required
        variables are passed in via the __init__ method.
        """

        self.logger.info(f"Area Aggregation selected: {self.AGG_BY}")

        if self.zone_region_sublist:
            self.logger.info(
                f"Only plotting {self.AGG_BY}: " f"{self.zone_region_sublist}"
            )

        processed_hdf5_folder = self.marmot_solutions_folder.joinpath(
            "Processed_HDF5_folder"
        )

        figure_format = mconfig.parser("figure_file_format")
        if figure_format == "nan":
            figure_format = "png"

        # Create an instance of MetaData.
        meta = MetaData(processed_hdf5_folder, region_mapping=self.region_mapping)
        Zones = self.get_geographic_regions(meta)

        # ================================================================================
        # Start Main plotting loop
        # ================================================================================

        # Filter for chosen figures to plot
        plot_selection = self.marmot_plot_select.loc[
            self.marmot_plot_select["Plot Graph"] == True
        ]
        plot_selection = plot_selection.sort_values(by=["Marmot Module", "Method"])

        list_modules = plot_selection["Marmot Module"].unique()

        start_timer = time.time()

        for module in list_modules:
            module_plots = plot_selection.loc[plot_selection["Marmot Module"] == module]
            # List of required arguments
            argument_list = [
                Zones,
                self.Scenarios,
                self.AGG_BY,
                self.ordered_gen_list,
                self.marmot_solutions_folder,
            ]
            # dictionary of keyword arguments passed to plotting modules;
            # key names match the instance variables in each module
            argument_dict = {
                "gen_names_dict": self.gen_names_dict,
                "gen_categories": GenCategories().set_categories(
                    self.ordered_gen_categories
                ),
                "marmot_color_dict": self.color_dictionary,
                "Scenario_Diff": self.Scenario_Diff,
                "ylabels": self.ylabels,
                "xlabels": self.xlabels,
                "custom_xticklabels": self.custom_xticklabels,
                "color_list": ColorList().colors,
                "marker_style": PlotMarkers().markers,
                "region_mapping": self.region_mapping,
                "TECH_SUBSET": self.TECH_SUBSET,
            }

            # Import plot module from plottingmodules package
            plot_module = importlib.import_module("marmot.plottingmodules." + module)
            # Instantiate the module class

            class_name = getattr(plot_module, Module_CLASS_MAPPING[module])
            instantiate_mplot = class_name(*argument_list, **argument_dict)
            # Create output folder for each plotting module
            figures: Path = instantiate_mplot.figure_folder.joinpath(
                f"{self.AGG_BY}_{module}"
            )
            figures.mkdir(exist_ok=True)

            # Main loop to process each figure and pass
            # plot specific variables to methods
            for _, row in module_plots.iterrows():

                print("\n\n\n")
                self.logger.info(f"Plot =  {row['Figure Output Name']}")

                if pd.isna(row.iloc[2]):
                    prop = None
                else:
                    prop = row.iloc[2]
                # Modifies timezone string before plotting
                if pd.isna(row.iloc[6]):
                    timezone_string: str = "Date"
                else:
                    timezone_string: str = f"Date ({row.iloc[6]})"

                if pd.isna(row.iloc[4]):
                    days_before = 2
                else:
                    days_before = float(row.iloc[4])
                if pd.isna(row.iloc[5]):
                    days_after = 2
                else:
                    days_after = float(row.iloc[5])

                if pd.notna(row["Custom Data File"]):
                    custom_data_file_path = Path(row["Custom Data File"])
                else:
                    custom_data_file_path = None

                if (
                    pd.notna(row["Timeseries Plot Resolution"])
                    and row["Timeseries Plot Resolution"] == "Annual"
                ):
                    data_resolution: str = "_Annual"
                else:
                    data_resolution: str = ""

                if row["Group by Scenario or Year-Scenario"] == "Year-Scenario":
                    scenario_groupby: str = "Year-Scenario"
                else:
                    scenario_groupby: str = "Scenario"

                # Get figure method and run plot
                try:
                    figure_method = getattr(instantiate_mplot, row["Method"])
                except AttributeError:
                    self.logger.warning(
                        f"{Module_CLASS_MAPPING[module]} has no attribute '{row['Method']}'"
                    )
                    continue
                Figure_Out = figure_method(
                    figure_name=row.iloc[0],
                    prop=prop,
                    y_axis_max=float(row.iloc[3]),
                    start=days_before,
                    end=days_after,
                    timezone=timezone_string,
                    start_date_range=row.iloc[7],
                    end_date_range=row.iloc[8],
                    custom_data_file_path=custom_data_file_path,
                    data_resolution=data_resolution,
                    scenario_groupby=scenario_groupby,
                )

                if isinstance(Figure_Out, MissingInputData):
                    self.logger.info(
                        "Add Inputs With Formatter Before " "Attempting to Plot!\n"
                    )
                    continue

                if isinstance(Figure_Out, DataSavedInModule):
                    self.logger.info(
                        f"Plotting Completed for " f'{row["Figure Output Name"]}\n'
                    )
                    self.logger.info("Plots & Data Saved Within Module!\n")
                    continue

                if isinstance(Figure_Out, UnderDevelopment):
                    self.logger.info("Plot is Under Development, Plotting Skipped!\n")
                    continue

                if isinstance(Figure_Out, InputSheetError):
                    self.logger.info("Input Sheet Data Incorrect!\n")
                    continue

                if isinstance(Figure_Out, MissingMetaData):
                    self.logger.info(
                        "Required Meta Data Not Available For " "This Plot!\n"
                    )
                    continue

                if isinstance(Figure_Out, UnsupportedAggregation):
                    self.logger.info(
                        f"Aggregation Type: '{self.AGG_BY}' "
                        "not supported for This plot!\n"
                    )
                    continue

                for zone_input in Zones:
                    if isinstance(Figure_Out[zone_input], MissingZoneData):

                        self.logger.info(f"No Data to Plot in {zone_input}")

                    else:
                        # Save figures
                        Figure_Out[zone_input]["fig"].savefig(
                            figures.joinpath(
                                f"{zone_input}_"
                                f'{row["Figure Output Name"]}'
                                f".{figure_format}"
                            ),
                            dpi=600,
                            bbox_inches="tight",
                        )
                        # Save .csv's.
                        if Figure_Out[zone_input]["data_table"].empty:
                            self.logger.info(
                                f'{row["Figure Output Name"]} '
                                "does not return a data table"
                            )
                        else:
                            Figure_Out[zone_input]["data_table"].to_csv(
                                figures.joinpath(
                                    f"{zone_input}_" f'{row["Figure Output Name"]}.csv'
                                )
                            )

                self.logger.info(
                    "Plotting Completed for " f'{row["Figure Output Name"]}\n'
                )

                # plt.tight_layout()
                # plt.show()
                plt.close("all")

        end_timer = time.time()
        time_elapsed = end_timer - start_timer
        self.logger.info(f"Main Plotting loop took {round(time_elapsed/60,2)} minutes")
        self.logger.info("All Plotting COMPLETED")
        meta.close_h5()


def main():
    """Run the plotting code and create desired plots and data-tables based on user input files."""
    # ====================================================================================
    # Load Input Properties
    # ====================================================================================

    Marmot_user_defined_inputs = pd.read_csv(
        INPUT_DIR.joinpath(mconfig.parser("user_defined_inputs_file")),
        usecols=["Input", "User_defined_value"],
        index_col="Input",
        skipinitialspace=True,
    )

    marmot_plot_select = pd.read_csv(
        INPUT_DIR.joinpath(mconfig.parser("plot_select_file"))
    )

    # Folder to save your processed solutions
    if pd.isna(
        Marmot_user_defined_inputs.loc["Marmot_Solutions_folder", "User_defined_value"]
    ):
        marmot_solutions_folder = None
    else:
        marmot_solutions_folder = Marmot_user_defined_inputs.loc[
            "Marmot_Solutions_folder", "User_defined_value"
        ].strip()

    Scenarios = Marmot_user_defined_inputs.loc["Scenarios", "User_defined_value"]

    # These variables (along with region_mapping) are used to initialize MetaData
    model_solutions_folder = Marmot_user_defined_inputs.loc[
        "Model_Solutions_folder", "User_defined_value"
    ].strip()

    # For plots using the difference of the values between two scenarios.
    # Max two entries, the second scenario is subtracted from the first.
    if pd.isna(
        Marmot_user_defined_inputs.loc["Scenario_Diff_plot", "User_defined_value"]
    ):
        Scenario_Diff = None
    else:
        Scenario_Diff = Marmot_user_defined_inputs.loc[
            "Scenario_Diff_plot", "User_defined_value"
        ]

    Mapping_folder = INPUT_DIR.joinpath("mapping_folder")

    if pd.isna(
        Marmot_user_defined_inputs.loc["Region_Mapping.csv_name", "User_defined_value"]
    ):
        region_mapping = pd.DataFrame()
    else:
        region_mapping = Mapping_folder.joinpath(
            Marmot_user_defined_inputs.loc[
                "Region_Mapping.csv_name", "User_defined_value"
            ]
        )

    gen_names_dict = Mapping_folder.joinpath(
        Marmot_user_defined_inputs.loc["gen_names.csv_name", "User_defined_value"]
    )

    ordered_gen_cat_file = Mapping_folder.joinpath(
        Marmot_user_defined_inputs.loc[
            "ordered_gen_categories_file", "User_defined_value"
        ]
    )
    color_dictionary = Mapping_folder.joinpath(
        Marmot_user_defined_inputs.loc["color_dictionary_file", "User_defined_value"]
    )

    AGG_BY = Marmot_user_defined_inputs.loc["AGG_BY", "User_defined_value"].strip()

    if pd.notna(Marmot_user_defined_inputs.loc["TECH_SUBSET", "User_defined_value"]):
        TECH_SUBSET = Marmot_user_defined_inputs.loc[
            "TECH_SUBSET", "User_defined_value"
        ].strip()
    else:
        TECH_SUBSET = None

    # Facet Grid Labels (Based on Scenarios)
    if pd.isna(
        Marmot_user_defined_inputs.loc["zone_region_sublist", "User_defined_value"]
    ):
        zone_region_sublist = None
    else:
        zone_region_sublist = Marmot_user_defined_inputs.loc[
            "zone_region_sublist", "User_defined_value"
        ]

    if pd.isna(Marmot_user_defined_inputs.loc["Facet_ylabels", "User_defined_value"]):
        ylabels = None
    else:
        ylabels = Marmot_user_defined_inputs.loc["Facet_ylabels", "User_defined_value"]

    if pd.isna(Marmot_user_defined_inputs.loc["Facet_xlabels", "User_defined_value"]):
        xlabels = None
    else:
        xlabels = Marmot_user_defined_inputs.loc["Facet_xlabels", "User_defined_value"]

    # option to change tick labels on plot
    if pd.isna(Marmot_user_defined_inputs.loc["Tick_labels", "User_defined_value"]):
        ticklabels = None
    else:
        ticklabels = Marmot_user_defined_inputs.loc["Tick_labels", "User_defined_value"]

    initiate = MarmotPlot(
        Scenarios,
        AGG_BY,
        model_solutions_folder,
        gen_names_dict,
        ordered_gen_cat_file,
        color_dictionary,
        marmot_plot_select,
        marmot_solutions_folder=marmot_solutions_folder,
        Scenario_Diff=Scenario_Diff,
        zone_region_sublist=zone_region_sublist,
        xlabels=xlabels,
        ylabels=ylabels,
        ticklabels=ticklabels,
        region_mapping=region_mapping,
        TECH_SUBSET=TECH_SUBSET,
    )

    initiate.run_plotter()


if __name__ == "__main__":
    main()
