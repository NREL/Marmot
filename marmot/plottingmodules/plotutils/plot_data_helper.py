# -*- coding: utf-8 -*-
"""Classes, Methods, and functions used to assist with the creation 
   of Marmot plots.

@author: Daniel Levie
"""

import re
import math
import logging
import pandas as pd
import functools
import concurrent.futures
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, List

import marmot.utils.mconfig as mconfig
import marmot.utils.dataio as dataio
from marmot.plottingmodules.plotutils.timeseries_modifiers import adjust_for_leapday

logger = logging.getLogger("plotter." + __name__)
shift_leapday: bool = mconfig.parser("shift_leapday")
curtailment_prop: str = mconfig.parser("plot_data", "curtailment_property")


@dataclass
class GenCategories:
    """Defines various generator categories."""

    vre: List[str] = field(default_factory=list)
    """vre (List[str]): List of variable renewable technologies.
    """
    pv: List[str] = field(default_factory=list)
    """pv (List[str]): List of PV technologies.
    """
    re: List[str] = field(default_factory=list)
    """re (List[str]): List of renewable technologies.
    """
    thermal: List[str] = field(default_factory=list)
    """thermal (List[str]): List of thermal technologies.
    """

    @classmethod
    def set_categories(cls, df: pd.DataFrame) -> "GenCategories":
        """Set generator categories from a dataframe

        Categories include the following:

        - vre
        - pv
        - re
        - thermal

        Args:
            df (pd.DataFrame): Dataframe containing an 'Ordered_Gen'
                column and a column for each generator category. The
                format should appear like the following.

                https://nrel.github.io/Marmot/references/input-files/mapping-folder/
                ordered_gen_categories.html#input-example

        Returns:
            GenCategories: returns instance of class.
        """
        gen_cats = ["vre", "pv", "re", "thermal"]
        gen_cat_dict = {}
        for category in gen_cats:
            if category in df.columns:
                gen_cat_dict[category] = (
                    df.loc[df[category] == True]["Ordered_Gen"].str.strip().tolist()
                )
            else:
                logger.warning(
                    f"'{category}' column was not found in the "
                    "ordered_gen_categories input. Check if the column "
                    "exists in the input file. This is required for "
                    "certain plots to display correctly"
                )
                if category == "vre":
                    logger.warning(
                        "'vre' generator categories not set, "
                        "curtailment will not be defined!"
                    )
        return cls(**gen_cat_dict)


class PlotDataStoreAndProcessor(dict):
    """Methods used to assist with the creation of Marmot plots

    Collection of Methods to assist with creation of figures,
    including getting and formatting data and modifying dataframes

    PlotDataStoreAndProcessor inherits the python class 'dict' so acts like a
    dictionary and stores the formatted data when retrieved by the
    get_formatted_data method.
    """

    def __init__(
        self,
        AGG_BY: str,
        ordered_gen: List[str],
        marmot_solutions_folder: Path,
        gen_names_dict: dict = None,
        TECH_SUBSET: List[str] = None,
        **_,
    ) -> None:
        """
        Args:
            AGG_BY (str): Informs region type to aggregate by when creating plots.
            ordered_gen (List[str]): Ordered list of generator technologies to plot,
                order defines the generator technology position in stacked bar and area plots.
            marmot_solutions_folder (Path): Directory containing Marmot solution outputs.
            gen_names_dict (dict, optional): Mapping dictionary to rename generator
                technologies.
                Default is None.
            TECH_SUBSET (List[str], optional): Tech subset category to plot.
                The TECH_SUBSET value should be a column in the
                ordered_gen_categories.csv. If left None all techs will be plotted
                Defaults to None.
        """
        self.AGG_BY = AGG_BY
        self.ordered_gen = ordered_gen
        self.marmot_solutions_folder = Path(marmot_solutions_folder)

        # Assign input/output folders
        self.processed_hdf5_folder = self.marmot_solutions_folder.joinpath(
            "Processed_HDF5_folder"
        )
        self.figure_folder = self.marmot_solutions_folder.joinpath("Figures_Output")
        self.figure_folder.mkdir(exist_ok=True)
        self.csv_properties_folder = self.marmot_solutions_folder.joinpath(
            "csv_properties"
        )
        self.csv_properties_folder.mkdir(exist_ok=True)

        if gen_names_dict is None:
            logger.warning("'gen_names_dict' is empty! Generators will not be renamed.")
            self.gen_names_dict = {}
        else:
            self.gen_names_dict = gen_names_dict
        self.TECH_SUBSET = TECH_SUBSET

    def get_formatted_data(self, properties: List[tuple]) -> list:
        """Get data from formatted h5 file or csv property input files.

        Adds data to dictionary with scenario name as key

        Args:
            properties (List[tuple]): list of tuples containing required
                plexos property information

        Returns:
            list: If 1 in list required data is missing.
        """
        check_input_data = []

        for prop in properties:
            required, plx_prop_name, scenario_list = prop
            if f"{plx_prop_name}" not in self:
                self[f"{plx_prop_name}"] = {}

            # Create new set of scenarios that are not yet in dictionary
            scen_list = set(scenario_list) - set(self[f"{plx_prop_name}"].keys())
            # If set is not empty add data to dict
            if scen_list:
                # Read data in with multi threading
                executor_func_setup = functools.partial(
                    dataio.read_processed_h5file,
                    self.processed_hdf5_folder,
                    plx_prop_name,
                )
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=mconfig.parser("multithreading_workers")
                ) as executor:
                    data_files = executor.map(executor_func_setup, scen_list)
                # Save data to dict
                for scenario, df in zip(scen_list, data_files):
                    self[f"{plx_prop_name}"][scenario] = df

            # If any of the dataframes are empty for given property log warning
            missing_scen_data = [
                scen for scen, df in self[f"{plx_prop_name}"].items() if df.empty
            ]
            if missing_scen_data:
                if mconfig.parser("read_csv_properties"):
                    logger.info(
                        f"{plx_prop_name} not found in Marmot formatted h5 files, "
                        "attempting to read from csv property file."
                    )
                    for scenario in missing_scen_data:
                        df = dataio.read_csv_property_file(
                            self.csv_properties_folder, plx_prop_name, scenario
                        )
                        self[f"{plx_prop_name}"][scenario] = df
                        if df.empty and required == True:
                            check_input_data.append(1)
                else:
                    logger.warning(
                        f"{plx_prop_name} is MISSING from the Marmot formatted h5 files"
                    )
                    if required == True:
                        check_input_data.append(1)
        return check_input_data

    def rename_gen_techs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Renames generator technologies based on the gen_names.csv file.

        Args:
            df (pd.DataFrame): Dataframe to process.

        Returns:
            pd.DataFrame: Processed DataFrame with renamed techs.
        """
        if self.gen_names_dict:
            # If tech is a column name
            if "tech" in df.columns:
                original_tech_index = df.tech.unique()
                # Checks if all generator tech categories have been identified and matched.
                # If not, lists categories that need a match
                unmapped_techs = set(original_tech_index) - set(
                    self.gen_names_dict.keys()
                )
                df["tech"] = pd.CategoricalIndex(
                    df.tech.map(lambda x: self.gen_names_dict.get(x, "Other"))
                )

            # If tech is in the index
            elif "tech" in df.index.names:
                original_tech_index = df.index.get_level_values(level="tech")
                # Checks if all generator tech categories have been identified and matched.
                # If not, lists categories that need a match
                unmapped_techs = set(original_tech_index) - set(
                    self.gen_names_dict.keys()
                )

                tech_index = pd.CategoricalIndex(
                    original_tech_index.map(
                        lambda x: self.gen_names_dict.get(x, "Other")
                    )
                )
                df.reset_index(level="tech", drop=True, inplace=True)

                idx_map = pd.MultiIndex(
                    levels=df.index.levels + [tech_index.categories],
                    codes=df.index.codes + [tech_index.codes],
                    names=df.index.names + tech_index.names,
                )

                df = pd.DataFrame(data=df.values.reshape(-1), index=idx_map)
                # Move tech back to position 1
                index_labels = list(df.index.names)
                index_labels.insert(1, index_labels.pop(index_labels.index("tech")))
                df = df.reorder_levels(index_labels, axis=0)

            if unmapped_techs:
                logger.warning(
                    "The following Generators could not be re-classified, "
                    f"they wil be renamed 'Other': {unmapped_techs}"
                )
        return df

    def assign_curtailment_techs(
        self, df: pd.DataFrame, vre_techs: list
    ) -> pd.DataFrame:
        """Assign technologies to Marmot's Curtailment property (generator_Curtailment).

        Args:
            df (pd.DataFrame): Dataframe to process.
            vre_techs (list): List of vre tech names, or technologies that should be
                included in curtailment calculations.
        Returns:
            pd.DataFrame: Dataframe containing only specified curtailment technologies.
        """

        # Adjust list of values to drop from vre_gen_cat depending
        # on if it exists in processed techs
        adjusted_vre_gen_list = [name for name in vre_techs if name in df.columns]

        if not adjusted_vre_gen_list:
            logger.warning(
                "Curtailment techs could not be identified correctly for Marmot's "
                "Curtailment property. This is likely happening as the 'vre' column was "
                "not present in the ordered_gen_categories.csv or there "
                "are no vre generators in the selected region"
            )

        # Retrun df with just vre techs
        return df[df.columns.intersection(vre_techs)]

    def df_process_gen_inputs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Processes generation data into a pivot table.

        Also calls rename_gen_techs() to rename technologies
        Technology names will be columns,
        Timeseries as index

        Args:
            df (pd.DataFrame): Dataframe to process.

        Returns:
            pd.DataFrame: Transformed Dataframe.
        """
        if "values" not in df.columns:
            df = df.rename(columns={0: "values"})
        if set(["timestamp", "tech"]).issubset(df.index.names):
            df = df.reset_index(["timestamp", "tech"])
        df = df.groupby(["timestamp", "tech"], as_index=False, observed=True).sum()
        # Rename generator technologies
        df = self.rename_gen_techs(df)
        # If duplicate rows remain, groupby again
        if df[["timestamp", "tech"]].duplicated().any():
            df = df.groupby(["timestamp", "tech"], as_index=False, observed=True).sum()
        # Filter for only data in ordered_gen
        df = df[df.tech.isin(self.ordered_gen)]
        # Check if data is not already categorical
        if df.tech.dtype.name != "category":
            df.tech = df.tech.astype("category")
        df.tech = df.tech.cat.set_categories(self.ordered_gen, ordered=True)
        df = df.sort_values(["tech"])
        df = df.pivot(index="timestamp", columns="tech", values="values")
        return df.fillna(0)

    def create_categorical_tech_index(self, df: pd.DataFrame, axis=0) -> pd.DataFrame:
        """Creates categorical index based on generators.

        Args:
            df (pd.DataFrame): Dataframe to process.

        Returns:
            pd.DataFrame: Processed DataFrame.
        """
        if axis == 0:
            df.index = df.index.astype("category")
            df.index = df.index.set_categories(self.ordered_gen)
        elif axis == 1:
            df.columns = df.columns.astype("category")
            df.columns = df.columns.set_categories(self.ordered_gen)
        df = df.sort_index(axis=axis)
        return df

    def include_net_imports(
        self,
        gen_df: pd.DataFrame,
        load_series: pd.Series,
        unsereved_energy: pd.Series = pd.Series(dtype="float64"),
    ) -> pd.DataFrame:
        """Adds net imports to total and timeseries generation plots.

        Net imports are calculated as load - total generation

        Args:
            gen_df (pd.DataFrame): generation dataframe
            load_series (pd.Series): load series
            unsereved_energy (pd.Series) : unsereved energy series,
                (optional)

        Returns:
            pd.DataFrame: Dataframe with net imports included
        """
        # Do not calculate net imports if using a subset of techs
        if self.TECH_SUBSET:
            logger.info("Net Imports can not be calculated when using TECH_SUBSET")
            return gen_df
        curtailment_name = self.gen_names_dict.get("Curtailment", "Curtailment")
        if curtailment_name in gen_df.columns:
            total_gen = gen_df.drop(curtailment_name, axis=1).sum(axis=1)
        else:
            total_gen = gen_df.sum(axis=1)
        net_imports = load_series.squeeze() - total_gen
        # Remove negative values (i.e exports)
        net_imports = net_imports.clip(lower=0)
        if not unsereved_energy.empty:
            net_imports -= unsereved_energy.squeeze()
        net_imports = net_imports.rename("Net Imports")
        net_imports = net_imports.fillna(0)
        gen_df = pd.concat([gen_df, net_imports], axis=1)
        # In the event of two Net Imports columns combine here
        gen_df = gen_df.groupby(level=0, axis=1, observed=True).sum()
        gen_df = self.create_categorical_tech_index(gen_df, axis=1)
        return gen_df

    def process_extra_properties(
        self,
        extra_properties: List[str],
        scenario: str,
        zone_input: str,
        agg: str,
        data_resolution: str = "",
    ) -> pd.DataFrame:
        """Processes a list of extra properties and saves them into a single dataframe.

        Use with properties that should be aggregated to a
        zonal/regional aggregation such as; Load, Demand and Unsereved Energy.

        Args:
            extra_properties (List[str]): list of extra property names to retrieve from formatted
                data file and process
            scenario (str): scenario to pull data from
            zone_input (str): zone to subset by.
            agg_by (str): Area aggregtaion, zone or region.
            data_resolution (str, optional):  Specifies the data resolution to
                pull from the formatted data and plot.
                Defaults to "".

        Returns:
            pd.DataFrame: Dataframe of extra properties with timeseries index.
        """
        extra_data_frames = []
        # Get and process extra properties
        for ext_prop in extra_properties:
            df: pd.DataFrame = self[ext_prop].get(scenario)
            if df.empty:
                date_index = pd.date_range(
                    start="2010-01-01", periods=1, freq="H", name="timestamp"
                )
                df = pd.DataFrame(data=[0], index=date_index, columns=["values"])
            else:
                df = df.xs(zone_input, level=self.AGG_BY)
                df = df.groupby(["timestamp"]).sum()
            df = df.rename(columns={"values": ext_prop})
            extra_data_frames.append(df)

        extra_plot_data = pd.concat(extra_data_frames, axis=1).fillna(0)

        if extra_plot_data.columns.str.contains("Unserved_Energy").any():
            if (
                extra_plot_data[f"{agg}_Unserved_Energy{data_resolution}"] == 0
            ).all() == False:
                extra_plot_data["Load-Unserved_Energy"] = (
                    extra_plot_data[f"{agg}_Demand{data_resolution}"]
                    - extra_plot_data[f"{agg}_Unserved_Energy{data_resolution}"]
                )

        extra_plot_data = extra_plot_data.rename(
            columns={
                f"{agg}_Load{data_resolution}": "Total Load",
                f"{agg}_Unserved_Energy{data_resolution}": "Unserved Energy",
                f"{agg}_Demand{data_resolution}": "Total Demand",
            }
        )
        return extra_plot_data

    def add_curtailment_to_df(
        self,
        df: pd.DataFrame,
        scenario: str,
        zone_input: str,
        vre_techs: list,
        data_resolution: str = "",
    ) -> pd.DataFrame:
        """Adds curtailment to the passed Dataframe as a new column

        Args:
            df (pd.DataFrame): DataFrame to add curtailment column to
            scenario (str): scenario to pull data from
            zone_input (str): zone to subset by
            vre_techs (list): List of vre tech names, or technologies that should be
                included in curtailment calculations.
            data_resolution (str, optional):  Specifies the data resolution to
                pull from the formatted data and plot.
                Defaults to "".

        Returns:
            pd.DataFrame: DataFrame with added curtailment column.
        """
        curt_df: pd.DataFrame = self[
            f"generator_{curtailment_prop}{data_resolution}"
        ].get(scenario)
        curtailment_name = self.gen_names_dict.get("Curtailment", "Curtailment")

        if not curt_df.empty:
            if shift_leapday:
                curt_df = adjust_for_leapday(curt_df)
            if zone_input in curt_df.index.get_level_values(self.AGG_BY).unique():
                curt_df = curt_df.xs(zone_input, level=self.AGG_BY)
                curt_df = self.df_process_gen_inputs(curt_df)
                # If using Marmot's curtailment property
                if curtailment_prop == "Curtailment":
                    curt_df = self.assign_curtailment_techs(curt_df, vre_techs)
                curt_df = curt_df.sum(axis=1)
                # Remove values less than 0.05 MW
                curt_df[curt_df < 0.05] = 0
                # Insert curtailment into
                df.insert(
                    len(df.columns),
                    column=curtailment_name,
                    value=curt_df,
                )
                # If columns are all 0 remove
                df = df.loc[:, (df != 0).any(axis=0)]
                df = df.fillna(0)
        return df

    def add_battery_gen_to_df(
        self,
        df: pd.DataFrame,
        scenario: str,
        zone_input: str,
        data_resolution: str = "",
    ) -> pd.DataFrame:
        """Adds Battery generation to the passed dataframe.

        Args:
            df (pd.DataFrame): DataFrame to add battery generation to.
            scenario (str): scenario to pull data from
            zone_input (str): zone to subset by
            data_resolution (str, optional):  Specifies the data resolution to
                pull from the formatted data and plot.
                Defaults to "".

        Returns:
            pd.DataFrame: DataFrame with added battery gen column.
        """
        battery_gen: pd.DataFrame = self[f"batterie_Generation{data_resolution}"].get(
            scenario
        )
        battery_discharge_name = self.gen_names_dict.get("battery", "Storage")
        if battery_gen.empty is True:
            logger.info("No Battery generation in selected Date Range")
        else:
            if shift_leapday:
                battery_gen = adjust_for_leapday(battery_gen)

            if zone_input in battery_gen.index.get_level_values(self.AGG_BY).unique():
                battery_gen = battery_gen.xs(zone_input, level=self.AGG_BY)
                battery_gen = battery_gen.groupby("timestamp").sum()
                df.insert(
                    len(df.columns),
                    column=battery_discharge_name,
                    value=battery_gen,
                )
                df = df.fillna(0)
                # In the event of two columns with the same name, combine here.
                df = df.groupby(level=0, axis=1, observed=True).sum()
        return df

    @staticmethod
    def year_scenario_grouper(
        df: pd.DataFrame,
        scenario: str,
        groupby: str = "Scenario",
        additional_groups: list = None,
        **kwargs,
    ) -> pd.DataFrame.groupby:
        """Special groupby method to group dataframes by Scenario or Year-Scenario.

        .. versionadded:: 0.10.0

        Grouping by Year-Scenario is useful for multi year results sets
        where examining results by year is of interest.

        This method is a wrapper around pd.DataFrame.groupby and takes all the
        same arguments.

        Args:
            df (pd.DataFrame): DataFrame to group
            scenario (str): name of the scenario to groupby
            groupby (str, optional): Groupby 'Scenario' or 'Year-Scenario'.
                If Year-Scenario is chosen the year is extracted from the
                DatetimeIndex and appended to the scenario name.
                Defaults to 'Scenario'.
            additional_groups (list, optional): List of any additional columns
                to groupby. Defaults to None.
            **kwargs
                These parameters will be passed to pandas.DataFrame.groupby
                function.

        Raises:
            ValueError: If df.index is not of type type pd.DatetimeIndex or
                            type pd.MultiIndex with level timestamp.
            ValueError: If additional_groups is not a list

        Returns:
            DataFrameGroupBy: Returns a groupby object that contains
                information about the groups.
        """

        if groupby == "Year-Scenario":
            if isinstance(df.index, pd.MultiIndex):
                grouper = [
                    (
                        df.index.get_level_values("timestamp").year.astype(str)
                        + f": {scenario}"
                    ).rename("Scenario")
                ]
            elif isinstance(df.index, pd.DatetimeIndex):
                grouper = [
                    (df.index.year.astype(str) + f": {scenario}").rename("Scenario")
                ]
            else:
                raise ValueError(
                    "'df.index' must be of type pd.DatetimeIndex or "
                    "type pd.MultiIndex with level 'timestamp'"
                )
        elif groupby == "Scenario":
            grouper = [pd.Index([scenario] * len(df.index), name="Scenario")]
        else:
            grouper = [groupby]

        if additional_groups:
            if isinstance(additional_groups, list):
                grouper.extend(additional_groups)
            else:
                raise ValueError("'additional_groups' must be a list")
        return df.groupby(grouper, **kwargs)

    @staticmethod
    def insert_custom_data_columns(
        existing_df: pd.DataFrame, custom_data_file_path: Path
    ) -> pd.DataFrame:
        """Insert custom columns into existing DataFrame before plotting.

        Custom data is loaded from passed custom_data_file_path,
        the custom data file must be a csv.
        Default position of new columns is at the end of the existing DataFrame.
        Specific positions can be selected by including a row with index label
        'column_position'.
        Corresponding column positions can then be included.
        -1 can be passed to insert the column at the end of the DataFrame (rightmost position).

        New rows can also be included but their position can not be changed and are
        appended to end of DataFrame.

        NaN values are returned as 0

        Args:
            existing_df (pd.DataFrame): DataFrame to modify
            custom_data_file_path (Path): path to custom data file

        Returns:
            pd.DataFrame: DataFrame with the newly inserted columns
        """

        if not custom_data_file_path.suffix == ".csv":
            logger.warning(
                "Custom datafile must be a csv, returning " "unmodified DataFrame"
            )
            return existing_df

        custom_input_df = pd.read_csv(custom_data_file_path, index_col=0)

        modifed_df = pd.concat([existing_df, custom_input_df], axis=1, copy=False)
        modifed_df.fillna(0, inplace=True)

        if "column_position" in custom_input_df.index:
            col_pos = custom_input_df.loc["column_position"]

            new_col_order = list(modifed_df.columns)
            for col in custom_input_df:
                if col_pos[col] == -1:
                    new_col_order.append(new_col_order.pop(new_col_order.index(col)))
                else:
                    new_col_order.remove(col)
                    new_col_order.insert(int(col_pos[col]), col)

            modifed_df = modifed_df.reindex(columns=new_col_order)
            modifed_df.drop("column_position", inplace=True)

        return modifed_df

    @staticmethod
    def capacity_energy_unitconversion(
        df: pd.DataFrame, Scenarios: List[str], sum_values: bool = False
    ) -> dict:
        """Unitconversion for capacity and energy figures.

        Takes a pd.DataFrame as input and will then determine the max value
        in the frame.

        If sum_values is True, either rows or columns will be summated before
        determining max value. The axis is chosen automatically based on where
        the scenario entries or datetime index is located. If correct axis
        cannot be determined axis 0 (rows) will be summed.
        This setting should mainly be set to True when potting stacked bar
        and area plots.

        Args:
            df (pd.DataFrame): pandas dataframe
            Scenarios (List[str]):  List of scenarios being processed.
            sum_values (bool, optional): Sum axis values if True.
                Should be set to True for stacked bar and area plots.
                Defaults to False.

        Returns:
            dict: Dictionary containing divisor and units.
        """
        if mconfig.parser("auto_convert_units"):
            if sum_values:
                # Check if scenarios are in index sum across columns
                if isinstance(df.index, pd.MultiIndex) and "Scenario" in df.index.names:
                    sum_axis = 1
                # If index datetime sum across columns
                elif isinstance(df.index, pd.DatetimeIndex):
                    sum_axis = 1
                # If any sceanrio is in the index
                elif any(scen in Scenarios for scen in df.index):
                    sum_axis = 0
                # If sceanrio is contained as a substring in the index
                # (only works for equal length lists scenario and index lists)
                elif [x for x, y in zip(Scenarios, df.index) if re.search(x, y)]:
                    sum_axis = 1
                elif any(scen in Scenarios for scen in df.columns):
                    sum_axis = 0
                else:
                    logger.warning(
                        "Could not determine axis to sum across, "
                        "defaulting to axis 0 (rows)"
                    )
                    sum_axis = 0
                max_value = df.abs().sum(axis=sum_axis).max()
            else:
                max_value = df.abs().to_numpy().max()

            if max_value < 1000 and max_value > 1:
                divisor = 1
                units = "MW"
            elif max_value < 1:
                divisor = 0.001
                units = "kW"
            elif max_value > 999999.9:
                divisor = 1000000
                units = "TW"
            else:
                divisor = 1000
                units = "GW"
        else:
            # Disables auto unit conversion, all values in MW
            divisor = 1
            units = "MW"

        return {"units": units, "divisor": divisor}


#################################################
## Other helper functions
#################################################


def merge_new_agg(
    Region_Mapping: pd.DataFrame, df: pd.DataFrame, AGG_BY: str
) -> pd.DataFrame:
    """Adds new region aggregation in the plotting step.

    This allows one to create a new aggregation without re-formatting the .h5 file.
    Args:
        df (pd.DataFrame): Dataframe to process.

    Returns:
        pd.DataFrame: Same dataframe, with new aggregation level added.
    """
    agg_new = Region_Mapping[["region", AGG_BY]]
    agg_new = agg_new.set_index("region")
    df = df.merge(agg_new, left_on="region", right_index=True)
    return df


def set_facet_col_row_dimensions(
    xlabels=None, ylabels=None, facet: bool = True, multi_scenario: list = None
) -> Tuple[int, int]:
    """Sets facet plot col and row dimensions based on user defined labeles

    Args:
        ylabels (List[str], optional): y-axis labels for facet plots.
            Defaults to None.
        xlabels (List[str], optional): x-axis labels for facet plots.
            Defaults to None.
        facet (bool, optional): Trigger for plotting facet plots.
            Defaults to True.
        multi_scenario (list, optional): List of scenarios.
            Defaults to None.

    Returns:
        Tuple[int, int]: Facet x,y dimensions.
    """

    if not xlabels:
        ncols = 1
    else:
        ncols = len(xlabels)
    if not ylabels:
        nrows = 1
    else:
        nrows = len(ylabels)
    # If the plot is not a facet plot, grid size should be 1x1
    if not facet:
        ncols = 1
        nrows = 1
        return ncols, nrows
    # If no labels were provided or dimensions less than len scenarios use
    # Marmot default dimension settings
    if not xlabels and not ylabels or ncols * nrows < len(multi_scenario):
        logger.info(
            "Dimensions could not be determined from x & y labels - Using Marmot "
            "default dimensions"
        )
        ncols, nrows = set_x_y_dimension(len(multi_scenario))
    return ncols, nrows


def set_x_y_dimension(region_number: int) -> Tuple[int, int]:
    """Sets X,Y dimension of plots without x,y labels.

    Args:
        region_number (int): # regions/scenarios

    Returns:
        Tuple[int, int]: Facet x,y dimensions.
    """
    if region_number >= 5:
        ncols = 3
        nrows = math.ceil(region_number / 3)
    if region_number <= 3:
        ncols = region_number
        nrows = 1
    if region_number == 4:
        ncols = 2
        nrows = 2
    return ncols, nrows
