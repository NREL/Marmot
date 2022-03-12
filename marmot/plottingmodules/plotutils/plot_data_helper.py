# -*- coding: utf-8 -*-
"""Methods used to assist with the creation of Marmot plots.

@author: Daniel Levie
"""

import os
import math
import logging
import datetime as dt
import pandas as pd
import numpy as np
import functools
import concurrent.futures
from typing import Tuple, Union, List

import marmot.utils.mconfig as mconfig


logger = logging.getLogger('plotter.'+__name__)


class PlotDataHelper(dict):
    """Methods used to assist with the creation of Marmot plots

    Collection of Methods to assist with creation of figures,
    including getting and formatting data, setting up plot sizes.

    PlotDataHelper inherits the python class 'dict' so acts like a dictionary and stores the
    formatted data when retrieved by the get_formatted_data method.
    """

    def __init__(self, Marmot_Solutions_folder: str, AGG_BY: str, ordered_gen: list, 
                 PLEXOS_color_dict: dict, Scenarios: list, ylabels: list, 
                 xlabels: list, gen_names_dict: dict, TECH_SUBSET: str,
                 Region_Mapping: pd.DataFrame = pd.DataFrame()):
        """
        Args:
            Marmot_Solutions_folder (str): Folder to save Marmot solution files.
            AGG_BY (str): Informs region type to aggregate by when creating plots.
            ordered_gen (list): Ordered list of generator technologies to plot, 
                order defines the generator technology position in stacked bar and area plots
            PLEXOS_color_dict (dict): Dictionary of colors to use for generation technologies
            Scenarios (list): Name of scenarios to process.
            ylabels (list): y-axis labels for facet plots.
            xlabels (list): x-axis labels for facet plots.
            gen_names_dict (dict): Mapping dictionary to rename generator technologies.
            Region_Mapping (pd.DataFrame, optional): Mapping file to map custom regions/zones 
                to create custom aggregations. Aggregations are created by grouping PLEXOS regions.
                Defaults to pd.DataFrame().
        """
        self.Marmot_Solutions_folder = Marmot_Solutions_folder
        self.AGG_BY = AGG_BY
        self.ordered_gen = ordered_gen
        self.PLEXOS_color_dict = PLEXOS_color_dict
        self.Scenarios = Scenarios
        self.ylabels = ylabels
        self.xlabels = xlabels
        self.gen_names_dict = gen_names_dict
        self.TECH_SUBSET = TECH_SUBSET
        self.Region_Mapping = Region_Mapping

    def get_formatted_data(self, properties: list) -> list:
        """Get data from formatted h5 file.
        
        Adds data to dictionary with scenario name as key

        Args:
            properties (list): list of tuples containing required 
                plexos property information

        Returns:
            list: If 1 in list required data is missing .
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
                #Read data in with multi threading
                executor_func_setup = functools.partial(self.read_processed_h5file, plx_prop_name)
                with concurrent.futures.ThreadPoolExecutor(max_workers=mconfig.parser("multithreading_workers")) as executor:
                    data_files = executor.map(executor_func_setup, scen_list)
                # Save data to dict
                for scenario, df in zip(scen_list, data_files):
                    self[f"{plx_prop_name}"][scenario] = df

            # If any of the dataframes are empty for given property log warning
            if True in [df.empty for df in self[f"{plx_prop_name}"].values()]:
                logger.warning(f"{plx_prop_name} is MISSING from the Marmot formatted h5 files")
                if required == True:
                    check_input_data.append(1)
        return check_input_data

    def read_processed_h5file(self, plx_prop_name: str, scenario: str) -> pd.DataFrame:
        """Reads Data from processed h5file.

        Args:
            plx_prop_name (str): Name of property, e.g generator_Generation
            scenario (str): Name of scenario.

        Returns:
            pd.DataFrame: Requested dataframe.
        """
        try:
            with pd.HDFStore(os.path.join(self.Marmot_Solutions_folder, "Processed_HDF5_folder", 
                                            f"{scenario}_formatted.h5"), 'r') as file:
                return file[plx_prop_name]
        except KeyError:
            return pd.DataFrame()
        
    def rename_gen_techs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Renames generator technologies based on the gen_names.csv file.

        Args:
            df (pd.DataFrame): Dataframe to process.

        Returns:
            pd.DataFrame: Processed DataFrame with renamed techs.
        """
        # If tech is a column name
        if 'tech' in df.columns:
            original_tech_index = df.tech.unique()
            # Checks if all generator tech categories have been identified and matched. If not, lists categories that need a match
            unmapped_techs = set(original_tech_index) - set(self.gen_names_dict.keys())
            df['tech'] = pd.CategoricalIndex(df.tech.map(lambda x: self.gen_names_dict.get(x, 'Other')))
        
        # If tech is in the index 
        elif 'tech' in df.index.names:
            original_tech_index = df.index.get_level_values(level='tech')
            # Checks if all generator tech categories have been identified and matched. If not, lists categories that need a match
            unmapped_techs = set(original_tech_index) - set(self.gen_names_dict.keys())
        
            tech_index = pd.CategoricalIndex(original_tech_index.map(lambda x: self.gen_names_dict.get(x, 'Other')))
            df.reset_index(level='tech', drop=True, inplace=True)

            idx_map = pd.MultiIndex(levels=df.index.levels + [tech_index.categories],
                                        codes=df.index.codes + [tech_index.codes],
                                        names=df.index.names + tech_index.names)

            df = pd.DataFrame(data=df.values.reshape(-1), index=idx_map)
            # Move tech back to position 1
            index_labels = list(df.index.names)
            index_labels.insert(1, index_labels.pop(index_labels.index("tech")))
            df = df.reorder_levels(index_labels, axis=0)

        if unmapped_techs:
            logger.warning(f"The following Generators could not be re-classified, they wil be renamed 'Other': {unmapped_techs}")
        return df

    def assign_curtailment_techs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign technologies to Marmot's Curtailment property (generator_Curtailment).

        Args:
            df (pd.DataFrame): Dataframe to process.

        Returns:
            pd.DataFrame: Dataframe containing only specified curtailment technologies.
        """

        # Adjust list of values to drop from vre_gen_cat depending on if it exists in processed techs
        adjusted_vre_gen_list = [name for name in self.vre_gen_cat if name in df.columns]

        if not adjusted_vre_gen_list:
            logger.warning("Curtailment techs could not be identified correctly for Marmot's Curtailment property. "
            "This is likely happening as the 'vre' column was not present in the ordered_gen_categories.csv or there "
            "are no vre generators in the selected region")
 
        # Retrun df with just vre techs
        return df[df.columns.intersection(self.vre_gen_cat)]

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
        if set(['timestamp','tech']).issubset(df.index.names):
            df = df.reset_index(['timestamp','tech'])
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
        df.tech.cat.set_categories(self.ordered_gen, inplace=True)
        df = df.sort_values(["tech"])
        df = df.pivot(index='timestamp', columns='tech', values=0)
        return df.fillna(0)

    def create_categorical_tech_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Creates categorical index based on generators.

        Args:
            df (pd.DataFrame): Dataframe to process.

        Returns:
            pd.DataFrame: Processed DataFrame.
        """
        df.index = df.index.astype("category")
        df.index = df.index.set_categories(self.ordered_gen)
        df = df.sort_index()
        return df

    def merge_new_agg(self, df: pd.DataFrame) -> pd.DataFrame:
        #TODO Needs fixing 
        """Adds new region aggregation in the plotting step.

        This allows one to create a new aggregation without re-formatting the .h5 file.
        Args:
            df (pd.DataFrame): Dataframe to process.

        Returns:
            pd.DataFrame: Same dataframe, with new aggregation level added.
        """
        agg_new = self.Region_Mapping[['region', self.AGG_BY]]
        agg_new = agg_new.set_index('region')
        df = df.merge(agg_new,left_on = 'region', right_index = True)
        return(df)

    def adjust_for_leapday(self, df: pd.DataFrame) -> pd.DataFrame:
        """Shifts dataframe ahead by one day, if a non-leap year time series is modeled with a leap year time index.

        Modeled year must be included in the scenario parent directory name.
        Args:
            df (pd.DataFrame): Dataframe to process.

        Returns:
            pd.DataFrame: Same dataframe, with time index shifted.
        """
        if ('2008' not in self.Marmot_Solutions_folder 
            and '2012' not in self.Marmot_Solutions_folder 
            and df.index.get_level_values('timestamp')[0] > dt.datetime(2024,2,28,0,0)):
            
            df.index = df.index.set_levels(
                df.index.levels[df.index.names.index('timestamp')].shift(1,freq = 'D'),
                level = 'timestamp')
            
        # # Special case where timezone shifting may also be necessary.
        #     df.index = df.index.set_levels(
        #         df.index.levels[df.index.names.index('timestamp')].shift(-3,freq = 'H'),
        #         level = 'timestamp')

        return df

    def set_facet_col_row_dimensions(self, facet: bool = True, 
                                  multi_scenario: list = None) -> Tuple[int, int]:
        """Sets facet plot col and row dimensions based on user defined labeles

        Args:
            facet (bool, optional): Trigger for plotting facet plots. 
                Defaults to True.
            multi_scenario (list, optional): List of scenarios.
                Defaults to None.

        Returns:
            Tuple[int, int]: Facet x,y dimensions.
        """
        ncols=len(self.xlabels)
        if self.xlabels == ['']:
            ncols = 1
        nrows=len(self.ylabels)
        if self.ylabels == ['']:
            nrows = 1
        # If the plot is not a facet plot, grid size should be 1x1
        if not facet:
            ncols = 1
            nrows = 1
            return ncols, nrows
        # If no labels were provided or dimensions less than len scenarios use Marmot default dimension settings
        if self.xlabels == [''] and self.ylabels == [''] or ncols*nrows<len(multi_scenario):
            logger.info("Dimensions could not be determined from x & y labels - Using Marmot default dimensions")
            ncols, nrows = self.set_x_y_dimension(len(multi_scenario))
        return ncols, nrows

    def set_x_y_dimension(self, region_number: int) -> Tuple[int, int]:
        """Sets X,Y dimension of plots without x,y labels.

        Args:
            region_number (int): # regions/scenarios

        Returns:
            Tuple[int, int]: Facet x,y dimensions.
        """
        if region_number >= 5:
            ncols = 3
            nrows = math.ceil(region_number/3)
        if region_number <= 3:
            ncols = region_number
            nrows = 1
        if region_number == 4:
            ncols = 2
            nrows = 2
        return ncols,nrows

    def include_net_imports(self, gen_df: pd.DataFrame, 
                            load_series: pd.Series,
                            unsereved_energy: pd.Series = pd.Series(dtype='float64')) -> pd.DataFrame:
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
        # Check if generators are in columns, if yes transpose gen_df
        if any(gen_type in self.ordered_gen for gen_type in gen_df.columns):
            gen_df = gen_df.T
            transpose_df = True
        else:
            transpose_df = False
        curtailment_name = self.gen_names_dict.get('Curtailment','Curtailment')
        if curtailment_name in gen_df.index:
            total_gen = gen_df.drop(curtailment_name).sum()
        else:
            total_gen = gen_df.sum()
        net_imports = load_series.squeeze() - total_gen
        # Remove negative values (i.e exports)
        net_imports = net_imports.clip(lower=0)
        if not unsereved_energy.empty:
            net_imports -= unsereved_energy.squeeze()
        net_imports = net_imports.rename("Net Imports")
        net_imports = net_imports.fillna(0)
        gen_df = gen_df.append(net_imports)
        gen_df = self.create_categorical_tech_index(gen_df)
        if transpose_df:
           gen_df = gen_df.T 
        return gen_df

    def capacity_energy_unitconversion(self, df: pd.DataFrame, 
                                        sum_values: bool = False) -> dict:
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
            sum_values (bool, optional): Sum axis values if True. 
                Should be set to True for stacked bar and area plots.
                Defaults to False.

        Returns:
            dict: Dictionary containing divisor and units.
        """
        if mconfig.parser("auto_convert_units"):
            if sum_values:
                # Check if scenarios are in index sum across columns
                if any(scen in self.Scenarios for scen in df.index):
                    sum_axis=1
                elif isinstance(df.index, pd.MultiIndex) and\
                    'Scenario' in df.index.names:
                    sum_axis=1
                # If index datetime sum across columns
                elif isinstance(df.index, pd.DatetimeIndex):
                    sum_axis=1  
                elif any(scen in self.Scenarios for scen in df.columns):
                    sum_axis=0
                else:
                    logger.warning("Could not determine axis to sum across, "
                                   "defaulting to axis 0 (rows)")
                    sum_axis=0
                max_value = df.abs().sum(axis=sum_axis).max()
            else:
                max_value = df.abs().to_numpy().max()

            if max_value < 1000 and max_value > 1:
                divisor = 1
                units = 'MW'
            elif max_value < 1:
                divisor = 0.001
                units = 'kW'
            elif max_value > 999999.9:
                divisor = 1000000
                units = 'TW'
            else:
                divisor = 1000
                units = 'GW'
        else:
            # Disables auto unit conversion, all values in MW
            divisor = 1
            units = 'MW'

        return {'units':units, 'divisor':divisor}

    @staticmethod
    def set_timestamp_date_range(dfs: Union[pd.DataFrame, List[pd.DataFrame]],
                        start_date: str, end_date: str) -> Tuple[pd.DataFrame, ...]:
        """Sets the timestamp date range based on start_date and end_date strings

        Takes either a single df or a list of dfs as input. 
        The index must be a pd.DatetimeIndex or a multiindex with level timestamp.

        Args:
            dfs (Union[pd.DataFrame, List[pd.DataFrame]]): df(s) to set date range for
            start_date (str): start date 
            end_date (str): end date

        Raises:
            ValueError: If df.index is not of type type pd.DatetimeIndex or
                            type pd.MultiIndex with level timestamp.

        Returns:
            Tuple[pd.DataFrame]: adjusted dataframes
        """

        logger.info(f"Plotting specific date range: \
                    {str(start_date)} to {str(end_date)}")

        df_list = []
        if isinstance(dfs, list):
            for df in dfs:
                if isinstance(df.index, pd.DatetimeIndex):
                    df = df.loc[start_date : end_date]
                elif isinstance(df.index, pd.MultiIndex):
                    df = df.xs(slice(start_date, end_date), level='timestamp', 
                                    drop_level=False)
                else:
                    raise ValueError("'df.index' must be of type pd.DatetimeIndex or "
                                 "type pd.MultiIndex with level 'timestamp'")
                df_list.append(df)
            return tuple(df_list)
        else:
            if isinstance(dfs.index, pd.DatetimeIndex):
                df = dfs.loc[start_date : end_date]
            elif isinstance(dfs.index, pd.MultiIndex):
                df = dfs.xs(slice(start_date, end_date), level='timestamp', 
                                    drop_level=False)
            else:
                raise ValueError("'df.index' must be of type pd.DatetimeIndex or "
                                "type pd.MultiIndex with level 'timestamp'")
            return df

    @staticmethod
    def year_scenario_grouper(df: pd.DataFrame, scenario: str, 
                                groupby: str='Scenario',
                                additional_groups: list=None,
                                **kwargs) -> pd.DataFrame.groupby:
        """Special groupby method to group dataframes by Scenario or Year-Scenario.

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

        Raises:
            ValueError: If df.index is not of type type pd.DatetimeIndex or
                            type pd.MultiIndex with level timestamp.
            ValueError: If additional_groups is not a list

        Returns:
            DataFrameGroupBy: Returns a groupby object that contains 
                information about the groups. 
        """
        
        if groupby == 'Year-Scenario':
            if isinstance(df.index, pd.MultiIndex):
                grouper = [(df.index.get_level_values('timestamp').year.astype(str) 
                                + f': {scenario}').rename('Scenario')]
            elif isinstance(df.index, pd.DatetimeIndex):
                grouper = [(df.index.year.astype(str) + f'_{scenario}').rename('Scenario')]
            else:
                raise ValueError("'df.index' must be of type pd.DatetimeIndex or "
                                 "type pd.MultiIndex with level 'timestamp'")
        elif groupby == 'Scenario':
            grouper = [pd.Index([scenario] * len(df.index), name='Scenario')]
        else:
            grouper = [groupby]

        if additional_groups:
            if isinstance(additional_groups, list):
                grouper.extend(additional_groups)
            else:
                raise ValueError("'additional_groups' must be a list")
        return df.groupby(grouper, **kwargs)

    @staticmethod
    def get_sub_hour_interval_count(df: pd.DataFrame) -> int:
        """Detects the interval spacing of timeseries data. 
        
        Used to adjust sums of certain variables for sub-hourly data.

        Args:
            df (pd.DataFrame): pandas dataframe with timestamp in index.

        Returns:
            int: Number of intervals per 60 minutes.
        """
        timestamps = df.index.get_level_values('timestamp').unique()
        time_delta = timestamps[1] - timestamps[0]
        # Finds intervals in 60 minute period
        intervals_per_hour = 60/(time_delta/np.timedelta64(1, 'm'))
        # If intervals are greater than 1 hour, returns 1
        return max(1, intervals_per_hour)

    @staticmethod
    def sort_duration(df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Converts a dataframe time series into a duration curve.

        Args:
            df (pd.DataFrame): pandas multiindex dataframe.
            col (str): Column name by which to sort.

        Returns:
            pd.DataFrame: Dataframe with values sorted from largest to smallest.
        """
        sorted_duration = (df.sort_values(by=col, ascending=False)
                        .reset_index()
                        .drop(columns=['timestamp']))
        
        return sorted_duration

    @staticmethod
    def insert_custom_data_columns(existing_df: pd.DataFrame, 
                                   custom_data_file_path: str) -> pd.DataFrame:
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
            custom_data_file_path (str): path to custom data file
            inplace (bool, optional): Modify the DataFrame in place 
                (do not create a new object). 
                Defaults to False.

        Returns:
            pd.DataFrame: DataFrame with the newly inserted columns
        """
        
        if not os.path.basename(custom_data_file_path).endswith('csv'):
            logger.warning("Custom datafile must be a csv, returning " 
                           "unmodified DataFrame")
            return existing_df

        custom_input_df = pd.read_csv(custom_data_file_path, index_col=0) 
        
        modifed_df = pd.concat([existing_df, custom_input_df], axis=1, copy=False)
        modifed_df.fillna(0, inplace=True)

        if 'column_position' in custom_input_df.index:
            col_pos = custom_input_df.loc['column_position']

            new_col_order = list(modifed_df.columns)
            for col in custom_input_df:
                if col_pos[col] == -1:
                    new_col_order.append(new_col_order.pop(new_col_order.index(col)))
                else:
                    new_col_order.remove(col)
                    new_col_order.insert(int(col_pos[col]), col)

            modifed_df = modifed_df.reindex(columns=new_col_order)
            modifed_df.drop('column_position', inplace=True)
        
        return modifed_df