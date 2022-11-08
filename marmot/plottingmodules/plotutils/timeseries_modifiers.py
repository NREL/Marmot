
import logging
import datetime as dt
from typing import Tuple, Union, List

import numpy as np
import pandas as pd
logger = logging.getLogger("plotter." + __name__)


def set_timestamp_date_range(
    dfs: Union[pd.DataFrame, List[pd.DataFrame]], start_date: str, end_date: str
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, ...]]:
    """Sets the timestamp date range based on start_date and end_date strings

    .. versionadded:: 0.10.0

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
        pd.DataFrame or Tuple[pd.DataFrame]: adjusted dataframes
    """

    logger.info(
        f"Plotting specific date range: \
                {str(start_date)} to {str(end_date)}"
    )

    df_list = []
    if isinstance(dfs, list):
        for df in dfs:
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.loc[start_date:end_date]
            elif isinstance(df.index, pd.MultiIndex):
                df = df.xs(
                    slice(start_date, end_date), level="timestamp", drop_level=False
                )
            else:
                raise ValueError(
                    "'df.index' must be of type pd.DatetimeIndex or "
                    "type pd.MultiIndex with level 'timestamp'"
                )
            df_list.append(df)
        return tuple(df_list)
    else:
        if isinstance(dfs.index, pd.DatetimeIndex):
            df = dfs.loc[start_date:end_date]
        elif isinstance(dfs.index, pd.MultiIndex):
            df = dfs.xs(
                slice(start_date, end_date), level="timestamp", drop_level=False
            )
        else:
            raise ValueError(
                "'df.index' must be of type pd.DatetimeIndex or "
                "type pd.MultiIndex with level 'timestamp'"
            )
        return df

def get_sub_hour_interval_count(df: pd.DataFrame) -> int:
    """Detects the interval spacing of timeseries data.

    Used to adjust sums of certain variables for sub-hourly data.

    Args:
        df (pd.DataFrame): pandas dataframe with timestamp in index.

    Returns:
        int: Number of intervals per 60 minutes.
    """
    timestamps = df.index.get_level_values("timestamp").unique()
    time_delta = timestamps[1] - timestamps[0]
    # Finds intervals in 60 minute period
    intervals_per_hour = 60 / (time_delta / np.timedelta64(1, "m"))
    # If intervals are greater than 1 hour, returns 1
    return max(1, intervals_per_hour)

def adjust_for_leapday(self, df: pd.DataFrame) -> pd.DataFrame:
    """Shifts dataframe ahead by one day.
    
    Use if a non-leap year time series is modeled with a leap year time index.

    Modeled year must be included in the scenario parent directory name.
    Args:
        df (pd.DataFrame): Dataframe to process.

    Returns:
        pd.DataFrame: Same dataframe, with time index shifted.
    """
    if (
        "2008" not in self.processed_hdf5_folder
        and "2012" not in self.processed_hdf5_folder
        and df.index.get_level_values("timestamp")[0]
        > dt.datetime(2024, 2, 28, 0, 0)
    ):

        df.index = df.index.set_levels(
            df.index.levels[df.index.names.index("timestamp")].shift(1, freq="D"),
            level="timestamp",
        )
        return df

def sort_duration(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Converts a dataframe time series into a duration curve.

    Args:
        df (pd.DataFrame): pandas multiindex dataframe.
        col (str): Column name by which to sort.

    Returns:
        pd.DataFrame: Dataframe with values sorted from largest to smallest.
    """
    sorted_duration = (
        df.sort_values(by=col, ascending=False)
        .reset_index()
        .drop(columns=["timestamp"])
    )

    return sorted_duration