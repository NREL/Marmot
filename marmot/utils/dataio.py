import logging
from pathlib import Path
import pandas as pd


def save_to_h5(
    df: pd.DataFrame,
    file_name: Path,
    key: str,
    mode: str = "a",
    complevel: int = 9,
    complib: str = "blosc:zlib",
    **kwargs,
) -> None:
    """Saves data to formatted hdf5 file

    Args:
        df (pd.DataFrame): Dataframe to save
        file_name (Path): name of hdf5 file
        key (str): formatted property identifier,
            e.g generator_Generation
        mode (str, optional): file access mode.
            Defaults to "a".
        complevel (int, optional): compression level.
            Defaults to 9.
        complib (str, optional): compression library.
            Defaults to 'blosc:zlib'.
        **kwargs
            These parameters will be passed pandas.to_hdf function.
    """
    logger = logging.getLogger("formatter." + __name__)
    logger.info("Saving data to h5 file...")
    df.to_hdf(
        file_name,
        key=key,
        mode=mode,
        complevel=complevel,
        complib=complib,
        **kwargs,
    )
    logger.info("Data saved to h5 file successfully\n")


def write_metadata_to_h5(
    df: pd.DataFrame,
    file_name: Path,
    key: str,
    partition: str = "model_metadata",
    mode: str = "a",
    **kwargs,
) -> None:
    """Save metadata to formatted h5 file.

    Args:
        df (pd.DataFrame): Dataframe to save
        file_name (Path): name of hdf5 file
        key (str): metadata key, e.g objects/generators
        partition (str, optional): metadata partition.
            Defaults to "model_metadata".
        mode (str, optional): file access mode.
        Defaults to "a".
        **kwargs
            These parameters will be passed pandas.to_hdf function.
    """
    df.to_hdf(
        file_name,
        key=f"metadata/{partition}/{key}",
        mode=mode,
        **kwargs,
    )


def read_processed_h5file(
    processed_hdf5_folder: Path, plx_prop_name: str, scenario: str
) -> pd.DataFrame:
    """Reads Data from processed h5file.

    Args:
        processed_hdf5_folder (Path): Directory containing Marmot h5 solution files.
        plx_prop_name (str): Name of property, e.g generator_Generation
        scenario (str): Name of scenario.

    Returns:
        pd.DataFrame: Requested dataframe.
    """
    logger = logging.getLogger("plotter." + __name__)

    try:
        with pd.HDFStore(
            processed_hdf5_folder.joinpath(f"{scenario}_formatted.h5"), "r"
        ) as file:
            return file[plx_prop_name]
    except KeyError:
        return pd.DataFrame()


def read_csv_property_file(
    csv_property_folder: Path, plx_prop_name: str, scenario: str
) -> pd.DataFrame:
    """Read formatted data from csv file.

    Allows data to be read in from a csv if it is missing from the
    formatted h5 file. Format of data must adhere to the standard
    Marmot formats for each data class, e.g generator, line etc.

    Filename should be of the following pattern:
    - {scenario}_{plx_prop_name}.csv

    An example of a line_Net_Import:
    - Base DA_line_Net_Import.csv

    The Marmot formatter will not create these files, they must be created manually.

    Args:
        csv_property_folder (Path): Directory containing csv property files.
        plx_prop_name (str): Name of property, e.g generator_Generation
        scenario (str): Name of scenario.

    Returns:
        pd.DataFrame: Requested dataframe or empty dataframe if file not found.
    """
    logger = logging.getLogger("plotter." + __name__)
    try:
        df = pd.read_csv(
            csv_property_folder.joinpath(f"{scenario}_{plx_prop_name}.csv"),
            index_col=False,
        )
        df.timestamp = pd.to_datetime(df.timestamp)
        df_cols = list(df.columns)
        df_cols.pop(df_cols.index("values"))
        df = df.set_index(df_cols)
        return df
    except FileNotFoundError:
        logger.warning(
            f"{scenario}_{plx_prop_name}.csv was not found in "
            f"{csv_property_folder}. Data is MISSING."
        )
        return pd.DataFrame()
