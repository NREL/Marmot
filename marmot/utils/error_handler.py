from dataclasses import dataclass
import pandas as pd


@dataclass
class ConfigFileReadError(KeyError):
    """Raise when a requested key cannot be found in the config.yml file"""

    key: str

    def __post_init__(self) -> None:
        self.message = (
            f"{self.key} could not be found in the config.yml. "
            "New config settings may have been added which will "
            "require the config.yml to be re-created.\n"
            "To continue delete config.yml located in the top directory level of Marmot."
        )

    def __str__(self) -> str:
        return self.message


@dataclass
class PropertyNotFound(KeyError):
    """Raise when the get_processed_data method cannot find the specified
    property in the simulation model solution files"""

    property: str
    prop_class: str

    def __post_init__(self) -> None:
        self.message = (
            f"CAN NOT FIND '{self.prop_class} {self.property}'. "
            f"'{self.property}' DOES NOT EXIST.\nSKIPPING PROPERTY\n"
        )

    def __str__(self) -> str:
        return self.message


@dataclass
class MissingH5PLEXOSDataError(Exception):
    """Raise when the data key in the H5PLEXOS file is empty"""

    h5_file: str

    def __post_init__(self) -> None:
        self.message = (
            f"There is no data in {self.h5_file} file, "
            "The file may be corrupted or missing information. "
            "Check the file before continuing the formatter."
        )

    def __str__(self) -> str:
        return self.message


@dataclass
class ReEDSColumnLengthError(ValueError):
    """Raised when there is a length mismatch between ReEDS df and ReEDSPropertyColumns"""

    df: pd.DataFrame
    prop_columns: list
    prop: str
    class_instance: str

    def __post_init__(self) -> None:
        self.message = (
            f"ReEDS df has {len(self.df.columns)} columns, "
            f"trying to assign {len(self.prop_columns)} elements for "
            f"property {self.prop}.\n\n"
            f"'{self.prop}' DataFrame has the following format:\n{self.df.head(5)}\n\n"
            "Trying to apply the following column names:\n"
            f"{self.prop_columns}\n\n"
            f"Adjust the {self.prop} field values in {self.class_instance} class"
        )

    def __str__(self) -> str:
        return self.message


@dataclass
class ReEDSYearTypeConvertError(ValueError):
    """Raised when ReEDS df.year column cannot be converted to type int"""

    df: pd.DataFrame
    prop: str
    class_instance: str

    def __post_init__(self) -> None:
        self.message = (
            "year column cannot be converted to type int.\nThis is likely due to an "
            f"incorrectly ordered {self.class_instance} variable, for property '{self.prop}'.\n\n"
            f"{self.prop} DataFrame has the following format:\n{self.df.head(5)}\n\n"
            f"Check column order in {self.class_instance} and try running the formatter again.\n"
            f"If the issue persists open a GitHub issue."
        )

    def __str__(self) -> str:
        return self.message
