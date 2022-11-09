"""Contains logger class used by Marmot

@author: Daniel Levie
"""

from pathlib import Path
import yaml
import logging
import logging.config
from marmot.utils.definitions import ROOT_DIR, LOG_DIR


class SetupLogger:
    """Sets up the python logger.

    This class handles the following.

    1. Configures logger from LOG_CONFIG_FILE file or DEFAULT_LOG_CONFIG.
    2. Handles rollover of log file on each instantiation.
    3. Sets log_directory.
    4. Append optional suffix to the end of the log file name

    Optional suffix is useful when running multiple processes in parallel to
    allow logging to separate files.
    """

    LOG_CONFIG_FILE: str = "marmot_logging_config.yml"
    """Name of default logging config file located in utils package"""

    DEFAULT_LOG_CONFIG: dict = {
        "version": 1,
        "formatters": {
            "info_format": {
                "format": "%(asctime)s:%(levelname)-8s- %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "warning_format": {
                "format": "%(asctime)s:%(levelname)s:%(module)s.%(funcName)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "stream": "ext://sys.stdout",
            },
            "warning_handler": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "WARNING",
                "formatter": "warning_format",
                "filename": "{}/WARNINGS_{}{}.log",
                "backupCount": 3,
                "encoding": "utf8",
                "mode": "a",
            },
            "info_handler": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "INFO",
                "formatter": "info_format",
                "filename": "{}/Log_{}{}.log",
                "backupCount": 3,
                "encoding": "utf8",
                "mode": "a",
            },
        },
        "loggers": {
            "formatter": {
                "level": "INFO",
                "handlers": ["console", "warning_handler", "info_handler"],
                "propagate": True,
            },
            "plotter": {
                "level": "INFO",
                "handlers": ["console", "warning_handler", "info_handler"],
                "propagate": True,
            },
            "root": {"level": "DEBUG", "handlers": ["console"]},
        },
    }
    """Default log config if LOG_CONFIG_FILE cannot be found"""

    def __init__(
        self,
        logger_type: str,
        log_directory: Path = LOG_DIR,
        log_suffix: str = None,
        **kwargs,
    ):
        """
        Args:
            logger_type (str): Type of logger defined in
                'utils/marmot_logging_config.yml'
            log_directory (Path, optional): log directory to save logs.
                Defaults to LOG_DIR.
            log_suffix (str, optional): Optional suffix to add to end of log file.
                Defaults to None.
        """
        if log_suffix is None:
            self.log_suffix = ""
        else:
            self.log_suffix = f"_{log_suffix}"

        if not ROOT_DIR.joinpath("utils", self.LOG_CONFIG_FILE).exists():
            conf = self.DEFAULT_LOG_CONFIG

        else:
            with open(ROOT_DIR.joinpath("utils", self.LOG_CONFIG_FILE), "rt") as f:
                conf = yaml.safe_load(f.read())

        conf["handlers"]["warning_handler"]["filename"] = conf["handlers"][
            "warning_handler"
        ]["filename"].format(log_directory, logger_type, self.log_suffix)
        conf["handlers"]["info_handler"]["filename"] = conf["handlers"]["info_handler"][
            "filename"
        ].format(log_directory, logger_type, self.log_suffix)

        logging.config.dictConfig(conf)

        self.logger = logging.getLogger(logger_type)
        # Creates a new log file for next run
        self.logger.handlers[1].doRollover()
        self.logger.handlers[2].doRollover()
