
"""Contains logger class used by Marmot

@author: Daniel Levie
"""

from pathlib import Path
import yaml
import logging
import logging.config
from marmot.utils.definitions import ROOT_DIR, LOG_DIR


class SetupLogger():
    """Sets up the python logger.

    This class handles the following.

    1. Configures logger from marmot_logging_config.yml file.
    2. Handles rollover of log file on each instantiation.
    3. Sets log_directory.
    4. Append optional suffix to the end of the log file name

    Optional suffix is useful when running multiple processes in parallel to 
    allow logging to separate files.
    """

    def __init__(self, logger_type: str, log_directory: Path = LOG_DIR, 
                 log_suffix: str = None):
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
            self.log_suffix = ''
        else:
             self.log_suffix = f'_{log_suffix}'

        with open(ROOT_DIR.joinpath('utils/marmot_logging_config.yml'), 'rt') as f:
            conf = yaml.safe_load(f.read())
            conf['handlers']['warning_handler']['filename'] = \
                (conf['handlers']['warning_handler']['filename']
                .format(log_directory, 'formatter', self.log_suffix))
            conf['handlers']['info_handler']['filename'] = \
                (conf['handlers']['info_handler']['filename']
                .format(log_directory, 'formatter', self.log_suffix))

            logging.config.dictConfig(conf)

        self.logger = logging.getLogger(logger_type)
        # Creates a new log file for next run
        self.logger.handlers[1].doRollover()
        self.logger.handlers[2].doRollover()
