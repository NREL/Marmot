from pathlib import Path

def get_project_root() -> Path:
    """Returns the root directory of the application. 
    Returns:
        [pathlib.Path]: path to root
    """
    ROOT_DIR = Path(__file__).parent.resolve().parent
    return ROOT_DIR

def log_dir() -> Path:
    """Returns the log directory of the application. 
    Returns:
        [pathlib.Path]: path to log dir
    """
    logs_folder =  get_project_root().parent.joinpath("logs")
    logs_folder.mkdir(exist_ok=True)
    return logs_folder

def input_dir() -> Path:
    """Returns the input directory of the application. 
    Returns:
        [pathlib.Path]: path to input dir
    """
    input_folder = get_project_root().joinpath('input_files')
    return input_folder

ROOT_DIR = get_project_root()
LOG_DIR = log_dir()
INPUT_DIR = input_dir()


INCORRECT_ENTRY_POINT = ("Beginning in Marmot v0.10.0, the abilty to run "
    "'python {}' directly has been removed. To run "
    "the code as before, run 'python bin/run_marmot_plotter.py' from " 
    "Marmot Package directory")