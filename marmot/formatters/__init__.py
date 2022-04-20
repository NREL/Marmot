import traceback

PROCESS_LIBRARY = {"PLEXOS": None, "ReEDS": None, "EGRET": None, "SIIP": None}

try:
    from .formatplexos import ProcessPLEXOS
    PROCESS_LIBRARY["PLEXOS"] = ProcessPLEXOS
except ModuleNotFoundError:
    PROCESS_LIBRARY["Error"] = traceback.format_exc()

try:
    from .formatreeds import ProcessReEDS
    PROCESS_LIBRARY["ReEDS"] = ProcessReEDS
except ModuleNotFoundError:
    PROCESS_LIBRARY["Error"] = traceback.format_exc()

try:
    from .formategret import ProcessEGRET
    PROCESS_LIBRARY["EGRET"] = ProcessEGRET
except ModuleNotFoundError:
    PROCESS_LIBRARY["Error"] = traceback.format_exc()

try:
    from .formatsiip import ProcessSIIP
    PROCESS_LIBRARY["SIIP"] = ProcessSIIP
except ModuleNotFoundError:
    PROCESS_LIBRARY["Error"] = traceback.format_exc()
