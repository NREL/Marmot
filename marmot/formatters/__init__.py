import traceback

def plexos():
    try:
        from .formatplexos import ProcessPLEXOS
        return ProcessPLEXOS
    except ModuleNotFoundError:
        return traceback.format_exc()

def reeds():
    try:
        from .formatreeds import ProcessReEDS
        return ProcessReEDS
    except ModuleNotFoundError:
        return traceback.format_exc()

def egret():
    try:
        from .formategret import ProcessEGRET
        return ProcessEGRET
    except ModuleNotFoundError:
        return traceback.format_exc()

