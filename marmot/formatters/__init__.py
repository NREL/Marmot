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


def reeds_india():
    try:
        from .formatreeds_india import ProcessReEDSIndia

        return ProcessReEDSIndia
    except ModuleNotFoundError:
        return traceback.format_exc()


def egret():
    try:
        from .formategret import ProcessEGRET

        return ProcessEGRET
    except ModuleNotFoundError:
        return traceback.format_exc()


def siip():
    try:
        from .formatsiip import ProcessSIIP

        return ProcessSIIP
    except ModuleNotFoundError:
        return traceback.format_exc()
