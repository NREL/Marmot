# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 16:00:00 2021

Exception classes to handle specific actions and results
@author: Daniel Levie
"""


class MissingInputData:
    """Exception Class for handling return of missing data."""
    pass

class MissingZoneData:
    """Exception Class for handling return of zones with no data."""
    pass

class DataSavedInModule:
    """Exception Class for handling data saved within modules."""
    pass

class UnderDevelopment:
    """Exception Class for handling methods under development."""
    pass

class InputSheetError:
    """Exception Class for handling user input sheet errors."""
    pass

class FacetLabelError:
    """Exception Class for incorrect facet labeling."""
    pass
    
class MissingMetaData:
    """Exception Class for missing meta data."""
    pass
    
class UnsupportedAggregation:
    """Exception Class for plotting using unsupported AGG_BY attribute."""
    pass
