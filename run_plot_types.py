# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 08:30:00 2020

@author: rhousema
"""


# import os
# import sys

import importlib

class plottypes:
    
    def __init__(self, figure_type, figure_output_name, argument_list):
        self.figure_type = figure_type
        self.figure_output_name = figure_output_name
        self.argument_list = argument_list
        
    def runmplot(self):
        plot = importlib.import_module(self.figure_type)
        fig = plot.mplot(self.argument_list)
        
        process_attr = getattr(fig, self.figure_output_name)
        
        Figure_Out = process_attr()
        return Figure_Out
    
    
    
    
# Cases that stil need to be worked out
            
        # Facet Plots
        
        # Plotting methods that return empty data frames for missing data
            # Should runmplot include that if/else for all methods?
        



