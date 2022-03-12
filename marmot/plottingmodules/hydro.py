# -*- coding: utf-8 -*-
"""Hydro generator plots.

This module creates hydro analysis plots.

DL: Oct 9th 2021, This plot is in need of work. 
It may not produce production ready figures.

@author: adyreson
"""

import os
import logging
import pandas as pd
import datetime as dt
import matplotlib.ticker as mtick

import marmot.utils.mconfig as mconfig

from marmot.plottingmodules.plotutils.plot_library import SetupSubplot
from marmot.plottingmodules.plotutils.plot_data_helper import PlotDataHelper
from marmot.plottingmodules.plotutils.plot_exceptions import (MissingInputData, DataSavedInModule,
            MissingZoneData)


class MPlot(PlotDataHelper):
    """hydro MPlot class.

    All the plotting modules use this same class name.
    This class contains plotting methods that are grouped based on the
    current module name.
    
    The hydro.py module contains methods that are
    related to hydro generators. 
    
    MPlot inherits from the PlotDataHelper class to assist in creating figures.
    """

    def __init__(self, argument_dict: dict):
        """
        Args:
            argument_dict (dict): Dictionary containing all
                arguments passed from MarmotPlot.
        """
        # iterate over items in argument_dict and set as properties of class
        # see key_list in Marmot_plot_main for list of properties
        for prop in argument_dict:
            self.__setattr__(prop, argument_dict[prop])
        
        # Instantiation of MPlotHelperFunctions
        super().__init__(self.Marmot_Solutions_folder, self.AGG_BY, self.ordered_gen, 
                    self.PLEXOS_color_dict, self.Scenarios, self.ylabels, 
                    self.xlabels, self.gen_names_dict, self.TECH_SUBSET, 
                    Region_Mapping=self.Region_Mapping) 
        
        self.logger = logging.getLogger('plotter.'+__name__)        

    def hydro_continent_net_load(self, start_date_range: str = None, 
                             end_date_range: str = None, **_):
        """Creates a scatter plot of hydro generation vs net load 

        Data is saved within this method.

        Args:
            start_date_range (str, optional): Defines a start date at which to represent data from. 
                Defaults to None.
            end_date_range (str, optional): Defines a end date at which to represent data to.
                Defaults to None.

        Returns:
            DataSavedInModule: DataSavedInModule exception 
        """
        outputs = {}
        
        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True, "generator_Generation", [self.Scenarios[0]])]
        
        # Runs get_formatted_data within PlotDataHelper to populate PlotDataHelper dictionary  
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)

        if 1 in check_input_data:
            return MissingInputData()
        
        for zone_input in self.Zones:
            #Location to save to
            hydro_figures = os.path.join(self.figure_folder, self.AGG_BY + '_Hydro')

            Stacked_Gen_read = self["generator_Generation"].get(self.Scenarios[0])

            self.logger.info("Zone = "+ zone_input)
            self.logger.info("Winter is defined as date range: \
            {} to {}".format(str(start_date_range),str(end_date_range)))
            Net_Load = self.df_process_gen_inputs(Stacked_Gen_read)

            # Calculates Net Load by removing variable gen
            # Adjust list of values to drop depending on if it exists in Stacked_Gen df
            vre_gen_cat = [name for name in self.vre_gen_cat if name in Net_Load.columns]
            Net_Load = Net_Load.drop(labels = vre_gen_cat, axis=1)
            Net_Load = Net_Load.sum(axis=1) # Continent net load
            
            try:
                Stacked_Gen = Stacked_Gen_read.xs(zone_input,level=self.AGG_BY)
            except KeyError:
                self.logger.warning("No Generation in %s",zone_input)
                continue
            del Stacked_Gen_read
            Stacked_Gen= self.df_process_gen_inputs(Stacked_Gen)
            #Removes columns only containing 0
            Stacked_Gen = Stacked_Gen.loc[:, (Stacked_Gen != 0).any(axis=0)] 

        #end weekly loop

            try:
                Hydro_Gen = Stacked_Gen['Hydro']
            except KeyError:
                self.logger.warning("No Hydro Generation in %s", zone_input)
                Hydro_Gen=MissingZoneData()
                continue

            del Stacked_Gen

            #Scatter plot by season
            mplt = SetupSubplot()
            fig, ax = mplt.get_figure()

            ax.scatter(Net_Load[end_date_range:start_date_range],
                        Hydro_Gen[end_date_range:start_date_range], color='black',
                        s=5, label='Non-winter')
            ax.scatter(Net_Load[start_date_range:],Hydro_Gen[start_date_range:],
                        color='blue', s=5, label='Winter', alpha=0.5)
            ax.scatter(Net_Load[:end_date_range],Hydro_Gen[:end_date_range],
                        color='blue', s=5, alpha=0.5)

            ax.set_ylabel('In Region Hydro Generation (MW)',  color='black', rotation='vertical')
            ax.set_xlabel('Continent Net Load (MW)',  color='black', rotation='horizontal')
            mplt.set_yaxis_major_tick_format()
            ax.xaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
            ax.margins(x=0.01)
            # Add title
            if mconfig.parser("plot_title_as_region"):
                mplt.add_main_title(zone_input)
            mplt.add_legend(reverse_legend=True)
            
            fig.savefig(os.path.join(hydro_figures, zone_input + 
                                      f"_Hydro_Versus_Continent_Net_Load_{self.Scenarios[0]}"), 
                         dpi=600, bbox_inches='tight')
        
        outputs = DataSavedInModule()
        return outputs

    def hydro_net_load(self, end: int = 7, timezone: str = "", **_):
        """Line plot of hydro generation vs net load.

        Creates separate plots for each week of the year, or longer depending 
        on 'Day After' value passed through plot_select.csv

        Data is saved within this method.

        Args:
            end (float, optional): Determines length of plot period. 
                Defaults to 7.
            timezone (str, optional): The timezone to display on the x-axes.
                Defaults to "".

        Returns:
            DataSavedInModule: DataSavedInModule exception 
        """
        outputs = {}
        
        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True, "generator_Generation", [self.Scenarios[0]])]
        
        # Runs get_formatted_data within PlotDataHelper to populate PlotDataHelper dictionary  
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)

        if 1 in check_input_data:
            return MissingInputData()
        
        for zone_input in self.Zones:
            self.logger.info("Zone = "+ zone_input)
            
            #Location to save to
            hydro_figures = os.path.join(self.figure_folder, self.AGG_BY + '_Hydro')

            Stacked_Gen_read = self["generator_Generation"].get(self.Scenarios[0])
            
           # The rest of the function won't work if this particular zone can't be found 
           # in the solution file (e.g. if it doesn't include Mexico)
            try:
                Stacked_Gen = Stacked_Gen_read.xs(zone_input,level=self.AGG_BY)
            except KeyError:
                self.logger.warning("No Generation in %s",zone_input)
                continue

            del Stacked_Gen_read
            Stacked_Gen = self.df_process_gen_inputs(Stacked_Gen)

            # Calculates Net Load by removing variable gen
            # Adjust list of values to drop depending on if it exists in Stacked_Gen df
            vre_gen_cat = [name for name in self.vre_gen_cat if name in Stacked_Gen.columns]
            Net_Load = Stacked_Gen.drop(labels = vre_gen_cat, axis=1)
            Net_Load = Net_Load.sum(axis=1)

            # Removes columns that only contain 0
            Stacked_Gen = Stacked_Gen.loc[:, (Stacked_Gen != 0).any(axis=0)]
            try:
                Hydro_Gen = Stacked_Gen['Hydro']
            except KeyError:
                self.logger.warning("No Hydro Generation in %s", zone_input)
                Hydro_Gen=MissingZoneData()
                continue

            del Stacked_Gen

            first_date=Net_Load.index[0]
            #assumes weekly, could be something else if user changes end Marmot_plot_select
            for wk in range(1,53): 

                period_start=first_date+dt.timedelta(days=(wk-1)*7)
                period_end=period_start+dt.timedelta(days=end)
                self.logger.info(str(period_start)+" and next "+str(end)+" days.")
                Hydro_Period = Hydro_Gen[period_start:period_end]
                Net_Load_Period = Net_Load[period_start:period_end]

                # Data table of values to return to main program
                Data_Table_Out = pd.concat([Net_Load_Period, Hydro_Period], axis=1, sort=False)

                #Scatter plot by season
                mplt = SetupSubplot()
                fig, ax = mplt.get_figure()

                ax.plot(Hydro_Period, linewidth=2,
                       color=self.PLEXOS_color_dict.get('Hydro','#333333'),
                       label='Hydro')

                ax.plot(Net_Load_Period, color='black',label='Load')

                ax.set_ylabel('Generation (MW)',  color='black', rotation='vertical')
                ax.set_xlabel(timezone,  color='black', rotation='horizontal')
                mplt.set_yaxis_major_tick_format()
                ax.margins(x=0.01)

                mplt.set_subplot_timeseries_format()

                # Add title                
                if mconfig.parser("plot_title_as_region"):
                    mplt.add_main_title(zone_input)
                # Add legend
                mplt.add_legend(reverse_legend=True)

                fig.savefig(os.path.join(hydro_figures, zone_input + 
                                          f"_Hydro_And_Net_Load_{self.Scenarios[0]}_period_{str(wk)}"),
                             dpi=600, bbox_inches='tight')
                Data_Table_Out.to_csv(os.path.join(hydro_figures, zone_input + 
                                          f"_Hydro_And_Net_Load_{self.Scenarios[0]}_period_{str(wk)}.csv"))
                del fig
                del Data_Table_Out
            #end weekly loop
            #Scatter plot
            
            mplt = SetupSubplot()
            fig, ax = mplt.get_figure()
            ax.scatter(Net_Load, Hydro_Gen, color='black', s=5)

            ax.set_ylabel('In-Region Hydro Generation (MW)',  color='black', rotation='vertical')
            ax.set_xlabel('In-Region Net Load (MW)',  color='black', rotation='horizontal')
            mplt.set_yaxis_major_tick_format()
            ax.xaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
            ax.margins(x=0.01)

            mplt.add_legend(reverse_legend=True)
            
            if mconfig.parser("plot_title_as_region"):
                mplt.add_main_title(zone_input)
            fig.savefig(os.path.join(hydro_figures, zone_input +
                                      f"_Hydro_Versus_Net_Load_{self.Scenarios[0]}"),
                         dpi=600, bbox_inches='tight')
        
        outputs = DataSavedInModule()
        return outputs

