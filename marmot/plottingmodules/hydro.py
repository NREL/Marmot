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
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
from matplotlib.patches import Patch

import marmot.config.mconfig as mconfig
from marmot.plottingmodules.plotutils.plot_data_helper import PlotDataHelper
from marmot.plottingmodules.plotutils.plot_exceptions import (MissingInputData, DataSavedInModule,
            MissingZoneData)


custom_legend_elements = [Patch(facecolor='#DD0200',
                            alpha=0.5, edgecolor='#DD0200',
                         label='Unserved Energy')]

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
        
        self.logger = logging.getLogger('marmot_plot.'+__name__)
        
        self.y_axes_decimalpt = mconfig.parser("axes_options","y_axes_decimalpt")
        

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
            fig2, ax2 = plt.subplots(figsize=(9,6))

            ax2.scatter(Net_Load[end_date_range:start_date_range],
                        Hydro_Gen[end_date_range:start_date_range], color='black',
                        s=5, label='Non-winter')
            ax2.scatter(Net_Load[start_date_range:],Hydro_Gen[start_date_range:],
                        color='blue', s=5, label='Winter', alpha=0.5)
            ax2.scatter(Net_Load[:end_date_range],Hydro_Gen[:end_date_range],
                        color='blue', s=5, alpha=0.5)


            ax2.set_ylabel('In Region Hydro Generation (MW)',  color='black', rotation='vertical')
            ax2.set_xlabel('Continent Net Load (MW)',  color='black', rotation='horizontal')
            ax2.spines['right'].set_visible(False)
            ax2.spines['top'].set_visible(False)
            ax2.tick_params(axis='y', which='major', length=5, width=1)
            ax2.tick_params(axis='x', which='major', length=5, width=1)
            ax2.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(
                                                lambda x, p: format(x, f',.{self.y_axes_decimalpt}f')))
            ax2.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
            ax2.margins(x=0.01)
            if mconfig.parser("plot_title_as_region"):
                ax2.set_title(zone_input)

            handles, labels = ax2.get_legend_handles_labels()

            leg1 = ax2.legend(reversed(handles), reversed(labels), 
                              loc='lower left', bbox_to_anchor=(1,0),
                              facecolor='inherit', frameon=True)

            ax2.add_artist(leg1)
            
            fig2.savefig(os.path.join(hydro_figures, zone_input + 
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
                #print(Net_Load_Period)

                # Data table of values to return to main program
                Data_Table_Out = pd.concat([Net_Load_Period, Hydro_Period], axis=1, sort=False)

                fig1, ax = plt.subplots(figsize=(9,6))

                ax.plot(Hydro_Period, linewidth=2,
                       color=self.PLEXOS_color_dict.get('Hydro','#333333'),label='Hydro')

                ax.plot(Net_Load_Period, color='black',label='Load')

                ax.set_ylabel('Generation (MW)',  color='black', rotation='vertical')
                ax.set_xlabel(timezone,  color='black', rotation='horizontal')
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.tick_params(axis='y', which='major', length=5, width=1)
                ax.tick_params(axis='x', which='major', length=5, width=1)
                ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(
                                                    lambda x, p: format(x, f',.{self.y_axes_decimalpt}f')))
                ax.margins(x=0.01)

                locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
                formatter = mdates.ConciseDateFormatter(locator)
                formatter.formats[2] = '%d\n %b'
                formatter.zero_formats[1] = '%b\n %Y'
                formatter.zero_formats[2] = '%d\n %b'
                formatter.zero_formats[3] = '%H:%M\n %d-%b'
                formatter.offset_formats[3] = '%b %Y'
                formatter.show_offset = False
                ax.xaxis.set_major_locator(locator)
                ax.xaxis.set_major_formatter(formatter)
                
                if mconfig.parser("plot_title_as_region"):
                    ax.set_title(zone_input)

                handles, labels = ax.get_legend_handles_labels()

                #Legend 1
                leg1 = ax.legend(reversed(handles), reversed(labels), 
                                 loc='lower left',bbox_to_anchor=(1,0),
                                 facecolor='inherit', frameon=True)

                # Manually add the first legend back
                ax.add_artist(leg1)

                fig1.savefig(os.path.join(hydro_figures, zone_input + 
                                          f"_Hydro_And_Net_Load_{self.Scenarios[0]}_period_{str(wk)}"),
                             dpi=600, bbox_inches='tight')
                Data_Table_Out.to_csv(os.path.join(hydro_figures, zone_input + 
                                          f"_Hydro_And_Net_Load_{self.Scenarios[0]}_period_{str(wk)}.csv"))
                del fig1
                del Data_Table_Out
                mpl.pyplot.close('all')
            #end weekly loop
            #Scatter plot
            fig2, ax2 = plt.subplots(figsize=(9,6))

            ax2.scatter(Net_Load,Hydro_Gen,color='black',s=5)

            ax2.set_ylabel('In-Region Hydro Generation (MW)',  color='black', rotation='vertical')
            ax2.set_xlabel('In-Region Net Load (MW)',  color='black', rotation='horizontal')
            ax2.spines['right'].set_visible(False)
            ax2.spines['top'].set_visible(False)
            ax2.tick_params(axis='y', which='major', length=5, width=1)
            ax2.tick_params(axis='x', which='major', length=5, width=1)
            ax2.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(
                                            lambda x, p: format(x, f',.{self.y_axes_decimalpt}f')))
            ax2.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
            ax2.margins(x=0.01)

            handles, labels = ax2.get_legend_handles_labels()

            leg1 = ax2.legend(reversed(handles), reversed(labels), 
                              loc='lower left',bbox_to_anchor=(1,0),
                              facecolor='inherit', frameon=True)

            ax2.add_artist(leg1)
            if mconfig.parser("plot_title_as_region"):
                ax2.set_title(zone_input)
            fig2.savefig(os.path.join(hydro_figures, zone_input +
                                      f"_Hydro_Versus_Net_Load_{self.Scenarios[0]}"),
                         dpi=600, bbox_inches='tight')
        
        outputs = DataSavedInModule()
        return outputs

