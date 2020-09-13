# -*- coding: utf-8 -*-
"""

price analysis plots, price duration cureves = timeseries plots

@author: adyreson and Daniel Levie
"""

import os
import pandas as pd
#import datetime as dt
import matplotlib.pyplot as plt
#import matplotlib as mpl
import matplotlib.dates as mdates
#import numpy as np
import math


#===============================================================================

class mplot(object):
    def __init__(self, argument_dict):
        # iterate over items in argument_dict and set as properties of class
        # see key_list in Marmot_plot_main for list of properties
        for prop in argument_dict:
            self.__setattr__(prop, argument_dict[prop])

    def region_pdc(self):

        """
        This method creates a price duration curve for each region.
        The code will create either a facet plot or a single plot depening on if the Facet argument is active.
        If a facet plot is created, each scenario is plotted on a seperate facet, otherwise all scenarios are
        plotted on a single plot.
        """
        #Duration curve of individual region prices
        # Create Dictionary to hold Datframes for each scenario
        Price_Collection = {}
        self._getdata(Price_Collection)

        outputs = {}
        for zone_input in self.Zones:
            print(self.AGG_BY + " = " + zone_input)

            all_prices=[]
            for scenario in self.Multi_Scenario:

                price = self._process_data(Price_Collection,scenario,zone_input)
                price.sort_values(by=scenario,ascending=False,inplace=True)
                price.reset_index(drop=True,inplace=True)
                all_prices.append(price)

            duration_curve = pd.concat(all_prices, axis=1)
            duration_curve.columns = duration_curve.columns.str.replace('_',' ')

            Data_Out = duration_curve.copy()

            xdimension=len(self.xlabels)
            if xdimension == 0:
                xdimension = 1
            ydimension=len(self.ylabels)
            if ydimension == 0:
                ydimension = 1

            # If the plot is not a facet plot, grid size should be 1x1
            if not self.facet:
                xdimension = 1
                ydimension = 1

            color_dict = dict(zip(duration_curve.columns,self.color_list))

            #setup plot
            fig1, axs = self._setup_plot(xdimension,ydimension)

            n=0
            for column in duration_curve:
                self._create_plot(axs,n,duration_curve,column,color_dict)
                axs[n].set_xlim(0,len(duration_curve))
                axs[n].legend(loc='best', facecolor='inherit', frameon=True)
                if self.facet:
                    n+=1

            fig1.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            plt.ylabel(self.AGG_BY + ' Price ($/MWh)',  color='black', rotation='vertical', labelpad=20)
            plt.xlabel('Intervals',  color='black', rotation='horizontal', labelpad=20)

            outputs[zone_input] = {'fig': fig1, 'data_table':Data_Out}
        return outputs


    def pdc_all_regions(self):

        """
        This method creates a price duration curve for all regions/zones and plots them on
        a single facet plot.
        The code automatically creates a facet plot based on the number of regions/zones in the input.
        All scenarios are plotted on a single facet for each region/zone
        """

        # Duration curve of individual region prices
        # Create Dictionary to hold Datframes for each scenario
        Price_Collection = {}
        self._getdata(Price_Collection)

        #Location to save to
        save_figures = os.path.join(self.figure_folder, self.AGG_BY + '_prices')

        outputs = {}
        n=0
        region_number = len(self.Zones)
        # determine x,y length for plot
        xdimension, ydimension =  self._set_x_y_dimension(region_number)

        #setup plot
        fig2, axs = self._setup_plot(xdimension,ydimension)

        data_table = []
        for zone_input in self.Zones:
            outputs[zone_input] = pd.DataFrame()
            all_prices=[]
            for scenario in self.Multi_Scenario:
                price = self._process_data(Price_Collection,scenario,zone_input)
                price.sort_values(by=scenario,ascending=False,inplace=True)
                price.reset_index(drop=True,inplace=True)
                all_prices.append(price)

            duration_curve = pd.concat(all_prices, axis=1)
            duration_curve.columns = duration_curve.columns.str.replace('_',' ')

            data_out = duration_curve.copy()
            data_out.columns = [zone_input + "_" + str(col) for col in data_out.columns]
            data_table.append(data_out)

            color_dict = dict(zip(duration_curve.columns,self.color_list))

            for column in duration_curve:
                self._create_plot(axs,n,duration_curve,column,color_dict)
                axs[n].set_xlim(0,len(duration_curve))
                axs[n].set_ylabel(zone_input.replace('_',' '), color='black', rotation='vertical')

                handles, labels = axs[region_number-1].get_legend_handles_labels()
                #Legend
                axs[region_number-1].legend((handles), (labels), loc='lower left',bbox_to_anchor=(1,0),
                              facecolor='inherit', frameon=True)
            n+=1

        fig2.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.ylabel(self.AGG_BY + ' Price ($/MWh)',  color='black', rotation='vertical', labelpad=30)
        plt.xlabel('Intervals',  color='black', rotation='horizontal', labelpad=20)

        Data_Table_Out = pd.concat(data_table, axis=1)
        fig2.savefig(os.path.join(save_figures, "Price_Duration_Curve_All_Regions.svg"), dpi=600, bbox_inches='tight')
        Data_Table_Out.to_csv(os.path.join(save_figures, "Price_Duration_Curve_All_Regions.csv"))
        return outputs


    def region_timeseries_price(self):

        """
        This method creates price timeseries plot for each region.
        The code will create either a facet plot or a single plot depening on if the Facet argument is active.
        If a facet plot is created, each scenario is plotted on a seperate facet, otherwise all scenarios are
        plotted on a single plot.
        """

        # Timeseries of individual region prices
        # Create Dictionary to hold Datframes for each scenario
        Price_Collection = {}
        self._getdata(Price_Collection)

        outputs = {}
        for zone_input in self.Zones:
            print(self.AGG_BY + " = " + zone_input)

            all_prices=[]
            for scenario in self.Multi_Scenario:
                price = self._process_data(Price_Collection,scenario,zone_input)
                price = price.groupby(["timestamp"]).sum()
                all_prices.append(price)

            timeseries = pd.concat(all_prices, axis=1)
            timeseries.columns = timeseries.columns.str.replace('_',' ')

            Data_Out = timeseries.copy()

            xdimension=len(self.xlabels)
            if xdimension == 0:
                xdimension = 1
            ydimension=len(self.ylabels)
            if ydimension == 0:
                ydimension = 1

            # If the plot is not a facet plot, grid size should be 1x1
            if not self.facet:
                xdimension = 1
                ydimension = 1

            color_dict = dict(zip(timeseries.columns,self.color_list))

            #setup plot
            fig3, axs = self._setup_plot(xdimension,ydimension)



            n=0 #Counter for scenario subplots
            for column in timeseries:
                self._create_plot(axs,n,timeseries,column,color_dict)
                axs[n].legend(loc='best',facecolor='inherit', frameon=True)
                self._set_plot_timeseries_format(axs,n)
                if self.facet:
                    n+=1

            fig3.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            plt.ylabel(self.AGG_BY + ' Price ($/MWh)',  color='black', rotation='vertical', labelpad=20)
            plt.xlabel(self.timezone,  color='black', rotation='horizontal', labelpad=20)

            outputs[zone_input] = {'fig': fig3, 'data_table':Data_Out}
        return outputs

    def timeseries_price_all_regions(self):

        """
        This method creates a price timeseries plot for all regions/zones and plots them on
        a single facet plot.
        The code automatically creates a facet plot based on the number of regions/zones in the input.
        All scenarios are plotted on a single facet for each region/zone
        """

        # Create Dictionary to hold Datframes for each scenario
        Price_Collection = {}
        self._getdata(Price_Collection)

        #Location to save to
        save_figures = os.path.join(self.figure_folder, self.AGG_BY + '_prices')

        outputs = {}
        n=0

        region_number = len(self.Zones)
        xdimension, ydimension =  self._set_x_y_dimension(region_number)

        #setup plot
        fig4, axs = self._setup_plot(xdimension,ydimension)

        data_table = []
        for zone_input in self.Zones:
            # print(zone_input)
            outputs[zone_input] = pd.DataFrame()
            all_prices=[]
            for scenario in self.Multi_Scenario:
                price = self._process_data(Price_Collection,scenario,zone_input)
                price = price.groupby(["timestamp"]).sum()
                all_prices.append(price)

            timeseries = pd.concat(all_prices, axis=1)
            timeseries.columns = timeseries.columns.str.replace('_',' ')

            data_out = timeseries.copy()
            data_out.columns = [zone_input + "_" + str(col) for col in data_out.columns]
            data_table.append(data_out)

            color_dict = dict(zip(timeseries.columns,self.color_list))

            for column in timeseries:
                axs[n].plot(timeseries[column], linewidth=1, color=color_dict[column],label=column)
                # self._create_plot(axs,n,timeseries,column,color_dict)
                axs[n].set_ylabel(zone_input.replace('_',' '), color='black', rotation='vertical')
                self._set_plot_timeseries_format(axs,n)

                handles, labels = axs[region_number-1].get_legend_handles_labels()
                #Legend
                axs[region_number-1].legend((handles), (labels), loc='lower left',bbox_to_anchor=(1,0),
                              facecolor='inherit', frameon=True)
            n+=1

        fig4.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.ylabel(self.AGG_BY + ' Price ($/MWh)',  color='black', rotation='vertical', labelpad=30)
        plt.xlabel(self.timezone,  color='black', rotation='horizontal', labelpad=20)

        Data_Table_Out = pd.concat(data_table, axis=1)
        fig4.savefig(os.path.join(save_figures, "Price_Timeseries_All_Regions.svg"), dpi=600, bbox_inches='tight')
        Data_Table_Out.to_csv(os.path.join(save_figures, "Price_Timeseries_All_Regions.csv"))

        return outputs

    # Internal methods to process data, not designed to be accessed from outside the mplot class.
    def _getdata(self,data_collection):
         for scenario in self.Multi_Scenario:
                if self.AGG_BY == "zone":
                    data_collection[scenario] = pd.read_hdf(os.path.join(self.Marmot_Solutions_folder, scenario,"Processed_HDF5_folder", scenario+ "_formatted.h5"),"zone_Price")
                else:
                    data_collection[scenario] = pd.read_hdf(os.path.join(self.Marmot_Solutions_folder, scenario,"Processed_HDF5_folder", scenario+ "_formatted.h5"),"region_Price")

    def _process_data(self,data_collection,scenario,zone_input):
        df = data_collection.get(scenario)
        df = df.xs(zone_input,level=self.AGG_BY)
        df = df.rename(columns={0:scenario})
        return df

    def _setup_plot(self,xdimension,ydimension):
        fig, axs = plt.subplots(ydimension,xdimension, figsize=((6*xdimension),(4*ydimension)), sharey=True, squeeze=False)
        plt.subplots_adjust(wspace=0.05, hspace=0.2)
        axs = axs.ravel()
        return fig,axs

    def _create_plot(self,axs,n,data,column,color_dict):
        axs[n].plot(data[column], linewidth=1, color=color_dict[column],label=column)
        axs[n].spines['right'].set_visible(False)
        axs[n].spines['top'].set_visible(False)
        # This checks for a nan in string. If no limit selected, do nothing
        if (self.prop!=self.prop)==False:
            axs[n].set_ylim(bottom=0,top=int(self.prop))

    def _set_x_y_dimension(self,region_number):
        if region_number >= 5:
            xdimension = 3
            ydimension = math.ceil(region_number/3)
        if region_number <= 3:
            xdimension = region_number
            ydimension = 1
        if region_number == 4:
            xdimension = 2
            ydimension = 2
        return xdimension,ydimension

    def _set_plot_timeseries_format(self,axs,n):
        locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
        formatter = mdates.ConciseDateFormatter(locator)
        formatter.formats[2] = '%d\n %b'
        formatter.zero_formats[1] = '%b\n %Y'
        formatter.zero_formats[2] = '%d\n %b'
        formatter.zero_formats[3] = '%H:%M\n %d-%b'
        formatter.offset_formats[3] = '%b %Y'
        formatter.show_offset = False
        axs[n].xaxis.set_major_locator(locator)
        axs[n].xaxis.set_major_formatter(formatter)
