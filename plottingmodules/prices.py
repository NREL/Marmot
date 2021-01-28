# -*- coding: utf-8 -*-
"""

price analysis plots, price duration cureves = timeseries plots

@author: adyreson and Daniel Levie
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import math
import logging
import plottingmodules.marmot_plot_functions as mfunc

#===============================================================================

class mplot(object):
    def __init__(self, argument_dict):
        # iterate over items in argument_dict and set as properties of class
        # see key_list in Marmot_plot_main for list of properties
        for prop in argument_dict:
            self.__setattr__(prop, argument_dict[prop])
        self.logger = logging.getLogger('marmot_plot.'+__name__)
        
    def region_pdc(self):

        """
        This method creates a price duration curve for each region.
        The code will create either a facet plot or a single plot depening on if the Facet argument is active.
        If a facet plot is created, each scenario is plotted on a seperate facet, otherwise all scenarios are
        plotted on a single plot.
        """
        outputs = {}
        price_collection = {}
        
        check_input_data = self._getdata(price_collection)
        
        if 1 in check_input_data:
            outputs = mfunc.MissingInputData()
            return outputs
        
        for zone_input in self.Zones:
            self.logger.info(self.AGG_BY + " = " + zone_input)

            all_prices=[]
            for scenario in self.Multi_Scenario:

                price = self._process_data(price_collection,scenario,zone_input)
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
            fig1, axs = mfunc.setup_plot(xdimension,ydimension)
            plt.subplots_adjust(wspace=0.05, hspace=0.2)

            n=0
            for column in duration_curve:
                mfunc.create_line_plot(axs,duration_curve,column,color_dict,n=n,label=column)
                if (self.prop!=self.prop)==False:
                    axs[n].set_ylim(bottom=0,top=int(self.prop))
                axs[n].set_xlim(0,len(duration_curve))
                axs[n].legend(loc='lower left',bbox_to_anchor=(1,0),
                              facecolor='inherit', frameon=True)
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
        outputs = {}
        price_collection = {}
        
        check_input_data = self._getdata(price_collection)
        
        if 1 in check_input_data:
            outputs = mfunc.MissingInputData()
            return outputs

        #Location to save to
        save_figures = os.path.join(self.figure_folder, self.AGG_BY + '_prices')

        n=0
        region_number = len(self.Zones)
        # determine x,y length for plot
        xdimension, ydimension =  mfunc.set_x_y_dimension(region_number)

        #setup plot
        fig2, axs = mfunc.setup_plot(xdimension,ydimension)
        plt.subplots_adjust(wspace=0.1, hspace=0.2)
        
        data_table = []
        for zone_input in self.Zones:

            all_prices=[]
            for scenario in self.Multi_Scenario:
                price = self._process_data(price_collection,scenario,zone_input)
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
                mfunc.create_line_plot(axs,duration_curve,column,color_dict,n=n,label=column)
                if (self.prop!=self.prop)==False:
                    axs[n].set_ylim(bottom=0,top=int(self.prop))
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
        outputs = mfunc.DataSavedInModule()
        return outputs


    def region_timeseries_price(self):

        """
        This method creates price timeseries plot for each region.
        The code will create either a facet plot or a single plot depening on if the Facet argument is active.
        If a facet plot is created, each scenario is plotted on a seperate facet, otherwise all scenarios are
        plotted on a single plot.
        """
        outputs = {}
        price_collection = {}
        
        check_input_data = self._getdata(price_collection)
        
        if 1 in check_input_data:
            outputs = mfunc.MissingInputData()
            return outputs
        
        for zone_input in self.Zones:
            self.logger.info(self.AGG_BY + " = " + zone_input)

            all_prices=[]
            for scenario in self.Multi_Scenario:
                price = self._process_data(price_collection,scenario,zone_input)
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
            fig3, axs = mfunc.setup_plot(xdimension,ydimension)
            plt.subplots_adjust(wspace=0.05, hspace=0.2)

            n=0 #Counter for scenario subplots
            for column in timeseries:
                mfunc.create_line_plot(axs,timeseries,column,color_dict,n=n,label=column)
                if (self.prop!=self.prop)==False:
                    axs[n].set_ylim(bottom=0,top=int(self.prop))
                axs[n].legend(loc='lower left',bbox_to_anchor=(1,0),
                              facecolor='inherit', frameon=True)
                
                mfunc.set_plot_timeseries_format(axs,n)
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
        outputs = {}
        price_collection = {}
        
        check_input_data = self._getdata(price_collection)
        
        if 1 in check_input_data:
            outputs = mfunc.MissingInputData()
            return outputs

        #Location to save to
        save_figures = os.path.join(self.figure_folder, self.AGG_BY + '_prices')

        outputs = {}
        n=0

        region_number = len(self.Zones)
        xdimension, ydimension =  mfunc.set_x_y_dimension(region_number)

        #setup plot
        fig4, axs = mfunc.setup_plot(xdimension,ydimension)
        plt.subplots_adjust(wspace=0.1, hspace=0.2)

        data_table = []
        for zone_input in self.Zones:
            self.logger.info(self.AGG_BY + " = " + zone_input)

            all_prices=[]
            for scenario in self.Multi_Scenario:
                price = self._process_data(price_collection,scenario,zone_input)
                price = price.groupby(["timestamp"]).sum()
                all_prices.append(price)

            timeseries = pd.concat(all_prices, axis=1)
            timeseries.columns = timeseries.columns.str.replace('_',' ')

            data_out = timeseries.copy()
            data_out.columns = [zone_input + "_" + str(col) for col in data_out.columns]
            data_table.append(data_out)

            color_dict = dict(zip(timeseries.columns,self.color_list))

            for column in timeseries:
                mfunc.create_line_plot(axs,timeseries,column,color_dict,n=n,label=column)
                axs[n].set_ylabel(zone_input.replace('_',' '), color='black', rotation='vertical')
                if (self.prop!=self.prop)==False:
                    axs[n].set_ylim(bottom=0,top=int(self.prop))
                mfunc.set_plot_timeseries_format(axs,n)

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
        outputs = mfunc.DataSavedInModule()
        return outputs

    # Internal methods to process data, not designed to be accessed from outside the mplot class.
    def _getdata(self,data_collection):
        check_input_data = []
        if self.AGG_BY == "zone":
            check_input_data.extend([mfunc.get_data(data_collection,"zone_Price",self.Marmot_Solutions_folder, self.Multi_Scenario)])
        else:
            check_input_data.extend([mfunc.get_data(data_collection,"region_Price",self.Marmot_Solutions_folder, self.Multi_Scenario)])
        return check_input_data
    
    def _process_data(self,data_collection,scenario,zone_input):
        df = data_collection.get(scenario)
        df = df.xs(zone_input,level=self.AGG_BY)
        df = df.rename(columns={0:scenario})
        return df

