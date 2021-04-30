# -*- coding: utf-8 -*-
"""

price analysis plots, price duration curves = timeseries plots

@author: adyreson and Daniel Levie
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import logging
import marmot.plottingmodules.marmot_plot_functions as mfunc
import marmot.config.mconfig as mconfig


#===============================================================================

class mplot(object):
    def __init__(self, argument_dict):
        # iterate over items in argument_dict and set as properties of class
        # see key_list in Marmot_plot_main for list of properties
        for prop in argument_dict:
            self.__setattr__(prop, argument_dict[prop])
        self.logger = logging.getLogger('marmot_plot.'+__name__)

        self.mplot_data_dict = {}

    def pdc_all_regions(self, figure_name=None, prop=None, start=None, end=None, 
                  timezone=None, start_date_range=None, end_date_range=None):
        """
        This method creates a price duration curve for all regions/zones and plots them on
        a single facet plot.
        The code automatically creates a facet plot based on the number of regions/zones in the input.
        All scenarios are plotted on a single facet for each region/zone
        """
            
        if self.AGG_BY == 'zone':
            agg = 'zone'
        else:
            agg = 'region'
            
        outputs = {}
        
        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True, f"{agg}_Price", self.Scenarios)]
        
        # Runs get_data to populate mplot_data_dict with all required properties, returns a 1 if required data is missing
        check_input_data = mfunc.get_data(self.mplot_data_dict, properties,self.Marmot_Solutions_folder)

        if 1 in check_input_data:
            return mfunc.MissingInputData()

        #Location to save to
        save_figures = os.path.join(self.figure_folder, self.AGG_BY + '_prices')

        region_number = len(self.Zones)
        # determine x,y length for plot
        xdimension, ydimension =  mfunc.set_x_y_dimension(region_number)

        #setup plot
        fig2, axs = mfunc.setup_plot(xdimension,ydimension)
        plt.subplots_adjust(wspace=0.1, hspace=0.3)
        
        data_table = []
        for n, zone_input in enumerate(self.Zones):

            all_prices=[]
            for scenario in self.Scenarios:
                price = self._process_data(self.mplot_data_dict[f"{agg}_Price"],scenario,zone_input)
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
                if (prop!=prop)==False:
                    axs[n].set_ylim(bottom=0,top=int(prop))
                axs[n].set_xlim(0,len(duration_curve))
                axs[n].set_title(zone_input.replace('_',' '))

                handles, labels = axs[region_number-1].get_legend_handles_labels()
                #Legend
                axs[region_number-1].legend((handles), (labels), loc='lower left',bbox_to_anchor=(1,0),
                              facecolor='inherit', frameon=True)

        fig2.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.ylabel(self.AGG_BY + ' Price ($/MWh)',  color='black', rotation='vertical', labelpad=30)
        plt.xlabel('Intervals',  color='black', rotation='horizontal', labelpad=20)

        Data_Table_Out = pd.concat(data_table, axis=1)

        Data_Table_Out = Data_Table_Out.add_suffix(" ($/MWh)")

        fig2.savefig(os.path.join(save_figures, "Price_Duration_Curve_All_Regions.svg"), dpi=600, bbox_inches='tight')
        Data_Table_Out.to_csv(os.path.join(save_figures, "Price_Duration_Curve_All_Regions.csv"))
        outputs = mfunc.DataSavedInModule()
        return outputs
    
    def region_pdc(self, figure_name=None, prop=None, start=None, end=None, 
                  timezone=None, start_date_range=None, end_date_range=None):

        """
        This method creates a price duration curve for each region.
        The code will create either a facet plot or a single plot depening on if the Facet argument is active.
        If a facet plot is created, each scenario is plotted on a seperate facet, otherwise all scenarios are
        plotted on a single plot.
        """
        outputs = {}
        
        facet=False
        if 'Facet' in figure_name:
            facet = True
            
        if self.AGG_BY == 'zone':
            agg = 'zone'
        else:
            agg = 'region'
            
        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True, f"{agg}_Price", self.Scenarios)]
        
        # Runs get_data to populate mplot_data_dict with all required properties, returns a 1 if required data is missing
        check_input_data = mfunc.get_data(self.mplot_data_dict, properties,self.Marmot_Solutions_folder)

        if 1 in check_input_data:
            return mfunc.MissingInputData()
        
        for zone_input in self.Zones:
            self.logger.info(f"{self.AGG_BY} = {zone_input}")

            all_prices=[]
            for scenario in self.Scenarios:

                price = self._process_data(self.mplot_data_dict[f"{agg}_Price"],scenario,zone_input)
                price.sort_values(by=scenario,ascending=False,inplace=True)
                price.reset_index(drop=True,inplace=True)
                all_prices.append(price)

            duration_curve = pd.concat(all_prices, axis=1)
            duration_curve.columns = duration_curve.columns.str.replace('_',' ')

            Data_Out = duration_curve.add_suffix(" ($/MWh)")

            xdimension=len(self.xlabels)
            if xdimension == 0:
                xdimension = 1
            ydimension=len(self.ylabels)
            if ydimension == 0:
                ydimension = 1

            # If the plot is not a facet plot, grid size should be 1x1
            if not facet:
                xdimension = 1
                ydimension = 1

            color_dict = dict(zip(duration_curve.columns,self.color_list))

            #setup plot
            fig1, axs = mfunc.setup_plot(xdimension,ydimension)
            plt.subplots_adjust(wspace=0.05, hspace=0.2)

            n=0
            for column in duration_curve:
                mfunc.create_line_plot(axs,duration_curve,column,color_dict,n=n,label=column)
                if (prop!=prop)==False:
                    axs[n].set_ylim(bottom=0,top=int(prop))
                axs[n].set_xlim(0,len(duration_curve))
                axs[n].legend(loc='lower left',bbox_to_anchor=(1,0),
                              facecolor='inherit', frameon=True)
                if facet:
                    n+=1

            fig1.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            plt.ylabel(f"{self.AGG_BY} Price ($/MWh)",  color='black', rotation='vertical', labelpad=20)
            plt.xlabel('Intervals',  color='black', rotation='horizontal', labelpad=20)
            if mconfig.parser("plot_title_as_region"):
                plt.title(zone_input)
            outputs[zone_input] = {'fig': fig1, 'data_table':Data_Out}
        return outputs


    def region_timeseries_price(self, figure_name=None, prop=None, start=None, end=None, 
                  timezone=None, start_date_range=None, end_date_range=None):

        """
        This method creates price timeseries plot for each region.
        The code will create either a facet plot or a single plot depening on if the Facet argument is active.
        If a facet plot is created, each scenario is plotted on a seperate facet, otherwise all scenarios are
        plotted on a single plot.
        """
        outputs = {}
        
        facet=False
        if 'Facet' in figure_name:
            facet = True
            
        if self.AGG_BY == 'zone':
            agg = 'zone'
        else:
            agg = 'region'
            
        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True, f"{agg}_Price", self.Scenarios)]
        
        # Runs get_data to populate mplot_data_dict with all required properties, returns a 1 if required data is missing
        check_input_data = mfunc.get_data(self.mplot_data_dict, properties,self.Marmot_Solutions_folder)

        if 1 in check_input_data:
            return mfunc.MissingInputData()
        
        for zone_input in self.Zones:
            self.logger.info(f"{self.AGG_BY} = {zone_input}")

            all_prices=[]
            for scenario in self.Scenarios:
                price = self._process_data(self.mplot_data_dict[f"{agg}_Price"],scenario,zone_input)
                price = price.groupby(["timestamp"]).sum()
                all_prices.append(price)

            timeseries = pd.concat(all_prices, axis=1)
            timeseries.columns = timeseries.columns.str.replace('_',' ')

            Data_Out = timeseries.add_suffix(" ($/MWh)")

            xdimension=len(self.xlabels)
            if xdimension == 0:
                xdimension = 1
            ydimension=len(self.ylabels)
            if ydimension == 0:
                ydimension = 1

            # If the plot is not a facet plot, grid size should be 1x1
            if not facet:
                xdimension = 1
                ydimension = 1

            color_dict = dict(zip(timeseries.columns,self.color_list))

            #setup plot
            fig3, axs = mfunc.setup_plot(xdimension,ydimension)
            plt.subplots_adjust(wspace=0.05, hspace=0.2)

            n=0 #Counter for scenario subplots
            for column in timeseries:
                mfunc.create_line_plot(axs,timeseries,column,color_dict,n=n,label=column)
                if (prop!=prop)==False:
                    axs[n].set_ylim(bottom=0,top=int(prop))
                axs[n].legend(loc='lower left',bbox_to_anchor=(1,0),
                              facecolor='inherit', frameon=True)
                
                mfunc.set_plot_timeseries_format(axs,n)
                if facet:
                    n+=1

            fig3.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            if mconfig.parser("plot_title_as_region"):
                plt.title(zone_input)
            plt.ylabel(f"{self.AGG_BY} Price ($/MWh)",  color='black', rotation='vertical', labelpad=20)
            plt.xlabel(timezone,  color='black', rotation='horizontal', labelpad=20)

            outputs[zone_input] = {'fig': fig3, 'data_table':Data_Out}
        return outputs

    def timeseries_price_all_regions(self, figure_name=None, prop=None, start=None, end=None, 
                  timezone=None, start_date_range=None, end_date_range=None):

        """
        This method creates a price timeseries plot for all regions/zones and plots them on
        a single facet plot.
        The code automatically creates a facet plot based on the number of regions/zones in the input.
        All scenarios are plotted on a single facet for each region/zone
        """
        outputs = {}
        
        if self.AGG_BY == 'zone':
            agg = 'zone'
        else:
            agg = 'region'
            
        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True, f"{agg}_Price", self.Scenarios)]
        
        # Runs get_data to populate mplot_data_dict with all required properties, returns a 1 if required data is missing
        check_input_data = mfunc.get_data(self.mplot_data_dict, properties,self.Marmot_Solutions_folder)

        if 1 in check_input_data:
            return mfunc.MissingInputData()

        #Location to save to
        save_figures = os.path.join(self.figure_folder, self.AGG_BY + '_prices')

        outputs = {}

        region_number = len(self.Zones)
        xdimension, ydimension =  mfunc.set_x_y_dimension(region_number)

        #setup plot
        fig4, axs = mfunc.setup_plot(xdimension,ydimension)
        plt.subplots_adjust(wspace=0.1, hspace=0.3)

        data_table = []
        for n, zone_input in enumerate(self.Zones):
            self.logger.info(f"{self.AGG_BY} = {zone_input}")

            all_prices=[]
            for scenario in self.Scenarios:
                price = self._process_data(self.mplot_data_dict[f"{agg}_Price"],scenario,zone_input)
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
                axs[n].set_title(zone_input.replace('_',' '))
                if (prop!=prop)==False:
                    axs[n].set_ylim(bottom=0,top=int(prop))
                mfunc.set_plot_timeseries_format(axs,n)

                handles, labels = axs[region_number-1].get_legend_handles_labels()
                #Legend
                axs[region_number-1].legend((handles), (labels), loc='lower left',bbox_to_anchor=(1,0),
                              facecolor='inherit', frameon=True)

        fig4.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.ylabel(f"{self.AGG_BY} Price ($/MWh)",  color='black', rotation='vertical', labelpad=30)
        plt.xlabel(timezone,  color='black', rotation='horizontal', labelpad=20)

        Data_Table_Out = pd.concat(data_table, axis=1)

        Data_Table_Out = Data_Table_Out.add_suffix(" ($/MWh)")

        fig4.savefig(os.path.join(save_figures, "Price_Timeseries_All_Regions.svg"), dpi=600, bbox_inches='tight')
        Data_Table_Out.to_csv(os.path.join(save_figures, "Price_Timeseries_All_Regions.csv"))
        outputs = mfunc.DataSavedInModule()
        return outputs

    # Internal methods to process data, not designed to be accessed from outside the mplot class.
    def _getdata(self,data_collection):
        check_input_data = []
        if self.AGG_BY == "zone":
            check_input_data.extend([mfunc.get_data(data_collection,"zone_Price",self.Marmot_Solutions_folder, self.Scenarios)])
        else:
            check_input_data.extend([mfunc.get_data(data_collection,"region_Price",self.Marmot_Solutions_folder, self.Scenarios)])
        return check_input_data
    
    def _process_data(self,data_collection,scenario,zone_input):
        df = data_collection.get(scenario)
        df = df.xs(zone_input,level=self.AGG_BY)
        df = df.rename(columns={0:scenario})
        return df

