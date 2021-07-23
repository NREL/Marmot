# -*- coding: utf-8 -*-
"""

price analysis plots, price duration curves = timeseries plots

@author: adyreson and Daniel Levie
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import marmot.plottingmodules.marmot_plot_functions as mfunc
import marmot.config.mconfig as mconfig
import math


#===============================================================================

class MPlot(object):
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
        
        grid_size = xdimension*ydimension
        # Used to calculate any excess axis to delete
        excess_axs = grid_size - region_number

        #setup plot
        fig2, axs = mfunc.setup_plot(xdimension,ydimension)
        plt.subplots_adjust(wspace=0.1, hspace=0.50)
        
        data_table = []
        for n, zone_input in enumerate(self.Zones):

            all_prices=[]
            for scenario in self.Scenarios:
                price = self._process_data(self.mplot_data_dict[f"{agg}_Price"],scenario,zone_input)
                price = price.groupby(["timestamp"]).sum()
                if not pd.isnull(start_date_range):
                    self.logger.info(f"Plotting specific date range: \
                                      {str(start_date_range)} to {str(end_date_range)}")
                    price = price[start_date_range:end_date_range]
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

                handles, labels = axs[n].get_legend_handles_labels()
                #Legend
                axs[grid_size-1].legend((handles), (labels), loc='lower left',bbox_to_anchor=(1,0),
                              facecolor='inherit', frameon=True)
        
        # Remove extra axes
        if excess_axs != 0:
            mfunc.remove_excess_axs(axs,excess_axs,grid_size)
        
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
                price = price.groupby(["timestamp"]).sum()
                if not pd.isnull(start_date_range):
                    self.logger.info(f"Plotting specific date range: \
                                      {str(start_date_range)} to {str(end_date_range)}")
                    price = price[start_date_range:end_date_range]
                
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
                
                if not pd.isnull(start_date_range):
                    self.logger.info(f"Plotting specific date range: \
                                      {str(start_date_range)} to {str(end_date_range)}")
                    price = price[start_date_range:end_date_range]
                
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
            if not math.isnan(timezone):
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
        
        grid_size = xdimension*ydimension
        # Used to calculate any excess axis to delete
        excess_axs = grid_size - region_number

        #setup plot
        fig4, axs = mfunc.setup_plot(xdimension,ydimension)
        plt.subplots_adjust(wspace=0.1, hspace=0.70)

        data_table = []
        for n, zone_input in enumerate(self.Zones):
            self.logger.info(f"{self.AGG_BY} = {zone_input}")

            all_prices=[]
            for scenario in self.Scenarios:
                price = self._process_data(self.mplot_data_dict[f"{agg}_Price"],scenario,zone_input)
                price = price.groupby(["timestamp"]).sum()
                
                if not pd.isnull(start_date_range):
                    self.logger.info(f"Plotting specific date range: \
                                      {str(start_date_range)} to {str(end_date_range)}")
                    price = price[start_date_range:end_date_range]
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

                handles, labels = axs[n].get_legend_handles_labels()
                #Legend
                axs[grid_size-1].legend((handles), (labels), loc='lower left',bbox_to_anchor=(1,0),
                              facecolor='inherit', frameon=True)
        
        # Remove extra axes
        if excess_axs != 0:
            mfunc.remove_excess_axs(axs,excess_axs,grid_size)
        
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
    
    
    
    def node_pdc(self, **kwargs):
        """
        This method creates a price duration curve for a set of specifc nodes.
        The code will create either a facet plot or a single plot depening on 
        the number of nodes included in plot_select.csv
        """    
        
        outputs = self._node_price(PDC=True, **kwargs)
        return outputs
    
        
    
    def node_timeseries_price(self, **kwargs):
        """
        This method creates a price timeseries plot for a set of specifc nodes.
        
        The code will create either a facet plot or a single plot depening on 
        the number of nodes included in plot_select.csv
        """    
        
        outputs = self._node_price(**kwargs)
        return outputs
        
        
        
    def _node_price(self, PDC=None, figure_name=None, prop=None, start=None, 
                    end=None, timezone=None, start_date_range=None, 
                    end_date_range=None):

        """
        This method creates a price duration curve or timeseries plot for a 
        specifc node. 
        If PDC == True, a price duration curve plot will be created
        The code will create either a facet plot or a single plot depening on 
        the number of nodes included in plot_select.csv
        Plots and Data are saved within the module
        """    
            
            
        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True, "node_Price", self.Scenarios)]
        
        # Runs get_data to populate mplot_data_dict with all required properties, returns a 1 if required data is missing
        check_input_data = mfunc.get_data(self.mplot_data_dict, properties, self.Marmot_Solutions_folder)

        if 1 in check_input_data:
            return mfunc.MissingInputData()
        
        node_figure_folder = os.path.join(self.figure_folder, 'node_prices')
        try:
            os.makedirs(node_figure_folder)
        except FileExistsError:
            # directory already exists
            pass

        #Select only node specified in Marmot_plot_select.csv.
        select_nodes = prop.split(",")
        if select_nodes == None:
            return mfunc.InputSheetError()
        
        self.logger.info(f'Plotting Prices for {select_nodes}')
        
        all_prices=[]
        for scenario in self.Scenarios:
            self.logger.info(f"Scenario = {scenario}")

            price = self.mplot_data_dict["node_Price"][scenario]
            price = price.loc[(slice(None), select_nodes),:]
            price = price.groupby(["timestamp","node"]).sum()
            price.rename(columns={0:scenario}, inplace=True)
            
            if not pd.isnull(start_date_range):
                self.logger.info(f"Plotting specific date range: \
                                  {str(start_date_range)} to {str(end_date_range)}")
                price = price[pd.to_datetime(start_date_range):pd.to_datetime(end_date_range)]
                
            if PDC:
                price.sort_values(by=['node',scenario], ascending=False,
                                  inplace=True)
                price.reset_index('timestamp', drop=True, inplace=True)                
            all_prices.append(price)
    
        pdc = pd.concat(all_prices,axis=1)
        pdc.columns = pdc.columns.str.replace('_',' ')

        Data_Out = pdc.add_suffix(" ($/MWh)")
        
        xdimension, ydimension =  mfunc.set_x_y_dimension(len(select_nodes))
        
        #setup plot
        fig, axs = mfunc.setup_plot(xdimension,ydimension)
        plt.subplots_adjust(wspace=0.1, hspace=0.70)
        
        color_dict = dict(zip(pdc.columns, self.color_list))

        for n, node in enumerate(select_nodes):
            
            if PDC:
                node_pdc = pdc.xs(node)
                node_pdc.reset_index(drop=True, inplace=True)
            else:
                node_pdc = pdc.xs(node, level='node')
            
            for column in node_pdc:
                mfunc.create_line_plot(axs,node_pdc, column, color_dict, 
                                       n=n, label=column)
                # if (prop!=prop)==False:
                axs[n].set_ylim(bottom=0,top=int(200))
                
                if not PDC:
                    mfunc.set_plot_timeseries_format(axs,n)
                # axs[n].set_xlim(0,len(node_pdc))
                
                handles, labels = axs[n].get_legend_handles_labels()
                #Legend
                axs[len(select_nodes)-1].legend((handles), (labels),
                                                loc='lower left', 
                                                bbox_to_anchor=(1,0),
                                                facecolor='inherit', 
                                                frameon=True)
                axs[n].set_title(node)
            
        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', top=False, bottom=False, 
                        left=False, right=False)
        plt.ylabel('Node Price ($/MWh)',  color='black', rotation='vertical', 
                   labelpad=30)
        if PDC:
            plt.xlabel('Intervals',  color='black', rotation='horizontal', 
                       labelpad=20)
        else:
            plt.xlabel(timezone,  color='black', rotation='horizontal', 
                       labelpad=20)

        fig.savefig(os.path.join(node_figure_folder, figure_name + ".svg"), 
                    dpi=600, bbox_inches='tight')
        Data_Out.to_csv(os.path.join(node_figure_folder, figure_name + ".csv"))
        outputs = mfunc.DataSavedInModule()
        return outputs
    
    
    def node_price_hist(self, **kwargs):
        """
        This method creates a price histogram for a specifc nodes.
        
        A facet plot will be created if more than one scenario are included on the 
        user input sheet
        Each scenario will be plotted on a seperate subplot.
        If a set of nodes are passed at input, each will be saved to a seperate 
        figure with node name as a suffix. 
        Plots and Data are saved within the module
        """    
        
        outputs = self._node_hist(**kwargs)
        return outputs
        
        
    def node_price_hist_diff(self, **kwargs):
        """
        This method creates a difference price histogram for a specifc nodes.
        
        This plot requires more than one scenario to display correctly.
        A facet plot will be created
        Each scenario will be plotted on a seperate subplot, with values disaplying 
        the relative difference to the first scenario in the list.
        If a set of nodes are passed at input, each will be saved to a seperate 
        figure with node name as a suffix. 
        Plots and Data are saved within the module
        """    
        
        outputs = self._node_hist(diff_plot=True, **kwargs)
        return outputs
        
    
    def _node_hist(self, diff_plot=None, figure_name=None, prop=None, start=None, end=None, 
                  timezone=None, start_date_range=None, end_date_range=None):
        """
        Internal code for hist plots. 
        """
        
        
        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True, "node_Price", self.Scenarios)]
        
        # Runs get_data to populate mplot_data_dict with all required properties, returns a 1 if required data is missing
        check_input_data = mfunc.get_data(self.mplot_data_dict, properties, self.Marmot_Solutions_folder)

        if 1 in check_input_data:
            return mfunc.MissingInputData()
        
        node_figure_folder = os.path.join(self.figure_folder, 'node_prices')
        try:
            os.makedirs(node_figure_folder)
        except FileExistsError:
            # directory already exists
            pass
        
        #Select only node specified in Marmot_plot_select.csv.
        select_nodes = prop.split(",")
        if select_nodes == None:
            return mfunc.InputSheetError()
        
        for node in select_nodes:
            self.logger.info(f'Plotting Prices for Node: {node}')
        
            all_prices=[]
            for scenario in self.Scenarios:
                self.logger.info(f"Scenario = {scenario}")
    
                price = self.mplot_data_dict["node_Price"][scenario]
                price = price.xs(node, level='node')
                # price = price.loc[(slice(None), select_nodes),:]
                price = price.groupby(["timestamp"]).sum()
                price.rename(columns={0:scenario}, inplace=True)
                
                if not pd.isnull(start_date_range):
                    self.logger.info(f"Plotting specific date range: \
                                      {str(start_date_range)} to {str(end_date_range)}")
                    price = price[pd.to_datetime(start_date_range):pd.to_datetime(end_date_range)]
                
                price.reset_index('timestamp',drop=True,inplace=True)                
                all_prices.append(price)
    
            p_hist = pd.concat(all_prices,axis=1)
            
            if diff_plot:
                p_hist = p_hist.subtract(p_hist[f'{self.Scenarios[0]}'],axis=0)
            
            p_hist.columns = p_hist.columns.str.replace('_',' ')
            data_out = p_hist.add_suffix(" ($/MWh)")
            
            xdimension, ydimension =  mfunc.setup_facet_xy_dimensions(self.xlabels,
                                                                      self.ylabels,
                                                                      multi_scenario=self.Scenarios)
            grid_size = xdimension*ydimension
             # Used to calculate any excess axis to delete
            plot_number = len(self.Scenarios)
            excess_axs = grid_size - plot_number
        
            #setup plot
            fig, axs = mfunc.setup_plot(xdimension,ydimension, sharey=True)
            axs = axs.ravel()
            plt.subplots_adjust(wspace=0.1, hspace=0.25)
            
            color_dict = dict(zip(p_hist.columns, self.color_list))
            
            # max, min values in histogram range and bin width
            # TODO: Determine a way to pass the following as an input. 
            range_max = 100
            range_min = -100
            bin_width = 2.5
            
            # no of bines
            bins = int((range_max + abs(range_min)) / bin_width)
            
            for n, column in enumerate(p_hist):
                
                # Set plot data eqaul to 0 if all zero, e.g diff plot
                if sum(p_hist[column]) == 0:
                    data = 0
                else:
                    data = p_hist[column]
                # values above range_max and below range_min are binned together
                axs[n].hist(np.clip(data, range_min, range_max), 
                                               bins=bins, range=(range_min, 
                                                                 range_max),
                                               color=color_dict[column],
                                               zorder=2, rwidth=0.8)
                # get xlabels and edit them 
                xticks = axs[n].get_xticks()
                # min range_min, max range_max
                xticks = np.unique(np.clip(xticks, range_min, range_max))
                xlabels = xticks.astype(int).astype(str)
                # adds a '+' to final xlabel
                xlabels[-1] += '+'
                if min(xticks) < 0 :
                    xlabels[0] += '-'
                # sets x_tick spacing
                axs[n].set_xticks(xticks)
                axs[n].set_xticklabels(xlabels)
                axs[n].spines['right'].set_visible(False)
                axs[n].spines['top'].set_visible(False)
            
            # Remove extra axes
            if excess_axs != 0:
                mfunc.remove_excess_axs(axs,excess_axs,grid_size)
            
            mfunc.add_facet_labels(fig, self.xlabels, self.ylabels)
            fig.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', top=False, bottom=False, 
                            left=False, right=False)
            plt.ylabel('Occurrence',  color='black', rotation='vertical', 
                       labelpad=60, fontsize=24)
            plt.title(node)
            if diff_plot:
                plt.xlabel(f"Node LMP Change ($/MWh) relative to {self.Scenarios[0].replace('_',' ')}",
                           color='black', labelpad=40)
            else:
                plt.xlabel("Node LMP ($/MWh)",
                           color='black', labelpad=40)
            
            fig.savefig(os.path.join(node_figure_folder, 
                                     f"{figure_name}_{node}.svg"), dpi=600, 
                        bbox_inches='tight')
            data_out.to_csv(os.path.join(node_figure_folder, 
                                         f"{figure_name}_{node}.csv"))
            
        outputs = mfunc.DataSavedInModule()
        return outputs
    
    
    def _process_data(self,data_collection,scenario,zone_input):
        df = data_collection.get(scenario)
        df = df.xs(zone_input,level=self.AGG_BY)
        df = df.rename(columns={0:scenario})
        return df

