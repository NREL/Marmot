# -*- coding: utf-8 -*-
"""
price analysis plots, price duration curves and timeseries plots.
Prices plotted in $/MWh
@author: adyreson and Daniel Levie
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import marmot.config.mconfig as mconfig
import marmot.plottingmodules.plotutils.plot_library as plotlib
from marmot.plottingmodules.plotutils.plot_data_helper import PlotDataHelper
from marmot.plottingmodules.plotutils.plot_exceptions import (MissingInputData, DataSavedInModule,
           InputSheetError)


class MPlot(PlotDataHelper):
    """Marmot MPlot class, common across all plotting modules.

    All the plotting modules use this same class name.
    This class contains plotting methods that are grouped based on the
    current module name.
    
    The price.py module contains methods that are
    related to grid prices at regions, zones, nodes etc. 

    MPlot inherits from the PlotDataHelper class to assist in creating figures.
    """

    def __init__(self, argument_dict: dict):
        """MPlot init method

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
                    self.xlabels, self.gen_names_dict, Region_Mapping=self.Region_Mapping) 

        self.logger = logging.getLogger('marmot_plot.'+__name__)


    def pdc_all_regions(self, y_axis_max: float = None, 
                        start_date_range: str = None,
                        end_date_range: str = None, **_):
        """Creates a price duration curve for all regions/zones and plots them on a single facet plot.
        
        Price is in $/MWh.
        The code automatically creates a facet plot based on the number of regions/zones in the input.
        All scenarios are plotted on a single facet for each region/zone

        Args:
            y_axis_max (float, optional): Max y-axis value. 
                Defaults to None.
            start_date_range (str, optional): Defines a start date at which to represent data from. 
                Defaults to None.
            end_date_range (str, optional): Defines a end date at which to represent data to.
                Defaults to None.

        Returns:
            dict: dictionary containing the created plot and its data table
        """              
        if self.AGG_BY == 'zone':
            agg = 'zone'
        else:
            agg = 'region'
            
        outputs = {}
        
        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True, f"{agg}_Price", self.Scenarios)]
        
        # Runs get_formatted_data within PlotDataHelper to populate PlotDataHelper dictionary  
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)

        if 1 in check_input_data:
            return MissingInputData()

        #Location to save to
        save_figures = os.path.join(self.figure_folder, self.AGG_BY + '_prices')

        region_number = len(self.Zones)
        # determine x,y length for plot
        xdimension, ydimension =  self.set_x_y_dimension(region_number)
        
        grid_size = xdimension*ydimension
        # Used to calculate any excess axis to delete
        excess_axs = grid_size - region_number

        #setup plot
        fig2, axs = plotlib.setup_plot(xdimension,ydimension)
        plt.subplots_adjust(wspace=0.1, hspace=0.50)
        
        data_table = []
        for n, zone_input in enumerate(self.Zones):

            all_prices=[]
            for scenario in self.Scenarios:
                price = self._process_data(self[f"{agg}_Price"],scenario,zone_input)
                price = price.groupby(["timestamp"]).sum()
                if pd.notna(start_date_range):
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
                plotlib.create_line_plot(axs,duration_curve, column, color_dict,
                                         n=n, label=column)
                if pd.notna(y_axis_max):
                    axs[n].set_ylim(bottom=0, top=float(y_axis_max))
                axs[n].set_xlim(0,len(duration_curve))
                axs[n].set_title(zone_input.replace('_',' '))

                handles, labels = axs[n].get_legend_handles_labels()
                #Legend
                axs[grid_size-1].legend((handles), (labels), loc='lower left',
                                        bbox_to_anchor=(1,0), facecolor='inherit', 
                                        frameon=True)
        
        # Remove extra axes
        if excess_axs != 0:
            PlotDataHelper.remove_excess_axs(axs,excess_axs,grid_size)
        
        fig2.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, 
                        right=False)
        plt.ylabel(self.AGG_BY + ' Price ($/MWh)',  color='black', rotation='vertical', 
                   labelpad=30)
        plt.xlabel('Intervals',  color='black', rotation='horizontal', labelpad=20)

        Data_Table_Out = pd.concat(data_table, axis=1)

        Data_Table_Out = Data_Table_Out.add_suffix(" ($/MWh)")

        fig2.savefig(os.path.join(save_figures, "Price_Duration_Curve_All_Regions.svg"), 
                     dpi=600, bbox_inches='tight')
        Data_Table_Out.to_csv(os.path.join(save_figures, "Price_Duration_Curve_All_Regions.csv"))
        outputs = DataSavedInModule()
        return outputs
    
    def region_pdc(self, figure_name: str = None, y_axis_max: float = None,
                   start_date_range: str = None, 
                   end_date_range: str = None, **_):
        """Creates a price duration curve for each region. Price in $/MWh

        The code will create either a facet plot or a single plot depending on 
        if the Facet argument is active.
        If a facet plot is created, each scenario is plotted on a separate facet, 
        otherwise all scenarios are plotted on a single plot.
        To make a facet plot, ensure the work 'Facet' is found in the figure_name.

        Args:
            figure_name (str, optional): User defined figure output name.
                Defaults to None.
            y_axis_max (float, optional): Max y-axis value. 
                Defaults to None.
            start_date_range (str, optional): Defines a start date at which to represent data from. 
                Defaults to None.
            end_date_range (str, optional): Defines a end date at which to represent data to.
                Defaults to None.

        Returns:
            dict: dictionary containing the created plot and its data table
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
        
        # Runs get_formatted_data within PlotDataHelper to populate PlotDataHelper dictionary  
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)

        if 1 in check_input_data:
            return MissingInputData()
        
        for zone_input in self.Zones:
            self.logger.info(f"{self.AGG_BY} = {zone_input}")

            all_prices=[]
            for scenario in self.Scenarios:

                price = self._process_data(self[f"{agg}_Price"],scenario,zone_input)
                price = price.groupby(["timestamp"]).sum()
                if pd.notna(start_date_range):
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
            fig1, axs = plotlib.setup_plot(xdimension,ydimension)
            plt.subplots_adjust(wspace=0.05, hspace=0.2)

            n=0
            for column in duration_curve:
                plotlib.create_line_plot(axs, duration_curve, column, color_dict, 
                                         n=n, label=column)
                if pd.notna(y_axis_max):
                    axs[n].set_ylim(bottom=0,top=float(y_axis_max))
                axs[n].set_xlim(0,len(duration_curve))
                axs[n].legend(loc='lower left',bbox_to_anchor=(1,0),
                              facecolor='inherit', frameon=True)
                if facet:
                    n+=1

            fig1.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, 
                            right=False)
            plt.ylabel(f"{self.AGG_BY} Price ($/MWh)",  color='black', rotation='vertical', 
                       labelpad=20)
            plt.xlabel('Intervals',  color='black', rotation='horizontal', labelpad=20)
            if mconfig.parser("plot_title_as_region"):
                plt.title(zone_input)
            outputs[zone_input] = {'fig': fig1, 'data_table':Data_Out}
        return outputs


    def region_timeseries_price(self, figure_name: str = None, y_axis_max: float = None,
                                timezone: str = "", start_date_range: str = None, 
                                end_date_range: str = None, **_):
        """Creates price timeseries line plot for each region. Price is $/MWh.

        The code will create either a facet plot or a single plot depending on 
        if the Facet argument is active.
        If a facet plot is created, each scenario is plotted on a separate facet, 
        otherwise all scenarios are plotted on a single plot. 
        To make a facet plot, ensure the work 'Facet' is found in the figure_name.

        Args:
            figure_name (str, optional): User defined figure output name.
                Defaults to None.
            y_axis_max (float, optional): Max y-axis value. 
                Defaults to None.
            timezone (str, optional): The timezone to display on the x-axes.
                Defaults to "".
            start_date_range (str, optional): Defines a start date at which to represent data from. 
                Defaults to None.
            end_date_range (str, optional): Defines a end date at which to represent data to.
                Defaults to None.

        Returns:
            dict: dictionary containing the created plot and its data table
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
        
        # Runs get_formatted_data within PlotDataHelper to populate PlotDataHelper dictionary  
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)

        if 1 in check_input_data:
            return MissingInputData()
        
        for zone_input in self.Zones:
            self.logger.info(f"{self.AGG_BY} = {zone_input}")

            all_prices=[]
            for scenario in self.Scenarios:
                price = self._process_data(self[f"{agg}_Price"],scenario,zone_input)
                price = price.groupby(["timestamp"]).sum()
                
                if pd.notna(start_date_range):
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
            fig3, axs = plotlib.setup_plot(xdimension,ydimension)
            plt.subplots_adjust(wspace=0.05, hspace=0.2)

            n=0 #Counter for scenario subplots
            for column in timeseries:
                plotlib.create_line_plot(axs, timeseries, column, 
                                         color_dict, n=n, label=column)
                if pd.notna(y_axis_max):
                    axs[n].set_ylim(bottom=0,top=float(y_axis_max))
                axs[n].legend(loc='lower left',bbox_to_anchor=(1,0),
                              facecolor='inherit', frameon=True)
                
                PlotDataHelper.set_plot_timeseries_format(axs,n)
                if facet:
                    n+=1

            fig3.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', top=False, bottom=False, 
                            left=False, right=False)
            if mconfig.parser("plot_title_as_region"):
                plt.title(zone_input)
            plt.ylabel(f"{self.AGG_BY} Price ($/MWh)", color='black', 
                       rotation='vertical', labelpad=20)
            plt.xlabel(timezone,  color='black', rotation='horizontal', labelpad=20)

            outputs[zone_input] = {'fig': fig3, 'data_table':Data_Out}
        return outputs

    def timeseries_price_all_regions(self, y_axis_max: float = None,
                                     timezone: str = "", start_date_range: str = None, 
                                     end_date_range: str = None, **_):
        """Creates a price timeseries plot for all regions/zones and plots them on a single facet plot.

        Price in $/MWh.
        The code automatically creates a facet plot based on the number of regions/zones in the input.
        All scenarios are plotted on a single facet for each region/zone.

        Args:
            y_axis_max (float, optional): Max y-axis value. 
                Defaults to None.
            timezone (str, optional): The timezone to display on the x-axes.
                Defaults to "".
            start_date_range (str, optional): Defines a start date at which to represent data from. 
                Defaults to None.
            end_date_range (str, optional): Defines a end date at which to represent data to.
                Defaults to None.

        Returns:
            dict: dictionary containing the created plot and its data table.
        """
        outputs = {}
        
        if self.AGG_BY == 'zone':
            agg = 'zone'
        else:
            agg = 'region'
            
        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True, f"{agg}_Price", self.Scenarios)]
        
        # Runs get_formatted_data within PlotDataHelper to populate PlotDataHelper dictionary  
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)

        if 1 in check_input_data:
            return MissingInputData()

        #Location to save to
        save_figures = os.path.join(self.figure_folder, self.AGG_BY + '_prices')

        outputs = {}

        region_number = len(self.Zones)
        xdimension, ydimension =  self.set_x_y_dimension(region_number)
        
        grid_size = xdimension*ydimension
        # Used to calculate any excess axis to delete
        excess_axs = grid_size - region_number

        #setup plot
        fig4, axs = plotlib.setup_plot(xdimension,ydimension)
        plt.subplots_adjust(wspace=0.1, hspace=0.70)

        data_table = []
        for n, zone_input in enumerate(self.Zones):
            self.logger.info(f"{self.AGG_BY} = {zone_input}")

            all_prices=[]
            for scenario in self.Scenarios:
                price = self._process_data(self[f"{agg}_Price"],scenario,zone_input)
                price = price.groupby(["timestamp"]).sum()
                
                if pd.notna(start_date_range):
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
                plotlib.create_line_plot(axs,timeseries,column,color_dict,n=n,label=column)
                axs[n].set_title(zone_input.replace('_',' '))
                if pd.notna(y_axis_max):
                    axs[n].set_ylim(bottom=0,top=float(y_axis_max))
                PlotDataHelper.set_plot_timeseries_format(axs,n)

                handles, labels = axs[n].get_legend_handles_labels()
                #Legend
                axs[grid_size-1].legend((handles), (labels), loc='lower left',bbox_to_anchor=(1,0),
                              facecolor='inherit', frameon=True)
        
        # Remove extra axes
        if excess_axs != 0:
            PlotDataHelper.remove_excess_axs(axs,excess_axs,grid_size)
        
        fig4.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, 
                        right=False)
        plt.ylabel(f"{self.AGG_BY} Price ($/MWh)",  color='black', rotation='vertical', 
                   labelpad=30)
        plt.xlabel(timezone,  color='black', rotation='horizontal', labelpad=20)

        Data_Table_Out = pd.concat(data_table, axis=1)

        Data_Table_Out = Data_Table_Out.add_suffix(" ($/MWh)")

        fig4.savefig(os.path.join(save_figures, "Price_Timeseries_All_Regions.svg"), 
                     dpi=600, bbox_inches='tight')
        Data_Table_Out.to_csv(os.path.join(save_figures, "Price_Timeseries_All_Regions.csv"))
        outputs = DataSavedInModule()
        return outputs
    
    def node_pdc(self, **kwargs):
        """Creates a price duration curve for a set of specifc nodes.

        Price in $/MWh.
        The code will create either a facet plot or a single plot depending on 
        the number of nodes included in plot_select.csv property entry.

        Returns:
            DataSavedInModule: DataSavedInModule exception.
        """
        outputs = self._node_price(PDC=True, **kwargs)
        return outputs
    
    def node_timeseries_price(self, **kwargs):
        """Creates a price timeseries plot for a set of specifc nodes.
        
        Price in $/MWh.
        The code will create either a facet plot or a single plot depending on 
        the number of nodes included in plot_select.csv property entry.

        Returns:
            DataSavedInModule: DataSavedInModule exception.
        """    
        outputs = self._node_price(**kwargs)
        return outputs
        
    def _node_price(self, PDC: bool = False, figure_name: str = None,
                    prop: str = None, y_axis_max: float = None,
                    timezone: str = "", 
                    start_date_range: str = None, 
                    end_date_range: str = None, **_):
        """Creates a price duration curve or timeseries plot for a set of specifc nodes. 

        This method is called from either node_pdc() or node_timeseries_price()
        
        If PDC == True, a price duration curve plot will be created
        The code will create either a facet plot or a single plot depending on 
        the number of nodes included in plot_select.csv property entry.
        Plots and Data are saved within the module

        Args:
            PDC (bool, optional): If True creates a price duration curve.
                Defaults to False.
            figure_name (str, optional): User defined figure output name.
                Defaults to None.
            prop (str, optional): comma seperated string of nodes to display. 
                Defaults to None.
            y_axis_max (float, optional): Max y-axis value. 
                Defaults to None.
            timezone (str, optional): The timezone to display on the x-axes.
                Defaults to "".
            start_date_range (str, optional): Defines a start date at which to represent data from. 
                Defaults to None.
            end_date_range (str, optional): Defines a end date at which to represent data to.
                Defaults to None.

        Returns:
            DataSavedInModule: DataSavedInModule exception.
        """
        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True, "node_Price", self.Scenarios)]
        
        # Runs get_formatted_data within PlotDataHelper to populate PlotDataHelper dictionary  
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)

        if 1 in check_input_data:
            return MissingInputData()
        
        node_figure_folder = os.path.join(self.figure_folder, 'node_prices')
        try:
            os.makedirs(node_figure_folder)
        except FileExistsError:
            # directory already exists
            pass

        #Select only node specified in Marmot_plot_select.csv.
        select_nodes = prop.split(",")
        if select_nodes == None:
            return InputSheetError()
        
        self.logger.info(f'Plotting Prices for {select_nodes}')
        
        all_prices=[]
        for scenario in self.Scenarios:
            self.logger.info(f"Scenario = {scenario}")

            price = self["node_Price"][scenario]
            price = price.loc[(slice(None), select_nodes),:]
            price = price.groupby(["timestamp","node"]).sum()
            price.rename(columns={0:scenario}, inplace=True)
            
            if pd.notna(start_date_range):
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
        
        xdimension, ydimension =  self.set_x_y_dimension(len(select_nodes))
        
        #setup plot
        fig, axs = plotlib.setup_plot(xdimension,ydimension)
        plt.subplots_adjust(wspace=0.1, hspace=0.70)
        
        color_dict = dict(zip(pdc.columns, self.color_list))

        for n, node in enumerate(select_nodes):
            
            if PDC:
                try:
                    node_pdc = pdc.xs(node)
                    node_pdc.reset_index(drop=True, inplace=True)
                except KeyError:
                    self.logger.info(f"{node} not found")
                    continue
            else:
                try:
                    node_pdc = pdc.xs(node, level='node')
                except KeyError:
                    self.logger.info(f"{node} not found")
                    continue
            
            for column in node_pdc:
                plotlib.create_line_plot(axs,node_pdc, column, color_dict, 
                                       n=n, label=column)
                if pd.notna(y_axis_max):
                    axs[n].set_ylim(bottom=0, top=float(y_axis_max))
                if not PDC:
                    PlotDataHelper.set_plot_timeseries_format(axs,n)
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
        outputs = DataSavedInModule()
        return outputs
    
    def node_price_hist(self, **kwargs):
        """Creates a price histogram for a specifc nodes. Price in $/MWh.
        
        A facet plot will be created if more than one scenario are included on the 
        user input sheet
        Each scenario will be plotted on a separate subplot.
        If a set of nodes are passed at input, each will be saved to a separate 
        figure with node name as a suffix. 
        Plots and Data are saved within the module

        Returns:
            DataSavedInModule: DataSavedInModule exception.
        """    
        outputs = self._node_hist(**kwargs)
        return outputs
        
    def node_price_hist_diff(self, **kwargs):
        """Creates a difference price histogram for a specifc nodes. Price in $/MWh.
        
        This plot requires more than one scenario to display correctly.
        A facet plot will be created
        Each scenario will be plotted on a separate subplot, with values displaying 
        the relative difference to the first scenario in the list.
        If a set of nodes are passed at input, each will be saved to a separate 
        figure with node name as a suffix. 
        Plots and Data are saved within the module

        Returns:
            DataSavedInModule: DataSavedInModule exception.
        """    
        outputs = self._node_hist(diff_plot=True, **kwargs)
        return outputs
        
    def _node_hist(self, diff_plot: bool = False, figure_name: str = None,
                   prop: str = None, start_date_range: str = None,
                   end_date_range: str = None, **_):
        """Internal code for hist plots.
        
        Called from node_price_hist() or node_price_hist_diff(). 

        Hist range and bin size is currently hardcoded from -100 to +100
        with a bin width of 2.5 $/MWh 

        Args:
            diff_plot (bool, optional): If True creates a diff plot. 
                Defaults to False.
            figure_name (str, optional): User defined figure output name.
                Defaults to None.
            prop (str, optional): comma seperated string of nodes to display. 
                Defaults to None.
            start_date_range (str, optional): Defines a start date at which to represent data from. 
                Defaults to None.
            end_date_range (str, optional): Defines a end date at which to represent data to.
                Defaults to None.

        Returns:
            DataSavedInModule: DataSavedInModule exception.
        """
        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True, "node_Price", self.Scenarios)]
        
        # Runs get_formatted_data within PlotDataHelper to populate PlotDataHelper dictionary  
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)

        if 1 in check_input_data:
            return MissingInputData()
        
        node_figure_folder = os.path.join(self.figure_folder, 'node_prices')
        try:
            os.makedirs(node_figure_folder)
        except FileExistsError:
            # directory already exists
            pass
        
        #Select only node specified in Marmot_plot_select.csv.
        select_nodes = prop.split(",")
        if select_nodes == None:
            return InputSheetError()
        
        for node in select_nodes:
            self.logger.info(f'Plotting Prices for Node: {node}')
        
            all_prices=[]
            for scenario in self.Scenarios:
                self.logger.info(f"Scenario = {scenario}")
    
                price = self["node_Price"][scenario]
                price = price.xs(node, level='node')
                # price = price.loc[(slice(None), select_nodes),:]
                price = price.groupby(["timestamp"]).sum()
                price.rename(columns={0:scenario}, inplace=True)
                
                if pd.notna(start_date_range):
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
            
            xdimension, ydimension = self.setup_facet_xy_dimensions(multi_scenario=self.Scenarios)
            grid_size = xdimension*ydimension
             # Used to calculate any excess axis to delete
            plot_number = len(self.Scenarios)
            excess_axs = grid_size - plot_number
        
            #setup plot
            fig, axs = plotlib.setup_plot(xdimension,ydimension, sharey=True)
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
                
                # Set plot data equal to 0 if all zero, e.g diff plot
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
                PlotDataHelper.remove_excess_axs(axs,excess_axs,grid_size)
            
            self.add_facet_labels(fig)
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
            
        outputs = DataSavedInModule()
        return outputs
    
    
    def _process_data(self, data_collection, scenario, zone_input):
        df = data_collection.get(scenario)
        df = df.xs(zone_input,level=self.AGG_BY)
        df = df.rename(columns={0:scenario})
        return df

