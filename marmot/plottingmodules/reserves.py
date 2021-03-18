# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 15:59:45 2020

Updated on Monday 21 Sep 2020

This module creates plots of reserve provision and shortage at the generation and region level

@author: Daniel Levie
"""

import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import marmot.plottingmodules.marmot_plot_functions as mfunc
import logging

#===============================================================================


class mplot(object):
    def __init__(self, argument_dict):
        # iterate over items in argument_dict and set as properties of class
        # see key_list in Marmot_plot_main for list of properties
        for prop in argument_dict:
            self.__setattr__(prop, argument_dict[prop])
        self.logger = logging.getLogger('marmot_plot.'+__name__)
        
    def reserve_gen_timeseries(self):
        """
        This method creates a generation stackplot of reserve provision for each region.
        A Facet Plot is created if multiple scenarios are compared.
        Generation is ordered by tech type that provides reserves
        Figures and data tables are returned to plot_main
        """
        # If not facet plot, only plot first sceanrio
        if not self.facet:
            self.Scenarios = [self.Scenarios[0]]
        outputs = {}
        check_input_data = []
        reserve_provision_collection = {}
        check_input_data.extend([mfunc.get_data(reserve_provision_collection,"reserves_generators_Provision",self.Marmot_Solutions_folder, self.Scenarios)])

        if 1 in check_input_data:
            outputs = mfunc.MissingInputData()
            return outputs

        for region in self.Zones:
            self.logger.info("Zone = "+ region)

            xdimension, ydimension = mfunc.setup_facet_xy_dimensions(self.xlabels,self.ylabels,self.facet,multi_scenario=self.Scenarios)
            grid_size = xdimension*ydimension
            excess_axs = grid_size - len(self.Scenarios)

            fig1, axs = mfunc.setup_plot(xdimension,ydimension)
            plt.subplots_adjust(wspace=0.05, hspace=0.2)

            data_tables = {}
            unique_tech_names = []
            n=0 #Counter for scenario subplots
            for scenario in self.Scenarios:
                self.logger.info("Scenario = " + scenario)

                reserve_provision_timeseries = reserve_provision_collection.get(scenario)

                #Check if zone has reserves, if not skips
                try:
                    reserve_provision_timeseries = reserve_provision_timeseries.xs(region,level=self.AGG_BY)
                except KeyError:
                    self.logger.info("No reserves deployed in : " + scenario)
                    continue
                reserve_provision_timeseries = mfunc.df_process_gen_inputs(reserve_provision_timeseries,self.ordered_gen)
                data_tables[scenario] = reserve_provision_timeseries

                if self.prop == "Peak Demand":
                    self.logger.info("Plotting Peak Demand period")

                    total_reserve = reserve_provision_timeseries.sum(axis=1)
                    peak_reserve_t =  total_reserve.idxmax()
                    start_date = peak_reserve_t - dt.timedelta(days=self.start)
                    end_date = peak_reserve_t + dt.timedelta(days=self.end)
                    reserve_provision_timeseries = reserve_provision_timeseries[start_date : end_date]
                    Peak_Reserve = total_reserve[peak_reserve_t]

                elif self.prop == 'Date Range':
                    self.logger.info("Plotting specific date range: \
                    {} to {}".format(str(self.start_date),str(self.end_date)))
                    reserve_provision_timeseries = reserve_provision_timeseries[self.start_date : self.end_date]
                else:
                    self.logger.info("Plotting graph for entire timeperiod")
                
                # unitconversion based off peak generation hour, only checked once 
                if n == 0:
                    unitconversion = mfunc.capacity_energy_unitconversion(max(reserve_provision_timeseries.sum(axis=1)))
                reserve_provision_timeseries = reserve_provision_timeseries/unitconversion['divisor']
                
                mfunc.create_stackplot(axs, reserve_provision_timeseries, self.PLEXOS_color_dict, label=reserve_provision_timeseries.columns,n=n)
                mfunc.set_plot_timeseries_format(axs,n=n,minticks=4, maxticks=8)

                if self.prop == "Peak Demand":
                    axs[n].annotate('Peak Reserve: \n' + str(format(int(Peak_Reserve/unitconversion['divisor']), '.2f')) + ' {}'.format(unitconversion['units']), 
                                    xy=(peak_reserve_t, Peak_Reserve),
                            xytext=((peak_reserve_t + dt.timedelta(days=0.25)), (Peak_Reserve + Peak_Reserve*0.05)),
                            fontsize=13, arrowprops=dict(facecolor='black', width=3, shrink=0.1))

                # create list of gen technologies
                l1 = reserve_provision_timeseries.columns.tolist()
                unique_tech_names.extend(l1)

                if self.facet:
                    n=n+1
            
            if not data_tables:
                self.logger.warning('No reserves in ' + region)
                out = mfunc.MissingZoneData()
                return out
                
            # create handles list of unique tech names then order
            handles = np.unique(np.array(unique_tech_names)).tolist()
            handles.sort(key = lambda i:self.ordered_gen.index(i))
            handles = reversed(handles)

            # create custom gen_tech legend
            gen_tech_legend = []
            for tech in handles:
                legend_handles = [Patch(facecolor=self.PLEXOS_color_dict[tech],
                            alpha=1.0,
                         label=tech)]
                gen_tech_legend.extend(legend_handles)

            # Add legend
            axs[grid_size-1].legend(handles=gen_tech_legend, loc='lower left',bbox_to_anchor=(1,0),
                     facecolor='inherit', frameon=True)

            #Remove extra axes
            if excess_axs != 0:
                mfunc.remove_excess_axs(axs,excess_axs,grid_size)

            # add facet labels
            mfunc.add_facet_labels(fig1, self.xlabels, self.ylabels)

            fig1.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            plt.ylabel('Reserve Provision ({})'.format(unitconversion['units']),  color='black', rotation='vertical', labelpad=30)

            if not self.facet:
                data_tables = data_tables[self.Scenarios[0]]

            outputs[region] = {'fig': fig1, 'data_table': data_tables}
        return outputs

    def total_reserves_by_gen(self):
        """
        This method creates a generation barplot of total reserve provision by generator for each region.
        Multiple scenarios are assigned to the x-axis
        Figures and data tables are returned to plot_main
        """

        outputs = {}
        check_input_data = []
        reserve_provision_collection = {}
        check_input_data.extend([mfunc.get_data(reserve_provision_collection,"reserves_generators_Provision",self.Marmot_Solutions_folder, self.Scenarios)])

        if 1 in check_input_data:
            outputs = mfunc.MissingInputData()
            return outputs

        for region in self.Zones:
            self.logger.info("Zone = "+ region)

            Total_Reserves_Out = pd.DataFrame()
            unique_tech_names = []
            for scenario in self.Scenarios:
                self.logger.info("Scenario = " + scenario)

                reserve_provision_timeseries = reserve_provision_collection.get(scenario)
                #Check if zone has reserves, if not skips
                try:
                    reserve_provision_timeseries = reserve_provision_timeseries.xs(region,level=self.AGG_BY)
                except KeyError:
                    self.logger.info("No reserves deployed in %s", scenario)
                    continue
                reserve_provision_timeseries = mfunc.df_process_gen_inputs(reserve_provision_timeseries,self.ordered_gen)

                # Calculates interval step to correct for MWh of generation
                time_delta = reserve_provision_timeseries.index[1]- reserve_provision_timeseries.index[0]
                # Finds intervals in 60 minute period
                interval_count = 60/(time_delta/np.timedelta64(1, 'm'))

                # sum totals by fuel types
                reserve_provision_timeseries = reserve_provision_timeseries/interval_count
                reserve_provision = reserve_provision_timeseries.sum(axis=0)
                reserve_provision.rename(scenario, inplace=True)
                Total_Reserves_Out = pd.concat([Total_Reserves_Out, reserve_provision], axis=1, sort=False).fillna(0)


            Total_Reserves_Out = mfunc.df_process_categorical_index(Total_Reserves_Out, self.ordered_gen)
            Total_Reserves_Out = Total_Reserves_Out.T
            Total_Reserves_Out = Total_Reserves_Out.loc[:, (Total_Reserves_Out != 0).any(axis=0)]
            
            if Total_Reserves_Out.empty:
                out = mfunc.MissingZoneData()
                outputs[region] = out
                continue
            
            Total_Reserves_Out.index = Total_Reserves_Out.index.str.replace('_',' ')
            Total_Reserves_Out.index = Total_Reserves_Out.index.str.wrap(5, break_long_words=False)
            data_table_out = Total_Reserves_Out
            
            # Convert units
            unitconversion = mfunc.capacity_energy_unitconversion(max(Total_Reserves_Out.sum()))
            Total_Reserves_Out = Total_Reserves_Out/unitconversion['divisor'] 
            
            # create figure
            fig1 = mfunc.create_stacked_bar_plot(Total_Reserves_Out, self.PLEXOS_color_dict)

            # additional figure formatting
            fig1.set_ylabel('Total Reserve Provision ({}h)'.format(unitconversion['units']),  color='black', rotation='vertical')

            # replace x-axis with custom labels
            if len(self.ticklabels) > 1:
                self.ticklabels = pd.Series(self.ticklabels).str.replace('-','- ').str.wrap(8, break_long_words=True)
                fig1.set_xticklabels(self.ticklabels)

            # create list of gen technologies
            l1 = Total_Reserves_Out.columns.tolist()
            unique_tech_names.extend(l1)

            # create handles list of unique tech names then order
            handles = np.unique(np.array(unique_tech_names)).tolist()
            handles.sort(key = lambda i:self.ordered_gen.index(i))
            handles = reversed(handles)

            # create custom gen_tech legend
            gen_tech_legend = []
            for tech in handles:
                legend_handles = [Patch(facecolor=self.PLEXOS_color_dict[tech],
                            alpha=1.0,label=tech)]
                gen_tech_legend.extend(legend_handles)

            # Add legend
            fig1.legend(handles=gen_tech_legend, loc='lower left',bbox_to_anchor=(1,0),
                     facecolor='inherit', frameon=True)

            outputs[region] = {'fig': fig1, 'data_table': data_table_out}
        return outputs

    def reg_reserve_shortage(self):
        """
        This method creates a bar plot of reserve shortage for each region in MWh.
        Bars are grouped by reserve type
        Figures and data tables are returned to plot_main
        """
        outputs = self._reserve_bar_plots("Shortage")
        return outputs

    def reg_reserve_provision(self):
        """
        This method creates a bar plot of reserve provision for each region in MWh.
        Bars are grouped by reserve type
        Figures and data tables are returned to plot_main
        """
        outputs = self._reserve_bar_plots("Provision")
        return outputs

    def reg_reserve_shortage_hrs(self):
        """
        This method creates a bar plot of reserve shortage for each region in hrs.
        Bars are grouped by reserve type
        Figures and data tables are returned to plot_main
        """
        outputs = self._reserve_bar_plots("Shortage",count_hours=True)
        return outputs

    def _reserve_bar_plots(self, data_set, count_hours=False):
        reserve_collection = {}
        check_input_data = []
        check_input_data.extend([mfunc.get_data(reserve_collection,"reserve_{}".format(data_set),self.Marmot_Solutions_folder, self.Scenarios)])

        if 1 in check_input_data:
            outputs = mfunc.MissingInputData()
            return outputs

        outputs = {}
        for region in self.Zones:
            self.logger.info("Zone = "+ region)

            Data_Table_Out=pd.DataFrame()
            reserve_total_chunk = []
            for scenario in self.Scenarios:

                self.logger.info('Scenario = ' + scenario)

                reserve_timeseries = reserve_collection.get(scenario)
                # Check if zone has reserves, if not skips
                try:
                    reserve_timeseries = reserve_timeseries.xs(region,level=self.AGG_BY)
                except KeyError:
                    self.logger.info("No reserves deployed in %s", scenario)
                    continue
                timestamps = reserve_timeseries.index.get_level_values('timestamp').unique()
                # Calculates interval step to correct for MWh of generation
                time_delta = timestamps[1]- timestamps[0]
                # Finds intervals in 60 minute period
                interval_count = 60/(time_delta/np.timedelta64(1, 'm'))

                reserve_timeseries = reserve_timeseries.reset_index(["timestamp","Type","parent"],drop=False)
                # Drop duplicates to remove double counting
                reserve_timeseries.drop_duplicates(inplace=True)
                # Set Type equal to parent value if Type eqauls '-'
                reserve_timeseries['Type'] = reserve_timeseries['Type'].mask(reserve_timeseries['Type'] == '-', reserve_timeseries['parent'])
                reserve_timeseries.set_index(["timestamp","Type","parent"],append=True,inplace=True)

                # Groupby Type
                if count_hours == False:
                    reserve_total = reserve_timeseries.groupby(["Type"]).sum()/interval_count
                elif count_hours == True:
                    reserve_total = reserve_timeseries[reserve_timeseries[0]>0] #Filter for non zero values
                    reserve_total = reserve_total.groupby("Type").count()/interval_count

                reserve_total.rename(columns={0:scenario},inplace=True)

                reserve_total_chunk.append(reserve_total)

            reserve_out = pd.concat(reserve_total_chunk,axis=1, sort='False')
            # remove any rows that all eqaul 0
            reserve_out = reserve_out.loc[(reserve_out != 0).any(axis=1),:]
            reserve_out.columns = reserve_out.columns.str.replace('_',' ')
        
            # If no reserves return nothing
            if reserve_out.empty:
                out = mfunc.MissingZoneData()
                outputs[region] = out
                continue
            
            Data_Table_Out=pd.concat([Data_Table_Out,reserve_out],axis=1)

            if count_hours == False:
                # Convert units
                unitconversion = mfunc.capacity_energy_unitconversion(max(reserve_out.sum()))
                reserve_out = reserve_out/unitconversion['divisor'] 

            # create color dictionary
            color_dict = dict(zip(reserve_out.columns,self.color_list))

            fig2 = mfunc.create_grouped_bar_plot(reserve_out,color_dict)
            if count_hours == False:
                fig2.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.1f}'))
                fig2.set_ylabel('Reserve {} [{}h]'.format(data_set,unitconversion['units'] ),  color='black', rotation='vertical')
            elif count_hours == True:
                fig2.set_ylabel('Reserve {} Hours'.format(data_set),  color='black', rotation='vertical')
            handles, labels = fig2.get_legend_handles_labels()
            fig2.legend(handles,labels, loc='lower left',bbox_to_anchor=(1,0),
                          facecolor='inherit', frameon=True)

            outputs[region] = {'fig': fig2,'data_table': Data_Table_Out}
        return outputs


    def reg_reserve_shortage_timeseries(self):
        """
        This method creates a timeseries line plot of reserve shortage for each region.
        A Facet Plot is created if multiple scenarios are compared.
        A line is plotted for each reserve type shortage.
        Figures and data tables are returned to plot_main
        """
        outputs = {}
        reserve_collection = {}
        check_input_data = []

        # If not facet plot, only plot first sceanrio
        if not self.facet:
            self.Scenarios = [self.Scenarios[0]]

        check_input_data.extend([mfunc.get_data(reserve_collection,"reserve_Shortage", self.Marmot_Solutions_folder, self.Scenarios)])

        if 1 in check_input_data:
            outputs = mfunc.MissingInputData()
            return outputs

        for region in self.Zones:
            self.logger.info("Zone = "+ region)

            xdimension, ydimension = mfunc.setup_facet_xy_dimensions(self.xlabels,self.ylabels,self.facet,multi_scenario = self.Scenarios)

            grid_size = xdimension*ydimension
            excess_axs = grid_size - len(self.Scenarios)

            fig3, axs = mfunc.setup_plot(xdimension,ydimension)
            plt.subplots_adjust(wspace=0.05, hspace=0.2)

            data_tables = {}
            unique_reserve_types = []
            n=0 #Counter for scenario subplots
            
            if not self.facet:
                self.Scenarios = [self.Scenarios[0]]
            for scenario in self.Scenarios:

                self.logger.info('Scenario = ' + scenario)

                reserve_timeseries = reserve_collection.get(scenario)
                # Check if zone has reserves, if not skips
                try:
                    reserve_timeseries = reserve_timeseries.xs(region,level=self.AGG_BY)
                except KeyError:
                    self.logger.info("No reserves deployed in %s", scenario)
                    continue
                
                reserve_timeseries.reset_index(["timestamp","Type","parent"],drop=False,inplace=True)
                reserve_timeseries = reserve_timeseries.drop_duplicates()
                # Set Type equal to parent value if Type eqauls '-'
                reserve_timeseries['Type'] = reserve_timeseries['Type'].mask(reserve_timeseries['Type'] == '-', reserve_timeseries['parent'])
                reserve_timeseries = reserve_timeseries.pivot(index='timestamp', columns='Type', values=0)

                if self.prop == 'Date Range':
                    self.logger.info("Plotting specific date range: \
                    {} to {}".format(str(self.start_date),str(self.end_date)))
                    reserve_timeseries = reserve_timeseries[self.start_date : self.end_date]
                else:
                    self.logger.info("Plotting graph for entire timeperiod")

                # create color dictionary
                color_dict = dict(zip(reserve_timeseries.columns,self.color_list))

                data_tables[scenario] = reserve_timeseries 

                for column in reserve_timeseries:
                    mfunc.create_line_plot(axs,reserve_timeseries,column,color_dict=color_dict,label=column, n=n)
                axs[n].yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.1f}'))
                axs[n].margins(x=0.01)
                mfunc.set_plot_timeseries_format(axs,n=n,minticks=6, maxticks=12)

                # scenario_names = pd.Series([scenario]*len(reserve_timeseries),name='Scenario')
                # reserve_timeseries = reserve_timeseries.set_index([scenario_names],append=True)
                # reserve_timeseries_chunk.append(reserve_timeseries)

                # create list of gen technologies
                l1 = reserve_timeseries.columns.tolist()
                unique_reserve_types.extend(l1)

                if self.facet:
                    n+=1
            
            if not data_tables:
                out = mfunc.MissingZoneData()
                outputs[region] = out
                continue
                
            # create handles list of unique reserve names
            handles = np.unique(np.array(unique_reserve_types)).tolist()

            # create color dictionary
            color_dict = dict(zip(handles,self.color_list))

            # create custom gen_tech legend
            reserve_legend = []
            for Type in handles:
                legend_handles = [Line2D([0], [0], color=color_dict[Type], lw=2, label=Type)]
                reserve_legend.extend(legend_handles)

            axs[grid_size-1].legend(handles=reserve_legend,loc='lower left',bbox_to_anchor=(1,0),facecolor='inherit', frameon=True)

            #Remove extra axes
            if excess_axs != 0:
                mfunc.remove_excess_axs(axs,excess_axs,grid_size)

            # add facet labels
            mfunc.add_facet_labels(fig3, self.xlabels, self.ylabels)

            fig3.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            # plt.xlabel('Date ' + '(' + self.timezone + ')',  color='black', rotation='horizontal',labelpad = 30)
            plt.ylabel('Reserve Shortage [MW]',  color='black', rotation='vertical',labelpad = 30)

            #Data_Out = pd.concat(reserve_timeseries_chunk,axis=0)
            if not self.facet:
                data_tables = data_tables[self.Scenarios[0]]

            outputs[region] =  {'fig': fig3, 'data_table': data_tables}

        return outputs
            
