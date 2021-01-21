# -*- coding: utf-8 -*-
"""
Created April 2020, updated August 2020

This code creates transmission line and interface plots and is called from Marmot_plot_main.py

@author: dlevie
"""

import os
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
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

    def net_export(self):

        """
        This method creates a net export line graph for each region.
        All scenarios are plotted on a single figure.
        Figures and data tables are returned to plot_main
        """
        net_interchange_collection = {}
        check_input_data = []
        if self.AGG_BY=='zone':
            check_input_data.extend([mfunc.get_data(net_interchange_collection,"zone_Net_Interchange",self.Marmot_Solutions_folder, self.Multi_Scenario)])
        else: 
            check_input_data.extend([mfunc.get_data(net_interchange_collection,"region_Net_Interchange",self.Marmot_Solutions_folder, self.Multi_Scenario)])
        
        if 1 in check_input_data:
            outputs = mfunc.MissingInputData()
            return outputs

        outputs = {}
        for zone_input in self.Zones:
            self.logger.info(self.AGG_BY + " = " + zone_input)

            net_export_all_scenarios = pd.DataFrame()

            for scenario in self.Multi_Scenario:

                self.logger.info("Scenario = " + str(scenario))
                net_export_read = net_interchange_collection.get(scenario)
                net_export = net_export_read.xs(zone_input, level = self.AGG_BY)
                net_export = net_export.groupby("timestamp").sum()
                net_export.columns = [scenario]

                if self.prop == 'Date Range':
                    self.logger.info("Plotting specific date range: \
                    {} to {}".format(str(self.start_date),str(self.end_date)))

                    net_export = net_export[self.start_date : self.end_date]

                net_export_all_scenarios = pd.concat([net_export_all_scenarios,net_export], axis = 1)
                net_export_all_scenarios.columns = net_export_all_scenarios.columns.str.replace('_',' ')

            # Data table of values to return to main program
            Data_Table_Out = net_export_all_scenarios.copy()
            #Make scenario/color dictionary.
            color_dict = dict(zip(net_export_all_scenarios.columns,self.color_list))

            # if '2008' not in self.Marmot_Solutions_folder and '2012' not in self.Marmot_Solutions_folder and Net_Export_all_scenarios.index[0] > dt.datetime(2024,2,28,0,0):
            #     Net_Export_all_scenarios.index = Net_Export_all_scenarios.index.shift(1,freq = 'D') #TO DEAL WITH LEAP DAYS, SPECIFIC TO MARTY'S PROJECT, REMOVE AFTER.

            fig1, axs = mfunc.setup_plot()
            plt.subplots_adjust(wspace=0.05, hspace=0.2)
            
            if net_export_all_scenarios.empty:
                out = mfunc.MissingZoneData()
                outputs[zone_input] = out
                continue
            
            n=0
            for column in net_export_all_scenarios:
                mfunc.create_line_plot(axs,net_export_all_scenarios,column,color_dict)
                axs[n].set_ylabel('Net exports (MW)',  color='black', rotation='vertical')
                axs[n].set_xlabel('Date ' + '(' + self.timezone + ')',  color='black', rotation='horizontal')
                axs[n].yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
                axs[n].margins(x=0.01)
                axs[n].hlines(y = 0, xmin = axs[n].get_xlim()[0], xmax = axs[n].get_xlim()[1], linestyle = ':')
                mfunc.set_plot_timeseries_format(axs,n)

            handles, labels = axs[n].get_legend_handles_labels()

            axs[n].legend(reversed(handles), reversed(labels), loc='lower left',bbox_to_anchor=(1,0),
                          facecolor='inherit', frameon=True)
            outputs[zone_input] = {'fig': fig1, 'data_table': Data_Table_Out}
        return outputs

    #Duration curve of individual line utilization for all hours
    def line_util(self):

        """
        This method creates a timeseries line plot of transmission lineflow utilization for each region.
        Utilization is plotted between 0 and 1 on the y-axis
        The plot will default to showing the 10 highest utilized lines, a Line category can also be passed
        instead using the property field in the Marmot_plot_select.csv
        Each scenarios is plotted on a seperate Facet plot.
        Figures and data tables are returned to plot_main
        """
        outputs = self._util()
        return outputs

    def line_hist(self):

        """
        This method creates a histogram of transmission lineflow utilization for each region.
        Utilization is plotted between 0 and 1 on the x-axis, with no. lines on the y-axis.
        Each bar is eqaul to a 0.05 utilization rate
        The plot will default to showing all lines, a Line category can also be passed
        instead using the property field in the Marmot_plot_select.csv
        Each scenarios is plotted on a seperate Facet plot.
        Figures and data tables are returned to plot_main
        """
        outputs = self._util(hist=True)
        return outputs

    def _util(self,hist=None):
        Flow_Collection = {}
        Limit_Collection = {}
        outputs = {}
        
        check_input_data = []
        check_input_data.extend([mfunc.get_data(Flow_Collection,"line_Flow",self.Marmot_Solutions_folder, self.Multi_Scenario)])
        check_input_data.extend([mfunc.get_data(Limit_Collection,"line_Import_Limit",self.Marmot_Solutions_folder, self.Multi_Scenario)])
        
        if 1 in check_input_data:
            outputs = mfunc.MissingInputData()
            return outputs

        for zone_input in self.Zones:
            self.logger.info("For all lines touching Zone = "+zone_input)
            
            fig2, axs = mfunc.setup_plot(ydimension=len(self.Multi_Scenario),sharey = False)
            plt.subplots_adjust(wspace=0.05, hspace=0.2)


            n=0 #Counter for scenario subplots
            Data_Out=pd.DataFrame()

            for scenario in self.Multi_Scenario:
                self.logger.info("Scenario = " + str(scenario))

                # gets correct metadata based on area aggregation
                if self.AGG_BY=='zone':
                    zone_lines = self.meta.zone_lines()
                else:
                    zone_lines = self.meta.region_lines()
                try:
                    zone_lines = zone_lines.set_index([self.AGG_BY])
                except:
                    self.logger.warning("Column to Aggregate by is missing")
                    continue

                zone_lines = zone_lines.xs(zone_input)
                zone_lines=zone_lines['line_name'].unique()

                flow = Flow_Collection.get(scenario)
                flow = flow[flow.index.get_level_values('line_name').isin(zone_lines)] #Limit to only lines touching to this zone

                limits = Limit_Collection.get(scenario).droplevel('timestamp')
                limits.mask(limits[0]==0.0,other=0.01,inplace=True) #if limit is zero set to small value

                # This checks for a nan in string. If no scenario selected, do nothing.
                if (self.prop != self.prop)==False:
                    self.logger.info("Line category = "+str(self.prop))
                    line_relations = self.meta.lines().rename(columns={"name":"line_name"}).set_index(["line_name"])
                    flow=pd.merge(flow,line_relations,left_index=True,right_index=True)
                    flow=flow[flow["category"] == self.prop]
                    flow=flow.drop('category',axis=1)

                flow = pd.merge(flow,limits[0].abs(),left_index=True, right_index=True,how='left')
                flow['Util']=flow['0_x'].abs()/flow['0_y']
                #If greater than 1 because exceeds flow limit, report as 1
                flow['Util'][flow['Util'] > 1] = 1
                annual_util=flow['Util'].groupby(["line_name"]).mean().rename(scenario)
                # top annual utilized lines
                top_utilization = annual_util.nlargest(10, keep='first')

                color_dict = dict(zip(self.Multi_Scenario,self.color_list))

                if hist == True:
                    mfunc.create_hist_plot(axs,annual_util,color_dict,label=scenario,n=n)
                    axs[n].set_ylabel(scenario.replace('_',' '),color='black', rotation='vertical')
                else:
                    for line in top_utilization.index.get_level_values(level='line_name').unique():
                        duration_curve = flow.xs(line,level="line_name").sort_values(by='Util',ascending=False).reset_index(drop=True)
                        mfunc.create_line_plot(axs,duration_curve,'Util',label=line,n=n)
                        axs[n].set_ylim((0,1.1))
                        axs[n].set_ylabel(scenario.replace('_',' '),color='black', rotation='vertical')
                        handles, labels = axs[n].get_legend_handles_labels()
                        axs[n].legend(handles,labels, loc='lower left',bbox_to_anchor=(1,0),
                              facecolor='inherit', frameon=True)
                n+=1
                Data_Out=pd.concat([Data_Out,annual_util],axis=1,sort=False)

            fig2.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            if hist == True:
                if (self.prop != self.prop)==True:
                    self.prop='All Lines'
                plt.ylabel('Number of lines',  color='black', rotation='vertical', labelpad=30)
                plt.xlabel('Line Utilization: {}'.format(self.prop),  color='black', rotation='horizontal', labelpad=20)
            else:
                if (self.prop != self.prop)==True:
                    self.prop='Top 10 Lines'
                plt.ylabel('Line Utilization: {}'.format(self.prop),  color='black', rotation='vertical', labelpad=30)
                plt.xlabel('Intervals',  color='black', rotation='horizontal', labelpad=20)
            del annual_util, limits
            outputs[zone_input] = {'fig': fig2,'data_table':Data_Out}
        return outputs

    def int_flow_ind(self):

        """
        This method plots flow, import and export limit, for individual transmission interchanges, with a facet for each interchange.
        The plot includes every interchange that originates or ends in the aggregation zone. 
        Figures and data tables are returned to plot_main
        """
        check_input_data = []
        Flow_Collection = {}
        Import_Limit_Collection = {}
        Export_Limit_Collection = {}

        check_input_data.extend([mfunc.get_data(Flow_Collection,"interface_Flow",self.Marmot_Solutions_folder, self.Multi_Scenario)])
        check_input_data.extend([mfunc.get_data(Import_Limit_Collection,"interface_Import_Limit",self.Marmot_Solutions_folder, self.Multi_Scenario)])
        check_input_data.extend([mfunc.get_data(Export_Limit_Collection,"interface_Export_Limit",self.Marmot_Solutions_folder, self.Multi_Scenario)])

        if 1 in check_input_data:
            outputs = mfunc.MissingInputData()
            return outputs

        scenario = self.Scenario_name

        outputs = {}
        for zone_input in self.Zones:
            self.logger.info("For all interfaces touching Zone = "+zone_input)

            Data_Table_Out = pd.DataFrame()

            # gets correct metadata based on area aggregation
            if self.AGG_BY=='zone':
                zone_lines = self.meta.zone_lines()
            else:
                zone_lines = self.meta.region_lines()
            try:
                zone_lines = zone_lines.set_index([self.AGG_BY])
            except:
                self.logger.info("Column to Aggregate by is missing")
                continue

            zone_lines = zone_lines.xs(zone_input)
            zone_lines = zone_lines['line_name'].unique()

            #Map lines to interfaces
            all_ints = self.meta.interface_lines() #Map lines to interfaces
            all_ints.index = all_ints.line
            ints = all_ints.loc[all_ints.index.intersection(zone_lines)]

            #flow = flow[flow.index.get_level_values('interface_name').isin(ints.interface)] #Limit to only interfaces touching to this zone
            #flow = flow.droplevel('interface_category')

            export_limits = Export_Limit_Collection.get(scenario).droplevel('timestamp')
            export_limits.mask(export_limits[0]==0.0,other=0.01,inplace=True) #if limit is zero set to small value
            export_limits = export_limits[export_limits.index.get_level_values('interface_name').isin(ints.interface)]
            export_limits = export_limits[export_limits[0].abs() < 99998] #Filter out unenforced interfaces.

            #Drop unnecessary columns.
            export_limits.reset_index(inplace = True)
            export_limits.drop(columns = 'interface_category',inplace = True)
            export_limits.set_index('interface_name',inplace = True)

            import_limits = Import_Limit_Collection.get(scenario).droplevel('timestamp')
            import_limits.mask(import_limits[0]==0.0,other=0.01,inplace=True) #if limit is zero set to small value
            import_limits = import_limits[import_limits.index.get_level_values('interface_name').isin(ints.interface)]
            import_limits = import_limits[import_limits[0].abs() < 99998] #Filter out unenforced interfaces.
            reported_ints = import_limits.index.get_level_values('interface_name').unique()

            #Drop unnecessary columns.
            import_limits.reset_index(inplace = True)
            import_limits.drop(columns = 'interface_category',inplace = True)
            import_limits.set_index('interface_name',inplace = True)

            if self.prop != '':
                interf_list = self.prop.split(',')
                self.logger.info('Plotting only interfaces specified in Marmot_plot_select.csv')
                self.logger.info(interf_list) 
            else:
                interf_list = reported_ints.copy()
            xdim,ydim = mfunc.set_x_y_dimension(len(interf_list))

            fig2, axs = mfunc.setup_plot(xdim,ydim,sharey = False)
            axs = axs.ravel()
            plt.subplots_adjust(wspace=0.05, hspace=0.2)
            missing_ints = 0
            chunks = []
            limits_chunks = []
            n = -1
            for interf in interf_list:
                n += 1

                #Remove leading spaces
                if interf[0] == ' ':
                    interf = interf[1:]
                if interf in reported_ints:
                    chunks_interf = []
                    single_exp_lim = export_limits.loc[interf].iloc[0] / 1000
                    single_imp_lim = import_limits.loc[interf].iloc[0] / 1000
                    limits = pd.concat([single_exp_lim,single_imp_lim])
                    limits_chunks.append(limits)

                    single_exp_lim = single_exp_lim.squeeze()
                    single_imp_lim = single_imp_lim.squeeze()

                    for scenario in self.Multi_Scenario:
                        flow = Flow_Collection.get(scenario)
                        single_int = flow.xs(interf,level = 'interface_name') / 1000
                        single_int.columns = [interf]
                        if self.duration_curve:
                            single_int.sort_values(by = interf,ascending = False,inplace = True)
                            single_int.reset_index(inplace = True)
                            single_int.drop(columns = ['timestamp'],inplace = True)
                        else:
                            single_int = single_int.reset_index().drop(columns = 'interface_category').set_index('timestamp')
                        mfunc.create_line_plot(axs,single_int,interf, label = scenario, n=n,alpha = 1)

                        #For output time series .csv
                        scenario_names = pd.Series([scenario] * len(single_int),name = 'Scenario')
                        single_int_out = single_int.set_index([scenario_names],append = True)
                        chunks_interf.append(single_int_out)

                    Data_out_line = pd.concat(chunks_interf,axis = 0)
                    chunks.append(Data_out_line)
                else:
                    self.logger.warning(interf + ' not found in results. Have you tagged it with the "Must Report" property in PLEXOS?')
                    excess_axs += 1
                    missing_lines += 1
                    continue

                axs[n].axhline(y = single_exp_lim, ls = '--',label = 'Export Limit')
                axs[n].axhline(y = single_imp_lim, ls = '--',label = 'Import Limit')
                axs[n].set_title(interf)
                handles, labels = axs[n].get_legend_handles_labels()
                if not self.duration_curve:
                    mfunc.set_plot_timeseries_format(axs, n=n)
                if n == len(interf_list) - 1:
                    axs[n].legend(loc='lower left',bbox_to_anchor=(1.05,-0.2))

                if missing_ints == len(interf_list):
                    outputs = mfunc.MissingInputData()
                    return outputs

            Data_Table_Out = pd.concat(chunks,axis = 1)
            Limits_Out = pd.concat(limits_chunks,axis = 1)
            Limits_Out.index = ['Export Limit','Import Limit']

            fig2.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            plt.ylabel('Flow (GW)',  color='black', rotation='vertical', labelpad=30)
            if self.duration_curve:
                plt.xlabel('Sorted hour of the year', color = 'black', labelpad = 30)
            plt.tight_layout(rect=[0, 0.03, 1, 0.97])
            outputs[zone_input] = {'fig': fig2, 'data_table': Data_Table_Out}
            Limits_Out.to_csv(os.path.join(self.Marmot_Solutions_folder, self.Scenario_name, 'Figures_Output',self.AGG_BY + '_transmission','Individual_Interface_Limits.csv'))

        return outputs

    def line_flow_ind(self):

        """
        This method plots flow, import and export limit, for individual transmission lines, with a facet for each line.
        The lines are specified in the plot properties field of Marmot_plot_select.csv (column 4).
        The plot includes every interchange that originates or ends in the aggregation zone. 
        Figures and data tables are returned to plot_main
        """

        check_input_data = []
        Flow_Collection = {}
        Import_Limit_Collection = {}
        Export_Limit_Collection = {}

        check_input_data.extend([mfunc.get_data(Flow_Collection,"line_Flow",self.Marmot_Solutions_folder, self.Multi_Scenario)])
        check_input_data.extend([mfunc.get_data(Import_Limit_Collection,"line_Import_Limit",self.Marmot_Solutions_folder, self.Multi_Scenario)])
        check_input_data.extend([mfunc.get_data(Export_Limit_Collection,"line_Export_Limit",self.Marmot_Solutions_folder, self.Multi_Scenario)])

        if 1 in check_input_data:
            outputs = mfunc.MissingInputData()
            return outputs

        #Select only lines specified in Marmot_plot_select.csv.
        select_lines = self.prop.split(",")
        if select_lines == None:
            outpus = mfunc.InputSheetError()
            return outputs

        self.logger.info('Plotting only lines specified in Marmot_plot_select.csv')
        self.logger.info(select_lines) 

        scenario = self.Scenario_name

        export_limits = Export_Limit_Collection.get(scenario).droplevel('timestamp')
        export_limits.mask(export_limits[0]==0.0,other=0.01,inplace=True) #if limit is zero set to small value
        export_limits = export_limits[export_limits[0].abs() < 99998] #Filter out unenforced lines.

        import_limits = Import_Limit_Collection.get(scenario).droplevel('timestamp')
        import_limits.mask(import_limits[0]==0.0,other=0.01,inplace=True) #if limit is zero set to small value
        import_limits = import_limits[import_limits[0].abs() < 99998] #Filter out unenforced lines.

        xdim,ydim = mfunc.set_x_y_dimension(len(select_lines))
        grid_size = xdim * ydim
        excess_axs = grid_size - len(select_lines)
        fig2, axs = mfunc.setup_plot(xdim,ydim,sharey = False)
        #plt.subplots_adjust(wspace=0.05, hspace=0.2)

        reported_lines = Flow_Collection[self.Multi_Scenario[0]].index.get_level_values('line_name').unique()
        n = -1
        missing_lines = 0
        chunks = []
        limits_chunks = []
        for line in select_lines:
            n += 1
            #Remove leading spaces
            if line[0] == ' ':
                line = line[1:]
            if line in reported_lines:
                chunks_line = []

                single_exp_lim = export_limits.loc[line]
                single_imp_lim = import_limits.loc[line]
                limits = pd.concat([single_exp_lim,single_imp_lim])
                limits_chunks.append(limits)
                single_exp_lim = single_exp_lim.squeeze()
                single_imp_lim = single_imp_lim.squeeze()

                for scenario in self.Multi_Scenario:
                    flow = Flow_Collection[scenario]
                    single_line = flow.xs(line,level = 'line_name')
                    single_line.columns = [line]
                    if self.duration_curve:
                        single_line.sort_values(by = line,ascending = False,inplace = True)
                        single_line.reset_index(inplace = True)
                        single_line.drop(columns = ['timestamp'],inplace = True)
                    mfunc.create_line_plot(axs,single_line,line, label = scenario + ' line flow', n = n)

                    #Add %congested number to plot.
                    if scenario == self.Scenario_name:
                        viol_exp = single_line[single_line[line] > single_exp_lim].count()
                        viol_imp = single_line[single_line[line] < single_imp_lim].count()
                        viol_perc = 100 * (viol_exp + viol_imp) / len(single_line)
                        viol_perc = round(viol_perc.squeeze(),3)
                        axs[n].annotate('Violation = ' + str(viol_perc) + '% of hours', xy = (0.1,0.15),xycoords='axes fraction')

                        cong_exp = single_line[single_line[line] == single_exp_lim].count()
                        cong_imp = single_line[single_line[line] == single_imp_lim].count()
                        cong_perc = 100 * (cong_exp + cong_imp) / len(single_line)
                        cong_perc = round(cong_perc.squeeze(),0)
                        axs[n].annotate('Congestion = ' + str(cong_perc) + '% of hours', xy = (0.1,0.1),xycoords='axes fraction')

                    #For output time series .csv
                    scenario_names = pd.Series([scenario] * len(single_line),name = 'Scenario')
                    single_line_out = single_line.set_index([scenario_names],append = True)
                    chunks_line.append(single_line_out)

                Data_out_line = pd.concat(chunks_line,axis = 0)
                chunks.append(Data_out_line)
            else:
                self.logger.warning(line + ' not found in results. Have you tagged it with the "Must Report" property in PLEXOS?')
                excess_axs += 1
                missing_lines += 1
                continue

            mfunc.remove_excess_axs(axs,excess_axs,grid_size)
            axs[n].axhline(y = single_exp_lim, ls = '--',label = 'Export Limit')
            axs[n].axhline(y = single_imp_lim, ls = '--',label = 'Import Limit')
            axs[n].set_title(line)
            handles, labels = axs[n].get_legend_handles_labels()
            if not self.duration_curve:
                mfunc.set_plot_timeseries_format(axs, n=n)
            if n == len(select_lines) - 1:
                axs[n].legend(loc='lower left',bbox_to_anchor=(1.05,-0.2))

        if missing_lines == len(select_lines):
            outputs = mfunc.MissingInputData()
            return outputs

        Data_Table_Out = pd.concat(chunks,axis = 1)
        Limits_Out = pd.concat(limits_chunks,axis = 1)
        Limits_Out.index = ['Export Limit','Import Limit']

        fig2.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.ylabel('Flow (MW)',  color='black', rotation='vertical', labelpad=30)
        #plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.tight_layout()

        fn_suffix = '_duration_curve' if self.duration_curve else ''

        fig2.savefig(os.path.join(self.Marmot_Solutions_folder, self.Scenario_name, 'Figures_Output',self.AGG_BY + '_transmission','Individual_Line_Flow' + fn_suffix + '.svg'), dpi=600, bbox_inches='tight')
        Data_Table_Out.to_csv(os.path.join(self.Marmot_Solutions_folder, self.Scenario_name, 'Figures_Output',self.AGG_BY + '_transmission','Individual_Line_Flow' + fn_suffix + '.csv'))
        Limits_Out.to_csv(os.path.join(self.Marmot_Solutions_folder, self.Scenario_name, 'Figures_Output',self.AGG_BY + '_transmission','Individual_Line_Limits.csv'))

        outputs = mfunc.DataSavedInModule()
        return outputs

    def extract_tx_cap(self):

        check_input_data = []
        Import_Limit_Collection = {}
        Export_Limit_Collection = {}
        Int_Import_Limit_Collection = {}
        Int_Export_Limit_Collection = {}

        check_input_data.extend([mfunc.get_data(Int_Import_Limit_Collection,"interface_Import_Limit",self.Marmot_Solutions_folder, self.Multi_Scenario)])
        check_input_data.extend([mfunc.get_data(Int_Export_Limit_Collection,"interface_Export_Limit",self.Marmot_Solutions_folder, self.Multi_Scenario)])
        check_input_data.extend([mfunc.get_data(Import_Limit_Collection,"line_Import_Limit",self.Marmot_Solutions_folder, self.Multi_Scenario)])
        check_input_data.extend([mfunc.get_data(Export_Limit_Collection,"line_Export_Limit",self.Marmot_Solutions_folder, self.Multi_Scenario)])

        for scenario in self.Multi_Scenario:
            self.logger.info(scenario)
            for zone_input in self.Zones:

                #Lines
                # lines = self.meta.region_interregionallines()
                # if scenario == 'ADS':
                #     zone_input = zone_input.split('_WI')[0]
                #     lines = self.meta_ADS.region_interregionallines()

                # lines = lines[lines['region'] == zone_input]
                # import_lim = Import_Limit_Collection[scenario].reset_index()
                # export_lim = Export_Limit_Collection[scenario].reset_index()
                # lines = lines.merge(import_lim,how = 'inner',on = 'line_name')
                # lines = lines[['line_name',0]]
                # lines.columns = ['line_name','import_limit']
                # lines = lines.merge(export_lim, how = 'inner',on = 'line_name')
                # lines = lines[['line_name','import_limit',0]]
                # lines.columns = ['line_name','import_limit','export_limit']

                # fn = os.path.join(self.Marmot_Solutions_folder, 'NARIS', 'Figures_Output',self.AGG_BY + '_transmission','Individual_Interregional_Line_Limits_' + scenario + '.csv')  
                # lines.to_csv(fn)

                # lines = self.meta.region_intraregionallines()
                # if scenario == 'ADS':
                #     lines = self.meta_ADS.region_intraregionallines()

                # lines = lines[lines['region'] == zone_input]
                # import_lim = Import_Limit_Collection[scenario].reset_index()
                # export_lim = Export_Limit_Collection[scenario].reset_index()
                # lines = lines.merge(import_lim,how = 'inner',on = 'line_name')
                # lines = lines[['line_name',0]]
                # lines.columns = ['line_name','import_limit']
                # lines = lines.merge(export_lim, how = 'inner',on = 'line_name')
                # lines = lines[['line_name','import_limit',0]]
                # lines.columns = ['line_name','import_limit','export_limit']

                # fn = os.path.join(self.Marmot_Solutions_folder, 'NARIS', 'Figures_Output',self.AGG_BY + '_transmission','Individual_Intraregional_Line_Limits_' + scenario + '.csv')  
                # lines.to_csv(fn)


                #Interfaces
                PSCo_ints = ['P39 TOT 5_WI','P40 TOT 7_WI']

                int_import_lim = Int_Import_Limit_Collection[scenario].reset_index()
                int_export_lim = Int_Export_Limit_Collection[scenario].reset_index()
                if scenario == 'NARIS':
                    last_timestamp = int_import_lim['timestamp'].unique()[-1] #Last because ADS uses the last timestamp.
                    int_import_lim = int_import_lim[int_import_lim['timestamp'] == last_timestamp]
                    int_export_lim = int_export_lim[int_export_lim['timestamp'] == last_timestamp]
                    lines2ints = self.meta_ADS.interface_lines()
                else:
                    lines2ints = self.meta.interface_lines()

                fn = os.path.join(self.Marmot_Solutions_folder, 'NARIS', 'Figures_Output',self.AGG_BY + '_transmission','test_meta_' + scenario + '.csv')  
                lines2ints.to_csv(fn)


                ints = pd.merge(int_import_lim,int_export_lim,how = 'inner', on = 'interface_name')
                ints.rename(columns = {'0_x':'import_limit','0_y': 'export_limit'},inplace = True)
                all_lines_in_ints = lines2ints['line'].unique()
                test = [line for line in lines['line_name'].unique() if line in all_lines_in_ints]
                print(test)
                ints = ints.merge(lines2ints, how = 'inner', left_on = 'interface_name',right_on = 'interface')

    def region_region_interchange_all_scenarios(self):  

        """
        This method creates a timeseries line plot of interchange flows between the selected region
        to each conecting region.
        If there are more than 4 total interchanges, all other interchanges are aggregated into an 'other' grouping
        Each scenarios is plotted on a seperate Facet plot.
        Figures and data tables are returned to plot_main
        """
        outputs = self._region_region_interchange(self.Multi_Scenario)
        return outputs

    def region_region_interchange_all_regions(self):

        """
        This method creates a timeseries line plot of interchange flows between the selected region
        to each conecting region. All regions are plotted on a single figure with each focus region placed on a seperate
        facet plot
        If there are more than 4 total interchanges, all other interchanges are aggregated into an 'other' grouping
        This figure only plots a single scenario that is defined by Main_scenario_plot in user_defined_inputs.csv.
        Figures and data tables are saved within method
        """
        outputs = self._region_region_interchange([self.Scenario_name],plot_scenario=False)
        return outputs

    def _region_region_interchange(self,scenario_type,plot_scenario=True):
        
        outputs = {}
        net_interchange_collection = {}
        check_input_data = []
        if self.AGG_BY=='zone':
            check_input_data.extend([mfunc.get_data(net_interchange_collection,"zone_zones_Net_Interchange",self.Marmot_Solutions_folder, self.Multi_Scenario)])
        else:
            check_input_data.extend([mfunc.get_data(net_interchange_collection,"region_regions_Net_Interchange",self.Marmot_Solutions_folder, self.Multi_Scenario)])
        
        if 1 in check_input_data:
            outputs = mfunc.MissingInputData()
            return outputs
            
        for zone_input in self.Zones:
            self.logger.info("Zone = %s",zone_input)
            xdimension=len(self.xlabels)
            ydimension=len(self.ylabels)
            if xdimension == 0 or ydimension == 0:
                xdimension, ydimension =  mfunc.set_x_y_dimension(len(scenario_type))

            fig3, axs = mfunc.setup_plot(xdimension,ydimension)
            plt.subplots_adjust(wspace=0.6, hspace=0.3)

            n=0 #Counter for subplots
            Data_Out=pd.DataFrame()

            for scenario in scenario_type:

                rr_int = net_interchange_collection.get(scenario)

                # For plot_main handeling - need to find better solution
                if plot_scenario == False:
                    outputs={}
                    for zone_input in self.Zones:
                        outputs[zone_input] = pd.DataFrame()

                if self.AGG_BY != 'region' and self.AGG_BY != 'zone':
                    agg_region_mapping = self.Region_Mapping[['region',self.AGG_BY]].set_index('region').to_dict()[self.AGG_BY]
                    rr_int = rr_int.reset_index()
                    rr_int['parent'] = rr_int['parent'].map(agg_region_mapping)
                    rr_int['child']  = rr_int['child'].map(agg_region_mapping)

                rr_int_agg = rr_int.groupby(['timestamp','parent','child'],as_index=True).sum()
                rr_int_agg.rename(columns = {0:'flow (MW)'}, inplace = True)
                rr_int_agg = rr_int_agg.reset_index()

                # if '2008' not in self.Marmot_Solutions_folder and '2012' not in self.Marmot_Solutions_folder and rr_int_agg.index[0] > dt.datetime(2024,2,28,0,0):
                #     rr_int_agg.index = rr_int_agg.index.shift(1,freq = 'D') #TO DEAL WITH LEAP DAYS, SPECIFIC TO MARTY'S PROJECT, REMOVE AFTER.

                # If plotting all regions update plot setup
                if plot_scenario == False:
                    #Make a facet plot, one panel for each parent zone.
                    parent_region = rr_int_agg['parent'].unique()
                    plot_number = len(parent_region)
                    xdimension, ydimension =  mfunc.set_x_y_dimension(plot_number)
                    fig3, axs = mfunc.setup_plot(xdimension,ydimension,sharey=False)
                    plt.subplots_adjust(wspace=0.6, hspace=0.3)

                else:
                    parent_region = [zone_input]
                    plot_number = len(scenario_type)

                grid_size = xdimension*ydimension
                excess_axs = grid_size - plot_number

                for parent in parent_region:
                    single_parent = rr_int_agg[rr_int_agg['parent'] == parent]
                    single_parent = single_parent.pivot(index = 'timestamp',columns = 'child',values = 'flow (MW)')
                    single_parent = single_parent.loc[:,(single_parent != 0).any(axis = 0)] #Remove all 0 columns (uninteresting).
                    if (parent in single_parent.columns):
                        single_parent = single_parent.drop(columns = [parent]) #Remove columns if parent = child

                    #Neaten up lines: if more than 4 total interchanges, aggregated all but the highest 3.
                    if len(single_parent.columns) > 4:
                        # Set the "three highest zonal interchanges" for all three scenarios.
                        cols_dontagg = single_parent.max().abs().sort_values(ascending = False)[0:3].index
                        df_dontagg = single_parent[cols_dontagg]
                        df_toagg = single_parent.drop(columns = cols_dontagg)
                        agged = df_toagg.sum(axis = 1)
                        agged_flows = agged.copy()
                        df_dontagg['Other'] = agged_flows.copy()
                        single_parent = df_dontagg.copy()

                    for column in single_parent.columns:

                        mfunc.create_line_plot(axs,single_parent,column,label=column,n=n)
                        axs[n].set_title(parent)
                        axs[n].yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
                        axs[n].margins(x=0.01)
                        mfunc.set_plot_timeseries_format(axs,n)
                        axs[n].hlines(y = 0, xmin = axs[n].get_xlim()[0], xmax = axs[n].get_xlim()[1], linestyle = ':') #Add horizontal line at 0.
                        axs[n].legend(loc='lower left',bbox_to_anchor=(1,0),facecolor='inherit', frameon=True)
                    n+=1

            #Remove extra axes
            if excess_axs != 0:
                while excess_axs > 0:
                    fig3.delaxes(axs[(grid_size)-excess_axs])
                    excess_axs-=1

            # if plotting all scenarios add facet labels
            if plot_scenario == True:
                all_axes = fig3.get_axes()
                j=0
                k=0
                for ax in all_axes:
                    if ax.is_last_row():
                        ax.set_xlabel(xlabel=(self.xlabels[j]),  color='black', fontsize=16)
                        j=j+1
                    if ax.is_first_col():
                        ax.set_ylabel(ylabel=(self.ylabels[k]),  color='black', rotation='vertical', fontsize=16)
                        k=k+1

            fig3.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            plt.xlabel('Date ' + '(' + self.timezone + ')',  color='black', rotation='horizontal',labelpad = 60)
            plt.ylabel('Net Interchange (MW)',  color='black', rotation='vertical', labelpad = 60)

            # If plotting all regions save output and return none plot_main
            if plot_scenario == False:
                # Location to save to
                Data_Table_Out = rr_int_agg
                save_figures = os.path.join(self.figure_folder, self.AGG_BY + '_transmission')
                fig3.savefig(os.path.join(save_figures, "Region_Region_Interchange_{}.svg".format(self.Scenario_name)), dpi=600, bbox_inches='tight')
                Data_Table_Out.to_csv(os.path.join(save_figures, "Region_Region_Interchange_{}.csv".format(self.Scenario_name)))
                outputs = mfunc.DataSavedInModule()
                return outputs

            # if plotting all scenarios return figures to plot_main
            outputs[zone_input] = {'fig': fig3,'data_table':Data_Out}
        return outputs

    def region_region_checkerboard(self):

        """
        This method creates a checkerboard/heatmap figure showing total interchanges between regions
        Each sceanrio is plotted on its own facet plot
        Figures and data tables are saved within method
        """
        outputs = {}
        net_interchange_collection = {}
        check_input_data = []
        if self.AGG_BY=='zone':
            check_input_data.extend([mfunc.get_data(net_interchange_collection,"zone_zones_Net_Interchange",self.Marmot_Solutions_folder, self.Multi_Scenario)])
        else:
            check_input_data.extend([mfunc.get_data(net_interchange_collection,"region_regions_Net_Interchange",self.Marmot_Solutions_folder, self.Multi_Scenario)])
        
        if 1 in check_input_data:
            outputs = mfunc.MissingInputData()
            return outputs

        xdimension,ydimension = mfunc.set_x_y_dimension(len(self.Multi_Scenario))
        grid_size = xdimension*ydimension
        excess_axs = grid_size - len(self.Multi_Scenario)

        fig4, axs = mfunc.setup_plot(xdimension,ydimension,sharey=False)
        plt.subplots_adjust(wspace=0.02, hspace=0.4)
        max_flow_group = []
        Data_Out = []
        n=0
        for scenario in self.Multi_Scenario:

            rr_int = net_interchange_collection.get(scenario)

            if self.AGG_BY != 'region' and self.AGG_BY != 'zone':
                    agg_region_mapping = self.Region_Mapping[['region',self.AGG_BY]].set_index('region').to_dict()[self.AGG_BY]
                    rr_int = rr_int.reset_index()
                    rr_int['parent'] = rr_int['parent'].map(agg_region_mapping)
                    rr_int['child']  = rr_int['child'].map(agg_region_mapping)

            rr_int_agg = rr_int.groupby(['parent','child'],as_index=True).sum()
            rr_int_agg.rename(columns = {0:'flow (MW)'}, inplace = True)
            rr_int_agg=rr_int_agg.loc[rr_int_agg['flow (MW)']>0.01] # Keep only positive flows
            rr_int_agg.sort_values(ascending=False,by='flow (MW)')
            rr_int_agg = rr_int_agg/1000 # MWh -> GWh

            data_out = rr_int_agg.copy()
            data_out.rename(columns={'flow (MW)':'{} flow (GWh)'.format(scenario)},inplace=True)
            max_flow = max(rr_int_agg['flow (MW)'])
            rr_int_agg = rr_int_agg.unstack('child')
            rr_int_agg = rr_int_agg.droplevel(level = 0, axis = 1)

            current_cmap = plt.cm.get_cmap()
            current_cmap.set_bad(color='grey')

            axs[n].imshow(rr_int_agg)
            axs[n].set_xticks(np.arange(rr_int_agg.shape[1]))
            axs[n].set_yticks(np.arange(rr_int_agg.shape[0]))
            axs[n].set_xticklabels(rr_int_agg.columns)
            axs[n].set_yticklabels(rr_int_agg.index)
            axs[n].set_title(scenario.replace('_',' '),fontweight='bold')

            # Rotate the tick labels and set their alignment.
            plt.setp(axs[n].get_xticklabels(), rotation=30, ha="right",
                 rotation_mode="anchor")

            axs[n].set_xticks(np.arange(rr_int_agg.shape[1]+1)-.5, minor=True) #Delineate the boxes and make room at top and bottom
            axs[n].set_yticks(np.arange(rr_int_agg.shape[0]+1)-.5, minor=True)
            axs[n].grid(which="minor", color="k", linestyle='-', linewidth=1)
            axs[n].tick_params(which="minor", bottom=False, left=False)

            max_flow_group.append(max_flow)
            Data_Out.append(data_out)
            n+=1

        #Remove extra axes
        if excess_axs != 0:
            while excess_axs > 0:
                fig4.delaxes(axs[(grid_size)-excess_axs])
                excess_axs-=1

        cmap = mpl.cm.inferno
        norm = mpl.colors.Normalize(vmin=0, vmax=max(max_flow_group))
        cax = plt.axes([0.90, 0.1, 0.035, 0.8])
        fig4.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cax, label='Total Net Interchange [GWh]')
        fig4.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.xlabel('To Region',  color='black', rotation='horizontal',labelpad = 40, fontsize=20)
        plt.ylabel('From Region', color='black', rotation='vertical', labelpad = 30, fontsize=20)

        Data_Table_Out = pd.concat(Data_Out,axis=1)
        save_figures = os.path.join(self.figure_folder, self.AGG_BY + '_transmission')
        fig4.savefig(os.path.join(save_figures, "region_region_checkerboard.svg"), dpi=600, bbox_inches='tight')
        Data_Table_Out.to_csv(os.path.join(save_figures, "region_region_checkerboard.csv"))

        outputs = mfunc.DataSavedInModule()
        return outputs


    def line_violations_totals(self):

        """
        This method creates a bar plot of total lineflow violations for each region.
        Each sceanrio is plotted on the same plot using vertical bars.
        Figures and data tables are returned to plot_main
        """
        outputs = self._violations(total_violations=True)
        return outputs

    def line_violations_timeseries(self):

        """
        This method creates a timeseries line plot of lineflow violations for each region.
        The magnitude of each violation is plotted on the y-axis
        Each sceanrio is plotted on the same plot.
        Figures and data tables are returned to plot_main
        """
        outputs = self._violations()
        return outputs

    def _violations(self,total_violations=None):
        
        outputs = {}
        line_violation_collection = {}
        check_input_data = []
        check_input_data.extend([mfunc.get_data(line_violation_collection,"line_Violation",self.Marmot_Solutions_folder, self.Multi_Scenario)])
        
        if 1 in check_input_data:
            outputs = mfunc.MissingInputData()
            return outputs
        
        for zone_input in self.Zones:
            self.logger.info('Zone = ' + zone_input)
            all_scenarios = pd.DataFrame()

            for scenario in self.Multi_Scenario:
                self.logger.info("Scenario = " + str(scenario))

                if self.AGG_BY == 'zone':
                    lines = self.meta.zone_lines()
                else:
                    lines = self.meta.region_lines()

                line_v = line_violation_collection.get(scenario)
                line_v = line_v.reset_index()

                viol = line_v.merge(lines,on = 'line_name',how = 'left')

                if self.AGG_BY == 'zone':
                    viol = viol.groupby(["timestamp", "zone"]).sum()
                else:
                    viol = viol.groupby(["timestamp", self.AGG_BY]).sum()

                one_zone = viol.xs(zone_input, level = self.AGG_BY)
                one_zone = one_zone.rename(columns = {0 : scenario})
                one_zone = one_zone.abs() #/ 1000 #We don't care the direction of the violation, convert MW -> GW.
                all_scenarios = pd.concat([all_scenarios,one_zone], axis = 1)

            all_scenarios.columns = all_scenarios.columns.str.replace('_',' ')
            #remove columns that are all equal to 0
            all_scenarios = all_scenarios.loc[:, (all_scenarios != 0).any(axis=0)]
            Data_Table_Out = all_scenarios

            if all_scenarios.empty==True:
                out = mfunc.MissingZoneData()
                outputs[zone_input] = out
                continue

            #Make scenario/color dictionary.
            color_dict = dict(zip(all_scenarios.columns,self.color_list))

            fig5, axs = mfunc.setup_plot()

            n=0
            if total_violations==True:
                all_scenarios_tot = all_scenarios.sum()
                all_scenarios_tot.plot.bar(stacked=False, rot=0,
                                                        color=[color_dict.get(x, '#333333') for x in all_scenarios_tot.index],
                                                       linewidth='0.1', width=0.35, ax=axs[n])
                axs[n].spines['right'].set_visible(False)
                axs[n].spines['top'].set_visible(False)
                axs[n].tick_params(axis='y', which='major', length=5, width=1)
                axs[n].tick_params(axis='x', which='major', length=5, width=1)

            else:
                for column in all_scenarios:
                    mfunc.create_line_plot(axs,all_scenarios,column,color_dict=color_dict,label=column)
                axs[n].yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
                axs[n].margins(x=0.01)
                mfunc.set_plot_timeseries_format(axs,minticks=6,maxticks=12)
                axs[n].set_xlabel('Date ' + '(' + self.timezone + ')',  color='black', rotation='horizontal')
                handles, labels = axs[n].get_legend_handles_labels()
                axs[n].legend(handles, labels, loc='best',facecolor='inherit', frameon=True)

            axs[n].set_ylabel('Line violations (MW)',  color='black', rotation='vertical')

            outputs[zone_input] = {'fig': fig5,'data_table':Data_Table_Out}

        return outputs


    ### Archived Code ####

    # def line_util_agged(self):

    #     line_flow_collection = {}
    #     interface_flow_collection = {}
    #     line_limit_collection = {}
    #     interface_limit_collection = {}

    #     #Load data
    #     self._getdata(line_flow_collection,"line_Flow")
    #     self._getdata(interface_flow_collection,"interface_Flow")
    #     self._getdata(line_limit_collection,"line_Export_Limit")
    #     self._getdata(interface_limit_collection,"interface_Export_Limit")

    #     outputs = {}

    #     for zone_input in self.Zones:
    #         self.logger.info('Zone = ' + str(zone_input))

    #         all_scenarios = pd.DataFrame()

    #         for scenario in self.Multi_Scenario:
    #             self.logger.info("Scenario = " + str(scenario))

    #             lineflow = line_flow_collection.get(scenario)
    #             linelim = line_limit_collection.get(scenario)
    #             linelim = linelim.reset_index().drop('timestamp', axis = 1).set_index('line_name')

    #             #Calculate utilization.
    #             alllines = pd.merge(lineflow,linelim,left_index=True,right_index=True)
    #             alllines = alllines.rename(columns = {'0_x':'flow','0_y':'capacity'})
    #             alllines['flow'] = abs(alllines['flow'])

    #             #Merge in region ID and aggregate by region.
    #             if self.AGG_BY == 'region':
    #                 line2region = self.region_lines
    #             elif self.AGG_BY == 'zone':
    #                 line2region = self.zone_lines

    #             alllines = alllines.reset_index()
    #             alllines = alllines.merge(line2region, on = 'line_name')

    #             #Extract ReEDS expansion lines for next section.
    #             reeds_exp = alllines.merge((self.lines.rename(columns={"name":"line_name"})), on = 'line_name')
    #             reeds_exp = reeds_exp[reeds_exp['category'] == 'ReEDS_Expansion']
    #             reeds_agg = reeds_exp.groupby(['timestamp',self.AGG_BY],as_index = False).sum()

    #             #Subset only enforced lines. This will subset to only EI lines.
    #             enforced = pd.read_csv('/projects/continental/pcm/Results/enforced_lines.csv')
    #             enforced.columns = ['line_name']
    #             lineutil = pd.merge(enforced,alllines, on = 'line_name')

    #             lineutil['line util'] = lineutil['flow'] / lineutil['capacity']
    #             lineutil = lineutil[lineutil.capacity < 10000]

    #             #Drop duplicates if AGG_BY == Interconnection.

    #             #Aggregate by region, merge in region mapping.
    #             agg = alllines.groupby(['timestamp',self.AGG_BY],as_index = False).sum()
    #             agg['util'] = agg['flow'] / agg['capacity']
    #             agg = agg.rename(columns = {'util' : scenario})
    #             onezone = agg[agg[self.AGG_BY] == zone_input]
    #             onezone = onezone.set_index('timestamp')[scenario]

    #             #If zone_input is in WI or ERCOT, the dataframe will be empty here. Lines are not enforced here. Instead, use interfaces.
    #             if onezone.empty:

    #                 #Start with interface flow.
    #                 intflow = interface_flow_collection.get(scenario)
    #                 intlim = interface_limit_collection.get(scenario)

    #                 allint = pd.merge(intflow,intlim,left_index=True,right_index=True)
    #                 allint = allint.rename(columns = {'0_x':'flow','0_y':'capacity'})
    #                 allint = allint.reset_index()

    #                 #Merge in interface/line/region mapping.
    #                 line2int = self.interface_lines
    #                 line2int = line2int.rename(columns = {'line' : 'line_name','interface' : 'interface_name'})
    #                 allint = allint.merge(line2int, on = 'interface_name', how = 'inner')
    #                 allint = allint.merge(line2region, on = 'line_name')
    #                 allint = allint.merge(self.Region_Mapping, on = 'region')
    #                 allint = allint.drop(columns = 'line_name')
    #                 allint = allint.drop_duplicates() #Merging in line info duplicated most of the interfaces.

    #                 agg = allint.groupby(['timestamp',self.AGG_BY],as_index = False).sum()
    #                 agg = pd.concat([agg,reeds_agg]) #Add in ReEDS expansion lines, re-aggregate.
    #                 agg = agg.groupby(['timestamp',self.AGG_BY],as_index = False).sum()
    #                 agg['util'] = agg['flow'] / agg['capacity']
    #                 agg = agg.rename(columns = {'util' : scenario})
    #                 onezone = agg[agg[self.AGG_BY] == zone_input]
    #                 onezone = onezone.set_index('timestamp')[scenario]

    #             if (self.prop != self.prop) == False: #Show only subset category of lines.
    #                 reeds_agg = reeds_agg.groupby('timestamp',as_index = False).sum()
    #                 reeds_agg['util'] = reeds_agg['flow'] / reeds_agg['capacity']
    #                 onezone = reeds_agg.rename(columns = {'util' : scenario})
    #                 onezone = onezone.set_index('timestamp')[scenario]

    #             all_scenarios = pd.concat([all_scenarios,onezone], axis = 1)

    #         # Data table of values to return to main program
    #         Data_Table_Out = all_scenarios.copy()

    #         #Make scenario/color dictionary.
    #         scenario_color_dict = {}
    #         for idx,column in enumerate(all_scenarios.columns):
    #             dictionary = {column : self.color_list[idx]}
    #             scenario_color_dict.update(dictionary)

    #         all_scenarios.index = pd.to_datetime(all_scenarios.index)
    #         if all_scenarios.empty == False and '2008' not in self.Marmot_Solutions_folder and '2012' not in self.Marmot_Solutions_folder and all_scenarios.index[0] > dt.datetime(2024,2,28,0,0):
    #             all_scenarios.index = all_scenarios.index.shift(1,freq = 'D') #TO DEAL WITH LEAP DAYS, SPECIFIC TO MARTY'S PROJECT, REMOVE AFTER.

    #         fig5, ax = plt.subplots(figsize=(9,6))
    #         for idx,column in enumerate(all_scenarios.columns):
    #             ax.plot(all_scenarios.index.values,all_scenarios[column], linewidth=2, color = scenario_color_dict.get(column,'#333333'),label=column)

    #         ax.set_ylabel('Transmission utilization (%)',  color='black', rotation='vertical')
    #         ax.set_xlabel('Date ' + '(' + self.timezone + ')',  color='black', rotation='horizontal')
    #         ax.spines['right'].set_visible(False)
    #         ax.spines['top'].set_visible(False)
    #         ax.tick_params(axis='y', which='major', length=5, width=1)
    #         ax.tick_params(axis='x', which='major', length=5, width=1)
    #         ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    #         ax.margins(x=0.01)

    #         locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
    #         formatter = mdates.ConciseDateFormatter(locator)
    #         formatter.formats[2] = '%d\n %b'
    #         formatter.zero_formats[1] = '%b\n %Y'
    #         formatter.zero_formats[2] = '%d\n %b'
    #         formatter.zero_formats[3] = '%H:%M\n %d-%b'
    #         formatter.offset_formats[3] = '%b %Y'
    #         formatter.show_offset = False
    #         ax.xaxis.set_major_locator(locator)
    #         ax.xaxis.set_major_formatter(formatter)
    #         ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    #         ax.hlines(y = 1, xmin = ax.get_xlim()[0], xmax = ax.get_xlim()[1], linestyle = ':') #Add horizontal line at 100%.

    #         handles, labels = ax.get_legend_handles_labels()

    #         #Legend 1
    #         leg1 = ax.legend(reversed(handles), reversed(labels), loc='best',facecolor='inherit', frameon=True)

    #         # Manually add the first legend back
    #         ax.add_artist(leg1)

    #         outputs[zone_input] = {'fig': fig5, 'data_table': Data_Table_Out}
    #     return outputs


#     def region_region_duration(self):

#         rr_int = pd.read_hdf(os.path.join(self.Marmot_Solutions_folder,self.Multi_Scenario[0],"Processed_HDF5_folder", self.Multi_Scenario[0] + "_formatted.h5"),"region_regions_Net_Interchange")
#         agg_region_mapping = self.Region_Mapping[['region',self.AGG_BY]].set_index('region').to_dict()[self.AGG_BY]

#         rr_int = rr_int.reset_index()
#         rr_int['parent'] = rr_int['parent'].map(agg_region_mapping)
#         rr_int['child']  = rr_int['child'].map(agg_region_mapping)

#         rr_int_agg = rr_int.groupby(['parent','child'],as_index=True).sum() # Determine annual net flow between regions.
#         rr_int_agg.rename(columns = {0:'flow (MW)'}, inplace = True)
#         rr_int_agg = rr_int_agg.reset_index(['parent','child'])
#         rr_int_agg=rr_int_agg.loc[rr_int_agg['flow (MW)']>0.01] # Keep only positive flows
# #        rr_int_agg.set_index(['parent','child'],inplace=True)

#         rr_int_agg['path']=rr_int_agg['parent']+"_"+rr_int_agg['child']

#         if (self.prop!=self.prop)==False: # This checks for a nan in string. If a number of paths is selected only plot those
#             pathlist=rr_int_agg.sort_values(ascending=False,by='flow (MW)')['path'][1:int(self.prop)+1] #list of top paths based on number selected
#         else:
#             pathlist=rr_int_agg['path'] #List of paths


#         rr_int_hr = rr_int.groupby(['timestamp','parent','child'],as_index=True).sum() # Hourly flow
#         rr_int_hr.rename(columns = {0:'flow (MW)'}, inplace = True)
#         rr_int_hr.reset_index(['timestamp','parent','child'],inplace=True)
#         rr_int_hr['path']=rr_int_hr['parent']+"_"+rr_int_hr['child']
#         rr_int_hr.set_index(['path'],inplace=True)
#         rr_int_hr['Abs MW']=abs(rr_int_hr['flow (MW)'])
#         rr_int_hr['Abs MW'].sum()
#         rr_int_hr.loc[pathlist]['Abs MW'].sum()*2  # Check that the sum of the absolute value of flows is the same. i.e. only redundant flows were eliminated.
#         rr_int_hr=rr_int_hr.loc[pathlist].drop(['Abs MW'],axis=1)

#         ## Plot duration curves
#         fig3, ax3 = plt.subplots(figsize=(9,6))
#         for i in pathlist:
#             duration_curve = rr_int_hr.loc[i].sort_values(ascending=False,by='flow (MW)').reset_index()
#             plt.plot(duration_curve['flow (MW)'],label=i)
#             del duration_curve

#         ax3.set_ylabel('flow MW',  color='black', rotation='vertical')
#         ax3.set_xlabel('Intervals',  color='black', rotation='horizontal')
#         ax3.spines['right'].set_visible(False)
#         ax3.spines['top'].set_visible(False)

# #        if (self.prop!=self.prop)==False: # This checks for a nan in string. If no limit selected, do nothing
#         ax3.legend(loc='best')
# #            plt.lim((0,int(self.prop)))

#         Data_Table_Out = rr_int_hr

        # return {'fig': fig3, 'data_table': Data_Table_Out}
