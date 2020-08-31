# -*- coding: utf-8 -*-
"""
Created April 2020, updated August 2020 

This code creates transmission line and interface plots and is called from Marmot_plot_main.py

@author: dlevie
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
import numpy as np
import datetime as dt
import matplotlib.ticker as mtick
import math

#===============================================================================

class mplot(object):
    def __init__(self,argument_list):

        self.prop = argument_list[0]
        self.start = argument_list[1]
        self.end = argument_list[2]
        self.timezone = argument_list[3]
        self.start_date = argument_list[4]
        self.end_date = argument_list[5]
        self.hdf_out_folder = argument_list[6]
        self.Zones = argument_list[7]
        self.AGG_BY = argument_list[8]
        self.ordered_gen = argument_list[9]
        self.PLEXOS_color_dict = argument_list[10]
        self.Multi_Scenario = argument_list[11]
        self.Scenario_Diff = argument_list[12]
        self.Marmot_Solutions_folder = argument_list[13]
        self.ylabels = argument_list[14]
        self.xlabels = argument_list[15]
        self.color_list = argument_list[16]
        self.gen_names_dict = argument_list[18]
        self.re_gen_cat = argument_list[20]
        self.Region_Mapping = argument_list[24]
        self.figure_folder = argument_list[25]
        self.meta = argument_list[26]
        self.Scenario_name = argument_list[28]
    

    def net_export(self):
        
        """
        This method creates a net export line graph for each region.
        All scenarios are plotted on a single figure.
        Figures and data tables are returned to plot_main
        """        
        net_interchange_collection = {} 
        
        if self.AGG_BY=='zone':
            self._getdata(net_interchange_collection,"zone_Net_Interchange")
        else:
            self._getdata(net_interchange_collection,"region_Net_Interchange")
        
        outputs = {}
        for zone_input in self.Zones:
            print(self.AGG_BY + " = " + zone_input)
            
            net_export_all_scenarios = pd.DataFrame()
    
            for scenario in self.Multi_Scenario:
                
                print("Scenario = " + str(scenario))
                net_export_read = net_interchange_collection.get(scenario)
                net_export = net_export_read.xs(zone_input, level = self.AGG_BY)
                net_export = net_export.groupby("timestamp").sum()
                net_export.columns = [scenario]
                
                if self.prop == 'Date Range':
                    print("Plotting specific date range:")
                    print(str(self.start_date) + '  to  ' + str(self.end_date))
                    
                    net_export = net_export[self.start_date : self.end_date]
    
                net_export_all_scenarios = pd.concat([net_export_all_scenarios,net_export], axis = 1) 
                net_export_all_scenarios.columns = net_export_all_scenarios.columns.str.replace('_',' ')  
                
            # Data table of values to return to main program
            Data_Table_Out = net_export_all_scenarios.copy()
            #Make scenario/color dictionary.
            color_dict = dict(zip(net_export_all_scenarios.columns,self.color_list))
            
            # if '2008' not in self.Marmot_Solutions_folder and '2012' not in self.Marmot_Solutions_folder and Net_Export_all_scenarios.index[0] > dt.datetime(2024,2,28,0,0):
            #     Net_Export_all_scenarios.index = Net_Export_all_scenarios.index.shift(1,freq = 'D') #TO DEAL WITH LEAP DAYS, SPECIFIC TO MARTY'S PROJECT, REMOVE AFTER.
                
            fig1, axs = self._setup_plot()
            plt.subplots_adjust(wspace=0.05, hspace=0.2)
            
            n=0
            for column in net_export_all_scenarios:
                self._create_line_plot(axs,net_export_all_scenarios,column,color_dict)
                axs[n].set_ylabel('Net exports (MW)',  color='black', rotation='vertical')
                axs[n].set_xlabel('Date ' + '(' + self.timezone + ')',  color='black', rotation='horizontal')
                axs[n].yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
                axs[n].margins(x=0.01)
                axs[n].hlines(y = 0, xmin = axs[n].get_xlim()[0], xmax = axs[n].get_xlim()[1], linestyle = ':')
                self._set_plot_timeseries_format(axs,n)
        
            handles, labels = axs[n].get_legend_handles_labels()
            
            axs[n].legend(reversed(handles), reversed(labels), loc='best',facecolor='inherit', frameon=True)   
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
        
        self._getdata(Flow_Collection,"line_Flow")
        self._getdata(Limit_Collection,"line_Import_Limit")
        
        outputs = {}
        for zone_input in self.Zones:    
            print("For all lines touching Zone = "+zone_input)
            
            fig2, axs = self._setup_plot(ydimension=len(self.Multi_Scenario))
            plt.subplots_adjust(wspace=0.05, hspace=0.2)

         
            n=0 #Counter for scenario subplots
            Data_Out=pd.DataFrame()
                
            for scenario in self.Multi_Scenario:
                print("Scenario = " + str(scenario))
                
                # gets correct metadata based on area aggregation 
                if self.AGG_BY=='zone':
                    zone_lines = self.meta.zone_lines()
                else:
                    zone_lines = self.meta.region_lines()                
                try:
                    zone_lines = zone_lines.set_index([self.AGG_BY])
                except:
                    print("Column to Aggregate by is missing")
                    continue
                
                zone_lines = zone_lines.xs(zone_input)            
                zone_lines=zone_lines['line_name'].unique()
                                
                flow = Flow_Collection.get(scenario)
                flow = flow[flow.index.get_level_values('line_name').isin(zone_lines)] #Limit to only lines touching to this zone
                                
                limits = Limit_Collection.get(scenario).droplevel('timestamp')
                limits.mask(limits[0]==0.0,other=0.01,inplace=True) #if limit is zero set to small value
                
                # This checks for a nan in string. If no scenario selected, do nothing.
                if (self.prop != self.prop)==False: 
                    print("Line category = "+str(self.prop))
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
                    self._create_hist_plot(axs,annual_util,color_dict,label=scenario,n=n)
                    axs[n].set_ylabel(scenario.replace('_',' '),color='black', rotation='vertical')
                else:
                    for line in top_utilization.index.get_level_values(level='line_name').unique():
                        duration_curve = flow.xs(line,level="line_name").sort_values(by='Util',ascending=False).reset_index(drop=True)
                        self._create_line_plot(axs,duration_curve,'Util',label=line,n=n)
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
        
        net_interchange_collection = {}
        if self.AGG_BY=='zone':
            self._getdata(net_interchange_collection,"zone_zones_Net_Interchange",scenario_type=scenario_type)
        else:
            self._getdata(net_interchange_collection,"region_regions_Net_Interchange",scenario_type=scenario_type)
        
        outputs = {}
        for zone_input in self.Zones:
            
            xdimension=len(self.xlabels)
            ydimension=len(self.ylabels)
            if xdimension == 0 or ydimension == 0:
                xdimension, ydimension =  self._set_x_y_dimension(len(scenario_type))

            fig3, axs = self._setup_plot(xdimension,ydimension)
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
                    xdimension, ydimension =  self._set_x_y_dimension(plot_number)
                    fig3, axs = self._setup_plot(xdimension,ydimension,sharey=False) 
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
                        df_dontagg['Other'] = agged_flows
                        single_parent = df_dontagg.copy()              
                    
                    for column in single_parent.columns:
                        
                        self._create_line_plot(axs,single_parent,column,label=column,n=n)
                        axs[n].set_title(parent)
                        axs[n].yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
                        axs[n].margins(x=0.01)
                        self._set_plot_timeseries_format(axs,n)
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
        net_interchange_collection = {}
        if self.AGG_BY=='zone':
            self._getdata(net_interchange_collection,"zone_zones_Net_Interchange")
        else:
            self._getdata(net_interchange_collection,"region_regions_Net_Interchange")
        
        outputs = {} 
        
        xdimension,ydimension = self._set_x_y_dimension(len(self.Multi_Scenario))
        grid_size = xdimension*ydimension
        excess_axs = grid_size - len(self.Multi_Scenario)
        
        fig4, axs = self._setup_plot(xdimension,ydimension,sharey=False) 
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
        
        # For plot_main handeling - need to find better solution
        for zone_input in self.Zones: 
                        outputs[zone_input] = pd.DataFrame()
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
        line_violation_collection = {}
        self._getdata(line_violation_collection,"line_Violation")
        
        outputs = {} 
        for zone_input in self.Zones:
            print('Zone = ' + zone_input)
            all_scenarios = pd.DataFrame()
        
            for scenario in self.Multi_Scenario:
                print("Scenario = " + str(scenario))
    
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
                df = pd.DataFrame()
                outputs[zone_input] = df
                continue
    
            #Make scenario/color dictionary.
            color_dict = dict(zip(all_scenarios.columns,self.color_list))
            
            fig5, axs = self._setup_plot() 
            
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
                    self._create_line_plot(axs,all_scenarios,column,color_dict=color_dict,label=column)
                axs[n].yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
                axs[n].margins(x=0.01)
                self._set_plot_timeseries_format(axs,minticks=6,maxticks=12)
                axs[n].set_xlabel('Date ' + '(' + self.timezone + ')',  color='black', rotation='horizontal')    
                handles, labels = axs[n].get_legend_handles_labels()
                axs[n].legend(handles, labels, loc='best',facecolor='inherit', frameon=True)
                
            axs[n].set_ylabel('Line violations (MW)',  color='black', rotation='vertical')
            
            outputs[zone_input] = {'fig': fig5,'data_table':Data_Table_Out}

        return outputs

    
    def _getdata(self,data_collection,data,scenario_type=None):
        if scenario_type == None:
            scenario_type = self.Multi_Scenario
        for scenario in scenario_type:
            data_collection[scenario] = pd.read_hdf(os.path.join(self.Marmot_Solutions_folder, scenario,"Processed_HDF5_folder", scenario+ "_formatted.h5"),data)

    def _setup_plot(self,xdimension=1,ydimension=1,sharey=True):
        fig, axs = plt.subplots(ydimension,xdimension, figsize=((6*xdimension),(4*ydimension)), sharey=sharey, squeeze=False)
        axs = axs.ravel()
        return fig,axs
    
    
    def _create_line_plot(self,axs,data,column,color_dict=None,label=None,n=0):
        if color_dict==None:
            axs[n].plot(data[column], linewidth=1,label=label)
        else:
            axs[n].plot(data[column], linewidth=1, color=color_dict[column],label=column)
        axs[n].spines['right'].set_visible(False)
        axs[n].spines['top'].set_visible(False)  
        axs[n].tick_params(axis='y', which='major', length=5, width=1)
        axs[n].tick_params(axis='x', which='major', length=5, width=1)
        
    def _create_hist_plot(self,axs,data,color_dict,label=None,n=0):
        axs[n].hist(data,bins=20, range=(0,1), color=color_dict[label], zorder=2, rwidth=0.8,label=label)
        axs[n].spines['right'].set_visible(False)
        axs[n].spines['top'].set_visible(False)
        axs[n].tick_params(axis='y', which='major', length=5, width=1)
    
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
    
    def _set_plot_timeseries_format(self,axs,n=0,minticks=6, maxticks=8):
        locator = mdates.AutoDateLocator(minticks=minticks, maxticks=maxticks)
        formatter = mdates.ConciseDateFormatter(locator)
        formatter.formats[2] = '%d\n %b'
        formatter.zero_formats[1] = '%b\n %Y'
        formatter.zero_formats[2] = '%d\n %b'
        formatter.zero_formats[3] = '%H:%M\n %d-%b'
        formatter.offset_formats[3] = '%b %Y'
        formatter.show_offset = False
        axs[n].xaxis.set_major_locator(locator)
        axs[n].xaxis.set_major_formatter(formatter)
        
    
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
    #         print('Zone = ' + str(zone_input))
    
    #         all_scenarios = pd.DataFrame()
    
    #         for scenario in self.Multi_Scenario:
    #             print("Scenario = " + str(scenario))
                
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