# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 15:59:45 2020

@author: dlevie
"""


import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
import os
import numpy as np


#===============================================================================

def df_process_gen_inputs(df, self):
    df = df.reset_index()
    df['tech'].replace(self.gen_names_dict, inplace=True)
    df = df.groupby(["timestamp", "tech"], as_index=False).sum()
    df.tech = df.tech.astype("category")
    df.tech.cat.set_categories(self.ordered_gen, inplace=True)
    df = df.sort_values(["tech"])
    df = df.pivot(index='timestamp', columns='tech', values=0)
    return df


class mplot(object):
    def __init__(self, argument_list):
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
        self.gen_names_dict = argument_list[18]
        self.re_gen_cat = argument_list[20]
        self.Reserve_Regions = argument_list[22]
        self.color_list = argument_list[16]
        self.Region_Mapping = argument_list[23]

    def reserve_timeseries(self):
        outputs = {}
        for region in self.Reserve_Regions:
            print("     "+ region)
            
            Reserve_Provision = pd.read_hdf(self.hdf_out_folder + "/" + self.Multi_Scenario[0]+"_formatted.h5",  "reserve_generators_Provision")
            
            Reserve_Provision_Timeseries = Reserve_Provision.xs(region,level="Reserve_Region")          
            Reserve_Provision_Timeseries = df_process_gen_inputs(Reserve_Provision_Timeseries, self)
        
            if self.prop == "Peak Demand":
                print("Plotting Peak Demand period")

                peak_reserve_t =  Reserve_Provision_Timeseries.sum(axis=1).idxmax()
                start_date = peak_reserve_t - dt.timedelta(days=self.start)
                end_date = peak_reserve_t + dt.timedelta(days=self.end)
                Reserve_Provision_Timeseries = Reserve_Provision_Timeseries[start_date : end_date]
                Peak_Reserve = Reserve_Provision_Timeseries.sum(axis=1)[peak_reserve_t]
             
            else:
                print("Plotting graph for entire timeperiod")
                
            # Data table of values to return to main program
            Data_Table_Out = Reserve_Provision_Timeseries
            
            fig1, ax = plt.subplots(figsize=(9,6))
            ax.stackplot(Reserve_Provision_Timeseries.index.values, 
                              Reserve_Provision_Timeseries.values.T, labels=Reserve_Provision_Timeseries.columns, 
                              linewidth=5, 
                              colors=[self.PLEXOS_color_dict.get(x, '#333333') for x in Reserve_Provision_Timeseries.T.index])
            
            
            ax.set_ylabel('Reserve Provision (MW)',  color='black', rotation='vertical')
            ax.set_xlabel('Date ' + '(' + self.timezone + ')',  color='black', rotation='horizontal')
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.tick_params(axis='y', which='major', length=5, width=1)
            ax.tick_params(axis='x', which='major', length=5, width=1)
            ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
            ax.margins(x=0.01)
            
            if self.prop == "Peak Demand":
                ax.annotate('Peak Reserve: \n' + str(format(int(Peak_Reserve), ',')) + ' MW', xy=(peak_reserve_t, Peak_Reserve), 
                            xytext=((peak_reserve_t + dt.timedelta(days=0.25)), (Peak_Reserve + Peak_Reserve*0.05)),
                            fontsize=13, arrowprops=dict(facecolor='black', width=3, shrink=0.1))
            
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
            
            handles, labels = ax.get_legend_handles_labels()
        
     
            ax.legend(reversed(handles), reversed(labels), loc='lower left',bbox_to_anchor=(1,0), 
                          facecolor='inherit', frameon=True)  
            
            outputs[region] = {'fig': fig1, 'data_table': Data_Table_Out}
        return outputs
    

    def reserve_timeseries_facet(self):
        outputs = {}
        for region in self.Reserve_Regions:
            print("     "+ region)
            
            # Create Dictionary to hold Datframes for each scenario 
            Reserve_Provision_Collection = {} 
             
            for scenario in self.Multi_Scenario:
                 Reserve_Provision_Collection[scenario] = pd.read_hdf(self.PLEXOS_Scenarios + r"\\" + scenario + r"\Processed_HDF5_folder" + "/" + scenario+"_formatted.h5",  "reserve_generators_Provision")
                 
            
            xdimension=len(self.xlabels)
            ydimension=len(self.ylabels)
            grid_size = xdimension*ydimension
           
            fig2, axs = plt.subplots(ydimension,xdimension, figsize=((4*xdimension),(4*ydimension)), sharey=True)
            plt.subplots_adjust(wspace=0.05, hspace=0.2)
            axs = axs.ravel()
            i=0
            
            Data_Out = pd.DataFrame()
            for scenario in self.Multi_Scenario:
                print("     " + scenario)
                
                Reserve_Provision_Timeseries = Reserve_Provision_Collection.get(scenario)
                Reserve_Provision_Timeseries = Reserve_Provision_Timeseries.xs(region,level="Reserve_Region")          
                Reserve_Provision_Timeseries = df_process_gen_inputs(Reserve_Provision_Timeseries, self)
            
                if self.prop == "Peak Demand":
                    print("Plotting Peak Demand period")
                    
                    peak_reserve_t =  Reserve_Provision_Timeseries.sum(axis=1).idxmax()
                    start_date = peak_reserve_t - dt.timedelta(days=self.start)
                    end_date = peak_reserve_t + dt.timedelta(days=self.end)
                    Reserve_Provision_Timeseries = Reserve_Provision_Timeseries[start_date : end_date]
                    Peak_Reserve = Reserve_Provision_Timeseries.sum(axis=1)[peak_reserve_t]
                
                else:
                    print("Plotting graph for entire timeperiod")
                
                Reserve_Provision_Timeseries.rename(columns={0:scenario},inplace=True)
                Data_Out=pd.concat([Data_Out,Reserve_Provision_Timeseries],axis=1)
                
                
                axs[i].stackplot(Reserve_Provision_Timeseries.index.values, Reserve_Provision_Timeseries.values.T, labels=Reserve_Provision_Timeseries.columns, linewidth=5,
                             colors=[self.PLEXOS_color_dict.get(x, '#333333') for x in Reserve_Provision_Timeseries.T.index])
                
                
                axs[i].spines['right'].set_visible(False)
                axs[i].spines['top'].set_visible(False)
                axs[i].tick_params(axis='y', which='major', length=5, width=1)
                axs[i].tick_params(axis='x', which='major', length=5, width=1)
                axs[i].yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
                axs[i].margins(x=0.01)
                
                if self.prop == "Peak Demand":
                    axs[i].annotate('Peak Reserve: \n' + str(format(int(Peak_Reserve), ',')) + ' MW', xy=(peak_reserve_t, Peak_Reserve), 
                            xytext=((peak_reserve_t + dt.timedelta(days=0.25)), (Peak_Reserve + Peak_Reserve*0.05)),
                            fontsize=13, arrowprops=dict(facecolor='black', width=3, shrink=0.1))
                
                locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
                formatter = mdates.ConciseDateFormatter(locator)
                formatter.formats[2] = '%d\n %b'
                formatter.zero_formats[1] = '%b\n %Y'
                formatter.zero_formats[2] = '%d\n %b'
                formatter.zero_formats[3] = '%H:%M\n %d-%b'
                formatter.offset_formats[3] = '%b %Y'
                formatter.show_offset = False
                axs[i].xaxis.set_major_locator(locator)
                axs[i].xaxis.set_major_formatter(formatter)
                
                handles, labels = axs[grid_size-1].get_legend_handles_labels()
                
                
                axs[grid_size-1].legend(reversed(handles), reversed(labels), loc='lower left',bbox_to_anchor=(1,0), 
                         facecolor='inherit', frameon=True)
                
                i=i+1
            
            all_axes = fig2.get_axes()
                
            self.xlabels = pd.Series(self.xlabels).str.replace('_',' ').str.wrap(10, break_long_words=False)
            
            j=0
            k=0
            for ax in all_axes:
                if ax.is_last_row():
                    ax.set_xlabel(xlabel=(self.xlabels[j]),  color='black')
                    j=j+1
                if ax.is_first_col(): 
                    ax.set_ylabel(ylabel=(self.ylabels[k]),  color='black', rotation='vertical')
                    k=k+1
            
            fig2.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            plt.ylabel('Reserve Provision (MW)',  color='black', rotation='vertical', labelpad=60)
                 
            outputs[region] = {'fig': fig2, 'data table': Data_Out}
        return outputs
        
    def reg_reserve_shortage(self):
        
        outputs = {}
        for region in self.Reserve_Regions:
            print("     "+ region)
            
            Reserve_Shortage_Collection = {}
            Data_Table_Out=pd.DataFrame()
    
            for scenario in self.Multi_Scenario:
                Reserve_Shortage_Collection[scenario] = pd.read_hdf(os.path.join(self.PLEXOS_Scenarios, scenario, "Processed_HDF5_folder", scenario + "_formatted.h5"), "reserve_Shortage")
                
            fig2, ax2 = plt.subplots(1,len(self.Multi_Scenario),figsize=(len(self.Multi_Scenario)*4,4),sharey=True)
    
            n=0 #Counter for scenario subplots
            for scenario in self.Multi_Scenario:
                
                print('Scenario = ' + scenario)
                
                reserve_short_timeseries = Reserve_Shortage_Collection.get(scenario)
                reserve_short_timeseries = reserve_short_timeseries.xs(region,level="Reserve_Region")
                timestamps=reserve_short_timeseries.reset_index(['timestamp'])['timestamp']
                time_delta = timestamps.iloc[1]- timestamps.iloc[0]                # Calculates interval step to correct for MWh of generation
                interval_count = 60/(time_delta/np.timedelta64(1, 'm'))            # Finds intervals in 60 minute period
                print("Identified timestep was: "+str(time_delta))
    
                reserve_short_total = reserve_short_timeseries.groupby(["Type"]).sum()/interval_count
    
        
                               
                if len(self.Multi_Scenario)>1:
                    ax2[n].bar(reserve_short_total.index,reserve_short_total[0])
                    ax2[n].set_ylabel(scenario,  color='black', rotation='vertical')
                    ax2[n].spines['right'].set_visible(False)
                    ax2[n].spines['top'].set_visible(False)
                    ax2[n].tick_params(axis='y', which='major', length=5, width=1)
                    ax2[n].tick_params(axis='x', which='major', length=5, width=1)
                    ax2[n].yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
                    ax2[n].margins(x=0.1)
                else:
                    ax2.bar(reserve_short_total.index,reserve_short_total[0])
                    ax2.set_ylabel(scenario,color='black',rotation='vertical')
                    ax2.spines['right'].set_visible(False)
                    ax2.spines['top'].set_visible(False)
                    ax2.tick_params(axis='y', which='major', length=5, width=1)
                    ax2.tick_params(axis='x', which='major', length=5, width=1)
                    ax2.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
                    ax2.margins(x=0.1)
                n=n+1
                reserve_short_total.rename(columns={0:scenario},inplace=True)
                Data_Table_Out=pd.concat([Data_Table_Out,reserve_short_total],axis=1) 
            #End scenario loop
    
            fig2.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            plt.ylabel('Reserve Shortage [MWh]',  color='black', rotation='vertical', labelpad=60)
    
            outputs[region] = {'fig': fig2,'data_table': Data_Table_Out}
        return outputs
    
    def reg_reserve_provision(self):
       outputs = {}
       for region in self.Reserve_Regions:
            print("     "+ region)
            
            Reserve_Provision_Collection = {}
            Data_Table_Out=pd.DataFrame()
            for scenario in self.Multi_Scenario:
                Reserve_Provision_Collection[scenario] = pd.read_hdf(os.path.join(self.PLEXOS_Scenarios, scenario, "Processed_HDF5_folder", scenario + "_formatted.h5"), "reserve_Provision")
                
            fig2, ax2 = plt.subplots(1,len(self.Multi_Scenario),figsize=(len(self.Multi_Scenario)*4,4),sharey=True)
    
            n=0 #Counter for scenario subplots
            for scenario in self.Multi_Scenario:
                
                print('Scenario = ' + scenario)
                
                reserve_provision_timeseries = Reserve_Provision_Collection.get(scenario)
                reserve_provision_timeseries = reserve_provision_timeseries.xs(region,level="Reserve_Region")
                timestamps=reserve_provision_timeseries.reset_index(['timestamp'])['timestamp']
                time_delta = timestamps.iloc[1]- timestamps.iloc[0]                # Calculates interval step to correct for MWh of generation
                print("Identified timestep was: "+str(time_delta))
                interval_count = 60/(time_delta/np.timedelta64(1, 'm'))            # Finds intervals in 60 minute period
                reserve_provision_total = reserve_provision_timeseries.groupby(["Type"]).sum()/interval_count
    
                               
                if len(self.Multi_Scenario)>1:
                    ax2[n].bar(reserve_provision_total.index,reserve_provision_total[0])
                    ax2[n].set_ylabel(scenario,  color='black', rotation='vertical')
                    ax2[n].spines['right'].set_visible(False)
                    ax2[n].spines['top'].set_visible(False)
                    ax2[n].tick_params(axis='y', which='major', length=5, width=1)
                    ax2[n].tick_params(axis='x', which='major', length=5, width=1)
                    ax2[n].yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
                    ax2[n].margins(x=0.1)
                else:
                    ax2.bar(reserve_provision_total.index,reserve_provision_total[0])
                    ax2.set_ylabel(scenario,color='black',rotation='vertical')
                    ax2.spines['right'].set_visible(False)
                    ax2.spines['top'].set_visible(False)
                    ax2.tick_params(axis='y', which='major', length=5, width=1)
                    ax2.tick_params(axis='x', which='major', length=5, width=1)
                    ax2.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
                    ax2.margins(x=0.1)
                n=n+1
                reserve_provision_total.rename(columns={0:scenario},inplace=True)
                Data_Table_Out=pd.concat([Data_Table_Out,reserve_provision_total],axis=1)
            #End scenario loop
    
            fig2.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            plt.ylabel('Reserve Provision [MWh]',  color='black', rotation='vertical', labelpad=60)
    
            outputs[region] = {'fig': fig2,'data_table': Data_Table_Out}
       return outputs
    
    def reg_reserve_shortage_timeseries(self):
        outputs = {}
        for region in self.Reserve_Regions:
            print("     "+ self.region)
            
            Reserve_Shortage_Collection = {}
            
            for scenario in self.Multi_Scenario:
                Reserve_Shortage_Collection[scenario] = pd.read_hdf(os.path.join(self.PLEXOS_Scenarios, scenario, "Processed_HDF5_folder", scenario + "_formatted.h5"), "reserve_Shortage")
            
            fig2, ax2 = plt.subplots(1,len(self.Multi_Scenario),figsize=(len(self.Multi_Scenario)*4,4),sharey=True)
            Data_Out = pd.DataFrame()
            n=0 #Counter for scenario subplots
            for scenario in self.Multi_Scenario:
                
                print('Scenario = ' + scenario)
                
                reserve_short_timeseries = Reserve_Shortage_Collection.get(scenario)
                reserve_short_timeseries = reserve_short_timeseries.xs(region,level="Reserve_Region").reset_index().set_index('timestamp')
                               
                if len(self.Multi_Scenario)>1:
                    ax2[n].plot(reserve_short_timeseries[0])
                    ax2[n].set_ylabel(scenario,  color='black', rotation='vertical')
                    ax2[n].spines['right'].set_visible(False)
                    ax2[n].spines['top'].set_visible(False)
                    ax2[n].tick_params(axis='y', which='major', length=5, width=1)
                    ax2[n].yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
                    
                    locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
                    formatter = mdates.ConciseDateFormatter(locator)
                    formatter.formats[2] = '%d\n %b'
                    formatter.zero_formats[1] = '%b\n %Y'
                    formatter.zero_formats[2] = '%d\n %b'
                    formatter.zero_formats[3] = '%H:%M\n %d-%b'
                    formatter.offset_formats[3] = '%b %Y'
                    formatter.show_offset = False
                    ax2[n].xaxis.set_major_locator(locator)
                    ax2[n].xaxis.set_major_formatter(formatter)
                    
                else:
                    ax2.plot(reserve_short_timeseries[0])
                    ax2.set_ylabel(scenario,color='black',rotation='vertical')
                    ax2.spines['right'].set_visible(False)
                    ax2.spines['top'].set_visible(False)
                    ax2.tick_params(axis='y', which='major', length=5, width=1)
                    ax2.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
                    
                    locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
                    formatter = mdates.ConciseDateFormatter(locator)
                    formatter.formats[2] = '%d\n %b'
                    formatter.zero_formats[1] = '%b\n %Y'
                    formatter.zero_formats[2] = '%d\n %b'
                    formatter.zero_formats[3] = '%H:%M\n %d-%b'
                    formatter.offset_formats[3] = '%b %Y'
                    formatter.show_offset = False
                    ax2.xaxis.set_major_locator(locator)
                    ax2.xaxis.set_major_formatter(formatter)
                n=n+1
            reserve_short_timeseries.rename(columns={0:scenario},inplace=True)
            Data_Out=pd.concat([Data_Out,reserve_short_timeseries],axis=1)   
            #End scenario loop
    
            fig2.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            plt.ylabel('Reserve Shortage [MW]',  color='black', rotation='vertical', labelpad=60)
    
            outputs[region] =  {'fig': fig2, 'data_table': Data_Out}
        return outputs
    
    def reg_reserve_shortage_hrs(self):
       outputs = {}
       for region in self.Reserve_Regions:
            print("     "+ region)
            
            Reserve_Shortage_Collection = {}
    
            for scenario in self.Multi_Scenario:
                Reserve_Shortage_Collection[scenario] = pd.read_hdf(os.path.join(self.PLEXOS_Scenarios, scenario, "Processed_HDF5_folder", scenario + "_formatted.h5"), "reserve_Shortage")
                
            fig2, ax2 = plt.subplots(1,len(self.Multi_Scenario),figsize=(len(self.Multi_Scenario)*4,4),sharey=True)
            Data_Out=pd.DataFrame()
            n=0 #Counter for scenario subplots
            for scenario in self.Multi_Scenario:
                
                print('Scenario = ' + scenario)
                
                reserve_short_timeseries = Reserve_Shortage_Collection.get(scenario)
                reserve_short_timeseries = reserve_short_timeseries.xs(region,level="Reserve_Region")
                reserve_short_hrs = reserve_short_timeseries[reserve_short_timeseries[0]>0] #Filter for non zero values
                reserve_short_hrs = reserve_short_hrs.groupby("Type").count()
                               
                if len(self.Multi_Scenario)>1:
                    ax2[n].bar(reserve_short_hrs.index,reserve_short_hrs[0])
                    ax2[n].set_ylabel(scenario,  color='black', rotation='vertical')
                    ax2[n].spines['right'].set_visible(False)
                    ax2[n].spines['top'].set_visible(False)
                    ax2[n].tick_params(axis='y', which='major', length=5, width=1)
                    ax2[n].tick_params(axis='x', which='major', length=5, width=1)
                    ax2[n].yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
                    ax2[n].margins(x=0.1)
                else:
                    ax2.bar(reserve_short_hrs.index,reserve_short_hrs[0])
                    ax2.set_ylabel(scenario,color='black',rotation='vertical')
                    ax2.spines['right'].set_visible(False)
                    ax2.spines['top'].set_visible(False)
                    ax2.tick_params(axis='y', which='major', length=5, width=1)
                    ax2.tick_params(axis='x', which='major', length=5, width=1)
                    ax2.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
                    ax2.margins(x=0.1)
                n=n+1
            reserve_short_hrs.rename(columns={0:scenario},inplace=True)
            Data_Out=pd.concat([Data_Out,reserve_short_hrs],axis=1)    
            #End scenario loop
    
            fig2.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            plt.ylabel('Reserve Shortage Intervals',  color='black', rotation='vertical', labelpad=60)
    
            outputs[region] = {'fig': fig2,'data_table':Data_Out}
       return outputs
            
#    def tot_reserve_shortage(self):
#
#        print("     "+ self.zone_input)
#
#        Reserve_Shortage_Collection = {}
#
#        for scenario in self.Multi_Scenario:
#            Reserve_Shortage_Collection[scenario] = pd.read_hdf(os.path.join(self.Marmot_Solutions_folder, scenario, "Processed_HDF5_folder", scenario + "_formatted.h5"), "reserve_Shortage")
#
#        Reserve_Shortage_Timeseries_Out = pd.DataFrame()
#        Total_Reserve_Shortage_Out = pd.DataFrame()
#
#        for scenario in self.Multi_Scenario:
#
#            print('Scenario = ' + scenario)
#
#            reserve_short_timeseries = Reserve_Shortage_Collection.get(scenario)
#            rto_Mapping=self.Region_Mapping[['rto',self.AGG_BY]].drop_duplicates().reset_index().drop('index',axis=1)
#            reserve_short_timeseries = pd.merge(reserve_short_timeseries.reset_index(),rto_Mapping,left_on='Reserve_Region',right_on='rto')
#            reserve_short_timeseries = reserve_short_timeseries.reset_index().set_index(['timestamp','Reserve_Region','Type','rto',AGG_BY])
#            reserve_short_timeseries = reserve_short_timeseries.xs(self.zone_input,level=self.AGG_BY)
#            reserve_short_timeseries = reserve_short_timeseries.groupby(["timestamp"]).sum()
#            reserve_short_timeseries = reserve_short_timeseries.squeeze() #Convert to Series
#            reserve_short_timeseries.rename(scenario, inplace=True)
#            Reserve_Shortage_Timeseries_Out = pd.concat([Reserve_Shortage_Timeseries_Out, reserve_short_timeseries], axis=1, sort=False).fillna(0)
#
#        Reserve_Shortage_Timeseries_Out.columns = Reserve_Shortage_Timeseries_Out.columns.str.replace('_',' ')
#
#        Total_Reserve_Shortage_Out = Reserve_Shortage_Timeseries_Out.sum(axis=0)
#
#        Total_Reserve_Shortage_Out.index = Total_Reserve_Shortage_Out.index.str.replace('_',' ')
#        Total_Reserve_Shortage_Out.index = Total_Reserve_Shortage_Out.index.str.wrap(10, break_long_words=False)
#
#        # Data table of values to return to main program
#        Data_Table_Out = Total_Reserve_Shortage_Out
#
#        # Converts color_list into an iterable list for use in a loop
#        iter_colour = iter(self.color_list)
#
#        fig2, ax = plt.subplots(figsize=(9,6))
#
#        bp = Total_Reserve_Shortage_Out.plot.bar(stacked=False, rot=0, edgecolor='black',
#                                                color=next(iter_colour), linewidth='0.1',
#                                                width=0.35, ax=ax)
#
#        ax.set_ylabel('Total Reserve Shortage (MWh)',  color='black', rotation='vertical')
#        ax.spines['right'].set_visible(False)
#        ax.spines['top'].set_visible(False)
#        ax.tick_params(axis='y', which='major', length=5, width=1)
#        ax.tick_params(axis='x', which='major', length=5, width=1)
#        ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
#        ax.margins(x=0.01)
#
#        for i in ax.patches:
#           width, height = i.get_width(), i.get_height()
#           if height<=1:
#               continue
#           x, y = i.get_xy()
#           ax.text(x+width/2,
#                y+(height+100)/2,
#                '{:,.0f}'.format(height),
#                horizontalalignment='center',
#                verticalalignment='center', fontsize=13)
#
#        return {'fig': fig2, 'data_table': Data_Table_Out}
