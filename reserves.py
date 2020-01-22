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
    def __init__(self, prop, start, end, timezone, hdf_out_folder, HDF5_output, 
                                     zone_input, AGG_BY, ordered_gen, PLEXOS_color_dict, 
                                     Multi_Scenario, Scenario_Diff, PLEXOS_Scenarios, ylabels, 
                                     xlabels, color_list, marker_style, gen_names_dict, pv_gen_cat, 
                                     re_gen_cat, vre_gen_cat, region):
        self.prop = prop
        self.start = start     
        self.end = end
        self.timezone = timezone
        self.hdf_out_folder = hdf_out_folder
        self.HDF5_output = HDF5_output
        self.zone_input =zone_input
        self.AGG_BY = AGG_BY
        self.ordered_gen = ordered_gen
        self.PLEXOS_color_dict = PLEXOS_color_dict
        self.Multi_Scenario = Multi_Scenario
        self.Scenario_Diff = Scenario_Diff
        self.PLEXOS_Scenarios = PLEXOS_Scenarios
        self.ylabels = ylabels
        self.xlabels = xlabels
        self.gen_names_dict = gen_names_dict
        self.re_gen_cat = re_gen_cat
        self.region = region
        
    def reserve_timeseries(self):
        
        print("     "+ self.region)
        
        Reserve_Provision = pd.read_hdf(self.hdf_out_folder + "/" + self.HDF5_output,  "reserve_generators_Provision")
        
        Reserve_Provision_Timeseries = Reserve_Provision.xs(self.region,level="Reserve_Region")          
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
        sp = ax.stackplot(Reserve_Provision_Timeseries.index.values, 
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
        
        return {'fig': fig1, 'data_table': Data_Table_Out}
    
    

    def reserve_timeseries_facet(self):
        
        print("     "+ self.region)
        
        # Create Dictionary to hold Datframes for each scenario 
        Reserve_Provision_Collection = {} 
         
        for scenario in self.Multi_Scenario:
             Reserve_Provision_Collection[scenario] = pd.read_hdf(self.PLEXOS_Scenarios + r"\\" + scenario + r"\Processed_HDF5_folder" + "/" + self.HDF5_output,  "reserve_generators_Provision")
             
        
        xdimension=len(self.xlabels)
        ydimension=len(self.ylabels)
        grid_size = xdimension*ydimension
       
        fig2, axs = plt.subplots(ydimension,xdimension, figsize=((4*xdimension),(4*ydimension)), sharey=True)
        plt.subplots_adjust(wspace=0.05, hspace=0.2)
        axs = axs.ravel()
        i=0
        
        for scenario in self.Multi_Scenario:
            print("     " + scenario)
            
            Reserve_Provision_Timeseries = Reserve_Provision_Collection.get(scenario)
            Reserve_Provision_Timeseries = Reserve_Provision_Timeseries.xs(self.region,level="Reserve_Region")          
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
            
            
            sp = axs[i].stackplot(Reserve_Provision_Timeseries.index.values, Reserve_Provision_Timeseries.values.T, labels=Reserve_Provision_Timeseries.columns, linewidth=5,
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
             
        return fig2
    
    