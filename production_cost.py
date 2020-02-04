# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 16:24:40 2019

@author: dlevie
"""

import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict
import matplotlib as mpl


#===============================================================================

def df_process_gen_inputs(df, self):
    df = df.reset_index()
    df = df.groupby(["timestamp", "tech"], as_index=False).sum()
    df.tech = df.tech.astype("category")
    df.tech.cat.set_categories(self.ordered_gen, inplace=True)
    df = df.sort_values(["tech"]) 
    df = df.pivot(index='timestamp', columns='tech', values=0)
    return df  



class mplot(object):
    def __init__(self, prop, start, end, timezone, hdf_out_folder, 
                                     zone_input, AGG_BY, ordered_gen, PLEXOS_color_dict, 
                                     Multi_Scenario, Scenario_Diff, PLEXOS_Scenarios, ylabels, 
                                     xlabels, color_list, marker_style, gen_names_dict, pv_gen_cat, 
                                     re_gen_cat, vre_gen_cat):
        self.prop = prop
        self.hdf_out_folder = hdf_out_folder
        self.zone_input =zone_input
        self.AGG_BY = AGG_BY
        self.ordered_gen = ordered_gen
        self.PLEXOS_color_dict = PLEXOS_color_dict
        self.Multi_Scenario = Multi_Scenario
        self.PLEXOS_Scenarios = PLEXOS_Scenarios
        self.color_list = color_list
        self.marker_style = marker_style
        self.gen_names_dict = gen_names_dict
        self.pv_gen_cat = pv_gen_cat
        self.re_gen_cat = re_gen_cat
        self.vre_gen_cat = vre_gen_cat
        
    def prod_cost(self):
        Total_Gen_Cost_Collection = {}
        Pool_Revenues_Collection = {}
        Reserve_Revenues_Collection = {}
        Installed_Capacity_Collection = {} 
        for scenario in self.Multi_Scenario:

            Total_Gen_Cost_Collection[scenario] = pd.read_hdf(self.PLEXOS_Scenarios + r"\\" + scenario + r"\Processed_HDF5_folder" + "/" + scenario+"_formatted.h5", "generator_Total_Generation_Cost")
            Pool_Revenues_Collection[scenario] = pd.read_hdf(self.PLEXOS_Scenarios + r"\\" + scenario + r"\Processed_HDF5_folder" + "/" + scenario+"_formatted.h5",  "generator_Pool_Revenue")
            Reserve_Revenues_Collection[scenario] = pd.read_hdf(self.PLEXOS_Scenarios + r"\\" + scenario + r"\Processed_HDF5_folder" + "/" + scenario+"_formatted.h5",  "generator_Reserves_Revenue")
            Installed_Capacity_Collection[scenario] = pd.read_hdf(self.PLEXOS_Scenarios + r"\\" + scenario + r"\Processed_HDF5_folder" + "/" + scenario+"_formatted.h5",  "generator_Installed_Capacity")
            

        Total_Systems_Cost_Out = pd.DataFrame()
        print(self.zone_input)
        
        for scenario in self.Multi_Scenario:
            print(scenario)
            Total_Systems_Cost = pd.DataFrame()
            
            Total_Installed_Capacity = Installed_Capacity_Collection.get(scenario)
            Total_Installed_Capacity = Total_Installed_Capacity.xs(self.zone_input,level=self.AGG_BY)
            Total_Installed_Capacity = df_process_gen_inputs(Total_Installed_Capacity, self)
            Total_Installed_Capacity.reset_index(drop=True, inplace=True)
            Total_Installed_Capacity = Total_Installed_Capacity.iloc[0]
            
            Total_Gen_Cost = Total_Gen_Cost_Collection.get(scenario)
            Total_Gen_Cost = Total_Gen_Cost.xs(self.zone_input,level=self.AGG_BY)
            Total_Gen_Cost = df_process_gen_inputs(Total_Gen_Cost, self)
            Total_Gen_Cost = Total_Gen_Cost.sum(axis=0)*-1
            Total_Gen_Cost = Total_Gen_Cost/Total_Installed_Capacity #Change to $/kW-year
            Total_Gen_Cost.rename("Total_Gen_Cost", inplace=True)
            
            Pool_Revenues = Pool_Revenues_Collection.get(scenario)
            Pool_Revenues = Pool_Revenues.xs(self.zone_input,level=self.AGG_BY)
            Pool_Revenues = df_process_gen_inputs(Pool_Revenues, self)
            Pool_Revenues = Pool_Revenues.sum(axis=0)
            Pool_Revenues = Pool_Revenues/Total_Installed_Capacity #Change to $/kW-year
            Pool_Revenues.rename("Energy_Revenues", inplace=True)
            
            Reserve_Revenues = Reserve_Revenues_Collection.get(scenario)
            Reserve_Revenues = Reserve_Revenues.xs(self.zone_input,level=self.AGG_BY)
            Reserve_Revenues = df_process_gen_inputs(Reserve_Revenues, self)
            Reserve_Revenues = Reserve_Revenues.sum(axis=0)
            Reserve_Revenues = Reserve_Revenues/Total_Installed_Capacity #Change to $/kW-year
            Reserve_Revenues.rename("Reserve_Revenues", inplace=True)
            
            Total_Systems_Cost = pd.concat([Total_Systems_Cost, Total_Gen_Cost, Pool_Revenues, Reserve_Revenues], axis=1, sort=False) 
        
            Total_Systems_Cost.columns = Total_Systems_Cost.columns.str.replace('_',' ')    
            Total_Systems_Cost = Total_Systems_Cost.sum(axis=0)
            Total_Systems_Cost = Total_Systems_Cost.rename(scenario)
            
            Total_Systems_Cost_Out = pd.concat([Total_Systems_Cost_Out, Total_Systems_Cost], axis=1, sort=False)
        
        Total_Systems_Cost_Out = Total_Systems_Cost_Out.T
        Total_Systems_Cost_Out.index = Total_Systems_Cost_Out.index.str.replace('_',' ')   
    #    Total_Systems_Cost_Out.index = Total_Systems_Cost_Out.index.str.wrap(8)
        
        
        Total_Systems_Cost_Out["Total Gen Cost"] = Total_Systems_Cost_Out["Total Gen Cost"]*1000
        Total_Systems_Cost_Out = Total_Systems_Cost_Out/1000
        Net_Revenue = Total_Systems_Cost_Out.sum(axis=1)
        
        # Data table of values to return to main program
        Data_Table_Out = Total_Systems_Cost_Out
        
        # names = list(Net_Revenue.index)
        # values = list(Net_Revenue.values)
        
        fig1, ax = plt.subplots(figsize=(9,6))
        
        net_rev = plt.plot(Net_Revenue.index, Net_Revenue.values, color='black', linestyle='None', marker='o')
        sb = Total_Systems_Cost_Out.plot.bar(stacked=True, rot=0, edgecolor='black', linewidth='0.1', ax=ax)
        
        
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylabel('Total System Net Rev, Rev, & Cost ($/KW-yr)',  color='black', rotation='vertical')
    #    ax.set_xticklabels(rotation='vertical')
        ax.tick_params(axis='y', which='major', length=5, width=1)
        ax.tick_params(axis='x', which='major', length=5, width=1)
        ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.2g}'))
        ax.margins(x=0.01)
        plt.xticks(rotation=90)
        
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(reversed(handles), reversed(labels), loc='upper center',bbox_to_anchor=(0.5,-0.15), 
                     facecolor='inherit', frameon=True, ncol=3)
    
        #Legend 1
        leg1 = ax.legend(reversed(handles), reversed(labels), loc='lower left',bbox_to_anchor=(1,0), 
                      facecolor='inherit', frameon=True)  
        #Legend 2
        leg2 = ax.legend(net_rev, ['Net Revenue'], loc='center left',bbox_to_anchor=(1, 0.9), 
                      facecolor='inherit', frameon=True)
        
        # Manually add the first legend back
        ax.add_artist(leg1)
        
        return {'fig': fig1, 'data_table': Data_Table_Out}
    
    
    def sys_cost(self):
        Total_Gen_Cost_Collection = {}
        Cost_Unserved_Energy_Collection = {}
        for scenario in self.Multi_Scenario:

            Total_Gen_Cost_Collection[scenario] = pd.read_hdf(self.PLEXOS_Scenarios + r"\\" + scenario + r"\Processed_HDF5_folder" + "/" + scenario+"_formatted.h5", "generator_Total_Generation_Cost")
            Cost_Unserved_Energy_Collection[scenario] = pd.read_hdf(self.PLEXOS_Scenarios + r"\\" + scenario + r"\Processed_HDF5_folder" + "/" + scenario+"_formatted.h5", "region_Cost_Unserved_Energy")

        Total_Systems_Cost_Out = pd.DataFrame()
        print(self.zone_input)
        
        for scenario in self.Multi_Scenario:
            print(scenario)
            Total_Systems_Cost = pd.DataFrame()
            
            Total_Gen_Cost = Total_Gen_Cost_Collection.get(scenario)
            Total_Gen_Cost = Total_Gen_Cost.xs(self.zone_input,level=self.AGG_BY)
            Total_Gen_Cost = Total_Gen_Cost.sum(axis=0)
            Total_Gen_Cost.rename("Total_Gen_Cost", inplace=True)
            
            Cost_Unserved_Energy = Cost_Unserved_Energy_Collection.get(scenario)
            Cost_Unserved_Energy = Cost_Unserved_Energy.xs(self.zone_input,level=self.AGG_BY)
            Cost_Unserved_Energy = Cost_Unserved_Energy.sum(axis=0)
            Cost_Unserved_Energy.rename("Cost_Unserved_Energy", inplace=True)
            
            Total_Systems_Cost = pd.concat([Total_Systems_Cost, Total_Gen_Cost, Cost_Unserved_Energy], axis=1, sort=False) 
            
            Total_Systems_Cost.columns = Total_Systems_Cost.columns.str.replace('_',' ')    
            Total_Systems_Cost.rename({0:scenario}, axis='index', inplace=True)
            
            
            Total_Systems_Cost_Out = pd.concat([Total_Systems_Cost_Out, Total_Systems_Cost], axis=0, sort=False)
        
        Total_Systems_Cost_Out["Total Gen Cost"] = Total_Systems_Cost_Out["Total Gen Cost"]
        Total_Systems_Cost_Out = Total_Systems_Cost_Out/1000000 #Convert cost to millions

        Total_Systems_Cost_Out.index = Total_Systems_Cost_Out.index.str.replace('_',' ')  
        Total_Systems_Cost_Out.index = Total_Systems_Cost_Out.index.str.wrap(10, break_long_words=False)
        
        # Data table of values to return to main program
        Data_Table_Out = Total_Systems_Cost_Out
        
        fig2, ax = plt.subplots(figsize=(9,6))
        
        sb = Total_Systems_Cost_Out.plot.bar(stacked=True, rot=0, edgecolor='black', linewidth='0.1', ax=ax)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylabel('Total System Cost (Million $)',  color='black', rotation='vertical')
        ax.tick_params(axis='y', which='major', length=5, width=1)
        ax.tick_params(axis='x', which='major', length=5, width=1)
        ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.2g}'))
        ax.margins(x=0.01)
       
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(reversed(handles), reversed(labels), loc='upper center',bbox_to_anchor=(0.5,-0.15), 
                     facecolor='inherit', frameon=True, ncol=2)
    
    
        """adds annotations to bar plots"""
        cost_values=[]  #holds cost of each stack
        cost_totals=[]  #holds total of each bar
        
        for i in ax.patches:
            cost_values.append(i.get_height())

        #calculates total value of bar
        q=0    
        j = int(len(cost_values)/2)   #total number of bars in plot
        for cost in cost_values: 
            out = cost + cost_values[q+j]
            cost_totals.append(out)
            q=q+1
            if q>=j:
                break
            
        #inserts values into bar stacks
        for i in ax.patches:
           width, height = i.get_width(), i.get_height()
           if height<=1:
               continue
           x, y = i.get_xy() 
           ax.text(x+width/2, 
                y+height/2, 
                '{:,.0f}'.format(height), 
                horizontalalignment='center', 
                verticalalignment='center', fontsize=13)
       
        #inserts total bar value above each bar
        k=0   
        for i in ax.patches:
            height = cost_totals[k]
            width = 0.5
            x, y = i.get_xy() 
            ax.text(x+width/2, 
                y+height + 0.05*max(ax.get_ylim()), 
                '{:#,.2g}'.format(height),  
                horizontalalignment='center', 
                verticalalignment='center', fontsize=15, color='red') 
            k=k+1
            if k>=j:
                break
            
        return {'fig': fig2, 'data_table': Data_Table_Out}
            
