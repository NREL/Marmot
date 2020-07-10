# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 16:24:40 2019

@author: dlevie
"""

import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict
import matplotlib as mpl
import os


#===============================================================================

def df_process_gen_inputs(df, self):
    df = df.reset_index()
    df = df.groupby(["timestamp", "tech"], as_index=False).sum()
    df.tech = df.tech.astype("category")
    df.tech.cat.set_categories(self.ordered_gen, inplace=True)
    df = df.sort_values(["tech"])
    df = df.pivot(index='timestamp', columns='tech', values=0)
    return df  

def df_process_categorical_index(df, self): 
    df=df
    df.index = df.index.astype("category")
    df.index = df.index.set_categories(self.ordered_gen)
    df = df.sort_index()
    return df   

class mplot(object):
    def __init__(self, argument_list):
        self.prop = argument_list[0]
        self.hdf_out_folder = argument_list[6]
        self.zone_input = argument_list[7]
        self.AGG_BY = argument_list[8]
        self.ordered_gen = argument_list[9]
        self.PLEXOS_color_dict = argument_list[10]
        self.Multi_Scenario = argument_list[11]
        self.Marmot_Solutions_folder = argument_list[13]
        self.xlabels = argument_list[15]
        self.color_list = argument_list[16]
        self.marker_style = argument_list[17]
        self.gen_names_dict = argument_list[18]
        self.pv_gen_cat = argument_list[19]
        self.re_gen_cat = argument_list[20]
        self.vre_gen_cat = argument_list[21]

    def prod_cost(self):
        Total_Gen_Cost_Collection = {}
        Pool_Revenues_Collection = {}
        Reserve_Revenues_Collection = {}
        Installed_Capacity_Collection = {}
        for scenario in self.Multi_Scenario:

            Total_Gen_Cost_Collection[scenario] = pd.read_hdf(os.path.join(self.Marmot_Solutions_folder, scenario, "Processed_HDF5_folder", scenario+"_formatted.h5"), "generator_Total_Generation_Cost")
            Pool_Revenues_Collection[scenario] = pd.read_hdf(os.path.join(self.Marmot_Solutions_folder, scenario, "Processed_HDF5_folder", scenario+"_formatted.h5"), "generator_Pool_Revenue")
            Reserve_Revenues_Collection[scenario] = pd.read_hdf(os.path.join(self.Marmot_Solutions_folder, scenario, "Processed_HDF5_folder", scenario+"_formatted.h5"),  "generator_Reserves_Revenue")
            Installed_Capacity_Collection[scenario] = pd.read_hdf(os.path.join(self.Marmot_Solutions_folder, scenario, "Processed_HDF5_folder", scenario+"_formatted.h5"),  "generator_Installed_Capacity")


        Total_Systems_Cost_Out = pd.DataFrame()
        print("Zone = "+ self.zone_input)

        for scenario in self.Multi_Scenario:
            print("Scenario = " + scenario)
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

            ### Might cvhnage to Net Reserve Revenue at later date
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
        Total_Systems_Cost_Out.index = Total_Systems_Cost_Out.index.str.wrap(10, break_long_words=False)

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

            Total_Gen_Cost_Collection[scenario] = pd.read_hdf(os.path.join(self.Marmot_Solutions_folder, scenario, "Processed_HDF5_folder", scenario+"_formatted.h5"), "generator_Total_Generation_Cost")
            # If data is to be agregated by zone, then zone properties are loaded, else region properties are loaded
            if self.AGG_BY == "zone":
                try:
                    Cost_Unserved_Energy_Collection[scenario] = pd.read_hdf(os.path.join(self.Marmot_Solutions_folder, scenario, "Processed_HDF5_folder", scenario+"_formatted.h5"), "zone_Cost_Unserved_Energy")
                except:
                    Cost_Unserved_Energy_Collection[scenario] = Total_Gen_Cost_Collection[scenario].copy()
                    Cost_Unserved_Energy_Collection[scenario].iloc[:,0] = 0
            else:
                try:
                    Cost_Unserved_Energy_Collection[scenario] = pd.read_hdf(os.path.join(self.Marmot_Solutions_folder, scenario, "Processed_HDF5_folder", scenario+"_formatted.h5"), "region_Cost_Unserved_Energy")
                except:
                    Cost_Unserved_Energy_Collection[scenario] = Total_Gen_Cost_Collection[scenario].copy()
                    Cost_Unserved_Energy_Collection[scenario].iloc[:,0] = 0

        Total_Systems_Cost_Out = pd.DataFrame()
        print("Zone = "+ self.zone_input)

        for scenario in self.Multi_Scenario:
            print("Scenario = " + scenario)
            Total_Systems_Cost = pd.DataFrame()
            Total_Gen_Cost = Total_Gen_Cost_Collection.get(scenario)

            try:
                Total_Gen_Cost = Total_Gen_Cost.xs(self.zone_input,level=self.AGG_BY)
            except KeyError:
                print("No Generators found for : "+self.zone_input)
                return pd.DataFrame()
            
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

        Total_Systems_Cost_Out = Total_Systems_Cost_Out/1000000 #Convert cost to millions

        Total_Systems_Cost_Out.index = Total_Systems_Cost_Out.index.str.replace('_',' ')
        Total_Systems_Cost_Out.index = Total_Systems_Cost_Out.index.str.wrap(10, break_long_words=False)

        # Data table of values to return to main program
        Data_Table_Out = Total_Systems_Cost_Out

        fig2, ax = plt.subplots(figsize=(9,6))

        sb = Total_Systems_Cost_Out.plot.bar(stacked=True, rot=0, edgecolor='black', linewidth='0.1', ax=ax)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylabel('Total Cost (Million $)',  color='black', rotation='vertical')
        ax.tick_params(axis='y', which='major', length=5, width=1)
        ax.tick_params(axis='x', which='major', length=5, width=1)
        ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.2g}'))
        ax.margins(x=0.01)

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(reversed(handles), reversed(labels), loc='upper center',bbox_to_anchor=(0.5,-0.15),
                     facecolor='inherit', frameon=True, ncol=2)


        """adds annotations to bar plots"""

        cost_totals = Total_Systems_Cost_Out.sum(axis=1) #holds total of each bar

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
                verticalalignment='center', fontsize=12)

        #inserts total bar value above each bar
        k=0
        for i in ax.patches:
            height = cost_totals[k]
            width = i.get_width()
            x, y = i.get_xy()
            ax.text(x+width/2,
                y+height + 0.05*max(ax.get_ylim()),
                '{:,.0f}'.format(height),
                horizontalalignment='center',
                verticalalignment='center', fontsize=15, color='red')
            k=k+1
            if k>=len(cost_totals):
                break

        return {'fig': fig2, 'data_table': Data_Table_Out}



    def detailed_production_cost(self):

        print("Zone = "+ self.zone_input)
        Detailed_Gen_Cost_Out = pd.DataFrame()

        for scenario in self.Multi_Scenario:
            print("Scenario = " + scenario)
            
            Gen = pd.read_hdf(os.path.join(self.Marmot_Solutions_folder, scenario, "Processed_HDF5_folder", scenario+"_formatted.h5"),  "generator_Generation")
            Gen = Gen.xs(self.zone_input,level=self.AGG_BY)
            Total_gen = Gen.sum().iloc[0] #Total energy produced, in MWh.

            Fuel_Cost = pd.read_hdf(os.path.join(self.Marmot_Solutions_folder, scenario, "Processed_HDF5_folder", scenario+"_formatted.h5"), "generator_Fuel_Cost")
            Fuel_Cost = Fuel_Cost.xs(self.zone_input,level=self.AGG_BY)
            Fuel_Cost = Fuel_Cost.sum(axis=0)
            Fuel_Cost.rename("Fuel_Cost", inplace=True)

            VOM_Cost = pd.read_hdf(os.path.join(self.Marmot_Solutions_folder, scenario, "Processed_HDF5_folder", scenario+"_formatted.h5"),  "generator_VO&M_Cost")
            VOM_Cost = VOM_Cost.xs(self.zone_input,level=self.AGG_BY)
            VOM_Cost = VOM_Cost.sum(axis=0)
            VOM_Cost.rename("VO&M_Cost", inplace=True)

            Start_Shutdown_Cost = pd.read_hdf(os.path.join(self.Marmot_Solutions_folder, scenario, "Processed_HDF5_folder", scenario+"_formatted.h5"),  "generator_Start_&_Shutdown_Cost")
            Start_Shutdown_Cost = Start_Shutdown_Cost.xs(self.zone_input,level=self.AGG_BY)
            Start_Shutdown_Cost = Start_Shutdown_Cost.sum(axis=0)
            Start_Shutdown_Cost.rename("Start_&_Shutdown_Cost", inplace=True)

            try:
                Emissions_Cost = pd.read_hdf(os.path.join(self.Marmot_Solutions_folder, scenario, "Processed_HDF5_folder", scenario+"_formatted.h5"), "generator_Emissions_Cost")
            except Exception:
                print("\ngenerator_Emissions_Cost not included in " + scenario + " results,\nEmissions_Cost will not be included in plot\n")
                Emissions_Cost = Start_Shutdown_Cost.copy()
                Emissions_Cost.iloc[:,0] = 0
                pass
            Emissions_Cost = Emissions_Cost.xs(self.zone_input,level=self.AGG_BY)
            Emissions_Cost = Emissions_Cost.sum(axis=0)
            Emissions_Cost.rename("Emissions_Cost", inplace=True)

            Detailed_Gen_Cost = pd.concat([Fuel_Cost, VOM_Cost, Start_Shutdown_Cost, Emissions_Cost], axis=1, sort=False)

            Detailed_Gen_Cost.columns = Detailed_Gen_Cost.columns.str.replace('_',' ')
            Detailed_Gen_Cost = Detailed_Gen_Cost.sum(axis=0)
            Detailed_Gen_Cost = Detailed_Gen_Cost.rename(scenario)

            Detailed_Gen_Cost_Out = pd.concat([Detailed_Gen_Cost_Out, Detailed_Gen_Cost], axis=1, sort=False)

        Detailed_Gen_Cost_Out = Detailed_Gen_Cost_Out.T/Total_gen #Normalize by total generation.

        Detailed_Gen_Cost_Out.index = Detailed_Gen_Cost_Out.index.str.replace('_',' ')
        Detailed_Gen_Cost_Out.index = Detailed_Gen_Cost_Out.index.str.wrap(10, break_long_words=False)
        # Deletes columns that are all 0
        Detailed_Gen_Cost_Out = Detailed_Gen_Cost_Out.loc[:, (Detailed_Gen_Cost_Out != 0).any(axis=0)]

        # Data table of values to return to main program
        Data_Table_Out = Detailed_Gen_Cost_Out

        fig3, ax = plt.subplots(figsize=(9,6))

        sb = Detailed_Gen_Cost_Out.plot.bar(stacked=True, rot=0, edgecolor='black', linewidth='0.1', ax=ax)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylabel('Total production cost, normalized by total generation ($/MWh)',  color='black', rotation='vertical')
        ax.tick_params(axis='y', which='major', length=5, width=1)
        ax.tick_params(axis='x', which='major', length=5, width=1)
        ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
        ax.margins(x=0.01)

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(reversed(handles), reversed(labels), loc='lower left',bbox_to_anchor=(1,0),
                      facecolor='inherit', frameon=True)


        """adds annotations to bar plots"""

        cost_totals = Detailed_Gen_Cost_Out.sum(axis=1) #holds total of each bar

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
                verticalalignment='center', fontsize=12)

        #inserts total bar value above each bar
        k=0
        for i in ax.patches:
            height = cost_totals[k]
            width = i.get_width()
            x, y = i.get_xy()
            ax.text(x+width/2,
                y+height + 0.05*max(ax.get_ylim()),
                '{:,.0f}'.format(height),
                horizontalalignment='center',
                verticalalignment='center', fontsize=15, color='red')
            k=k+1
            if k>=len(cost_totals):
                break

            
        return {'fig': fig3, 'data_table': Data_Table_Out}
    
    
    def sys_cost_type(self):
        # Create Dictionary to hold Datframes for each scenario 
        Stacked_Gen_Collection = {} 

        
        for scenario in self.Multi_Scenario:
            Stacked_Gen_Collection[scenario] = pd.read_hdf(os.path.join(self.Marmot_Solutions_folder, scenario,"Processed_HDF5_folder", scenario+ "_formatted.h5"),"generator_Total_Generation_Cost")
            # If data is to be agreagted by zone, then zone properties are loaded, else region properties are loaded

        Total_Generation_Stack_Out = pd.DataFrame()
        
        print("Zone = " + self.zone_input)
            
            
        for scenario in self.Multi_Scenario:
            
            print("Scenario = " + scenario)
            
            Total_Gen_Stack = Stacked_Gen_Collection.get(scenario)
            Total_Gen_Stack = Total_Gen_Stack.xs(self.zone_input,level=self.AGG_BY)
            Total_Gen_Stack = df_process_gen_inputs(Total_Gen_Stack, self)
            
            
           
            
            Total_Gen_Stack = Total_Gen_Stack.sum(axis=0)
            Total_Gen_Stack.rename(scenario, inplace=True)
            Total_Generation_Stack_Out = pd.concat([Total_Generation_Stack_Out, Total_Gen_Stack], axis=1, sort=False).fillna(0)
            
           
        
        Total_Generation_Stack_Out = df_process_categorical_index(Total_Generation_Stack_Out, self)
        Total_Generation_Stack_Out = Total_Generation_Stack_Out.T/1000000 #Convert to millions
        Total_Generation_Stack_Out = Total_Generation_Stack_Out.loc[:, (Total_Generation_Stack_Out != 0).any(axis=0)]
        
        # Data table of values to return to main program
        Data_Table_Out = pd.concat([Total_Generation_Stack_Out],  axis=1, sort=False)

        Total_Generation_Stack_Out.index = Total_Generation_Stack_Out.index.str.replace('_',' ')
        Total_Generation_Stack_Out.index = Total_Generation_Stack_Out.index.str.wrap(10, break_long_words=False)
    
        fig1, ax = plt.subplots(figsize=(9,6))

        bp = Total_Generation_Stack_Out.plot.bar(stacked=True, figsize=(9,6), rot=0, 
                         color=[self.PLEXOS_color_dict.get(x, '#333333') for x in Total_Generation_Stack_Out.columns], edgecolor='black', linewidth='0.1',ax=ax)
        
        
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
         
        ax.set_ylabel('Total System Cost (Million $)',  color='black', rotation='vertical')
        ax.tick_params(axis='y', which='major', length=5, width=1)
        ax.tick_params(axis='x', which='major', length=5, width=1)
        ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

        ax.margins(x=0.01)
       
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(reversed(handles), reversed(labels), loc='upper center',bbox_to_anchor=(0.5,-0.15), 
                     facecolor='inherit', frameon=True, ncol=2)
        
#        handles, labels = fig1.get_legend_handles_labels()
        
        #Legend 1
#        leg1 = fig1.legend(reversed(handles), reversed(labels), loc='lower left',bbox_to_anchor=(1,0), 
#                      facecolor='inherit', frameon=True)  
        
        
        # Manually add the first legend back
#        fig1.add_artist(leg1)
       
        
        
        return {'fig': fig1, 'data_table': Data_Table_Out}
    
    def sys_cost_diff(self):
        Total_Gen_Cost_Collection = {}
        Cost_Unserved_Energy_Collection = {}
        for scenario in self.Multi_Scenario:

            Total_Gen_Cost_Collection[scenario] = pd.read_hdf(os.path.join(self.Marmot_Solutions_folder, scenario, "Processed_HDF5_folder", scenario+"_formatted.h5"), "generator_Total_Generation_Cost")
            # If data is to be agregated by zone, then zone properties are loaded, else region properties are loaded
            if self.AGG_BY == "zone":
                Cost_Unserved_Energy_Collection[scenario] = pd.read_hdf(os.path.join(self.Marmot_Solutions_folder, scenario, "Processed_HDF5_folder", scenario+"_formatted.h5"), "zone_Cost_Unserved_Energy")
            else:
                Cost_Unserved_Energy_Collection[scenario] = pd.read_hdf(os.path.join(self.Marmot_Solutions_folder, scenario, "Processed_HDF5_folder", scenario+"_formatted.h5"), "region_Cost_Unserved_Energy")
                            
        Total_Systems_Cost_Out = pd.DataFrame()
        print("Zone = "+ self.zone_input)
        
        for scenario in self.Multi_Scenario:
            print("Scenario = " + scenario)
            Total_Systems_Cost = pd.DataFrame()
            
            
            Total_Gen_Cost = Total_Gen_Cost_Collection.get(scenario)

            try:
                Total_Gen_Cost = Total_Gen_Cost.xs(self.zone_input,level=self.AGG_BY)
            except KeyError:
                print("No Generators found for : "+self.zone_input)
                return pd.DataFrame()
            
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
        
        Total_Systems_Cost_Out = Total_Systems_Cost_Out/1000000 #Convert cost to millions
        Total_Systems_Cost_Out=Total_Systems_Cost_Out-Total_Systems_Cost_Out.xs(self.Multi_Scenario[0]) #Change to a diff on first scenario
        Total_Systems_Cost_Out.drop(self.Multi_Scenario[0],inplace=True) #Drop base entry
        
#        Total_Systems_Cost_Out.index = Total_Systems_Cost_Out.index.str.replace('_',' ')  
#        Total_Systems_Cost_Out.index = Total_Systems_Cost_Out.index.str.wrap(10, break_long_words=False)
        
        # Data table of values to return to main program
        Data_Table_Out = Total_Systems_Cost_Out
        
        fig2, ax = plt.subplots(figsize=(9,6))
        
        sb = Total_Systems_Cost_Out.plot.bar(stacked=True, rot=0, edgecolor='black', linewidth='0.1', ax=ax)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        ax.tick_params(axis='y', which='major', length=5, width=1)
        ax.tick_params(axis='x', which='major', length=5, width=1)
        locs,labels=plt.xticks()
        ax.set_ylabel('Generation Cost Change (Million $) \n relative to '+self.xlabels[0],  color='black', rotation='vertical')
        self.xlabels = pd.Series(self.xlabels).str.replace('_',' ').str.wrap(10, break_long_words=False)
        plt.xticks(ticks=locs,labels=self.xlabels[1:])

        ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
        ax.margins(x=0.01)
#        plt.ylim((0,600)) 
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(reversed(handles), reversed(labels), loc='upper center',bbox_to_anchor=(0.5,-0.15), 
                     facecolor='inherit', frameon=True, ncol=2)
    
    
       
        return {'fig': fig2, 'data_table': Data_Table_Out}
    
    def sys_cost_type_diff(self):
        # Create Dictionary to hold Datframes for each scenario 
        Stacked_Gen_Collection = {} 

        
        for scenario in self.Multi_Scenario:
            Stacked_Gen_Collection[scenario] = pd.read_hdf(os.path.join(self.Marmot_Solutions_folder, scenario,"Processed_HDF5_folder", scenario+ "_formatted.h5"),"generator_Total_Generation_Cost")
            # If data is to be agreagted by zone, then zone properties are loaded, else region properties are loaded

        Total_Generation_Stack_Out = pd.DataFrame()
        
        print("Zone = " + self.zone_input)
            
            
        for scenario in self.Multi_Scenario:
            
            print("Scenario = " + scenario)
            
            Total_Gen_Stack = Stacked_Gen_Collection.get(scenario)
            Total_Gen_Stack = Total_Gen_Stack.xs(self.zone_input,level=self.AGG_BY)
            Total_Gen_Stack = df_process_gen_inputs(Total_Gen_Stack, self)
            
            
           
            
            Total_Gen_Stack = Total_Gen_Stack.sum(axis=0)
            Total_Gen_Stack.rename(scenario, inplace=True)
            Total_Generation_Stack_Out = pd.concat([Total_Generation_Stack_Out, Total_Gen_Stack], axis=1, sort=False).fillna(0)
            
           
        
        Total_Generation_Stack_Out = df_process_categorical_index(Total_Generation_Stack_Out, self)
        Total_Generation_Stack_Out = Total_Generation_Stack_Out.T/1000000 #Convert to millions
        Total_Generation_Stack_Out = Total_Generation_Stack_Out.loc[:, (Total_Generation_Stack_Out != 0).any(axis=0)]
        Total_Generation_Stack_Out = Total_Generation_Stack_Out-Total_Generation_Stack_Out.xs(self.Multi_Scenario[0]) #Change to a diff on first scenario
        Total_Generation_Stack_Out.drop(self.Multi_Scenario[0],inplace=True) #Drop base entry
        
        
        # Data table of values to return to main program
        Data_Table_Out = pd.concat([Total_Generation_Stack_Out],  axis=1, sort=False)

        Total_Generation_Stack_Out.index = Total_Generation_Stack_Out.index.str.replace('_',' ')
        Total_Generation_Stack_Out.index = Total_Generation_Stack_Out.index.str.wrap(10, break_long_words=False)
    
        fig1, ax = plt.subplots(figsize=(9,6))

        bp = Total_Generation_Stack_Out.plot.bar(stacked=True, figsize=(9,6), rot=0, 
                         color=[self.PLEXOS_color_dict.get(x, '#333333') for x in Total_Generation_Stack_Out.columns], edgecolor='black', linewidth='0.1',ax=ax)
        
        
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
         
        
        ax.tick_params(axis='y', which='major', length=5, width=1)
        ax.tick_params(axis='x', which='major', length=5, width=1)
        ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
        
        locs,labels=plt.xticks()
        ax.set_ylabel('Generation Cost Change (Million $) \n relative to '+self.xlabels[0],  color='black', rotation='vertical')
        self.xlabels = pd.Series(self.xlabels).str.replace('_',' ').str.wrap(10, break_long_words=False)
        plt.xticks(ticks=locs,labels=self.xlabels[1:])


        ax.margins(x=0.01)
#        plt.ylim((0,600)) 

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(reversed(handles), reversed(labels), loc='upper center',bbox_to_anchor=(0.5,-0.15), 
                     facecolor='inherit', frameon=True, ncol=2)
        
#        handles, labels = fig1.get_legend_handles_labels()
        
        #Legend 1
#        leg1 = fig1.legend(reversed(handles), reversed(labels), loc='lower left',bbox_to_anchor=(1,0), 
#                      facecolor='inherit', frameon=True)  
        
        
        # Manually add the first legend back
#        fig1.add_artist(leg1)
       
        
        
        return {'fig': fig1, 'data_table': Data_Table_Out}
    
    def detailed_gen_cost_diff(self):
        Total_Gen_Cost_Collection = {}
        Fuel_Cost_Collection = {}
        VOM_Cost_Collection = {}
        Start_Shutdown_Cost_Collection = {} 
        Emissions_Cost_Collection = {}
        for scenario in self.Multi_Scenario:
            
            Total_Gen_Cost_Collection[scenario] = pd.read_hdf(os.path.join(self.Marmot_Solutions_folder, scenario, "Processed_HDF5_folder", scenario+"_formatted.h5"), "generator_Total_Generation_Cost")
            Fuel_Cost_Collection[scenario] = pd.read_hdf(os.path.join(self.Marmot_Solutions_folder, scenario, "Processed_HDF5_folder", scenario+"_formatted.h5"), "generator_Fuel_Cost")
            VOM_Cost_Collection[scenario] = pd.read_hdf(os.path.join(self.Marmot_Solutions_folder, scenario, "Processed_HDF5_folder", scenario+"_formatted.h5"),  "generator_VO&M_Cost")
            Start_Shutdown_Cost_Collection[scenario] = pd.read_hdf(os.path.join(self.Marmot_Solutions_folder, scenario, "Processed_HDF5_folder", scenario+"_formatted.h5"),  "generator_Start_&_Shutdown_Cost")
            try:
                Emissions_Cost_Collection[scenario] = pd.read_hdf(os.path.join(self.Marmot_Solutions_folder, scenario, "Processed_HDF5_folder", scenario+"_formatted.h5"), "generator_Emissions_Cost")
            except Exception:
                print("\ngenerator_Emissions_Cost not included in " + scenario + " results,\nEmissions_Cost will not be included in plot\n")
                Emissions_Cost_Collection[scenario] = Start_Shutdown_Cost_Collection[scenario].copy()
                Emissions_Cost_Collection[scenario].iloc[:,0] = 0
                pass   
            
        print("Zone = "+ self.zone_input)
        
        Detailed_Gen_Cost_Out = pd.DataFrame()
        
        for scenario in self.Multi_Scenario:
            print("Scenario = " + scenario)
            
            Fuel_Cost = Fuel_Cost_Collection.get(scenario)
            try:
                Fuel_Cost = Fuel_Cost.xs(self.zone_input,level=self.AGG_BY)
            except KeyError:
                return pd.DataFrame()
            Fuel_Cost = Fuel_Cost.sum(axis=0)
            Fuel_Cost.rename("Fuel_Cost", inplace=True)
            
            VOM_Cost = VOM_Cost_Collection.get(scenario)
            VOM_Cost = VOM_Cost.xs(self.zone_input,level=self.AGG_BY)
            VOM_Cost = VOM_Cost.sum(axis=0)
            VOM_Cost.rename("VO&M_Cost", inplace=True)
            
            Start_Shutdown_Cost = Start_Shutdown_Cost_Collection.get(scenario)
            Start_Shutdown_Cost = Start_Shutdown_Cost.xs(self.zone_input,level=self.AGG_BY)
            Start_Shutdown_Cost = Start_Shutdown_Cost.sum(axis=0)
            Start_Shutdown_Cost.rename("Start_&_Shutdown_Cost", inplace=True)
            
            Emissions_Cost = Emissions_Cost_Collection.get(scenario)
            Emissions_Cost = Emissions_Cost.xs(self.zone_input,level=self.AGG_BY)
            Emissions_Cost = Emissions_Cost.sum(axis=0)
            Emissions_Cost.rename("Emissions_Cost", inplace=True)
            
            Detailed_Gen_Cost = pd.concat([Fuel_Cost, VOM_Cost, Start_Shutdown_Cost, Emissions_Cost], axis=1, sort=False) 
        
            Detailed_Gen_Cost.columns = Detailed_Gen_Cost.columns.str.replace('_',' ')    
            Detailed_Gen_Cost = Detailed_Gen_Cost.sum(axis=0)
            Detailed_Gen_Cost = Detailed_Gen_Cost.rename(scenario)
            
            Detailed_Gen_Cost_Out = pd.concat([Detailed_Gen_Cost_Out, Detailed_Gen_Cost], axis=1, sort=False)
            
        Detailed_Gen_Cost_Out = Detailed_Gen_Cost_Out.T/1000000 #Convert cost to millions
        Detailed_Gen_Cost_Out = Detailed_Gen_Cost_Out-Detailed_Gen_Cost_Out.xs(self.Multi_Scenario[0]) #Change to a diff on first scenario
        Detailed_Gen_Cost_Out.drop(self.Multi_Scenario[0],inplace=True) #Drop base entry
        
        Detailed_Gen_Cost_Out.index = Detailed_Gen_Cost_Out.index.str.replace('_',' ')  
        Detailed_Gen_Cost_Out.index = Detailed_Gen_Cost_Out.index.str.wrap(10, break_long_words=False)
        # Deletes columns that are all 0 
        Detailed_Gen_Cost_Out = Detailed_Gen_Cost_Out.loc[:, (Detailed_Gen_Cost_Out != 0).any(axis=0)]
        
        # Data table of values to return to main program
        Data_Table_Out = Detailed_Gen_Cost_Out
        
        fig3, ax = plt.subplots(figsize=(9,6))
        
        sb = Detailed_Gen_Cost_Out.plot.bar(stacked=True, rot=0, edgecolor='black', linewidth='0.1', ax=ax)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='y', which='major', length=5, width=1)
        ax.tick_params(axis='x', which='major', length=5, width=1)
        locs,labels=plt.xticks()
        ax.set_ylabel('Generation Cost Change (Million $) \n relative to '+self.xlabels[0],  color='black', rotation='vertical')
        self.xlabels = pd.Series(self.xlabels).str.replace('_',' ').str.wrap(10, break_long_words=False)
        plt.xticks(ticks=locs,labels=self.xlabels[1:])
        
        ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
        ax.margins(x=0.01)
#        plt.ylim((0,600))
        
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(reversed(handles), reversed(labels), loc='lower left',bbox_to_anchor=(1,0), 
                      facecolor='inherit', frameon=True)  
                   
            
        return {'fig': fig3, 'data_table': Data_Table_Out}
