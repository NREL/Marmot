# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 13:20:56 2019

This code creates total generation stacked bar plots and is called from Marmot_plot_main.py


@author: dlevie
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


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

def df_process_categorical_index(df, self): 
    df=df
    df.index = df.index.astype("category")
    df.index = df.index.set_categories(self.ordered_gen)
    df = df.sort_index()
    return df           



class mplot(object):
    def __init__(self, prop, start, end, timezone, hdf_out_folder,  
                                     zone_input, AGG_BY, ordered_gen, PLEXOS_color_dict, 
                                     Multi_Scenario, Scenario_Diff, PLEXOS_Scenarios, ylabels, 
                                     xlabels, color_list, marker_style, gen_names_dict, pv_gen_cat, 
                                     re_gen_cat, vre_gen_cat):
        
        self.hdf_out_folder = hdf_out_folder
        self.zone_input =zone_input
        self.AGG_BY = AGG_BY
        self.ordered_gen = ordered_gen
        self.PLEXOS_color_dict = PLEXOS_color_dict
        self.Multi_Scenario = Multi_Scenario
        self.PLEXOS_Scenarios = PLEXOS_Scenarios
        self.ylabels = ylabels
        self.xlabels = xlabels
        self.gen_names_dict = gen_names_dict
        
    def total_gen(self):
        # Create Dictionary to hold Datframes for each scenario 
        Stacked_Gen_Collection = {} 
        Stacked_Load_Collection = {}
        Pump_Load_Collection = {}
        Curtailment_Collection = {}
        
        for scenario in self.Multi_Scenario:
            Stacked_Gen_Collection[scenario] = pd.read_hdf(self.PLEXOS_Scenarios + r"\\" + scenario + r"\Processed_HDF5_folder" + "/" + self.Multi_Scenario[0]+"_formatted.h5","generator_Generation")
            Stacked_Load_Collection[scenario] = pd.read_hdf(self.PLEXOS_Scenarios + r"\\" + scenario + r"\Processed_HDF5_folder" + "/" + self.Multi_Scenario[0]+"_formatted.h5",  "region_Load")
            Pump_Load_Collection[scenario] =pd.read_hdf(self.PLEXOS_Scenarios + r"\\" + scenario + r"\Processed_HDF5_folder" + "/" + self.Multi_Scenario[0]+"_formatted.h5", "generator_Pump_Load" )
            Curtailment_Collection[scenario] = pd.read_hdf(self.PLEXOS_Scenarios + r"\\" + scenario + r"\Processed_HDF5_folder" + "/" + self.Multi_Scenario[0]+"_formatted.h5",  "generator_Curtailment")
            
            
        Total_Generation_Stack_Out = pd.DataFrame()
        Total_Load_Out = pd.DataFrame()
        Pump_Load_Out = pd.DataFrame()
        print("     "+ self.zone_input)
            
            
        for scenario in self.Multi_Scenario:
            
            Total_Gen_Stack = Stacked_Gen_Collection.get(scenario)
            Total_Gen_Stack = Total_Gen_Stack.xs(self.zone_input,level=self.AGG_BY)
            Total_Gen_Stack = df_process_gen_inputs(Total_Gen_Stack, self)
            
            try:
                Stacked_Curt = Curtailment_Collection.get(scenario)
                Stacked_Curt = Stacked_Curt.xs(self.zone_input,level=self.AGG_BY)
                Stacked_Curt = df_process_gen_inputs(Stacked_Curt, self)
                Stacked_Curt = Stacked_Curt.sum(axis=1)
                Total_Gen_Stack.insert(len(Total_Gen_Stack.columns),column='Curtailment',value=Stacked_Curt) #Insert curtailment into 
                Total_Gen_Stack = Total_Gen_Stack.loc[:, (Total_Gen_Stack != 0).any(axis=0)]
            except Exception:
                pass
            
            Total_Gen_Stack = Total_Gen_Stack.sum(axis=0)
            Total_Gen_Stack.rename(scenario, inplace=True)
            Total_Generation_Stack_Out = pd.concat([Total_Generation_Stack_Out, Total_Gen_Stack], axis=1, sort=False).fillna(0)
            
            Total_Load = Stacked_Load_Collection.get(scenario)
            Total_Load = Total_Load.xs(self.zone_input,level=self.AGG_BY)
            Total_Load = Total_Load.groupby(["timestamp"]).sum()
            Total_Load = Total_Load.rename(columns={0:scenario}).sum(axis=0)
            Total_Load_Out = pd.concat([Total_Load_Out, Total_Load], axis=0, sort=False)
            
            Pump_Load = Pump_Load_Collection.get(scenario)
            Pump_Load = Pump_Load.xs(self.zone_input,level=self.AGG_BY)
            Pump_Load = Pump_Load.groupby(["timestamp"]).sum()
            Pump_Load = Pump_Load.rename(columns={0:scenario}).sum(axis=0)
            if (Pump_Load == 0).all() == False:
                Pump_Load = Total_Load - Pump_Load
            Pump_Load_Out = pd.concat([Pump_Load_Out, Pump_Load], axis=0, sort=False)
              
        
        Total_Load_Out = Total_Load_Out.rename(columns={0:'Total Load'})
        
        Total_Generation_Stack_Out = df_process_categorical_index(Total_Generation_Stack_Out, self)
        Total_Generation_Stack_Out = Total_Generation_Stack_Out.T/1000 
        Total_Generation_Stack_Out = Total_Generation_Stack_Out.loc[:, (Total_Generation_Stack_Out != 0).any(axis=0)]
        
        # Data table of values to return to main program
        Data_Table_Out = pd.concat([Total_Load_Out/1000, Total_Generation_Stack_Out],  axis=1, sort=False)

        Total_Generation_Stack_Out.index = Total_Generation_Stack_Out.index.str.replace('_',' ')
        Total_Generation_Stack_Out.index = Total_Generation_Stack_Out.index.str.wrap(10, break_long_words=False)
    
        Total_Load_Out = Total_Load_Out.T/1000
        Pump_Load_Out = Pump_Load_Out.T/1000
        
        print(Pump_Load_Out)

        
        fig1 = Total_Generation_Stack_Out.plot.bar(stacked=True, figsize=(9,6), rot=0, 
                         color=[self.PLEXOS_color_dict.get(x, '#333333') for x in Total_Generation_Stack_Out.columns], edgecolor='black', linewidth='0.1')
        fig1.spines['right'].set_visible(False)
        fig1.spines['top'].set_visible(False)
        fig1.set_ylabel('Total Genertaion (GWh)',  color='black', rotation='vertical')
        #adds comma to y axis data 
        fig1.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
        fig1.tick_params(axis='y', which='major', length=5, width=1)
        fig1.tick_params(axis='x', which='major', length=5, width=1)
        
        n=0
        for scenario in self.Multi_Scenario:
            
            x = [fig1.patches[n].get_x(), fig1.patches[n].get_x() + fig1.patches[n].get_width()]
            height1 = [int(Total_Load_Out[scenario])]*2
            lp1 = plt.plot(x,height1, c='black', linewidth=1.5)
            if Pump_Load_Out.values.sum() > 0:
                height2 = [int(Pump_Load_Out[scenario])]*2
                lp2 = plt.plot(x,height2, 'r--', c='black', linewidth=1.5)   
            n=n+1
         
        #Legend 1 
        handles, labels = fig1.get_legend_handles_labels()
        leg1 = fig1.legend(reversed(handles), reversed(labels), loc='lower left',bbox_to_anchor=(1,0), 
                      facecolor='inherit', frameon=True)
        
        if (Pump_Load_Out.values.sum()) == 0:
            leg2 = fig1.legend(lp1, ['Demand'], loc='upper left',bbox_to_anchor=(1, 0.95), 
                      facecolor='inherit', frameon=True)
            fig1.add_artist(leg1)
        
        #Legend 2
        if (Pump_Load_Out.values.sum()) > 0:
            leg2 = fig1.legend(lp1, ['Demand + Pumped Load'], loc='upper left',bbox_to_anchor=(1, 0.95), 
                      facecolor='inherit', frameon=True)
        
            leg3 = fig1.legend(lp2, ['Demand'], loc='upper left',bbox_to_anchor=(1, 0.85), 
                      facecolor='inherit', frameon=True)
        
            fig1.add_artist(leg1)
            fig1.add_artist(leg2)
        
        return {'fig': fig1, 'data_table': Data_Table_Out}
    
    
    def total_gen_facet(self):
        Gen_Collection = {} 
        Load_Collection = {}
        Curtailment_Collection = {}
        
        for scenario in self.Multi_Scenario:
            try:
                Gen_Collection[scenario] = pd.read_hdf(self.PLEXOS_Scenarios + r"\\" + scenario + r"\Processed_HDF5_folder" + "/" + scenario+"_formatted.h5", "generator_Generation")
                Load_Collection[scenario] = pd.read_hdf(self.PLEXOS_Scenarios + r"\\" + scenario + r"\Processed_HDF5_folder" + "/" + scenario+"_formatted.h5",  "region_Load")
                Curtailment_Collection[scenario] = pd.read_hdf(self.PLEXOS_Scenarios + r"\\" + scenario + r"\Processed_HDF5_folder" + "/" + scenario+"_formatted.h5",  "generator_Curtailment")
            except Exception:
                pass
            
    
        Total_Generation_Stack_Out = pd.DataFrame()
        Total_Load_Out = pd.DataFrame()
        print("     "+ self.zone_input)
        
        
        for scenario in self.Multi_Scenario:      
            print("     " + scenario)
            try:
                Total_Gen_Stack = Gen_Collection.get(scenario)
                Total_Gen_Stack = Total_Gen_Stack.xs(self.zone_input,level=self.AGG_BY)
                Total_Gen_Stack = df_process_gen_inputs(Total_Gen_Stack, self)            
                Stacked_Curt = Curtailment_Collection.get(scenario)
                Stacked_Curt = Stacked_Curt.xs(self.zone_input,level=self.AGG_BY)
                Stacked_Curt = df_process_gen_inputs(Stacked_Curt, self)
                Stacked_Curt = Stacked_Curt.sum(axis=1)
                Total_Gen_Stack.insert(len(Total_Gen_Stack.columns),column='Curtailment',value=Stacked_Curt) #Insert curtailment into 
                Total_Gen_Stack = Total_Gen_Stack.loc[:, (Total_Gen_Stack != 0).any(axis=0)]
                
                Total_Gen_Stack = Total_Gen_Stack.sum(axis=0)
                Total_Gen_Stack.rename(scenario, inplace=True)
    
                Total_Generation_Stack_Out = pd.concat([Total_Generation_Stack_Out, Total_Gen_Stack], axis=1, sort=False).fillna(0)
            
                Total_Load = Load_Collection.get(scenario)
                Total_Load = Total_Load.xs(self.zone_input,level=self.AGG_BY)
                Total_Load = Total_Load.groupby(["timestamp"]).sum()
                Total_Load = Total_Load.rename(columns={0:scenario}).sum(axis=0)
                Total_Load_Out = pd.concat([Total_Load_Out, Total_Load], axis=0, sort=False)
            except Exception:
                print("Error: Skipping " + scenario)
                pass
        
        Total_Load_Out = Total_Load_Out.rename(columns={0:'Total Load'})
        
        Total_Generation_Stack_Out = df_process_categorical_index(Total_Generation_Stack_Out, self)
        Total_Generation_Stack_Out = Total_Generation_Stack_Out.T/1000 
        Total_Generation_Stack_Out = Total_Generation_Stack_Out.loc[:, (Total_Generation_Stack_Out != 0).any(axis=0)]
        
        # Data table of values to return to main program
        Data_Table_Out = pd.concat([Total_Load_Out/1000, Total_Generation_Stack_Out],  axis=1, sort=False)
        
        Total_Generation_Stack_Out.index = Total_Generation_Stack_Out.index.str.replace('_',' ')
        Total_Generation_Stack_Out.index = Total_Generation_Stack_Out.index.str.wrap(11, break_long_words=False)
        
        Total_Load_Out.index = Total_Load_Out.index.str.replace('_',' ')
        Total_Load_Out.index = Total_Load_Out.index.str.wrap(11, break_long_words=False)
    
        Total_Load_Out = Total_Load_Out.T/1000             
    
        xdimension=len(self.xlabels)
        ydimension=len(self.ylabels)
        grid_size = xdimension*ydimension
    
        fig2, axs = plt.subplots(ydimension,xdimension, figsize=((2*xdimension), (4*ydimension)), sharey=True)
        axs = axs.ravel()
        plt.subplots_adjust(wspace=0, hspace=0.01)
        
        i=0
        for index in Total_Generation_Stack_Out.index:
        
            sb = Total_Generation_Stack_Out.iloc[i:i+1].plot.bar(stacked=True, rot=0,
            color=[self.PLEXOS_color_dict.get(x, '#333333') for x in Total_Generation_Stack_Out.columns], edgecolor='black', linewidth='0.1', 
                                         ax=axs[i])
                
            axs[i].get_legend().remove()
            axs[i].spines['right'].set_visible(False)
            axs[i].spines['top'].set_visible(False)
            axs[i].xaxis.set_ticklabels([])
            axs[i].tick_params(axis='y', which='major', length=5, width=1)
            axs[i].tick_params(axis='x', which='major', length=5, width=1)
            axs[i].yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
            axs[i].margins(x=0.01)
            
            height = [int(Total_Load_Out[index])]
            axs[i].axhline(y=height,xmin=0.25,xmax=0.75, linestyle ='--', c="black",linewidth=1.5)
            
            handles, labels = axs[1].get_legend_handles_labels()
            
            leg1 = axs[grid_size-1].legend(reversed(handles), reversed(labels), loc='lower left',bbox_to_anchor=(1,0), 
                      facecolor='inherit', frameon=True)
            
            #Legend 2
            leg2 = axs[grid_size-1].legend(['Load'], loc='upper left',bbox_to_anchor=(1, 0.95), 
                          facecolor='inherit', frameon=True)
            
            fig2.add_artist(leg1)
            
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
        plt.ylabel('Total Genertaion (GWh)',  color='black', rotation='vertical', labelpad=60)
        
        return {'fig': fig2, 'data_table': Data_Table_Out}
        
        