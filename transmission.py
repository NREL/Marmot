# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 10:34:48 2019

This code creates generation stack plots and is called from Marmot_plot_main.py

@author: dlevie
"""

import os
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
   def __init__(self,argument_list):
        
        self.prop = argument_list[0]
        self.start = argument_list[1]     
        self.end = argument_list[2]
        self.timezone = argument_list[3]
        self.start_date = argument_list[4]
        self.end_date = argument_list[5]
        self.hdf_out_folder = argument_list[6]
        self.zone_input =argument_list[7]
        self.AGG_BY = argument_list[8]
        self.ordered_gen = argument_list[9]
        self.PLEXOS_color_dict = argument_list[10]
        self.Multi_Scenario = argument_list[11]
        self.Scenario_Diff = argument_list[12]
        self.PLEXOS_Scenarios = argument_list[13]
        self.ylabels = argument_list[14]
        self.xlabels = argument_list[15]
        self.gen_names_dict = argument_list[18]
        self.re_gen_cat = argument_list[20]

   
    # def net_interchange(self):
    #       Net_Interchange_read = pd.read_hdf(os.path.join(self.hdf_out_folder,self.Multi_Scenario[0] + '_formatted.h5'),'region_Net_Interchange')
    #       print("     "+ self.zone_input)
    #       return(Net_Interchange_read)
    
    
    #     Stacked_Gen = Stacked_Gen_read.xs(self.zone_input,level=self.AGG_BY)        
    #     Stacked_Gen = df_process_gen_inputs(Stacked_Gen, self)
        
    #     Avail_Gen = Avail_Gen_read.xs(self.zone_input,level=self.AGG_BY)
    #     Avail_Gen = df_process_gen_inputs(Avail_Gen, self)
        
    #     try:
    #         Stacked_Curt = Stacked_Curt_read.xs(self.zone_input,level=self.AGG_BY)
    #         Stacked_Curt = df_process_gen_inputs(Stacked_Curt, self)
    #         Stacked_Curt = Stacked_Curt.sum(axis=1)
    #         Stacked_Curt[Stacked_Curt<0.05] = 0 #Remove values less than 0.05 MW
    #         Stacked_Gen.insert(len(Stacked_Gen.columns),column='Curtailment',value=Stacked_Curt) #Insert curtailment into 
    #     except Exception:
    #         pass
        
    #     # Calculates Net Load by removing variable gen + curtailment
    #     self.re_gen_cat = self.re_gen_cat + ['Curtailment']
    #     # Adjust list of values to drop depending on if it exhists in Stacked_Gen df
    #     self.re_gen_cat = [name for name in self.re_gen_cat if name in Stacked_Gen.columns]
    #     Net_Load = Stacked_Gen.drop(labels = self.re_gen_cat, axis=1)
    #     Net_Load = Net_Load.sum(axis=1)
        
    #     Stacked_Gen = Stacked_Gen.loc[:, (Stacked_Gen != 0).any(axis=0)]
        
    #     Load = Load_read.xs(self.zone_input,level=self.AGG_BY)
    #     Load = Load.groupby(["timestamp"]).sum()
    #     Load = Load.squeeze() #Convert to Series
    

    #     Pump_Load = Pump_Load_read.xs(self.zone_input,level=self.AGG_BY)
    #     Pump_Load = Pump_Load.groupby(["timestamp"]).sum()
    #     Pump_Load = Pump_Load.squeeze() #Convert to Series
    #     if (Pump_Load == 0).all() == False:
    #         Pump_Load = Load - Pump_Load


        
    #     Unserved_Energy = Unserved_Energy_read.xs(self.zone_input,level=self.AGG_BY)
    #     Unserved_Energy = Unserved_Energy.groupby(["timestamp"]).sum()
    #     Unserved_Energy = Unserved_Energy.squeeze() #Convert to Series
    #     if (Unserved_Energy == 0).all() == False:
    #         Unserved_Energy = Load - Unserved_Energy
        

    #     if self.prop == "Peak Demand":
    #          peak_pump_load_t = Pump_Load.idxmax() 
    #          end_date = peak_pump_load_t + dt.timedelta(days=self.end)
    #          start_date = peak_pump_load_t - dt.timedelta(days=self.start)
    #          Peak_Pump_Load = Pump_Load[peak_pump_load_t]
    #          Stacked_Gen = Stacked_Gen[start_date : end_date]
    #          Load = Load[start_date : end_date]
    #          Unserved_Energy = Unserved_Energy[start_date : end_date]
    #          Pump_Load = Pump_Load[start_date : end_date]

             
    #     elif self.prop == "Min Net Load":
    #         min_net_load_t = Net_Load.idxmin()
    #         end_date = min_net_load_t + dt.timedelta(days=self.end)
    #         start_date = min_net_load_t - dt.timedelta(days=self.start)
    #         Min_Net_Load = Net_Load[min_net_load_t]
    #         Stacked_Gen = Stacked_Gen[start_date : end_date]
    #         Load = Load[start_date : end_date]
    #         Unserved_Energy = Unserved_Energy[start_date : end_date]
    #         Pump_Load = Pump_Load[start_date : end_date]
            


    #     else:
    #         print("Plotting graph for entire timeperiod")
        
    #     # Data table of values to return to main program
    #     Data_Table_Out = Stacked_Gen
        
    #     fig1, ax = plt.subplots(figsize=(9,6))
    #     sp = ax.stackplot(Stacked_Gen.index.values, Stacked_Gen.values.T, labels=Stacked_Gen.columns, linewidth=5,
    #                  colors=[self.PLEXOS_color_dict.get(x, '#333333') for x in Stacked_Gen.T.index])
        
    #     if (Unserved_Energy == 0).all() == False:
    #         lp2 = plt.plot(Unserved_Energy, color='#EE1289')
        
    #     lp = plt.plot(Load, color='black')
        
    #     if (Pump_Load == 0).all() == False:
    #         lp3 = plt.plot(Pump_Load, color='black', linestyle="--")
        
        
    #     ax.set_ylabel('Generation (MW)',  color='black', rotation='vertical')
    #     ax.set_xlabel('Date ' + '(' + self.timezone + ')',  color='black', rotation='horizontal')
    #     ax.spines['right'].set_visible(False)
    #     ax.spines['top'].set_visible(False)
    #     ax.tick_params(axis='y', which='major', length=5, width=1)
    #     ax.tick_params(axis='x', which='major', length=5, width=1)
    #     ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    #     ax.margins(x=0.01)
        
    #     if self.prop == "Min Net Load":
    #         ax.annotate('Min Net Load: \n' + str(format(int(Min_Net_Load), ',')) + ' MW', xy=(min_net_load_t, Min_Net_Load), xytext=((min_net_load_t + dt.timedelta(days=0.1)), (Min_Net_Load + max(Load)/4)),
    #                 fontsize=13, arrowprops=dict(facecolor='black', width=3, shrink=0.1))
        
    #     elif self.prop == "Peak Demand":
    #             ax.annotate('Peak Demand: \n' + str(format(int(Load[peak_pump_load_t]), ',')) + ' MW', xy=(peak_pump_load_t, Peak_Pump_Load), xytext=((peak_pump_load_t + dt.timedelta(days=0.1)), (max(Load) + Load[peak_pump_load_t]*0.1)),
    #                         fontsize=13, arrowprops=dict(facecolor='black', width=3, shrink=0.1))

    #     locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
    #     formatter = mdates.ConciseDateFormatter(locator)
    #     formatter.formats[2] = '%d\n %b'
    #     formatter.zero_formats[1] = '%b\n %Y'
    #     formatter.zero_formats[2] = '%d\n %b'
    #     formatter.zero_formats[3] = '%H:%M\n %d-%b'
    #     formatter.offset_formats[3] = '%b %Y'
    #     formatter.show_offset = False
    #     ax.xaxis.set_major_locator(locator)
    #     ax.xaxis.set_major_formatter(formatter)
         
                
    #     if (Unserved_Energy == 0).all() == False:
    #         ax.fill_between(Load.index, Load,Unserved_Energy, facecolor='#EE1289')
        
    #     handles, labels = ax.get_legend_handles_labels()
        
     
    #     #Legend 1
    #     leg1 = ax.legend(reversed(handles), reversed(labels), loc='lower left',bbox_to_anchor=(1,0), 
    #                   facecolor='inherit', frameon=True)  
    #     #Legend 2
    #     if (Pump_Load == 0).all() == False:
    #         leg2 = ax.legend(lp, ['Demand + Pumped Load'], loc='center left',bbox_to_anchor=(1, 0.9), 
    #                   facecolor='inherit', frameon=True)
    #     else:
    #         leg2 = ax.legend(lp, ['Demand'], loc='center left',bbox_to_anchor=(1, 0.9), 
    #                   facecolor='inherit', frameon=True)
        
    #     #Legend 3
    #     if (Unserved_Energy == 0).all() == False:
    #         leg3 = ax.legend(lp2, ['Unserved Energy'], loc='upper left',bbox_to_anchor=(1, 0.82), 
    #                   facecolor='inherit', frameon=True)
            
    #     #Legend 4
    #     if (Pump_Load == 0).all() == False:
    #         leg4 = ax.legend(lp3, ['Demand'], loc='upper left',bbox_to_anchor=(1, 0.885), 
    #                   facecolor='inherit', frameon=True)
        
    #     # Manually add the first legend back
    #     ax.add_artist(leg1)
    #     ax.add_artist(leg2)
    #     if (Unserved_Energy == 0).all() == False:
    #         ax.add_artist(leg3)
            
    #     return {'fig': fig1, 'data_table': Data_Table_Out}
    
    # def gen_stack_facet(self):
        
    #     # Create Dictionary to hold Datframes for each scenario 
    #     Gen_Collection = {} 
    #     Load_Collection = {}
    #     Avail_Gen_Collection = {}
    #     Pump_Load_Collection = {}
    #     Unserved_Energy_Collection = {}
    #     Curtailment_Collection = {}
        
        
    #     for scenario in self.Multi_Scenario:
    #         Gen_Collection[scenario] = pd.read_hdf(self.PLEXOS_Scenarios + r"\\" + scenario + r"\Processed_HDF5_folder" + "/" + scenario+"_formatted.h5", "generator_Generation")
    #         Curtailment_Collection[scenario] = pd.read_hdf(self.PLEXOS_Scenarios + r"\\" + scenario + r"\Processed_HDF5_folder" + "/" + scenario+"_formatted.h5",  "generator_Curtailment")
    #         Avail_Gen_Collection[scenario] = pd.read_hdf(self.PLEXOS_Scenarios + r"\\" + scenario + r"\Processed_HDF5_folder" + "/" + scenario+"_formatted.h5", "generator_Available_Capacity")
    #         Pump_Load_Collection[scenario] = pd.read_hdf(self.PLEXOS_Scenarios + r"\\" + scenario + r"\Processed_HDF5_folder" + "/" + scenario+"_formatted.h5",  "generator_Pump_Load" )
    #         if self.AGG_BY == "zone":
    #             Unserved_Energy_Collection[scenario] = pd.read_hdf(self.PLEXOS_Scenarios + r"\\" + scenario + r"\Processed_HDF5_folder" + "/" + scenario+"_formatted.h5", "zone_Unserved_Energy" )
    #             Load_Collection[scenario] = pd.read_hdf(self.PLEXOS_Scenarios + r"\\" + scenario + r"\Processed_HDF5_folder" + "/" + scenario+"_formatted.h5",  "zone_Load")
    #         else:
    #             Unserved_Energy_Collection[scenario] = pd.read_hdf(self.PLEXOS_Scenarios + r"\\" + scenario + r"\Processed_HDF5_folder" + "/" + scenario+"_formatted.h5", "region_Unserved_Energy" )
    #             Load_Collection[scenario] = pd.read_hdf(self.PLEXOS_Scenarios + r"\\" + scenario + r"\Processed_HDF5_folder" + "/" + scenario+"_formatted.h5",  "region_Load")
            
            
    #     print("     "+ self.zone_input)
        
    #     xdimension=len(self.xlabels)
    #     ydimension=len(self.ylabels)
    #     grid_size = xdimension*ydimension
        
    #     fig2, axs = plt.subplots(ydimension,xdimension, figsize=((4*xdimension),(4*ydimension)), sharey=True)
    #     plt.subplots_adjust(wspace=0.05, hspace=0.2)
    #     axs = axs.ravel()
    #     i=0
        
    #     for scenario in self.Multi_Scenario:
    #         print("     " + scenario)
    #         Pump_Load = pd.Series() # Initiate pump load 
            
    #         try:
    #             Stacked_Gen = Gen_Collection.get(scenario)
    #             Stacked_Gen = Stacked_Gen.xs(self.zone_input,level=self.AGG_BY)  
    #         except Exception:
    #             i=i+1
    #             continue
            
    #         if Stacked_Gen.empty == True:
    #             continue
    #         Stacked_Gen = df_process_gen_inputs(Stacked_Gen, self)
            
    #         Avail_Gen = Avail_Gen_Collection.get(scenario)
    #         Avail_Gen = Avail_Gen.xs(self.zone_input,level=self.AGG_BY)
    #         Avail_Gen = df_process_gen_inputs(Avail_Gen, self)
            
    #         try:
    #             Stacked_Curt = Curtailment_Collection.get(scenario)
    #             Stacked_Curt = Stacked_Curt.xs(self.zone_input,level=self.AGG_BY)
    #             Stacked_Curt = df_process_gen_inputs(Stacked_Curt, self)
    #             Stacked_Curt = Stacked_Curt.sum(axis=1)
    #             Stacked_Curt[Stacked_Curt<0.05] = 0 #Remove values less than 0.05 MW
    #             Stacked_Gen.insert(len(Stacked_Gen.columns),column='Curtailment',value=Stacked_Curt) #Insert curtailment into 
    #         except Exception:
    #             pass
            
    #         # Calculates Net Load by removing variable gen + curtailment
    #         self.re_gen_cat = self.re_gen_cat + ['Curtailment']
    #         # Adjust list of values to drop depending on if it exhists in Stacked_Gen df
    #         self.re_gen_cat = [name for name in self.re_gen_cat if name in Stacked_Gen.columns]
    #         Net_Load = Stacked_Gen.drop(labels = self.re_gen_cat, axis=1)
    #         Net_Load = Net_Load.sum(axis=1)

    #         Stacked_Gen = Stacked_Gen.loc[:, (Stacked_Gen != 0).any(axis=0)]

    #         Load = Load_Collection.get(scenario)
    #         Load = Load.xs(self.zone_input,level=self.AGG_BY)
    #         Load = Load.groupby(["timestamp"]).sum()
    #         Load = Load.squeeze() #Convert to Series
                    
    #         Pump_Load = Pump_Load_Collection.get(scenario)
    #         Pump_Load = Pump_Load.xs(self.zone_input,level=self.AGG_BY)
    #         Pump_Load = Pump_Load.groupby(["timestamp"]).sum()
    #         Pump_Load = Pump_Load.squeeze() #Convert to Series
    #         if (Pump_Load == 0).all() == False:
    #             Pump_Load = Load - Pump_Load
       
    #         Unserved_Energy = Unserved_Energy_Collection.get(scenario)
    #         Unserved_Energy = Unserved_Energy.xs(self.zone_input,level=self.AGG_BY)
    #         Unserved_Energy = Unserved_Energy.groupby(["timestamp"]).sum()
    #         Unserved_Energy = Unserved_Energy.squeeze() #Convert to Series
    #         if (Unserved_Energy == 0).all() == False:
    #             Unserved_Energy = Load - Unserved_Energy
          
            
    #         if self.prop == "Peak Demand":
    #             peak_pump_load_t = Pump_Load.idxmax() 
    #             end_date = peak_pump_load_t + dt.timedelta(days=self.end)
    #             start_date = peak_pump_load_t - dt.timedelta(days=self.start)
    #             Peak_Pump_Load = Pump_Load[peak_pump_load_t]
    #             Stacked_Gen = Stacked_Gen[start_date : end_date]
    #             Load = Load[start_date : end_date]
    #             Unserved_Energy = Unserved_Energy[start_date : end_date]
    #             Pump_Load = Pump_Load[start_date : end_date]

             
    #         elif self.prop == "Min Net Load":
    #             min_net_load_t = Net_Load.idxmin()
    #             end_date = min_net_load_t + dt.timedelta(days=self.end)
    #             start_date = min_net_load_t - dt.timedelta(days=self.start)
    #             Min_Net_Load = Net_Load[min_net_load_t]
    #             Stacked_Gen = Stacked_Gen[start_date : end_date]
    #             Load = Load[start_date : end_date]
    #             Unserved_Energy = Unserved_Energy[start_date : end_date]
    #             Pump_Load = Pump_Load[start_date : end_date]

    #         else:
    #             print("Plotting graph for entire timeperiod")
            
        
    #         sp = axs[i].stackplot(Stacked_Gen.index.values, Stacked_Gen.values.T, labels=Stacked_Gen.columns, linewidth=0,
    #                      colors=[self.PLEXOS_color_dict.get(x, '#333333') for x in Stacked_Gen.T.index])
             
            
    #         if (Unserved_Energy == 0).all() == False:
    #             lp2 = axs[i].plot(Unserved_Energy, color='#EE1289')
            
    #         lp = axs[i].plot(Load, color='black')
            
    #         if (Pump_Load == 0).all() == False:
    #             lp3 = axs[i].plot(Pump_Load, color='black', linestyle="--")
            
            
    #         axs[i].spines['right'].set_visible(False)
    #         axs[i].spines['top'].set_visible(False)
    #         axs[i].tick_params(axis='y', which='major', length=5, width=1)
    #         axs[i].tick_params(axis='x', which='major', length=5, width=1)
    #         axs[i].yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    #         axs[i].margins(x=0.01)
    
    #         if self.prop == "Min Net Load":
    #             axs[i].annotate('Min Net Load: \n' + str(format(int(Min_Net_Load), ',')) + ' MW', xy=(min_net_load_t, Min_Net_Load), xytext=((min_net_load_t + dt.timedelta(days=0.1)), (Min_Net_Load + max(Load)/4)),
    #                 fontsize=13, arrowprops=dict(facecolor='black', width=3, shrink=0.1))
        
    #         elif self.prop == "Peak Demand":
    #             axs[i].annotate('Peak Demand: \n' + str(format(int(Load[peak_pump_load_t]), ',')) + ' MW', xy=(peak_pump_load_t, Peak_Pump_Load), xytext=((peak_pump_load_t + dt.timedelta(days=0.1)), (max(Load) + Load[peak_pump_load_t]*0.1)),
    #                         fontsize=13, arrowprops=dict(facecolor='black', width=3, shrink=0.1))
            
            
    #         locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    #         formatter = mdates.ConciseDateFormatter(locator)
    #         formatter.formats[2] = '%d\n %b'
    #         formatter.zero_formats[1] = '%b\n %Y'
    #         formatter.zero_formats[2] = '%d\n %b'
    #         formatter.zero_formats[3] = '%H:%M\n %d-%b'
    #         formatter.offset_formats[3] = '%b %Y'
    #         formatter.show_offset = False
    #         axs[i].xaxis.set_major_locator(locator)
    #         axs[i].xaxis.set_major_formatter(formatter)
            
    #         if (Unserved_Energy == 0).all() == False:
    #             axs[i].fill_between(Load.index, Load,Unserved_Energy, facecolor='#EE1289')
            
    #         handles, labels = axs[grid_size-1].get_legend_handles_labels()
            
         
    #         #Legend 1
    #         leg1 = axs[grid_size-1].legend(reversed(handles), reversed(labels), loc='lower left',bbox_to_anchor=(1,0), 
    #                       facecolor='inherit', frameon=True)  
    #         #Legend 2
    #         leg2 = axs[grid_size-1].legend(lp, ['Demand + Pumped Load'], loc='upper left',bbox_to_anchor=(1, 1.45), 
    #                       facecolor='inherit', frameon=True)
            
    #         #Legend 3
    #         if (Unserved_Energy == 0).all() == False:
    #             leg3 = axs[grid_size-1].legend(lp2, ['Unserved Energy'], loc='upper left',bbox_to_anchor=(1, 1.55), 
    #                       facecolor='inherit', frameon=True)
                
    #         #Legend 4
    #         if (Pump_Load == 0).all() == False:
    #             leg4 = axs[grid_size-1].legend(lp3, ['Demand'], loc='upper left',bbox_to_anchor=(1, 1.35), 
    #                       facecolor='inherit', frameon=True)
            
    #         # Manually add the first legend back
    #         axs[grid_size-1].add_artist(leg1)
    #         axs[grid_size-1].add_artist(leg2)
    #         if (Unserved_Energy == 0).all() == False:
    #             axs[grid_size-1].add_artist(leg3)
                
    #         i=i+1  
            
    #     all_axes = fig2.get_axes()
        
    #     self.xlabels = pd.Series(self.xlabels).str.replace('_',' ').str.wrap(10, break_long_words=False)
        
    #     j=0
    #     k=0
    #     for ax in all_axes:
    #         if ax.is_last_row():
    #             ax.set_xlabel(xlabel=(self.xlabels[j]),  color='black')
    #             j=j+1
    #         if ax.is_first_col(): 
    #             ax.set_ylabel(ylabel=(self.ylabels[k]),  color='black', rotation='vertical')
    #             k=k+1
                
    #     fig2.add_subplot(111, frameon=False)
    #     plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    #     plt.ylabel('Genertaion (MW)',  color='black', rotation='vertical', labelpad=60)
        
    #     return fig2
    
    # def gen_diff(self):
        
    #     # Create Dictionary to hold Datframes for each scenario 
    #     Gen_Collection = {} 
        
    #     for scenario in self.Scenario_Diff:
    #         Gen_Collection[scenario] = pd.read_hdf(self.PLEXOS_Scenarios + r"\\" + scenario + r"\Processed_HDF5_folder" + "/" + scenario+"_formatted.h5", "generator_Generation")
            
            
    #     Total_Gen_Stack_1 = Gen_Collection.get(self.Scenario_Diff[0])
    #     Total_Gen_Stack_1 = Total_Gen_Stack_1.xs(self.zone_input,level=self.AGG_BY)
    #     Total_Gen_Stack_1 = df_process_gen_inputs(Total_Gen_Stack_1, self)
    #     #Adds in all possible columns from ordered gen to ensure the two dataframes have same column names
    #     Total_Gen_Stack_1 = pd.DataFrame(Total_Gen_Stack_1, columns = self.ordered_gen).fillna(0)
        
    #     Total_Gen_Stack_2 = Gen_Collection.get(self.Scenario_Diff[1])
    #     Total_Gen_Stack_2 = Total_Gen_Stack_2.xs(self.zone_input,level=self.AGG_BY)
    #     Total_Gen_Stack_2 = df_process_gen_inputs(Total_Gen_Stack_2, self)
    #     #Adds in all possible columns from ordered gen to ensure the two dataframes have same column names
    #     Total_Gen_Stack_2 = pd.DataFrame(Total_Gen_Stack_2, columns = self.ordered_gen).fillna(0)
        
    #     print(self.Scenario_Diff[0])
    #     print(self.Scenario_Diff[1])
    #     Gen_Stack_Out = Total_Gen_Stack_1-Total_Gen_Stack_2
    #     # Removes columns that only equal 0
    #     Gen_Stack_Out = Gen_Stack_Out.loc[:, (Gen_Stack_Out != 0).any(axis=0)]
        
    #     # Data table of values to return to main program
    #     Data_Table_Out = Gen_Stack_Out
    #     # Reverses order of columns
    #     Gen_Stack_Out = Gen_Stack_Out.iloc[:, ::-1]
       
    #     fig3, ax = plt.subplots(figsize=(9,6))
        
    #     for column in Gen_Stack_Out:
    #         ax.plot(Gen_Stack_Out[column], linewidth=3, color=self.PLEXOS_color_dict[column], 
    #                 label=column)
    #         ax.legend(loc='lower left',bbox_to_anchor=(1,0), 
    #                       facecolor='inherit', frameon=True)
        

    #     ax.set_title(self.Scenario_Diff[0].replace('_', ' ') + " vs. " + self.Scenario_Diff[1].replace('_', ' '))
    #     ax.set_ylabel('Generation Difference (MW)',  color='black', rotation='vertical')
    #     ax.set_xlabel('Date ' + '(' + self.timezone + ')',  color='black', rotation='horizontal')
    #     ax.spines['right'].set_visible(False)
    #     ax.spines['top'].set_visible(False)
    #     ax.tick_params(axis='y', which='major', length=5, width=1)
    #     ax.tick_params(axis='x', which='major', length=5, width=1)
    #     ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    #     ax.margins(x=0.01)
        
    #     locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
    #     formatter = mdates.ConciseDateFormatter(locator)
    #     formatter.formats[2] = '%d\n %b'
    #     formatter.zero_formats[1] = '%b\n %Y'
    #     formatter.zero_formats[2] = '%d\n %b'
    #     formatter.zero_formats[3] = '%H:%M\n %d-%b'
    #     formatter.offset_formats[3] = '%b %Y'
    #     formatter.show_offset = False
    #     ax.xaxis.set_major_locator(locator)
    #     ax.xaxis.set_major_formatter(formatter)
        
    #     return {'fig': fig3, 'data_table': Data_Table_Out}
            