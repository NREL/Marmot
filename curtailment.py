# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 15:23:06 2019

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
        self.hdf_out_folder = argument_list[6]
        self.Zones = argument_list[7]
        self.AGG_BY = argument_list[8]
        self.ordered_gen = argument_list[9]
        self.PLEXOS_color_dict = argument_list[10]
        self.Multi_Scenario = argument_list[11]
        self.Marmot_Solutions_folder = argument_list[13]
        self.color_list = argument_list[16]
        self.marker_style = argument_list[17]
        self.gen_names_dict = argument_list[18]
        self.pv_gen_cat = argument_list[19]
        self.re_gen_cat = argument_list[20]
        self.vre_gen_cat = argument_list[21]
        
    def curt_pen(self):
        # Create Dictionary to hold Datframes for each scenario 
        Gen_Collection = {} 
        Avail_Gen_Collection = {}
        Curtailment_Collection = {}
        Total_Gen_Cost_Collection = {}
        
        for scenario in self.Multi_Scenario:
            Gen_Collection[scenario] = pd.read_hdf(os.path.join(self.PLEXOS_Scenarios, scenario,"Processed_HDF5_folder", scenario + "_formatted.h5"), "generator_Generation")
            Avail_Gen_Collection[scenario] = pd.read_hdf(os.path.join(self.PLEXOS_Scenarios, scenario, "Processed_HDF5_folder",scenario + "_formatted.h5"), "generator_Available_Capacity")
            Curtailment_Collection[scenario] = pd.read_hdf(os.path.join(self.PLEXOS_Scenarios, scenario, "Processed_HDF5_folder",scenario + "_formatted.h5"),  "generator_Curtailment")
            Total_Gen_Cost_Collection[scenario] = pd.read_hdf(os.path.join(self.PLEXOS_Scenarios, scenario, "Processed_HDF5_folder", scenario + "_formatted.h5"), "generator_Total_Generation_Cost")
                
        outputs = {}
        for zone_input in self.Zones:     
            Penetration_Curtailment_out = pd.DataFrame()
            
            print(self.AGG_BY +  " = " + zone_input)
            
            for scenario in self.Multi_Scenario:
                print("Scenario = " + scenario)
                
                gen = Gen_Collection.get(scenario)
                gen = gen.xs(zone_input,level=self.AGG_BY)
                
                avail_gen = Avail_Gen_Collection.get(scenario)
                avail_gen = avail_gen.xs(zone_input,level=self.AGG_BY) 
                
                re_curt = Curtailment_Collection.get(scenario)
                re_curt = re_curt.xs(zone_input,level=self.AGG_BY)
        
                # Finds the number of unique hours in the year
                no_hours_year = len(gen.index.unique(level="timestamp"))
                
                # Total generation across all technologies [MWh]
                total_gen = float(gen.sum())
                
                # Timeseries [MW] and Total VRE generation [MWh]
                vre_gen = (gen.loc[(slice(None), self.vre_gen_cat),:])
                total_vre_gen = float(vre_gen.sum())
                
                # Timeseries [MW] and Total RE generation [MWh]
                re_gen = (gen.loc[(slice(None), self.re_gen_cat),:])
                total_re_gen = float(re_gen.sum())
                
                # Timeseries [MW] and Total PV generation [MWh]
                pv_gen = (gen.loc[(slice(None), self.pv_gen_cat),:])
                total_pv_gen = float(pv_gen.sum())
                
                # % Penetration of generation classes across the year
                VRE_Penetration = (total_vre_gen/total_gen)*100
                RE_Penetration = (total_re_gen/total_gen)*100
                PV_Penetration = (total_pv_gen/total_gen)*100
                
                # Timeseries [MW] and Total RE available [MWh]
                re_avail = (avail_gen.loc[(slice(None), self.re_gen_cat),:])
                total_re_avail = float(re_avail.sum())
                
                # Timeseries [MW] and Total PV available [MWh]
                pv_avail = (avail_gen.loc[(slice(None), self.pv_gen_cat),:])
                total_pv_avail = float(pv_avail.sum())
            
                # Total RE curtailment [MWh]
                total_re_curt = float(re_curt.sum())
                
                # Timeseries [MW] and Total PV curtailment [MWh]
                pv_curt = (re_curt.loc[(slice(None), self.pv_gen_cat),:])
                total_pv_curt = float(pv_curt.sum())
                
                # % of hours with curtailment
                Prct_hr_RE_curt = (len((re_curt.sum(axis=1)).loc[(re_curt.sum(axis=1))>0])/no_hours_year)*100
                Prct_hr_PV_curt = (len((pv_curt.sum(axis=1)).loc[(pv_curt.sum(axis=1))>0])/no_hours_year)*100
                            
                # Max instantaneous curtailment 
                Max_RE_Curt = max(re_curt.sum(axis=1))
                Max_PV_Curt = max(pv_curt.sum(axis=1))
        
                # % RE and PV Curtailment Capacity Factor
                if total_pv_curt > 0:
                    RE_Curt_Cap_factor = (total_re_curt/Max_RE_Curt)/no_hours_year
                    PV_Curt_Cap_factor = (total_pv_curt/Max_PV_Curt)/no_hours_year
                else:
                    RE_Curt_Cap_factor = 0
                    PV_Curt_Cap_factor = 0
                
                # % Curtailment across the year
                Prct_RE_curt = (total_re_curt/total_re_avail)*100
                Prct_PV_curt = (total_pv_curt/total_pv_avail)*100
                
                # Total generation cost
                Total_Gen_Cost = Total_Gen_Cost_Collection.get(scenario)
                Total_Gen_Cost = Total_Gen_Cost.xs(zone_input,level=self.AGG_BY)
                Total_Gen_Cost = float(Total_Gen_Cost.sum())
            
                
                vg_out = pd.Series([PV_Penetration ,RE_Penetration, VRE_Penetration, Max_PV_Curt, 
                                    Max_RE_Curt, Prct_PV_curt, Prct_RE_curt, Prct_hr_PV_curt,
                                    Prct_hr_RE_curt, PV_Curt_Cap_factor, RE_Curt_Cap_factor, Total_Gen_Cost], 
                                   index=["% PV Penetration", "% RE Penetration", "% VRE Penetration",
                                          "Max PV Curtailment [MW]", "Max RE Curtailment [MW]",
                                          "% PV Curtailment", '% RE Curtailment',"% PV hrs Curtailed", 
                                          "% RE hrs Curtailed", "PV Curtailment Capacity Factor", 
                                          "RE Curtailment Capacity Factor", "Gen Cost"])
                vg_out = vg_out.rename(scenario)
                
                Penetration_Curtailment_out = pd.concat([Penetration_Curtailment_out, vg_out], axis=1, sort=False)
                
             
            Penetration_Curtailment_out = Penetration_Curtailment_out.T
            
            # Data table of values to return to main program
            Data_Table_Out = Penetration_Curtailment_out 
            
            VG_index = pd.Series(Penetration_Curtailment_out.index)
            # VG_index = VG_index.str.split(n=1, pat="_", expand=True)
            # VG_index.rename(columns = {0:"Scenario"}, inplace=True) 
            VG_index.rename("Scenario", inplace=True)
            # VG_index = VG_index["Scenario"]
            Penetration_Curtailment_out.loc[:, "Scenario"] = VG_index[:,].values     
                
            marker_dict = dict(zip(VG_index.unique(), self.marker_style))
            colour_dict = dict(zip(VG_index.unique(), self.color_list))
            
            Penetration_Curtailment_out["colour"] = [colour_dict.get(x, '#333333') for x in Penetration_Curtailment_out.Scenario]
            Penetration_Curtailment_out["marker"] = [marker_dict.get(x, '.') for x in Penetration_Curtailment_out.Scenario]
            
            
            fig1, ax = plt.subplots(figsize=(9,6))
            for index, row in Penetration_Curtailment_out.iterrows():
                if self.prop == "PV":
                    ax.scatter(row["% PV Penetration"], row["% PV Curtailment"],
                          marker=row["marker"],  c=row["colour"], s=100, label = row["Scenario"])
                    ax.set_ylabel('% PV Curtailment',  color='black', rotation='vertical')
                    ax.set_xlabel('% PV Penetration',  color='black', rotation='horizontal')
    
                elif self.prop == "PV+Wind":
                    ax.scatter(row["% RE Penetration"], row["% RE Curtailment"],
                          marker=row["marker"],  c=row["colour"], s=100, label = row["Scenario"])
                    ax.set_ylabel('% PV + Wind Curtailment',  color='black', rotation='vertical')
                    ax.set_xlabel('% PV + Wind Penetration',  color='black', rotation='horizontal')
            
            ax.set_ylim(bottom=0)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.tick_params(axis='y', which='major', length=5, width=1)
            ax.tick_params(axis='x', which='major', length=5, width=1)
            ax.margins(x=0.01)
        
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())
            
            outputs[zone_input] = {'fig': fig1, 'data_table': Data_Table_Out}
        return outputs
            
    def curt_duration_curve(self):
        Curtailment_Collection = {}
        for scenario in self.Multi_Scenario:
            Curtailment_Collection[scenario] = pd.read_hdf(os.path.join(self.Marmot_Solutions_folder, scenario, "Processed_HDF5_folder", scenario + "_formatted.h5"),  "generator_Curtailment")
            
        RE_Curtailment_DC = pd.DataFrame()
        PV_Curtailment_DC = pd.DataFrame()
        
        outputs = {}
        for zone_input in self.Zones:
            print(self.AGG_BY +  " = " + zone_input)
            
            for scenario in self.Multi_Scenario:
                print("Scenario = " + scenario)
                
                re_curt = Curtailment_Collection.get(scenario)
                
                # Timeseries [MW] RE curtailment [MWh]
                re_curt = re_curt.xs(zone_input,level=self.AGG_BY)
                
                # Timeseries [MW] PV curtailment [MWh]
                pv_curt = (re_curt.loc[(slice(None), self.pv_gen_cat),:])
                
                re_curt = re_curt.groupby(["timestamp"]).sum()
                pv_curt = pv_curt.groupby(["timestamp"]).sum()
                
                re_curt = re_curt.squeeze() #Convert to Series
                pv_curt = pv_curt.squeeze() #Convert to Series
                
                # Sort from larget to smallest             
                re_cdc = re_curt.sort_values(ascending=False).reset_index(drop=True)
                pv_cdc = pv_curt.sort_values(ascending=False).reset_index(drop=True)
                
                re_cdc.rename(scenario, inplace=True)
                pv_cdc.rename(scenario, inplace=True)
                
                RE_Curtailment_DC = pd.concat([RE_Curtailment_DC, re_cdc], axis=1, sort=False)
                PV_Curtailment_DC = pd.concat([PV_Curtailment_DC, pv_cdc], axis=1, sort=False)
            
            # Remove columns that have values less than 1 
            RE_Curtailment_DC = RE_Curtailment_DC.loc[:, (RE_Curtailment_DC >= 1).any(axis=0)]
            PV_Curtailment_DC = PV_Curtailment_DC.loc[:, (PV_Curtailment_DC >= 1).any(axis=0)]
            # Replace _ with white space
            RE_Curtailment_DC.columns = RE_Curtailment_DC.columns.str.replace('_',' ')   
            PV_Curtailment_DC.columns = PV_Curtailment_DC.columns.str.replace('_',' ')   
            
            # Create Dictionary from scenario names and color list
            colour_dict = dict(zip(RE_Curtailment_DC.columns, self.color_list))
            
            
            fig2, ax = plt.subplots(figsize=(9,6))
            
            if self.prop == "PV":
                Data_Table_Out = PV_Curtailment_DC
                
                for column in PV_Curtailment_DC:
                    ax.plot(PV_Curtailment_DC[column], linewidth=3, color=colour_dict[column], 
                            label=column)
                    ax.legend(loc='lower left',bbox_to_anchor=(1,0), 
                              facecolor='inherit', frameon=True)
                    ax.set_ylabel('PV Curtailment (MW)',  color='black', rotation='vertical')
            
            if self.prop == "PV+Wind":
                Data_Table_Out = RE_Curtailment_DC
                
                for column in RE_Curtailment_DC:
                    ax.plot(RE_Curtailment_DC[column], linewidth=3, color=colour_dict[column], 
                            label=column)
                    ax.legend(loc='lower left',bbox_to_anchor=(1,0), 
                              facecolor='inherit', frameon=True)
                    ax.set_ylabel('PV + Wind Curtailment (MW)',  color='black', rotation='vertical')
            
            ax.set_xlabel('Hours',  color='black', rotation='horizontal')
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.tick_params(axis='y', which='major', length=5, width=1)
            ax.tick_params(axis='x', which='major', length=5, width=1)
            ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
            ax.margins(x=0.01)
            ax.set_xlim=(0, 9490)
            ax.set_ylim(bottom=0)
            
            outputs[zone_input] = {'fig': fig2, 'data_table': Data_Table_Out}
        return outputs
        
        