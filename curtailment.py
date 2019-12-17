# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 15:23:06 2019

@author: dlevie
"""

import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict


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
    def __init__(self, prop, hdf_out_folder, HDF5_output, zone_input, AGG_BY, ordered_gen, PLEXOS_color_dict, Multi_Scenario, 
                 PLEXOS_Scenarios, color_list, marker_style, gen_names_dict, pv_gen_cat, re_gen_cat, vre_gen_cat):
        self.prop = prop
        self.hdf_out_folder = hdf_out_folder
        self.HDF5_output = HDF5_output
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
        
    def curt_pen(self):
        # Create Dictionary to hold Datframes for each scenario 
        Gen_Collection = {} 
        Avail_Gen_Collection = {}
        Curtailment_Collection = {}
        Installed_Capacity_Collection = {} 
        Total_Gen_Cost_Collection = {}
        
        for scenario in self.Multi_Scenario:
            Installed_Capacity_Collection[scenario] = pd.read_hdf(self.PLEXOS_Scenarios + r"\\" + scenario + r"\Processed_HDF5_folder" + "/" + self.HDF5_output,  "generator_Installed_Capacity")
            Gen_Collection[scenario] = pd.read_hdf(self.PLEXOS_Scenarios + r"\\" + scenario + r"\Processed_HDF5_folder" + "/" + self.HDF5_output, "generator_Generation")
            Avail_Gen_Collection[scenario] = pd.read_hdf(self.PLEXOS_Scenarios + r"\\" + scenario + r"\Processed_HDF5_folder" + "/" + self.HDF5_output, "generator_Available_Capacity")
            Curtailment_Collection[scenario] = pd.read_hdf(self.PLEXOS_Scenarios + r"\\" + scenario + r"\Processed_HDF5_folder" + "/" + self.HDF5_output,  "generator_Curtailment")
            Total_Gen_Cost_Collection[scenario] = pd.read_hdf(self.PLEXOS_Scenarios + r"\\" + scenario + r"\Processed_HDF5_folder" + "/" + self.HDF5_output, "generator_Total_Generation_Cost")
            
        
        Penetration_Curtailment_out = pd.DataFrame()
        
        for scenario in self.Multi_Scenario:
            print("     " + scenario)
    
            gen = Gen_Collection.get(scenario)
            gen = gen.xs(self.zone_input,level=self.AGG_BY)
            
            total_gen = gen.sum(axis=1)
            total_gen = total_gen.sum(axis=0)
            
            vre_gen = (gen.loc[(slice(None), self.vre_gen_cat),:]).sum(axis=1)
            vre_gen = vre_gen.sum(axis=0)
            
            re_gen = (gen.loc[(slice(None), self.re_gen_cat),:]).sum(axis=1)
            re_gen = re_gen.sum(axis=0)
            
            pv_gen = (gen.loc[(slice(None), self.pv_gen_cat),:]).sum(axis=1)   
            pv_gen = pv_gen.sum(axis=0)
            
            VRE_Penetration = (vre_gen/total_gen)*100
            RE_Penetration = (re_gen/total_gen)*100
            PV_Penetration = (pv_gen/total_gen)*100
            
            avail_gen = Avail_Gen_Collection.get(scenario)
            avail_gen = avail_gen.xs(self.zone_input,level=self.AGG_BY)  
            
            re_avail = (avail_gen.loc[(slice(None), self.re_gen_cat),:]).sum(axis=1)
            re_avail = re_avail.sum(axis=0)
            
            pv_avail = (avail_gen.loc[(slice(None), self.pv_gen_cat),:]).sum(axis=1)
            pv_avail = pv_avail.sum(axis=0)

            total_curt = Curtailment_Collection.get(scenario)
            re_curt = total_curt.xs(self.zone_input,level=self.AGG_BY).sum(axis=1)
            re_curt = re_curt.sum(axis=0)
            
            pv_curt = total_curt.xs(self.zone_input,level=self.AGG_BY)
            pv_curt = (pv_curt.loc[(slice(None), self.pv_gen_cat),:]).sum(axis=1) 
            pv_curt = pv_curt.sum(axis=0)
                    
            Prct_RE_curt = (re_curt/re_avail)*100
            Prct_PV_curt = (pv_curt/pv_avail)*100
            
            Total_Gen_Cost = Total_Gen_Cost_Collection.get(scenario)
            Total_Gen_Cost = Total_Gen_Cost.xs(self.zone_input,level=self.AGG_BY).sum(axis=1)
            Total_Gen_Cost = Total_Gen_Cost.sum(axis=0)

            
            vg_out = pd.Series([PV_Penetration ,RE_Penetration, VRE_Penetration, Prct_PV_curt, Prct_RE_curt, Total_Gen_Cost], 
                               index=["% PV Penetration", "% RE Penetration", "% VRE Penetration", "% PV Curtailment", '% RE Curtailment', "Gen Cost"])
            vg_out = vg_out.rename(scenario)
            
            Penetration_Curtailment_out = pd.concat([Penetration_Curtailment_out, vg_out], axis=1, sort=False)
         
        Penetration_Curtailment_out = Penetration_Curtailment_out.T
        
        # Data table of values to return to main program
        Data_Table_Out = Penetration_Curtailment_out 
        
        VG_index = pd.Series(Penetration_Curtailment_out.index)
        VG_index = VG_index.str.split(n=1, pat="_", expand=True)
        VG_index.rename(columns = {0:"Scenario"}, inplace=True) 
        VG_index = VG_index["Scenario"]
        Penetration_Curtailment_out.loc[:, "Scenario"] = VG_index[:,].values     
            
        marker_dict = dict(zip(VG_index.unique(), self.marker_style))
        colour_dict = dict(zip(VG_index.unique(), self.color_list))
        
        Penetration_Curtailment_out["colour"] = [colour_dict.get(x, '#333333') for x in Penetration_Curtailment_out.Scenario]
        Penetration_Curtailment_out["marker"] = [marker_dict.get(x, '+') for x in Penetration_Curtailment_out.Scenario]
        
        
        fig1, ax = plt.subplots(figsize=(9,6))
        for index, row in Penetration_Curtailment_out.iterrows():
            if self.prop == "PV":
                sp = ax.scatter(row["% PV Penetration"], row["% PV Curtailment"],
                      marker=row["marker"],  c=row["colour"], s=100, label = row["Scenario"])
                ax.set_ylabel('% PV Curtailment',  color='black', rotation='vertical')
                ax.set_xlabel('% PV Penetration',  color='black', rotation='horizontal')

            elif self.prop == "PV+Wind":
                sp = ax.scatter(row["% RE Penetration"], row["% RE Curtailment"],
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
        
        return {'fig': fig1, 'data_table': Data_Table_Out}