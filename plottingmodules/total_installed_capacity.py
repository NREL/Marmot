# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 08:51:15 2019

@author: dlevie
"""

import pandas as pd
#import matplotlib.pyplot as plt
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
    def __init__(self, argument_dict):
        # iterate over items in argument_dict and set as properties of class
        # see key_list in Marmot_plot_main for list of properties
        for prop in argument_dict:
            self.__setattr__(prop, argument_dict[prop])

    def total_cap(self):
        # Create Dictionary to hold Datframes for each scenario
        Installed_Capacity_Collection = {}

        for scenario in self.Multi_Scenario:
            Installed_Capacity_Collection[scenario] = pd.read_hdf(os.path.join(self.Marmot_Solutions_folder, scenario, "Processed_HDF5_folder", scenario + "_formatted.h5"),   "generator_Installed_Capacity")

        outputs = {}
        for zone_input in self.Zones:
            Total_Installed_Capacity_Out = pd.DataFrame()
            Data_Table_Out = pd.DataFrame()
            print(self.AGG_BY + " = " + zone_input)

            for scenario in self.Multi_Scenario:

                print("Scenario = " + scenario)

                Total_Installed_Capacity = Installed_Capacity_Collection.get(scenario)
                try:
                    Total_Installed_Capacity = Total_Installed_Capacity.xs(zone_input,level=self.AGG_BY)
                except KeyError:
                    print("No installed capacity in : "+zone_input)
                    break

                Total_Installed_Capacity = df_process_gen_inputs(Total_Installed_Capacity, self)
                Total_Installed_Capacity.reset_index(drop=True, inplace=True)
                Total_Installed_Capacity.rename(index={0:scenario}, inplace=True)
                Total_Installed_Capacity_Out = pd.concat([Total_Installed_Capacity_Out, Total_Installed_Capacity], axis=0, sort=False).fillna(0)


            Total_Installed_Capacity_Out = Total_Installed_Capacity_Out/1000 #Convert to GW
            Total_Installed_Capacity_Out = Total_Installed_Capacity_Out.loc[:, (Total_Installed_Capacity_Out != 0).any(axis=0)]

            # If Total_Installed_Capacity_Out df is empty returns a empty dataframe and does not plot
            if Total_Installed_Capacity_Out.empty:
                df = pd.DataFrame()
                outputs[zone_input] = df
                continue

            # Data table of values to return to main program
            Data_Table_Out = pd.concat([Data_Table_Out, Total_Installed_Capacity_Out],  axis=1, sort=False)

            Total_Installed_Capacity_Out.index = Total_Installed_Capacity_Out.index.str.replace('_',' ')
            Total_Installed_Capacity_Out.index = Total_Installed_Capacity_Out.index.str.wrap(5, break_long_words=False)


            fig1 = Total_Installed_Capacity_Out.plot.bar(stacked=True, figsize=(6,4), rot=0,
                                 color=[self.PLEXOS_color_dict.get(x, '#333333') for x in Total_Installed_Capacity_Out.columns], edgecolor='black', linewidth='0.1')

            fig1.spines['right'].set_visible(False)
            fig1.spines['top'].set_visible(False)
            fig1.set_ylabel('Total Installed Capacity (GW)',  color='black', rotation='vertical')
            #adds comma to y axis data
            fig1.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
            fig1.tick_params(axis='y', which='major', length=5, width=1)
            fig1.tick_params(axis='x', which='major', length=5, width=1)

            handles, labels = fig1.get_legend_handles_labels()
            fig1.legend(reversed(handles), reversed(labels), loc='lower left',bbox_to_anchor=(1,0),
                          facecolor='inherit', frameon=True)


            outputs[zone_input] = {'fig': fig1, 'data_table': Data_Table_Out}
        return outputs
