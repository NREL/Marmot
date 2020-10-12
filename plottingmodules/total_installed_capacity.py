# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 08:51:15 2019

@author: dlevie
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import total_generation as gen
import pdb
from matplotlib.patches import Patch
import numpy as np

import marmot_plot_functions as mfunc

#===============================================================================

custom_legend_elements = [Patch(facecolor='#DD0200',
                            alpha=0.5, edgecolor='#DD0200',
                         label='Unserved Energy')]

class mplot(object):
    def __init__(self, argument_dict):
        # iterate over items in argument_dict and set as properties of class
        # see key_list in Marmot_plot_main for list of properties
        for prop in argument_dict:
            self.__setattr__(prop, argument_dict[prop])

        # used for combined cap/gen plot
        self.argument_dict = argument_dict

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

                Total_Installed_Capacity = mfunc.df_process_gen_inputs(Total_Installed_Capacity, self.ordered_gen)
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

    def total_cap_and_gen_facet(self):
        # generation figure
        print("Generation data")
        gen_obj = gen.mplot(self.argument_dict)
        gen_outputs = gen_obj.total_gen()

        print("Installed capacity data")
        cap_outputs = self.total_cap()

        outputs = {}
        for zone_input in self.Zones:

            fig, axs = plt.subplots(1, 2, figsize=(10, 4))

            plt.subplots_adjust(wspace=0.35, hspace=0.2)
            axs = axs.ravel()

            # left panel: installed capacity
            Total_Installed_Capacity_Out = cap_outputs[zone_input]["data_table"]
            Total_Installed_Capacity_Out.index = Total_Installed_Capacity_Out.index.str.replace('_',' ')
            Total_Installed_Capacity_Out.index = Total_Installed_Capacity_Out.index.str.wrap(5, break_long_words=False)

            Total_Installed_Capacity_Out.plot.bar(stacked=True, rot=0, ax=axs[0],
                                 color=[self.PLEXOS_color_dict.get(x, '#333333') for x in Total_Installed_Capacity_Out.columns],
                                 edgecolor='black', linewidth='0.1')

            axs[0].spines['right'].set_visible(False)
            axs[0].spines['top'].set_visible(False)
            axs[0].set_ylabel('Total Installed Capacity (GW)',  color='black', rotation='vertical')
            #adds comma to y axis data
            axs[0].yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
            axs[0].tick_params(axis='y', which='major', length=5, width=1)
            axs[0].tick_params(axis='x', which='major', length=5, width=1)
            axs[0].get_legend().remove()

            # replace x-axis with custom labels
            if len(self.ticklabels) > 1:
                self.ticklabels = pd.Series(self.ticklabels).str.replace('-','- ').str.wrap(8, break_long_words=True)
                axs[0].set_xticklabels(self.ticklabels)

            # right panel: annual generation
            Total_Gen_Results = gen_outputs[zone_input]["data_table"]

            Total_Load_Out = Total_Gen_Results.loc[:, "Total Load (Demand + Pumped Load)"]
            Total_Demand_Out = Total_Gen_Results.loc[:, "Total Demand"]
            Unserved_Energy_Out = Total_Gen_Results.loc[:, "Unserved Energy"]
            Total_Generation_Stack_Out = Total_Gen_Results.drop(["Total Load (Demand + Pumped Load)", "Total Demand", "Unserved Energy"], axis=1)
            Pump_Load_Out = Total_Load_Out - Total_Demand_Out

            Total_Generation_Stack_Out.index = Total_Generation_Stack_Out.index.str.replace('_',' ')
            Total_Generation_Stack_Out.index = Total_Generation_Stack_Out.index.str.wrap(5, break_long_words=False)

            Total_Generation_Stack_Out.plot.bar(stacked=True, rot=0, ax=axs[1],
                             color=[self.PLEXOS_color_dict.get(x, '#333333') for x in Total_Generation_Stack_Out.columns], edgecolor='black', linewidth='0.1')

            axs[1].spines['right'].set_visible(False)
            axs[1].spines['top'].set_visible(False)
            axs[1].set_ylabel('Total Genertaion (GWh)',  color='black', rotation='vertical')
            #adds comma to y axis data
            axs[1].yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
            axs[1].tick_params(axis='y', which='major', length=5, width=1)
            axs[1].tick_params(axis='x', which='major', length=5, width=1)

            n=0
            for scenario in self.Multi_Scenario:

                x = [axs[1].patches[n].get_x(), axs[1].patches[n].get_x() + axs[1].patches[n].get_width()]
                height1 = [int(Total_Load_Out[scenario])]*2
                lp1 = plt.plot(x,height1, c='black', linewidth=1.5)
                height2 = [int(Total_Demand_Out[scenario])]*2
                lp2 = plt.plot(x,height2, 'r--', c='black', linewidth=1.5)
                if Unserved_Energy_Out[scenario].sum() > 0:
                    height3 = [int(Unserved_Energy_Out[scenario])]*2
                    plt.plot(x,height3, c='#DD0200', linewidth=1.5)
                    axs[1].fill_between(x, height3, height1,
                                facecolor = '#DD0200',
                                alpha=0.5)
                n=n+1

            # replace x-axis with custom labels
            if len(self.ticklabels) > 1:
                self.ticklabels = pd.Series(self.ticklabels).str.replace('-','- ').str.wrap(8, break_long_words=True)
                axs[1].set_xticklabels(self.ticklabels)


            # get names of generator to create custom legend
            l1 = Total_Installed_Capacity_Out.columns.tolist()
            l2 = Total_Generation_Stack_Out.columns.tolist()
            l1.extend(l2)

            handles = np.unique(np.array(l1)).tolist()
            handles.sort(key = lambda i:self.ordered_gen.index(i))
            handles = reversed(handles)

            # create custom gen_tech legend
            gen_tech_legend = []
            for tech in handles:
                legend_handles = [Patch(facecolor=self.PLEXOS_color_dict[tech],
                            alpha=1.0,
                         label=tech)]
                gen_tech_legend.extend(legend_handles)

            #Legend 1
            leg1 = axs[1].legend(handles = gen_tech_legend, loc='lower left', bbox_to_anchor=(1,-0.1),
                          facecolor='inherit', frameon=True, prop={"size":10})

            #Legend 2
            if Pump_Load_Out.values.sum() > 0:
              leg2 = axs[1].legend(lp1, ['Demand + Pumped Load'], loc='center left',bbox_to_anchor=(1, 1.2),
                        facecolor='inherit', frameon=True, prop={"size":10})
            else:
              leg2 = axs[1].legend(lp1, ['Demand'], loc='center left',bbox_to_anchor=(1, 1.2),
                        facecolor='inherit', frameon=True, prop={"size":10})

            #Legend 3
            if Unserved_Energy_Out.values.sum() > 0:
                leg3 = axs[1].legend(handles=custom_legend_elements, loc='upper left',bbox_to_anchor=(1, 1.15),
                          facecolor='inherit', frameon=True, prop={"size":10})

            #Legend 4
            if Pump_Load_Out.values.sum() > 0:
                leg4 = axs[1].legend(lp2, ['Demand'], loc='upper left',bbox_to_anchor=(1, 1.18),
                          facecolor='inherit', frameon=True, prop={"size":10})

            # Manually add the first legend back
            fig.add_artist(leg1)
            fig.add_artist(leg2)
            if Unserved_Energy_Out.values.sum() > 0:
                fig.add_artist(leg3)
            if Pump_Load_Out.values.sum() > 0:
                fig.add_artist(leg4)

            # add labels to panels
            axs[0].set_title("A.", fontdict={"weight":"bold"}, loc='left')
            axs[1].set_title("B.", fontdict={"weight":"bold"}, loc='left')

            # output figure
            df = pd.DataFrame()
            outputs[zone_input] = {'fig': fig, 'data_table': df}


        return outputs
