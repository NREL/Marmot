# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 15:23:06 2019

@author: dlevie

This module creates plots that show curtailment

"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict
import matplotlib as mpl
import matplotlib.dates as mdates

import logging
import marmot.plottingmodules.marmot_plot_functions as mfunc
import marmot.config.mconfig as mconfig
import matplotlib.ticker as mtick

#===============================================================================

class MPlot(object):
    def __init__(self, argument_dict):
        # iterate over items in argument_dict and set as properties of class
        # see key_list in Marmot_plot_main for list of properties
        for prop in argument_dict:
            self.__setattr__(prop, argument_dict[prop])
        self.logger = logging.getLogger('marmot_plot.'+__name__)
        self.x = mconfig.parser("figure_size","xdimension")
        self.y = mconfig.parser("figure_size","ydimension")
        self.y_axes_decimalpt = mconfig.parser("axes_options","y_axes_decimalpt")
        self.curtailment_prop = mconfig.parser("plot_data","curtailment_property")
        
        self.mplot_data_dict = {}


    def curt_duration_curve(self, figure_name=None, prop=None, start=None, end=None, 
                  timezone="", start_date_range=None, end_date_range=None):
        
        outputs = {}
        
        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True,f"generator_{self.curtailment_prop}",self.Scenarios)]
        
        # Runs get_data to populate mplot_data_dict with all required properties, returns a 1 if required data is missing
        check_input_data = mfunc.get_data(self.mplot_data_dict, properties,self.Marmot_Solutions_folder)

        if 1 in check_input_data:
            return mfunc.MissingInputData()

        for zone_input in self.Zones:
            self.logger.info(f"{self.AGG_BY} = {zone_input}")
            
            RE_Curtailment_DC = pd.DataFrame()
            PV_Curtailment_DC = pd.DataFrame()
            
            for scenario in self.Scenarios:
                self.logger.info(f"Scenario = {scenario}")

                re_curt = self.mplot_data_dict[f"generator_{self.curtailment_prop}"].get(scenario)

                # Timeseries [MW] RE curtailment [MWh]
                try: #Check for regions missing all generation.
                    re_curt = re_curt.xs(zone_input,level = self.AGG_BY)
                except KeyError:
                        self.logger.info(f'No curtailment in {zone_input}')
                        continue

                # Timeseries [MW] PV curtailment [MWh]
                pv_curt = (re_curt.loc[(slice(None), self.pv_gen_cat),:])

                re_curt = re_curt.groupby(["timestamp"]).sum()
                pv_curt = pv_curt.groupby(["timestamp"]).sum()

                re_curt = re_curt.squeeze() #Convert to Series
                pv_curt = pv_curt.squeeze() #Convert to Series
                
                
                if pd.isna(start_date_range) == False:
                    self.logger.info(f"Plotting specific date range: \
                    {str(start_date_range)} to {str(end_date_range)}")
                    re_curt = re_curt[start_date_range : end_date_range]
                    pv_curt = pv_curt[start_date_range : end_date_range]
                    
                    if re_curt.empty is True and prop == "PV+Wind": 
                        self.logger.warning('No data in selected Date Range')
                        continue
                    
                    if pv_curt.empty is True and prop == "PV":
                        self.logger.warning('No data in selected Date Range')
                        continue
                
                
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

            fig2, ax = plt.subplots(figsize=(self.x,self.y))

            if prop == "PV":
                
                if PV_Curtailment_DC.empty:
                    out = mfunc.MissingZoneData()
                    outputs[zone_input] = out
                    continue
                # unit conversion return divisor and energy units
                unitconversion = mfunc.capacity_energy_unitconversion(PV_Curtailment_DC.values.max())
                PV_Curtailment_DC = PV_Curtailment_DC/unitconversion['divisor'] 
                Data_Table_Out = PV_Curtailment_DC
                Data_Table_Out = Data_Table_Out.add_suffix(f" ({unitconversion['units']})")
                
                x_axis_lim = 1.25 * len(PV_Curtailment_DC)
                for column in PV_Curtailment_DC:
                    ax.plot(PV_Curtailment_DC[column], linewidth=3, color=colour_dict[column],
                            label=column)
                    ax.legend(loc='lower left',bbox_to_anchor=(1,0),
                              facecolor='inherit', frameon=True)
                    ax.set_ylabel(f"PV Curtailment ({unitconversion['units']})",  color='black', rotation='vertical')

            if prop == "PV+Wind":
                
                if RE_Curtailment_DC.empty:
                    out = mfunc.MissingZoneData()
                    outputs[zone_input] = out
                    continue
                # unit conversion return divisor and energy units
                unitconversion = mfunc.capacity_energy_unitconversion(RE_Curtailment_DC.values.max())
                RE_Curtailment_DC = RE_Curtailment_DC/unitconversion['divisor'] 
                Data_Table_Out = RE_Curtailment_DC
                Data_Table_Out = Data_Table_Out.add_suffix(f" ({unitconversion['units']})")
                
                x_axis_lim = 1.25 * len(RE_Curtailment_DC)
                for column in RE_Curtailment_DC:
                    ax.plot(RE_Curtailment_DC[column], linewidth=3, color=colour_dict[column],
                            label=column)
                    ax.legend(loc='lower left',bbox_to_anchor=(1,0),
                              facecolor='inherit', frameon=True)
                    ax.set_ylabel(f"PV + Wind Curtailment ({unitconversion['units']})",  color='black', rotation='vertical')

            ax.set_xlabel('Hours',  color='black', rotation='horizontal')
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.tick_params(axis='y', which='major', length=5, width=1)
            ax.tick_params(axis='x', which='major', length=5, width=1)
            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(x, f',.{self.y_axes_decimalpt}f')))
            ax.margins(x=0.01)
            #ax.set_xlim(0, 9490)
            ax.set_xlim(0,x_axis_lim)
            
            ax.set_ylim(bottom=0)
            if mconfig.parser("plot_title_as_region"):
                ax.set_title(zone_input)

            outputs[zone_input] = {'fig': fig2, 'data_table': Data_Table_Out}
        return outputs


    def curt_pen(self, figure_name=None, prop=None, start=None, end=None, 
                  timezone="", start_date_range=None, end_date_range=None):
        
        outputs = {}
        
        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True, "generator_Generation", self.Scenarios),
                      (True, "generator_Available_Capacity", self.Scenarios),
                      (True, f"generator_{self.curtailment_prop}", self.Scenarios),
                      (True, "generator_Total_Generation_Cost", self.Scenarios)]
        
        # Runs get_data to populate mplot_data_dict with all required properties, returns a 1 if required data is missing
        check_input_data = mfunc.get_data(self.mplot_data_dict, properties,self.Marmot_Solutions_folder)

        if 1 in check_input_data:
            return mfunc.MissingInputData()
        
        for zone_input in self.Zones:
            Penetration_Curtailment_out = pd.DataFrame()

            self.logger.info(f"{self.AGG_BY } = {zone_input}")

            for scenario in self.Scenarios:
                self.logger.info(f"Scenario = {scenario}")

                gen = self.mplot_data_dict["generator_Generation"].get(scenario)
                try: #Check for regions missing all generation.
                    gen = gen.xs(zone_input,level = self.AGG_BY)
                except KeyError:
                        self.logger.info(f'No generation in {zone_input}')
                        continue

                avail_gen = self.mplot_data_dict["generator_Available_Capacity"].get(scenario)
                avail_gen = avail_gen.xs(zone_input,level=self.AGG_BY)
                
                re_curt = self.mplot_data_dict[f"generator_{self.curtailment_prop}"].get(scenario)
                try:
                    re_curt = re_curt.xs(zone_input,level=self.AGG_BY)
                except KeyError:
                        self.logger.info(f'No curtailment in {zone_input}')
                        continue
                
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
                if re_curt.empty == True:
                    continue
                else:
                    Max_RE_Curt = max(re_curt.sum(axis=1))
                if pv_curt.empty == True:
                    continue
                else:
                    Max_PV_Curt = max(pv_curt.sum(axis=1))

                # % RE and PV Curtailment Capacity Factor
                if total_pv_curt > 0:
                    RE_Curt_Cap_factor = (total_re_curt/Max_RE_Curt)/no_hours_year
                    PV_Curt_Cap_factor = (total_pv_curt/Max_PV_Curt)/no_hours_year
                else:
                    RE_Curt_Cap_factor = 0
                    PV_Curt_Cap_factor = 0

                # % Curtailment across the year
                if total_re_avail == 0:
                    continue
                else:
                    Prct_RE_curt = (total_re_curt/total_re_avail)*100
                if total_pv_avail == 0:
                    continue
                else:
                    Prct_PV_curt = (total_pv_curt/total_pv_avail)*100

                # Total generation cost
                Total_Gen_Cost = self.mplot_data_dict["generator_Total_Generation_Cost"].get(scenario)
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
            
            if Penetration_Curtailment_out.empty:
                self.logger.warning(f'No Generation in {zone_input}')
                out = mfunc.MissingZoneData()
                outputs[zone_input] = out
                continue
            
            fig1, ax = plt.subplots(figsize=(self.x,self.y))
            for index, row in Penetration_Curtailment_out.iterrows():
                if prop == "PV":
                    ax.scatter(row["% PV Penetration"], row["% PV Curtailment"],
                          marker=row["marker"],  c=row["colour"], s=100, label = row["Scenario"])
                    ax.set_ylabel('% PV Curtailment',  color='black', rotation='vertical')
                    ax.set_xlabel('% PV Penetration',  color='black', rotation='horizontal')

                elif prop == "PV+Wind":
                    ax.scatter(row["% RE Penetration"], row["% RE Curtailment"],
                          marker=row["marker"],  c=row["colour"], s=40, label = row["Scenario"])
                    ax.set_ylabel('% PV + Wind Curtailment',  color='black', rotation='vertical')
                    ax.set_xlabel('% PV + Wind Penetration',  color='black', rotation='horizontal')

            ax.set_ylim(bottom=0)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.tick_params(axis='y', which='major', length=5, width=1)
            ax.tick_params(axis='x', which='major', length=5, width=1)
            ax.margins(x=0.01)
            if mconfig.parser("plot_title_as_region"):
                ax.set_title(zone_input)

            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys(), loc = 'lower right')

            outputs[zone_input] = {'fig': fig1, 'data_table': Data_Table_Out}
        return outputs

    def curt_total(self, figure_name=None, prop=None, start=None, end=None, 
                  timezone="", start_date_range=None, end_date_range=None):

        """
        This module calculates the total curtailment, broken down by technology. 
        It produces a stacked bar plot, with a bar for each scenario.
        """

        outputs = {}
        
        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True, f"generator_{self.curtailment_prop}", self.Scenarios),
                      (True, "generator_Available_Capacity", self.Scenarios)]
        
        # Runs get_data to populate mplot_data_dict with all required properties, returns a 1 if required data is missing
        check_input_data = mfunc.get_data(self.mplot_data_dict, properties,self.Marmot_Solutions_folder)

        if 1 in check_input_data:
            return mfunc.MissingInputData()
        
        for zone_input in self.Zones:
            self.logger.info(f"{self.AGG_BY} = {zone_input}")

            Total_Curtailment_out = pd.DataFrame()
            Total_Available_gen = pd.DataFrame()
            vre_curt_chunks = []
            avail_gen_chunks = []

            for scenario in self.Scenarios:

                self.logger.info(f"Scenario = {scenario}")
                # Adjust list of values to drop from vre_gen_cat depending on if it exhists in processed techs
                #self.vre_gen_cat = [name for name in self.vre_gen_cat if name in self.mplot_data_dict[f"generator_{self.curtailment_prop}"].get(scenario).index.unique(level="tech")]

                vre_collection = {}
                avail_vre_collection = {}

                vre_curt = self.mplot_data_dict[f"generator_{self.curtailment_prop}"].get(scenario)
                try:
                    vre_curt = vre_curt.xs(zone_input,level=self.AGG_BY)
                except KeyError:
                    self.logger.info(f'No curtailment in {zone_input}')
                    continue
                vre_curt = vre_curt[vre_curt.index.isin(self.vre_gen_cat,level='tech')]

                avail_gen = self.mplot_data_dict["generator_Available_Capacity"].get(scenario)
                try: #Check for regions missing all generation.
                    avail_gen = avail_gen.xs(zone_input,level = self.AGG_BY)
                except KeyError:
                        self.logger.info(f'No available generation in {zone_input}')
                        continue
                avail_gen = avail_gen[avail_gen.index.isin(self.vre_gen_cat,level='tech')]
                
                all_empty = True
                if pd.isna(start_date_range) == False:
                    self.logger.info(f"Plotting specific date range: \
                    {str(start_date_range)} to {str(end_date_range)}")
                for vre_type in self.vre_gen_cat:
                    try:
                        vre_curt_type = vre_curt.xs(vre_type,level='tech')
                    except KeyError:
                        self.logger.info(f'No {vre_type} in {zone_input}')
                        continue

                    avail_gen_type = avail_gen.xs(vre_type,level='tech')
                    
                    # Code to index data by date range, if a date range is listed in marmot_plot_select.csv
                    if pd.isna(start_date_range) == False:
                        avail_gen_type = avail_gen_type.groupby(['timestamp']).sum()
                        vre_curt_type = vre_curt_type.groupby(['timestamp']).sum()
                        vre_curt_type = vre_curt_type[start_date_range : end_date_range]
                        avail_gen_type = avail_gen_type[start_date_range : end_date_range]
                    
                    if vre_curt_type.empty is False and avail_gen_type.empty is False:
                        all_empty = False
                    vre_collection[vre_type] = float(vre_curt_type.sum())
                    avail_vre_collection[vre_type] = float(avail_gen_type.sum())
                if all_empty:
                    self.logger.warning('No data in selected Date Range')
                    continue
                
                vre_table = pd.DataFrame(vre_collection,index=[scenario])
                avail_gen_table = pd.DataFrame(avail_vre_collection,index=[scenario])

                vre_curt_chunks.append(vre_table)
                avail_gen_chunks.append(avail_gen_table)
            
            if not vre_curt_chunks:
                outputs[zone_input] = mfunc.MissingZoneData()
                continue
                
            Total_Curtailment_out = pd.concat(vre_curt_chunks, axis=0, sort=False)
            Total_Available_gen = pd.concat(avail_gen_chunks, axis=0, sort=False)
            
            vre_pct_curt = Total_Curtailment_out.sum(axis=1)/Total_Available_gen.sum(axis=1)
            
            Total_Curtailment_out.index = Total_Curtailment_out.index.str.replace('_',' ')
            
            if Total_Curtailment_out.empty == True:
                outputs[zone_input] = mfunc.MissingZoneData()
                continue
            
            
            # unit conversion return divisor and energy units
            unitconversion = mfunc.capacity_energy_unitconversion(max(Total_Curtailment_out.sum()))
            Total_Curtailment_out = Total_Curtailment_out/unitconversion['divisor'] 
            
            # Data table of values to return to main program            
            Data_Table_Out = Total_Curtailment_out
            Data_Table_Out = Data_Table_Out.add_suffix(f" ({unitconversion['units']}h)")
            
            fig3, ax = plt.subplots(figsize=(self.x,self.y))
            Total_Curtailment_out.plot.bar(stacked=True, 
                             color=[self.PLEXOS_color_dict.get(x, '#333333') for x in Total_Curtailment_out.columns],
                             edgecolor='black', linewidth='0.1', ax=ax)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_ylabel(f"Total Curtailment ({unitconversion['units']}h)",  color='black', rotation='vertical')
            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(x, f',.{self.y_axes_decimalpt}f')))
            
            # Set x-tick labels 
            if len(self.custom_xticklabels) > 1:
                tick_labels = self.custom_xticklabels
            else:
                tick_labels = Total_Curtailment_out.index
            mfunc.set_barplot_xticklabels(tick_labels, ax=ax)

            ax.tick_params(axis='y', which='major', length=5, width=1)
            ax.tick_params(axis='x', which='major', length=5, width=1)
            ax.margins(x=0.01)
            if mconfig.parser("plot_title_as_region"):
                ax.set_title(zone_input)

            handles, labels = ax.get_legend_handles_labels()
            ax.legend(reversed(handles), reversed(labels), loc='lower left',bbox_to_anchor=(1,0),
                          facecolor='inherit', frameon=True)

            curt_totals = Total_Curtailment_out.sum(axis=1)
            #inserts total bar value above each bar
            for k, patch in enumerate(ax.patches):
                height = curt_totals[k]
                width = patch.get_width()
                x, y = patch.get_xy()
                ax.text(x+width/2,
                    y+height + 0.05*max(ax.get_ylim()),
                    '{:.2%}\n|{:,.2f}|'.format(vre_pct_curt[k],curt_totals[k]),
                    horizontalalignment='center',
                    verticalalignment='center', fontsize=11, color='red')
                
                if k>=len(vre_pct_curt)-1:
                    break

            outputs[zone_input] = {'fig': fig3, 'data_table': Data_Table_Out}
        return outputs

    def curt_total_diff(self,figure_name=None, prop=None, start=None, end=None, 
                  timezone="", start_date_range=None, end_date_range=None):

        """
        This module calculates the total curtailment, broken down by technology. 
        It produces a stacked bar plot, with a bar for each scenario.
        """

        return mfunc.UnderDevelopment()

        outputs = {}
        properties = [(True, f"generator_{self.curtailment_prop}", self.Scenarios),
                      (True, "generator_Available_Capacity", self.Scenarios)]
        
        # Runs get_data to populate mplot_data_dict with all required properties, returns a 1 if required data is missing
        check_input_data = mfunc.get_data(self.mplot_data_dict, properties,self.Marmot_Solutions_folder)

        # Checks if all data required by plot is available, if 1 in list required data is missing
        if 1 in check_input_data:
            outputs = mfunc.MissingInputData()
            return outputs
        
        for zone_input in self.Zones:
            self.logger.info(self.AGG_BY +  " = " + zone_input)

            Total_Curtailment_out = pd.DataFrame()
            Total_Available_gen = pd.DataFrame()
            vre_curt_chunks = []
            avail_gen_chunks = []

            for scenario in self.Scenarios:

                self.logger.info("Scenario = " + scenario)
                # Adjust list of values to drop from vre_gen_cat depending on if it exhists in processed techs
                #self.vre_gen_cat = [name for name in self.vre_gen_cat if name in curtailment_collection.get(scenario).index.unique(level="tech")]

                vre_collection = {}
                avail_vre_collection = {}

                vre_curt = self.mplot_data_dict[f"generator_{self.curtailment_prop}"].get(scenario)
                try:
                    vre_curt = vre_curt.xs(zone_input,level=self.AGG_BY)
                except KeyError:
                    self.logger.info('No curtailment in ' + zone_input)
                    continue
                vre_curt = vre_curt[vre_curt.index.isin(self.vre_gen_cat,level='tech')]

                avail_gen = self.mplot_data_dict["generator_Available_Capacity"].get(scenario)
                try: #Check for regions missing all generation.
                    avail_gen = avail_gen.xs(zone_input,level = self.AGG_BY)
                except KeyError:
                        self.logger.info('No available generation in ' + zone_input)
                        continue
                avail_gen = avail_gen[avail_gen.index.isin(self.vre_gen_cat,level='tech')]

                for vre_type in self.vre_gen_cat:
                    try:
                        vre_curt_type = vre_curt.xs(vre_type,level='tech')
                    except KeyError:
                        self.logger.info('No ' + vre_type + ' in ' + zone_input)
                        continue
                    vre_collection[vre_type] = float(vre_curt_type.sum())

                    avail_gen_type = avail_gen.xs(vre_type,level='tech')
                    avail_vre_collection[vre_type] = float(avail_gen_type.sum())

                vre_table = pd.DataFrame(vre_collection,index=[scenario])
                avail_gen_table = pd.DataFrame(avail_vre_collection,index=[scenario])

                vre_curt_chunks.append(vre_table)
                avail_gen_chunks.append(avail_gen_table)
            
            Total_Curtailment_out = pd.concat(vre_curt_chunks, axis=0, sort=False)
            Total_Available_gen = pd.concat(avail_gen_chunks, axis=0, sort=False)
            
            vre_pct_curt = Total_Curtailment_out.sum(axis=1)/Total_Available_gen.sum(axis=1)

            #Change to a diff on the first scenario.
            print(Total_Curtailment_out)
            Total_Curtailment_out = Total_Curtailment_out-Total_Curtailment_out.xs(self.Scenarios[0]) 
            Total_Curtailment_out.drop(self.Scenarios[0],inplace=True) #Drop base entry
            
            Total_Curtailment_out.index = Total_Curtailment_out.index.str.replace('_',' ')
            
            # Data table of values to return to main program
            Data_Table_Out = Total_Curtailment_out

            if Total_Curtailment_out.empty == True:
                outputs[zone_input] = mfunc.MissingZoneData()
                continue
            
            # unit conversion return divisor and energy units
            unitconversion = mfunc.capacity_energy_unitconversion(max(Total_Curtailment_out.sum()))
            Total_Curtailment_out = Total_Curtailment_out/unitconversion['divisor'] 
            
            fig3, ax= plt.subplots(figsize=(self.x,self.y))
            Total_Curtailment_out.plot.bar(stacked=True,
                             color=[self.PLEXOS_color_dict.get(x, '#333333') for x in Total_Curtailment_out.columns],
                             edgecolor='black', linewidth='0.1',ax=ax)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_ylabel('Total Curtailment ({}h)'.format(unitconversion['units']),  color='black', rotation='vertical')
            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(x, f',.{self.y_axes_decimalpt}f')))
            
            # Set x-tick labels 
            if len(self.custom_xticklabels) > 1:
                tick_labels = self.custom_xticklabels
            else:
                tick_labels = Total_Curtailment_out.index
            mfunc.set_barplot_xticklabels(tick_labels, ax=ax)

            ax.tick_params(axis='y', which='major', length=5, width=1)
            ax.tick_params(axis='x', which='major', length=5, width=1)
            ax.margins(x=0.01)
            if mconfig.parser("plot_title_as_region"):
                ax.set_title(zone_input)

            handles, labels = ax.get_legend_handles_labels()
            ax.legend(reversed(handles), reversed(labels), loc='lower left',bbox_to_anchor=(1,0),
                          facecolor='inherit', frameon=True)

            curt_totals = Total_Curtailment_out.sum(axis=1)
            print(Total_Curtailment_out)
            print(curt_totals)
            #inserts total bar value above each bar
            k=0
            for i in ax.patches:
                height = curt_totals[k]
                width = i.get_width()
                x, y = i.get_xy()
                ax.text(x+width/2,
                    y+height + 0.05*max(ax.get_ylim()),
                    '{:.2%}\n|{:,.2f}|'.format(vre_pct_curt[k],curt_totals[k]),
                    horizontalalignment='center',
                    verticalalignment='center', fontsize=11, color='red')
                k += 1
                if k>=len(vre_pct_curt):
                    break

            outputs[zone_input] = {'fig': fig3, 'data_table': Data_Table_Out}
        return outputs

    def curt_ind(self, figure_name=None, prop=None, start=None, end=None, 
                  timezone="", start_date_range=None, end_date_range=None):

        """
        This module calculates the curtailment, as a percentage of total generation, of individual generators.
        The generators are specified as a list of strings in the fourth column of Marmot_plot_select.csv.  
        The module prints out two .csvs: 
            -one that contains curtailment, in percent, for each scenario and site. 
            -the other contains total generation, in TWh, for each scenario and site.
    
        """

        outputs = {}
        
        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True, "generator_Generation", self.Scenarios),
                      (True, "generator_Available_Capacity", self.Scenarios),
                      (True, f"generator_{self.curtailment_prop}", self.Scenarios)]
        
        # Runs get_data to populate mplot_data_dict with all required properties, returns a 1 if required data is missing
        check_input_data = mfunc.get_data(self.mplot_data_dict, properties,self.Marmot_Solutions_folder)

        if 1 in check_input_data:
            return mfunc.MissingInputData()

        Total_Curtailment_Out_perc = pd.DataFrame()
        Total_Curt = pd.DataFrame()
        Total_Gen = pd.DataFrame()
        Gen_8760 = pd.DataFrame()
        scen_idx = -1

        chunks = []

        for scenario in self.Scenarios:
            scen_idx += 1
            self.logger.info(f"Scenario = {scenario}")

            vre_curt = self.mplot_data_dict[f"generator_{self.curtailment_prop}"].get(scenario)
            gen = self.mplot_data_dict["generator_Generation"].get(scenario)
            cap = self.mplot_data_dict["generator_Available_Capacity"].get(scenario)
            
            #Select only lines specified in Marmot_plot_select.csv.
            select_sites = prop.split(",") 
            select_sites = [site[1:] if site[0] == ' ' else site for site in select_sites]
            self.logger.info('Plotting curtailment only for sites specified in Marmot_plot_select.csv')

            ti = gen.index.get_level_values('timestamp').unique()

            site_idx = -1
            sites = pd.Series()
            sites_gen = pd.Series()
            curt_tots = pd.Series()
            chunks_scen = []
            #vre_curt = vre_curt.reset_index()

            for site in select_sites:
                if site in gen.index.get_level_values('gen_name').unique():
                    site_idx += 1
                    gen_site = gen.xs(site,level = 'gen_name')
                    cap_site = cap.xs(site,level = 'gen_name') 
                    curt = cap_site - gen_site
                    curt = vre_curt.xs(site,level = 'gen_name')

                    curt_tot = curt.sum()
                    gen_tot = gen_site.sum()
                    curt_perc = pd.Series(curt_tot / gen_tot)

                    levels2drop = [level for level in gen_site.index.names if level != 'timestamp']
                    gen_site = gen_site.droplevel(levels2drop)


                else:
                    curt_perc = pd.Series([0])
                    curt_tot = pd.Series([0])
                    gen_tot = pd.Series([0])
                    curt = pd.Series([0] * len(ti),name = site,index = ti)

                gen_tot.columns = [site]
                curt_perc.columns = [site]
                curt_tot.columns = [site]
                curt.columns = [site]

                sites_gen = sites_gen.append(gen_tot)
                sites = sites.append(curt_perc)
                curt_tots = curt_tots.append(curt_tot)
                chunks_scen.append(curt)

            curt_8760_scen = pd.concat(chunks_scen,axis = 1)
            scen_name = pd.Series([scenario] * len(curt_8760_scen),name = 'Scenario')
            curt_8760_scen = curt_8760_scen.set_index([scen_name],append = True)
            chunks.append(curt_8760_scen)

            sites.name = scenario
            sites.index = select_sites
            curt_tots.name = scenario
            curt_tots.index = select_sites
            sites_gen.name = scenario
            sites_gen.index = select_sites
            Total_Curtailment_Out_perc = pd.concat([Total_Curtailment_Out_perc,sites],axis = 1)
            Total_Gen = pd.concat([Total_Gen,sites_gen],axis = 1)
            Total_Curt = pd.concat([Total_Curt,curt_tots],axis = 1)

        Curt_8760 = pd.concat(chunks,axis = 0, copy = False)
        Curt_8760.to_csv(os.path.join(self.Marmot_Solutions_folder, 'Figures_Output',self.AGG_BY + '_curtailment',figure_name + '_8760.csv'))

        Total_Gen = Total_Gen / 1000000
        Total_Curtailment_Out_perc.T.to_csv(os.path.join(self.Marmot_Solutions_folder, 'Figures_Output',self.AGG_BY + '_curtailment',figure_name + '.csv'))
        Total_Gen.T.to_csv(os.path.join(self.Marmot_Solutions_folder, 'Figures_Output',self.AGG_BY + '_curtailment',figure_name + '_gen.csv'))
                
        fig1, ax = plt.subplots(figsize=(9,6))
        Total_Curtailment_Out_perc.plot.bar(stacked = False, edgecolor='black', linewidth='0.1',ax=ax)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylabel('Curtailment (%)',  color='black', rotation='vertical')
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1,decimals = 0))         #adds % to y axis data
        
        # Set x-tick labels 
        tick_labels = Total_Curtailment_Out_perc.index
        mfunc.set_barplot_xticklabels(tick_labels, ax=ax)

        ax.tick_params(axis='y', which='major', length=5, width=1)
        ax.tick_params(axis='x', which='major', length=5, width=1)
        
        unitconversion = mfunc.capacity_energy_unitconversion(Total_Curt.values.max())
        Total_Curt = Total_Curt/unitconversion['divisor'] 
        
        Total_Curt = round(Total_Curt,2)
        Total_Curt = Total_Curt.melt()
        #inserts total bar value above each bar, 
        #but only if it is the max in the bar group.
        #to do this, take the n highest patches, where n is the number of bar broups (select_sites)
        heights = [patch.get_height() for patch in ax.patches]
        heights.sort(reverse = True)
        toph = heights[0:len(select_sites)]
        for k, patch in enumerate(ax.patches):
            height = patch.get_height()
            if height in toph:
                width = patch.get_width()
                x, y = patch.get_xy()
                ax.text(x+width/2,y + height + 0.05*max(ax.get_ylim()),
                    str(Total_Curt.iloc[k][1]) + f" {unitconversion['units']}h",
                    horizontalalignment='center',
                    verticalalignment='center', fontsize=11)

        fig1.savefig(os.path.join(self.Marmot_Solutions_folder,'Figures_Output',self.AGG_BY + '_curtailment',figure_name + '.svg'),dpi=600, bbox_inches='tight')
        outputs = mfunc.DataSavedInModule()
        return outputs


    def average_diurnal_curt(self, figure_name=None, prop=None, start=None, end=None, 
                  timezone=None, start_date_range=None, end_date_range=None):
        """Average diurnal renewable curtailment plot
        """
        
        outputs = {}
        
        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True,f"generator_{self.curtailment_prop}",self.Scenarios)]
        
        # Runs get_data to populate mplot_data_dict with all required properties, returns a 1 if required data is missing
        check_input_data = mfunc.get_data(self.mplot_data_dict, properties,self.Marmot_Solutions_folder)

        if 1 in check_input_data:
            return mfunc.MissingInputData()

        for zone_input in self.Zones:
            self.logger.info(f"{self.AGG_BY} = {zone_input}")
                        
            chunks = []
            for scenario in self.Scenarios:
                self.logger.info(f"Scenario = {scenario}")

                re_curt = self.mplot_data_dict[f"generator_{self.curtailment_prop}"].get(scenario)

                # Timeseries [MW] RE curtailment [MWh]
                try: #Check for regions missing all generation.
                    re_curt = re_curt.xs(zone_input,level = self.AGG_BY)
                except KeyError:
                        self.logger.info(f'No curtailment in {zone_input}')
                        continue

                re_curt = re_curt.groupby(["timestamp"]).sum()
                re_curt = re_curt.squeeze() #Convert to Series
                
                if pd.isna(start_date_range) == False:
                    self.logger.info(f"Plotting specific date range: \
                    {str(start_date_range)} to {str(end_date_range)}")
                    re_curt = re_curt[start_date_range : end_date_range]
                    
                    if re_curt.empty is True: 
                        self.logger.warning('No data in selected Date Range')
                        continue
                
                interval_count = mfunc.get_sub_hour_interval_count(re_curt)
                re_curt = re_curt/interval_count
                # Group data by hours and find mean across entire range 
                re_curt = re_curt.groupby([re_curt.index.hour]).mean()
                
                # reset index to datetime 
                re_curt.index = pd.date_range("2024-01-01", periods=24, freq="H")
                re_curt.rename(scenario, inplace=True)
                chunks.append(re_curt)

            RE_Curtailment_DC = pd.concat(chunks, axis=1, sort=False)

            # Replace _ with white space
            RE_Curtailment_DC.columns = RE_Curtailment_DC.columns.str.replace('_',' ')

            # Create Dictionary from scenario names and color list
            colour_dict = dict(zip(RE_Curtailment_DC.columns, self.color_list))

            fig, axs = mfunc.setup_plot(squeeze=False)
            # flatten object
            ax = axs[0]

            unitconversion = mfunc.capacity_energy_unitconversion(RE_Curtailment_DC.values.max())
            RE_Curtailment_DC = RE_Curtailment_DC / unitconversion['divisor']
            Data_Table_Out = RE_Curtailment_DC.copy()
            Data_Table_Out.index = pd.date_range("2024-01-01", periods=24, freq="H").time
            Data_Table_Out = Data_Table_Out.add_suffix(f" ({unitconversion['units']})")
            
            for column in RE_Curtailment_DC:
                ax.plot(RE_Curtailment_DC[column], linewidth=2, color=colour_dict[column],
                        label=column)
                
            ax.legend(loc='lower left',bbox_to_anchor=(1,0),
                          facecolor='inherit', frameon=True)
            ax.set_ylabel(f"Average Diurnal Curtailment ({unitconversion['units']})",  color='black', rotation='vertical')
            
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.tick_params(axis='y', which='major', length=5, width=1)
            ax.tick_params(axis='x', which='major', length=5, width=1)
            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(x, f',.{self.y_axes_decimalpt}f')))
            ax.margins(x=0.01)
            
            # Set time ticks
            locator = mdates.AutoDateLocator(minticks=8, maxticks=12)
            formatter = mdates.ConciseDateFormatter(locator)
            formatter.zero_formats[3] = '%H:%M'
            formatter.show_offset = False
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)
        
            ax.set_ylim(bottom=0)
            if mconfig.parser("plot_title_as_region"):
                ax.set_title(zone_input)

            outputs[zone_input] = {'fig': fig, 'data_table': Data_Table_Out}
        return outputs

