# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 11:40:37 2021

@author: rhousema
"""

#import os
import pandas as pd
import matplotlib.pyplot as plt
#from collections import OrderedDict
import matplotlib as mpl
import logging
import marmot.plottingmodules.marmot_plot_functions as mfunc
import marmot.config.mconfig as mconfig
#import matplotlib.ticker as mtick


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
        
        self.mplot_data_dict = {}


# More philippines plots

# Monthly average diurnal re curtailment plot

    def average_diurnal_curt(self, figure_name=None, prop=None, start=None, end=None, 
                  timezone=None, start_date_range=None, end_date_range=None):
        
        outputs = {}
        
        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True,"generator_Capacity_Curtailed",self.Scenarios)]
        
        # Runs get_data to populate mplot_data_dict with all required properties, returns a 1 if required data is missing
        check_input_data = mfunc.get_data(self.mplot_data_dict, properties,self.Marmot_Solutions_folder)

        if 1 in check_input_data:
            return mfunc.MissingInputData()

        for zone_input in self.Zones:
            self.logger.info(f"{self.AGG_BY} = {zone_input}")
            
            RE_Curtailment_DC = pd.DataFrame()
            #PV_Curtailment_DC = pd.DataFrame()
            
            for scenario in self.Scenarios:
                self.logger.info(f"Scenario = {scenario}")

                re_curt = self.mplot_data_dict["generator_Capacity_Curtailed"].get(scenario)

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
                    #pv_curt = pv_curt[start_date_range : end_date_range]
                    
                    if re_curt.empty is True: 
                        self.logger.warning('No data in selected Date Range')
                        continue
                    
                re_curt = re_curt.groupby([re_curt.index.floor('d')]).sum()
                interval_count = mfunc.get_interval_count(re_curt)
                re_curt = re_curt*interval_count
                


                re_curt.rename(scenario, inplace=True)
                # pv_cdc.rename(scenario, inplace=True)

                RE_Curtailment_DC = pd.concat([RE_Curtailment_DC, re_curt], axis=1, sort=False)
                #PV_Curtailment_DC = pd.concat([PV_Curtailment_DC, pv_cdc], axis=1, sort=False)

            # Remove columns that have values less than 1
            #RE_Curtailment_DC = RE_Curtailment_DC.loc[:, (RE_Curtailment_DC >= 1).any(axis=0)]


            # Replace _ with white space
            RE_Curtailment_DC.columns = RE_Curtailment_DC.columns.str.replace('_',' ')
            #PV_Curtailment_DC.columns = PV_Curtailment_DC.columns.str.replace('_',' ')

            # Create Dictionary from scenario names and color list
            colour_dict = dict(zip(RE_Curtailment_DC.columns, self.color_list))

            fig2, ax = plt.subplots(figsize=(self.x*1.5,self.y))

            unitconversion = mfunc.capacity_energy_unitconversion(RE_Curtailment_DC.values.max())
            RE_Curtailment_DC = RE_Curtailment_DC / unitconversion['divisor']
            Data_Table_Out = RE_Curtailment_DC
            Data_Table_Out = Data_Table_Out.add_suffix(f" ({unitconversion['units']})")
            
            for column in RE_Curtailment_DC:
                ax.plot(RE_Curtailment_DC[column], linewidth=3, color=colour_dict[column],
                        label=column)
                ax.legend(loc='lower left',bbox_to_anchor=(1,0),
                          facecolor='inherit', frameon=True)
                ax.set_ylabel(f"RE Curtailment ({unitconversion['units']})",  color='black', rotation='vertical')
            
            #ax.set_xlabel('Hours',  color='black', rotation='horizontal')
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.tick_params(axis='y', which='major', length=5, width=1)
            ax.tick_params(axis='x', which='major', length=5, width=1)
            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(x, f',.{self.y_axes_decimalpt}f')))
            ax.margins(x=0.01)

            
            ax.set_ylim(bottom=0)
            if mconfig.parser("plot_title_as_region"):
                ax.set_title(zone_input)

            outputs[zone_input] = {'fig': fig2, 'data_table': Data_Table_Out}
        return outputs




# Monthly average diurnal re curtailment plot

    def average_diurnal_ue(self, figure_name=None, prop=None, start=None, end=None, 
                  timezone=None, start_date_range=None, end_date_range=None):
        
        outputs = {}
        
        if self.AGG_BY == 'zone':
            agg = 'zone'
        else:
            agg = 'region'
        
        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True,f"{agg}_Unserved_Energy",self.Scenarios)]
        
        # Runs get_data to populate mplot_data_dict with all required properties, returns a 1 if required data is missing
        check_input_data = mfunc.get_data(self.mplot_data_dict, properties,self.Marmot_Solutions_folder)

        if 1 in check_input_data:
            return mfunc.MissingInputData()

        for zone_input in self.Zones:
            self.logger.info(f"{self.AGG_BY} = {zone_input}")
            
            Unserved_Energy_Out = pd.DataFrame()
            #PV_Curtailment_DC = pd.DataFrame()
            
            for scenario in self.Scenarios:
                self.logger.info(f"Scenario = {scenario}")
                
                Unserved_Energy = self.mplot_data_dict[f"{agg}_Unserved_Energy"][scenario]
                try:
                    Unserved_Energy = Unserved_Energy.xs(zone_input,level=self.AGG_BY)
                except KeyError:
                    self.logger.info(f'No unserved energy in {zone_input}')
                
                Unserved_Energy = Unserved_Energy.groupby(["timestamp"]).sum()
                Unserved_Energy = Unserved_Energy.squeeze()

                if pd.isna(start_date_range) == False:
                    self.logger.info(f"Plotting specific date range: \
                    {str(start_date_range)} to {str(end_date_range)}")
                    Unserved_Energy = Unserved_Energy[start_date_range : end_date_range]
                    #pv_curt = pv_curt[start_date_range : end_date_range]
                    
                    if Unserved_Energy.empty is True: 
                        self.logger.warning('No data in selected Date Range')
                        continue
                    
                Unserved_Energy = Unserved_Energy.groupby([Unserved_Energy.index.floor('d')]).sum()
                interval_count = mfunc.get_interval_count(Unserved_Energy)
                Unserved_Energy = Unserved_Energy*interval_count
                


                Unserved_Energy.rename(scenario, inplace=True)
                # pv_cdc.rename(scenario, inplace=True)

                Unserved_Energy_Out = pd.concat([Unserved_Energy_Out, Unserved_Energy], axis=1, sort=False)
                #PV_Curtailment_DC = pd.concat([PV_Curtailment_DC, pv_cdc], axis=1, sort=False)

            # Remove columns that have values less than 1
            #Unserved_Energy_Out = Unserved_Energy_Out.loc[:, (Unserved_Energy_Out >= 1).any(axis=0)]

            # Replace _ with white space
            Unserved_Energy_Out.columns = Unserved_Energy_Out.columns.str.replace('_',' ')
            #PV_Curtailment_DC.columns = PV_Curtailment_DC.columns.str.replace('_',' ')

            # Create Dictionary from scenario names and color list
            colour_dict = dict(zip(Unserved_Energy_Out.columns, self.color_list))

            fig2, ax = plt.subplots(figsize=(self.x*1.5,self.y))

            unitconversion = mfunc.capacity_energy_unitconversion(Unserved_Energy_Out.values.max())
            Unserved_Energy_Out = Unserved_Energy_Out / unitconversion['divisor']
            Data_Table_Out = Unserved_Energy_Out
            Data_Table_Out = Data_Table_Out.add_suffix(f" ({unitconversion['units']})")
            
            for column in Unserved_Energy_Out:
                ax.plot(Unserved_Energy_Out[column], linewidth=3, color=colour_dict[column],
                        label=column)
                ax.legend(loc='lower left',bbox_to_anchor=(1,0),
                          facecolor='inherit', frameon=True)
                ax.set_ylabel(f"Unserved Energy ({unitconversion['units']})",  color='black', rotation='vertical')
            
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.tick_params(axis='y', which='major', length=5, width=1)
            ax.tick_params(axis='x', which='major', length=5, width=1)
            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(x, f',.{self.y_axes_decimalpt}f')))
            ax.margins(x=0.01)

            
            ax.set_ylim(bottom=0)
            if mconfig.parser("plot_title_as_region"):
                ax.set_title(zone_input)

            outputs[zone_input] = {'fig': fig2, 'data_table': Data_Table_Out}
        return outputs