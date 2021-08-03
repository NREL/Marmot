# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 13:20:56 2019

This module creates bar plot of the total volume of generator starts in MW,GW,etc.


@author: dlevie
"""

import pandas as pd
import logging
import matplotlib as mpl
import matplotlib.pyplot as plt
import marmot.plottingmodules.marmot_plot_functions as mfunc
import marmot.config.mconfig as mconfig


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
    
    def capacity_started(self, figure_name=None, prop=None, start=None, end=None, 
                  timezone=None, start_date_range=None, end_date_range=None):
       
        outputs = {}
        
        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True,"generator_Generation",self.Scenarios),
                      (True,"generator_Installed_Capacity",self.Scenarios)]
        
        # Runs get_data to populate mplot_data_dict with all required properties, returns a 1 if required data is missing
        check_input_data = mfunc.get_data(self.mplot_data_dict, properties,self.Marmot_Solutions_folder)

        if 1 in check_input_data:
            return mfunc.MissingInputData()
        
        for zone_input in self.Zones:
            self.logger.info(f"{self.AGG_BY} = {zone_input}")
            cap_started_all_scenarios = pd.DataFrame()

            for scenario in self.Scenarios:

                self.logger.info(f"Scenario = {str(scenario)}")

                Gen = self.mplot_data_dict["generator_Generation"].get(scenario)
                
                try:
                    Gen = Gen.xs(zone_input,level = self.AGG_BY)
                except KeyError:
                    self.logger.warning(f"No installed capacity in : {zone_input}")
                    break
                
                Gen = Gen.reset_index()
                Gen.tech = Gen.tech.astype("category")
                Gen.tech.cat.set_categories(self.ordered_gen, inplace=True)
                # Gen = Gen.drop(columns = ['region'])
                Gen = Gen.rename(columns = {0:"Output (MWh)"})
                Gen = Gen[Gen['tech'].isin(self.thermal_gen_cat)]    #We are only interested in thermal starts/stops.

                Cap = self.mplot_data_dict["generator_Installed_Capacity"].get(scenario)
                Cap = Cap.xs(zone_input,level = self.AGG_BY)
                Cap = Cap.reset_index()
                Cap = Cap.drop(columns = ['timestamp','tech'])
                Cap = Cap.rename(columns = {0:"Installed Capacity (MW)"})
                Gen = pd.merge(Gen,Cap, on = 'gen_name')
                Gen.set_index('timestamp',inplace=True)
                
                if prop == 'Date Range':
                    self.logger.info(f"Plotting specific date range: \
                    {str(start_date_range)} to {str(end_date_range)}")
                    # sort_index added see https://github.com/pandas-dev/pandas/issues/35509
                    Gen = Gen.sort_index()[start_date_range : end_date_range]

                tech_names = Gen['tech'].unique()
                Cap_started = pd.DataFrame(columns = tech_names,index = [scenario])

                for tech_name in tech_names:
                    stt = Gen.loc[Gen['tech'] == tech_name]

                    gen_names = stt['gen_name'].unique()

                    cap_started = 0


                    for gen in gen_names:
                        sgt = stt.loc[stt['gen_name'] == gen]
                        if any(sgt["Output (MWh)"] == 0) and not all(sgt["Output (MWh)"] == 0):   #Check that this generator has some, but not all, uncommited hours.
                            #print('Couting starts for: ' + gen)
                            for idx in range(len(sgt['Output (MWh)']) - 1):
                                    if sgt["Output (MWh)"].iloc[idx] == 0 and not sgt["Output (MWh)"].iloc[idx + 1] == 0:
                                        cap_started = cap_started + sgt["Installed Capacity (MW)"].iloc[idx]
                                      # print('started on '+ timestamp)
                                    # if sgt[0].iloc[idx] == 0 and not idx == 0 and not sgt[0].iloc[idx - 1] == 0:
                                    #     stops = stops + 1

                    Cap_started[tech_name] = cap_started

                cap_started_all_scenarios = cap_started_all_scenarios.append(Cap_started)


                # import time
                    # start = time.time()
                    # for gen in gen_names:
                    #     sgt = stt.loc[stt['gen_name'] == gen]

                    #     if any(sgt[0] == 0) and not all(sgt[0] == 0):   #Check that this generator has some, but not all, uncommited hours.
                    #         zeros = sgt.loc[sgt[0] == 0]

                    #         print('Couting starts and stops for: ' + gen)
                    #         for idx in range(len(zeros['timestamp']) - 1):
                    #                if not zeros['timestamp'].iloc[idx + 1] == pd.Timedelta(1,'h'):
                    #                    starts = starts + 1
                    #                   # print('started on '+ timestamp)
                    #                if not zeros['timestamp'].iloc[idx - 1] == pd.Timedelta(1,'h'):
                    #                    stops = stops + 1

                    # starts_and_stops = [starts,stops]
                    # counts[tech_name] = starts_and_stops


                # end = time.time()
                # elapsed = end - start
                # print('Method 2 (first making a data frame with only 0s, then checking if timestamps > 1 hour) took ' + str(elapsed) + ' seconds')

            if cap_started_all_scenarios.empty == True:
                out = mfunc.MissingZoneData()
                outputs[zone_input] = out
                continue

            unitconversion = mfunc.capacity_energy_unitconversion(cap_started_all_scenarios.values.max())
            
            cap_started_all_scenarios = cap_started_all_scenarios/unitconversion['divisor'] 
            Data_Table_Out = cap_started_all_scenarios.T.add_suffix(f" ({unitconversion['units']}-starts)")
            
            cap_started_all_scenarios.index = cap_started_all_scenarios.index.str.replace('_',' ')
            
            cap_started_all_scenarios, angle = mfunc.check_label_angle(cap_started_all_scenarios, True)
            
            fig1, ax = plt.subplots(figsize=(self.x,self.y))
            cap_started_all_scenarios.T.plot.bar(stacked = False, rot=angle,
                                 color = self.color_list,edgecolor='black', linewidth='0.1',ax=ax)

            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_ylabel(f"Capacity Started ({unitconversion['units']}-starts)",  color='black', rotation='vertical')
            if angle > 0:
                ax.set_xticklabels(cap_started_all_scenarios.columns, ha="right")
                tick_length = 8
            else:
                tick_length = 5
            ax.tick_params(axis='y', which='major', length=tick_length, width=1)
            ax.tick_params(axis='x', which='major', length=tick_length, width=1)
            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(x, f',.{self.y_axes_decimalpt}f')))
            ax.legend(loc='lower left',bbox_to_anchor=(1,0),
                          facecolor='inherit', frameon=True)
            if mconfig.parser("plot_title_as_region"):
                ax.set_title(zone_input)

            outputs[zone_input] = {'fig': fig1, 'data_table': Data_Table_Out}
        return outputs


    def count_ramps(self, figure_name=None, prop=None, start=None, end=None, 
                  timezone=None, start_date_range=None, end_date_range=None):
        
        outputs = {}
        
        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True,"generator_Generation",self.Scenarios),
                      (True,"generator_Installed_Capacity",self.Scenarios)]
        
        # Runs get_data to populate mplot_data_dict with all required properties, returns a 1 if required data is missing
        check_input_data = mfunc.get_data(self.mplot_data_dict, properties,self.Marmot_Solutions_folder)

        if 1 in check_input_data:
            return mfunc.MissingInputData()
        
        for zone_input in self.Zones:
            self.logger.info(f"Zone =  {zone_input}")
            cap_started_chunk = []

            for scenario in self.Scenarios:

                self.logger.info(f"Scenario = {str(scenario)}")
                Gen = self.mplot_data_dict["generator_Generation"].get(scenario)
                Gen = Gen.xs(zone_input,level = self.AGG_BY)

                Gen = Gen.reset_index()
                Gen.tech = Gen.tech.astype("category")
                Gen.tech.cat.set_categories(self.ordered_gen, inplace=True)
                Gen = Gen.rename(columns = {0:"Output (MWh)"})
                Gen = Gen[['timestamp','gen_name','tech','Output (MWh)']]
                Gen = Gen[Gen['tech'].isin(self.thermal_gen_cat)]    #We are only interested in thermal starts/stops.tops.

                Cap = self.mplot_data_dict["generator_Installed_Capacity"].get(scenario)
                Cap = Cap.xs(zone_input,level = self.AGG_BY)
                Cap = Cap.reset_index()
                Cap = Cap.rename(columns = {0:"Installed Capacity (MW)"})
                Cap = Cap[['gen_name','Installed Capacity (MW)']]
                Gen = pd.merge(Gen,Cap, on = ['gen_name'])
                Gen.index = Gen.timestamp
                Gen = Gen.drop(columns = ['timestamp'])

                # Min = pd.read_hdf(os.path.join(Marmot_Solutions_folder, scenario,"Processed_HDF5_folder", scenario + "_formatted.h5"),"generator_Hours_at_Minimum")
                # Min = Min.xs(zone_input, level = AGG_BY)

                if prop == 'Date Range':
                    self.logger.info(f"Plotting specific date range: \
                    {str(start_date_range)} to {str(end_date_range)}")
                    Gen = Gen[start_date_range : end_date_range]

                tech_names = Gen['tech'].unique()
                ramp_counts = pd.DataFrame(columns = tech_names,index = [scenario])

                for tech_name in tech_names:
                    stt = Gen.loc[Gen['tech'] == tech_name]

                    gen_names = stt['gen_name'].unique()

                    up_ramps = 0

                    for gen in gen_names:
                        sgt = stt.loc[stt['gen_name'] == gen]
                        if any(sgt["Output (MWh)"] == 0) and not all(sgt["Output (MWh)"] == 0):   #Check that this generator has some, but not all, uncommited hours.
                            #print('Couting starts for: ' + gen)
                            for idx in range(len(sgt['Output (MWh)']) - 1):
                                    if sgt["Output (MWh)"].iloc[idx] == 0 and not sgt["Output (MWh)"].iloc[idx + 1] == 0:
                                        up_ramps = up_ramps + sgt["Installed Capacity (MW)"].iloc[idx]
                                      # print('started on '+ timestamp)
                                    # if sgt[0].iloc[idx] == 0 and not idx == 0 and not sgt[0].iloc[idx - 1] == 0:
                                    #     stops = stops + 1

                    ramp_counts[tech_name] = up_ramps

                cap_started_chunk.append(ramp_counts)
            
            cap_started_all_scenarios = pd.concat(cap_started_chunk)
            
            if cap_started_all_scenarios.empty == True:
                out = mfunc.MissingZoneData()
                outputs[zone_input] = out
                continue
            
            cap_started_all_scenarios.index = cap_started_all_scenarios.index.str.replace('_',' ')

            unitconversion = mfunc.capacity_energy_unitconversion(cap_started_all_scenarios.values.max())
            
            cap_started_all_scenarios = cap_started_all_scenarios/unitconversion['divisor'] 
            Data_Table_Out = cap_started_all_scenarios.T.add_suffix(f" ({unitconversion['units']}-starts)")
            
            #TODO: use mfunc functions
            cap_started_all_scenarios, angle = mfunc.check_label_angle(cap_started_all_scenarios, True)
            
            fig2, ax = plt.subplots(figsize=(self.x,self.y))
            cap_started_all_scenarios.T.plot.bar(stacked = False, rot=angle,
                                  color = self.color_list,edgecolor='black', linewidth='0.1',ax=ax)

            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_ylabel(f"Capacity Started ({unitconversion['units']}-starts)",  color='black', rotation='vertical')
            if angle > 0:
                ax.set_xticklabels(cap_started_all_scenarios.columns, ha="right")
                tick_length = 8
            else:
                tick_length = 5
            ax.tick_params(axis='y', which='major', length=tick_length, width=1)
            ax.tick_params(axis='x', which='major', length=tick_length, width=1)
            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(x, f',.{self.y_axes_decimalpt}f')))
            ax.legend(loc='lower left',bbox_to_anchor=(1,0),
                          facecolor='inherit', frameon=True)
            if mconfig.parser("plot_title_as_region"):
                ax.set_title(zone_input)

            outputs[zone_input] = {'fig': fig2, 'data_table': Data_Table_Out}
        return outputs
