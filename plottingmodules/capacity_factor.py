# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 13:20:56 2019

This code creates total generation stacked bar plots and is called from Marmot_plot_main.py


@author: dlevie
"""

import pandas as pd
import matplotlib.ticker as mtick
import numpy as np
import os
import marmot_plot_functions as mfunc
import logging

#===============================================================================

class mplot(object):

    def __init__(self, argument_dict):
        # iterate over items in argument_dict and set as properties of class
        # see key_list in Marmot_plot_main for list of properties
        for prop in argument_dict:
            self.__setattr__(prop, argument_dict[prop])
        self.logger = logging.getLogger('marmot_plot.'+__name__)

    def cf(self):
        outputs = {}
        gen_collection = {}
        cap_collection = {}
        check_input_data = []
        
        check_input_data.extend([mfunc.get_data(cap_collection,"generator_Installed_Capacity", self.Marmot_Solutions_folder, self.Multi_Scenario)])
        check_input_data.extend([mfunc.get_data(gen_collection,"generator_Generation", self.Marmot_Solutions_folder, self.Multi_Scenario)])
        
        # Checks if all data required by plot is available, if 1 in list required data is missing
        if 1 in check_input_data:
            outputs = mfunc.MissingInputData()
            return outputs
        
        for zone_input in self.Zones:
            CF_all_scenarios = pd.DataFrame()
            self.logger.info(self.AGG_BY + " = " + zone_input)

            for scenario in self.Multi_Scenario:

                self.logger.info("Scenario = " + str(scenario))
                Gen = gen_collection.get(scenario)
                try: #Check for regions missing all generation.
                    Gen = Gen.xs(zone_input,level = self.AGG_BY)
                except KeyError:
                        self.logger.warning('No data in ' + zone_input)
                        continue
                Gen = mfunc.df_process_gen_inputs(Gen,self.ordered_gen)
                
                # Calculates interval step to correct for MWh of generation
                time_delta = Gen.index[1] - Gen.index[0]
                duration = Gen.index[len(Gen)-1] - Gen.index[0]
                duration = duration + time_delta #Account for last timestep.
                # Finds intervals in 60 minute period
                #interval_count = 60/(time_delta/np.timedelta64(1, 'm'))
                duration_hours = duration/np.timedelta64(1,'h')     #Get length of time series in hours for CF calculation.

                if self.prop == 'Date Range':
                    self.logger.info("Plotting specific date range: \
                    {} to {}".format(str(self.start_date),str(self.end_date)))
                    Gen = Gen[self.start_date : self.end_date]
                

                #Gen = Gen/interval_count
                Total_Gen = Gen.sum(axis=0)
                Total_Gen.rename(scenario, inplace = True)

                Cap = cap_collection.get(scenario)
                Cap = Cap.xs(zone_input,level = self.AGG_BY)
                Cap = mfunc.df_process_gen_inputs(Cap, self.ordered_gen)
                Cap = Cap.T.sum(axis = 1)  #Rotate and force capacity to a series.
                Cap.rename(scenario, inplace = True)

                #Calculate CF
                CF = Total_Gen/(Cap * duration_hours)
                CF.rename(scenario, inplace = True)
                CF_all_scenarios = pd.concat([CF_all_scenarios, CF], axis=1, sort=False)
                CF_all_scenarios = CF_all_scenarios.fillna(0, axis = 0)

            CF_all_scenarios.columns = CF_all_scenarios.columns.str.replace('_',' ')
            CF_all_scenarios.index = CF_all_scenarios.index.str.wrap(10, break_long_words = False)

            if CF_all_scenarios.empty == True:
                outputs[zone_input] = mfunc.MissingZoneData()
                continue
            fig1 = CF_all_scenarios.plot.bar(stacked = False, figsize=(9,6), rot=0,
                                 color = self.color_list,edgecolor='black', linewidth='0.1')

            fig1.spines['right'].set_visible(False)
            fig1.spines['top'].set_visible(False)
            fig1.set_ylabel('Capacity Factor',  color='black', rotation='vertical')
            #adds % to y axis data
            fig1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            fig1.tick_params(axis='y', which='major', length=5, width=1)
            fig1.tick_params(axis='x', which='major', length=5, width=1)

            # handles, labels = fig1.get_legend_handles_labels()

            # #Legend 1
            # leg1 = fig1.legend(handles, labels, loc='lower left',bbox_to_anchor=(1,0),
            #               facecolor='inherit', frameon=True)

            outputs[zone_input] = {'fig': fig1, 'data_table': CF_all_scenarios}

        return outputs

    def avg_output_when_committed(self):
        outputs = {}
        gen_collection = {}
        cap_collection = {}
        check_input_data = []
        
        check_input_data.extend([mfunc.get_data(cap_collection,"generator_Installed_Capacity", self.Marmot_Solutions_folder, self.Multi_Scenario)])
        check_input_data.extend([mfunc.get_data(gen_collection,"generator_Generation", self.Marmot_Solutions_folder, self.Multi_Scenario)])
        
        # Checks if all data required by plot is available, if 1 in list required data is missing
        if 1 in check_input_data:
            outputs = mfunc.MissingInputData()
            return outputs
        
        for zone_input in self.Zones:
            CF_all_scenarios = pd.DataFrame()
            self.logger.info(self.AGG_BY + " = " + zone_input)

            for scenario in self.Multi_Scenario:
                self.logger.info("Scenario = " + str(scenario))
                Gen = gen_collection.get(scenario)
                try: #Check for regions missing all generation.
                    Gen = Gen.xs(zone_input,level = self.AGG_BY)
                except KeyError:
                        self.logger.warning('No data in ' + zone_input)
                        continue
                Gen = Gen.reset_index()
                Gen.tech = Gen.tech.astype("category")
                Gen.tech.cat.set_categories(self.ordered_gen, inplace=True)
                Gen = Gen.rename(columns = {0:"Output (MWh)"})
                techs = list(Gen['tech'].unique())
                Gen = Gen[Gen['tech'].isin(self.thermal_gen_cat)]
                Cap = cap_collection.get(scenario)
                Cap = Cap.xs(zone_input,level = self.AGG_BY)
                Cap = Cap.reset_index()
                Cap = Cap.drop(columns = ['timestamp','tech'])
                Cap = Cap.rename(columns = {0:"Installed Capacity (MW)"})
                Gen = pd.merge(Gen,Cap, on = 'gen_name')
                Gen.set_index('timestamp',inplace=True)
                
                if self.prop == 'Date Range':
                    self.logger.info("Plotting specific date range: \
                    {} to {}".format(str(self.start_date),str(self.end_date)))
                    # sort_index added see https://github.com/pandas-dev/pandas/issues/35509
                    Gen = Gen.sort_index()[self.start_date : self.end_date]

                #Calculate CF individually for each plant, since we need to take out all zero rows.
                tech_names = Gen['tech'].unique()
                CF = pd.DataFrame(columns = tech_names,index = [scenario])
                for tech_name in tech_names:
                    stt = Gen.loc[Gen['tech'] == tech_name]
                    if not all(stt['Output (MWh)'] == 0):

                        gen_names = stt['gen_name'].unique()
                        cfs = []
                        caps = []
                        for gen in gen_names:
                            sgt = stt.loc[stt['gen_name'] == gen]
                            if not all(sgt['Output (MWh)'] == 0):

                                time_delta = sgt.index[1] - sgt.index[0]  # Calculates interval step to correct for MWh of generation.
                                sgt = sgt[sgt['Output (MWh)'] !=0] #Remove time intervals when output is zero.
                                duration_hours = (len(sgt) * time_delta + time_delta)/np.timedelta64(1,'h')     #Get length of time series in hours for CF calculation
                                total_gen = sgt['Output (MWh)'].sum()
                                cap = sgt['Installed Capacity (MW)'].mean()

                                #Calculate CF
                                cf = total_gen/(cap * duration_hours)
                                cfs.append(cf)
                                caps.append(cap)

                        #Find average "CF" (average output when committed) for this technology, weighted by capacity.
                        cf = np.average(cfs,weights = caps)
                        CF[tech_name] = cf

                CF_all_scenarios = CF_all_scenarios.append(CF)

            if CF_all_scenarios.empty == True:
                outputs[zone_input] = mfunc.MissingZoneData()
                continue
            fig2 = CF_all_scenarios.T.plot.bar(stacked = False, figsize=(6,4), rot=0,
                                 color = self.color_list,edgecolor='black', linewidth='0.1')

            fig2.spines['right'].set_visible(False)
            fig2.spines['top'].set_visible(False)
            fig2.set_ylabel('Average Output When Committed',  color='black', rotation='vertical')
            #adds % to y axis data
            fig2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            fig2.tick_params(axis='y', which='major', length=5, width=1)
            fig2.tick_params(axis='x', which='major', length=5, width=1)

            outputs[zone_input] = {'fig': fig2, 'data_table': CF_all_scenarios.T}
        return outputs


    def time_at_min_gen(self):
        outputs = {}
        gen_collection = {}
        cap_collection = {}
        gen_hours_at_min_collection = {}
        check_input_data = []
        
        check_input_data.extend([mfunc.get_data(cap_collection,"generator_Installed_Capacity", self.Marmot_Solutions_folder, self.Multi_Scenario)])
        check_input_data.extend([mfunc.get_data(gen_collection,"generator_Generation", self.Marmot_Solutions_folder, self.Multi_Scenario)])
        check_input_data.extend([mfunc.get_data(gen_hours_at_min_collection,"generator_Hours_at_Minimum", self.Marmot_Solutions_folder, self.Multi_Scenario)])
        
        # Checks if all data required by plot is available, if 1 in list required data is missing
        if 1 in check_input_data:
            outputs = mfunc.MissingInputData()
            return outputs
        
        for zone_input in self.Zones:
            self.logger.info(self.AGG_BY + " = " + zone_input)

            time_at_min = pd.DataFrame()

            for scenario in self.Multi_Scenario:
                self.logger.info("Scenario = " + str(scenario))

                Min = gen_hours_at_min_collection.get(scenario)
                Min = Min.xs(zone_input,level = self.AGG_BY)
                Min = Min.reset_index()
                Min.index = Min.gen_name
                Min = Min.drop(columns = ['gen_name','timestamp','region','zone','Usual','Country','CountryInterconnect'])
                Min = Min.rename(columns = {0:"Hours at Minimum"})


                Gen = gen_collection.get(scenario)
                try: #Check for regions missing all generation.
                    Gen = Gen.xs(zone_input,level = self.AGG_BY)
                except KeyError:
                        self.logger.warning('No data in ' + zone_input)
                        continue
                Gen = Gen.reset_index()
                Gen.tech = Gen.tech.astype("category")
                Gen.tech.cat.set_categories(self.ordered_gen, inplace=True)
                Gen = Gen.drop(columns = ['region'])
                Gen = Gen.rename(columns = {0:"Output (MWh)"})
                Gen = Gen[~Gen['tech'].isin(['PV','Wind','Hydro','CSP','Storage','Other'])]
                Gen.index = Gen.timestamp

                Cap = cap_collection.get(scenario)
                Cap = Cap.xs(zone_input,level = self.AGG_BY)
                Caps = Cap.groupby('gen_name').mean()
                Caps.reset_index()
                Caps = Caps.rename(columns = {0: 'Installed Capacity (MW)'})
                Min = pd.merge(Min,Caps, on = 'gen_name')

                #Find how many hours each generator was operating, for the denominator of the % time at min gen.
                #So remove all zero rows.
                Gen = Gen.loc[Gen['Output (MWh)'] != 0]
                online_gens = Gen.gen_name.unique()
                Min = Min.loc[online_gens]
                Min['hours_online'] = Gen.groupby('gen_name')['Output (MWh)'].count()
                Min['fraction_at_min'] = Min['Hours at Minimum'] / Min.hours_online

                tech_names = Min.tech.unique()
                time_at_min_individ = pd.DataFrame(columns = tech_names, index = [scenario])
                for tech_name in tech_names:
                    stt = Min.loc[Min['tech'] == tech_name]
                    output = np.average(stt.fraction_at_min,weights = stt['Installed Capacity (MW)'])
                    time_at_min_individ[tech_name] = output

                time_at_min = time_at_min.append(time_at_min_individ)

            if time_at_min.empty == True:
                outputs[zone_input] = mfunc.MissingZoneData()
                continue
            fig3 = time_at_min.T.plot.bar(stacked = False, figsize=(9,6), rot=0,
                                 color = self.color_list,edgecolor='black', linewidth='0.1')

            fig3.spines['right'].set_visible(False)
            fig3.spines['top'].set_visible(False)
            fig3.set_ylabel('Percentage of time online at minimum generation',  color='black', rotation='vertical')
            #adds % to y axis data
            fig3.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            fig3.tick_params(axis='y', which='major', length=5, width=1)
            fig3.tick_params(axis='x', which='major', length=5, width=1)

            outputs[zone_input] = {'fig': fig3, 'data_table': time_at_min.T}
        return outputs
