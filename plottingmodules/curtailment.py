# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 15:23:06 2019

@author: dlevie

This module creates plots that show curtailment

"""

import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict
import matplotlib as mpl
import logging
import marmot_plot_functions as mfunc
import os

#===============================================================================

class mplot(object):
    def __init__(self, argument_dict):
        # iterate over items in argument_dict and set as properties of class
        # see key_list in Marmot_plot_main for list of properties
        for prop in argument_dict:
            self.__setattr__(prop, argument_dict[prop])
        self.logger = logging.getLogger('marmot_plot.'+__name__)
        
    def curt_pen(self):
        outputs = {}
        generation_collection = {}
        avail_gen_collection = {}
        curtailment_collection = {}
        total_gen_cost_collection = {}
        check_input_data = []
        
        check_input_data.extend([mfunc.get_data(generation_collection,"generator_Generation", self.Marmot_Solutions_folder, self.Multi_Scenario)])
        check_input_data.extend([mfunc.get_data(avail_gen_collection,"generator_Available_Capacity", self.Marmot_Solutions_folder, self.Multi_Scenario)])
        check_input_data.extend([mfunc.get_data(curtailment_collection,"generator_Curtailment", self.Marmot_Solutions_folder, self.Multi_Scenario)])
        check_input_data.extend([mfunc.get_data(total_gen_cost_collection,"generator_Total_Generation_Cost", self.Marmot_Solutions_folder, self.Multi_Scenario)])
        
        # Checks if all data required by plot is available, if 1 in list required data is missing
        if 1 in check_input_data:
            outputs = mfunc.MissingInputData()
            return outputs
        
        for zone_input in self.Zones:
#        for zone_input in ['TWPR']:
            Penetration_Curtailment_out = pd.DataFrame()

            self.logger.info(self.AGG_BY +  " = " + zone_input)

            for scenario in self.Multi_Scenario:
                self.logger.info("Scenario = " + scenario)

                gen = generation_collection.get(scenario)
                try: #Check for regions missing all generation.
                    gen = gen.xs(zone_input,level = self.AGG_BY)
                except KeyError:
                        self.logger.info('No generation in %s',zone_input)
                        continue

                avail_gen = avail_gen_collection.get(scenario)
                avail_gen = avail_gen.xs(zone_input,level=self.AGG_BY)

                re_curt = curtailment_collection.get(scenario)
                try:
                    re_curt = re_curt.xs(zone_input,level=self.AGG_BY)
                except KeyError:
                        self.logger.info('No curtailment in %s',zone_input)
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
                Total_Gen_Cost = total_gen_cost_collection.get(scenario)
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
                self.logger.warning('No Generation in %s',zone_input)
                out = mfunc.MissingZoneData()
                outputs[zone_input] = out
                continue

            fig1, ax = plt.subplots(figsize=(6,4))
            for index, row in Penetration_Curtailment_out.iterrows():
                if self.prop == "PV":
                    ax.scatter(row["% PV Penetration"], row["% PV Curtailment"],
                          marker=row["marker"],  c=row["colour"], s=100, label = row["Scenario"])
                    ax.set_ylabel('% PV Curtailment',  color='black', rotation='vertical')
                    ax.set_xlabel('% PV Penetration',  color='black', rotation='horizontal')

                elif self.prop == "PV+Wind":
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

            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys(), loc = 'lower right')

            outputs[zone_input] = {'fig': fig1, 'data_table': Data_Table_Out}
        return outputs

    def curt_duration_curve(self):
        outputs = {}
        curtailment_collection = {}
        check_input_data = []
        
        check_input_data.extend([mfunc.get_data(curtailment_collection,"generator_Curtailment", self.Marmot_Solutions_folder, self.Multi_Scenario)])
        
        # Checks if all data required by plot is available, if 1 in list required data is missing
        if 1 in check_input_data:
            outputs = mfunc.MissingInputData()
            return outputs
        
        RE_Curtailment_DC = pd.DataFrame()
        PV_Curtailment_DC = pd.DataFrame()

        for zone_input in self.Zones:
            self.logger.info(self.AGG_BY +  " = " + zone_input)

            for scenario in self.Multi_Scenario:
                self.logger.info("Scenario = " + scenario)

                re_curt = curtailment_collection.get(scenario)

                # Timeseries [MW] RE curtailment [MWh]
                try: #Check for regions missing all generation.
                    re_curt = re_curt.xs(zone_input,level = self.AGG_BY)
                except KeyError:
                        self.logger.info('No curtailment in ' + zone_input)
                        continue

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


            fig2, ax = plt.subplots(figsize=(6,4))

            if self.prop == "PV":
                Data_Table_Out = PV_Curtailment_DC
                
                if PV_Curtailment_DC.empty:
                    out = mfunc.MissingZoneData()
                    outputs[zone_input] = out
                    continue
                
                for column in PV_Curtailment_DC:
                    ax.plot(PV_Curtailment_DC[column], linewidth=3, color=colour_dict[column],
                            label=column)
                    ax.legend(loc='lower left',bbox_to_anchor=(1,0),
                              facecolor='inherit', frameon=True)
                    ax.set_ylabel('PV Curtailment (MW)',  color='black', rotation='vertical')

            if self.prop == "PV+Wind":
                Data_Table_Out = RE_Curtailment_DC
                
                if RE_Curtailment_DC.empty:
                    out = mfunc.MissingZoneData()
                    outputs[zone_input] = out
                    continue

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
            ax.set_xlim(0, 9490)
            ax.set_ylim(bottom=0)

            outputs[zone_input] = {'fig': fig2, 'data_table': Data_Table_Out}
        return outputs


    def curt_total(self):

        """
        This module calculates the total curtailment, broken down by technology. 
        It produces a stacked bar plot, with a bar for each scenario.
        """

        outputs = {}
        curtailment_collection = {}
        avail_gen_collection = {}
        check_input_data = []
        
        check_input_data.extend([mfunc.get_data(curtailment_collection,"generator_Curtailment", self.Marmot_Solutions_folder, self.Multi_Scenario)])
        check_input_data.extend([mfunc.get_data(avail_gen_collection,"generator_Available_Capacity", self.Marmot_Solutions_folder, self.Multi_Scenario)])
        
        # Checks if all data required by plot is available, if 1 in list required data is missing
        if 1 in check_input_data:
            outputs = mfunc.MissingInputData()
            return outputs
        
        for zone_input in self.Zones:
            self.logger.info(self.AGG_BY +  " = " + zone_input)

            Total_Curtailment_out = pd.DataFrame()
            Total_Available_gen = pd.DataFrame()

            for scenario in self.Multi_Scenario:
                self.logger.info("Scenario = " + scenario)
                # Adjust list of values to drop from vre_gen_cat depending on if it exhists in processed techs
                #self.vre_gen_cat = [name for name in self.vre_gen_cat if name in curtailment_collection.get(scenario).index.unique(level="tech")]

                vre_collection = {}
                avail_vre_collection = {}

                vre_curt = curtailment_collection.get(scenario)
                try:
                    vre_curt = vre_curt.xs(zone_input,level=self.AGG_BY)
                except KeyError:
                    self.logger.info('No curtailment in ' + zone_input)
                    continue
                vre_curt = (vre_curt.loc[(slice(None), self.vre_gen_cat),:])

                avail_gen = avail_gen_collection.get(scenario)
                try: #Check for regions missing all generation.
                    avail_gen = avail_gen.xs(zone_input,level = self.AGG_BY)
                except KeyError:
                        self.logger.info('No available generation in ' + zone_input)
                        continue
                avail_gen = (avail_gen.loc[(slice(None), self.vre_gen_cat),:])

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


                Total_Curtailment_out = pd.concat([Total_Curtailment_out, vre_table], axis=0, sort=False)
                Total_Available_gen = pd.concat([Total_Available_gen, avail_gen_table], axis=0, sort=False)

            vre_pct_curt = Total_Curtailment_out.sum(axis=1)/Total_Available_gen.sum(axis=1)

            Total_Curtailment_out = Total_Curtailment_out/1000000 #Convert to TWh

            # Data table of values to return to main program
            Data_Table_Out = Total_Curtailment_out

            Total_Curtailment_out.index = Total_Curtailment_out.index.str.replace('_',' ')
            Total_Curtailment_out.index = Total_Curtailment_out.index.str.wrap(5, break_long_words=False)

            if Total_Curtailment_out.empty == True:
                outputs[zone_input] = mfunc.MissingZoneData()
                continue
            fig3 = Total_Curtailment_out.plot.bar(stacked=True, figsize=(6,4), rot=0,
                             color=[self.PLEXOS_color_dict.get(x, '#333333') for x in Total_Curtailment_out.columns],
                             edgecolor='black', linewidth='0.1')
            fig3.spines['right'].set_visible(False)
            fig3.spines['top'].set_visible(False)
            fig3.set_ylabel('Total Curtailment (TWh)',  color='black', rotation='vertical')
            fig3.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.2f}'))
            fig3.tick_params(axis='y', which='major', length=5, width=1)
            fig3.tick_params(axis='x', which='major', length=5, width=1)
            fig3.margins(x=0.01)

            handles, labels = fig3.get_legend_handles_labels()
            fig3.legend(reversed(handles), reversed(labels), loc='lower left',bbox_to_anchor=(1,0),
                          facecolor='inherit', frameon=True)

            curt_totals = Total_Curtailment_out.sum(axis=1)
            #inserts total bar value above each bar
            k=0
            for i in fig3.patches:
                height = curt_totals[k]
                width = i.get_width()
                x, y = i.get_xy()
                fig3.text(x+width/2,
                    y+height + 0.05*max(fig3.get_ylim()),
                    '{:.2%}\n|{:,.0f}|'.format(vre_pct_curt[k],curt_totals[k]),
                    horizontalalignment='center',
                    verticalalignment='center', fontsize=11, color='red')
                k=k+1
                if k>=len(vre_pct_curt):
                    break

            outputs[zone_input] = {'fig': fig3, 'data_table': Data_Table_Out}
        return outputs



    def curt_ind(self):

        """
        This module calculates the curtailment, as a percentage of total generation, of individual generators.
        The generators are specified as a list of strings in the fourth column of Marmot_plot_select.csv.  
        The module prints out two .csvs: 
            -one that contains curtailment, in percent, for each scenario and site. 
            -the other contains total generation, in TWh, for each scenario and site.
    
        """

        outputs = {}
        curtailment_collection = {}
        cap_collection = {}
        gen_collection = {}
        check_input_data = []
        
        check_input_data.extend([mfunc.get_data(curtailment_collection,"generator_Curtailment", self.Marmot_Solutions_folder, self.Multi_Scenario)])
        check_input_data.extend([mfunc.get_data(gen_collection,"generator_Generation", self.Marmot_Solutions_folder, self.Multi_Scenario)])
        check_input_data.extend([mfunc.get_data(cap_collection,"generator_Available_Capacity", self.Marmot_Solutions_folder, self.Multi_Scenario)])

        # Checks if all data required by plot is available, if 1 in list required data is missing
        if 1 in check_input_data:
            outputs = None
            return outputs

        Total_Curtailment_Out = pd.DataFrame()
        Total_Gen = pd.DataFrame()
        Gen_8760 = pd.DataFrame()
        scen_idx = -1

        chunks = []

        for scenario in self.Multi_Scenario:
            scen_idx += 1
            self.logger.info("Scenario = " + scenario)

            vre_curt = curtailment_collection.get(scenario)
            gen = gen_collection.get(scenario)
            cap = cap_collection.get(scenario)
            
            #Select only lines specified in Marmot_plot_select.csv.
            select_sites = self.prop.split(",") 
            select_sites = [site[1:] if site[0] == ' ' else site for site in select_sites]
            self.logger.info('Plotting curtailment only for sites specified in Marmot_plot_select.csv')

            ti = gen.index.get_level_values('timestamp').unique()

            site_idx = -1
            sites = pd.Series()
            sites_gen = pd.Series()
            chunks_scen = []
            for site in select_sites:
                if site in gen.index.get_level_values('gen_name').unique():
                    site_idx += 1
                    curt2 = vre_curt.xs(site,level = 'gen_name')
                    gen_site = gen.xs(site,level = 'gen_name')
                    cap_site = cap.xs(site,level = 'gen_name') 
                    curt = cap_site - gen_site
                    curt = vre_curt.xs(site,level = 'gen_name')
                    curt_tot = curt.sum()
                    gen_tot = gen_site.sum()
                    curt_perc = pd.Series(100 * curt_tot / gen_tot)

                    levels2drop = [level for level in gen_site.index.names if level != 'timestamp']
                    gen_site = gen_site.droplevel(levels2drop)
                    gen_site.columns = [site]

                else:
                    curt_perc = pd.Series([0])
                    gen_tot = pd.Series([0])
                    gen_site = pd.Series([0] * len(ti),name = site,index = ti)
                sites_gen = sites_gen.append(gen_tot)
                sites = sites.append(curt_perc)
                chunks_scen.append(gen_site)

            gen_8760_scen = pd.concat(chunks_scen,axis = 1)
            scen_name = pd.Series([scenario] * len(ti),name = 'Scenario')
            gen_8760_scen = gen_8760_scen.set_index(scen_name,append = True)
            chunks.append(gen_8760_scen)

            sites.name = scenario
            sites.index = select_sites
            sites_gen.name = scenario
            sites_gen.index = select_sites
            Total_Curtailment_Out = pd.concat([Total_Curtailment_Out,sites],axis = 1)
            Total_Gen = pd.concat([Total_Gen,sites_gen],axis = 1)

        Gen_8760 = pd.concat(chunks,axis = 0, copy = False)
        Gen_8760.to_csv(os.path.join(self.Marmot_Solutions_folder, self.Scenario_name, 'Figures_Output',self.AGG_BY + '_curtailment','Individual_gen_8760.csv'))

        Total_Gen = Total_Gen / 1000000
        Total_Curtailment_Out.T.to_csv(os.path.join(self.Marmot_Solutions_folder, self.Scenario_name, 'Figures_Output',self.AGG_BY + '_curtailment','Individual_curtailment.csv'))
        Total_Gen.T.to_csv(os.path.join(self.Marmot_Solutions_folder, self.Scenario_name, 'Figures_Output',self.AGG_BY + '_curtailment','Individual_gen.csv'))


        outputs = mfunc.DataSavedInModule()
        return outputs


