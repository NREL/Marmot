# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 13:20:56 2019

This code creates total generation stacked bar plots and is called from Marmot_plot_main.py

@author: dlevie
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
import logging
import marmot.plottingmodules.marmot_plot_functions as mfunc
import marmot.config.mconfig as mconfig
import textwrap

#===============================================================================

custom_legend_elements = Patch(facecolor='#DD0200',
                            alpha=0.5, edgecolor='#DD0200',
                         label='Unserved Energy')

custom_legend_elements_month = Patch(facecolor='#DD0200',alpha=0.7,edgecolor='#DD0200',
                                     label='Unserved_Energy')

class MPlot(object):

    MONTHS = {  1 : "January",
                2 : "February",
                3 : "March",
                4 : "April",
                5 : "May",
                6 : "June",
                7 : "July",
                8 : "August",
                9 : "September",
                10 : "October",
                11 : "November",
                12 : "December"
            }


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

    def total_gen(self, figure_name=None, prop=None, start=None, end=None,
                  timezone=None, start_date_range=None, end_date_range=None):
        # Create Dictionary to hold Datframes for each scenario
        outputs = {}

        if self.AGG_BY == 'zone':
            agg = 'zone'
        else:
            agg = 'region'
        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True,"generator_Generation",self.Scenarios),
                      (False,f"generator_{self.curtailment_prop}",self.Scenarios),
                      (False,"generator_Pump_Load",self.Scenarios),
                      (True,f"{agg}_Load",self.Scenarios),
                      (False,f"{agg}_Unserved_Energy",self.Scenarios)]

        # Runs get_data to populate mplot_data_dict with all required properties, returns a 1 if required data is missing
        check_input_data = mfunc.get_data(self.mplot_data_dict, properties,self.Marmot_Solutions_folder)

        if 1 in check_input_data:
            outputs = mfunc.MissingInputData()
            return outputs

        for zone_input in self.Zones:
            Total_Generation_Stack_Out = pd.DataFrame()
            Total_Load_Out = pd.DataFrame()
            Pump_Load_Out = pd.DataFrame()
            Total_Demand_Out = pd.DataFrame()
            Unserved_Energy_Out = pd.DataFrame()
            self.logger.info(f"Zone = {zone_input}")

            for scenario in self.Scenarios:

                self.logger.info(f"Scenario = {scenario}")
                Total_Gen_Stack = self.mplot_data_dict['generator_Generation'].get(scenario)

                #Check if zone has generation, if not skips
                try:
                    Total_Gen_Stack = Total_Gen_Stack.xs(zone_input,level=self.AGG_BY)
                except KeyError:
                    self.logger.warning(f"No installed capacity in: {zone_input}")
                    continue

                Total_Gen_Stack = mfunc.df_process_gen_inputs(Total_Gen_Stack, self.ordered_gen)

                # Calculates interval step to correct for MWh of generation
                interval_count = mfunc.get_sub_hour_interval_count(Total_Gen_Stack)

                curtailment_name = self.gen_names_dict.get('Curtailment','Curtailment')

                # Insert Curtailmnet into gen stack if it exhists in database
                if self.mplot_data_dict[f"generator_{self.curtailment_prop}"]:
                    Stacked_Curt = self.mplot_data_dict[f"generator_{self.curtailment_prop}"].get(scenario)
                    if zone_input in Stacked_Curt.index.get_level_values(self.AGG_BY).unique():
                        Stacked_Curt = Stacked_Curt.xs(zone_input,level=self.AGG_BY)
                        Stacked_Curt = mfunc.df_process_gen_inputs(Stacked_Curt, self.ordered_gen)
                        Stacked_Curt = Stacked_Curt.sum(axis=1)
                        Total_Gen_Stack.insert(len(Total_Gen_Stack.columns),column=curtailment_name,value=Stacked_Curt) #Insert curtailment into
                        Total_Gen_Stack = Total_Gen_Stack.loc[:, (Total_Gen_Stack != 0).any(axis=0)]

                Total_Gen_Stack = Total_Gen_Stack/interval_count

                if not pd.isnull(start_date_range):
                    self.logger.info(f"Plotting specific date range: \
                                      {str(start_date_range)} to {str(end_date_range)}")
                    Total_Gen_Stack = Total_Gen_Stack[start_date_range:end_date_range]

                Total_Gen_Stack = Total_Gen_Stack.sum(axis=0)
                Total_Gen_Stack.rename(scenario, inplace=True)
                Total_Generation_Stack_Out = pd.concat([Total_Generation_Stack_Out, Total_Gen_Stack], axis=1, sort=False).fillna(0)

                Total_Load = self.mplot_data_dict[f"{agg}_Load"].get(scenario)
                Total_Load = Total_Load.xs(zone_input,level=self.AGG_BY)
                Total_Load = Total_Load.groupby(["timestamp"]).sum()

                if not pd.isnull(start_date_range):
                    Total_Load = Total_Load[start_date_range:end_date_range]

                Total_Load = Total_Load.rename(columns={0:scenario}).sum(axis=0)
                Total_Load = Total_Load/interval_count
                Total_Load_Out = pd.concat([Total_Load_Out, Total_Load], axis=0, sort=False)


                if self.mplot_data_dict[f"{agg}_Unserved_Energy"] == {}:
                    Unserved_Energy = self.mplot_data_dict[f"{agg}_Load"][scenario].copy()
                    Unserved_Energy.iloc[:,0] = 0
                else:
                    Unserved_Energy = self.mplot_data_dict[f"{agg}_Unserved_Energy"][scenario]
                Unserved_Energy = Unserved_Energy.xs(zone_input,level=self.AGG_BY)
                Unserved_Energy = Unserved_Energy.groupby(["timestamp"]).sum()
                
                
                if not pd.isnull(start_date_range):
                    Unserved_Energy = Unserved_Energy[start_date_range:end_date_range]
                Unserved_Energy = Unserved_Energy.rename(columns={0:scenario}).sum(axis=0)
                Unserved_Energy = Unserved_Energy/interval_count

                # save for output
                Unserved_Energy_Out = pd.concat([Unserved_Energy_Out, Unserved_Energy], axis=0, sort=False)

                # subtract unserved energy from load for graphing (not sure this is actually used)
                if (Unserved_Energy == 0).all() == False:
                    Unserved_Energy = Total_Load - Unserved_Energy

                if self.mplot_data_dict["generator_Pump_Load"] == {}:
                    Pump_Load = self.mplot_data_dict['generator_Generation'][scenario].copy()
                    Pump_Load.iloc[:,0] = 0
                else:
                    Pump_Load = self.mplot_data_dict["generator_Pump_Load"][scenario]
                Pump_Load = Pump_Load.xs(zone_input,level=self.AGG_BY)
                Pump_Load = Pump_Load.groupby(["timestamp"]).sum()
                if not pd.isnull(start_date_range):
                    Pump_Load = Pump_Load[start_date_range:end_date_range]
                
                Pump_Load = Pump_Load.rename(columns={0:scenario}).sum(axis=0)
                Pump_Load = Pump_Load/interval_count
                if (Pump_Load == 0).all() == False:
                    Total_Demand = Total_Load - Pump_Load
                else:
                    Total_Demand = Total_Load
                Pump_Load_Out = pd.concat([Pump_Load_Out, Pump_Load], axis=0, sort=False)
                Total_Demand_Out = pd.concat([Total_Demand_Out, Total_Demand], axis=0, sort=False)

            Total_Load_Out = Total_Load_Out.rename(columns={0:'Total Load (Demand + \n Storage Charging)'})
            Total_Demand_Out = Total_Demand_Out.rename(columns={0:'Total Demand'})
            Unserved_Energy_Out = Unserved_Energy_Out.rename(columns={0: 'Unserved Energy'})

            Total_Generation_Stack_Out = mfunc.df_process_categorical_index(Total_Generation_Stack_Out, self.ordered_gen)
            Total_Generation_Stack_Out = Total_Generation_Stack_Out.T
            Total_Generation_Stack_Out = Total_Generation_Stack_Out.loc[:, (Total_Generation_Stack_Out != 0).any(axis=0)]

            if Total_Generation_Stack_Out.empty:
                out = mfunc.MissingZoneData()
                outputs[zone_input] = out
                continue

            # unit conversion return divisor and energy units
            unitconversion = mfunc.capacity_energy_unitconversion(max(Total_Generation_Stack_Out.sum()))

            Total_Generation_Stack_Out = Total_Generation_Stack_Out/unitconversion['divisor']
            Total_Load_Out = Total_Load_Out.T/unitconversion['divisor']
            Pump_Load_Out = Pump_Load_Out.T/unitconversion['divisor']
            Total_Demand_Out = Total_Demand_Out.T/unitconversion['divisor']
            Unserved_Energy_Out = Unserved_Energy_Out.T/unitconversion['divisor']

            # Data table of values to return to main program
            Data_Table_Out = pd.concat([Total_Load_Out.T,
                                        Total_Demand_Out.T,
                                        Unserved_Energy_Out.T,
                                        Total_Generation_Stack_Out],  axis=1, sort=False)
            Data_Table_Out = Data_Table_Out.add_suffix(f" ({unitconversion['units']}h)")

            Total_Generation_Stack_Out.index = Total_Generation_Stack_Out.index.str.replace('_',' ')
            
            Total_Generation_Stack_Out, angle = mfunc.check_label_angle(Total_Generation_Stack_Out, False)
            fig1, ax = plt.subplots(figsize=(self.x,self.y))

            Total_Generation_Stack_Out.plot.bar(stacked=True, rot=angle, ax=ax,
                             color=[self.PLEXOS_color_dict.get(x, '#333333') for x in Total_Generation_Stack_Out.columns], edgecolor='black', linewidth='0.1')
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_ylabel(f"Total Genertaion ({unitconversion['units']}h)",  color='black', rotation='vertical')
            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(x, f',.{self.y_axes_decimalpt}f')))
            if angle > 0:
                ax.set_xticklabels(Total_Generation_Stack_Out.index, ha="right")
                tick_length = 8
            else:
                tick_length = 5
            ax.tick_params(axis='y', which='major', length=tick_length, width=1)
            ax.tick_params(axis='x', which='major', length=tick_length, width=1)
            if mconfig.parser("plot_title_as_region"):
                ax.set_title(zone_input)
            
            # replace x-axis with custom labels if present 
            if len(self.ticklabels) > 1:
                ticklabels = [textwrap.fill(x.replace('_', ' '), 8) for x in self.ticklabels]
                ax.set_xticklabels(ticklabels)
            
            
            for n, scenario in enumerate(self.Scenarios):

                x = [ax.patches[n].get_x(), ax.patches[n].get_x() + ax.patches[n].get_width()]
                height1 = [int(Total_Load_Out[scenario].sum())]*2
                #print("total load height: " + str(height1))
                lp1 = plt.plot(x,height1, c='black', linewidth=3)
                if Pump_Load_Out[scenario].values.sum() > 0:
                    height2 = [int(Total_Demand_Out[scenario])]*2
                    lp2 = plt.plot(x,height2, 'r--', c='black', linewidth=1.5)

                if Unserved_Energy_Out[scenario].values.sum() > 0:
                    height3 = [int(Unserved_Energy_Out[scenario])]*2
                    plt.plot(x,height3, c='#DD0200', linewidth=1.5)
                    ax.fill_between(x, height3, height1,
                                facecolor = '#DD0200',
                                alpha=0.5)

            handles, labels = ax.get_legend_handles_labels()

            #Combine all legends into one.
            if Pump_Load_Out.values.sum() > 0:
                handles.append(lp2[0])
                handles.append(lp1[0])
                labels += ['Demand','Demand + \n Storage Charging']
            else:
                handles.append(lp1[0])
                labels += ['Demand']

            if Unserved_Energy_Out.values.sum() > 0:
                handles.append(custom_legend_elements)
                labels += ['Unserved Energy']

            ax.legend(reversed(handles),reversed(labels),loc = 'lower left',bbox_to_anchor=(1.05,0),facecolor='inherit', frameon=True)

            outputs[zone_input] = {'fig': fig1, 'data_table': Data_Table_Out}

        return outputs

    def total_gen_diff(self, figure_name=None, prop=None, start=None, end=None,
                       timezone=None, start_date_range=None, end_date_range=None):
        # Create Dictionary to hold Datframes for each scenario
        outputs = {}

        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True,"generator_Generation",self.Scenarios),
                      (False,f"generator_{self.curtailment_prop}",self.Scenarios)]

        # Runs get_data to populate mplot_data_dict with all required properties, returns a 1 if required data is missing
        check_input_data = mfunc.get_data(self.mplot_data_dict, properties,self.Marmot_Solutions_folder)

        if 1 in check_input_data:
            outputs = mfunc.MissingInputData()
            return outputs

        for zone_input in self.Zones:
            Total_Generation_Stack_Out = pd.DataFrame()

            self.logger.info(f"Zone = {zone_input}")

            for scenario in self.Scenarios:

                self.logger.info(f"Scenario = {scenario}")

                Total_Gen_Stack = self.mplot_data_dict['generator_Generation'].get(scenario)

                #Check if zone has generation, if not skips and breaks out of Multi_Scenario loop
                try:
                    Total_Gen_Stack = Total_Gen_Stack.xs(zone_input,level=self.AGG_BY)
                except KeyError:
                    self.logger.warning(f"No installed capacity in : {zone_input}")
                    break

                Total_Gen_Stack = mfunc.df_process_gen_inputs(Total_Gen_Stack, self.ordered_gen)

                # Calculates interval step to correct for MWh of generation
                interval_count = mfunc.get_sub_hour_interval_count(Total_Gen_Stack)

                curtailment_name = self.gen_names_dict.get('Curtailment','Curtailment')

                # Insert Curtailmnet into gen stack if it exhists in database
                if self.mplot_data_dict[f"generator_{self.curtailment_prop}"]:
                    Stacked_Curt = self.mplot_data_dict[f"generator_{self.curtailment_prop}"].get(scenario)
                    if zone_input in Stacked_Curt.index.get_level_values(self.AGG_BY).unique():
                        Stacked_Curt = Stacked_Curt.xs(zone_input,level=self.AGG_BY)
                        Stacked_Curt = mfunc.df_process_gen_inputs(Stacked_Curt, self.ordered_gen)
                        Stacked_Curt = Stacked_Curt.sum(axis=1)
                        Total_Gen_Stack.insert(len(Total_Gen_Stack.columns),column=curtailment_name,value=Stacked_Curt) #Insert curtailment into
                        Total_Gen_Stack = Total_Gen_Stack.loc[:, (Total_Gen_Stack != 0).any(axis=0)]

                Total_Gen_Stack = Total_Gen_Stack/interval_count
                if not pd.isnull(start_date_range):
                    self.logger.info(f"Plotting specific date range: \
                                     {str(start_date_range)} to {str(end_date_range)}")
                    Total_Gen_Stack = Total_Gen_Stack[start_date_range:end_date_range]
                Total_Gen_Stack = Total_Gen_Stack.sum(axis=0)
                Total_Gen_Stack.rename(scenario, inplace=True)
                Total_Generation_Stack_Out = pd.concat([Total_Generation_Stack_Out, Total_Gen_Stack], axis=1, sort=False).fillna(0)

            Total_Generation_Stack_Out = mfunc.df_process_categorical_index(Total_Generation_Stack_Out, self.ordered_gen)
            Total_Generation_Stack_Out = Total_Generation_Stack_Out.T
            Total_Generation_Stack_Out = Total_Generation_Stack_Out.loc[:, (Total_Generation_Stack_Out != 0).any(axis=0)]

            #Ensures region has generation, else skips
            try:
                Total_Generation_Stack_Out = Total_Generation_Stack_Out-Total_Generation_Stack_Out.xs(self.Scenarios[0]) #Change to a diff on first scenario
            except KeyError:
                out = mfunc.MissingZoneData()
                outputs[zone_input] = out
                continue

            Total_Generation_Stack_Out.drop(self.Scenarios[0],inplace=True) #Drop base entry

            if Total_Generation_Stack_Out.empty == True:
                out = mfunc.MissingZoneData()
                outputs[zone_input] = out
                continue

            unitconversion = mfunc.capacity_energy_unitconversion(max(abs(Total_Generation_Stack_Out.sum())))
            Total_Generation_Stack_Out = Total_Generation_Stack_Out/unitconversion['divisor']

            # Data table of values to return to main program
            Data_Table_Out = Total_Generation_Stack_Out.add_suffix(f" ({unitconversion['units']}h)")

            net_diff = Total_Generation_Stack_Out
            try:
                net_diff.drop(columns = curtailment_name,inplace=True)
            except KeyError:
                pass
            net_diff = net_diff.sum(axis = 1)

            Total_Generation_Stack_Out.index = Total_Generation_Stack_Out.index.str.replace('_',' ')

            Total_Generation_Stack_Out, angle = mfunc.check_label_angle(Total_Generation_Stack_Out, False)
            
            fig1, ax = plt.subplots(figsize=(self.x,self.y))
            Total_Generation_Stack_Out.plot.bar(stacked=True, rot=angle,
                             color=[self.PLEXOS_color_dict.get(x, '#333333') for x in Total_Generation_Stack_Out.columns], edgecolor='black', linewidth='0.1',ax=ax)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(x, f',.{self.y_axes_decimalpt}f')))
            if angle > 0:
                ax.set_xticklabels(Total_Generation_Stack_Out.index, ha="right")
                tick_length = 8
            else:
                tick_length = 5
            ax.tick_params(axis='y', which='major', length=tick_length, width=1)
            ax.tick_params(axis='x', which='major', length=tick_length, width=1)

            #Add net gen difference line.
            for n, scenario in enumerate(self.Scenarios[1:]):
                x = [ax.patches[n].get_x(), ax.patches[n].get_x() + ax.patches[n].get_width()]
                y_net = [net_diff.loc[scenario]] * 2
                net_line = plt.plot(x,y_net, c='black', linewidth=1.5)

            locs,labels=plt.xticks()

            ax.set_ylabel(f"Generation Change ({format(unitconversion['units'])}h) \n relative to {self.Scenarios[0].replace('_',' ')}",  color='black', rotation='vertical')
            
            # xlabels = [textwrap.fill(x.replace('_',' '),10) for x in self.xlabels]

            # plt.xticks(ticks=locs,labels=xlabels[1:])
            # ax.margins(x=0.01)

            plt.axhline(linewidth=0.5,linestyle='--',color='grey')

            handles, labels = ax.get_legend_handles_labels()

            handles.append(net_line[0])
            labels += ['Net Gen Change']

            #Main legend
            ax.legend(reversed(handles), reversed(labels), loc='lower left',bbox_to_anchor=(1,0),facecolor='inherit', frameon=True)
            if mconfig.parser("plot_title_as_region"):
                ax.set_title(zone_input)
            outputs[zone_input] = {'fig': fig1, 'data_table': Data_Table_Out}
        return outputs

    #===============================================================================
    ## Total Gen Facet Plots removed for now, code not stable and needs testing
    #===============================================================================

    def total_gen_facet(self, figure_name=None, prop=None, start=None, end=None,
                        timezone=None, start_date_range=None, end_date_range=None):
        outputs = mfunc.UnderDevelopment()
        self.logger.warning('total_gen_facet is under development')
        return outputs

    #     self.mplot_data_dict['generator_Generation'] = {}
    #     self.mplot_data_dictf"{self.AGG_BY}_Load"] = {}
    #     self.mplot_data_dict[f"generator_{self.curtailment_prop}"] = {}

    #     for scenario in self.Scenarios:
    #         try:
    #             self.mplot_data_dict['generator_Generation'][scenario] = pd.read_hdf(os.path.join(self.Marmot_Solutions_folder, scenario,"Processed_HDF5_folder", scenario+ "_formatted.h5"), "generator_Generation")
    #             self.mplot_data_dict[f"generator_{self.curtailment_prop}"][scenario] = pd.read_hdf(os.path.join(self.Marmot_Solutions_folder, scenario,"Processed_HDF5_folder", scenario+ "_formatted.h5"),  f"generator_{self.curtailment_prop}")
    #             # If data is to be agreagted by zone, then zone properties are loaded, else region properties are loaded
    #             if self.AGG_BY == "zone":
    #                 self.mplot_data_dictf"{self.AGG_BY}_Load"][scenario] = pd.read_hdf(os.path.join(self.Marmot_Solutions_folder, scenario,"Processed_HDF5_folder", scenario+ "_formatted.h5"), "zone_Load")
    #             else:
    #                 self.mplot_data_dictf"{self.AGG_BY}_Load"][scenario] = pd.read_hdf(os.path.join(self.Marmot_Solutions_folder, scenario,"Processed_HDF5_folder", scenario+ "_formatted.h5"),  "region_Load")

    #         except Exception:
    #             pass


    #     Total_Generation_Stack_Out = pd.DataFrame()
    #     Total_Load_Out = pd.DataFrame()
    #     self.logger.info("Zone = " + self.zone_input)


    #     for scenario in self.Scenarios:
    #         self.logger.info("Scenario = " + scenario)
    #         try:
    #             Total_Gen_Stack = self.mplot_data_dict['generator_Generation'].get(scenario)
    #             Total_Gen_Stack = Total_Gen_Stack.xs(self.zone_input,level=self.AGG_BY)
    #             Total_Gen_Stack = df_process_gen_inputs(Total_Gen_Stack, self)
    #             Stacked_Curt = self.mplot_data_dict[f"generator_{self.curtailment_prop}"].get(scenario)
    #             Stacked_Curt = Stacked_Curt.xs(self.zone_input,level=self.AGG_BY)
    #             Stacked_Curt = df_process_gen_inputs(Stacked_Curt, self)
    #             Stacked_Curt = Stacked_Curt.sum(axis=1)
    #             Total_Gen_Stack.insert(len(Total_Gen_Stack.columns),column='Curtailment',value=Stacked_Curt) #Insert curtailment into
    #             Total_Gen_Stack = Total_Gen_Stack.loc[:, (Total_Gen_Stack != 0).any(axis=0)]

    #             Total_Gen_Stack = Total_Gen_Stack.sum(axis=0)
    #             Total_Gen_Stack.rename(scenario, inplace=True)

    #             Total_Generation_Stack_Out = pd.concat([Total_Generation_Stack_Out, Total_Gen_Stack], axis=1, sort=False).fillna(0)

    #             Total_Load = self.mplot_data_dictf"{self.AGG_BY}_Load"].get(scenario)
    #             Total_Load = Total_Load.xs(self.zone_input,level=self.AGG_BY)
    #             Total_Load = Total_Load.groupby(["timestamp"]).sum()
    #             Total_Load = Total_Load.rename(columns={0:scenario}).sum(axis=0)
    #             Total_Load_Out = pd.concat([Total_Load_Out, Total_Load], axis=0, sort=False)
    #         except Exception:
    #             self.logger.warning("Error: Skipping " + scenario)
    #             pass

    #     Total_Load_Out = Total_Load_Out.rename(columns={0:'Total Load'})

    #     Total_Generation_Stack_Out = df_process_categorical_index(Total_Generation_Stack_Out, self)
    #     Total_Generation_Stack_Out = Total_Generation_Stack_Out.T/1000
    #     Total_Generation_Stack_Out = Total_Generation_Stack_Out.loc[:, (Total_Generation_Stack_Out != 0).any(axis=0)]

    #     # Data table of values to return to main program
    #     Data_Table_Out = pd.concat([Total_Load_Out/1000, Total_Generation_Stack_Out],  axis=1, sort=False)

    #     Total_Generation_Stack_Out.index = Total_Generation_Stack_Out.index.str.replace('_',' ')
    
    #     Total_Generation_Stack_Out.index = mfunc.check_label_angle(Total_Generation_Stack_Out,False)

    #     Total_Load_Out.index = Total_Load_Out.index.str.replace('_',' ')
    #     Total_Load_Out.index = Total_Load_Out.index.str.wrap(10, break_long_words=False)
    #     
    #     Total_Load_Out = Total_Load_Out.T/1000

    #     xdimension=len(self.xlabels)
    #     ydimension=len(self.ylabels)
    #     grid_size = xdimension*ydimension

    #     fig2, axs = plt.subplots(ydimension,xdimension, figsize=((2*xdimension), (4*ydimension)), sharey=True)
    #     axs = axs.ravel()
    #     plt.subplots_adjust(wspace=0, hspace=0.01)

    #     i=0
    #     for index in Total_Generation_Stack_Out.index:

    #         sb = Total_Generation_Stack_Out.iloc[i:i+1].plot.bar(stacked=True, rot=angle,
    #         color=[self.PLEXOS_color_dict.get(x, '#333333') for x in Total_Generation_Stack_Out.columns], edgecolor='black', linewidth='0.1',
    #                                      ax=axs[i])

    #         axs[i].get_legend().remove()
    #         axs[i].spines['right'].set_visible(False)
    #         axs[i].spines['top'].set_visible(False)
    #         axs[i].xaxis.set_ticklabels([])
    #         if angle > 0:
    #             ax.set_xticklabels(Total_Generation_Stack_Out.iloc[i:i+1].index, ha="right")
    #             tick_length = 8
    #         else:
    #             tick_length = 5
    #         ax.tick_params(axis='y', which='major', length=tick_length, width=1)
    #         ax.tick_params(axis='x', which='major', length=tick_length, width=1)
    #         axs[i].yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    #         axs[i].margins(x=0.01)

    #         height = [int(Total_Load_Out[index])]
    #         axs[i].axhline(y=height,xmin=0.25,xmax=0.75, linestyle ='--', c="black",linewidth=1.5)

    #         handles, labels = axs[1].get_legend_handles_labels()

    #         leg1 = axs[grid_size-1].legend(reversed(handles), reversed(labels), loc='lower left',bbox_to_anchor=(1,0),
    #                   facecolor='inherit', frameon=True)

    #         #Legend 2
    #         leg2 = axs[grid_size-1].legend(['Load'], loc='upper left',bbox_to_anchor=(1, 0.95),
    #                       facecolor='inherit', frameon=True)

    #         fig2.add_artist(leg1)

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
    #     plt.ylabel('Total Genertaion (GWh)',  color='black', rotation='vertical', labelpad=60)

    #     return {'fig': fig2, 'data_table': Data_Table_Out}
    
    
    
    
###############################################################################

#################################################################################

    
    def total_gen_monthly(self, figure_name=None, prop=None, start=None, end=None,
                  timezone=None, start_date_range=None, end_date_range=None):
        """ Total generation by Month"""
        
        # Create Dictionary to hold Datframes for each scenario
        
        outputs = {}

        if self.AGG_BY == 'zone':
            agg = 'zone'
        else:
            agg = 'region'
        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True,"generator_Generation",self.Scenarios),
                      (False,"generator_Energy_Curtailed",self.Scenarios),
                      (False,"generator_Pump_Load",self.Scenarios),
                      (True,f"{agg}_Load",self.Scenarios),
                      (False,f"{agg}_Unserved_Energy",self.Scenarios)]

        # Runs get_data to populate mplot_data_dict with all required properties, returns a 1 if required data is missing
        check_input_data = mfunc.get_data(self.mplot_data_dict, properties,self.Marmot_Solutions_folder)

        if 1 in check_input_data:
            outputs = mfunc.MissingInputData()
            return outputs
        
        xdimension, ydimension = mfunc.setup_facet_xy_dimensions(self.xlabels,self.ylabels,multi_scenario=self.Scenarios)
        grid_size = xdimension*ydimension
        
        if xdimension > 1:
            font_scaling_ratio = 1 + ((xdimension-1)*0.09)
            plt.rcParams['xtick.labelsize'] = plt.rcParams['xtick.labelsize']*font_scaling_ratio
            plt.rcParams['ytick.labelsize'] = plt.rcParams['ytick.labelsize']*font_scaling_ratio
            plt.rcParams['legend.fontsize'] = plt.rcParams['legend.fontsize']*font_scaling_ratio
            plt.rcParams['axes.labelsize'] = plt.rcParams['axes.labelsize']*font_scaling_ratio
            plt.rcParams['axes.titlesize'] =  plt.rcParams['axes.titlesize']*font_scaling_ratio
         
        for zone_input in self.Zones:
            Monthly_Gen_Stack_Out = pd.DataFrame()
            #monthly_gen_stack_collect = pd.DataFrame()
            Total_Load_Out = pd.DataFrame()
            Pump_Load_Out = pd.DataFrame()
            Total_Demand_Out = pd.DataFrame()
            Unserved_Energy_Out = pd.DataFrame()
            self.logger.info(f"Zone = {zone_input}")

            #for scenario in self.Scenarios:
            fig1, axs = plt.subplots(ydimension,xdimension, figsize=((self.x*xdimension),(self.y*ydimension)), sharey=True, squeeze=False)
            plt.subplots_adjust(wspace=0.05, hspace=0.5)
            axs = axs.ravel()
            #data_tables = []    
            
            for i, scenario in enumerate(self.Scenarios):
                
                self.logger.info(f"Scenario = {scenario}")
                Total_Gen_Stack = self.mplot_data_dict['generator_Generation'].get(scenario)
    
                #Check if zone has generation, if not skips
                try:
                    Total_Gen_Stack = Total_Gen_Stack.xs(zone_input,level=self.AGG_BY)
                except KeyError:
                    self.logger.warning(f"No installed capacity in: {zone_input}")
                    continue
    
                Total_Gen_Stack = mfunc.df_process_gen_inputs(Total_Gen_Stack, self.ordered_gen)

                # Calculates interval step to correct for MWh of generation if data is subhourly
                interval_count = mfunc.get_sub_hour_interval_count(Total_Gen_Stack)

                curtailment_name = self.gen_names_dict.get('Curtailment','Curtailment')
    
                # Insert Curtailmnet into gen stack if it exhists in database
                if self.mplot_data_dict["generator_Energy_Curtailed"]:
                    Stacked_Curt = self.mplot_data_dict["generator_Energy_Curtailed"].get(scenario)
                    if zone_input in Stacked_Curt.index.get_level_values(self.AGG_BY).unique():
                        Stacked_Curt = Stacked_Curt.xs(zone_input,level=self.AGG_BY)
                        Stacked_Curt = mfunc.df_process_gen_inputs(Stacked_Curt, self.ordered_gen)
                        Stacked_Curt = Stacked_Curt.sum(axis=1)
                        Total_Gen_Stack.insert(len(Total_Gen_Stack.columns),column=curtailment_name,value=Stacked_Curt) #Insert curtailment into
                        Total_Gen_Stack = Total_Gen_Stack.loc[:, (Total_Gen_Stack != 0).any(axis=0)]
    
                
                monthly_gen_stack = Total_Gen_Stack/interval_count
                monthly_gen_stack = monthly_gen_stack.groupby(pd.Grouper(freq='M')).sum()

                if len(monthly_gen_stack.index) > 12:
                    monthly_gen_stack = monthly_gen_stack[:-1]
                
                monthly_gen_stack.columns = monthly_gen_stack.columns.add_categories(['scenario','timestamp'])
                monthly_gen_stack.reset_index(drop=False, inplace=True)
                monthly_gen_stack['timestamp'] = monthly_gen_stack['timestamp'].dt.month.apply(lambda x: self.MONTHS[x])
                monthly_gen_stack.set_index('timestamp', inplace=True)

                monthly_gen_stack["scenario"] = scenario
                Monthly_Gen_Stack_Out = pd.concat([Monthly_Gen_Stack_Out,monthly_gen_stack],axis=0,sort=False)
                    
                Total_Load = self.mplot_data_dict[f"{agg}_Load"].get(scenario)
                Total_Load = Total_Load.xs(zone_input,level=self.AGG_BY)
                Total_Load = Total_Load.groupby(["timestamp"]).sum()
                Total_Load = Total_Load/interval_count
                Total_Load = Total_Load.groupby(pd.Grouper(freq='M')).sum()
                if len(Total_Load.index) > 12:
                    Total_Load = Total_Load[:-1]
                
                Total_Load.reset_index(drop=False, inplace=True)
                Total_Load['timestamp'] = Total_Load['timestamp'].dt.month.apply(lambda x: self.MONTHS[x])
                Total_Load.set_index('timestamp', inplace=True)
    
                Total_Load_Out = pd.concat([Total_Load_Out, Total_Load], axis=0, sort=False)
                
                if self.mplot_data_dict[f"{agg}_Unserved_Energy"] == {}:
                    Unserved_Energy = self.mplot_data_dict[f"{agg}_Load"][scenario].copy()
                    Unserved_Energy.iloc[:,0] = 0
                else:
                    Unserved_Energy = self.mplot_data_dict[f"{agg}_Unserved_Energy"][scenario]
                Unserved_Energy = Unserved_Energy.xs(zone_input,level=self.AGG_BY)
                Unserved_Energy = Unserved_Energy.groupby(["timestamp"]).sum()
                Unserved_Energy = Unserved_Energy/interval_count
                Unserved_Energy = Unserved_Energy.groupby(pd.Grouper(freq='M')).sum()
                if len(Unserved_Energy.index) > 12:
                    Unserved_Energy = Unserved_Energy[:-1]
                
                Unserved_Energy.reset_index(drop=False, inplace=True)
                Unserved_Energy['timestamp'] = Unserved_Energy['timestamp'].dt.month.apply(lambda x: self.MONTHS[x])
                Unserved_Energy.set_index('timestamp', inplace=True)
    
                # save for output
                Unserved_Energy_Out = pd.concat([Unserved_Energy_Out, Unserved_Energy], axis=0, sort=False)
    
                if self.mplot_data_dict["generator_Pump_Load"] == {}:
                    Pump_Load = self.mplot_data_dict['generator_Generation'][scenario].copy()
                    Pump_Load.iloc[:,0] = 0
                else:
                    Pump_Load = self.mplot_data_dict["generator_Pump_Load"][scenario]
                Pump_Load = Pump_Load.xs(zone_input,level=self.AGG_BY)
                Pump_Load = Pump_Load.groupby(["timestamp"]).sum()
                Pump_Load = Pump_Load/interval_count
                Pump_Load = Pump_Load.groupby(pd.Grouper(freq='M')).sum()
                if len(Pump_Load.index) > 12:
                    Pump_Load = Pump_Load[:-1]
                Pump_Load.reset_index(drop=False, inplace=True)
                Pump_Load['timestamp'] = Pump_Load['timestamp'].dt.month.apply(lambda x: self.MONTHS[x])
                Pump_Load.set_index('timestamp', inplace=True)
                
                Total_Demand = Total_Load - Pump_Load
                
                Pump_Load_Out = pd.concat([Pump_Load_Out, Pump_Load], axis=0, sort=False)
                Total_Demand_Out = pd.concat([Total_Demand_Out, Total_Demand], axis=0, sort=False)

            
            Pump_Load_Out = Pump_Load_Out.rename(columns={0:'Pump Load'})
            Total_Load_Out = Total_Load_Out.rename(columns={0:'Total Load (Demand + \n Storage Charging)'})
            Total_Demand_Out = Total_Demand_Out.rename(columns={0:'Total Demand'})
            Unserved_Energy_Out = Unserved_Energy_Out.rename(columns={0: 'Unserved Energy'})

            
            Monthly_Gen_Stack_Out = Monthly_Gen_Stack_Out.loc[:, (Monthly_Gen_Stack_Out != 0).any(axis=0)]

            if Monthly_Gen_Stack_Out.empty:
                out = mfunc.MissingZoneData()
                outputs[zone_input] = out
                continue
            
            scenario_data = pd.DataFrame(Monthly_Gen_Stack_Out['scenario'])
            Monthly_Gen_Stack_Out = Monthly_Gen_Stack_Out.drop('scenario',axis=1)
            Monthly_Gen_Stack_Out = Monthly_Gen_Stack_Out.T
            #print(Monthly_Gen_Stack_Out)
            
            # unit conversion return divisor and energy units, only check once

            unitconversion = mfunc.capacity_energy_unitconversion(max(Monthly_Gen_Stack_Out.sum()))

            #monthly_gen_stack = monthly_gen_stack/unitconversion['divisor']
            Monthly_Gen_Stack_Out = Monthly_Gen_Stack_Out/unitconversion['divisor']
            Total_Load_Out = Total_Load_Out/unitconversion['divisor']
            Pump_Load_Out = Pump_Load_Out/unitconversion['divisor']
            Total_Demand_Out = Total_Demand_Out/unitconversion['divisor']
            Unserved_Energy_Out = Unserved_Energy_Out/unitconversion['divisor']
        
            
            #Data table of values to return to main program
            Data_Table_Out = pd.concat([scenario_data.T,
                                        Total_Load_Out.T,
                                        Total_Demand_Out.T,
                                        Unserved_Energy_Out.T,
                                        Monthly_Gen_Stack_Out],  axis=0, sort=False)

            Data_Table_Out = Data_Table_Out.add_suffix(f" ({unitconversion['units']}h)")
            
            
            for i, scenario in enumerate(self.Scenarios):
                month_data_scenario = pd.concat([scenario_data.T,Monthly_Gen_Stack_Out],axis=0,sort=False)
                month_data_scenario = month_data_scenario.T
                month_data_scenario = month_data_scenario.loc[month_data_scenario['scenario']==scenario]
                month_data_scenario = month_data_scenario.drop('scenario',axis=1)
                
                month_data_scenario, angle = mfunc.check_label_angle(month_data_scenario, False)
                
                month_data_scenario.plot.bar(stacked=True, rot=angle, ax=axs[i],
                                  color=[self.PLEXOS_color_dict.get(x, '#333333') for x in month_data_scenario.columns], edgecolor='black', linewidth='0.1')
                axs[i].spines['right'].set_visible(False)
                axs[i].spines['top'].set_visible(False)
                axs[i].margins(x=0.01)
                axs[i].set_ylabel(f"Total Genertaion ({unitconversion['units']}h)",  color='black', rotation='vertical')
                axs[i].yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(x, f',.{self.y_axes_decimalpt}f')))
                if angle > 0:
                    axs[i].set_xticklabels(month_data_scenario.index, ha="right")
                    tick_length = 8
                else:
                    tick_length = 5
                axs[i].tick_params(axis='y', which='major', length=tick_length, width=1)
                axs[i].tick_params(axis='x', which='major', length=tick_length, width=1)
                
                handles, labels = axs[i].get_legend_handles_labels()
                axs[i].legend().set_visible(False)
            
            axs[grid_size-1].legend(reversed(handles),reversed(labels),
                                    loc = 'lower left',bbox_to_anchor=(1.05,0),
                                    facecolor='inherit', frameon=True)

            xlabels = [x.replace('_',' ') for x in self.xlabels]
            if self.ylabels == ['']:
                ylabels = [f"Total Generation ({unitconversion['units']}h)"]
            else:
                self.ylabels = [y.replace('_',' ') for y in self.ylabels]


            # add facet labels
            mfunc.add_facet_labels(fig1, xlabels, ylabels)
            
            fig1.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            if mconfig.parser('plot_title_as_region'):
                plt.title(zone_input)
            
            outputs[zone_input] = {'fig': fig1, 'data_table': Data_Table_Out}

        return outputs
    

    def monthly_vre_percentage_generation(self, **kwargs):
        """Monthly Total Variable Renewable Generation by technology percentage,
           Each vre technology is plotted as a bar, the total of all bars add to 100%
           Each sceanrio is plotted on a seperate facet plot 
           Technologies that belong to VRE can be set in the vre_gen_cat.csv file 
           in the Mapping folder
        """
        outputs = self._monthly_vre_gen(plot_as_percnt=True, **kwargs)
        return outputs
    
    def monthly_vre_generation(self, **kwargs):
        """Monthly Total Variable Renewable Generation
            Each vre technology is plotted as a bar
            Each sceanrio is plotted on a seperate facet plot 
           Technologies that belong to VRE can be set in the vre_gen_cat.csv file 
           in the Mapping folder
        """
        outputs = self._monthly_vre_gen(**kwargs)
        return outputs

    def _monthly_vre_gen(self, plot_as_percnt=False, figure_name=None, prop=None, start=None, end=None,
                  timezone=None, start_date_range=None, end_date_range=None):
        """ Creates monthly generation plot, internal method called from 
            monthly_vre_percentage_generation or monthly_vre_generation
        """
        
        # Create Dictionary to hold Datframes for each scenario
        outputs = {}
            
        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True,"generator_Generation",self.Scenarios), 
                      (False,"generator_Energy_Curtailed",self.Scenarios)]


        # Runs get_data to populate mplot_data_dict with all required properties, returns a 1 if required data is missing
        check_input_data = mfunc.get_data(self.mplot_data_dict, properties,self.Marmot_Solutions_folder)

        if 1 in check_input_data:
            outputs = mfunc.MissingInputData()
            return outputs
        
        xdimension, ydimension = mfunc.setup_facet_xy_dimensions(self.xlabels,self.ylabels,multi_scenario=self.Scenarios)
        grid_size = xdimension*ydimension
        
        if xdimension > 1:
            font_scaling_ratio = 1 + ((xdimension-1)*0.09)
            plt.rcParams['xtick.labelsize'] = plt.rcParams['xtick.labelsize']*font_scaling_ratio
            plt.rcParams['ytick.labelsize'] = plt.rcParams['ytick.labelsize']*font_scaling_ratio
            plt.rcParams['legend.fontsize'] = plt.rcParams['legend.fontsize']*font_scaling_ratio
            plt.rcParams['axes.labelsize'] = plt.rcParams['axes.labelsize']*font_scaling_ratio
            plt.rcParams['axes.titlesize'] =  plt.rcParams['axes.titlesize']*font_scaling_ratio
         
        for zone_input in self.Zones:
            Wind_Solar_Out = pd.DataFrame()
            self.logger.info(f"Zone = {zone_input}")

            #for scenario in self.Scenarios:
            fig1, axs = plt.subplots(ydimension,xdimension, figsize=((self.x*xdimension),(self.y*ydimension)), sharey=True, squeeze=False)
            plt.subplots_adjust(wspace=0.05, hspace=0.5)
            axs = axs.ravel()
               
            for i, scenario in enumerate(self.Scenarios):
                
                self.logger.info(f"Scenario = {scenario}")
                Total_Gen_Stack = self.mplot_data_dict['generator_Generation'].get(scenario)
    
                #Check if zone has generation, if not skips
                try:
                    Total_Gen_Stack = Total_Gen_Stack.xs(zone_input,level=self.AGG_BY)
                except KeyError:
                    self.logger.warning(f"No installed capacity in: {zone_input}")
                    continue
    
                Total_Gen_Stack = (Total_Gen_Stack.loc[(slice(None), self.vre_gen_cat),:])
                Total_Gen_Stack = mfunc.df_process_gen_inputs(Total_Gen_Stack, self.ordered_gen)
    
                # Calculates interval step to correct for MWh of generation if data is subhourly
                interval_count = mfunc.get_sub_hour_interval_count(Total_Gen_Stack)

                curtailment_name = self.gen_names_dict.get('Curtailment','Curtailment')
    
                #Insert Curtailmnet into gen stack if it exhists in database
                if self.mplot_data_dict["generator_Energy_Curtailed"]:
                    Stacked_Curt = self.mplot_data_dict["generator_Energy_Curtailed"].get(scenario)
                    if zone_input in Stacked_Curt.index.get_level_values(self.AGG_BY).unique():
                        Stacked_Curt = Stacked_Curt.xs(zone_input,level=self.AGG_BY)
                        Stacked_Curt = mfunc.df_process_gen_inputs(Stacked_Curt, self.ordered_gen)
                        Stacked_Curt = Stacked_Curt.sum(axis=1)
                        Total_Gen_Stack.insert(len(Total_Gen_Stack.columns),column=curtailment_name,value=Stacked_Curt) #Insert curtailment into
                        Total_Gen_Stack = Total_Gen_Stack.loc[:, (Total_Gen_Stack != 0).any(axis=0)]

                monthly_gen_stack = Total_Gen_Stack/interval_count
                
                monthly_gen_stack = monthly_gen_stack.groupby(pd.Grouper(freq='M')).sum()
                
                if len(monthly_gen_stack.index) > 12:
                    monthly_gen_stack = monthly_gen_stack[:-1]
                
                monthly_gen_stack.columns = monthly_gen_stack.columns.add_categories('timestamp')
                monthly_gen_stack.reset_index(drop=False, inplace=True)
                monthly_gen_stack['timestamp'] = monthly_gen_stack['timestamp'].dt.month.apply(lambda x: self.MONTHS[x])
                monthly_gen_stack.set_index('timestamp', inplace=True)
                                
                wind_solar = monthly_gen_stack.copy()
                monthly_total_gen = pd.DataFrame(monthly_gen_stack.T.sum(),columns=['Total Generation'])
                
                if plot_as_percnt:
                    for vre_col in wind_solar.columns:
                        wind_solar[vre_col] = (wind_solar[vre_col] / monthly_total_gen['Total Generation']) * 100
                    
                wind_solar.columns = wind_solar.columns.add_categories('scenario')
                wind_solar["scenario"] = scenario

                Wind_Solar_Out = pd.concat([Wind_Solar_Out,wind_solar],axis=0,sort=False)
                
            Wind_Solar_Out = Wind_Solar_Out.loc[:, (Wind_Solar_Out != 0).any(axis=0)]

            if Wind_Solar_Out.empty:
                out = mfunc.MissingZoneData()
                outputs[zone_input] = out
                continue
            
            scenario_data = pd.DataFrame(Wind_Solar_Out['scenario'])
            Wind_Solar_Out = Wind_Solar_Out.drop('scenario',axis=1)
            Wind_Solar_Out = Wind_Solar_Out.T
            
            # unit conversion return divisor and energy units
            if not plot_as_percnt:
                unitconversion = mfunc.capacity_energy_unitconversion(Wind_Solar_Out.to_numpy().max())
                Wind_Solar_Out = Wind_Solar_Out/unitconversion['divisor']
    
            #Data table of values to return to main program
            Data_Table_Out = pd.concat([scenario_data.T,
                                        Wind_Solar_Out],  axis=0, sort=False)

            if not plot_as_percnt:
                Data_Table_Out = Data_Table_Out.add_suffix(f" ({unitconversion['units']}h)")
            
            
            for i, scenario in enumerate(self.Scenarios):
                month_data_scenario = pd.concat([scenario_data.T,Wind_Solar_Out],axis=0,sort=False)
                month_data_scenario = month_data_scenario.T
                month_data_scenario = month_data_scenario.loc[month_data_scenario['scenario']==scenario]
                month_data_scenario = month_data_scenario.drop('scenario',axis=1)
                
                month_data_scenario, angle = mfunc.check_label_angle(month_data_scenario, False)
                
                month_data_scenario.plot.bar(stacked=False, rot=angle, ax=axs[i],
                                  color=[self.PLEXOS_color_dict.get(x, '#333333') for x in month_data_scenario.columns], edgecolor='black', linewidth='0.1')
                axs[i].spines['right'].set_visible(False)
                axs[i].spines['top'].set_visible(False)
                axs[i].margins(x=0.01)
                axs[i].yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(x, f',.{self.y_axes_decimalpt}f')))
                if angle > 0:
                    axs[i].set_xticklabels(month_data_scenario.index, ha="right")
                    tick_length = 8
                else:
                    tick_length = 5
                axs[i].tick_params(axis='y', which='major', length=tick_length, width=1)
                axs[i].tick_params(axis='x', which='major', length=tick_length, width=1)
                
                handles, labels = axs[i].get_legend_handles_labels()
                axs[i].legend().set_visible(False)
            
            axs[grid_size-1].legend(reversed(handles),reversed(labels),
                                    loc = 'lower left',bbox_to_anchor=(1.05,0),
                                    facecolor='inherit', frameon=True)

            xlabels = [x.replace('_',' ') for x in self.xlabels]
            if self.ylabels == ['']:
                if plot_as_percnt:
                    ylabels = ["% of Generation"]
                else:
                    ylabels = [f"Generation ({unitconversion['units']}h)"]
            else:
                self.ylabels = [y.replace('_',' ') for y in self.ylabels]


            # add facet labels
            mfunc.add_facet_labels(fig1, xlabels, ylabels)
            
            fig1.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            if mconfig.parser('plot_title_as_region'):
                plt.title(zone_input)
            
            outputs[zone_input] = {'fig': fig1, 'data_table': Data_Table_Out}

        return outputs
    

    def total_gen_pie(self, figure_name=None, prop=None, start=None, end=None,
                  timezone=None, start_date_range=None, end_date_range=None):
        """Total Generation Pie Chart """
        
        # Create Dictionary to hold Datframes for each scenario
        
        outputs = {}
                    
        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True,"generator_Generation",self.Scenarios), 
                      (False,"generator_Energy_Curtailed",self.Scenarios)]


        # Runs get_data to populate mplot_data_dict with all required properties, returns a 1 if required data is missing
        check_input_data = mfunc.get_data(self.mplot_data_dict, properties,self.Marmot_Solutions_folder)

        if 1 in check_input_data:
            outputs = mfunc.MissingInputData()
            return outputs
        
        xdimension, ydimension = mfunc.setup_facet_xy_dimensions(self.xlabels,self.ylabels,multi_scenario=self.Scenarios)
        grid_size = xdimension*ydimension
        
         
        for zone_input in self.Zones:
            Total_Gen_Out = pd.DataFrame()
            self.logger.info(f"Zone = {zone_input}")

            #for scenario in self.Scenarios:
            fig1, axs = plt.subplots(ydimension,xdimension, figsize=((self.x*xdimension),(self.y*ydimension)), sharey=True, squeeze=False)
            plt.subplots_adjust(wspace=0.05, hspace=0.5)
            axs = axs.ravel()
               
            for i, scenario in enumerate(self.Scenarios):
                
                self.logger.info(f"Scenario = {scenario}")
                Total_Gen_Stack = self.mplot_data_dict['generator_Generation'].get(scenario)
    
                #Check if zone has generation, if not skips
                try:
                    Total_Gen_Stack = Total_Gen_Stack.xs(zone_input,level=self.AGG_BY)
                except KeyError:
                    self.logger.warning(f"No installed capacity in: {zone_input}")
                    continue
    
                Total_Gen_Stack = mfunc.df_process_gen_inputs(Total_Gen_Stack, self.ordered_gen)
                
                curtailment_name = self.gen_names_dict.get('Curtailment','Curtailment')
    
                #Insert Curtailmnet into gen stack if it exhists in database
                if self.mplot_data_dict["generator_Energy_Curtailed"]:
                    Stacked_Curt = self.mplot_data_dict["generator_Energy_Curtailed"].get(scenario)
                    if zone_input in Stacked_Curt.index.get_level_values(self.AGG_BY).unique():
                        Stacked_Curt = Stacked_Curt.xs(zone_input,level=self.AGG_BY)
                        Stacked_Curt = mfunc.df_process_gen_inputs(Stacked_Curt, self.ordered_gen)
                        Stacked_Curt = Stacked_Curt.sum(axis=1)
                        Total_Gen_Stack.insert(len(Total_Gen_Stack.columns),column=curtailment_name,value=Stacked_Curt) #Insert curtailment into
                        Total_Gen_Stack = Total_Gen_Stack.loc[:, (Total_Gen_Stack != 0).any(axis=0)]
                
                Total_Gen_Stack = Total_Gen_Stack.sum(axis=0)
                Total_Gen_Stack.rename(scenario, inplace=True)
                Total_Gen_Stack = (Total_Gen_Stack/sum(Total_Gen_Stack))*100
                
                Total_Gen_Out = pd.concat([Total_Gen_Out, Total_Gen_Stack], axis=1, sort=False).fillna(0)
                
            Total_Gen_Out = Total_Gen_Out.loc[:, (Total_Gen_Out != 0).any(axis=0)]
            
            if Total_Gen_Out.empty:
                out = mfunc.MissingZoneData()
                outputs[zone_input] = out
                continue
                        
            for i, scenario in enumerate(self.Scenarios):
                
                scenario_data = Total_Gen_Out[scenario]
               
                axs[i].pie(scenario_data,labels=scenario_data.index, 
                                       shadow=True,startangle=90, labeldistance=None,
                                       colors=[self.PLEXOS_color_dict.get(x, '#333333') for x in scenario_data.index]) #,
                
                handles, labels = axs[i].get_legend_handles_labels()
                axs[i].legend().set_visible(False)
            
            axs[grid_size-1].legend(reversed(handles),reversed(labels),
                                    loc = 'lower left',bbox_to_anchor=(1.05,0),
                                    facecolor='inherit', frameon=True)

            xlabels = [x.replace('_',' ') for x in self.xlabels]

            ylabels = ['']

            # add facet labels
            mfunc.add_facet_labels(fig1, xlabels, ylabels)
            
            fig1.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            if mconfig.parser('plot_title_as_region'):
                plt.title(zone_input)
            
            outputs[zone_input] = {'fig': fig1, 'data_table': Total_Gen_Out}

        return outputs
     