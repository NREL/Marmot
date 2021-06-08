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

class mplot(object):

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
                      (False,"generator_Curtailment",self.Scenarios),
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
                interval_count = mfunc.get_interval_count(Total_Gen_Stack)

                curtailment_name = self.gen_names_dict.get('Curtailment','Curtailment')

                # Insert Curtailmnet into gen stack if it exhists in database
                if self.mplot_data_dict["generator_Curtailment"]:
                    Stacked_Curt = self.mplot_data_dict["generator_Curtailment"].get(scenario)
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
                    self.logger.info(f"Plotting specific date range: \
                                     {str(start_date_range)} to {str(end_date_range)}")
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
                    self.logger.info(f"Plotting specific date range: \
                                     {str(start_date_range)} to {str(end_date_range)}")
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
                    self.logger.info(f"Plotting specific date range: \
                                     {str(start_date_range)} to {str(end_date_range)}")
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
            Total_Generation_Stack_Out.index = Total_Generation_Stack_Out.index.str.wrap(5, break_long_words=False)

            fig1, ax = plt.subplots(figsize=(self.x,self.y))

            Total_Generation_Stack_Out.plot.bar(stacked=True, rot=0, ax=ax,
                             color=[self.PLEXOS_color_dict.get(x, '#333333') for x in Total_Generation_Stack_Out.columns], edgecolor='black', linewidth='0.1')
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_ylabel(f"Total Genertaion ({unitconversion['units']}h)",  color='black', rotation='vertical')
            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(x, f',.{self.y_axes_decimalpt}f')))
            ax.tick_params(axis='y', which='major', length=5, width=1)
            ax.tick_params(axis='x', which='major', length=5, width=1)
            if mconfig.parser("plot_title_as_region"):
                ax.set_title(zone_input)

            for n, scenario in enumerate(self.Scenarios):

                x = [ax.patches[n].get_x(), ax.patches[n].get_x() + ax.patches[n].get_width()]
                height1 = [int(Total_Load_Out[scenario].sum())]*2
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
                      (False,"generator_Curtailment",self.Scenarios)]

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
                interval_count = mfunc.get_interval_count(Total_Gen_Stack)

                curtailment_name = self.gen_names_dict.get('Curtailment','Curtailment')

                # Insert Curtailmnet into gen stack if it exhists in database
                if self.mplot_data_dict["generator_Curtailment"]:
                    Stacked_Curt = self.mplot_data_dict["generator_Curtailment"].get(scenario)
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
            Total_Generation_Stack_Out.index = Total_Generation_Stack_Out.index.str.wrap(10, break_long_words=False)

            fig1, ax = plt.subplots(figsize=(self.x,self.y))
            Total_Generation_Stack_Out.plot.bar(stacked=True, rot=0,
                             color=[self.PLEXOS_color_dict.get(x, '#333333') for x in Total_Generation_Stack_Out.columns], edgecolor='black', linewidth='0.1',ax=ax)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(x, f',.{self.y_axes_decimalpt}f')))
            ax.tick_params(axis='y', which='major', length=5, width=1)
            ax.tick_params(axis='x', which='major', length=5, width=1)

            #Add net gen difference line.
            for n, scenario in enumerate(self.Scenarios[1:]):
                x = [ax.patches[n].get_x(), ax.patches[n].get_x() + ax.patches[n].get_width()]
                y_net = [net_diff.loc[scenario]] * 2
                net_line = plt.plot(x,y_net, c='black', linewidth=1.5)

            locs,labels=plt.xticks()

            ax.set_ylabel(f"Generation Change ({format(unitconversion['units'])}h) \n relative to {self.Scenarios[0].replace('_',' ')}",  color='black', rotation='vertical')

            xlabels = [textwrap.fill(x.replace('_',' '),10) for x in self.xlabels]

            plt.xticks(ticks=locs,labels=xlabels[1:])
            ax.margins(x=0.01)

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
    #     self.mplot_data_dict["generator_Curtailment"] = {}

    #     for scenario in self.Scenarios:
    #         try:
    #             self.mplot_data_dict['generator_Generation'][scenario] = pd.read_hdf(os.path.join(self.Marmot_Solutions_folder, scenario,"Processed_HDF5_folder", scenario+ "_formatted.h5"), "generator_Generation")
    #             self.mplot_data_dict["generator_Curtailment"][scenario] = pd.read_hdf(os.path.join(self.Marmot_Solutions_folder, scenario,"Processed_HDF5_folder", scenario+ "_formatted.h5"),  "generator_Curtailment")
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
    #             Stacked_Curt = self.mplot_data_dict["generator_Curtailment"].get(scenario)
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
    #     Total_Generation_Stack_Out.index = Total_Generation_Stack_Out.index.str.wrap(11, break_long_words=False)

    #     Total_Load_Out.index = Total_Load_Out.index.str.replace('_',' ')
    #     Total_Load_Out.index = Total_Load_Out.index.str.wrap(11, break_long_words=False)

    #     Total_Load_Out = Total_Load_Out.T/1000

    #     xdimension=len(self.xlabels)
    #     ydimension=len(self.ylabels)
    #     grid_size = xdimension*ydimension

    #     fig2, axs = plt.subplots(ydimension,xdimension, figsize=((2*xdimension), (4*ydimension)), sharey=True)
    #     axs = axs.ravel()
    #     plt.subplots_adjust(wspace=0, hspace=0.01)

    #     i=0
    #     for index in Total_Generation_Stack_Out.index:

    #         sb = Total_Generation_Stack_Out.iloc[i:i+1].plot.bar(stacked=True, rot=0,
    #         color=[self.PLEXOS_color_dict.get(x, '#333333') for x in Total_Generation_Stack_Out.columns], edgecolor='black', linewidth='0.1',
    #                                      ax=axs[i])

    #         axs[i].get_legend().remove()
    #         axs[i].spines['right'].set_visible(False)
    #         axs[i].spines['top'].set_visible(False)
    #         axs[i].xaxis.set_ticklabels([])
    #         axs[i].tick_params(axis='y', which='major', length=5, width=1)
    #         axs[i].tick_params(axis='x', which='major', length=5, width=1)
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
