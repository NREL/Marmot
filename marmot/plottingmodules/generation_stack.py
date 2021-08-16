"""
Created on Mon Dec  9 10:34:48 2019
This code creates generation stack plots and is called from Marmot_plot_main.py
@author: dlevie
"""

import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
from matplotlib.patches import Patch
import numpy as np
import marmot.plottingmodules.marmot_plot_functions as mfunc
import marmot.config.mconfig as mconfig
import logging
#import textwrap


#mpl.rcParams['axes.titlesize'] = mconfig.parser("font_settings","title_size")
#===============================================================================

custom_legend_elements = Patch(facecolor='#DD0200',
                               alpha=0.5, edgecolor='#DD0200')

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


    def committed_stack(self, figure_name=None, prop=None, start=None, end=None,
                        timezone=None, start_date_range=None, end_date_range=None):
        outputs = {}

        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True,"generator_Installed_Capacity",[self.Scenarios[0]]),
                      (True,"generator_Generation",self.Scenarios),
                      (True,"generator_Units_Generating",self.Scenarios),
                      (True,"generator_Available_Capacity",self.Scenarios)]

        # Runs get_data to populate mplot_data_dict with all required properties, returns a 1 if required data is missing
        check_input_data = mfunc.get_data(self.mplot_data_dict, properties,self.Marmot_Solutions_folder)

        # Checks if all data required by plot is available, if 1 in list required data is missing
        if 1 in check_input_data:
            outputs = mfunc.MissingInputData()
            return outputs

        for zone_input in self.Zones:
            self.logger.info(f'Zone = {str(zone_input)}')

            #Get technology list.
            gens = self.mplot_data_dict['generator_Installed_Capacity'].get(self.Scenarios[0])
            try:
                gens = gens.xs(zone_input,level=self.AGG_BY)
            except KeyError:
                self.logger.warning(f"No Generation in: {zone_input}")
                out = mfunc.MissingZoneData()
                outputs[zone_input] = out
                continue

            tech_list = list(gens.reset_index().tech.unique())
            tech_list_sort = [tech_type for tech_type in self.ordered_gen if tech_type in tech_list and tech_type in self.thermal_gen_cat]

            if not tech_list_sort:
                self.logger.info(f'No Thermal Generation in: {zone_input}')
                out = mfunc.MissingZoneData()
                outputs[zone_input] = out
                continue

            xdimension = len(self.Scenarios)
            ydimension = len(tech_list_sort)

            fig4, axs = plt.subplots(ydimension,xdimension, figsize=((self.x*xdimension),(self.y*ydimension)), sharex = True, sharey='row',squeeze=False)
            plt.subplots_adjust(wspace=0.1, hspace=0.2)

            for i, scenario in enumerate(self.Scenarios):
                self.logger.info(f"Scenario = {scenario}")

                locator = mdates.AutoDateLocator(minticks = self.minticks, maxticks = self.maxticks)
                formatter = mdates.ConciseDateFormatter(locator)
                formatter.formats[2] = '%d\n %b'
                formatter.zero_formats[1] = '%b\n %Y'
                formatter.zero_formats[2] = '%d\n %b'
                formatter.zero_formats[3] = '%H:%M\n %d-%b'
                formatter.offset_formats[3] = '%b %Y'
                formatter.show_offset = False

                units_gen = self.mplot_data_dict['generator_Units_Generating'].get(scenario)
                avail_cap = self.mplot_data_dict['generator_Available_Capacity'].get(scenario)

                #Calculate  committed cap (for thermal only).
                thermal_commit_cap = units_gen * avail_cap
                thermal_commit_cap = thermal_commit_cap.xs(zone_input,level = self.AGG_BY)
                thermal_commit_cap = mfunc.df_process_gen_inputs(thermal_commit_cap,self.ordered_gen)
                thermal_commit_cap = thermal_commit_cap.loc[:, (thermal_commit_cap != 0).any(axis=0)]

                # unitconversion based off peak generation hour, only checked once
                if i == 0:
                    unitconversion = mfunc.capacity_energy_unitconversion(thermal_commit_cap.values.max())
                thermal_commit_cap = thermal_commit_cap/unitconversion['divisor']

                #Process generation.
                gen = self.mplot_data_dict['generator_Generation'].get(scenario)
                gen = gen.xs(zone_input,level = self.AGG_BY)
                gen = mfunc.df_process_gen_inputs(gen,self.ordered_gen)
                gen = gen.loc[:, (gen != 0).any(axis=0)]
                gen = gen/unitconversion['divisor']

                #Process available capacity (for VG only).
                avail_cap = avail_cap.xs(zone_input, level = self.AGG_BY)
                avail_cap = mfunc.df_process_gen_inputs(avail_cap,self.ordered_gen)
                avail_cap = avail_cap.loc[:, (avail_cap !=0).any(axis=0)]
                avail_cap = avail_cap/unitconversion['divisor']

                gen_lines = []
                for j,tech in enumerate(tech_list_sort):
                    if tech not in gen.columns:
                        gen_one_tech = pd.Series(0,index = gen.index)
                        commit_cap = pd.Series(0,index = gen.index) #Add dummy columns to deal with coal retirements (coal showing up in 2024, but not future years).
                    elif tech in self.thermal_gen_cat:
                        gen_one_tech = gen[tech]
                        commit_cap = thermal_commit_cap[tech]
                    else:
                        gen_one_tech = gen[tech]
                        commit_cap = avail_cap[tech]

                    gen_line = axs[j,i].plot(gen_one_tech,alpha = 0, color = self.PLEXOS_color_dict[tech])[0]
                    gen_lines.append(gen_line)
                    gen_fill = axs[j,i].fill_between(gen_one_tech.index,gen_one_tech,0, color = self.PLEXOS_color_dict[tech], alpha = 0.5)
                    if tech != 'Hydro':
                        cc = axs[j,i].plot(commit_cap, color = self.PLEXOS_color_dict[tech])

                    axs[j,i].spines['right'].set_visible(False)
                    axs[j,i].spines['top'].set_visible(False)
                    axs[j,i].tick_params(axis='y', which='major', length=5, width=1)
                    axs[j,i].tick_params(axis='x', which='major', length=5, width=1)
                    axs[j,i].yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(x, f',.{self.y_axes_decimalpt}f')))
                    axs[j,i].margins(x=0.01)
                    axs[j,i].xaxis.set_major_locator(locator)
                    axs[j,i].xaxis.set_major_formatter(formatter)
                    if j == 0:
                        axs[j,i].set_xlabel(xlabel = scenario, color = 'black')
                        axs[j,i].xaxis.set_label_position('top')
                    if i == 0:
                        axs[j,i].set_ylabel(ylabel = tech, rotation = 'vertical', color = 'black')

            #fig4.legend(gen_lines,labels = tech_list_sort, loc = 'right', title = 'RT Generation')
            fig4.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            if mconfig.parser("plot_title_as_region"):
                plt.title(zone_input)
            plt.ylabel(f"Generation or Committed Capacity ({unitconversion['units']})",  color='black', rotation='vertical', labelpad=60)
            data_table = pd.DataFrame() #TODO: write actual data out
            outputs[zone_input] = {'fig':fig4, 'data_table':data_table}
        return outputs


    def gen_stack(self, figure_name=None, prop=None, start=None, end=None,
                  timezone=None, start_date_range=None, end_date_range=None):

        facet=False
        if 'Facet' in figure_name:
            facet = True

        if self.AGG_BY == 'zone':
                agg = 'zone'
        else:
            agg = 'region'

        def set_dicts(scenario_list):


            # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
            # required True/False, property name and scenarios required, scenarios must be a list.
            properties = [(True,"generator_Generation",scenario_list),
                          (False,"generator_Curtailment",scenario_list),
                          (False,"generator_Pump_Load",scenario_list),
                          (True,f"{agg}_Load",scenario_list),
                          (False,f"{agg}_Unserved_Energy",scenario_list)]

            # Runs get_data to populate mplot_data_dict with all required properties, returns a 1 if required data is missing
            return mfunc.get_data(self.mplot_data_dict, properties,self.Marmot_Solutions_folder)


        def setup_data(zone_input, scenario, Stacked_Gen):

            curtailment_name = self.gen_names_dict.get('Curtailment','Curtailment')

            # Insert Curtailmnet into gen stack if it exhists in database
            if self.mplot_data_dict['generator_Curtailment']:
                Stacked_Curt = self.mplot_data_dict['generator_Curtailment'].get(scenario).copy()
                if self.shift_leapday == True:
                    Stacked_Curt = mfunc.shift_leapday(Stacked_Curt,self.Marmot_Solutions_folder)
                if zone_input in Stacked_Curt.index.get_level_values(self.AGG_BY).unique():
                    Stacked_Curt = Stacked_Curt.xs(zone_input,level=self.AGG_BY)
                    Stacked_Curt = mfunc.df_process_gen_inputs(Stacked_Curt, self.ordered_gen)
                    Stacked_Curt = Stacked_Curt.sum(axis=1)
                    Stacked_Curt[Stacked_Curt<0.05] = 0 #Remove values less than 0.05 MW
                    Stacked_Gen.insert(len(Stacked_Gen.columns),column=curtailment_name,value=Stacked_Curt) #Insert curtailment into
                    # Calculates Net Load by removing variable gen + curtailment
                    vre_gen_cat = self.vre_gen_cat + [curtailment_name]
                else:
                    vre_gen_cat = self.vre_gen_cat
                    
            else:
                vre_gen_cat = self.vre_gen_cat
            # Adjust list of values to drop depending on if it exhists in Stacked_Gen df
            vre_gen_cat = [name for name in vre_gen_cat if name in Stacked_Gen.columns]
            Net_Load = Stacked_Gen.drop(labels = vre_gen_cat, axis=1)
            Net_Load = Net_Load.sum(axis=1)

            # Removes columns that only contain 0
            Stacked_Gen = Stacked_Gen.loc[:, (Stacked_Gen != 0).any(axis=0)]


            Load = self.mplot_data_dict[f'{agg}_Load'].get(scenario).copy()
            if self.shift_leapday == True:
                Load = mfunc.shift_leapday(Load,self.Marmot_Solutions_folder)
            Load = Load.xs(zone_input,level=self.AGG_BY)
            Load = Load.groupby(["timestamp"]).sum()
            Load = Load.squeeze() #Convert to Series

            #######################
            ###DO NOT COMMIT
            #Use input load instead of zonal load.
            # Total_Demand = pd.read_csv('/Users/jnovache/Volumes/nrelnas01/PLEXOS CEII/Projects/Xcel_Weather/Load/load_2028_2011_EST.csv',index_col = 'DATETIME')
            # Total_Demand = Total_Demand['PSCO_WI']
            # Total_Demand.index = pd.to_datetime(Total_Demand.index)
            # Total_Demand.index = Total_Demand.index.shift(1,freq = 'D')
            # Total_Demand.index = Total_Demand.index.shift(-2,freq = 'H')
            # Total_Demand = Total_Demand.loc[Stacked_Gen.index]
            # Total_Demand = Total_Demand.squeeze()

            ###DO NOT COMMIT
            #######################

            if self.mplot_data_dict["generator_Pump_Load"] == {} or not mconfig.parser("plot_data","include_timeseries_pumped_load_line"):
                Pump_Load = self.mplot_data_dict['generator_Generation'][scenario].copy()
                Pump_Load.iloc[:,0] = 0
            else:
                Pump_Load = self.mplot_data_dict["generator_Pump_Load"][scenario]

            if self.shift_leapday == True:
                Pump_Load = mfunc.shift_leapday(Pump_Load,self.Marmot_Solutions_folder)
            Pump_Load = Pump_Load.xs(zone_input,level=self.AGG_BY)
            Pump_Load = Pump_Load.groupby(["timestamp"]).sum()
            Pump_Load = Pump_Load.squeeze() #Convert to Series
            if (Pump_Load == 0).all() == False:
                Total_Demand = Load - Pump_Load
                #Load = Total_Demand + Pump_Load
            else:
                Total_Demand = Load
                #Load = Total_Demand

            try:
                Unserved_Energy = self.mplot_data_dict[f'{agg}_Unserved_Energy'][scenario].copy()
            except KeyError:
                Unserved_Energy = self.mplot_data_dict[f'{agg}_Load'][scenario].copy()
                Unserved_Energy.iloc[:,0] = 0
            if self.shift_leapday == True:
                Unserved_Energy = mfunc.shift_leapday(Unserved_Energy,self.Marmot_Solutions_folder)

            Unserved_Energy = Unserved_Energy.xs(zone_input,level=self.AGG_BY)
            Unserved_Energy = Unserved_Energy.groupby(["timestamp"]).sum()
            Unserved_Energy = Unserved_Energy.squeeze() #Convert to Series

            unserved_eng_data_table = Unserved_Energy # Used for output to data table csv
            if (Unserved_Energy == 0).all() == False:
                Unserved_Energy = Load - Unserved_Energy

            data = {"Stacked_Gen":Stacked_Gen, "Load":Load, "Net_Load":Net_Load, "Pump_Load":Pump_Load, "Total_Demand":Total_Demand, "Unserved_Energy":Unserved_Energy,"ue_data_table":unserved_eng_data_table}
            return data

        def data_prop(data):

            Stacked_Gen = data["Stacked_Gen"]
            Load = data["Load"]
            Net_Load = data["Net_Load"]
            Pump_Load = data["Pump_Load"]
            Total_Demand = data["Total_Demand"]
            Unserved_Energy = data["Unserved_Energy"]
            unserved_eng_data_table = data["ue_data_table"]
            peak_demand_t = None
            Peak_Demand = 0
            min_net_load_t = None
            Min_Net_Load = 0

            if prop == "Peak Demand":
                peak_demand_t = Total_Demand.idxmax()
                end_date = peak_demand_t + dt.timedelta(days=end)
                start_date = peak_demand_t - dt.timedelta(days=start)
                Peak_Demand = Total_Demand[peak_demand_t]
                Stacked_Gen = Stacked_Gen[start_date : end_date]
                Load = Load[start_date : end_date]
                Unserved_Energy = Unserved_Energy[start_date : end_date]
                Total_Demand = Total_Demand[start_date : end_date]

                unserved_eng_data_table = unserved_eng_data_table[start_date : end_date]

            elif prop == "Min Net Load":
                min_net_load_t = Net_Load.idxmin()
                end_date = min_net_load_t + dt.timedelta(days=end)
                start_date = min_net_load_t - dt.timedelta(days=start)
                Min_Net_Load = Net_Load[min_net_load_t]
                Stacked_Gen = Stacked_Gen[start_date : end_date]
                Load = Load[start_date : end_date]
                Unserved_Energy = Unserved_Energy[start_date : end_date]
                Total_Demand = Total_Demand[start_date : end_date]

                unserved_eng_data_table = unserved_eng_data_table[start_date : end_date]

            elif prop == 'Date Range':

                self.logger.info(f"Plotting specific date range: \
                {str(start_date_range)} to {str(end_date_range)}")
                Stacked_Gen = Stacked_Gen[start_date_range : end_date_range]
                Load = Load[start_date_range : end_date_range]
                Unserved_Energy = Unserved_Energy[start_date_range : end_date_range]
                Total_Demand = Total_Demand[start_date_range : end_date_range]
                unserved_eng_data_table = unserved_eng_data_table[start_date_range : end_date_range]

            else:
                self.logger.info("Plotting graph for entire timeperiod")

            data = {"Stacked_Gen":Stacked_Gen, "Load":Load, "Pump_Load":Pump_Load, "Total_Demand":Total_Demand, "Unserved_Energy":Unserved_Energy,"ue_data_table":unserved_eng_data_table}
            data["peak_demand_t"] = peak_demand_t
            data["Peak_Demand"] = Peak_Demand
            data["min_net_load_t"] = min_net_load_t
            data["Min_Net_Load"] = Min_Net_Load
            return data

        def mkplot(outputs, zone_input, all_scenarios):

            # sets up x, y dimensions of plot
            xdimension, ydimension = mfunc.setup_facet_xy_dimensions(self.xlabels,self.ylabels,multi_scenario=all_scenarios)

            # If the plot is not a facet plot, grid size should be 1x1
            if not facet:
                xdimension = 1
                ydimension = 1

            grid_size = xdimension*ydimension

            # Used to calculate any excess axis to delete
            plot_number = len(all_scenarios)
            excess_axs = grid_size - plot_number

            fig1, axs = plt.subplots(ydimension,xdimension, figsize=((self.x*xdimension),(self.y*ydimension)), sharey=True, squeeze=False)
            plt.subplots_adjust(wspace=0.05, hspace=0.5)
            axs = axs.ravel()
            data_tables = []
            unique_tech_names = []

            for i, scenario in enumerate(all_scenarios):
                self.logger.info(f"Scenario = {scenario}")

                try:

                    Stacked_Gen = self.mplot_data_dict['generator_Generation'].get(scenario).copy()
                    if self.shift_leapday == True:
                        Stacked_Gen = mfunc.shift_leapday(Stacked_Gen,self.Marmot_Solutions_folder)
                    Stacked_Gen = Stacked_Gen.xs(zone_input,level=self.AGG_BY)
                except KeyError:
                    self.logger.warning(f'No generation in {zone_input}')
                    out = mfunc.MissingZoneData()
                    return out

                Stacked_Gen = mfunc.df_process_gen_inputs(Stacked_Gen, self.ordered_gen)
                data = setup_data(zone_input, scenario, Stacked_Gen)
                data = data_prop(data)
                
                # if no Generation return empty dataframe
                if data["Stacked_Gen"].empty == True:
                    self.logger.warning(f'No generation during time period in {zone_input}')
                    out = mfunc.MissingZoneData()
                    return out

                Stacked_Gen = data["Stacked_Gen"]
                Load = data["Load"]
                Pump_Load = data["Pump_Load"]
                Total_Demand = data["Total_Demand"]
                Unserved_Energy = data["Unserved_Energy"]
                unserved_eng_data_table = data["ue_data_table"]
                Peak_Demand = data["Peak_Demand"]
                peak_demand_t = data["peak_demand_t"]
                min_net_load_t = data["min_net_load_t"]
                Min_Net_Load = data["Min_Net_Load"]

                # unitconversion based off peak generation hour, only checked once
                if i == 0:
                    unitconversion = mfunc.capacity_energy_unitconversion(max(Stacked_Gen.sum(axis=1)))

                #Convert units
                Stacked_Gen = Stacked_Gen / unitconversion['divisor']
                Load = Load / unitconversion['divisor']
                Pump_Load = Pump_Load / unitconversion['divisor']
                Total_Demand = Total_Demand / unitconversion['divisor']
                Unserved_Energy = Unserved_Energy / unitconversion['divisor']
                unserved_eng_data_table = unserved_eng_data_table / unitconversion['divisor']
                Peak_Demand = Peak_Demand / unitconversion['divisor']
                Min_Net_Load = Min_Net_Load / unitconversion['divisor']

                Load = Load.rename('Total Load \n (Demand + Storage Charging)')
                Total_Demand = Total_Demand.rename('Total Demand')
                unserved_eng_data_table = unserved_eng_data_table.rename("Unserved Energy")
                
                # Data table of values to return to main program
                single_scen_out = pd.concat([Load, Total_Demand, unserved_eng_data_table, Stacked_Gen], axis=1, sort=False)
                scenario_names = pd.Series([scenario] * len(single_scen_out),name = 'Scenario')
                single_scen_out = single_scen_out.add_suffix(f" ({unitconversion['units']})")
                single_scen_out = single_scen_out.set_index([scenario_names],append = True)
                data_tables.append(single_scen_out)

                ##DO NOT COMMIT
                #Pull P05 hourly flow
                # interface_Flow_collection = {}
                # mfunc.get_data(interface_Flow_collection,"interface_Flow", self.Marmot_Solutions_folder, self.Scenarios)
                # int_flow = interface_Flow_collection[scenario]
                # int_flow = int_flow.xs('P05 West of Cascades-South_WI',level = 'interface_name')
                # int_flow = mfunc.shift_leapday(int_flow,self.Marmot_Solutions_folder)
                # int_flow = int_flow.droplevel('interface_category')
                # int_flow = int_flow[self.start_date : self.end_date]
                # int_flow = int_flow /unitconversion['divisor']
                # int_flow.columns = ['P05 flow']

                # # only difference linewidth = 0,5
                axs[i].stackplot(Stacked_Gen.index.values, Stacked_Gen.values.T, labels=Stacked_Gen.columns, linewidth=0,
                      colors=[self.PLEXOS_color_dict.get(x, '#333333') for x in Stacked_Gen.T.index])


                if (Unserved_Energy == 0).all() == False:
                    axs[i].plot(Unserved_Energy,
                                      #color='#EE1289'  OLD MARMOT COLOR
                                      color = '#DD0200' #SEAC STANDARD COLOR (AS OF MARCH 9, 2020)
                                      )

                lp = axs[i].plot(Load, color='black')

                if (Pump_Load == 0).all() == False:
                    lp3 = axs[i].plot(Total_Demand, color='black', linestyle="--")

                axs[i].spines['right'].set_visible(False)
                axs[i].spines['top'].set_visible(False)
                axs[i].tick_params(axis='y', which='major', length=5, width=1)
                axs[i].tick_params(axis='x', which='major', length=5, width=1)
                axs[i].yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(x, f',.{self.y_axes_decimalpt}f')))
                axs[i].margins(x=0.01)
                mfunc.set_plot_timeseries_format(axs,i)

                if prop == "Min Net Load":
                    axs[i].annotate(f"Min Net Load: \n{str(format(Min_Net_Load, '.2f'))} {unitconversion['units']}",
                                    xy=(min_net_load_t, Min_Net_Load), xytext=((min_net_load_t + dt.timedelta(days=0.1)),
                                                                               (max(Load))),
                        fontsize=13, arrowprops=dict(facecolor='black', width=3, shrink=0.1))

                # Peak Demand label overlaps other labels on a facet plot
                elif prop == "Peak Demand":
                    axs[i].annotate(f"Peak Demand: \n{str(format(Total_Demand[peak_demand_t], '.2f'))} {unitconversion['units']}",
                                    xy=(peak_demand_t, Peak_Demand), xytext=((peak_demand_t + dt.timedelta(days=0.1)),
                                                                             (max(Total_Demand) + Total_Demand[peak_demand_t]*0.1)),
                                fontsize=13, arrowprops=dict(facecolor='black', width=3, shrink=0.1))

                if (Unserved_Energy == 0).all() == False:
                    axs[i].fill_between(Load.index, Load,Unserved_Energy,
                                        # facecolor='#EE1289' OLD MARMOT COLOR
                                        facecolor = '#DD0200', #SEAC STANDARD COLOR (AS OF MARCH 9, 2020)
                                        alpha=0.5)

                # create list of gen technologies
                l1 = Stacked_Gen.columns.tolist()
                unique_tech_names.extend(l1)

            # create labels list of unique tech names then order
            labels = np.unique(np.array(unique_tech_names)).tolist()
            labels.sort(key = lambda i:self.ordered_gen.index(i))

            handles = []
            # create custom gen_tech legend
            for tech in labels:
                gen_legend_patches = Patch(facecolor=self.PLEXOS_color_dict[tech],
                            alpha=1.0)
                handles.append(gen_legend_patches)

            if (Pump_Load == 0).all() == False:
                handles.append(lp3[0])
                handles.append(lp[0])
                labels += ['Demand','Demand + \n Storage Charging']

            else:
                handles.append(lp[0])
                labels += ['Demand']

            if (Unserved_Energy == 0).all() == False:
                handles.append(custom_legend_elements)
                labels += ['Unserved Energy']

            axs[grid_size-1].legend(reversed(handles),reversed(labels),
                                    loc = 'lower left',bbox_to_anchor=(1.05,0),
                                    facecolor='inherit', frameon=True)

            xlabels = [x.replace('_',' ') for x in self.xlabels]
            ylabels = [y.replace('_',' ') for y in self.ylabels]

            # add facet labels
            mfunc.add_facet_labels(fig1, xlabels, ylabels)

            fig1.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            if mconfig.parser('plot_title_as_region'):
                plt.title(zone_input)

            #Ylabel should change if there are facet labels, leave at 40 for now, works for all values in spacing
            labelpad = 40
            plt.ylabel(f"Generation ({unitconversion['units']})", color='black', rotation='vertical', labelpad = labelpad)

            #Remove extra axes
            if excess_axs != 0:
                mfunc.remove_excess_axs(axs,excess_axs,grid_size)
            Data_Table_Out = pd.concat(data_tables)
            out = {'fig':fig1, 'data_table':Data_Table_Out}
            return out

        #TODO: combine data_prop(), setup_data(), mkplot(), into gen_stack()
                                                             
        # Main loop for gen_stack
        outputs = {}
        if facet:
            check_input_data = set_dicts(self.Scenarios)
        else:
            check_input_data = set_dicts([self.Scenarios[0]])

        # Checks if all data required by plot is available, if 1 in list required data is missing
        if 1 in check_input_data:
            outputs = mfunc.MissingInputData()
            return outputs

        xdimension=len(self.xlabels)
        if xdimension == 0:
                xdimension = 1

        # If the plot is not a facet plot, grid size should be 1x1
        if not facet:
            xdimension = 1

        # If creating a facet plot the font is scaled by 9% for each added x dimesion fact plot
        if xdimension > 1:
            font_scaling_ratio = 1 + ((xdimension-1)*0.09)
            plt.rcParams['xtick.labelsize'] = plt.rcParams['xtick.labelsize']*font_scaling_ratio
            plt.rcParams['ytick.labelsize'] = plt.rcParams['ytick.labelsize']*font_scaling_ratio
            plt.rcParams['legend.fontsize'] = plt.rcParams['legend.fontsize']*font_scaling_ratio
            plt.rcParams['axes.labelsize'] = plt.rcParams['axes.labelsize']*font_scaling_ratio
            plt.rcParams['axes.titlesize'] =  plt.rcParams['axes.titlesize']*font_scaling_ratio
 

        for zone_input in self.Zones:
            self.logger.info(f"Zone = {zone_input}")

            if facet:
                outputs[zone_input] = mkplot(outputs, zone_input, self.Scenarios)
            else:
                outputs[zone_input] = mkplot(outputs, zone_input, [self.Scenarios[0]])
        return outputs


    def gen_diff(self, figure_name=None, prop=None, start=None, end=None,
                 timezone=None, start_date_range=None, end_date_range=None):
        outputs = {}

        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True,"generator_Generation",self.Scenarios)]

        # Runs get_data to populate mplot_data_dict with all required properties, returns a 1 if required data is missing
        check_input_data = mfunc.get_data(self.mplot_data_dict, properties,self.Marmot_Solutions_folder)

        if 1 in check_input_data:
            outputs = mfunc.MissingInputData()
            return outputs

        for zone_input in self.Zones:
            self.logger.info(f"Zone = {zone_input}")
            # Create Dictionary to hold Datframes for each scenario

            Total_Gen_Stack_1 = self.mplot_data_dict['generator_Generation'].get(self.Scenario_Diff[0])
            if Total_Gen_Stack_1 is None:
                self.logger.warning(f'Scenario_Diff "{self.Scenario_Diff[0]}" is not in data. Ensure User Input Sheet is set up correctly!')
                outputs = mfunc.InputSheetError()
                return outputs
            
            if zone_input not in Total_Gen_Stack_1.index.get_level_values(self.AGG_BY).unique():
                outputs[zone_input] = mfunc.MissingZoneData()
                continue
                
            Total_Gen_Stack_1 = Total_Gen_Stack_1.xs(zone_input,level=self.AGG_BY)
            Total_Gen_Stack_1 = mfunc.df_process_gen_inputs(Total_Gen_Stack_1, self.ordered_gen)
            #Adds in all possible columns from ordered gen to ensure the two dataframes have same column names
            Total_Gen_Stack_1 = pd.DataFrame(Total_Gen_Stack_1, columns = self.ordered_gen).fillna(0)

            Total_Gen_Stack_2 = self.mplot_data_dict['generator_Generation'].get(self.Scenario_Diff[1])
            if Total_Gen_Stack_2 is None:
                self.logger.warning(f'Scenario_Diff "{self.Scenario_Diff[1]}" is not in data. Ensure User Input Sheet is set up correctly!')
                outputs = mfunc.InputSheetError()
                return outputs

            Total_Gen_Stack_2 = Total_Gen_Stack_2.xs(zone_input,level=self.AGG_BY)
            Total_Gen_Stack_2 = mfunc.df_process_gen_inputs(Total_Gen_Stack_2, self.ordered_gen)
            #Adds in all possible columns from ordered gen to ensure the two dataframes have same column names
            Total_Gen_Stack_2 = pd.DataFrame(Total_Gen_Stack_2, columns = self.ordered_gen).fillna(0)

            self.logger.info(f'Scenario 1 = {self.Scenario_Diff[0]}')
            self.logger.info(f'Scenario 2 = {self.Scenario_Diff[1]}')
            Gen_Stack_Out = Total_Gen_Stack_1-Total_Gen_Stack_2

            if prop == 'Date Range':
                self.logger.info(f"Plotting specific date range: \
                {str(start_date_range)} to {str(end_date_range)}")
                Gen_Stack_Out = Gen_Stack_Out[start_date_range : end_date_range]
            else:
                self.logger.info("Plotting graph for entire timeperiod")

            # Removes columns that only equal 0
            Gen_Stack_Out.dropna(inplace=True)
            Gen_Stack_Out = Gen_Stack_Out.loc[:, (Gen_Stack_Out != 0).any(axis=0)]

            if Gen_Stack_Out.empty == True:
                outputs[zone_input] = mfunc.MissingZoneData()
                continue

            # Reverses order of columns
            Gen_Stack_Out = Gen_Stack_Out.iloc[:, ::-1]

            unitconversion = mfunc.capacity_energy_unitconversion(max(Gen_Stack_Out.sum(axis=1)))
            Gen_Stack_Out = Gen_Stack_Out/unitconversion['divisor']

            # Data table of values to return to main program
            Data_Table_Out = Gen_Stack_Out.add_suffix(f" ({unitconversion['units']})")

            fig3, axs = mfunc.setup_plot()
            # Flatten object
            ax = axs[0]

            for column in Gen_Stack_Out:
                ax.plot(Gen_Stack_Out[column], linewidth=3, color=self.PLEXOS_color_dict[column],
                        label=column)
                ax.legend(loc='lower left',bbox_to_anchor=(1,0),
                              facecolor='inherit', frameon=True)

            ax.set_title(self.Scenario_Diff[0].replace('_', ' ') + " vs. " + self.Scenario_Diff[1].replace('_', ' '))
            ax.set_ylabel(f"Generation Difference ({unitconversion['units']})",  color='black', rotation='vertical')
            ax.set_xlabel(f'Date ({timezone})',  color='black', rotation='horizontal')
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.tick_params(axis='y', which='major', length=5, width=1)
            ax.tick_params(axis='x', which='major', length=5, width=1)
            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(x, f',.{self.y_axes_decimalpt}f')))
            ax.margins(x=0.01)

            mfunc.set_plot_timeseries_format(axs)
            outputs[zone_input] = {'fig': fig3, 'data_table': Data_Table_Out}
        return outputs


    def gen_stack_all_periods(self, figure_name=None, prop=None, start=None, end=None,
                              timezone=None, start_date_range=None, end_date_range=None):
        '''
        DEPRCIATED FOR NOW

        Returns
        -------
        outputs : mfunc.UnderDevelopment()

        '''

        outputs = mfunc.UnderDevelopment()
        self.logger.warning('total_gen_facet is under development')
        return outputs

    #     #Location to save to
    #     gen_stack_figures = os.path.join(self.figure_folder, self.AGG_BY + '_Gen_Stack')

    #     Stacked_Gen_read = pd.read_hdf(self.hdf_out_folder + "/" + self.Scenarios[0]+"_formatted.h5", 'generator_Generation')
    #     try:
    #         Pump_Load_read =pd.read_hdf(self.hdf_out_folder + "/" + self.Scenarios[0]+"_formatted.h5", "generator_Pump_Load" )
    #     except:
    #         Pump_Load_read = Stacked_Gen_read.copy()
    #         Pump_Load_read.iloc[:,0] = 0
    #     Stacked_Curt_read = pd.read_hdf(self.hdf_out_folder + "/" + self.Scenarios[0]+"_formatted.h5", "generator_Curtailment" )

    #     # If data is to be aggregated by zone, then zone properties are loaded, else region properties are loaded
    #     if self.AGG_BY == "zone":
    #         Load_read = pd.read_hdf(self.hdf_out_folder + "/" + self.Scenarios[0]+"_formatted.h5", "zone_Load")
    #         try:
    #             Unserved_Energy_read = pd.read_hdf(self.hdf_out_folder + "/" + self.Scenarios[0]+"_formatted.h5", "zone_Unserved_Energy" )
    #         except:
    #             Unserved_Energy_read = Load_read.copy()
    #             Unserved_Energy_read.iloc[:,0] = 0
    #     else:
    #         Load_read = pd.read_hdf(self.hdf_out_folder + "/" + self.Scenarios[0]+"_formatted.h5", "region_Load")
    #         try:
    #             Unserved_Energy_read = pd.read_hdf(self.hdf_out_folder + "/" + self.Scenarios[0]+"_formatted.h5", "region_Unserved_Energy" )
    #         except:
    #             Unserved_Energy_read = Load_read.copy()
    #             Unserved_Energy_read.iloc[:,0] = 0

    #     outputs = {}
    #     for zone_input in self.Zones:

    #         self.logger.info("Zone = "+ zone_input)


    #        # try:   #The rest of the function won't work if this particular zone can't be found in the solution file (e.g. if it doesn't include Mexico)
    #         Stacked_Gen = Stacked_Gen_read.xs(zone_input,level=self.AGG_BY)
    #         del Stacked_Gen_read
    #         Stacked_Gen = mfunc.df_process_gen_inputs(Stacked_Gen, self.ordered_gen)

    #         try:
    #             Stacked_Curt = Stacked_Curt_read.xs(zone_input,level=self.AGG_BY)
    #             del Stacked_Curt_read
    #             Stacked_Curt = mfunc.df_process_gen_inputs(Stacked_Curt, self.ordered_gen)
    #             Stacked_Curt = Stacked_Curt.sum(axis=1)
    #             Stacked_Curt[Stacked_Curt<0.05] = 0 #Remove values less than 0.05 MW
    #             Stacked_Gen.insert(len(Stacked_Gen.columns),column='Curtailment',value=Stacked_Curt) #Insert curtailment into
    #         except Exception:
    #             pass

    #         # Calculates Net Load by removing variable gen + curtailment
    #         self.vre_gen_cat = self.vre_gen_cat + ['Curtailment']
    #         # Adjust list of values to drop depending on if it exhists in Stacked_Gen df
    #         self.vre_gen_cat = [name for name in self.vre_gen_cat if name in Stacked_Gen.columns]
    #         Net_Load = Stacked_Gen.drop(labels = self.vre_gen_cat, axis=1)
    #         Net_Load = Net_Load.sum(axis=1)

    #         # Removes columns that only contain 0
    #         Stacked_Gen = Stacked_Gen.loc[:, (Stacked_Gen != 0).any(axis=0)]

    #         Load = Load_read.xs(zone_input,level=self.AGG_BY)
    #         del Load_read
    #         Load = Load.groupby(["timestamp"]).sum()
    #         Load = Load.squeeze() #Convert to Series

    #         Pump_Load = Pump_Load_read.xs(zone_input,level=self.AGG_BY)
    #         del Pump_Load_read
    #         Pump_Load = Pump_Load.groupby(["timestamp"]).sum()
    #         Pump_Load = Pump_Load.squeeze() #Convert to Series
    #         if (Pump_Load == 0).all() == False:
    #             Total_Demand = Load - Pump_Load
    #         else:
    #             Total_Demand = Load

    #         Unserved_Energy = Unserved_Energy_read.xs(zone_input,level=self.AGG_BY)
    #         del Unserved_Energy_read
    #         Unserved_Energy = Unserved_Energy.groupby(["timestamp"]).sum()
    #         Unserved_Energy = Unserved_Energy.squeeze() #Convert to Series
    #         unserved_eng_data_table = Unserved_Energy # Used for output to data table csv
    #         if (Unserved_Energy == 0).all() == False:
    #             Unserved_Energy = Load - Unserved_Energy

    #         Load = Load.rename('Total Load (Demand + Storage Charging)')
    #         Total_Demand = Total_Demand.rename('Total Demand')
    #         unserved_eng_data_table = unserved_eng_data_table.rename("Unserved Energy")


    #         first_date=Stacked_Gen.index[0]
    #         for wk in range(1,53): #assumes weekly, could be something else if user changes end Marmot_plot_select

    #             period_start=first_date+dt.timedelta(days=(wk-1)*7)
    #             period_end=period_start+dt.timedelta(days=end)
    #             self.logger.info(str(period_start)+" and next "+str(end)+" days.")
    #             Stacked_Gen_Period = Stacked_Gen[period_start:period_end]
    #             Load_Period = Load[period_start:period_end]
    #             Unserved_Energy_Period = Unserved_Energy[period_start:period_end]
    #             Total_Demand_Period = Total_Demand[period_start:period_end]
    #             unserved_eng_data_table_period = unserved_eng_data_table[period_start:period_end]


    #             # Data table of values to return to main program
    #             Data_Table_Out = pd.concat([Load_Period, Total_Demand_Period, unserved_eng_data_table_period, Stacked_Gen_Period], axis=1, sort=False)

    #             fig1, ax = plt.subplots(figsize=(9,6))
    #             ax.stackplot(Stacked_Gen_Period.index.values, Stacked_Gen_Period.values.T, labels=Stacked_Gen_Period.columns, linewidth=5,colors=[self.PLEXOS_color_dict.get(x, '#333333') for x in Stacked_Gen_Period.T.index])

    #             if (Unserved_Energy_Period == 0).all() == False:
    #                 plt.plot(Unserved_Energy_Period,
    #                                #color='#EE1289'  OLD MARMOT COLOR
    #                                color = '#DD0200' #SEAC STANDARD COLOR (AS OF MARCH 9, 2020)
    #                                )

    #             lp1 = plt.plot(Load_Period, color='black')

    #             if (Pump_Load == 0).all() == False:
    #                 lp3 = plt.plot(Total_Demand_Period, color='black', linestyle="--")


    #             ax.set_ylabel('Generation (MW)',  color='black', rotation='vertical')
    #             ax.set_xlabel('Date ' + '(' + str(timezone) + ')',  color='black', rotation='horizontal')
    #             ax.spines['right'].set_visible(False)
    #             ax.spines['top'].set_visible(False)
    #             ax.tick_params(axis='y', which='major', length=5, width=1)
    #             ax.tick_params(axis='x', which='major', length=5, width=1)
    #             ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    #             ax.margins(x=0.01)

    #             locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
    #             formatter = mdates.ConciseDateFormatter(locator)
    #             formatter.formats[2] = '%d\n %b'
    #             formatter.zero_formats[1] = '%b\n %Y'
    #             formatter.zero_formats[2] = '%d\n %b'
    #             formatter.zero_formats[3] = '%H:%M\n %d-%b'
    #             formatter.offset_formats[3] = '%b %Y'
    #             formatter.show_offset = False
    #             ax.xaxis.set_major_locator(locator)
    #             ax.xaxis.set_major_formatter(formatter)


    #             if (Unserved_Energy_Period == 0).all() == False:
    #                 ax.fill_between(Load_Period.index, Load_Period,Unserved_Energy_Period,
    #                                 #facecolor='#EE1289'
    #                                 facecolor = '#DD0200',
    #                                 alpha=0.5)

    #             handles, labels = ax.get_legend_handles_labels()

    #             if (Pump_Load == 0).all() == False:
    #                 handles.append(lp3[0])
    #                 handles.append(lp1[0])
    #                 labels += ['Demand','Demand + \n Storage Charging']

    #             else:
    #                 handles.append(lp1[0])
    #                 labels += ['Demand']

    #             if (Unserved_Energy_Period == 0).all() == False:
    #                 handles.append(custom_legend_elements)
    #                 labels += ['Unserved Energy']

    #             ax.legend(reversed(handles),reversed(labels),
    #                                     loc = 'lower left',bbox_to_anchor=(1.05,0),
    #                                     facecolor='inherit', frameon=True)

    #             fig1.savefig(os.path.join(gen_stack_figures, zone_input + "_" + "Stacked_Gen_All_Periods" + "_" + self.Scenarios[0]+"_period_"+str(wk)), dpi=600, bbox_inches='tight')
    #             Data_Table_Out.to_csv(os.path.join(gen_stack_figures, zone_input + "_" + "Stacked_Gen_All_Periods" + "_" + self.Scenarios[0]+"_period_"+str(wk)+ ".csv"))
    #             del fig1
    #             del Data_Table_Out
    #             mpl.pyplot.close('all')

    #     outputs = mfunc.DataSavedInModule()
    #     #end weekly loop
    #     return outputs


