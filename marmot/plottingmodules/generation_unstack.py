"""
Created on Mon Dec  9 10:34:48 2019
This code creates generation UNstacked plots and is called from Marmot_plot_main.py
@author: Daniel Levie
"""
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np
import marmot.plottingmodules.marmot_plot_functions as mfunc
import marmot.config.mconfig as mconfig
import logging
import textwrap

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


    def gen_unstack(self, figure_name=None, prop=None, start=None, end=None, 
                        timezone="", start_date_range=None, end_date_range=None):
        outputs = {}  
        
        facet=False
        if 'Facet' in figure_name:
            facet = True
            
        if self.AGG_BY == 'zone':
                agg = 'zone'
        else:
            agg = 'region'
            
        def getdata(scenario_list):
            
            # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
            # required True/False, property name and scenarios required, scenarios must be a list.
            properties = [(True,"generator_Generation",scenario_list),
                          (False,f"generator_{self.curtailment_prop}",scenario_list),
                          (False,"generator_Pump_Load",scenario_list),
                          (True,f"{agg}_Load",scenario_list),
                          (False,f"{agg}_Unserved_Energy",scenario_list),
                          (True,"line_Flow",scenario_list)]
            
            # Runs get_data to populate mplot_data_dict with all required properties, returns a 1 if required data is missing
            return mfunc.get_data(self.mplot_data_dict, properties,self.Marmot_Solutions_folder)
        
        if facet:
            check_input_data = getdata(self.Scenarios)
            all_scenarios = self.Scenarios
        else:
            check_input_data = getdata([self.Scenarios[0]])  
            all_scenarios = [self.Scenarios[0]]
        
        # Checks if all data required by plot is available, if 1 in list required data is missing
        if 1 in check_input_data:
            outputs = mfunc.MissingInputData()
            return outputs
            
        # sets up x, y dimensions of plot
        xdimension, ydimension = mfunc.setup_facet_xy_dimensions(self.xlabels,self.ylabels,multi_scenario=all_scenarios)

        # If the plot is not a facet plot, grid size should be 1x1
        if not facet:
            xdimension = 1
            ydimension = 1
        
        if prop=='VRE_compare':
            xdimension=(len(self.vre_gen_cat)-1)

        # If creating a facet plot the font is scaled by 9% for each added x dimesion fact plot
        if xdimension > 1:
            font_scaling_ratio = 1 + ((xdimension-1)*0.09)
            plt.rcParams['xtick.labelsize'] = plt.rcParams['xtick.labelsize']*font_scaling_ratio
            plt.rcParams['ytick.labelsize'] = plt.rcParams['ytick.labelsize']*font_scaling_ratio
            plt.rcParams['legend.fontsize'] = plt.rcParams['legend.fontsize']*font_scaling_ratio
            plt.rcParams['axes.labelsize'] = plt.rcParams['axes.labelsize']*font_scaling_ratio
            plt.rcParams['axes.titlesize'] =  plt.rcParams['axes.titlesize']*font_scaling_ratio
        
        grid_size = xdimension*ydimension
            
        # Used to calculate any excess axis to delete
        plot_number = len(all_scenarios)*xdimension
        
        for zone_input in self.Zones:
            self.logger.info(f"Zone = {zone_input}")
        
            excess_axs = grid_size - plot_number
        
            fig1, axs = plt.subplots(ydimension,xdimension, figsize=((self.x*xdimension),(self.y*ydimension)), sharey=True, squeeze=False)
            plt.subplots_adjust(wspace=0.05, hspace=0.5)
            axs = axs.ravel()
            data_tables = []
            unique_tech_names = []


            for i, scenario in enumerate(all_scenarios):
                self.logger.info(f"Scenario = {scenario}")
                # Pump_Load = pd.Series() # Initiate pump load

                #here we can also decide to grab imports to plot them vs. generation to see if some correlations going on?
                if prop=='VRE_compare':
                    
                    #record which interfaces are in the zone?
                    if self.AGG_BY=='zone':
                        zone_lines = self.meta.zone_lines()
                    else:
                        zone_lines = self.meta.region_lines()
                    try:
                        zone_lines = zone_lines.set_index([self.AGG_BY])
                    except:
                        self.logger.info("Column to Aggregate by is missing")
                        continue

                    zone_lines = zone_lines.xs(zone_input)
                    zone_lines = zone_lines['line_name'].unique()

                    #Map lines to interfaces
                    # all_ints = self.meta.interface_lines() #Map lines to interfaces
                    all_ints = self.meta.region_interregionallines()
                    all_ints.index = all_ints.line_name
                    ints = all_ints.loc[all_ints.index.intersection(zone_lines)]

                    #flow = flow[flow.index.get_level_values('interface_name').isin(ints.interface)] #Limit to only interfaces touching to this zone
                    #flow = flow.droplevel('interface_category')

                    # export_limits = self.mplot_data_dict["interface_Export_Limit"].get(scenario).droplevel('timestamp')
                    # export_limits.mask(export_limits[0]==0.0,other=0.01,inplace=True) #if limit is zero set to small value
                    # export_limits = export_limits[export_limits.index.get_level_values('interface_name').isin(ints.interface)]
                    # export_limits = export_limits[export_limits[0].abs() < 99998] #Filter out unenforced interfaces.

                    # #Drop unnecessary columns.
                    # export_limits.reset_index(inplace = True)
                    # export_limits.drop(columns = 'interface_category',inplace = True)
                    # export_limits.set_index('interface_name',inplace = True)

                    # import_limits = self.mplot_data_dict["interface_Import_Limit"].get(scenario).droplevel('timestamp')
                    # import_limits.mask(import_limits[0]==0.0,other=0.01,inplace=True) #if limit is zero set to small value
                    # import_limits = import_limits[import_limits.index.get_level_values('interface_name').isin(ints.interface)]
                    # import_limits = import_limits[import_limits[0].abs() < 99998] #Filter out unenforced interfaces.
                    # reported_ints = import_limits.index.get_level_values('interface_name').unique()
                    # print(reported_ints)

                    # flow_all = self.mplot_data_dict["interface_Flow"][scenario]
                    flow_all = self.mplot_data_dict['line_Flow'][scenario].copy()
                    # self.mplot_data_dict["line_Flow"][scenario].copy()
                    # self.meta.region_interregionallines()
                    flow_zone = flow_all[flow_all.index.get_level_values('line_name').isin(ints.index)]
                    reported_ints = flow_zone.index.get_level_values('line_name').unique()
                    

                    flow_zone.reset_index(inplace = True)
                    #flow_zone.drop(columns = 'interface_category',inplace = True)
                    fz = flow_zone.groupby(['timestamp'],as_index=True).sum()
                    print(fz) #hopefully close to ready to plot!!!
                    print('sdf')
                    #import_limits.set_index('interface_name',inplace = True)

                    # flow = flow_all.xs(inter,level = 'interface_name')
                    # flow = flow.reset_index()
                    
                    # if pd.notna(start_date_range):
                    #     self.logger.info("Plotting specific date range: \
                    #     {} to {}".format(str(start_date_range),str(end_date_range)))
                    #     flow = flow[start_date_range : end_date_range]
                
                    # flow = flow[0]

                    # pos_sing = pd.Series(flow.where(flow > 0).sum())
                    # pos = pos.append(pos_sing)
                    # neg_sing = pd.Series(flow.where(flow < 0).sum())
                    # neg = neg.append(neg_sing)
                    
                    # if scenario == self.Scenarios[0]:
                    #     max_val = max(pos.max(),abs(neg.max()))
                    #     unitconversion = mfunc.capacity_energy_unitconversion(max_val)

                    # both = pd.concat([pos,neg],axis = 1)
                    # both.columns = ['Total Export','Total Import']
                    # both = both / unitconversion['divisor']
                    # both.index = available_inter
                    # net_flows_all.append(both)

                    # #Add scenario column to output table.
                    # scenario_names = pd.Series([scenario] * len(both),name = 'Scenario')
                    # data_table = both.set_index([scenario_names],append = True)
                    # data_table = data_table.add_suffix(f" ({unitconversion['units']})")
                    # data_out_chunk.append(data_table)

                # the comparison generation stack, if desired, should filter on interconnect
                if prop=='VRE_compare':
                    try:
                        interconnect_Stacked_Gen = self.mplot_data_dict["generator_Generation"].get(scenario).copy()
                        interconnect_Stacked_Gen_reset = interconnect_Stacked_Gen.reset_index() 
                        interconnect = interconnect_Stacked_Gen_reset[interconnect_Stacked_Gen_reset[self.AGG_BY]==zone_input]['Interconnection'].unique()[0]
                        if self.shift_leapday == True:
                            interconnect_Stacked_Gen = mfunc.shift_leapday(interconnect_Stacked_Gen,self.Marmot_Solutions_folder)
                        interconnect_Stacked_Gen = interconnect_Stacked_Gen.xs(interconnect,level='Interconnection')
                        
                    except KeyError:
                        continue

                    if interconnect_Stacked_Gen.empty == True:
                        continue

                    interconnect_Stacked_Gen = mfunc.df_process_gen_inputs(interconnect_Stacked_Gen, self.ordered_gen)

                try:

                    Stacked_Gen = self.mplot_data_dict["generator_Generation"].get(scenario).copy()
                    if self.shift_leapday == True:
                        Stacked_Gen = mfunc.shift_leapday(Stacked_Gen,self.Marmot_Solutions_folder)
                    Stacked_Gen = Stacked_Gen.xs(zone_input,level=self.AGG_BY)
                except KeyError:
                    # self.logger.info('No generation in %s',zone_input)
                    continue

                if Stacked_Gen.empty == True:
                    continue

                Stacked_Gen = mfunc.df_process_gen_inputs(Stacked_Gen, self.ordered_gen)

                curtailment_name = self.gen_names_dict.get('Curtailment','Curtailment')
            
                # Insert Curtailmnet into gen stack if it exhists in database

                if self.mplot_data_dict[f"generator_{self.curtailment_prop}"]:
                    Stacked_Curt = self.mplot_data_dict[f"generator_{self.curtailment_prop}"].get(scenario).copy()
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

                Stacked_Gen = Stacked_Gen.loc[:, (Stacked_Gen != 0).any(axis=0)]

                Load = self.mplot_data_dict[f"{agg}_Load"].get(scenario).copy()
                if self.shift_leapday == True:
                    Load = mfunc.shift_leapday(Load,self.Marmot_Solutions_folder)     
                Load = Load.xs(zone_input,level=self.AGG_BY)
                Load = Load.groupby(["timestamp"]).sum()
                Load = Load.squeeze() #Convert to Series

                if self.mplot_data_dict["generator_Pump_Load"] == {}:
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
                    Pump_Load = Load - Pump_Load
                else:
                    Pump_Load = Load
                
                if self.mplot_data_dict[f"{agg}_Unserved_Energy"] == {}:
                    Unserved_Energy = self.mplot_data_dict[f"{agg}_Load"][scenario].copy()
                    Unserved_Energy.iloc[:,0] = 0
                else:
                    Unserved_Energy = self.mplot_data_dict[f"{agg}_Unserved_Energy"][scenario].copy()                
                if self.shift_leapday == True:
                    Unserved_Energy = mfunc.shift_leapday(Unserved_Energy,self.Marmot_Solutions_folder)                    
                Unserved_Energy = Unserved_Energy.xs(zone_input,level=self.AGG_BY)
                Unserved_Energy = Unserved_Energy.groupby(["timestamp"]).sum()
                Unserved_Energy = Unserved_Energy.squeeze() #Convert to Series

                if prop == "Peak Demand":
                    peak_pump_load_t = Pump_Load.idxmax()
                    end_date = peak_pump_load_t + dt.timedelta(days=end)
                    start_date = peak_pump_load_t - dt.timedelta(days=start)
                    # Peak_Pump_Load = Pump_Load[peak_pump_load_t]
                    Stacked_Gen = Stacked_Gen[start_date : end_date]
                    Load = Load[start_date : end_date]
                    Unserved_Energy = Unserved_Energy[start_date : end_date]
                    Pump_Load = Pump_Load[start_date : end_date]

                elif prop == "Min Net Load":
                    min_net_load_t = Net_Load.idxmin()
                    end_date = min_net_load_t + dt.timedelta(days=end)
                    start_date = min_net_load_t - dt.timedelta(days=start)
                    # Min_Net_Load = Net_Load[min_net_load_t]
                    Stacked_Gen = Stacked_Gen[start_date : end_date]
                    Load = Load[start_date : end_date]
                    Unserved_Energy = Unserved_Energy[start_date : end_date]
                    Pump_Load = Pump_Load[start_date : end_date]

                elif prop == 'Date Range':
                	self.logger.info(f"Plotting specific date range: \
                	{str(start_date_range)} to {str(end_date_range)}")

	                Stacked_Gen = Stacked_Gen[start_date_range : end_date_range]
	                Load = Load[start_date_range : end_date_range]
	                Unserved_Energy = Unserved_Energy[start_date_range : end_date_range]

                else:
                    self.logger.info("Plotting graph for entire timeperiod")
                
                # unitconversion based off peak generation hour, only checked once 
                if i == 0:
                    unitconversion = mfunc.capacity_energy_unitconversion(max(Stacked_Gen.max()))
                Stacked_Gen = Stacked_Gen/unitconversion['divisor']
                Unserved_Energy = Unserved_Energy/unitconversion['divisor']
                if prop=='VRE_compare':
                    interconnect_Stacked_Gen = interconnect_Stacked_Gen/unitconversion['divisor']
                
                scenario_names = pd.Series([scenario]*len(Stacked_Gen),name='Scenario')
                data_table = Stacked_Gen.add_suffix(f" ({unitconversion['units']})")
                data_table = data_table.set_index([scenario_names],append=True)
                data_tables.append(data_table)

                #add the flow lines first so they are on bottom by default
                if prop=='VRE_compare':
                    for c in range(len(self.vre_gen_cat)-1):
                        axs[i+c].plot(fz.index.values,fz[fz.columns[0]]/max(fz[fz.columns[0]]),
                        linewidth=2,color='lightgrey',label='net flow')
                
                vre_counter = 0
                for column in Stacked_Gen.columns:
                    if prop=='VRE_compare':
                        if column in self.vre_gen_cat:
                            #normalize generation to allow for better comparison
                            # print(max(Stacked_Gen[column]))
                            axs[i+vre_counter].plot(Stacked_Gen.index.values,Stacked_Gen[column]/max(Stacked_Gen[column]), linewidth=2,
                            color=self.PLEXOS_color_dict.get(column,'#333333'),label=column)
                            # print(interconnect_Stacked_Gen)
                            axs[i+vre_counter].plot(interconnect_Stacked_Gen.index.values,interconnect_Stacked_Gen[column]/max(interconnect_Stacked_Gen[column]), linewidth=2,linestyle='dashed',
                            color=self.PLEXOS_color_dict.get(column,'#333333'),label=column+"_"+interconnect)

                            #let's also try to do some annotation of correlations?
                            column_1 = Stacked_Gen[column]
                            column_2 = interconnect_Stacked_Gen[column]
                            column_3 = fz[fz.columns[0]]
                            correlation1,correlation2,correlation3 = column_1.corr(column_2),column_1.corr(column_3),column_2.corr(column_3)
                            l2 = str(column+"_"+interconnect)
                            l3 = 'net flow'
                            axs[i+vre_counter].annotate(f"{column}-{l2} corr= {round(correlation1,3)}",
                                    xy=(Stacked_Gen.index.values[10], -0.5),fontsize=10)
                            axs[i+vre_counter].annotate(f"{column}-{l3} corr= {round(correlation2,3)}",
                                    xy=(Stacked_Gen.index.values[10], -0.7),fontsize=10)
                            axs[i+vre_counter].annotate(f"{l2}-{l3} corr= {round(correlation3,3)}",
                                    xy=(Stacked_Gen.index.values[10], -0.9),fontsize=10)
                            axs[i+vre_counter].set_ylim((-1.0,1.0)) #set normalized axes
                            vre_counter+=1 
                    else:
                        axs[i].plot(Stacked_Gen.index.values,Stacked_Gen[column], linewidth=2,
                        color=self.PLEXOS_color_dict.get(column,'#333333'),label=column)
                
                if (Unserved_Energy == 0).all() == False:
                    lp2 = axs[i].plot(Unserved_Energy, color='#DD0200')

                for c in range(vre_counter):
                    axs[i+c].spines['right'].set_visible(False)
                    axs[i+c].spines['top'].set_visible(False)
                    #if c==0:
                    axs[i+c].tick_params(axis='y', which='major', length=5, width=1)
                    axs[i+c].tick_params(axis='x', which='major', length=5, width=1)
                    axs[i+c].yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(x, f',.{self.y_axes_decimalpt}f')))
                    #else:
                    #    axs[i+c].spines['left'].set_visible(False)
                        # axs[i+c].set_yticks([])
                    axs[i+c].margins(x=0.01)
                    mfunc.set_plot_timeseries_format(axs,i+c)

                # create list of gen technologies
                l1 = Stacked_Gen.columns.tolist()
                unique_tech_names.extend(l1)
            
            if not data_tables:
                self.logger.warning(f'No generation in {zone_input}')
                out = mfunc.MissingZoneData()
                outputs[zone_input] = out
                continue
            
            # create handles list of unique tech names then order
            labels = np.unique(np.array(unique_tech_names)).tolist()
            labels.sort(key = lambda i:self.ordered_gen.index(i))
            
            # create custom gen_tech legend
            
            handles = []
            new_labels = []
            for tech in labels:
                if prop == 'VRE_compare':
                    if tech in self.vre_gen_cat:
                        gen_tech_legend = Line2D([0], [0], color=self.PLEXOS_color_dict[tech], linewidth=2, linestyle='solid')
                        gen_dot_legend = Line2D([0], [0], color=self.PLEXOS_color_dict[tech], linewidth=2, linestyle='dashed')
                        handles.append(gen_tech_legend)
                        handles.append(gen_dot_legend)
                        new_labels.append(tech)
                        new_labels.append(tech+"_"+interconnect)
                else:
                    gen_tech_legend = Patch(facecolor=self.PLEXOS_color_dict[tech],
                                alpha=1.0)
                    handles.append(gen_tech_legend)

            if prop=='VRE_compare':
                flow_legend = Line2D([0], [0], color='lightgrey', linewidth=2, linestyle='solid')
                handles.append(flow_legend)
                new_labels.append('net flow')
            
            if (Unserved_Energy == 0).all() == False:
                handles.append(lp2[0])
                labels += ['Unserved Energy']

            if prop == 'VRE_compare':
                axs[grid_size-1].legend(reversed(handles),reversed(new_labels),
                                        loc = 'lower left',bbox_to_anchor=(1.05,0),
                                        facecolor='inherit', frameon=True)
            else:
                axs[grid_size-1].legend(reversed(handles),reversed(labels),
                                        loc = 'lower left',bbox_to_anchor=(1.05,0),
                                        facecolor='inherit', frameon=True)
            
            xlabels = [x.replace('_',' ') for x in self.xlabels]
            ylabels = [y.replace('_',' ') for y in self.ylabels]
            
            # add facet labels
            mfunc.add_facet_labels(fig1, xlabels, ylabels)
                        
            fig1.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            if mconfig.parser("plot_title_as_region"):
                plt.title(zone_input)
            labelpad = 40
            if prop=='VRE_compare':
                plt.ylabel(f"Normalized Generation/Flow",  color='black', rotation='vertical', labelpad=labelpad)
            else:
                plt.ylabel(f"Generation ({unitconversion['units']})",  color='black', rotation='vertical', labelpad=labelpad)
            
             #Remove extra axis
            if excess_axs != 0:
                mfunc.remove_excess_axs(axs,excess_axs,grid_size)

            data_table_out = pd.concat(data_tables)
                
            outputs[zone_input] = {'fig':fig1, 'data_table':data_table_out}
        return outputs
