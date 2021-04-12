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
import numpy as np
import marmot.plottingmodules.marmot_plot_functions as mfunc
import marmot.config.mconfig as mconfig
import logging
import textwrap

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


    def gen_unstack(self):

        outputs = {}   
        gen_collection = {}
        load_collection = {}
        pump_load_collection = {}
        unserved_energy_collection = {}
        curtailment_collection = {}
        
        def getdata(scenario_list):
            
            check_input_data = []
            check_input_data.extend([mfunc.get_data(gen_collection,"generator_Generation", self.Marmot_Solutions_folder, scenario_list)])
            mfunc.get_data(curtailment_collection,"generator_Curtailment", self.Marmot_Solutions_folder, scenario_list)
            mfunc.get_data(pump_load_collection,"generator_Pump_Load", self.Marmot_Solutions_folder, self.Scenarios)
            
            if self.AGG_BY == "zone":
                check_input_data.extend([mfunc.get_data(load_collection,"zone_Load", self.Marmot_Solutions_folder, scenario_list)])
                mfunc.get_data(unserved_energy_collection,"zone_Unserved_Energy", self.Marmot_Solutions_folder, scenario_list)
            else:
                check_input_data.extend([mfunc.get_data(load_collection,"region_Load", self.Marmot_Solutions_folder, scenario_list)])
                mfunc.get_data(unserved_energy_collection,"region_Unserved_Energy", self.Marmot_Solutions_folder, scenario_list)
            
            return check_input_data
        
        if self.facet:
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
        if not self.facet:
            xdimension = 1
            ydimension = 1

        # If creating a facet plot the font is scaled by 9% for each added x dimesion fact plot
        if xdimension > 1:
            font_scaling_ratio = 1 + ((xdimension-1)*0.09)
            plt.rcParams['xtick.labelsize'] = plt.rcParams['xtick.labelsize']*font_scaling_ratio
            plt.rcParams['ytick.labelsize'] = plt.rcParams['ytick.labelsize']*font_scaling_ratio
            plt.rcParams['legend.fontsize'] = plt.rcParams['legend.fontsize']*font_scaling_ratio
            plt.rcParams['axes.labelsize'] = plt.rcParams['axes.labelsize']*font_scaling_ratio
        
        grid_size = xdimension*ydimension
            
        # Used to calculate any excess axis to delete
        plot_number = len(all_scenarios)
        
        for zone_input in self.Zones:
            self.logger.info(f"Zone = {zone_input}")
        
            excess_axs = grid_size - plot_number
        
            fig1, axs = plt.subplots(ydimension,xdimension, figsize=((self.x*xdimension),(self.y*ydimension)), sharey=True, squeeze=False)
            plt.subplots_adjust(wspace=0.05, hspace=0.25)
            axs = axs.ravel()
            data_table = {}
            unique_tech_names = []

            for i, scenario in enumerate(all_scenarios):
                self.logger.info(f"Scenario = {scenario}")
                Pump_Load = pd.Series() # Initiate pump load

                try:
                    Stacked_Gen = gen_collection.get(scenario).copy()
                    if self.shift_leapday:
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
                if curtailment_collection:
                    Stacked_Curt = curtailment_collection.get(scenario).copy()
                    if self.shift_leapday:
                        Stacked_Curt = mfunc.shift_leapday(Stacked_Curt,self.Marmot_Solutions_folder)
                    Stacked_Curt = Stacked_Curt.xs(zone_input,level=self.AGG_BY)
                    Stacked_Curt = mfunc.df_process_gen_inputs(Stacked_Curt, self.ordered_gen)
                    Stacked_Curt = Stacked_Curt.sum(axis=1)
                    Stacked_Curt[Stacked_Curt<0.05] = 0 #Remove values less than 0.05 MW
                    Stacked_Gen.insert(len(Stacked_Gen.columns),column=curtailment_name,value=Stacked_Curt) #Insert curtailment into

                    # Calculates Net Load by removing variable gen + curtailment
                    self.re_gen_cat = self.re_gen_cat + [curtailment_name]
                    
                # Adjust list of values to drop depending on if it exhists in Stacked_Gen df
                self.re_gen_cat = [name for name in self.re_gen_cat if name in Stacked_Gen.columns]
                Net_Load = Stacked_Gen.drop(labels = self.re_gen_cat, axis=1)
                Net_Load = Net_Load.sum(axis=1)

                Stacked_Gen = Stacked_Gen.loc[:, (Stacked_Gen != 0).any(axis=0)]

                Load = load_collection.get(scenario).copy()
                if self.shift_leapday:
                    Load = mfunc.shift_leapday(Load,self.Marmot_Solutions_folder)     
                Load = Load.xs(zone_input,level=self.AGG_BY)
                Load = Load.groupby(["timestamp"]).sum()
                Load = Load.squeeze() #Convert to Series
           
                try:
                    pump_load_collection[scenario]
                except KeyError:
                    pump_load_collection[scenario] = gen_collection[scenario].copy()
                    pump_load_collection[scenario].iloc[:,0] = 0

                Pump_Load = pump_load_collection.get(scenario).copy()
                if self.shift_leapday:
                    Pump_Load = mfunc.shift_leapday(Pump_Load,self.Marmot_Solutions_folder)                                
                Pump_Load = Pump_Load.xs(zone_input,level=self.AGG_BY)
                Pump_Load = Pump_Load.groupby(["timestamp"]).sum()
                Pump_Load = Pump_Load.squeeze() #Convert to Series
                if (Pump_Load == 0).all() == False:
                    Pump_Load = Load - Pump_Load
                else:
                    Pump_Load = Load
                try:
                    unserved_energy_collection[scenario]
                except KeyError:
                    unserved_energy_collection[scenario] = load_collection[scenario].copy()
                    unserved_energy_collection[scenario].iloc[:,0] = 0
                Unserved_Energy = unserved_energy_collection.get(scenario).copy()
                if self.shift_leapday:
                    Unserved_Energy = mfunc.shift_leapday(Unserved_Energy,self.Marmot_Solutions_folder)                    
                Unserved_Energy = Unserved_Energy.xs(zone_input,level=self.AGG_BY)
                Unserved_Energy = Unserved_Energy.groupby(["timestamp"]).sum()
                Unserved_Energy = Unserved_Energy.squeeze() #Convert to Series

                if self.prop == "Peak Demand":
                    peak_pump_load_t = Pump_Load.idxmax()
                    end_date = peak_pump_load_t + dt.timedelta(days=self.end)
                    start_date = peak_pump_load_t - dt.timedelta(days=self.start)
                    # Peak_Pump_Load = Pump_Load[peak_pump_load_t]
                    Stacked_Gen = Stacked_Gen[start_date : end_date]
                    Load = Load[start_date : end_date]
                    Unserved_Energy = Unserved_Energy[start_date : end_date]
                    Pump_Load = Pump_Load[start_date : end_date]

                elif self.prop == "Min Net Load":
                    min_net_load_t = Net_Load.idxmin()
                    end_date = min_net_load_t + dt.timedelta(days=self.end)
                    start_date = min_net_load_t - dt.timedelta(days=self.start)
                    # Min_Net_Load = Net_Load[min_net_load_t]
                    Stacked_Gen = Stacked_Gen[start_date : end_date]
                    Load = Load[start_date : end_date]
                    Unserved_Energy = Unserved_Energy[start_date : end_date]
                    Pump_Load = Pump_Load[start_date : end_date]

                elif self.prop == 'Date Range':
                	self.logger.info(f"Plotting specific date range: \
                	{str(self.start_date)} to {str(self.end_date)}")

	                Stacked_Gen = Stacked_Gen[self.start_date : self.end_date]
	                Load = Load[self.start_date : self.end_date]
	                Unserved_Energy = Unserved_Energy[self.start_date : self.end_date]

                else:
                    self.logger.info("Plotting graph for entire timeperiod")
                
                # unitconversion based off peak generation hour, only checked once 
                if i == 0:
                    unitconversion = mfunc.capacity_energy_unitconversion(max(Stacked_Gen.max()))
                Stacked_Gen = Stacked_Gen/unitconversion['divisor']
                Unserved_Energy = Unserved_Energy/unitconversion['divisor']
                
                Data_Table_Out = Stacked_Gen
                data_table[scenario] = Data_Table_Out.add_suffix(f" ({unitconversion['units']})")
                
                for column in Stacked_Gen.columns:
                    axs[i].plot(Stacked_Gen.index.values,Stacked_Gen[column], linewidth=2,
                       color=self.PLEXOS_color_dict.get(column,'#333333'),label=column)

                if (Unserved_Energy == 0).all() == False:
                    lp2 = axs[i].plot(Unserved_Energy, color='#DD0200')

                axs[i].spines['right'].set_visible(False)
                axs[i].spines['top'].set_visible(False)
                axs[i].tick_params(axis='y', which='major', length=5, width=1)
                axs[i].tick_params(axis='x', which='major', length=5, width=1)
                axs[i].yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(x, f',.{self.y_axes_decimalpt}f')))
                axs[i].margins(x=0.01)
                mfunc.set_plot_timeseries_format(axs,i)

                # create list of gen technologies
                l1 = Stacked_Gen.columns.tolist()
                unique_tech_names.extend(l1)
            
            if not data_table:
                self.logger.warning(f'No generation in {zone_input}')
                out = mfunc.MissingZoneData()
                outputs[zone_input] = out
                continue
            
            # create handles list of unique tech names then order
            labels = np.unique(np.array(unique_tech_names)).tolist()
            labels.sort(key = lambda i:self.ordered_gen.index(i))
            
            # create custom gen_tech legend
            handles = []
            for tech in labels:
                gen_tech_legend = Patch(facecolor=self.PLEXOS_color_dict[tech],
                            alpha=1.0)
                handles.append(gen_tech_legend)
            
            if (Unserved_Energy == 0).all() == False:
                handles.append(lp2[0])
                labels += ['Unserved Energy']
                

            axs[grid_size-1].legend(reversed(handles),reversed(labels),
                                    loc = 'lower left',bbox_to_anchor=(1.05,0),
                                    facecolor='inherit', frameon=True)
            
            self.xlabels = [textwrap.fill(x.replace('_',' '),10) for x in self.xlabels]
            self.ylabels = [textwrap.fill(y.replace('_',' '),10) for y in self.ylabels]
            
            # add facet labels
            mfunc.add_facet_labels(fig1, self.xlabels, self.ylabels)
                        
            fig1.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            labelpad = 60 if self.facet else 25
            plt.ylabel(f"Genertaion ({unitconversion['units']})",  color='black', rotation='vertical', labelpad=labelpad)
            
             #Remove extra axis
            if excess_axs != 0:
                mfunc.remove_excess_axs(axs,excess_axs,grid_size)

            if not self.facet:
                data_table = data_table[self.Scenarios[0]]
                
            outputs[zone_input] = {'fig':fig1, 'data_table':data_table}
        return outputs
