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

        self.mplot_data_dict = {}


    def gen_unstack(self, figure_name=None, prop=None, start=None, end=None, 
                        timezone=None, start_date_range=None, end_date_range=None):
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
                          (False,"generator_Curtailment",scenario_list),
                          (False,"generator_Pump_Load",scenario_list),
                          (True,f"{agg}_Load",scenario_list),
                          (False,f"{agg}_Unserved_Energy",scenario_list)]
            
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
                # Pump_Load = pd.Series() # Initiate pump load

                try:
                    Stacked_Gen = self.mplot_data_dict["generator_Generation"].get(scenario).copy()
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
                if self.mplot_data_dict["generator_Curtailment"]:
                    Stacked_Curt = self.mplot_data_dict["generator_Curtailment"].get(scenario).copy()
                    if self.shift_leapday:
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
                if self.shift_leapday:
                    Load = mfunc.shift_leapday(Load,self.Marmot_Solutions_folder)     
                Load = Load.xs(zone_input,level=self.AGG_BY)
                Load = Load.groupby(["timestamp"]).sum()
                Load = Load.squeeze() #Convert to Series
           
                if self.mplot_data_dict["generator_Pump_Load"] == {}:
                    Pump_Load = self.mplot_data_dict['generator_Generation'][scenario].copy()
                    Pump_Load.iloc[:,0] = 0
                else:
                    Pump_Load = self.mplot_data_dict["generator_Pump_Load"][scenario]
                if self.shift_leapday:
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
                if self.shift_leapday:
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
            
            xlabels = [textwrap.fill(x.replace('_',' '),10) for x in self.xlabels]
            ylabels = [textwrap.fill(y.replace('_',' '),10) for y in self.ylabels]
            
            # add facet labels
            mfunc.add_facet_labels(fig1, xlabels, ylabels)
                        
            fig1.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            labelpad = 50 if facet else 30
            plt.ylabel(f"Genertaion ({unitconversion['units']})",  color='black', rotation='vertical', labelpad=labelpad)
            
             #Remove extra axis
            if excess_axs != 0:
                mfunc.remove_excess_axs(axs,excess_axs,grid_size)

            if not facet:
                data_table = data_table[self.Scenarios[0]]
                
            outputs[zone_input] = {'fig':fig1, 'data_table':data_table}
        return outputs
