"""
Created on Mon Dec  9 10:34:48 2019
This code creates generation UNstacked plots and is called from Marmot_plot_main.py
@author: Daniel Levie
"""
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
from matplotlib.patches import Patch
import numpy as np
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
            check_input_data.extend([mfunc.get_data(curtailment_collection,"generator_Curtailment", self.Marmot_Solutions_folder, scenario_list)])
            mfunc.get_data(pump_load_collection,"generator_Pump_Load", self.Marmot_Solutions_folder, self.Multi_Scenario)
            
            if self.AGG_BY == "zone":
                check_input_data.extend([mfunc.get_data(load_collection,"zone_Load", self.Marmot_Solutions_folder, scenario_list)])
                mfunc.get_data(unserved_energy_collection,"zone_Unserved_Energy", self.Marmot_Solutions_folder, scenario_list)
            else:
                check_input_data.extend([mfunc.get_data(load_collection,"region_Load", self.Marmot_Solutions_folder, scenario_list)])
                mfunc.get_data(unserved_energy_collection,"region_Unserved_Energy", self.Marmot_Solutions_folder, scenario_list)
            
            return check_input_data
        
        if self.facet:
            check_input_data = getdata(self.Multi_Scenario)
            all_scenarios = self.Multi_Scenario
        else:
            check_input_data = getdata([self.Multi_Scenario[0]])  
            all_scenarios = [self.Multi_Scenario[0]]
        
        # Checks if all data required by plot is available, if 1 in list required data is missing
        if 1 in check_input_data:
            outputs = mfunc.MissingInputData()
            return outputs

        for zone_input in self.Zones:
            self.logger.info("Zone = "+ zone_input)
            
            
            xdimension=len(self.xlabels)
            if xdimension == 0:
                xdimension = 1
            ydimension=len(self.ylabels)
            if ydimension == 0:
                ydimension = 1

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
            excess_axs = grid_size - plot_number

            fig1, axs = plt.subplots(ydimension,xdimension, figsize=((6*xdimension),(4*ydimension)), sharey=True, squeeze=False)
            plt.subplots_adjust(wspace=0.05, hspace=0.25)
            axs = axs.ravel()
            i=0
            data_table = {}
            unique_tech_names = []

            for scenario in all_scenarios:
                self.logger.info("     " + scenario)
                Pump_Load = pd.Series() # Initiate pump load

                try:
                    Stacked_Gen = gen_collection.get(scenario)
                    Stacked_Gen = Stacked_Gen.xs(zone_input,level=self.AGG_BY)
                except KeyError:
                    # self.logger.info('No generation in %s',zone_input)
                    i=i+1
                    continue

                if Stacked_Gen.empty == True:
                    continue
                Stacked_Gen = mfunc.df_process_gen_inputs(Stacked_Gen, self.ordered_gen)

                try:
                    Stacked_Curt = curtailment_collection.get(scenario)
                    Stacked_Curt = Stacked_Curt.xs(zone_input,level=self.AGG_BY)
                    Stacked_Curt = mfunc.df_process_gen_inputs(Stacked_Curt, self.ordered_gen)
                    Stacked_Curt = Stacked_Curt.sum(axis=1)
                    Stacked_Curt[Stacked_Curt<0.05] = 0 #Remove values less than 0.05 MW
                    Stacked_Gen.insert(len(Stacked_Gen.columns),column='Curtailment',value=Stacked_Curt) #Insert curtailment into
                except Exception:
                    pass

                # Calculates Net Load by removing variable gen + curtailment
                self.re_gen_cat = self.re_gen_cat + ['Curtailment']
                # Adjust list of values to drop depending on if it exhists in Stacked_Gen df
                self.re_gen_cat = [name for name in self.re_gen_cat if name in Stacked_Gen.columns]
                Net_Load = Stacked_Gen.drop(labels = self.re_gen_cat, axis=1)
                Net_Load = Net_Load.sum(axis=1)

                Stacked_Gen = Stacked_Gen.loc[:, (Stacked_Gen != 0).any(axis=0)]

                Load = load_collection.get(scenario)
                Load = Load.xs(zone_input,level=self.AGG_BY)
                Load = Load.groupby(["timestamp"]).sum()
                Load = Load.squeeze() #Convert to Series
                
                try:
                    pump_load_collection[scenario]
                except KeyError:
                    pump_load_collection[scenario] = gen_collection[scenario].copy()
                    pump_load_collection[scenario].iloc[:,0] = 0
                Pump_Load = pump_load_collection.get(scenario)
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
                Unserved_Energy = unserved_energy_collection.get(scenario)
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

                else:
                    self.logger.info("Plotting graph for entire timeperiod")
                
                data_table[scenario] = Stacked_Gen

                for column in Stacked_Gen.columns:
                    axs[i].plot(Stacked_Gen.index.values,Stacked_Gen[column], linewidth=2,
                       color=self.PLEXOS_color_dict.get(column,'#333333'),label=column)

                if (Unserved_Energy == 0).all() == False:
                    lp2 = axs[i].plot(Unserved_Energy, color='#DD0200')


                axs[i].spines['right'].set_visible(False)
                axs[i].spines['top'].set_visible(False)
                axs[i].tick_params(axis='y', which='major', length=5, width=1)
                axs[i].tick_params(axis='x', which='major', length=5, width=1)
                axs[i].yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
                axs[i].margins(x=0.01)

                locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
                formatter = mdates.ConciseDateFormatter(locator)
                formatter.formats[2] = '%d\n %b'
                formatter.zero_formats[1] = '%b\n %Y'
                formatter.zero_formats[2] = '%d\n %b'
                formatter.zero_formats[3] = '%H:%M\n %d-%b'
                formatter.offset_formats[3] = '%b %Y'
                formatter.show_offset = False
                axs[i].xaxis.set_major_locator(locator)
                axs[i].xaxis.set_major_formatter(formatter)

                # create list of gen technologies
                l1 = Stacked_Gen.columns.tolist()
                unique_tech_names.extend(l1)

                i=i+1
            
            if not data_table:
                self.logger.warning('No generation in %s',zone_input)
                out = mfunc.MissingZoneData()
                outputs[zone_input] = out
                continue
            
            # create handles list of unique tech names then order
            handles = np.unique(np.array(unique_tech_names)).tolist()
            handles.sort(key = lambda i:self.ordered_gen.index(i))
            handles = reversed(handles)
            
            # create custom gen_tech legend
            gen_tech_legend = []
            for tech in handles:
                legend_handles = [Patch(facecolor=self.PLEXOS_color_dict[tech],
                            alpha=1.0,
                         label=tech)]
                gen_tech_legend.extend(legend_handles)
            
            #Legend 1
            leg1 = axs[grid_size-1].legend(handles=gen_tech_legend, loc='lower left',bbox_to_anchor=(1.05,-0.2),
                          facecolor='inherit', frameon=True)

            #Legend 3
            if (Unserved_Energy == 0).all() == False:
                leg3 = axs[grid_size-1].legend(lp2, ['Unserved Energy'], loc='upper left',bbox_to_anchor=(1.05, 1.15),
                      facecolor='inherit', frameon=True)
            
            # Manually add the first legend back
            axs[grid_size-1].add_artist(leg1)
            if (Unserved_Energy == 0).all() == False:
                axs[grid_size-1].add_artist(leg3)
            
            all_axes = fig1.get_axes()

            self.xlabels = pd.Series(self.xlabels).str.replace('_',' ').str.wrap(10, break_long_words=False)

            j=0
            k=0
            for ax in all_axes:
                if ax.is_last_row():
                    ax.set_xlabel(xlabel=(self.xlabels[j]),  color='black')
                    j=j+1
                if ax.is_first_col():
                    ax.set_ylabel(ylabel=(self.ylabels[k]),  color='black', rotation='vertical')
                    k=k+1

            fig1.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            plt.ylabel('Genertaion (MW)',  color='black', rotation='vertical', labelpad=60)
            
             #Remove extra axis
            if excess_axs != 0:
                while excess_axs > 0:
                    axs[(grid_size)-excess_axs].spines['right'].set_visible(False)
                    axs[(grid_size)-excess_axs].spines['left'].set_visible(False)
                    axs[(grid_size)-excess_axs].spines['bottom'].set_visible(False)
                    axs[(grid_size)-excess_axs].spines['top'].set_visible(False)
                    axs[(grid_size)-excess_axs].tick_params(axis='both',
                                                            which='both',
                                                            colors='white')
                    excess_axs-=1

            if not self.facet:
                data_table = data_table[self.Multi_Scenario[0]]
                
            outputs[zone_input] = {'fig':fig1, 'data_table':data_table}
        return outputs
