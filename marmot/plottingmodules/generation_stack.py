"""Timeseries generation stacked area plots.

This code creates generation stack plots.

@author: Daniel Levie
"""

import logging
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

import marmot.config.mconfig as mconfig
from marmot.plottingmodules.plotutils.plot_library import SetupSubplot, PlotLibrary
from marmot.plottingmodules.plotutils.plot_data_helper import PlotDataHelper
from marmot.plottingmodules.plotutils.plot_exceptions import (MissingInputData, 
            UnderDevelopment, InputSheetError, MissingZoneData)


class MPlot(PlotDataHelper):
    """generation_stack MPlot class.

    All the plotting modules use this same class name.
    This class contains plotting methods that are grouped based on the
    current module name.
    
    The generation_stack.py contains methods that are
    related to the timeseries generation of generators, in a stacked area format.  
    
    MPlot inherits from the PlotDataHelper class to assist in creating figures.
    """

    def __init__(self, argument_dict: dict):
        """
        Args:
            argument_dict (dict): Dictionary containing all
                arguments passed from MarmotPlot.
        """
        # iterate over items in argument_dict and set as properties of class
        # see key_list in Marmot_plot_main for list of properties
        for prop in argument_dict:
            self.__setattr__(prop, argument_dict[prop])

        # Instantiation of MPlotHelperFunctions
        super().__init__(self.Marmot_Solutions_folder, self.AGG_BY, self.ordered_gen, 
                    self.PLEXOS_color_dict, self.Scenarios, self.ylabels, 
                    self.xlabels, self.gen_names_dict, self.TECH_SUBSET, 
                    Region_Mapping=self.Region_Mapping) 

        self.logger = logging.getLogger('marmot_plot.'+__name__)

        self.curtailment_prop = mconfig.parser("plot_data","curtailment_property")

        
    def committed_stack(self, start_date_range: str = None, 
                        end_date_range: str = None, **_):
        """Plots the timeseries of committed generation compared to the total available capacity 
        
        The upper line shows the total available cpacity that can be committed 
        The area between the lower line and the x-axis plots the total capacity that is 
        committed and producing energy.
        
        Any gap that exists between the upper and lower line is generation that is 
        not committed but available to use.  

        Data is plotted in a facet plot, each row of the facet plot represents 
        separate generation technologies.
        Each bar the facet plot represents separate scenarios.

        Args:
            start_date_range (str, optional): Defines a start date at which to represent data from. 
                Defaults to None.
            end_date_range (str, optional): Defines a end date at which to represent data to.
                Defaults to None.

        Returns:
            dict: dictionary containing the created plot and its data table.
        """
        outputs = {}

        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True,"generator_Installed_Capacity",[self.Scenarios[0]]),
                      (True,"generator_Generation",self.Scenarios),
                      (True,"generator_Units_Generating",self.Scenarios),
                      (True,"generator_Available_Capacity",self.Scenarios)]

        # Runs get_formatted_data within PlotDataHelper to populate PlotDataHelper dictionary  
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)

        # Checks if all data required by plot is available, if 1 in list required data is missing
        if 1 in check_input_data:
            return MissingInputData()

        for zone_input in self.Zones:
            self.logger.info(f'Zone = {str(zone_input)}')

            #Get technology list.
            gens = self['generator_Installed_Capacity'].get(self.Scenarios[0])
            try:
                gens = gens.xs(zone_input,level=self.AGG_BY)
            except KeyError:
                self.logger.warning(f"No Generation in: {zone_input}")
                outputs[zone_input] = MissingZoneData()
                continue
            
            gens = self.df_process_gen_inputs(gens)
            tech_list = list(gens.columns)
            tech_list_sort = [tech_type for tech_type in self.ordered_gen 
                              if tech_type in tech_list and tech_type in self.thermal_gen_cat]

            if not tech_list_sort:
                self.logger.info(f'No Thermal Generation in: {zone_input}')
                outputs[zone_input] = MissingZoneData()
                continue

            ncols = len(self.Scenarios)
            nrows = len(tech_list_sort)
            
            mplt = SetupSubplot(nrows, ncols, sharex=True,
                                sharey='row', squeeze=False)
            fig, axs = mplt.get_figure()
            plt.subplots_adjust(wspace=0.1, hspace=0.2)

            for i, scenario in enumerate(self.Scenarios):
                self.logger.info(f"Scenario = {scenario}")

                units_gen = self['generator_Units_Generating'].get(scenario).copy()
                avail_cap = self['generator_Available_Capacity'].get(scenario)

                # Drop units index to allow multiplication  
                units_gen.reset_index(level='units', drop=True, inplace=True)
        
                #Calculate  committed cap (for thermal only).
                thermal_commit_cap = units_gen * avail_cap
                thermal_commit_cap = thermal_commit_cap.xs(zone_input, level=self.AGG_BY)
                thermal_commit_cap = self.df_process_gen_inputs(thermal_commit_cap)
                # Drop all zero columns
                thermal_commit_cap = thermal_commit_cap.loc[:, (thermal_commit_cap != 0).any(axis=0)]

                # unitconversion based off peak generation hour, only checked once
                if i == 0:
                    unitconversion = self.capacity_energy_unitconversion(thermal_commit_cap)
                thermal_commit_cap = thermal_commit_cap/unitconversion['divisor']

                #Process generation.
                gen = self['generator_Generation'].get(scenario)
                gen = gen.xs(zone_input,level = self.AGG_BY)
                gen = self.df_process_gen_inputs(gen)
                gen = gen.loc[:, (gen != 0).any(axis=0)]
                gen = gen/unitconversion['divisor']

                #Process available capacity (for VG only).
                avail_cap = avail_cap.xs(zone_input, level=self.AGG_BY)
                avail_cap = self.df_process_gen_inputs(avail_cap)
                avail_cap = avail_cap.loc[:, (avail_cap !=0).any(axis=0)]
                avail_cap = avail_cap/unitconversion['divisor']

                for j,tech in enumerate(tech_list_sort):
                    if tech not in gen.columns:
                        gen_one_tech = pd.Series(0,index = gen.index)
                        # Add dummy columns to deal with coal retirements 
                        # (coal showing up in 2024, but not future years).
                        commit_cap = pd.Series(0,index = gen.index) 
                    elif tech in self.thermal_gen_cat:
                        gen_one_tech = gen[tech]
                        commit_cap = thermal_commit_cap[tech]
                    else:
                        gen_one_tech = gen[tech]
                        commit_cap = avail_cap[tech]

                    axs[j,i].plot(gen_one_tech, alpha=0, 
                                    color=self.PLEXOS_color_dict[tech])[0]
                    axs[j,i].fill_between(gen_one_tech.index, gen_one_tech, 0, 
                                            color=self.PLEXOS_color_dict[tech], alpha=0.5)
                    if tech != 'Hydro':
                        axs[j,i].plot(commit_cap, color=self.PLEXOS_color_dict[tech])

                    mplt.set_yaxis_major_tick_format(sub_pos=(j,i))
                    axs[j,i].margins(x=0.01)
                    mplt.set_subplot_timeseries_format(sub_pos=(j,i))

            mplt.add_facet_labels(xlabels_bottom=False, xlabels=self.Scenarios,
                                    ylabels=tech_list_sort)
            
            if mconfig.parser("plot_title_as_region"):
                mplt.add_main_title(zone_input)
            plt.ylabel(f"Generation or Committed Capacity ({unitconversion['units']})", 
                       color='black', rotation='vertical', labelpad=60)

            data_table = pd.DataFrame() #TODO: write actual data out
            outputs[zone_input] = {'fig':fig, 'data_table':data_table}
        return outputs


    def gen_stack(self, figure_name: str = None, prop: str = None,
                  start: float = None, end: float= None,
                  timezone: str = "", start_date_range: str = None,
                  end_date_range: str = None, **_):
        """Creates a timeseries stacked area plot of generation by technology.

        The stack order of technologies is determined by the ordered_gen_categories.csv
        
        If multiple scenarios are passed they will be plotted in a facet plot.
        The plot can be further customized by passing specific values to the
        prop argument.

        Args:
            figure_name (str, optional): User defined figure output name.
                Defaults to None.
            prop (str, optional): Special argument used to adjust specific 
                plot settings. Controlled through the plot_select.csv.
                Opinions available are:

                - Peak Demand
                - Min Net Load
                - Date Range
                - Peak RE
                - Peak Unserved Energy
                - Peak Curtailment
                
                Defaults to None.
            start (float, optional): Used in conjunction with the prop argument.
                Will define the number of days to plot before a certain event in 
                a timeseries plot, e.g Peak Demand.
                Defaults to None.
            end (float, optional): Used in conjunction with the prop argument.
                Will define the number of days to plot after a certain event in 
                a timeseries plot, e.g Peak Demand.
                Defaults to None.
            timezone (str, optional): The timezone to display on the x-axes.
                Defaults to "".
            start_date_range (str, optional): Defines a start date at which to represent data from. 
                Defaults to None.
            end_date_range (str, optional): Defines a end date at which to represent data to.
                Defaults to None.

        Returns:
            dict: dictionary containing the created plot and its data table.

        """
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
                          (False,f"generator_{self.curtailment_prop}",scenario_list),
                          (False,"generator_Pump_Load",scenario_list),
                          (True,f"{agg}_Load",scenario_list),
                          (False,f"{agg}_Unserved_Energy",scenario_list)]

            # Runs get_formatted_data within PlotDataHelper to populate PlotDataHelper dictionary  
        # with all required properties, returns a 1 if required data is missing
            return self.get_formatted_data(properties)

        def setup_data(zone_input, scenario, Stacked_Gen):

            # Insert Curtailment into gen stack if it exists in database
            Stacked_Curt = self[f"generator_{self.curtailment_prop}"].get(scenario).copy()
            if not Stacked_Curt.empty:
                curtailment_name = self.gen_names_dict.get('Curtailment','Curtailment')
                if self.shift_leapday == True:
                    Stacked_Curt = self.adjust_for_leapday(Stacked_Curt)
                if zone_input in Stacked_Curt.index.get_level_values(self.AGG_BY).unique():
                    Stacked_Curt = Stacked_Curt.xs(zone_input,level=self.AGG_BY)
                    Stacked_Curt = self.df_process_gen_inputs(Stacked_Curt)
                    # If using Marmot's curtailment property
                    if self.curtailment_prop == 'Curtailment':
                        Stacked_Curt = self.assign_curtailment_techs(Stacked_Curt)
                    Stacked_Curt = Stacked_Curt.sum(axis=1)
                    Stacked_Curt[Stacked_Curt<0.05] = 0 #Remove values less than 0.05 MW
                    Stacked_Gen.insert(len(Stacked_Gen.columns), 
                                       column=curtailment_name, value=Stacked_Curt) #Insert curtailment into
                    # Calculates Net Load by removing variable gen + curtailment
                    vre_gen_cat = self.vre_gen_cat + [curtailment_name]
                else:
                    vre_gen_cat = self.vre_gen_cat
            else:
                vre_gen_cat = self.vre_gen_cat
            # Adjust list of values to drop depending on if it exists in Stacked_Gen df
            vre_gen_cat = [name for name in vre_gen_cat if name in Stacked_Gen.columns]
            Net_Load = Stacked_Gen.drop(labels = vre_gen_cat, axis=1)
            Net_Load = Net_Load.sum(axis=1)

            # Removes columns that only contain 0
            Stacked_Gen = Stacked_Gen.loc[:, (Stacked_Gen != 0).any(axis=0)]

            Load = self[f'{agg}_Load'].get(scenario).copy()
            if self.shift_leapday == True:
                Load = self.adjust_for_leapday(Load)
            Load = Load.xs(zone_input,level=self.AGG_BY)
            Load = Load.groupby(["timestamp"]).sum()
            Load = Load.squeeze() #Convert to Series

            Pump_Load = self["generator_Pump_Load"][scenario]
            if Pump_Load.empty or not mconfig.parser("plot_data","include_timeseries_pumped_load_line"):
                Pump_Load = self['generator_Generation'][scenario].copy()
                Pump_Load.iloc[:,0] = 0
            if self.shift_leapday == True:
                Pump_Load = self.adjust_for_leapday(Pump_Load)
            Pump_Load = Pump_Load.xs(zone_input,level=self.AGG_BY)
            Pump_Load = Pump_Load.groupby(["timestamp"]).sum()
            Pump_Load = Pump_Load.squeeze() #Convert to Series
            if (Pump_Load == 0).all() == False:
                Total_Demand = Load - Pump_Load
            else:
                Total_Demand = Load

            Unserved_Energy = self[f'{agg}_Unserved_Energy'][scenario].copy()
            if Unserved_Energy.empty:
                Unserved_Energy = self[f'{agg}_Load'][scenario].copy()
                Unserved_Energy.iloc[:,0] = 0
            if self.shift_leapday == True:
                Unserved_Energy = self.adjust_for_leapday(Unserved_Energy)
            Unserved_Energy = Unserved_Energy.xs(zone_input,level=self.AGG_BY)
            Unserved_Energy = Unserved_Energy.groupby(["timestamp"]).sum()
            Unserved_Energy = Unserved_Energy.squeeze() #Convert to Series

            unserved_eng_data_table = Unserved_Energy # Used for output to data table csv
            if (Unserved_Energy == 0).all() == False:
                Unserved_Energy = Load - Unserved_Energy
            
            data = {"Stacked_Gen": Stacked_Gen,
                    "Load": Load,
                    "Net_Load": Net_Load,
                    "Pump_Load": Pump_Load,
                    "Total_Demand": Total_Demand,
                    "Unserved_Energy": Unserved_Energy,
                    "ue_data_table": unserved_eng_data_table}
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
            peak_re_t = None
            peak_re = 0
            gen_peak_re = 0
            peak_ue = 0
            peak_ue_t = None
            peak_curt = 0
            peak_curt_t = None
            gen_peak_curt = 0

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
            
            elif prop == 'Peak RE':
                re_gen_cat = [name for name in self.re_gen_cat if name in Stacked_Gen.columns]
                all_gen = [name for name in Stacked_Gen.columns]
                if len(re_gen_cat) == 0:
                    return None
                else:
                    re_total = Stacked_Gen[re_gen_cat[0]]
                    i = 1
                    while i < len(re_gen_cat):
                        re_total = re_total + Stacked_Gen[re_gen_cat[i]]
                        i += 1
                gen_total = Stacked_Gen[all_gen[0]]
                j = 1
                while j < len(all_gen):
                    gen_total = gen_total + Stacked_Gen[all_gen[j]]
                    j += 1
                peak_re_t = re_total.idxmax()
                peak_re = re_total[peak_re_t]
                gen_peak_re = gen_total[peak_re_t]
                end_date = peak_re_t + dt.timedelta(days=end)
                start_date = peak_re_t - dt.timedelta(days=start)
                Min_Net_Load = Net_Load[peak_re_t]
                Stacked_Gen = Stacked_Gen[start_date : end_date]
                Load = Load[start_date : end_date]
                Unserved_Energy = Unserved_Energy[start_date : end_date]
                Total_Demand = Total_Demand[start_date : end_date]

                unserved_eng_data_table = unserved_eng_data_table[start_date : end_date]
            
            elif prop == 'Peak Unserved Energy':
                peak_ue_t = unserved_eng_data_table.idxmax()
                peak_ue = unserved_eng_data_table[peak_ue_t]
                end_date = peak_ue_t + dt.timedelta(days=end)
                start_date = peak_ue_t - dt.timedelta(days=start)
                Min_Net_Load = Net_Load[peak_ue_t]
                Stacked_Gen = Stacked_Gen[start_date : end_date]
                Load = Load[start_date : end_date]
                Unserved_Energy = Unserved_Energy[start_date : end_date]
                Total_Demand = Total_Demand[start_date : end_date]

                unserved_eng_data_table = unserved_eng_data_table[start_date : end_date]
                
            elif prop == 'Peak Curtailment':
                all_gen = [name for name in Stacked_Gen.columns]
                gen_total = Stacked_Gen[all_gen[0]]
                j = 1
                while j < len(all_gen):
                    gen_total = gen_total + Stacked_Gen[all_gen[j]]
                    j += 1

                curtailment_name = self.gen_names_dict.get('Curtailment','Curtailment')
                if curtailment_name not in Stacked_Gen:
                    return None
                curtailment = Stacked_Gen['Curtailment']
                peak_curt_t = curtailment.idxmax()
                peak_curt = curtailment[peak_curt_t]
                gen_peak_curt = gen_total[peak_curt_t]
                end_date = peak_curt_t + dt.timedelta(days=end)
                start_date = peak_curt_t - dt.timedelta(days=start)
                Min_Net_Load = Net_Load[peak_curt_t]
                Stacked_Gen = Stacked_Gen[start_date : end_date]
                Load = Load[start_date : end_date]
                Unserved_Energy = Unserved_Energy[start_date : end_date]
                Total_Demand = Total_Demand[start_date : end_date]
                unserved_eng_data_table = unserved_eng_data_table[start_date : end_date]
            else:
                self.logger.info("Plotting graph for entire timeperiod")

            data = {"Stacked_Gen": Stacked_Gen, 
                    "Load": Load, 
                    "Pump_Load": Pump_Load, 
                    "Total_Demand": Total_Demand, 
                    "Unserved_Energy": Unserved_Energy,
                    "ue_data_table": unserved_eng_data_table}

            data["peak_demand_t"] = peak_demand_t
            data["Peak_Demand"] = Peak_Demand
            data["min_net_load_t"] = min_net_load_t
            data["Min_Net_Load"] = Min_Net_Load
            data["peak_re_t"] = peak_re_t
            data["Peak_RE"] = peak_re
            data["Gen_peak_re"] = gen_peak_re
            data["Peak_Unserved_Energy"] = peak_ue
            data["peak_ue_t"] = peak_ue_t
            data["peak_curt"] = peak_curt
            data["peak_curt_t"] = peak_curt_t
            data["gen_peak_curt"] = gen_peak_curt
            return data

        def mkplot(outputs, zone_input, all_scenarios):

            # sets up x, y dimensions of plot
            ncols, nrows = self.set_facet_col_row_dimensions(multi_scenario=all_scenarios)

            # If the plot is not a facet plot, grid size should be 1x1
            if not facet:
                ncols = 1
                nrows = 1

            grid_size = ncols*nrows

            # Used to calculate any excess axis to delete
            plot_number = len(all_scenarios)
            excess_axs = grid_size - plot_number

            mplt = PlotLibrary(nrows, ncols, sharey=True, 
                                squeeze=False, ravel_axs=True)
            fig, axs = mplt.get_figure()

            plt.subplots_adjust(wspace=0.05, hspace=0.5)
            
            # If creating a facet plot the font is scaled by 9% for each added x dimesion fact plot
            if ncols > 1:
                font_scaling_ratio = 1 + ((ncols-1)*0.09)
                plt.rcParams['xtick.labelsize'] *= font_scaling_ratio
                plt.rcParams['ytick.labelsize'] *= font_scaling_ratio
                plt.rcParams['legend.fontsize'] *= font_scaling_ratio
                plt.rcParams['axes.labelsize'] *= font_scaling_ratio
                plt.rcParams['axes.titlesize'] *= font_scaling_ratio
                
            data_tables = []

            for i, scenario in enumerate(all_scenarios):
                self.logger.info(f"Scenario = {scenario}")

                try:
                    Stacked_Gen = self['generator_Generation'].get(scenario).copy()
                    if self.shift_leapday == True:
                        Stacked_Gen = self.adjust_for_leapday(Stacked_Gen)
                    Stacked_Gen = Stacked_Gen.xs(zone_input,level=self.AGG_BY)
                except KeyError:
                    self.logger.warning(f'No generation in {zone_input}')
                    out = MissingZoneData()
                    return out
                Stacked_Gen = self.df_process_gen_inputs(Stacked_Gen)
                data = setup_data(zone_input, scenario, Stacked_Gen)
                data = data_prop(data)
                
                # if no Generation return empty dataframe
                if not data or data["Stacked_Gen"].empty == True:
                    self.logger.warning(f'No generation during time period in {zone_input}')
                    out = MissingZoneData()
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
                Peak_RE = data["Peak_RE"]
                peak_re_t = data["peak_re_t"]
                gen_peak_re2 = data["Gen_peak_re"]
                peak_ue = data["Peak_Unserved_Energy"]
                peak_ue_t = data["peak_ue_t"]
                peak_curt = data["peak_curt"]
                peak_curt_t = data["peak_curt_t"]
                gen_peak_curt = data["gen_peak_curt"]

                # unitconversion based off peak generation hour, only checked once
                if i == 0:
                    unitconversion = self.capacity_energy_unitconversion(Stacked_Gen, 
                                                                            sum_values=True)

                #Convert units
                Stacked_Gen = Stacked_Gen / unitconversion['divisor']
                Load = Load / unitconversion['divisor']
                Pump_Load = Pump_Load / unitconversion['divisor']
                Total_Demand = Total_Demand / unitconversion['divisor']
                Unserved_Energy = Unserved_Energy / unitconversion['divisor']
                unserved_eng_data_table = unserved_eng_data_table / unitconversion['divisor']
                Peak_Demand = Peak_Demand / unitconversion['divisor']
                Peak_RE = Peak_RE / unitconversion['divisor']
                gen_peak_re2 = gen_peak_re2/ unitconversion['divisor']
                gen_peak_curt = gen_peak_curt / unitconversion['divisor']
                peak_ue = peak_ue/ unitconversion['divisor']
                peak_curt = peak_curt/ unitconversion['divisor']
                Min_Net_Load = Min_Net_Load / unitconversion['divisor']

                Load = Load.rename('Total Load \n (Demand + Storage Charging)')
                Total_Demand = Total_Demand.rename('Total Demand')
                unserved_eng_data_table = unserved_eng_data_table.rename("Unserved Energy")
                
                # Add Net Imports if desired
                if mconfig.parser("plot_data","include_timeseries_net_imports"):
                    Stacked_Gen = self.include_net_imports(Stacked_Gen, Load,
                                                           Unserved_Energy)
                    
                # Data table of values to return to main program
                single_scen_out = pd.concat([Load, Total_Demand, unserved_eng_data_table, Stacked_Gen], 
                                            axis=1, sort=False)
                scenario_names = pd.Series([scenario] * len(single_scen_out),name = 'Scenario')
                single_scen_out = single_scen_out.add_suffix(f" ({unitconversion['units']})")
                single_scen_out = single_scen_out.set_index([scenario_names],append = True)
                data_tables.append(single_scen_out)

                mplt.stackplot(Stacked_Gen, self.PLEXOS_color_dict, 
                                      labels=Stacked_Gen.columns, sub_pos=i)
                mplt.set_subplot_timeseries_format(sub_pos=i)

                if (Unserved_Energy == 0).all() == False:
                    axs[i].plot(Unserved_Energy,
                                      #color='#EE1289'  OLD MARMOT COLOR
                                      color = '#DD0200', #SEAC STANDARD COLOR (AS OF MARCH 9, 2020)
                                      label='Unserved Energy')
                    axs[i].fill_between(Load.index, Load,Unserved_Energy,
                                        # facecolor='#EE1289' OLD MARMOT COLOR
                                        facecolor = '#DD0200', #SEAC STANDARD COLOR (AS OF MARCH 9, 2020)
                                        alpha=0.5)

                if (Pump_Load == 0).all() == False:
                    axs[i].plot(Load, color='black', 
                                     label='Demand + \n Storage Charging')
                    axs[i].plot(Total_Demand, color='black', linestyle="--",
                                        label='Demand')
                else:
                    axs[i].plot(Load, color='black', label='Demand')                    

                if prop == "Min Net Load":
                    axs[i].annotate(f"Min Net Load: \n{str(format(Min_Net_Load, '.2f'))} {unitconversion['units']}",
                                    xy=(min_net_load_t, Min_Net_Load), 
                                    xytext=((min_net_load_t + dt.timedelta(days=0.1)), (max(Load))),
                                    fontsize=13, arrowprops=dict(facecolor='black', width=3, shrink=0.1))

                # Peak Demand label overlaps other labels on a facet plot
                elif prop == "Peak Demand":
                    axs[i].annotate(f"Peak Demand: \n{str(format(Total_Demand[peak_demand_t], '.2f'))} {unitconversion['units']}",
                                    xy=(peak_demand_t, Peak_Demand), 
                                    xytext=((peak_demand_t + dt.timedelta(days=0.1)),
                                            (max(Total_Demand) + Total_Demand[peak_demand_t]*0.1)),
                                    fontsize=13, arrowprops=dict(facecolor='black', width=3, shrink=0.1))
                
                if prop == "Peak RE":
                    axs[i].annotate(f"Peak RE: \n{str(format(Peak_RE, '.2f'))} {unitconversion['units']}",
                                    xy=(peak_re_t, gen_peak_re2), 
                                    xytext=((peak_re_t + dt.timedelta(days=0.5)),
                                            (max(Total_Demand))),
                                    fontsize=13, arrowprops=dict(facecolor='black', width=3, shrink=0.1))
                
                if prop == "Peak Unserved Energy":
                    axs[i].annotate(f"Peak Unserved Energy: \n{str(format(peak_ue, '.2f'))} {unitconversion['units']}",
                                    xy=(peak_ue_t, Total_Demand[peak_ue_t]), 
                                    xytext=((peak_ue_t + dt.timedelta(days=0.5)),
                                            (max(Total_Demand))),
                                    fontsize=13, arrowprops=dict(facecolor='black', width=3, shrink=0.1))
                
                if prop == "Peak Curtailment":
                    axs[i].annotate(f"Peak Curtailment: \n{str(format(peak_curt, '.2f'))} {unitconversion['units']}",
                                    xy=(peak_curt_t, gen_peak_curt), 
                                    xytext=((peak_curt_t + dt.timedelta(days=0.5)),
                                            (max(Total_Demand))),
                                    fontsize=13, arrowprops=dict(facecolor='black', width=3, shrink=0.1))

            # Add facet labels
            mplt.add_facet_labels(xlabels=self.xlabels,
                                  ylabels = self.ylabels)
            # Add legend
            mplt.add_legend(reverse_legend=True, sort_by=self.ordered_gen)
            # Remove extra axes
            mplt.remove_excess_axs(excess_axs, grid_size)
            # Add title
            if mconfig.parser('plot_title_as_region'):
                mplt.add_main_title(zone_input)

            #Ylabel should change if there are facet labels, leave at 40 for now, 
            # works for all values in spacing
            labelpad = 40
            plt.ylabel(f"Generation ({unitconversion['units']})", color='black', 
                       rotation='vertical', labelpad=labelpad)

            Data_Table_Out = pd.concat(data_tables)
            out = {'fig':fig, 'data_table':Data_Table_Out}
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
            outputs = MissingInputData()
            return outputs

        ncols=len(self.xlabels)
        if ncols == 0:
                ncols = 1

        # If the plot is not a facet plot, grid size should be 1x1
        if not facet:
            ncols = 1

        for zone_input in self.Zones:
            self.logger.info(f"Zone = {zone_input}")

            if facet:
                outputs[zone_input] = mkplot(outputs, zone_input, self.Scenarios)
            else:
                outputs[zone_input] = mkplot(outputs, zone_input, [self.Scenarios[0]])
        return outputs


    def gen_diff(self, timezone: str = "", start_date_range: str = None,
                 end_date_range: str = None, **_):
        """Plots the difference in generation between two scenarios.

        A line plot is created for each technology representing the difference 
        between the scenarios.

        Args:
            timezone (str, optional): The timezone to display on the x-axes.
                Defaults to "".
            start_date_range (str, optional): Defines a start date at which to represent data from. 
                Defaults to None.
            end_date_range (str, optional): Defines a end date at which to represent data to.
                Defaults to None.

        Returns:
            dict: dictionary containing the created plot and its data table.
        """
        outputs = {}

        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True,"generator_Generation",self.Scenarios)]

        # Runs get_formatted_data within PlotDataHelper to populate PlotDataHelper dictionary  
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)

        if 1 in check_input_data:
            outputs = MissingInputData()
            return outputs

        for zone_input in self.Zones:
            self.logger.info(f"Zone = {zone_input}")
            # Create Dictionary to hold Datframes for each scenario

            Total_Gen_Stack_1 = self['generator_Generation'].get(self.Scenario_Diff[0])
            if Total_Gen_Stack_1 is None:
                self.logger.warning(f"Scenario_Diff '{self.Scenario_Diff[0]}'' is not in data. "
                                    "Ensure User Input Sheet is set up correctly!")
                outputs = InputSheetError()
                return outputs
            
            if zone_input not in Total_Gen_Stack_1.index.get_level_values(self.AGG_BY).unique():
                outputs[zone_input] = MissingZoneData()
                continue
                
            Total_Gen_Stack_1 = Total_Gen_Stack_1.xs(zone_input,level=self.AGG_BY)
            Total_Gen_Stack_1 = self.df_process_gen_inputs(Total_Gen_Stack_1)
            #Adds in all possible columns from ordered gen to ensure the two dataframes have same column names
            Total_Gen_Stack_1 = pd.DataFrame(Total_Gen_Stack_1, columns=self.ordered_gen).fillna(0)

            Total_Gen_Stack_2 = self['generator_Generation'].get(self.Scenario_Diff[1])
            if Total_Gen_Stack_2 is None:
                self.logger.warning(f"Scenario_Diff '{self.Scenario_Diff[1]}' is not in data. "
                                    "Ensure User Input Sheet is set up correctly!")
                outputs = InputSheetError()
                return outputs

            Total_Gen_Stack_2 = Total_Gen_Stack_2.xs(zone_input,level=self.AGG_BY)
            Total_Gen_Stack_2 = self.df_process_gen_inputs(Total_Gen_Stack_2)
            #Adds in all possible columns from ordered gen to ensure the two dataframes have same column names
            Total_Gen_Stack_2 = pd.DataFrame(Total_Gen_Stack_2, columns=self.ordered_gen).fillna(0)

            self.logger.info(f'Scenario 1 = {self.Scenario_Diff[0]}')
            self.logger.info(f'Scenario 2 = {self.Scenario_Diff[1]}')
            Gen_Stack_Out = Total_Gen_Stack_1-Total_Gen_Stack_2

            if pd.notna(start_date_range):
                self.logger.info(f"Plotting specific date range: \
                {str(start_date_range)} to {str(end_date_range)}")
                Gen_Stack_Out = Gen_Stack_Out[start_date_range : end_date_range]
            else:
                self.logger.info("Plotting graph for entire timeperiod")

            # Removes columns that only equal 0
            Gen_Stack_Out.dropna(inplace=True)
            Gen_Stack_Out = Gen_Stack_Out.loc[:, (Gen_Stack_Out != 0).any(axis=0)]

            if Gen_Stack_Out.empty == True:
                outputs[zone_input] = MissingZoneData()
                continue

            # Reverses order of columns
            Gen_Stack_Out = Gen_Stack_Out.iloc[:, ::-1]

            unitconversion = self.capacity_energy_unitconversion(Gen_Stack_Out)
            Gen_Stack_Out = Gen_Stack_Out/unitconversion['divisor']

            # Data table of values to return to main program
            Data_Table_Out = Gen_Stack_Out.add_suffix(f" ({unitconversion['units']})")

            mplt = SetupSubplot()
            fig, ax = mplt.get_figure()

            for column in Gen_Stack_Out:
                ax.plot(Gen_Stack_Out[column], linewidth=3, 
                        color=self.PLEXOS_color_dict[column], label=column)

            mplt.add_main_title(self.Scenario_Diff[0].replace('_', ' ') + " vs. " + self.Scenario_Diff[1].replace('_', ' '))
            ax.set_ylabel(f"Generation Difference ({unitconversion['units']})",  color='black', rotation='vertical')
            ax.set_xlabel(timezone,  color='black', rotation='horizontal')
            mplt.set_yaxis_major_tick_format()
            ax.margins(x=0.01)
            mplt.set_subplot_timeseries_format()
            mplt.add_legend()
            
            outputs[zone_input] = {'fig': fig, 'data_table': Data_Table_Out}
        return outputs