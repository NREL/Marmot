# -*- coding: utf-8 -*-
"""Total generation plots.

This module plots figures of total generation for a year, month etc.

@author: Daniel Levie 
"""

import logging
import pandas as pd
import matplotlib.pyplot as plt

import marmot.config.mconfig as mconfig
from marmot.plottingmodules.plotutils.plot_library import PlotLibrary, SetupSubplot
from marmot.plottingmodules.plotutils.plot_data_helper import PlotDataHelper
from marmot.plottingmodules.plotutils.plot_exceptions import (MissingInputData,
            MissingZoneData)


class MPlot(PlotDataHelper):
    """total_generation MPlot class.

    All the plotting modules use this same class name.
    This class contains plotting methods that are grouped based on the
    current module name.
    
    The total_genertion.py module contains methods that are
    display the total amount of generation over a given time period.
    
    MPlot inherits from the PlotDataHelper class to assist in creating figures.
    
    Attributes:
        MONTHS (dict) = dictionary of months.
    """

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
        
    def total_gen(self, start_date_range: str = None, 
                  end_date_range: str = None, **_):
        """Creates a stacked bar plot of total generation by technology type.

        A separate bar is created for each scenario.

        Args:
            start_date_range (str, optional): Defines a start date at which to represent data from. 
                Defaults to None.
            end_date_range (str, optional): Defines a end date at which to represent data to.
                Defaults to None.

        Returns:
            dict: Dictionary containing the created plot and its data table.
        """
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
                      (False,f"{agg}_Unserved_Energy",self.Scenarios),
                      (False,"batterie_Generation",self.Scenarios),
                      (False,"batterie_Load",self.Scenarios)]

        # Runs get_formatted_data within PlotDataHelper to populate PlotDataHelper dictionary  
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)

        if 1 in check_input_data:
            outputs = MissingInputData()
            return outputs

        for zone_input in self.Zones:

            # Will hold retrieved data for each scenario
            gen_chunks = []
            load_chunk = []
            pumped_load_chunk = []
            total_demand_chunk = []
            unserved_energy_chunk = []
            batt_load_chunk = []

            self.logger.info(f"Zone = {zone_input}")

            for scenario in self.Scenarios:

                self.logger.info(f"Scenario = {scenario}")
                Total_Gen_Stack = self['generator_Generation'].get(scenario)

                #Check if zone has generation, if not skips
                try:
                    Total_Gen_Stack = Total_Gen_Stack.xs(zone_input,level=self.AGG_BY)
                except KeyError:
                    self.logger.warning(f"No installed capacity in: {zone_input}")
                    continue

                Total_Gen_Stack = self.df_process_gen_inputs(Total_Gen_Stack)

                # Calculates interval step to correct for MWh of generation
                interval_count = self.get_sub_hour_interval_count(Total_Gen_Stack)

                curtailment_name = self.gen_names_dict.get('Curtailment','Curtailment')

                # Insert Curtailment into gen stack if it exists in database
                Stacked_Curt = self[f"generator_{self.curtailment_prop}"].get(scenario)
                if not Stacked_Curt.empty:
                    if zone_input in Stacked_Curt.index.get_level_values(self.AGG_BY).unique():
                        Stacked_Curt = Stacked_Curt.xs(zone_input,level=self.AGG_BY)
                        Stacked_Curt = self.df_process_gen_inputs(Stacked_Curt)
                        # If using Marmot's curtailment property
                        if self.curtailment_prop == 'Curtailment':
                            Stacked_Curt = self.assign_curtailment_techs(Stacked_Curt)
                        Stacked_Curt = Stacked_Curt.sum(axis=1)
                        Total_Gen_Stack.insert(len(Total_Gen_Stack.columns), 
                                               column=curtailment_name, value=Stacked_Curt) #Insert curtailment into


                # Insert battery generation into gen stack.
                batt_gen = self["batterie_Generation"].get(scenario).copy()
                if not batt_gen.empty:
                    if self.shift_leapday == True:
                        batt_gen = self.adjust_for_leapday(batt_gen)
                #if zone_input in batt_gen.index.get_level_values(self.AGG_BY).unique():
                #batt_gen = batt_gen.xs(zone_input,level=self.AGG_BY)
                batt_gen = batt_gen.groupby(level = ['timestamp']).sum()
                Total_Gen_Stack.insert(len(Total_Gen_Stack.columns),column='BESS discharging', value=batt_gen)
                
                Total_Gen_Stack = Total_Gen_Stack.loc[:, (Total_Gen_Stack != 0).any(axis=0)]
                Total_Gen_Stack = Total_Gen_Stack/interval_count

                if pd.notna(start_date_range):
                    self.logger.info(f"Plotting specific date range: \
                                      {str(start_date_range)} to {str(end_date_range)}")
                    Total_Gen_Stack = Total_Gen_Stack[start_date_range:end_date_range]

                Total_Gen_Stack = Total_Gen_Stack.sum(axis=0)
                Total_Gen_Stack.rename(scenario, inplace=True)
                
                Total_Load = self[f"{agg}_Load"].get(scenario)
                Total_Load = Total_Load.xs(zone_input,level=self.AGG_BY)
                Total_Load = Total_Load.groupby(["timestamp"]).sum()

                if pd.notna(start_date_range):
                    Total_Load = Total_Load[start_date_range:end_date_range]

                Total_Load = Total_Load.rename(columns={0:scenario}).sum(axis=0)
                Total_Load = Total_Load/interval_count

                Unserved_Energy = self[f"{agg}_Unserved_Energy"][scenario]
                if Unserved_Energy.empty:
                    Unserved_Energy = self[f"{agg}_Load"][scenario].copy()
                    Unserved_Energy.iloc[:,0] = 0
                Unserved_Energy = Unserved_Energy.xs(zone_input,level=self.AGG_BY)
                Unserved_Energy = Unserved_Energy.groupby(["timestamp"]).sum()
                
                if pd.notna(start_date_range):
                    Unserved_Energy = Unserved_Energy[start_date_range:end_date_range]

                Unserved_Energy = Unserved_Energy.rename(columns={0:scenario}).sum(axis=0)
                Unserved_Energy = Unserved_Energy/interval_count

                # subtract unserved energy from load for graphing (not sure this is actually used)
                if (Unserved_Energy == 0).all() == False:
                    Unserved_Energy = Total_Load - Unserved_Energy

                Pump_Load = self["generator_Pump_Load"][scenario]
                if Pump_Load.empty or not mconfig.parser("plot_data","include_total_pumped_load_line"):
                    Pump_Load = self['generator_Generation'][scenario].copy()
                    Pump_Load.iloc[:,0] = 0
                Pump_Load = Pump_Load.xs(zone_input,level=self.AGG_BY)
                Pump_Load = Pump_Load.groupby(["timestamp"]).sum()
                if pd.notna(start_date_range):
                    Pump_Load = Pump_Load[start_date_range:end_date_range]
                
                Pump_Load = Pump_Load.rename(columns={0:scenario}).sum(axis=0)
                Pump_Load = Pump_Load/interval_count

                batt_load = self['batterie_Load'][scenario].copy()
                if self.shift_leapday == True: batt_load = self.adjust_for_leapday(batt_load)
                batt_load = batt_load.groupby('timestamp').sum().squeeze()

                if (Pump_Load == 0).all() == False:
                    Total_Demand = Total_Load - Pump_Load
                elif (batt_load == 0).all() == False:
                    Total_Demand = Total_Load - batt_load
                else:
                    Total_Demand = Total_Load.copy()

                batt_load = pd.DataFrame(batt_load)
                batt_load.rename(columns = {0:scenario},inplace = True)
                batt_load = batt_load.sum(axis = 0)

                gen_chunks.append(Total_Gen_Stack)
                load_chunk.append(Total_Load)
                pumped_load_chunk.append(Pump_Load)
                total_demand_chunk.append(Total_Demand)
                unserved_energy_chunk.append(Unserved_Energy)
                batt_load_chunk.append(batt_load)

            if not gen_chunks:
                outputs[zone_input] = MissingZoneData()
                continue

            Total_Generation_Stack_Out = pd.concat(gen_chunks, axis=1, sort=False).fillna(0)
            Total_Load_Out = pd.concat(load_chunk, axis=0, sort=False)
            Pump_Load_Out = pd.concat(pumped_load_chunk, axis=0, sort=False)
            Total_Demand_Out = pd.concat(total_demand_chunk, axis=0, sort=False)
            Unserved_Energy_Out = pd.concat(unserved_energy_chunk, axis=0, sort=False)
            batt_load_out = pd.concat(batt_load_chunk, axis = 0,sort = False)

            Total_Load_Out = Total_Load_Out.rename('Total Load (Demand + \n Storage Charging)')
            Total_Demand_Out = Total_Demand_Out.rename('Total Demand')
            Unserved_Energy_Out = Unserved_Energy_Out.rename('Unserved Energy')

            # Add Net Imports if desired
            if mconfig.parser("plot_data","include_total_net_imports"):
                Total_Generation_Stack_Out =\
                    self.include_net_imports(Total_Generation_Stack_Out, 
                                             Total_Load_Out,
                                             Unserved_Energy_Out)

            Total_Generation_Stack_Out = Total_Generation_Stack_Out.T
            Total_Generation_Stack_Out = Total_Generation_Stack_Out.loc[:, (Total_Generation_Stack_Out != 0).any(axis=0)]

            # unit conversion return divisor and energy units
            unitconversion = self.capacity_energy_unitconversion(Total_Generation_Stack_Out,
                                                                    sum_values=True)

            Total_Generation_Stack_Out = Total_Generation_Stack_Out/unitconversion['divisor']
            Total_Load_Out = Total_Load_Out.T/unitconversion['divisor']
            Pump_Load_Out = Pump_Load_Out.T/unitconversion['divisor']
            Total_Demand_Out = Total_Demand_Out.T/unitconversion['divisor']
            Unserved_Energy_Out = Unserved_Energy_Out.T/unitconversion['divisor']
            batt_load_out = batt_load_out.T/unitconversion['divisor']

            # Data table of values to return to main program
            # Data_Table_Out = pd.concat([Total_Load_Out.T,
            #                             Total_Demand_Out.T,
            #                             Unserved_Energy_Out.T,
            #                             Total_Generation_Stack_Out],  axis=1, sort=False)
            # Data_Table_Out = Data_Table_Out.add_suffix(f" ({unitconversion['units']}h)")

            Total_Generation_Stack_Out.index = Total_Generation_Stack_Out.index.str.replace('_',' ')
            
            mplt = PlotLibrary()
            fig, ax = mplt.get_figure()

            # Set x-tick labels 
            if len(self.custom_xticklabels) > 1:
                tick_labels = self.custom_xticklabels
            else:
                tick_labels = Total_Generation_Stack_Out.index
            
            mplt.barplot(Total_Generation_Stack_Out, color=self.PLEXOS_color_dict,
                         stacked=True, custom_tick_labels=tick_labels)

            ax.set_ylabel(f"Total Generation ({unitconversion['units']}h)", 
                          color='black', rotation='vertical')
            
            for n, scenario in enumerate(self.Scenarios):

                x = [ax.patches[n].get_x(), ax.patches[n].get_x() + ax.patches[n].get_width()]
                
                height1 = [float(Total_Load_Out[scenario].sum())]*2
                if Pump_Load_Out[scenario] > 0:
                    ax.plot(x, height1, c='black', linewidth=3,
                                  label='Demand + \n Storage Charging')
                    height2 = [float(Total_Demand_Out[scenario])]*2
                    ax.plot(x, height2, 'r--', c='black', linewidth=1.5, 
                                  label='Demand')
                else:
                    ax.plot(x, height1, c='black', linewidth=3, 
                           label='Demand')

                if Unserved_Energy_Out[scenario] > 0:
                    height3 = [float(Unserved_Energy_Out[scenario])]*2
                    ax.plot(x, height3, c='#DD0200', linewidth=1.5,
                            label='Unserved Energy')
                    ax.fill_between(x, height3, height1,
                                facecolor = '#DD0200',
                                alpha=0.5)
            # Add legend
            mplt.add_legend(reverse_legend=True, sort_by=self.ordered_gen)
            # Add title
            if mconfig.parser("plot_title_as_region"):
                mplt.add_main_title(zone_input)

            outputs[zone_input] = {'fig': fig, 'data_table': Data_Table_Out}

        return outputs

    def total_gen_diff(self, start_date_range: str = None, 
                       end_date_range: str = None, **_):
        """Creates a stacked bar plot of total generation by technology type, relative to a base scenario.

        Barplots show the change in total generation relative to a base scenario.
        The default is to comapre against the first scenario provided in the inputs list.
        Each sceanrio is plotted as a separate bar.

        Args:
            start_date_range (str, optional): Defines a start date at which to represent data from. 
                Defaults to None.
            end_date_range (str, optional): Defines a end date at which to represent data to.
                Defaults to None.

        Returns:
            dict: Dictionary containing the created plot and its data table.
        """
        # Create Dictionary to hold Datframes for each scenario
        outputs = {}

        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True, "generator_Generation", self.Scenarios),
                      (False, f"generator_{self.curtailment_prop}", self.Scenarios)]

        # Runs get_formatted_data within PlotDataHelper to populate PlotDataHelper dictionary  
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)

        if 1 in check_input_data:
            outputs = MissingInputData()
            return outputs

        for zone_input in self.Zones:

            self.logger.info(f"Zone = {zone_input}")

            gen_chunks =[]
            for scenario in self.Scenarios:

                self.logger.info(f"Scenario = {scenario}")

                Total_Gen_Stack = self['generator_Generation'].get(scenario)

                #Check if zone has generation, if not skips and breaks out of Multi_Scenario loop
                try:
                    Total_Gen_Stack = Total_Gen_Stack.xs(zone_input,level=self.AGG_BY)
                except KeyError:
                    self.logger.warning(f"No installed capacity in : {zone_input}")
                    break

                Total_Gen_Stack = self.df_process_gen_inputs(Total_Gen_Stack)

                # Calculates interval step to correct for MWh of generation
                interval_count = self.get_sub_hour_interval_count(Total_Gen_Stack)

                # Insert Curtailment into gen stack if it exists in database
                Stacked_Curt = self[f"generator_{self.curtailment_prop}"].get(scenario)
                if not Stacked_Curt.empty:
                    curtailment_name = self.gen_names_dict.get('Curtailment','Curtailment')
                    if zone_input in Stacked_Curt.index.get_level_values(self.AGG_BY).unique():
                        Stacked_Curt = Stacked_Curt.xs(zone_input,level=self.AGG_BY)
                        Stacked_Curt = self.df_process_gen_inputs(Stacked_Curt)
                        # If using Marmot's curtailment property
                        if self.curtailment_prop == 'Curtailment':
                            Stacked_Curt = self.assign_curtailment_techs(Stacked_Curt)
                        Stacked_Curt = Stacked_Curt.sum(axis=1)
                        Total_Gen_Stack.insert(len(Total_Gen_Stack.columns), 
                                               column=curtailment_name, value=Stacked_Curt) #Insert curtailment into
                        Total_Gen_Stack = Total_Gen_Stack.loc[:, (Total_Gen_Stack != 0).any(axis=0)]

                Total_Gen_Stack = Total_Gen_Stack/interval_count
                if pd.notna(start_date_range):
                    self.logger.info(f"Plotting specific date range: \
                                     {str(start_date_range)} to {str(end_date_range)}")
                    Total_Gen_Stack = Total_Gen_Stack[start_date_range:end_date_range]
                
                Total_Gen_Stack = Total_Gen_Stack.sum(axis=0)
                Total_Gen_Stack.rename(scenario, inplace=True)
                
                gen_chunks.append(Total_Gen_Stack)
            
            if not gen_chunks:
                outputs[zone_input] = MissingZoneData()
                continue

            Total_Generation_Stack_Out = pd.concat(gen_chunks, axis=1, sort=False).fillna(0)

            Total_Generation_Stack_Out = self.create_categorical_tech_index(Total_Generation_Stack_Out)
            Total_Generation_Stack_Out = Total_Generation_Stack_Out.T
            Total_Generation_Stack_Out = Total_Generation_Stack_Out.loc[:, (Total_Generation_Stack_Out != 0).any(axis=0)]

            #Ensures region has generation, else skips
            try:
                #Change to a diff on first scenario
                Total_Generation_Stack_Out = Total_Generation_Stack_Out-Total_Generation_Stack_Out.xs(self.Scenarios[0]) 
            except KeyError:
                outputs[zone_input] = MissingZoneData()
                continue

            Total_Generation_Stack_Out.drop(self.Scenarios[0],inplace=True) #Drop base entry

            if Total_Generation_Stack_Out.empty == True:
                outputs[zone_input] = MissingZoneData()
                continue
            
            unitconversion = self.capacity_energy_unitconversion(Total_Generation_Stack_Out,
                                                                            sum_values=True)
            Total_Generation_Stack_Out = Total_Generation_Stack_Out/unitconversion['divisor']

            # Data table of values to return to main program
            Data_Table_Out = Total_Generation_Stack_Out.add_suffix(f" ({unitconversion['units']}h)")

            net_diff = Total_Generation_Stack_Out
            try:
                net_diff.drop(columns = curtailment_name,inplace=True)
            except KeyError:
                pass
            net_diff = net_diff.sum(axis = 1)
            
            mplt = PlotLibrary()
            fig, ax = mplt.get_figure()

            mplt.barplot(Total_Generation_Stack_Out, stacked=True, 
                         color=self.PLEXOS_color_dict)
            
            # Set x-tick labels 
            tick_labels = Total_Generation_Stack_Out.index
            mplt.set_barplot_xticklabels(tick_labels)

            #Add net gen difference line.
            for n, scenario in enumerate(self.Scenarios[1:]):
                x = [ax.patches[n].get_x(), ax.patches[n].get_x() + ax.patches[n].get_width()]
                y_net = [net_diff.loc[scenario]] * 2
                ax.plot(x, y_net, c='black', linewidth=1.5, 
                        label='Net Gen Change')

            ax.set_ylabel((f"Generation Change ({format(unitconversion['units'])}h) \n "
                          f"relative to {self.Scenarios[0].replace('_',' ')}"), 
                          color='black', rotation='vertical')
            
            ax.axhline(linewidth=0.5, linestyle='--', color='grey')
            # Add legend
            mplt.add_legend(reverse_legend=True, sort_by=self.ordered_gen)

            if mconfig.parser("plot_title_as_region"):
                mplt.add_main_title(zone_input)
            outputs[zone_input] = {'fig': fig, 'data_table': Data_Table_Out}
        return outputs

    def total_gen_monthly(self, **kwargs):
        """Creates stacked bar plot of total generation by technology by month.

        A separate bar is created for each scenario.

        This methods calls _monthly_gen() to create the figure.

        Returns:
            dict: Dictionary containing the created plot and its data table.
        """

        outputs = self._monthly_gen(**kwargs)
        return outputs

    def monthly_vre_generation_percentage(self, **kwargs):
        """Creates clustered barplot of total monthly percentage variable renewable generation by technology.

           Each vre technology + curtailment if present is plotted as a separate clustered bar, 
           the total of all bars add to 100%.
           Each scenario is plotted on a separate facet plot.
           Technologies that belong to VRE can be set in the ordered_gen_catagories.csv file 
           in the Mapping folder.

           This methods calls _monthly_gen() and passes the vre_only=True and 
           plot_as_percnt=True arguments to create the figure.

        Returns:
            dict: Dictionary containing the created plot and its data table.
        """
        
        outputs = self._monthly_gen(vre_only=True, plot_as_percnt=True, **kwargs)
        return outputs
    
    def monthly_vre_generation(self, **kwargs):
        """Creates clustered barplot of total monthly variable renewable generation by technology.

           Each vre technology + curtailment if present is plotted as a separate clustered bar
           Each scenario is plotted on a separate facet plot.
           Technologies that belong to VRE can be set in the ordered_gen_catagories.csv file 
           in the Mapping folder.

           This methods calls _monthly_gen() and passes the vre_only=True arguments to 
           create the figure.

        Returns:
            dict: Dictionary containing the created plot and its data table.
        """
        
        outputs = self._monthly_gen(vre_only=True, **kwargs)
        return outputs

    def _monthly_gen(self, vre_only: bool = False, 
                     plot_as_percnt: bool = False, **_):
        """Creates monthly generation plot, internal method called from 
            monthly_vre_percentage_generation or monthly_vre_generation

        Args:
            vre_only (bool, optional): If True only plots vre technologies.
                Defaults to False.
            plot_as_percnt (bool, optional): If True only plots data as a percentage.
                Defaults to False.

        Returns:
            dict: Dictionary containing the created plot and its data table.
        """
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
                      (True,f"{agg}_Load",self.Scenarios)]

        # Runs get_formatted_data within PlotDataHelper to populate PlotDataHelper dictionary  
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)

        if 1 in check_input_data:
            outputs = MissingInputData()
            return outputs
        
        ncols, nrows = self.set_facet_col_row_dimensions(multi_scenario=self.Scenarios)
        grid_size = ncols*nrows

        # Used to calculate any excess axis to delete
        plot_number = len(self.Scenarios)
        excess_axs = grid_size - plot_number
         
        for zone_input in self.Zones:
            
            self.logger.info(f"Zone = {zone_input}")
            
            # Will hold retrieved data for each scenario
            gen_chunks = []
            load_chunk = []
            pumped_load_chunk = []
            total_demand_chunk = []

            # Loop gets all data by scenario
            for i, scenario in enumerate(self.Scenarios):
                
                self.logger.info(f"Scenario = {scenario}")
                Total_Gen_Stack = self['generator_Generation'].get(scenario)
    
                #Check if zone has generation, if not skips
                try:
                    Total_Gen_Stack = Total_Gen_Stack.xs(zone_input,level=self.AGG_BY)
                except KeyError:
                    self.logger.warning(f"No installed capacity in: {zone_input}")
                    continue
                       
                Total_Gen_Stack = self.df_process_gen_inputs(Total_Gen_Stack)
                if vre_only:
                    Total_Gen_Stack = Total_Gen_Stack[Total_Gen_Stack.columns.intersection(self.vre_gen_cat)]

                if Total_Gen_Stack.empty:
                    if vre_only:
                        self.logger.warning(f"No vre in: {zone_input}")
                    gen_chunks.append(pd.DataFrame())
                    continue

                Total_Gen_Stack.columns = Total_Gen_Stack.columns.add_categories('timestamp')
    
                # Calculates interval step to correct for MWh of generation if data is subhourly
                interval_count = self.get_sub_hour_interval_count(Total_Gen_Stack)
    
                #Insert Curtailment into gen stack if it exists in database
                Stacked_Curt = self[f"generator_{self.curtailment_prop}"].get(scenario)
                if not Stacked_Curt.empty:
                    curtailment_name = self.gen_names_dict.get('Curtailment','Curtailment')
                    if zone_input in Stacked_Curt.index.get_level_values(self.AGG_BY).unique():
                        Stacked_Curt = Stacked_Curt.xs(zone_input, level=self.AGG_BY)
                        Stacked_Curt = self.df_process_gen_inputs(Stacked_Curt)
                        # If using Marmot's curtailment property
                        if self.curtailment_prop == 'Curtailment':
                            Stacked_Curt = self.assign_curtailment_techs(Stacked_Curt)
                        Stacked_Curt = Stacked_Curt.sum(axis=1)
                        Total_Gen_Stack.insert(len(Total_Gen_Stack.columns), 
                                               column=curtailment_name, value=Stacked_Curt) 
                        Total_Gen_Stack = Total_Gen_Stack.loc[:, (Total_Gen_Stack != 0).any(axis=0)]

                # Get Total Load 
                Total_Load = self[f"{agg}_Load"].get(scenario)
                Total_Load = Total_Load.xs(zone_input, level=self.AGG_BY)
                Total_Load = Total_Load.groupby(["timestamp"]).sum()

                # Get Pumped Load 
                Pump_Load = self["generator_Pump_Load"][scenario]
                if Pump_Load.empty or not mconfig.parser("plot_data","include_total_pumped_load_line"):
                    Pump_Load = self['generator_Generation'][scenario].copy()
                    Pump_Load.iloc[:,0] = 0
                Pump_Load = Pump_Load.xs(zone_input, level=self.AGG_BY)
                Pump_Load = Pump_Load.groupby(["timestamp"]).sum()

                def _group_monthly_data(input_df, interval_count):
                    '''Groups data into months'''

                    monthly_df = input_df/interval_count
                    monthly_df = monthly_df.groupby(pd.Grouper(freq='M')).sum()
                    if len(monthly_df.index) > 12:
                        monthly_df = monthly_df[:-1]
                    monthly_df.reset_index(drop=False, inplace=True)
                    monthly_df['timestamp'] = monthly_df['timestamp'].dt.month.apply(lambda x: self.MONTHS[x])
                    monthly_df.set_index('timestamp', inplace=True)
                    return monthly_df

                # Group data into months
                monthly_gen_stack = _group_monthly_data(Total_Gen_Stack, interval_count)  
                monthly_total_load = _group_monthly_data(Total_Load, interval_count)  
                monthly_pumped_load = _group_monthly_data(Pump_Load, interval_count)         
                # Calculate Total Demand 
                monthly_total_demand = monthly_total_load - monthly_pumped_load
                
                # If plotting percentage data convert to percentages 
                if plot_as_percnt:
                    monthly_total_gen = pd.DataFrame(monthly_gen_stack.T.sum(), columns=['Total Generation'])
                    for vre_col in monthly_gen_stack.columns:
                        monthly_gen_stack[vre_col] = (monthly_gen_stack[vre_col] / monthly_total_gen['Total Generation'])

                # Add scenario index 
                scenario_names = pd.Series([scenario] * len(monthly_gen_stack), name='Scenario')
                monthly_gen_stack = monthly_gen_stack.set_index([scenario_names], append = True)
                monthly_total_load = monthly_total_load.set_index([scenario_names], append = True)
                monthly_pumped_load = monthly_pumped_load.set_index([scenario_names], append = True)
                monthly_total_demand = monthly_total_demand.set_index([scenario_names], append = True)

                # Add all data to lists
                gen_chunks.append(monthly_gen_stack)
                load_chunk.append(monthly_total_load)
                pumped_load_chunk.append(monthly_pumped_load)
                total_demand_chunk.append(monthly_total_demand)

            if not gen_chunks:
                # If no generation in select zone/region
                outputs[zone_input] = MissingZoneData()
                continue

            # Concat all data into single data-frames
            Gen_Out = pd.concat(gen_chunks,axis=0, sort=False)
            # Drop any technologies with 0 Gen
            Gen_Out = Gen_Out.loc[:, (Gen_Out != 0).any(axis=0)]

            if Gen_Out.empty:
                outputs[zone_input] = MissingZoneData()
                continue
            
            # Concat all data into single data-frames
            Pump_Load_Out = pd.concat(pumped_load_chunk, axis=0, sort=False)
            Total_Demand_Out = pd.concat(total_demand_chunk, axis=0, sort=False)
            Total_Load_Out = pd.concat(load_chunk, axis=0, sort=False)
            
            # Rename load columns
            Pump_Load_Out.rename(columns={0:'Pump Load'}, inplace=True)
            Total_Demand_Out.rename(columns={0:'Total Demand'}, inplace=True)
            Total_Load_Out.rename(columns={0:'Total Load (Demand + \n Storage Charging)'}, inplace=True)

            # Add Net Imports if desired
            if mconfig.parser("plot_data","include_total_net_imports") and \
                not vre_only:
                Gen_Out = self.include_net_imports(Gen_Out, Total_Load_Out)

            if not plot_as_percnt:
                # unit conversion return divisor and energy units
                if vre_only:
                    unitconversion = self.capacity_energy_unitconversion(Gen_Out)
                else:
                    unitconversion = self.capacity_energy_unitconversion(Gen_Out,
                                                                            sum_values=True)                
                Gen_Out = Gen_Out/unitconversion['divisor']
                Total_Demand_Out = Total_Demand_Out/unitconversion['divisor']
                Total_Load_Out = Total_Load_Out/unitconversion['divisor']
                Pump_Load_Out = Pump_Load_Out/unitconversion['divisor']
                #Data table of values to return to main program
                Data_Table_Out = pd.concat([Total_Load_Out, Total_Demand_Out, Pump_Load_Out, 
                                            Gen_Out], axis=1).add_suffix(f" ({unitconversion['units']}h)")
            else:
                Data_Table_Out = Gen_Out.add_suffix(f" (%-Gen)") * 100

            mplt = PlotLibrary(nrows, ncols, sharey=True,
                                squeeze=False, ravel_axs=True)
            fig, axs = mplt.get_figure()
            plt.subplots_adjust(wspace=0.05, hspace=0.5)

            if ncols > 1:
                font_scaling_ratio = 1 + ((ncols-1)*0.09)
                plt.rcParams['xtick.labelsize'] *= font_scaling_ratio
                plt.rcParams['ytick.labelsize'] *= font_scaling_ratio
                plt.rcParams['legend.fontsize'] *= font_scaling_ratio
                plt.rcParams['axes.labelsize'] *= font_scaling_ratio
                plt.rcParams['axes.titlesize'] *= font_scaling_ratio

            for i, scenario in enumerate(self.Scenarios):

                month_gen = Gen_Out.xs(scenario, level="Scenario")
                month_gen = month_gen.loc[:, (month_gen != 0).any(axis=0)]
                month_total_load = Total_Load_Out.xs(scenario, level="Scenario")
                month_pumped_load = Pump_Load_Out.xs(scenario, level="Scenario")
                month_total_demand = Total_Demand_Out.xs(scenario, level="Scenario")
                
                if vre_only:
                    stack = False
                else:
                    stack = True

                mplt.barplot(month_gen, color=self.PLEXOS_color_dict,
                            stacked=stack,
                            sub_pos=i)

                axs[i].margins(x=0.01)
                axs[i].set_xlabel("")
                
                if plot_as_percnt:
                    mplt.set_yaxis_major_tick_format(tick_format='percent', sub_pos=i)     
                
                if not vre_only:
                    for n, _m in enumerate(month_total_load.index):
                        x = [axs[i].patches[n].get_x(), axs[i].patches[n].get_x() +\
                             axs[i].patches[n].get_width()]
                        height1 = [float(month_total_load.loc[_m])]*2
                        
                        if month_pumped_load.loc[_m].values.sum() > 0:
                            axs[i].plot(x, height1, c='black', linewidth=3,
                                                label='Demand + \n Storage Charging')
                            height2 = [float(month_total_demand.loc[_m])]*2
                            axs[i].plot(x,height2, 'r--', c='black', linewidth=1.5,
                                                label='Demand')
                        else:
                            axs[i].plot(x, height1, c='black', linewidth=3,
                                               label='Demand')

            # add facet labels
            mplt.add_facet_labels(xlabels=self.xlabels,
                                  ylabels = self.ylabels)
            # Add legend
            mplt.add_legend(reverse_legend=True, sort_by=self.ordered_gen)
            #Remove extra axes
            mplt.remove_excess_axs(excess_axs, grid_size)

            #Y-label should change if there are facet labels, leave at 40 for now, works for all values in spacing
            labelpad = 40
            if plot_as_percnt:
                plt.ylabel(f"% of Generation",  color='black', rotation='vertical', labelpad=labelpad)
            else:
                plt.ylabel(f"Total Generation ({unitconversion['units']}h)", color='black', 
                           rotation='vertical', labelpad=labelpad)

            if mconfig.parser('plot_title_as_region'):
                mplt.add_main_title(zone_input)
            
            outputs[zone_input] = {'fig': fig, 'data_table': Data_Table_Out}

        return outputs

    def total_gen_pie(self, start_date_range: str = None, 
                      end_date_range: str = None, **_):
        """Creates a pie chart of total generation and curtailment.

        Each sceanrio is plotted as a separate pie chart.

        Args:
            start_date_range (str, optional): Defines a start date at which to represent data from. 
                Defaults to None.
            end_date_range (str, optional): Defines a end date at which to represent data to.
                Defaults to None.

        Returns:
            dict: Dictionary containing the created plot and its data table.
        """
        # Create Dictionary to hold Datframes for each scenario
        outputs = {}
                    
        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True,"generator_Generation",self.Scenarios), 
                      (False,f"generator_{self.curtailment_prop}",self.Scenarios)]

        # Runs get_formatted_data within PlotDataHelper to populate PlotDataHelper dictionary  
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)

        if 1 in check_input_data:
            outputs = MissingInputData()
            return outputs
        
        ncols, nrows = self.set_facet_col_row_dimensions(multi_scenario=self.Scenarios)
        grid_size = ncols*nrows

        # Used to calculate any excess axis to delete
        plot_number = len(self.Scenarios)
        excess_axs = grid_size - plot_number
        
        for zone_input in self.Zones:
            Total_Gen_Out = pd.DataFrame()
            self.logger.info(f"Zone = {zone_input}")

            mplt = SetupSubplot(nrows, ncols, sharey=True,
                                squeeze=False, ravel_axs=True)
            fig, axs = mplt.get_figure()
            
            gen_chunks = []
            for i, scenario in enumerate(self.Scenarios):
                
                self.logger.info(f"Scenario = {scenario}")
                Total_Gen_Stack = self['generator_Generation'].get(scenario)
    
                #Check if zone has generation, if not skips
                try:
                    Total_Gen_Stack = Total_Gen_Stack.xs(zone_input,level=self.AGG_BY)
                except KeyError:
                    self.logger.warning(f"No installed capacity in: {zone_input}")
                    continue
    
                Total_Gen_Stack = self.df_process_gen_inputs(Total_Gen_Stack)
                
                #Insert Curtailment into gen stack if it exists in database
                Stacked_Curt = self[f"generator_{self.curtailment_prop}"].get(scenario)
                if not Stacked_Curt.empty:
                    curtailment_name = self.gen_names_dict.get('Curtailment','Curtailment')
                    if zone_input in Stacked_Curt.index.get_level_values(self.AGG_BY).unique():
                        Stacked_Curt = Stacked_Curt.xs(zone_input,level=self.AGG_BY)
                        Stacked_Curt = self.df_process_gen_inputs(Stacked_Curt)
                        # If using Marmot's curtailment property
                        if self.curtailment_prop == 'Curtailment':
                            Stacked_Curt = self.assign_curtailment_techs(Stacked_Curt)
                        Stacked_Curt = Stacked_Curt.sum(axis=1)
                        #Insert curtailment into
                        Total_Gen_Stack.insert(len(Total_Gen_Stack.columns), 
                                               column=curtailment_name, value=Stacked_Curt) 
                        Total_Gen_Stack = Total_Gen_Stack.loc[:, (Total_Gen_Stack != 0).any(axis=0)]
                
                Total_Gen_Stack = Total_Gen_Stack.sum(axis=0)
                Total_Gen_Stack.rename(scenario, inplace=True)
                Total_Gen_Stack = (Total_Gen_Stack/sum(Total_Gen_Stack))*100
                gen_chunks.append(Total_Gen_Stack)
            
            if not gen_chunks:
                outputs[zone_input] = MissingZoneData()
                continue

            Total_Gen_Out = pd.concat(gen_chunks, axis=1, sort=False).fillna(0)
            Total_Gen_Out = Total_Gen_Out.loc[:, (Total_Gen_Out != 0).any(axis=0)]
            
            if Total_Gen_Out.empty:
                outputs[zone_input] = MissingZoneData()
                continue

            for i, scenario in enumerate(self.Scenarios):
                
                scenario_data = Total_Gen_Out[scenario]
               
                axs[i].pie(scenario_data, labels=scenario_data.index, 
                                       shadow=True, startangle=90, labeldistance=None,
                                       colors=[self.PLEXOS_color_dict.get(x, '#333333') 
                                               for x in scenario_data.index])
                axs[i].legend().set_visible(False)

            # add facet labels
            mplt.add_facet_labels(xlabels=self.xlabels,
                                  ylabels = self.ylabels)          
            # Add legend
            mplt.add_legend(reverse_legend=True, sort_by=self.ordered_gen)
            #Remove extra axes
            mplt.remove_excess_axs(excess_axs, grid_size)

            plt.tick_params(labelcolor='none', top=False, bottom=False, 
                            left=False, right=False)
            plt.ylabel(f"Total Generation (%)",  color='black', rotation='vertical')
            if mconfig.parser('plot_title_as_region'):
                mplt.add_main_title(zone_input)
            
            outputs[zone_input] = {'fig': fig, 'data_table': Total_Gen_Out}

        return outputs