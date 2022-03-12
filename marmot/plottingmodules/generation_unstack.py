"""Timeseries generation line plots. 

This code creates generation non-stacked line plots.

@author: Daniel Levie
"""
import logging
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import marmot.utils.mconfig as mconfig


from marmot.plottingmodules.plotutils.plot_library import SetupSubplot
from marmot.plottingmodules.plotutils.plot_data_helper import PlotDataHelper
from marmot.plottingmodules.plotutils.plot_exceptions import (MissingInputData, MissingZoneData)


class MPlot(PlotDataHelper):
    """generation_unstack MPlot class.

    All the plotting modules use this same class name.
    This class contains plotting methods that are grouped based on the
    current module name.
    
    The generation_unstack.py module contains methods that are
    related to the timeseries generation of generators, displayed in an unstacked line format. 
    
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

        self.logger = logging.getLogger('plotter.'+__name__)
        
        self.curtailment_prop = mconfig.parser("plot_data","curtailment_property")

        
    def gen_unstack(self, figure_name: str = None, prop: str = None,
                    start: float = None, end: float= None,
                    timezone: str = "", start_date_range: str = None,
                    end_date_range: str = None,
                    data_resolution: str = "", **_):
        """Creates a timeseries plot of generation by technology each plotted as a line.

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
            properties = [(True,f"generator_Generation{data_resolution}",scenario_list),
                          (False,f"generator_{self.curtailment_prop}{data_resolution}",scenario_list),
                          (False,f"generator_Pump_Load{data_resolution}",scenario_list),
                          (True,f"{agg}_Load{data_resolution}",scenario_list),
                          (False,f"{agg}_Unserved_Energy{data_resolution}",scenario_list)]
            
            # Runs get_formatted_data within PlotDataHelper to populate PlotDataHelper dictionary  
        # with all required properties, returns a 1 if required data is missing
            return self.get_formatted_data(properties)
        
        if facet:
            check_input_data = getdata(self.Scenarios)
            all_scenarios = self.Scenarios
        else:
            check_input_data = getdata([self.Scenarios[0]])  
            all_scenarios = [self.Scenarios[0]]
        
        # Checks if all data required by plot is available, if 1 in list required data is missing
        if 1 in check_input_data:
            outputs = MissingInputData()
            return outputs
            
        # sets up x, y dimensions of plot
        ncols, nrows = self.set_facet_col_row_dimensions(multi_scenario=all_scenarios)

        # If the plot is not a facet plot, grid size should be 1x1
        if not facet:
            ncols = 1
            nrows = 1
        
        grid_size = ncols*nrows
            
        # Used to calculate any excess axis to delete
        plot_number = len(all_scenarios)
        
        for zone_input in self.Zones:
            self.logger.info(f"Zone = {zone_input}")
        
            excess_axs = grid_size - plot_number
            
            mplt = SetupSubplot(nrows, ncols, sharey=True, 
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
                    Stacked_Gen = self[f"generator_Generation{data_resolution}"].get(scenario).copy()
                    if self.shift_leapday == True:
                        Stacked_Gen = self.adjust_for_leapday(Stacked_Gen)
                    Stacked_Gen = Stacked_Gen.xs(zone_input,level=self.AGG_BY)
                except KeyError:
                    # self.logger.info('No generation in %s',zone_input)
                    continue

                if Stacked_Gen.empty == True:
                    continue

                Stacked_Gen = self.df_process_gen_inputs(Stacked_Gen)

                # Insert Curtailment into gen stack if it exists in database
                Stacked_Curt = self[f"generator_{self.curtailment_prop}{data_resolution}"].get(scenario).copy()
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
                        #Insert curtailment into
                        Stacked_Gen.insert(len(Stacked_Gen.columns),
                                            column=curtailment_name,
                                            value=Stacked_Curt) 
                        Stacked_Gen = Stacked_Gen.fillna(0)
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

                Load = self[f"{agg}_Load{data_resolution}"].get(scenario).copy()
                if self.shift_leapday == True:
                    Load = self.adjust_for_leapday(Load)     
                Load = Load.xs(zone_input,level=self.AGG_BY)
                Load = Load.groupby(["timestamp"]).sum()
                Load = Load.squeeze() #Convert to Series

                Pump_Load = self[f"generator_Pump_Load{data_resolution}"][scenario].copy()
                if Pump_Load.empty:
                    Pump_Load = self[f"generator_Generation{data_resolution}"][scenario].copy()
                    Pump_Load.iloc[:,0] = 0
                if self.shift_leapday == True:
                    Pump_Load = self.adjust_for_leapday(Pump_Load)                                
                Pump_Load = Pump_Load.xs(zone_input,level=self.AGG_BY)
                Pump_Load = Pump_Load.groupby(["timestamp"]).sum()
                Pump_Load = Pump_Load.squeeze() #Convert to Series
                if (Pump_Load == 0).all() == False:
                    Pump_Load = Load - Pump_Load
                else:
                    Pump_Load = Load
                
                Unserved_Energy = self[f"{agg}_Unserved_Energy{data_resolution}"][scenario].copy()    
                if Unserved_Energy.empty:
                    Unserved_Energy = self[f"{agg}_Load{data_resolution}"][scenario].copy()
                    Unserved_Energy.iloc[:,0] = 0           
                if self.shift_leapday == True:
                    Unserved_Energy = self.adjust_for_leapday(Unserved_Energy)                    
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

                elif pd.notna(start_date_range):
                    self.logger.info(f"Plotting specific date range: \
                    {str(start_date_range)} to {str(end_date_range)}")
                    Stacked_Gen = Stacked_Gen[start_date_range : end_date_range]
                    Load = Load[start_date_range : end_date_range]
                    Unserved_Energy = Unserved_Energy[start_date_range : end_date_range]
                else:
                    self.logger.info("Plotting graph for entire timeperiod")
                
                # unitconversion based off peak generation hour, only checked once 
                if i == 0:
                    unitconversion = self.capacity_energy_unitconversion(Stacked_Gen)
                Stacked_Gen = Stacked_Gen/unitconversion['divisor']
                Unserved_Energy = Unserved_Energy/unitconversion['divisor']
                # Remove any all 0 columns
                Stacked_Gen = Stacked_Gen.loc[:, (Stacked_Gen != 0).any(axis=0)]

                scenario_names = pd.Series([scenario]*len(Stacked_Gen),name='Scenario')
                data_table = Stacked_Gen.add_suffix(f" ({unitconversion['units']})")
                data_table = data_table.set_index([scenario_names],append=True)
                data_tables.append(data_table)
                
                for column in Stacked_Gen.columns:
                    axs[i].plot(Stacked_Gen.index.values, Stacked_Gen[column], 
                                linewidth=2,
                                color=self.PLEXOS_color_dict.get(column,'#333333'),
                                label=column)

                if (Unserved_Energy == 0).all() == False:
                    axs[i].plot(Unserved_Energy, color='#DD0200',
                                label='Unserved Energy')

                mplt.set_yaxis_major_tick_format(sub_pos=i)
                axs[i].margins(x=0.01)
                mplt.set_subplot_timeseries_format(sub_pos=i)

            if not data_tables:
                self.logger.warning(f'No generation in {zone_input}')
                out = MissingZoneData()
                outputs[zone_input] = out
                continue

            data_table_out = pd.concat(data_tables)

            # add facet labels
            mplt.add_facet_labels(xlabels=self.xlabels,
                                  ylabels = self.ylabels)
            # Add legend
            mplt.add_legend(reverse_legend=True, sort_by=self.ordered_gen)
            # Remove extra supl
            mplt.remove_excess_axs(excess_axs, grid_size)
            # Add title
            mplt.add_main_title(zone_input)

            labelpad = 40
            plt.ylabel(f"Generation ({unitconversion['units']})", 
                        color='black', rotation='vertical', labelpad=labelpad)
 
            outputs[zone_input] = {'fig':fig, 'data_table':data_table_out}
        return outputs
