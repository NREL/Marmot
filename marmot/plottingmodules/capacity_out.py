"""Generator outage plots.

This module contain methods that arerelated to related to 
generators that are on an outage.  

@author: Daniel Levie 
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import marmot.utils.mconfig as mconfig

from marmot.plottingmodules.plotutils.plot_library import PlotLibrary
from marmot.plottingmodules.plotutils.plot_data_helper import MPlotDataHelper
from marmot.plottingmodules.plotutils.plot_exceptions import (MissingInputData, 
            UnderDevelopment, MissingZoneData)

logger = logging.getLogger('plotter.'+__name__)
plot_data_settings = mconfig.parser("plot_data")
shift_leapday : bool = mconfig.parser("shift_leapday")

class CapacityOut(MPlotDataHelper):
    """Generator outage plots.

    The capacity_out.py module contains methods that are
    related to generators that are on an outage. 

    CapacityOut inherits from the MPlotDataHelper class to assist 
    in creating figures.
    """

    def __init__(self, **kwargs):
        # Instantiation of MPlotHelperFunctions
        super().__init__(**kwargs)

    def capacity_out_stack(self, start_date_range: str = None, 
                             end_date_range: str = None,
                             data_resolution: str = "", **_):
        """Creates Timeseries stacked area plots of generation on outage by technology.

        Each scenario is plotted by a separate facet plot. 

        Args:
            start_date_range (str, optional): Defines a start date at which to represent data from. 
                Defaults to None.
            end_date_range (str, optional): Defines a end date at which to represent data to.
                Defaults to None.

        Returns:
            dict: dictionary containing the created plot and its data table.
        """
        outputs : dict = {}
        
        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True,f"generator_Installed_Capacity{data_resolution}",self.Scenarios),
                      (True,f"generator_Available_Capacity{data_resolution}",self.Scenarios)]
        
        # Runs get_formatted_data within MPlotDataHelper to populate MPlotDataHelper dictionary  
        # with all required properties, returns a 1 if required data is missing
        check_input_data = self.get_formatted_data(properties)

        if 1 in check_input_data:
            return MissingInputData()
        
        # sets up x, y dimensions of plot
        ncols, nrows = self.set_facet_col_row_dimensions(multi_scenario=self.Scenarios)

        grid_size = ncols*nrows

        # Used to calculate any excess axis to delete
        plot_number = len(self.Scenarios)
        excess_axs = grid_size - plot_number

        for zone_input in self.Zones:
            logger.info(f'Zone = {str(zone_input)}')

            mplt = PlotLibrary(nrows, ncols, sharex=True,
                                sharey='row', 
                                squeeze=False, ravel_axs=True)
            fig, axs = mplt.get_figure()

            plt.subplots_adjust(wspace=0.1, hspace=0.2)

            chunks = []
            i=-1
            for scenario in (self.Scenarios):
                logger.info(f"Scenario = {scenario}")
                i+=1
                
                install_cap = self["generator_Installed_Capacity"].get(scenario).copy()
                avail_cap = self["generator_Available_Capacity"].get(scenario).copy()
                if shift_leapday:
                    avail_cap = self.adjust_for_leapday(avail_cap)
                if zone_input in avail_cap.index.get_level_values(self.AGG_BY).unique():
                    avail_cap = avail_cap.xs(zone_input,level=self.AGG_BY)
                else:
                    logger.warning(f"No Generation in: {zone_input}")
                    outputs[zone_input] = MissingZoneData()
                    continue
                avail_cap.columns = ['avail']
                install_cap.columns = ['cap']
                avail_cap['year'] = avail_cap.index.get_level_values('timestamp').year
                install_cap['year'] = install_cap.index.get_level_values('timestamp').year
                avail_cap.reset_index(inplace=True)
                
                cap_out = avail_cap.merge(install_cap, on=['year','tech','gen_name'])
                cap_out[0] = cap_out['cap'] - cap_out['avail']
                
                cap_out = self.df_process_gen_inputs(cap_out)
                #Subset only thermal gen categories
                thermal_gens = [therm for therm in self.thermal_gen_cat if therm in cap_out.columns]
                cap_out = cap_out[thermal_gens]
                
                if cap_out.empty is True:
                    logger.warning(f"No Thermal Generation in: {zone_input}")
                    continue
                if pd.notna(start_date_range):
                    cap_out = self.set_timestamp_date_range(cap_out,
                                    start_date_range, end_date_range)
                    if cap_out.empty is True:
                        logger.warning('No generation in selected Date Range')
                        continue

                # unitconversion based off peak outage hour, only checked once 
                if i == 0:
                    unitconversion = self.capacity_energy_unitconversion(cap_out,
                                                                            sum_values=True)
               
                cap_out = cap_out / unitconversion['divisor']

                scenario_names = pd.Series([scenario] * len(cap_out),name = 'Scenario')
                single_scen_out = cap_out.set_index([scenario_names],append = True)
                chunks.append(single_scen_out)
                
                mplt.stackplot(df=cap_out, color_dict=self.PLEXOS_color_dict, 
                                      labels=cap_out.columns, sub_pos=i)
                mplt.set_subplot_timeseries_format(sub_pos=i)
            
            if not chunks:
                outputs[zone_input] = MissingZoneData()
                continue
                
            Data_Table_Out = pd.concat(chunks,axis = 0)
            Data_Table_Out = Data_Table_Out.add_suffix(f" ({unitconversion['units']})")

            # Add legend
            mplt.add_legend(reverse_legend=True)
            plt.ylabel(f"Capacity out ({unitconversion['units']})",  color='black', 
                       rotation='vertical', labelpad=30)
            # Looks better for a one scenario plot
            #plt.tight_layout(rect=[0, 0.03, 1.25, 0.97])
            
            # plt.tight_layout(rect=[0,0.03,1,0.97])
            # Add title
            if plot_data_settings["plot_title_as_region"]:
                mplt.add_main_title(zone_input)

            outputs[zone_input] = {'fig': fig, 'data_table': Data_Table_Out}
        return outputs


    def capacity_out_stack_PASA(self, start: float = None, 
                                end: float= None, timezone: str = "", **_):
        
        outputs = UnderDevelopment()
        logger.warning('capacity_out_stack_PASA requires PASA files, and is under development. Skipping plot.')
        return outputs 
    
        outputs : dict = {}
        for zone_input in self.Zones:
            
            logger.info('Zone = ' + str(zone_input))

            ncols=len(self.xlabels)
            if ncols == 0:
                ncols = 1
            nrows=len(self.ylabels)
            if nrows == 0:
                nrows = 1
            grid_size = ncols*nrows
            fig1, axs = plt.subplots(nrows,ncols, figsize=((8*ncols),(4*nrows)), sharey=True)
            plt.subplots_adjust(wspace=0.05, hspace=0.2)
            if len(self.Multi_Scenario) > 1:
                axs = axs.ravel()
            i=0

            met_year = self.processed_hdf5_folder[-4:] #Extract met year from PLEXOS parent scenario.

            for scenario in self.Multi_Scenario:
                logger.info("Scenario = " + str(scenario))

                infra_year = scenario[-4:] #Extract infra year from scenario name.
                capacity_out = pd.read_csv(os.path.join('/projects/continental/pcm/Outage Profiles/capacity out for plotting/',infra_year + '_' + met_year + '_capacity out.csv'))
                capacity_out.index = pd.to_datetime(capacity_out.DATETIME)
                one_zone = capacity_out[capacity_out[self.AGG_BY] == zone_input]    #Select only this particular zone.
                one_zone = one_zone.drop(columns = ['DATETIME',self.AGG_BY])
                one_zone = one_zone.dropna(axis = 'columns')
                one_zone = one_zone / 1000 #MW -> GW
                #Calculate average outage for all technology for all year.
                sum_ts = one_zone.sum(axis = 'columns')
                overall_avg = sum_ts.mean()

               #Subset to match dispatch time horizon.
                Gen = pd.read_hdf(os.path.join(self. Marmot_Solutions_folder, scenario, "Processed_HDF5_folder", scenario+"_formatted.h5"),  "generator_Generation")
                start = Gen.index.get_level_values('timestamp')[0]
                end =  Gen.index.get_level_values('timestamp')[-1]

               #OR select only time period of interest.
                if self.prop == 'Date Range':
                    logger.info("Plotting specific date range: \
                    {} to {}".format(str(self.start_date),str(self.end_date)))
                    one_zone = one_zone[self.start_date : self.end_date]
                else:
                    one_zone = one_zone[start:end]

                tech_list = [tech_type for tech_type in self.ordered_gen if tech_type in one_zone.columns]  #Order columns.
                one_zone = one_zone[tech_list]

                overall_avg_vec = pd.DataFrame(np.repeat(np.array(overall_avg),len(one_zone.index)), index = one_zone.index, columns = ['Annual average'])
                overall_avg_vec = overall_avg_vec / 1000
                Data_Table_Out = pd.concat([one_zone,overall_avg_vec], axis = 'columns')

                locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
                formatter = mdates.ConciseDateFormatter(locator)
                formatter.formats[2] = '%d\n %b'
                formatter.zero_formats[1] = '%b\n %Y'
                formatter.zero_formats[2] = '%d\n %b'
                formatter.zero_formats[3] = '%H:%M\n %d-%b'
                formatter.offset_formats[3] = '%b %Y'
                formatter.show_offset = False

                if len(self.Multi_Scenario) > 1 :
                    sp = axs[i].stackplot(one_zone.index.values, one_zone.values.T, labels=one_zone.columns, linewidth=0,
                              colors=[self.PLEXOS_color_dict.get(x, '#333333') for x in one_zone.T.index])

                    axs[i].spines['right'].set_visible(False)
                    axs[i].spines['top'].set_visible(False)
                    axs[i].tick_params(axis='y', which='major', length=5, width=1)
                    axs[i].tick_params(axis='x', which='major', length=5, width=1)
                    axs[i].margins(x=0.01)
                    axs[i].xaxis.set_major_locator(locator)
                    axs[i].xaxis.set_major_formatter(formatter)
                    axs[i].hlines(y = overall_avg, xmin = axs[i].get_xlim()[0], xmax = axs[i].get_xlim()[1], label = 'Annual average') #Add horizontal line at average.
                    if i == (len(self.Multi_Scenario) - 1) :
                        handles, labels = axs[i].get_legend_handles_labels()
                        leg1 = axs[i].legend(reversed(handles), reversed(labels), loc='lower left',bbox_to_anchor=(1,0),facecolor='inherit', frameon=True)
                        axs[i].add_artist(leg1)
                else:
                    sp = axs.stackplot(one_zone.index.values, one_zone.values.T, labels=one_zone.columns, linewidth=0,
                              colors=[self.PLEXOS_color_dict.get(x, '#333333') for x in one_zone.T.index])

                    axs.spines['right'].set_visible(False)
                    axs.spines['top'].set_visible(False)
                    axs.tick_params(axis='y', which='major', length=5, width=1)
                    axs.tick_params(axis='x', which='major', length=5, width=1)
                    axs.margins(x=0.01)
                    axs.xaxis.set_major_locator(locator)
                    axs.xaxis.set_major_formatter(formatter)
                    axs.hlines(y = overall_avg, xmin = axs.get_xlim()[0], xmax = axs.get_xlim()[1], label = 'Annual average') #Add horizontal line at average.
                    if i == (len(self.Multi_Scenario) - 1) :
                        handles, labels = axs.get_legend_handles_labels()
                        leg1 = axs.legend(reversed(handles), reversed(labels), loc='lower left',bbox_to_anchor=(1,0),facecolor='inherit', frameon=True)
                        axs.add_artist(leg1)

                i = i + 1

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
            plt.xlabel(timezone,  color='black', rotation='horizontal', labelpad = 40)
            plt.ylabel('Capacity out (MW)',  color='black', rotation='vertical', labelpad = 60)
            if plot_data_settings["plot_title_as_region"]:
                mplt.add_main_title(zone_input)


            outputs[zone_input] = {'fig' : fig1, 'data_table' : Data_Table_Out}
        return outputs
