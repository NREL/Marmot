import os
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
import numpy as np
import logging
import plottingmodules.marmot_plot_functions as mfunc
import config.mconfig as mconfig

#===============================================================================

class mplot(object):
    def __init__(self, argument_dict):
        # iterate over items in argument_dict and set as properties of class
        # see key_list in Marmot_plot_main for list of properties
        for prop in argument_dict:
            self.__setattr__(prop, argument_dict[prop])
        self.logger = logging.getLogger('marmot_plot.'+__name__)
        self.x = mconfig.parser("figure_size","xdimension")
        self.y = mconfig.parser("figure_size","ydimension")

    def capacity_out_stack(self):
        
        outputs = {}
        installed_cap_collection = {}
        gen_available_capacity_collection = {}
        check_input_data = []
        
        check_input_data.extend([mfunc.get_data(installed_cap_collection,"generator_Installed_Capacity", self.Marmot_Solutions_folder, [self.Scenarios[0]])])
        check_input_data.extend([mfunc.get_data(gen_available_capacity_collection,"generator_Available_Capacity", self.Marmot_Solutions_folder, self.Scenarios)])
        
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

        grid_size = xdimension*ydimension

        # Used to calculate any excess axis to delete
        plot_number = len(self.Scenarios)
        excess_axs = grid_size - plot_number

        for zone_input in self.Zones:
            self.logger.info('Zone = ' + str(zone_input))

            fig2, axs = plt.subplots(ydimension,xdimension, figsize=((self.x*xdimension),(self.y*ydimension)), sharex = True, sharey='row',squeeze=False)
            plt.subplots_adjust(wspace=0.1, hspace=0.2)
            axs = axs.ravel()

            i = -1
            chunks = []
            
            for scenario in self.Scenarios:
                self.logger.info("Scenario = " + scenario)
                i += 1
                
                install_cap = installed_cap_collection.get(scenario).copy()
                avail_cap = gen_available_capacity_collection.get(scenario).copy()
                if self.shift_leapday:
                    avail_cap = mfunc.shift_leapday(avail_cap,self.Marmot_Solutions_folder)
                avail_cap = avail_cap.xs(zone_input,level=self.AGG_BY)
                avail_cap.columns = ['avail']
                install_cap.columns = ['cap']
                avail_cap.reset_index(inplace = True)
                
                cap_out = avail_cap.merge(install_cap,left_on = ['gen_name'],right_on = ['gen_name'])
                cap_out['Capacity out'] = cap_out['cap'] - cap_out['avail']
                
                cap_out = cap_out.groupby(["timestamp", "tech"], as_index=False).sum()
                cap_out.tech = cap_out.tech.astype("category")
                cap_out.tech.cat.set_categories(self.ordered_gen, inplace=True)
                cap_out = cap_out.sort_values(["tech"])
                cap_out = cap_out.pivot(index = 'timestamp', columns = 'tech', values = 'Capacity out')
                #Subset only thermal gen categories
                thermal_gens = [therm for therm in self.thermal_gen_cat if therm in cap_out.columns]
                cap_out = cap_out[thermal_gens]

                # unitconversion based off peak outage hour, only checked once 
                if i == 0:
                    unitconversion = mfunc.capacity_energy_unitconversion(max(cap_out.sum(axis=1)))
               
                cap_out = cap_out / unitconversion['divisor']

                scenario_names = pd.Series([scenario] * len(cap_out),name = 'Scenario')
                single_scen_out = cap_out.set_index([scenario_names],append = True)
                chunks.append(single_scen_out)
                        
                mfunc.create_stackplot(axs = axs, data = cap_out,color_dict = self.PLEXOS_color_dict, label = cap_out.columns, n = i)
                mfunc.set_plot_timeseries_format(axs, n = i, minticks = self.minticks, maxticks = self.maxticks)
                axs[i].legend(loc = 'lower left',bbox_to_anchor=(1.05,0),facecolor='inherit', frameon=True)

            Data_Table_Out = pd.concat(chunks,axis = 1)

            fig2.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            plt.ylabel('Capacity out ({})'.format(unitconversion['units']),  color='black', rotation='vertical', labelpad=30)
            plt.tight_layout(rect=[0, 0.03, 1, 0.97])
            outputs[zone_input] = {'fig': fig2, 'data_table': Data_Table_Out}
        return outputs


    def capacity_out_stack_PASA(self):
        outputs = mfunc.UnderDevelopment()
        self.logger.warning('capacity_out_stack_PASA requires PASA files, and is under development. Skipping plot.')
        return outputs 
    
        outputs = {}
        for zone_input in self.Zones:
            
            self.logger.info('Zone = ' + str(zone_input))

            xdimension=len(self.xlabels)
            if xdimension == 0:
                xdimension = 1
            ydimension=len(self.ylabels)
            if ydimension == 0:
                ydimension = 1
            grid_size = xdimension*ydimension
            fig1, axs = plt.subplots(ydimension,xdimension, figsize=((8*xdimension),(4*ydimension)), sharey=True)
            plt.subplots_adjust(wspace=0.05, hspace=0.2)
            if len(self.Multi_Scenario) > 1:
                axs = axs.ravel()
            i=0

            met_year = self.Marmot_Solutions_folder[-4:] #Extract met year from PLEXOS parent scenario.

            for scenario in self.Multi_Scenario:
                self.logger.info("Scenario = " + str(scenario))

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
                    self.logger.info("Plotting specific date range: \
                    {} to {}".format(str(self.start_date),str(self.end_date)))
                    one_zone = one_zone[self.start_date : self.end_date]
                else:
                    one_zone = one_zone[start:end]

                tech_list = [tech_type for tech_type in self.ordered_gen if tech_type in one_zone.columns]  #Order columns.
                one_zone = one_zone[tech_list]

                if '2008' not in self.Marmot_Solutions_folder and '2012' not in self.Marmot_Solutions_folder and one_zone.index[0] > dt.datetime(2024,2,28,0,0):
                    one_zone.index = one_zone.index.shift(1,freq = 'D') #TO DEAL WITH LEAP DAYS, SPECIFIC TO MARTY'S PROJECT, REMOVE AFTER.

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
                    axs[i].yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
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
                    axs.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
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
            plt.xlabel('Date ' + '(' + self.timezone + ')',  color='black', rotation='horizontal', labelpad = 40)
            plt.ylabel('Capacity out (MW)',  color='black', rotation='vertical', labelpad = 60)

           #fig1.savefig('/home/mschwarz/PLEXOS results analysis/test/PJM_outages_2024_2011_test', dpi=600, bbox_inches='tight') #Test

            outputs[zone_input] = {'fig' : fig1, 'data_table' : Data_Table_Out}
        return outputs
