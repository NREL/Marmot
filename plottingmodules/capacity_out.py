import os
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
import numpy as np



#===============================================================================

class mplot(object):
    def __init__(self, argument_dict):
        # iterate over items in argument_dict and set as properties of class
        # see key_list in Marmot_plot_main for list of properties
        for prop in argument_dict:
            self.__setattr__(prop, argument_dict[prop])

    def capacity_out_stack(self):
        outputs = {}
        for zone_input in self.Zones:
            print('Zone = ' + str(zone_input))

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
                print("Scenario = " + str(scenario))

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
                    print("Plotting specific date range:")
                    print(str(self.start_date) + '  to  ' + str(self.end_date))
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
