
"""
Created on Mon Dec  9 10:34:48 2019
This code creates generation stack plots and is called from Marmot_plot_main.py
@author: dlevie
"""

import pandas as pd

import datetime as dt
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
import os
from matplotlib.patches import Patch

#===============================================================================

def df_process_gen_inputs(df, self):
    df = df.reset_index(['timestamp','tech'])
    df = df.groupby(["timestamp", "tech"], as_index=False).sum()
    df.tech = df.tech.astype("category")
    df.tech.cat.set_categories(self.ordered_gen, inplace=True)
    df = df.sort_values(["tech"])
    df = df.pivot(index='timestamp', columns='tech', values=0)

    return df

custom_legend_elements = [Patch(facecolor='#DD0200',
                            alpha=0.5, edgecolor='#DD0200',
                         label='Unserved Energy')]

class mplot(object):

    def __init__(self,argument_list):

        self.prop = argument_list[0]
        self.start = argument_list[1]
        self.end = argument_list[2]
        self.timezone = argument_list[3]
        self.start_date = argument_list[4]
        self.end_date = argument_list[5]
        self.hdf_out_folder = argument_list[6]
        self.Zones =argument_list[7]
        self.AGG_BY = argument_list[8]
        self.ordered_gen = argument_list[9]
        self.PLEXOS_color_dict = argument_list[10]
        self.Multi_Scenario = argument_list[11]
        self.Scenario_Diff = argument_list[12]
        self.Marmot_Solutions_folder = argument_list[13]
        self.ylabels = argument_list[14]
        self.xlabels = argument_list[15]
        self.gen_names_dict = argument_list[18]
        self.vre_gen_cat = argument_list[21]
        self.figure_folder = argument_list[25]
        self.facet = argument_list[27]

###############################################################################

    def gen_stack(self):
        # Create a dictionary to hold Dataframes
        Gen_Collection = {}
        Load_Collection = {}
        Pump_Load_Collection = {}
        Unserved_Energy_Collection = {}
        Curtailment_Collection = {}

        def set_dicts(scenario):
            Gen_Collection[scenario] = pd.read_hdf(os.path.join(self.Marmot_Solutions_folder, scenario, "Processed_HDF5_folder", scenario + "_formatted.h5"), "generator_Generation")
            Curtailment_Collection[scenario] = pd.read_hdf(os.path.join(self.Marmot_Solutions_folder, scenario, "Processed_HDF5_folder", scenario + "_formatted.h5"),  "generator_Curtailment")
            try:
                Pump_Load_Collection[scenario] = pd.read_hdf(os.path.join(self.Marmot_Solutions_folder, scenario, "Processed_HDF5_folder", scenario+"_formatted.h5"),  "generator_Pump_Load" )
            except:
                Pump_Load_Collection[scenario] = Gen_Collection[scenario].copy()
                Pump_Load_Collection[scenario].iloc[:,0] = 0
            if self.AGG_BY == "zone":
                Load_Collection[scenario] = pd.read_hdf(os.path.join(self.Marmot_Solutions_folder, scenario, "Processed_HDF5_folder", scenario + "_formatted.h5"),  "zone_Load")
                try:
                    Unserved_Energy_Collection[scenario] = pd.read_hdf(os.path.join(self.Marmot_Solutions_folder, scenario, "Processed_HDF5_folder", scenario + "_formatted.h5"), "zone_Unserved_Energy" )
                except:
                    Unserved_Energy_Collection[scenario] = Load_Collection[scenario].copy()
                    Unserved_Energy_Collection[scenario].iloc[:,0] = 0
            else:
                Load_Collection[scenario] = pd.read_hdf(os.path.join(self.Marmot_Solutions_folder, scenario, "Processed_HDF5_folder", scenario + "_formatted.h5"),  "region_Load")
                try:
                    Unserved_Energy_Collection[scenario] = pd.read_hdf(os.path.join(self.Marmot_Solutions_folder, scenario, "Processed_HDF5_folder", scenario + "_formatted.h5"), "region_Unserved_Energy" )
                except:
                    Unserved_Energy_Collection[scenario] = Load_Collection[scenario].copy()
                    Unserved_Energy_Collection[scenario].iloc[:,0] = 0

        def setup_data(zone_input, scenario, Stacked_Gen):
            try:
                Stacked_Curt = Curtailment_Collection.get(scenario)
                Stacked_Curt = Stacked_Curt.xs(zone_input,level=self.AGG_BY)
                Stacked_Curt = df_process_gen_inputs(Stacked_Curt, self)
                Stacked_Curt = Stacked_Curt.sum(axis=1)
                Stacked_Curt[Stacked_Curt<0.05] = 0 #Remove values less than 0.05 MW
                Stacked_Gen.insert(len(Stacked_Gen.columns),column='Curtailment',value=Stacked_Curt) #Insert curtailment into
            except KeyError:
                pass

            # Calculates Net Load by removing variable gen + curtailment
            self.vre_gen_cat = self.vre_gen_cat + ['Curtailment']
            # Adjust list of values to drop depending on if it exhists in Stacked_Gen df
            self.vre_gen_cat = [name for name in self.vre_gen_cat if name in Stacked_Gen.columns]
            Net_Load = Stacked_Gen.drop(labels = self.vre_gen_cat, axis=1)
            Net_Load = Net_Load.sum(axis=1)

            # Removes columns that only contain 0
            Stacked_Gen = Stacked_Gen.loc[:, (Stacked_Gen != 0).any(axis=0)]
            Stacked_Gen = Stacked_Gen #/ 1000 #MW -> GW

            Load = Load_Collection.get(scenario)
            Load = Load.xs(zone_input,level=self.AGG_BY)
            Load = Load.groupby(["timestamp"]).sum()
            Load = Load.squeeze() #Convert to Series
            Load = Load #/ 1000 #MW -> GW

            Pump_Load = Pump_Load_Collection.get(scenario)
            Pump_Load = Pump_Load.xs(zone_input,level=self.AGG_BY)
            Pump_Load = Pump_Load.groupby(["timestamp"]).sum()
            Pump_Load = Pump_Load.squeeze() #Convert to Series
            Pump_Load = Pump_Load #/ 1000 #MW -> GW
            if (Pump_Load == 0).all() == False:
                Total_Demand = Load - Pump_Load
            else:
                Total_Demand = Load

            Unserved_Energy = Unserved_Energy_Collection.get(scenario)
            Unserved_Energy = Unserved_Energy.xs(zone_input,level=self.AGG_BY)
            Unserved_Energy = Unserved_Energy.groupby(["timestamp"]).sum()
            Unserved_Energy = Unserved_Energy.squeeze() #Convert to Series

            unserved_eng_data_table = Unserved_Energy # Used for output to data table csv
            if (Unserved_Energy == 0).all() == False:
                Unserved_Energy = Load - Unserved_Energy

            data = {"Stacked_Gen":Stacked_Gen, "Load":Load, "Net_Load":Net_Load, "Pump_Load":Pump_Load, "Total_Demand":Total_Demand, "Unserved_Energy":Unserved_Energy,"ue_data_table":unserved_eng_data_table}
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
            Peak_Demand = None
            min_net_load_t = None
            Min_Net_Load = None

            if self.prop == "Peak Demand":

                peak_demand_t = Total_Demand.idxmax()
                end_date = peak_demand_t + dt.timedelta(days=self.end)
                start_date = peak_demand_t - dt.timedelta(days=self.start)
                Peak_Demand = Total_Demand[peak_demand_t]
                Stacked_Gen = Stacked_Gen[start_date : end_date]
                Load = Load[start_date : end_date]
                Unserved_Energy = Unserved_Energy[start_date : end_date]
                Total_Demand = Total_Demand[start_date : end_date]

                unserved_eng_data_table = unserved_eng_data_table[start_date : end_date]

            elif self.prop == "Min Net Load":
                min_net_load_t = Net_Load.idxmin()
                end_date = min_net_load_t + dt.timedelta(days=self.end)
                start_date = min_net_load_t - dt.timedelta(days=self.start)
                Min_Net_Load = Net_Load[min_net_load_t]
                Stacked_Gen = Stacked_Gen[start_date : end_date]
                Load = Load[start_date : end_date]
                Unserved_Energy = Unserved_Energy[start_date : end_date]
                Total_Demand = Total_Demand[start_date : end_date]

                unserved_eng_data_table = unserved_eng_data_table[start_date : end_date]

            elif self.prop == 'Date Range':
                ("Plotting specific date range:")
                print(str(self.start_date) + '  to  ' + str(self.end_date))

                Stacked_Gen = Stacked_Gen[self.start_date : self.end_date]
                Load = Load[self.start_date : self.end_date]
                Unserved_Energy = Unserved_Energy[self.start_date : self.end_date]
                Total_Demand = Total_Demand[self.start_date : self.end_date]
                unserved_eng_data_table = unserved_eng_data_table[self.start_date : self.end_date]

            else:
                print("Plotting graph for entire timeperiod")
            data = {"Stacked_Gen":Stacked_Gen, "Load":Load, "Pump_Load":Pump_Load, "Total_Demand":Total_Demand, "Unserved_Energy":Unserved_Energy,"ue_data_table":unserved_eng_data_table}
            data["peak_demand_t"] = peak_demand_t
            data["Peak_Demand"] = Peak_Demand
            data["min_net_load_t"] = min_net_load_t
            data["Min_Net_Load"] = Min_Net_Load
            return data

        def mkplot(outputs, zone_input, all_scenarios):
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

            fig1, axs = plt.subplots(ydimension,xdimension, figsize=((6*xdimension),(4*ydimension)), sharey=True, squeeze=False)
            plt.subplots_adjust(wspace=0.05, hspace=0.2)
            axs = axs.ravel()
            i=0
            data_tables = {}
            for scenario in all_scenarios:
                print("Scenario = " + scenario)

                try:
                    Stacked_Gen = Gen_Collection.get(scenario)
                    Stacked_Gen = Stacked_Gen.xs(zone_input,level=self.AGG_BY)
                except KeyError:
                    i=i+1
                    print('No generation in ' + zone_input + '\n')
                    out = pd.DataFrame()
                    return out

                Stacked_Gen = df_process_gen_inputs(Stacked_Gen, self)
                data = setup_data(zone_input, scenario, Stacked_Gen)
                
                # if no Generation return empty dataframe
                if data["Stacked_Gen"].empty == True:
                    print('No generation during time period in ' + zone_input + '\n')
                    out = pd.DataFrame()
                    return out
                
                data = data_prop(data)

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

                Load = Load.rename('Total Load (Demand + Storage Charging)')
                Total_Demand = Total_Demand.rename('Total Demand')
                unserved_eng_data_table = unserved_eng_data_table.rename("Unserved Energy")
                # Data table of values to return to main program
                Data_Table_Out = pd.concat([Load, Total_Demand, unserved_eng_data_table, Stacked_Gen], axis=1, sort=False)
                data_tables[scenario] = Data_Table_Out
                

                # only difference linewidth = 0,5
                axs[i].stackplot(Stacked_Gen.index.values, Stacked_Gen.values.T, labels=Stacked_Gen.columns, linewidth=0,
                      colors=[self.PLEXOS_color_dict.get(x, '#333333') for x in Stacked_Gen.T.index])


                if (Unserved_Energy == 0).all() == False:
                    axs[i].plot(Unserved_Energy,
                                      #color='#EE1289'  OLD MARMOT COLOR
                                      color = '#DD0200' #SEAC STANDARD COLOR (AS OF MARCH 9, 2020)
                                      )

                lp = axs[i].plot(Load, color='black')

                if (Pump_Load == 0).all() == False:
                    lp3 = axs[i].plot(Total_Demand, color='black', linestyle="--")

                # ax.set_ylabel('Generation (MW)',  color='black', rotation='vertical')
                # ax.set_xlabel('Date ' + '(' + self.timezone + ')',  color='black', rotation='horizontal')
                axs[i].spines['right'].set_visible(False)
                axs[i].spines['top'].set_visible(False)
                axs[i].tick_params(axis='y', which='major', length=5, width=1)
                axs[i].tick_params(axis='x', which='major', length=5, width=1)
                axs[i].yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
                axs[i].margins(x=0.01)

                if self.prop == "Min Net Load":
                    axs[i].annotate('Min Net Load: \n' + str(format(int(Min_Net_Load), ',')) + ' MW', xy=(min_net_load_t, Min_Net_Load), xytext=((min_net_load_t + dt.timedelta(days=0.1)), (Min_Net_Load + max(Load)/4)),
                        fontsize=13, arrowprops=dict(facecolor='black', width=3, shrink=0.1))

# Peak Demand label overlaps other labels on a facet plot
                elif self.prop == "Peak Demand":
                    axs[i].annotate('Peak Demand: \n' + str(format(int(Total_Demand[peak_demand_t]), ',')) + ' MW', xy=(peak_demand_t, Peak_Demand), xytext=((peak_demand_t + dt.timedelta(days=0.1)), (max(Total_Demand) + Total_Demand[peak_demand_t]*0.1)),
                                fontsize=13, arrowprops=dict(facecolor='black', width=3, shrink=0.1))


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

                if (Unserved_Energy == 0).all() == False:
                    axs[i].fill_between(Load.index, Load,Unserved_Energy,
                                        # facecolor='#EE1289' OLD MARMOT COLOR
                                        facecolor = '#DD0200', #SEAC STANDARD COLOR (AS OF MARCH 9, 2020)
                                        alpha=0.5)

                handles, labels = axs[grid_size-1].get_legend_handles_labels()

                #Legend 1
                leg1 = axs[grid_size-1].legend(reversed(handles), reversed(labels), loc='lower left',bbox_to_anchor=(1,0),
                              facecolor='inherit', frameon=True)
                #Legend 2
                if (Pump_Load == 0).all() == False:
                    leg2 = axs[grid_size-1].legend(lp, ['Demand + Storage Charging'], loc='upper left',bbox_to_anchor=(1, 1.2),
                              facecolor='inherit', frameon=True)
                else:
                    leg2 = axs[grid_size-1].legend(lp, ['Demand'], loc='upper left',bbox_to_anchor=(1, 1.2),
                              facecolor='inherit', frameon=True)

                #Legend 3
                if (Unserved_Energy == 0).all() == False:
                    leg3 = axs[grid_size-1].legend(handles=custom_legend_elements, loc='upper left',bbox_to_anchor=(1, 1.15),
                          facecolor='inherit', frameon=True)

                # Variable defined, but never used
                #Legend 4
                if (Pump_Load == 0).all() == False:
                    axs[grid_size-1].legend(lp3, ['Demand'], loc='upper left',bbox_to_anchor=(1, 1.1),
                              facecolor='inherit', frameon=True)

                # Manually add the first legend back
                axs[grid_size-1].add_artist(leg1)
                axs[grid_size-1].add_artist(leg2)
                if (Unserved_Energy == 0).all() == False:
                    axs[grid_size-1].add_artist(leg3)

                i=i+1

            print(" ")
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
            plt.ylabel('Generation (MW)',  color='black', rotation='vertical', labelpad=60)


            if not self.facet:
                data_tables = data_tables[self.Multi_Scenario[0]]
            out = {'fig':fig1, 'data_table':data_tables}

            plt.close(fig1)

            return out

# Main loop for gen_stack
        if self.facet:
            for scenario in self.Multi_Scenario:
                set_dicts(scenario)
        else:
            set_dicts(self.Multi_Scenario[0])

        outputs = {}
        for zone_input in self.Zones:
            print("Zone = "+ zone_input)
            #data_tables = {}
            if self.facet:
                #for scenario in self.Multi_Scenario:
                    #set_dicts(scenario)
                outputs[zone_input] = mkplot(outputs, zone_input, self.Multi_Scenario)

            else:
                #set_dicts(self.Multi_Scenario[0])
                outputs[zone_input] = mkplot(outputs, zone_input, [self.Multi_Scenario[0]])
        return outputs

###############################################################################

###############################################################################
    def gen_diff(self):
        outputs = {}
        for zone_input in self.Zones:
            print("Zone = "+ zone_input)
            # Create Dictionary to hold Datframes for each scenario
            Gen_Collection = {}

            for scenario in self.Scenario_Diff:
                Gen_Collection[scenario] = pd.read_hdf(os.path.join(self.Marmot_Solutions_folder, scenario, "Processed_HDF5_folder", scenario + "_formatted.h5"), "generator_Generation")


            Total_Gen_Stack_1 = Gen_Collection.get(self.Scenario_Diff[0])
            Total_Gen_Stack_1 = Total_Gen_Stack_1.xs(zone_input,level=self.AGG_BY)
            Total_Gen_Stack_1 = df_process_gen_inputs(Total_Gen_Stack_1, self)
            #Adds in all possible columns from ordered gen to ensure the two dataframes have same column names
            Total_Gen_Stack_1 = pd.DataFrame(Total_Gen_Stack_1, columns = self.ordered_gen).fillna(0)

            Total_Gen_Stack_2 = Gen_Collection.get(self.Scenario_Diff[1])
            Total_Gen_Stack_2 = Total_Gen_Stack_2.xs(zone_input,level=self.AGG_BY)
            Total_Gen_Stack_2 = df_process_gen_inputs(Total_Gen_Stack_2, self)
            #Adds in all possible columns from ordered gen to ensure the two dataframes have same column names
            Total_Gen_Stack_2 = pd.DataFrame(Total_Gen_Stack_2, columns = self.ordered_gen).fillna(0)

            print('Scenario 1 = ' + self.Scenario_Diff[0])
            print('Scenario 2 =  ' + self.Scenario_Diff[1])
            Gen_Stack_Out = Total_Gen_Stack_1-Total_Gen_Stack_2
            # Removes columns that only equal 0
            Gen_Stack_Out.dropna(inplace=True)
            Gen_Stack_Out = Gen_Stack_Out.loc[:, (Gen_Stack_Out != 0).any(axis=0)]

            # Data table of values to return to main program
            Data_Table_Out = Gen_Stack_Out
            # Reverses order of columns
            Gen_Stack_Out = Gen_Stack_Out.iloc[:, ::-1]

            fig3, ax = plt.subplots(figsize=(9,6))

            for column in Gen_Stack_Out:
                ax.plot(Gen_Stack_Out[column], linewidth=3, color=self.PLEXOS_color_dict[column],
                        label=column)
                ax.legend(loc='lower left',bbox_to_anchor=(1,0),
                              facecolor='inherit', frameon=True)


            ax.set_title(self.Scenario_Diff[0].replace('_', ' ') + " vs. " + self.Scenario_Diff[1].replace('_', ' '))
            ax.set_ylabel('Generation Difference (MW)',  color='black', rotation='vertical')
            ax.set_xlabel('Date ' + '(' + self.timezone + ')',  color='black', rotation='horizontal')
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.tick_params(axis='y', which='major', length=5, width=1)
            ax.tick_params(axis='x', which='major', length=5, width=1)
            ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
            ax.margins(x=0.01)

            locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
            formatter = mdates.ConciseDateFormatter(locator)
            formatter.formats[2] = '%d\n %b'
            formatter.zero_formats[1] = '%b\n %Y'
            formatter.zero_formats[2] = '%d\n %b'
            formatter.zero_formats[3] = '%H:%M\n %d-%b'
            formatter.offset_formats[3] = '%b %Y'
            formatter.show_offset = False
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)

            outputs[zone_input] = {'fig': fig3, 'data_table': Data_Table_Out}
        return outputs

    def gen_stack_all_periods(self):
        #Location to save to
        gen_stack_figures = os.path.join(self.figure_folder, self.AGG_BY + '_Gen_Stack')

        Stacked_Gen_read = pd.read_hdf(self.hdf_out_folder + "/" + self.Multi_Scenario[0]+"_formatted.h5", 'generator_Generation')
        try:
            Pump_Load_read =pd.read_hdf(self.hdf_out_folder + "/" + self.Multi_Scenario[0]+"_formatted.h5", "generator_Pump_Load" )
        except:
            Pump_Load_read = Stacked_Gen_read.copy()
            Pump_Load_read.iloc[:,0] = 0
        Stacked_Curt_read = pd.read_hdf(self.hdf_out_folder + "/" + self.Multi_Scenario[0]+"_formatted.h5", "generator_Curtailment" )

        # If data is to be aggregated by zone, then zone properties are loaded, else region properties are loaded
        if self.AGG_BY == "zone":
            Load_read = pd.read_hdf(self.hdf_out_folder + "/" + self.Multi_Scenario[0]+"_formatted.h5", "zone_Load")
            try:
                Unserved_Energy_read = pd.read_hdf(self.hdf_out_folder + "/" + self.Multi_Scenario[0]+"_formatted.h5", "zone_Unserved_Energy" )
            except:
                Unserved_Energy_read = Load_read.copy()
                Unserved_Energy_read.iloc[:,0] = 0
        else:
            Load_read = pd.read_hdf(self.hdf_out_folder + "/" + self.Multi_Scenario[0]+"_formatted.h5", "region_Load")
            try:
                Unserved_Energy_read = pd.read_hdf(self.hdf_out_folder + "/" + self.Multi_Scenario[0]+"_formatted.h5", "region_Unserved_Energy" )
            except:
                Unserved_Energy_read = Load_read.copy()
                Unserved_Energy_read.iloc[:,0] = 0

        outputs = {}
        for zone_input in self.Zones:

            print("Zone = "+ zone_input)


           # try:   #The rest of the function won't work if this particular zone can't be found in the solution file (e.g. if it doesn't include Mexico)
            Stacked_Gen = Stacked_Gen_read.xs(zone_input,level=self.AGG_BY)
            del Stacked_Gen_read
            Stacked_Gen = df_process_gen_inputs(Stacked_Gen, self)

            try:
                Stacked_Curt = Stacked_Curt_read.xs(zone_input,level=self.AGG_BY)
                del Stacked_Curt_read
                Stacked_Curt = df_process_gen_inputs(Stacked_Curt, self)
                Stacked_Curt = Stacked_Curt.sum(axis=1)
                Stacked_Curt[Stacked_Curt<0.05] = 0 #Remove values less than 0.05 MW
                Stacked_Gen.insert(len(Stacked_Gen.columns),column='Curtailment',value=Stacked_Curt) #Insert curtailment into
            except Exception:
                pass

            # Calculates Net Load by removing variable gen + curtailment
            self.vre_gen_cat = self.vre_gen_cat + ['Curtailment']
            # Adjust list of values to drop depending on if it exhists in Stacked_Gen df
            self.vre_gen_cat = [name for name in self.vre_gen_cat if name in Stacked_Gen.columns]
            Net_Load = Stacked_Gen.drop(labels = self.vre_gen_cat, axis=1)
            Net_Load = Net_Load.sum(axis=1)

            # Removes columns that only contain 0
            Stacked_Gen = Stacked_Gen.loc[:, (Stacked_Gen != 0).any(axis=0)]

            Load = Load_read.xs(zone_input,level=self.AGG_BY)
            del Load_read
            Load = Load.groupby(["timestamp"]).sum()
            Load = Load.squeeze() #Convert to Series

            Pump_Load = Pump_Load_read.xs(zone_input,level=self.AGG_BY)
            del Pump_Load_read
            Pump_Load = Pump_Load.groupby(["timestamp"]).sum()
            Pump_Load = Pump_Load.squeeze() #Convert to Series
            if (Pump_Load == 0).all() == False:
                Total_Demand = Load - Pump_Load
            else:
                Total_Demand = Load

            Unserved_Energy = Unserved_Energy_read.xs(zone_input,level=self.AGG_BY)
            del Unserved_Energy_read
            Unserved_Energy = Unserved_Energy.groupby(["timestamp"]).sum()
            Unserved_Energy = Unserved_Energy.squeeze() #Convert to Series
            unserved_eng_data_table = Unserved_Energy # Used for output to data table csv
            if (Unserved_Energy == 0).all() == False:
                Unserved_Energy = Load - Unserved_Energy

            Load = Load.rename('Total Load (Demand + Storage Charging)')
            Total_Demand = Total_Demand.rename('Total Demand')
            unserved_eng_data_table = unserved_eng_data_table.rename("Unserved Energy")


            first_date=Stacked_Gen.index[0]
            for wk in range(1,53): #assumes weekly, could be something else if user changes self.end Marmot_plot_select

                period_start=first_date+dt.timedelta(days=(wk-1)*7)
                period_end=period_start+dt.timedelta(days=self.end)
                print(str(period_start)+" and next "+str(self.end)+" days.")
                Stacked_Gen_Period = Stacked_Gen[period_start:period_end]
                Load_Period = Load[period_start:period_end]
                Unserved_Energy_Period = Unserved_Energy[period_start:period_end]
                Total_Demand_Period = Total_Demand[period_start:period_end]
                unserved_eng_data_table_period = unserved_eng_data_table[period_start:period_end]


                # Data table of values to return to main program
                Data_Table_Out = pd.concat([Load_Period, Total_Demand_Period, unserved_eng_data_table_period, Stacked_Gen_Period], axis=1, sort=False)

                fig1, ax = plt.subplots(figsize=(9,6))
                ax.stackplot(Stacked_Gen_Period.index.values, Stacked_Gen_Period.values.T, labels=Stacked_Gen_Period.columns, linewidth=5,colors=[self.PLEXOS_color_dict.get(x, '#333333') for x in Stacked_Gen_Period.T.index])

                if (Unserved_Energy_Period == 0).all() == False:
                    plt.plot(Unserved_Energy_Period,
                                   #color='#EE1289'  OLD MARMOT COLOR
                                   color = '#DD0200' #SEAC STANDARD COLOR (AS OF MARCH 9, 2020)
                                   )

                lp1 = plt.plot(Load_Period, color='black')

                if (Pump_Load == 0).all() == False:
                    lp3 = plt.plot(Total_Demand_Period, color='black', linestyle="--")


                ax.set_ylabel('Generation (MW)',  color='black', rotation='vertical')
                ax.set_xlabel('Date ' + '(' + str(self.timezone) + ')',  color='black', rotation='horizontal')
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.tick_params(axis='y', which='major', length=5, width=1)
                ax.tick_params(axis='x', which='major', length=5, width=1)
                ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
                ax.margins(x=0.01)

                locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
                formatter = mdates.ConciseDateFormatter(locator)
                formatter.formats[2] = '%d\n %b'
                formatter.zero_formats[1] = '%b\n %Y'
                formatter.zero_formats[2] = '%d\n %b'
                formatter.zero_formats[3] = '%H:%M\n %d-%b'
                formatter.offset_formats[3] = '%b %Y'
                formatter.show_offset = False
                ax.xaxis.set_major_locator(locator)
                ax.xaxis.set_major_formatter(formatter)


                if (Unserved_Energy_Period == 0).all() == False:
                    ax.fill_between(Load_Period.index, Load_Period,Unserved_Energy_Period,
                                    #facecolor='#EE1289'
                                    facecolor = '#DD0200',
                                    alpha=0.5)

                handles, labels = ax.get_legend_handles_labels()


                #Legend 1
                leg1 = ax.legend(reversed(handles), reversed(labels), loc='lower left',bbox_to_anchor=(1,0),
                              facecolor='inherit', frameon=True)
                #Legend 2
                if (Pump_Load == 0).all() == False:
                    leg2 = ax.legend(lp1, ['Demand + Storage Charging'], loc='center left',bbox_to_anchor=(1, 0.9),
                              facecolor='inherit', frameon=True)
                else:
                    leg2 = ax.legend(lp1, ['Demand'], loc='center left',bbox_to_anchor=(1, 0.9),
                              facecolor='inherit', frameon=True)

                #Legend 3
                if (Unserved_Energy_Period == 0).all() == False:
                    leg3 = ax.legend(handles=custom_legend_elements, loc='upper left',bbox_to_anchor=(1, 0.82),
                              facecolor='inherit', frameon=True)

                #Legend 4
                if (Pump_Load == 0).all() == False:
                    ax.legend(lp3, ['Demand'], loc='upper left',bbox_to_anchor=(1, 0.885),
                              facecolor='inherit', frameon=True)

                # Manually add the first legend back
                ax.add_artist(leg1)
                ax.add_artist(leg2)
                if (Unserved_Energy_Period == 0).all() == False:
                    ax.add_artist(leg3)


                fig1.savefig(os.path.join(gen_stack_figures, zone_input + "_" + "Stacked_Gen_All_Periods" + "_" + self.Multi_Scenario[0]+"_period_"+str(wk)), dpi=600, bbox_inches='tight')
                Data_Table_Out.to_csv(os.path.join(gen_stack_figures, zone_input + "_" + "Stacked_Gen_All_Periods" + "_" + self.Multi_Scenario[0]+"_period_"+str(wk)+ ".csv"))
                del fig1
                del Data_Table_Out
                mpl.pyplot.close('all')

            outputs[zone_input] = pd.DataFrame()
        #end weekly loop
        return outputs

    def committed_stack(self):
        outputs = {}
        for zone_input in self.Zones:
            print('Zone = ' + str(zone_input))

            #Get technology list.
            gens = pd.read_hdf(os.path.join(self.Marmot_Solutions_folder, self.Multi_Scenario[0], "Processed_HDF5_folder", self.Multi_Scenario[0] + "_formatted.h5"), "generator_Installed_Capacity")
            tech_list = list(gens.reset_index().tech.unique())
            tech_list_sort = [tech_type for tech_type in self.ordered_gen if tech_type in tech_list and tech_type in self.facet_gen_cat]

            xdimension = len(self.Multi_Scenario)
            ydimension = len(tech_list_sort)

            fig4, axs = plt.subplots(ydimension,xdimension, figsize=((8*xdimension),(4*ydimension)), sharex = True, sharey='row')
            plt.subplots_adjust(wspace=0.1, hspace=0.2)

            i=0

            for scenario in self.Multi_Scenario:
                print("Scenario = " + scenario)

                locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
                formatter = mdates.ConciseDateFormatter(locator)
                formatter.formats[2] = '%d\n %b'
                formatter.zero_formats[1] = '%b\n %Y'
                formatter.zero_formats[2] = '%d\n %b'
                formatter.zero_formats[3] = '%H:%M\n %d-%b'
                formatter.offset_formats[3] = '%b %Y'
                formatter.show_offset = False

                gen = pd.read_hdf(os.path.join(self.Marmot_Solutions_folder, scenario, "Processed_HDF5_folder", scenario + "_formatted.h5"), "generator_Generation")
                units_gen = pd.read_hdf(os.path.join(self.Marmot_Solutions_folder, scenario, "Processed_HDF5_folder", scenario + "_formatted.h5"), "generator_Units_Generating")
                avail_cap = pd.read_hdf(os.path.join(self.Marmot_Solutions_folder, scenario, "Processed_HDF5_folder", scenario + "_formatted.h5"), "generator_Available_Capacity")

                #Calculate  committed cap (for thermal only).
                thermal_commit_cap = units_gen * avail_cap
                thermal_commit_cap = thermal_commit_cap.xs(zone_input,level = self.AGG_BY)
                thermal_commit_cap = df_process_gen_inputs(thermal_commit_cap,self)
                thermal_commit_cap = thermal_commit_cap.loc[:, (thermal_commit_cap != 0).any(axis=0)]
                thermal_commit_cap = thermal_commit_cap / 1000 #MW -> GW

                #Process generation.
                gen = gen.xs(zone_input,level = self.AGG_BY)
                gen = df_process_gen_inputs(gen,self)
                gen = gen.loc[:, (gen != 0).any(axis=0)]
                gen = gen / 1000

                #Process available capacity (for VG only).
                avail_cap = avail_cap.xs(zone_input, level = self.AGG_BY)
                avail_cap = df_process_gen_inputs(avail_cap,self)
                avail_cap = avail_cap.loc[:, (avail_cap !=0).any(axis=0)]
                avail_cap = avail_cap / 1000

                j = 0
                gen_lines = []
                for tech in tech_list_sort:
                    if tech not in gen.columns:
                        gen_one_tech = pd.Series(0,index = gen.index)
                        commit_cap = pd.Series(0,index = gen.index) #Add dummy columns to deal with coal retirements (coal showing up in 2024, but not future years).
                    elif tech in self.thermal_gen_cat:
                        gen_one_tech = gen[tech]
                        commit_cap = thermal_commit_cap[tech]
                    else:
                        gen_one_tech = gen[tech]
                        commit_cap = avail_cap[tech]

                    gen_line = axs[j,i].plot(gen_one_tech,alpha = 0, color = self.PLEXOS_color_dict[tech])[0]
                    gen_lines.append(gen_line)
                    gen_fill = axs[j,i].fill_between(gen_one_tech.index,gen_one_tech,0, color = self.PLEXOS_color_dict[tech], alpha = 0.5)
                    if tech != 'Hydro':
                        cc = axs[j,i].plot(commit_cap, color = self.PLEXOS_color_dict[tech])

                    axs[j,i].spines['right'].set_visible(False)
                    axs[j,i].spines['top'].set_visible(False)
                    axs[j,i].tick_params(axis='y', which='major', length=5, width=1)
                    axs[j,i].tick_params(axis='x', which='major', length=5, width=1)
                    axs[j,i].yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.1f}'))
                    axs[j,i].margins(x=0.01)
                    axs[j,i].xaxis.set_major_locator(locator)
                    axs[j,i].xaxis.set_major_formatter(formatter)
                    if j == 0:
                        axs[j,i].set_xlabel(xlabel = scenario, color = 'black')
                        axs[j,i].xaxis.set_label_position('top')
                    if i == 0:
                        axs[j,i].set_ylabel(ylabel = tech, rotation = 'vertical', color = 'black')

                    j = j + 1
                i = i + 1

            #fig4.legend(gen_lines,labels = tech_list_sort, loc = 'right', title = 'RT Generation')
            fig4.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            plt.ylabel('Generation or Committed Capacity (GW)',  color='black', rotation='vertical', labelpad=60)

            data_table = pd.DataFrame()
            outputs[zone_input] = {'fig':fig4, 'data_table':data_table}
        return outputs

            # fig4.savefig('/home/mschwarz/PLEXOS results analysis/test/WI_commited_cap_2011heatwave_test.png', dpi=100, bbox_inches='tight') #Test
            # mpl.pyplot.close('all')
