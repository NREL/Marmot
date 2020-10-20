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

    def __init__(self, argument_dict):
        # iterate over items in argument_dict and set as properties of class
        # see key_list in Marmot_plot_main for list of properties
        for prop in argument_dict:
            self.__setattr__(prop, argument_dict[prop])

    def _process_ts(self,df,zone_input):
        oz = df.xs(zone_input, level = self.AGG_BY)
        oz = oz.reset_index()
        oz = oz.groupby('timestamp').sum()
        return(oz)

    def sensitivities_gas(self):

        """
        This method highlights the difference in generation between two scenarios of a single resource. 
        The two scenarios are specified in the "Scenario_Diff_plot" field of Marmot_user_defined_inputs.csv
        The single resource is specfied in the "properties" field of Marmot_plot_select.csv.
        Blue hatches represent additional energy produced by the resource, and red hatches represent decreased energy.
        The difference in Gas-CC and Gas-CT generation, curtailment, and net interchange are also plotted.
        Each zone is plotted on a separate figure.
        Figures and data tables are returned to plot_main
        """

        outputs = {}

        bc = pd.read_hdf(os.path.join(self.Marmot_Solutions_folder, self.Scenario_Diff[0],"Processed_HDF5_folder", self.Scenario_Diff[0] + "_formatted.h5"),"generator_Generation")
        bc = mfunc.shift_leap_day(bc,self.Marmot_Solutions_folder,self.shift_leap_day)
        bc_tech = bc.xs(self.prop,level = 'tech')
        bc_CT = bc.xs('Gas-CT',level = 'tech')
        bc_CC = bc.xs('Gas-CC',level = 'tech')

        scen = pd.read_hdf(os.path.join(self.Marmot_Solutions_folder, self.Scenario_Diff[1],"Processed_HDF5_folder", self.Scenario_Diff[1] + "_formatted.h5"),"generator_Generation")
        scen = mfunc.shift_leap_day(scen,self.Marmot_Solutions_folder,self.shift_leap_day)
        scen_tech = scen.xs(self.prop,level = 'tech')
        scen_CT = scen.xs('Gas-CT',level = 'tech')
        scen_CC = scen.xs('Gas-CC',level = 'tech')

        curt_bc = pd.read_hdf(os.path.join(self.Marmot_Solutions_folder, self.Scenario_Diff[0], "Processed_HDF5_folder", self.Scenario_Diff[0] + "_formatted.h5"),  "generator_Curtailment")
        curt_scen = pd.read_hdf(os.path.join(self.Marmot_Solutions_folder, self.Scenario_Diff[1], "Processed_HDF5_folder", self.Scenario_Diff[1] + "_formatted.h5"),  "generator_Curtailment")
        curt_bc = self._shift_leap_day(curt_bc,self.Marmot_Solutions_folder,self.shift_leap_day)
        curt_scen = self._shift_leap_day(curt_scen,self.Marmot_Solutions_folder,self.shift_leap_day)
        curt_diff_all = curt_scen - curt_bc

        regions = list(bc.index.get_level_values(self.AGG_BY).unique())
        tech_regions = list(scen_tech.index.get_level_values(self.AGG_BY).unique()) 

        CT_diff_all = scen_CT - bc_CT
        CT_regions = list(CT_diff_all.index.get_level_values(self.AGG_BY).unique())
        CC_diff_all = scen_CC - bc_CC
        CC_regions = list(CC_diff_all.index.get_level_values(self.AGG_BY).unique())

        diff_csv = pd.DataFrame(index = bc_tech.index.get_level_values('timestamp').unique())
        diff_csv_perc = pd.DataFrame(index = bc_tech.index.get_level_values('timestamp').unique())

        #Add net interchange difference to icing plot.
        bc_int = pd.read_hdf(os.path.join(self.Marmot_Solutions_folder, self.Scenario_Diff[0],"Processed_HDF5_folder", self.Scenario_Diff[0] + "_formatted.h5"),"region_Net_Interchange")
        bc_int = shift_leap_day(bc_int,self.Marmot_Solutions_folder,self.shift_leap_day)
        scen_int = pd.read_hdf(os.path.join(self.Marmot_Solutions_folder, self.Scenario_Diff[1],"Processed_HDF5_folder", self.Scenario_Diff[1] + "_formatted.h5"),"region_Net_Interchange")
        scen_int = shift_leap_day(scen_int,self.Marmot_Solutions_folder,self.shift_leap_day)

        int_diff_all = scen_int - bc_int

        for zone_input in self.Zones:
            print(self.AGG_BY + " = " + zone_input)


            if zone_input not in regions or zone_input not in tech_regions:
                outputs[zone_input] = pd.DataFrame()

            else:

                oz_bc = self._process_ts(bc_tech,zone_input)
                oz_scen = self._process_ts(scen_tech,zone_input)
                icing_diff = oz_scen - oz_bc
                icing_diff_perc = 100 * icing_diff / oz_bc

                oz_bc.columns = [str(self.Scenario_Diff[0])] 
                oz_scen.columns = [str(self.Scenario_Diff[1])]
                Data_Table_Out = pd.concat([oz_bc,oz_scen],axis = 1)

                # icing_diff.columns = [zone_input]
                # icing_diff_perc.columns = [zone_input]  
                # diff_csv = pd.concat([diff_csv, icing_diff], axis = 1)  
                # diff_csv_perc = pd.concat([diff_csv_perc, icing_diff_perc], axis = 1)
                # continue

                diffs = Data_Table_Out.copy()

                curt_diff = curt_diff_all.xs(zone_input,level=self.AGG_BY)
                curt_diff = mfunc.df_process_gen_inputs(curt_diff)
                curt_diff = curt_diff.sum(axis=1)
                curt_diff.columns = ['Curtailment difference']

                int_diff_all = int_diff_all.reset_index()
                int_diff_all = self._merge_new_agg(int_diff_all)
                int_diff = int_diff_all[int_diff_all[self.AGG_BY] == zone_input]
                int_diff = int_diff.groupby('timestamp').sum()

                Data_Table_Out = pd.concat([Data_Table_Out,curt_diff], axis = 1)

                fig2, axs = mfunc.setup_plot()
                axs = axs[0]
                plt.subplots_adjust(wspace=0.05, hspace=0.2)

                if zone_input in CT_regions:
                    CT_diff = self._process_ts(CT_diff_all,zone_input)
                    CT_diff.columns = ['Gas-CT difference']
                    axs.plot(CT_diff,linewidth = 1, label = 'Gas-CT difference', color = self.PLEXOS_color_dict['Gas-CT'])
                    Data_Table_Out = pd.concat([Data_Table_Out,CT_diff],axis = 1)
                if zone_input in CC_regions:
                    CC_diff = self._process_ts(CC_diff_all,zone_input)
                    CC_diff.columns = ['Gas-CC difference']
                    axs.plot(CC_diff,linewidth = 1, label = 'Gas-CC difference',color = self.PLEXOS_color_dict['Gas-CC'])
                    Data_Table_Out = pd.concat([Data_Table_Out,CC_diff],axis = 1)

                axs.plot(oz_scen, linewidth = 1, label = self.Scenario_Diff[1],color = self.PLEXOS_color_dict[self.prop],linestyle = ':')
                axs.plot(oz_bc, linewidth = 1, label = self.prop + ' ' +  self.Scenario_Diff[0],color = self.PLEXOS_color_dict[self.prop])
                axs.plot(curt_diff, label = 'Curtailment difference', color = self.PLEXOS_color_dict['Curtailment'])
                axs.plot(int_diff, label = 'Net export difference', linestyle = '--')

                #Make two hatches: blue for when scenario > basecase, and red for when scenario < basecase.
                if self.Scenario_name != 'Icing' and self.Scenario_name != 'DryHydro':
                    axs.fill_between(diffs.index,diffs[str(self.Scenario_Diff[0])],diffs[str(self.Scenario_Diff[1])],
                        where = diffs[str(self.Scenario_Diff[1])] > diffs[str(self.Scenario_Diff[0])],
                        label = 'Increased ' + self.prop.lower() + ' generation', facecolor = 'blue', hatch = '///',alpha = 0.5)
                axs.fill_between(diffs.index,diffs[str(self.Scenario_Diff[0])],diffs[str(self.Scenario_Diff[1])],
                    where = diffs[str(self.Scenario_Diff[1])] < diffs[str(self.Scenario_Diff[0])],
                    label = 'Decreased ' + self.prop.lower() + ' generation', facecolor = 'red', hatch = '///',alpha = 0.5)
                axs.hlines(y = 0, xmin = axs.get_xlim()[0], xmax = axs.get_xlim()[1], linestyle = '--')
                axs.spines['right'].set_visible(False)
                axs.spines['top'].set_visible(False)
                axs.tick_params(axis='y', which='major', length=5, width=1)
                axs.tick_params(axis='x', which='major', length=5, width=1)
                axs.set_ylabel('Generation (MW)',  color='black', rotation='vertical')
                axs.set_xlabel('Date ' + '(' + self.timezone + ')',  color='black', rotation='horizontal')
                axs.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
                axs.margins(x=0.01)
                mfunc.set_plot_timeseries_format(axs)
                handles, labels = axs.get_legend_handles_labels()
                axs.legend(reversed(handles), reversed(labels),facecolor='inherit', frameon=True,loc='lower left',bbox_to_anchor=(1,0))
                outputs[zone_input] = {'fig': fig2, 'data_table': Data_Table_Out}

        # diff_csv.to_csv(self.Marmot_Solutions_folder + '/' + self.Scenario_name + '/icing_regional_MWdiffs.csv')
        # diff_csv_perc.to_csv(self.Marmot_Solutions_folder + '/' + self.Scenario_name + '/icing_regional_percdiffs.csv')

        return outputs