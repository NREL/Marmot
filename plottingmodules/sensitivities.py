import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
import marmot_plot_functions as mfunc
import os
from matplotlib.patches import Patch
import logging


#===============================================================================

custom_legend_elements = [Patch(facecolor='#DD0200',
                            alpha=0.5, edgecolor='#DD0200',
                         label='Unserved Energy')]

class mplot(object):

    def __init__(self, argument_dict):
        # iterate over items in argument_dict and set as properties of class
        # see key_list in Marmot_plot_main for list of properties
        for prop in argument_dict:
            self.__setattr__(prop, argument_dict[prop])
        self.logger = logging.getLogger('marmot_plot.'+__name__)

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

        check_input_data = [] 
        gen_collection = {}
        curt_collection = {}
        int_collection = {}
        
        if self.Scenario_Diff == ['']:
            self.logger.warning('Scenario_Diff field is empty. Ensure User Input Sheet is set up correctly!')
            outputs = mfunc.InputSheetError()
            return outputs 
        if len(self.Scenario_Diff) == 1:
            self.logger.warning('Scenario_Diff field only contains 1 entry, two are required. Ensure User Input Sheet is set up correctly!')
            outputs = mfunc.InputSheetError()
            return outputs 
        
        check_input_data.extend([mfunc.get_data(gen_collection,'generator_Generation',self.Marmot_Solutions_folder,self.Scenario_Diff)])
        check_input_data.extend([mfunc.get_data(curt_collection,'generator_Curtailment',self.Marmot_Solutions_folder,self.Scenario_Diff)])
        check_input_data.extend([mfunc.get_data(int_collection,'region_Net_Interchange',self.Marmot_Solutions_folder,self.Scenario_Diff)])

        if 1 in check_input_data:
            outputs = mfunc.MissingInputData()
            return outputs

        try:
            bc = mfunc.shift_leap_day(gen_collection.get(self.Scenario_Diff[0]),self.Marmot_Solutions_folder,self.shift_leap_day)
        except IndexError:
            self.logger.warning('Scenario_Diff "%s" is not in data. Ensure User Input Sheet is set up correctly!',self.Scenario_Diff[0])
            outputs = mfunc.InputSheetError()
            return outputs 
        
        bc_tech = bc.xs(self.prop,level = 'tech')
        bc_CT = bc.xs('Gas-CT',level = 'tech')
        bc_CC = bc.xs('Gas-CC',level = 'tech')
        try:
            scen = mfunc.shift_leap_day(gen_collection.get(self.Scenario_Diff[1]),self.Marmot_Solutions_folder,self.shift_leap_day)
        except IndexError:
            self.logger.warning('Scenario_Diff "%s" is not in data. Ensure User Input Sheet is set up correctly!',self.Scenario_Diff[0])
            outputs = mfunc.InputSheetError()
            return outputs 
        
        scen_tech = scen.xs(self.prop,level = 'tech')
        scen_CT = scen.xs('Gas-CT',level = 'tech')
        scen_CC = scen.xs('Gas-CC',level = 'tech')

        curt_bc = mfunc.shift_leap_day(curt_collection.get(self.Scenario_Diff[0]),self.Marmot_Solutions_folder,self.shift_leap_day)
        curt_scen = mfunc.shift_leap_day(curt_collection.get(self.Scenario_Diff[1]),self.Marmot_Solutions_folder,self.shift_leap_day)
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
        bc_int = mfunc.shift_leap_day(int_collection.get(self.Scenario_Diff[0]),self.Marmot_Solutions_folder,self.shift_leap_day)
        scen_int = mfunc.shift_leap_day(int_collection.get(self.Scenario_Diff[1]),self.Marmot_Solutions_folder,self.shift_leap_day)

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

                oz_bc.columns = [self.prop + ' ' + str(self.Scenario_Diff[0])] 
                oz_scen.columns = [str(self.Scenario_Diff[1])]

                Data_Out_List = []
                Data_Out_List.append(oz_bc)
                Data_Out_List.append(oz_scen)

                diffs = pd.concat(Data_Out_List,axis = 1, copy = False)

                # icing_diff.columns = [zone_input]
                # icing_diff_perc.columns = [zone_input]  
                # diff_csv = pd.concat([diff_csv, icing_diff], axis = 1)  
                # diff_csv_perc = pd.concat([diff_csv_perc, icing_diff_perc], axis = 1)
                # continue

                curt_diff = curt_diff_all.xs(zone_input,level=self.AGG_BY)
                curt_diff = mfunc.df_process_gen_inputs(curt_diff,self.ordered_gen)
                curt_diff = curt_diff.sum(axis=1)
                curt_diff = curt_diff.rename('Curtailment difference')
                Data_Out_List.append(curt_diff)

                int_diff_all = int_diff_all.reset_index()
                if self.AGG_BY not in int_diff_all.columns:
                    int_diff_all = mfunc.merge_new_agg(int_diff_all,self.AGG_BY)
                int_diff = int_diff_all[int_diff_all[self.AGG_BY] == zone_input]
                int_diff = int_diff.groupby('timestamp').sum()
                int_diff.columns = ['Net export difference']
                Data_Out_List.append(int_diff)


                fig2, axs = mfunc.setup_plot()

                plt.subplots_adjust(wspace=0.05, hspace=0.2)

                if zone_input in CT_regions:
                    CT_diff = self._process_ts(CT_diff_all,zone_input)
                    CT_diff.columns = ['Gas-CT difference']
                    Data_Out_List.append(CT_diff)
                if zone_input in CC_regions:
                    CC_diff = self._process_ts(CC_diff_all,zone_input)
                    CC_diff.columns = ['Gas-CC difference']
                    Data_Out_List.append(CC_diff)

                Data_Table_Out = pd.concat(Data_Out_List,axis = 1,copy = False)

                custom_color_dict = {'Curtailment difference' : self.PLEXOS_color_dict['Curtailment'],
                                     self.prop + ' ' + self.Scenario_Diff[0] : self.PLEXOS_color_dict[self.prop],
                                     self.Scenario_Diff[1] : self.PLEXOS_color_dict[self.prop],
                                     'Gas-CC difference' : self.PLEXOS_color_dict['Gas-CC'],
                                     'Gas-CT difference' : self.PLEXOS_color_dict['Gas-CT'],
                                     'Net export difference': 'black'}

                ls_dict = {'Curtailment difference' : 'solid',
                                     self.prop + ' ' + self.Scenario_Diff[0] : 'solid',
                                     self.Scenario_Diff[1] :':',
                                     'Gas-CC difference' : 'solid',
                                     'Gas-CT difference' : 'solid',
                                     'Net export difference': '--'}

                for col in Data_Table_Out.columns: 
                    mfunc.create_line_plot(axs,Data_Table_Out,col,color_dict = custom_color_dict, label = col, linestyle = ls_dict[col], n=0)
                    
                #Make two hatches: blue for when scenario > basecase, and red for when scenario < basecase.
                if self.Scenario_name != 'Icing' and self.Scenario_name != 'DryHydro':
                    axs[0].fill_between(diffs.index,diffs[self.prop + ' ' + str(self.Scenario_Diff[0])],diffs[str(self.Scenario_Diff[1])],
                        where = diffs[str(self.Scenario_Diff[1])] > diffs[self.prop + ' ' + str(self.Scenario_Diff[0])],
                        label = 'Increased ' + self.prop.lower() + ' generation', facecolor = 'blue', hatch = '///',alpha = 0.5)
                axs[0].fill_between(diffs.index,diffs[self.prop + ' ' + str(self.Scenario_Diff[0])],diffs[str(self.Scenario_Diff[1])],
                    where = diffs[str(self.Scenario_Diff[1])] < diffs[self.prop + ' ' + str(self.Scenario_Diff[0])],
                    label = 'Decreased ' + self.prop.lower() + ' generation', facecolor = 'red', hatch = '///',alpha = 0.5)
                axs[0].hlines(y = 0, xmin = axs[0].get_xlim()[0], xmax = axs[0].get_xlim()[1], linestyle = '--')
                axs[0].spines['right'].set_visible(False)
                axs[0].spines['top'].set_visible(False)
                axs[0].tick_params(axis='y', which='major', length=5, width=1)
                axs[0].tick_params(axis='x', which='major', length=5, width=1)
                axs[0].set_ylabel('Generation (MW)',  color='black', rotation='vertical')
                axs[0].set_xlabel('Date ' + '(' + self.timezone + ')',  color='black', rotation='horizontal')
                axs[0].yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
                axs[0].margins(x=0.01)
                mfunc.set_plot_timeseries_format(axs)
                handles, labels = axs[0].get_legend_handles_labels()
                axs[0].legend(reversed(handles), reversed(labels),facecolor='inherit', frameon=True,loc='lower left',bbox_to_anchor=(1,0))
                outputs[zone_input] = {'fig': fig2, 'data_table': Data_Table_Out}

        # diff_csv.to_csv(self.Marmot_Solutions_folder + '/' + self.Scenario_name + '/icing_regional_MWdiffs.csv')
        # diff_csv_perc.to_csv(self.Marmot_Solutions_folder + '/' + self.Scenario_name + '/icing_regional_percdiffs.csv')

        return outputs