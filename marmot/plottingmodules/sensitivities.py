# -*- coding: utf-8 -*-

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
import logging
import marmot.plottingmodules.marmot_plot_functions as mfunc
import marmot.config.mconfig as mconfig


#===============================================================================

custom_legend_elements = [Patch(facecolor='#DD0200',
                            alpha=0.5, edgecolor='#DD0200',
                         label='Unserved Energy')]

class MPlot(object):

    def __init__(self, argument_dict):
        # iterate over items in argument_dict and set as properties of class
        # see key_list in Marmot_plot_main for list of properties
        for prop in argument_dict:
            self.__setattr__(prop, argument_dict[prop])
        self.logger = logging.getLogger('marmot_plot.'+__name__)
        self.y_axes_decimalpt = mconfig.parser("axes_options","y_axes_decimalpt")
        self.mplot_data_dict = {}
        self.curtailment_prop = mconfig.parser("plot_data","curtailment_property")


    def _process_ts(self,df,zone_input):
        oz = df.xs(zone_input, level = self.AGG_BY)
        oz = oz.reset_index()
        oz = oz.groupby('timestamp').sum()
        return(oz)

    def sensitivities_gas(self, figure_name=None, prop=None, start=None, 
                             end=None, timezone="", start_date_range=None, 
                             end_date_range=None):

        """
        This method highlights the difference in generation between two scenarios of a single resource. 
        The two scenarios are specified in the "Scenario_Diff_plot" field of Marmot_user_defined_inputs.csv
        The single resource is specfied in the "properties" field of Marmot_plot_select.csv.
        Blue hatches represent additional energy produced by the resource, and red hatches represent decreased energy.
        The difference in Gas-CC and Gas-CT generation, curtailment, and net interchange are also plotted.
        Each zone is plotted on a separate figure.
        Figures and data tables are returned to plot_main
        """
        return mfunc.UnderDevelopment()

        outputs = {}
        
        if self.Scenario_Diff == ['']:
            self.logger.warning('Scenario_Diff field is empty. Ensure User Input Sheet is set up correctly!')
            outputs = mfunc.InputSheetError()
            return outputs 
        if len(self.Scenario_Diff) == 1:
            self.logger.warning('Scenario_Diff field only contains 1 entry, two are required. Ensure User Input Sheet is set up correctly!')
            outputs = mfunc.InputSheetError()
            return outputs 
        
        # List of properties needed by the plot, properties are a set of tuples and contain 3 parts:
        # required True/False, property name and scenarios required, scenarios must be a list.
        properties = [(True,"generator_Generation",self.Scenario_Diff),
                      (True,f"generator_{self.curtailment_prop}",self.Scenario_Diff),
                      (True,"region_Net_Interchange",self.Scenario_Diff)]
        
        # Runs get_data to populate mplot_data_dict with all required properties, returns a 1 if required data is missing
        check_input_data = mfunc.get_data(self.mplot_data_dict, properties,self.Marmot_Solutions_folder)

        if 1 in check_input_data:
            return mfunc.MissingInputData()
        
        try:
            bc = mfunc.shift_leapday(self.mplot_data_dict["generator_Generation"].get(self.Scenario_Diff[0]),self.Marmot_Solutions_folder)
        except IndexError:
            self.logger.warning('Scenario_Diff "%s" is not in data. Ensure User Input Sheet is set up correctly!',self.Scenario_Diff[0])
            outputs = mfunc.InputSheetError()
            return outputs 
        
        bc_tech = bc.xs(prop,level = 'tech')
        bc_CT = bc.xs('Gas-CT',level = 'tech')
        bc_CC = bc.xs('Gas-CC',level = 'tech')
        try:
            scen = mfunc.shift_leapday(self.mplot_data_dict["generator_Generation"].get(self.Scenario_Diff[1]),self.Marmot_Solutions_folder)
        except IndexError:
            self.logger.warning('Scenario_Diff "%s" is not in data. Ensure User Input Sheet is set up correctly!',self.Scenario_Diff[0])
            outputs = mfunc.InputSheetError()
            return outputs 
        
        scen_tech = scen.xs(prop,level = 'tech')
        scen_CT = scen.xs('Gas-CT',level = 'tech')
        scen_CC = scen.xs('Gas-CC',level = 'tech')

        curt_bc = mfunc.shift_leapday(self.mplot_data_dict[f"generator_{self.curtailment_prop}"].get(self.Scenario_Diff[0]),self.Marmot_Solutions_folder)
        curt_scen = mfunc.shift_leapday(self.mplot_data_dict[f"generator_{self.curtailment_prop}"].get(self.Scenario_Diff[1]),self.Marmot_Solutions_folder)
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
        bc_int = pd.read_hdf(os.path.join(self.Marmot_Solutions_folder, "Processed_HDF5_folder", self.Scenario_Diff[0] + "_formatted.h5"),"region_Net_Interchange")
        bc_int = mfunc.shift_leapday(self.mplot_data_dict["region_Net_Interchange"].get(self.Scenario_Diff[0]),self.Marmot_Solutions_folder)
        scen_int = mfunc.shift_leapday(self.mplot_data_dict["region_Net_Interchange"].get(self.Scenario_Diff[1]),self.Marmot_Solutions_folder)

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

                oz_bc.columns = [prop + ' ' + str(self.Scenario_Diff[0])] 
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
                                     prop + ' ' + self.Scenario_Diff[0] : self.PLEXOS_color_dict[prop],
                                     self.Scenario_Diff[1] : self.PLEXOS_color_dict[prop],
                                     'Gas-CC difference' : self.PLEXOS_color_dict['Gas-CC'],
                                     'Gas-CT difference' : self.PLEXOS_color_dict['Gas-CT'],
                                     'Net export difference': 'black'}

                ls_dict = {'Curtailment difference' : 'solid',
                                     prop + ' ' + self.Scenario_Diff[0] : 'solid',
                                     self.Scenario_Diff[1] :':',
                                     'Gas-CC difference' : 'solid',
                                     'Gas-CT difference' : 'solid',
                                     'Net export difference': '--'}

                for col in Data_Table_Out.columns: 
                    mfunc.create_line_plot(axs,Data_Table_Out,col,color_dict = custom_color_dict, label = col, linestyle = ls_dict[col], n=0)
                    
                #Make two hatches: blue for when scenario > basecase, and red for when scenario < basecase.
                if self.Scenario_Diff[1]!= 'Icing' and self.Scenario_Diff[1]!= 'DryHydro':
                    axs[0].fill_between(diffs.index,diffs[prop + ' ' + str(self.Scenario_Diff[0])],diffs[str(self.Scenario_Diff[1])],
                        where = diffs[str(self.Scenario_Diff[1])] > diffs[prop + ' ' + str(self.Scenario_Diff[0])],
                        label = 'Increased ' + prop.lower() + ' generation', facecolor = 'blue', hatch = '///',alpha = 0.5)
                axs[0].fill_between(diffs.index,diffs[prop + ' ' + str(self.Scenario_Diff[0])],diffs[str(self.Scenario_Diff[1])],
                    where = diffs[str(self.Scenario_Diff[1])] < diffs[prop + ' ' + str(self.Scenario_Diff[0])],
                    label = 'Decreased ' + prop.lower() + ' generation', facecolor = 'red', hatch = '///',alpha = 0.5)
                axs[0].hlines(y = 0, xmin = axs[0].get_xlim()[0], xmax = axs[0].get_xlim()[1], linestyle = '--')
                axs[0].spines['right'].set_visible(False)
                axs[0].spines['top'].set_visible(False)
                axs[0].tick_params(axis='y', which='major', length=5, width=1)
                axs[0].tick_params(axis='x', which='major', length=5, width=1)
                axs[0].set_ylabel('Generation (MW)',  color='black', rotation='vertical')
                axs[0].set_xlabel(timezone,  color='black', rotation='horizontal')
                axs[0].yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(x, f',.{self.y_axes_decimalpt}f')))
                axs[0].margins(x=0.01)
                mfunc.set_plot_timeseries_format(axs)
                handles, labels = axs[0].get_legend_handles_labels()
                axs[0].legend(reversed(handles), reversed(labels),facecolor='inherit', frameon=True,loc='lower left',bbox_to_anchor=(1,0))
                if mconfig.parser("plot_title_as_region"):
                	fig2.title(zone_input)
                outputs[zone_input] = {'fig': fig2, 'data_table': Data_Table_Out}

        # diff_csv.to_csv(self.Marmot_Solutions_folder + '/' + self.Scenario_name + '/icing_regional_MWdiffs.csv')
        # diff_csv_perc.to_csv(self.Marmot_Solutions_folder + '/' + self.Scenario_name + '/icing_regional_percdiffs.csv')

        return outputs