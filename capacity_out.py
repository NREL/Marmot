import os
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
import numpy as np



#===============================================================================

class mplot(object):
    def __init__(self,argument_list):

        self.prop = argument_list[0]
        self.start = argument_list[1]
        self.end = argument_list[2]
        self.timezone = argument_list[3]
        self.start_date = argument_list[4]
        self.end_date = argument_list[5]
        self.hdf_out_folder = argument_list[6]
        self.zone_input =argument_list[7]
        self.AGG_BY = argument_list[8]
        self.ordered_gen = argument_list[9]
        self.PLEXOS_color_dict = argument_list[10]
        self.Multi_Scenario = argument_list[11]
        self.Scenario_Diff = argument_list[12]
        self.PLEXOS_Scenarios = argument_list[13]
        self.ylabels = argument_list[14]
        self.xlabels = argument_list[15]
        self.color_list = argument_list[16]
        self.gen_names_dict = argument_list[18]
        self.re_gen_cat = argument_list[20]
        self.Region_Mapping = argument_list[24]


    def capacity_out_stack_facet(self):

      print('Zone = ' + str(self.zone_input))

      xdimension=len(self.xlabels)
      if xdimension == 0:
          xdimension = 1
      ydimension=len(self.ylabels)
      if ydimension == 0:
          ydimension = 1
      grid_size = xdimension*ydimension
      fig1, axs = plt.subplots(ydimension,xdimension, figsize=((8*xdimension),(4*ydimension)), sharey=True)
      plt.subplots_adjust(wspace=0.05, hspace=0.2)
      axs = axs.ravel()
      i=0

      met_year = self.PLEXOS_Scenarios[-4:] #Extract met year from PLEXOS parent scenario.

      for scenario in self.Multi_Scenario:
          print("Scenario = " + str(scenario))

          infra_year = scenario[-4:] #Extract infra year from scenario name.
          capacity_out = pd.read_csv(os.path.join('/projects/continental/pcm/Results/capacity out timeseries',infra_year + '_' + met_year + '_capacity out.csv'))
          capacity_out.index = pd.to_datetime(capacity_out.DATETIME)
          one_zone = capacity_out[capacity_out['PLEXOS_Zone'] == self.zone_input]    #Select only this particular zone.
          one_zone = one_zone.drop(columns = ['DATETIME','PLEXOS_Zone'])

         #Select only time period of interest.
          if self.prop == 'Date Range':
              print("Plotting specific date range:")
              print(str(self.start_date) + '  to  ' + str(self.end_date))
              one_zone = one_zone[self.start_date : self.end_date]

          sp = axs[i].stackplot(one_zone.index.values, one_zone.values.T, labels=one_zone.columns, linewidth=0,
                      colors=[self.PLEXOS_color_dict.get(x, '#333333') for x in one_zone.T.index])

          axs[i].spines['right'].set_visible(False)
          axs[i].spines['top'].set_visible(False)
          axs[i].tick_params(axis='y', which='major', length=5, width=1)
          axs[i].tick_params(axis='x', which='major', length=5, width=1)
          axs[i].yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
          axs[i].margins(x=0.01)

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
          if i == (len(self.Multi_Scenario) - 1) :
              handles, labels = axs[i].get_legend_handles_labels()
              leg1 = axs[i].legend(reversed(handles), reversed(labels), loc='lower left',bbox_to_anchor=(1,0),facecolor='inherit', frameon=True)
              axs[i].add_artist(leg1)
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

      return fig1
