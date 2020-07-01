# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 10:34:48 2019

This code creates generation stack plots and is called from Marmot_plot_main.py

@author: dlevie
"""

import os
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
import numpy as np 

from metadata import MetaData



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
        self.HDF5_folder_in = argument_list[25]
        
    
    def net_export(self):
        
        print("Zone = " + self.zone_input)
        
        Net_Export_all_scenarios = pd.DataFrame()

        for scenario in self.Multi_Scenario:
            
            print("Scenario = " + str(scenario))
    
            Net_Export_read = pd.read_hdf(os.path.join(self.PLEXOS_Scenarios,scenario, 'Processed_HDF5_folder', scenario + '_formatted.h5'),'region_Net_Interchange')
                
            Net_Export = Net_Export_read.xs(self.zone_input, level = self.AGG_BY)
            Net_Export = Net_Export.reset_index()
            Net_Export = Net_Export.groupby(["timestamp"]).sum()
            Net_Export.columns = [scenario]
            
            if self.prop == 'Date Range':
                print("Plotting specific date range:")
                print(str(self.start_date) + '  to  ' + str(self.end_date))
                
                Net_Export = Net_Export[self.start_date : self.end_date]

            Net_Export_all_scenarios = pd.concat([Net_Export_all_scenarios,Net_Export], axis = 1) 
            
        # Data table of values to return to main program
        Data_Table_Out = Net_Export_all_scenarios
        
        #Make scenario/color dictionary.
        scenario_color_dict = {}
        for idx,column in enumerate(Net_Export_all_scenarios.columns):
            dictionary = {column : self.color_list[idx]}
            scenario_color_dict.update(dictionary)
            
        fig1, ax = plt.subplots(figsize=(9,6))
        for idx,column in enumerate(Net_Export_all_scenarios.columns):
            ax.plot(Net_Export_all_scenarios.index.values,Net_Export_all_scenarios[column], linewidth=2, color = scenario_color_dict.get(column,'#333333'),label=column)

        
        ax.set_ylabel('Net exports (MW)',  color='black', rotation='vertical')
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
        
        handles, labels = ax.get_legend_handles_labels()
        
        #Legend 1
        leg1 = ax.legend(reversed(handles), reversed(labels), loc='best',facecolor='inherit', frameon=True)  
        
        # Manually add the first legend back
        ax.add_artist(leg1)

        return {'fig': fig1, 'data_table': Data_Table_Out}
    
  
    def line_util(self):          #Duration curve of individual line utilization for all hours
        
        meta = MetaData(self.HDF5_folder_in, self.Region_Mapping)
        
        Flow_Collection = {}        # Create Dictionary to hold Datframes for each scenario 
        Limit_Collection = {}
        
        for scenario in self.Multi_Scenario:
            Flow_Collection[scenario] = pd.read_hdf(os.path.join(self.PLEXOS_Scenarios, scenario,"Processed_HDF5_folder", scenario+ "_formatted.h5"),"line_Flow")
            Limit_Collection[scenario]= pd.read_hdf(os.path.join(self.PLEXOS_Scenarios, scenario,"Processed_HDF5_folder", scenario+ "_formatted.h5"),"line_Import_Limit")
        
        print("For all lines touching Zone = "+self.zone_input)
        
        fig3, ax3 = plt.subplots(len(self.Multi_Scenario),figsize=(9,6)) # Set up subplots for all scenarios
     
        n=0 #Counter for scenario subplots
        
        Data_Out=pd.DataFrame()
                
        for scenario in self.Multi_Scenario:
            
            print("Scenario = " + str(scenario))
            lines_interregional = meta.regional_line_relations()
            lines_intraregional = meta.region_lines()
            
            # lines_interregional=pd.read_pickle(os.path.join(self.PLEXOS_Scenarios,scenario,"line_relations_interregional.pkl")).set_index([self.AGG_BY])
            # lines_intraregional=pd.read_pickle(os.path.join(self.PLEXOS_Scenarios,scenario,"line_relations_intraregional.pkl")).set_index([self.AGG_BY])
            try:
                lines_interregional=lines_interregional.xs(self.zone_input)            
            except KeyError:
                lines_interregional=pd.DataFrame()
                print("No interregional lines touching "+self.zone_input+".")
            try:
                lines_intraregional=lines_intraregional.xs(self.zone_input)            
            except KeyError:
                lines_intraregional=pd.DataFrame()
                print("No intraregional lines in "+self.zone_input+".")
            
            zone_lines=pd.concat([lines_interregional,lines_intraregional],axis=0,sort=False)
            zone_lines=zone_lines['line_name'].unique()
            
            Flow = Flow_Collection.get(scenario).reset_index('line_name')
            Flow = Flow[Flow['line_name'].isin(zone_lines)]                 #Limit to only lines touching to this zone
            Flow.set_index(['line_name'],inplace=True,append=True)            
            
            Limits= Limit_Collection.get(scenario).droplevel('timestamp')
            Limits.mask(Limits[0]==0.0,other=0.01,inplace=True) #if limit is zero set to small value
            
            if (self.prop!=self.prop)==False: # This checks for a nan in string. If no scenario selected, do nothing.
                print("Line category = "+str(self.prop))
                line_relations = meta.line_relations()
                #line_relations=pd.read_pickle(os.path.join(self.PLEXOS_Scenarios,scenario,"line_relations.pkl")).rename(columns={"name":"line_name"}).set_index(["line_name"])
                Flow=pd.merge(Flow,line_relations,left_index=True,right_index=True)
                Flow=Flow[Flow["category"]==self.prop] 
                Flow=Flow.drop('category',axis=1) 
                
#            AbsMaxFlow = Flow.abs().groupby(["line_name"]).max()
#            AbsMaxFlow = pd.merge(AbsMaxFlow[0],Limits[0].abs(),left_index=True,right_index=True,how='left')
#            AbsMaxFlow['Limit']=AbsMaxFlow[['0_x','0_y']]
            Flow = pd.merge(Flow,Limits[0].abs(),left_index=True, right_index=True,how='left')
            del Limits
            Flow['Util']=Flow['0_x'].abs()/Flow['0_y']
#            Flow[Flow['Util'].isna()==True]['Util']=1.0
            Flow.mask(Flow['Util']>1.0,other=1.0,inplace=True) #If greater than 1 because exceeds flow limit, report as 1
            Annual_Util=Flow['Util'].groupby(["line_name"]).mean()      
            
            for line in Flow.index.get_level_values(level='line_name').unique() :
                duration_curve = Flow.xs(line,level="line_name").sort_values(by='Util',ascending=False).reset_index()
                        
                if len(self.Multi_Scenario)>1:
                    ax3[n].plot(duration_curve['Util'])
                    if (self.prop!=self.prop)==False: # This checks if a category was passed
                            ax3[n].set_ylabel(scenario+' Line Utilization '+'\n'+'Line cateogory: '+str(self.prop),  color='black', rotation='vertical')
                    else:
                            ax3[n].set_ylabel(scenario+' Line Utilization',  color='black', rotation='vertical')
                    ax3[n].set_xlabel('Intervals',  color='black', rotation='horizontal')
                    ax3[n].spines['right'].set_visible(False)
                    ax3[n].spines['top'].set_visible(False)                         
                    plt.ylim((0,1.1))           

                else:
                    ax3.plot(duration_curve['Util'])
                    if (self.prop!=self.prop)==False: # This checks if a category was passed
                            ax3.set_ylabel(scenario+' Line Utilization '+'\n'+'Line cateogory: '+str(self.prop),  color='black', rotation='vertical')
                    else:
                            ax3.set_ylabel(scenario+' Line Utilization',  color='black', rotation='vertical')
                    ax3.set_xlabel('Intervals',  color='black', rotation='horizontal')
                    ax3.spines['right'].set_visible(False)
                    ax3.spines['top'].set_visible(False)   
                    plt.ylim((0,1.1))           
                
                del duration_curve
            
            Annual_Util=Annual_Util.groupby("line_name").mean()
            Annual_Util.rename(columns={0:scenario},inplace=True)
            Data_Out=pd.concat([Data_Out,Annual_Util],axis=1,sort=False)
            
            del Flow, Annual_Util
            
               
            n=n+1
        #end scenario loop
                              
        return {'fig': fig3,'data_table':Data_Out}
    
    
    
    def line_hist(self):                #Histograms of individual line utilization factor for entire year
        
        meta = MetaData(self.HDF5_folder_in, self.Region_Mapping)
        
        print("For all lines touching Zone = "+self.zone_input)
        
        Flow_Collection = {}            # Create Dictionary to hold Datframes for each scenario 
        Limit_Collection = {}
        for scenario in self.Multi_Scenario:
            Flow_Collection[scenario] = pd.read_hdf(os.path.join(self.PLEXOS_Scenarios, scenario,"Processed_HDF5_folder", scenario+ "_formatted.h5"),"line_Flow")
            Limit_Collection[scenario]= pd.read_hdf(os.path.join(self.PLEXOS_Scenarios, scenario,"Processed_HDF5_folder", scenario+ "_formatted.h5"),"line_Import_Limit")
        
        fig3, ax3 = plt.subplots(len(self.Multi_Scenario),figsize=(9,6)) # Set up subplots for all scenarios
     
        n=0 #Counter for scenario subplots
                 
        Data_Out=pd.DataFrame()             
        for scenario in self.Multi_Scenario:
                    
            print("Scenario = " + str(scenario))
            #lines_interregional = meta.regional_line_relations()
            #lines_intraregional = meta.region_lines()
            
            lines_interregional=pd.read_pickle(os.path.join(self.PLEXOS_Scenarios,scenario,"line_relations_interregional.pkl")).set_index([self.AGG_BY])
            lines_intraregional=pd.read_pickle(os.path.join(self.PLEXOS_Scenarios,scenario,"line2region.pkl")).set_index([self.AGG_BY]) # "line_relations_intraregional.pkl")).set_index([self.AGG_BY])
            try:
                lines_interregional=lines_interregional.xs(self.zone_input)            
            except KeyError:
                lines_interregional=pd.DataFrame()
                print("No interregional lines touching "+self.zone_input+".")
            try:
                lines_intraregional=lines_intraregional.xs(self.zone_input)            
            except KeyError:
                lines_intraregional=pd.DataFrame()
                print("No intraregional lines in "+self.zone_input+".")
            

            zone_lines=pd.concat([lines_interregional,lines_intraregional],axis=0,sort=False)
            zone_lines=zone_lines['line_name'].unique()
            
            Flow = Flow_Collection.get(scenario).reset_index('line_name')
            Flow = Flow[Flow['line_name'].isin(zone_lines)] #Limit to only lines touching to this zone
            Flow.set_index(['line_name'],inplace=True,append=True)
            
            Limits= Limit_Collection.get(scenario).droplevel('timestamp')
            Limits.mask(Limits[0]==0.0,other=0.01,inplace=True) #if limit is zero set to small value

            if (self.prop!=self.prop)==False: # This checks for a nan in string. If no category selected, do nothing.
                print("Line category = "+str(self.prop))
                line_relations = meta.line_relations()
                #line_relations=pd.read_pickle(os.path.join(self.PLEXOS_Scenarios,scenario,"line_relations.pkl")).rename(columns={"name":"line_name"}).set_index(["line_name"])
                Flow=pd.merge(Flow,line_relations,left_index=True,right_index=True)
                Flow=Flow[Flow["category"]==self.prop] 
                Flow=Flow.drop('category',axis=1)
            
            Flow = pd.merge(Flow,Limits[0].abs(),left_index=True, right_index=True,how='left')
            del Limits
            Flow['Util']=Flow['0_x'].abs()/Flow['0_y']
#            Flow[Flow['Util'].isna()==True]['Util']=1.0
            Flow.mask(Flow['Util']>1.0,other=1.0,inplace=True) #If greater than 1 because exceeds flow limit, report as 1
            Annual_Util=Flow['Util'].groupby(["line_name"]).mean() 
            del Flow
                                
            if len(self.Multi_Scenario)>1:
                ax3[n].hist(Annual_Util.replace([np.inf,np.nan]),bins=20,range=(0,1),label=scenario)
                if (self.prop!=self.prop)==False: # This checks if a category was passed
                    ax3[n].set_ylabel(scenario+' Number of lines '+'\n'+'Line cateogory: '+str(self.prop),  color='black', rotation='vertical')
                else:
                    ax3[n].set_ylabel(scenario+' Number of lines',  color='black', rotation='vertical')

                ax3[n].set_xlabel('Utilization',  color='black', rotation='horizontal')
                ax3[n].spines['right'].set_visible(False)
                ax3[n].spines['top'].set_visible(False)                       
            
            else:
                ax3.hist(Annual_Util.replace([np.inf,np.nan]),bins=20,range=(0,1),label=scenario)
                if (self.prop!=self.prop)==False: # This checks if a category was passed
                    ax3.set_ylabel(scenario+' Number of lines '+'\n'+'Line cateogory: '+str(self.prop),  color='black', rotation='vertical')
                else:
                    ax3.set_ylabel(scenario+' Number of lines',  color='black', rotation='vertical')

                ax3.set_xlabel('Utilization',  color='black', rotation='horizontal')
                ax3.spines['right'].set_visible(False)
                ax3.spines['top'].set_visible(False)   
            
            Annual_Util.rename(columns={0:scenario},inplace=True)
            Data_Out=pd.concat([Data_Out,Annual_Util],axis=1,sort=False)
            del Annual_Util  
            n=n+1
        #end scenario loop
                              
        return {'fig': fig3,'data_table':Data_Out}
    
    def region_region_interchange(self): 

        rr_int = pd.read_hdf(os.path.join(self.PLEXOS_Scenarios,self.Multi_Scenario[0],"Processed_HDF5_folder", self.Multi_Scenario[0] + "_formatted.h5"),"region_regions_Net_Interchange")
        agg_region_mapping = self.Region_Mapping[['region',self.AGG_BY]].set_index('region').to_dict()[self.AGG_BY]

        rr_int = rr_int.reset_index()
        rr_int['parent'] = rr_int['parent'].map(agg_region_mapping)
        rr_int['child']  = rr_int['child'].map(agg_region_mapping)
        rr_int_agg = rr_int.groupby(['timestamp','parent','child'],as_index=True).sum() 
        rr_int_agg.rename(columns = {0:'Flow (MW)'}, inplace = True)
        rr_int_agg = rr_int_agg.unstack(level = 'child')
        rr_int_agg = rr_int_agg.droplevel(level = 0, axis = 1)
        rr_int_agg = rr_int_agg.stack(level = 'child')
        rr_int_agg = rr_int_agg.reset_index()

        Data_Table_Out = rr_int_agg
        
        #Make a facet plot, one panel for each parent zone. 
        parents = rr_int_agg['parent'].unique()

        fig4, axs = plt.subplots(nrows = 5, ncols = 4, figsize=(6*5,4*4), sharey = False)
        plt.subplots_adjust(wspace=0.6, hspace=0.5)

        axs = axs.ravel()
        i=0
        
        for parent in parents:
            single_parent = rr_int_agg[rr_int_agg['parent'] == parent]
            single_parent = single_parent.pivot(index = 'timestamp',columns = 'child',values = 0)
            single_parent = single_parent.loc[:,(single_parent != 0).any(axis = 0)] #Remove all 0 columns (uninteresting).
            if (parent in single_parent.columns):
                single_parent = single_parent.drop(columns = [parent]) #Remove columns if parent = child (strange).
            
            for column in single_parent.columns:
                axs[i].plot(single_parent.index.values,single_parent[column], linewidth=2, label=column)
            
            axs[i].set_title(parent)
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
            axs[i].legend(loc='lower left',bbox_to_anchor=(1,0),facecolor='inherit', frameon=True)  
            
            i = i + 1 
        if len(parents) % 2 !=0:    #Remove extra plot
            fig4.delaxes(axs[len(parents)])
            
        fig4.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.xlabel('Date ' + '(' + self.timezone + ')',  color='black', rotation='horizontal',labelpad = 60)
        plt.ylabel('Flow from zone indicated in panel title to zone indicated in legend (MW)',  color='black', rotation='vertical', labelpad = 60)
        
            
        return {'fig': fig4, 'data_table': Data_Table_Out}

    def region_region_checkerboard(self): 

        rr_int = pd.read_hdf(os.path.join(self.PLEXOS_Scenarios,self.Multi_Scenario[0],"Processed_HDF5_folder", self.Multi_Scenario[0] + "_formatted.h5"),"region_regions_Net_Interchange")
        agg_region_mapping = self.Region_Mapping[['region',self.AGG_BY]].set_index('region').to_dict()[self.AGG_BY]

        rr_int = rr_int.reset_index()
        rr_int['parent'] = rr_int['parent'].map(agg_region_mapping)
        rr_int['child']  = rr_int['child'].map(agg_region_mapping)
        rr_int_agg = rr_int.groupby(['parent','child'],as_index=True).sum() # Determine annual net flow between regions.
        rr_int_agg.rename(columns = {0:'Flow (MW)'}, inplace = True)
        rr_int_agg = rr_int_agg.reset_index(['parent','child'])
        rr_int_agg=rr_int_agg.loc[rr_int_agg['Flow (MW)']>0.01] # Keep only positive flows                            
        rr_int_agg.sort_values(ascending=False,by='Flow (MW)')
        rr_int_agg.set_index(['parent','child'],inplace=True)
        rr_int_agg = rr_int_agg.unstack('child')
        rr_int_agg = rr_int_agg.droplevel(level = 0, axis = 1)
        
#        rr_int_agg['path']=rr_int_agg['parent']+"_"+rr_int_agg['child']
#        pathlist=rr_int_agg['path'] #List of paths

        ## Annual summary
        fig, ax = plt.subplots(figsize=(9,6))
        current_cmap = plt.cm.get_cmap()
        current_cmap.set_bad(color='grey')

        im = ax.imshow(rr_int_agg,interpolation='none')
        cbar=fig.colorbar(im,ax=ax)
        cbar.ax.set_ylabel("Net Interchange [GWh]", rotation=-90, va="bottom")

        ax.set_xticks(np.arange(rr_int_agg.shape[1]))
        ax.set_yticks(np.arange(rr_int_agg.shape[0]))
        ax.set_xticklabels(rr_int_agg.columns)
        ax.set_yticklabels(rr_int_agg.index)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

        ax.set_xticks(np.arange(rr_int_agg.shape[1]+1)-.5, minor=True) #Delineate the boxes and make room at top and bottom
        ax.set_yticks(np.arange(rr_int_agg.shape[0]+1)-.5, minor=True)
        ax.grid(which="minor", color="k", linestyle='-', linewidth=1)
        ax.tick_params(which="minor", bottom=False, left=False)

        ax.set_ylabel('From Region',  color='black')
        ax.set_xlabel('To Region',color='black')
        fig.tight_layout()
                
        Data_Table_Out = rr_int_agg
           
                   
        return {'fig': fig, 'data_table': Data_Table_Out}
    
    def region_region_duration(self): 

        rr_int = pd.read_hdf(os.path.join(self.PLEXOS_Scenarios,self.Multi_Scenario[0],"Processed_HDF5_folder", self.Multi_Scenario[0] + "_formatted.h5"),"region_regions_Net_Interchange")
        agg_region_mapping = self.Region_Mapping[['region',self.AGG_BY]].set_index('region').to_dict()[self.AGG_BY]

        rr_int = rr_int.reset_index()
        rr_int['parent'] = rr_int['parent'].map(agg_region_mapping)
        rr_int['child']  = rr_int['child'].map(agg_region_mapping)
        
        rr_int_agg = rr_int.groupby(['parent','child'],as_index=True).sum() # Determine annual net flow between regions.
        rr_int_agg.rename(columns = {0:'Flow (MW)'}, inplace = True)
        rr_int_agg = rr_int_agg.reset_index(['parent','child'])
        rr_int_agg=rr_int_agg.loc[rr_int_agg['Flow (MW)']>0.01] # Keep only positive flows                            
#        rr_int_agg.set_index(['parent','child'],inplace=True)
   
        rr_int_agg['path']=rr_int_agg['parent']+"_"+rr_int_agg['child']
        
        if (self.prop!=self.prop)==False: # This checks for a nan in string. If a number of paths is selected only plot those
            pathlist=rr_int_agg.sort_values(ascending=False,by='Flow (MW)')['path'][1:int(self.prop)+1] #list of top paths based on number selected
        else:
            pathlist=rr_int_agg['path'] #List of paths

        
        rr_int_hr = rr_int.groupby(['timestamp','parent','child'],as_index=True).sum() # Hourly flow
        rr_int_hr.rename(columns = {0:'Flow (MW)'}, inplace = True)
        rr_int_hr.reset_index(['timestamp','parent','child'],inplace=True)
        rr_int_hr['path']=rr_int_hr['parent']+"_"+rr_int_hr['child']
        rr_int_hr.set_index(['path'],inplace=True)
        rr_int_hr['Abs MW']=abs(rr_int_hr['Flow (MW)'])
        rr_int_hr['Abs MW'].sum()        
        rr_int_hr.loc[pathlist]['Abs MW'].sum()*2  # Check that the sum of the absolute value of flows is the same. i.e. only redundant flows were eliminated.
        rr_int_hr=rr_int_hr.loc[pathlist].drop(['Abs MW'],axis=1)
        
        ## Plot duration curves 
        fig3, ax3 = plt.subplots(figsize=(9,6))
        for i in pathlist:
            duration_curve = rr_int_hr.loc[i].sort_values(ascending=False,by='Flow (MW)').reset_index()
            plt.plot(duration_curve['Flow (MW)'],label=i)
            del duration_curve
    
        ax3.set_ylabel('Flow MW',  color='black', rotation='vertical')
        ax3.set_xlabel('Intervals',  color='black', rotation='horizontal')
        ax3.spines['right'].set_visible(False)
        ax3.spines['top'].set_visible(False)

#        if (self.prop!=self.prop)==False: # This checks for a nan in string. If no limit selected, do nothing
        ax3.legend(loc='best')            
#            plt.lim((0,int(self.prop)))  

            
        Data_Table_Out = rr_int_hr
           
                   
        return {'fig': fig3, 'data_table': Data_Table_Out}
    