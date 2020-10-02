# -*- coding: utf-8 -*-
"""
Created on Wed May 22 14:29:48 2019

@author: Daniel Levie

This code was written to process PLEXOS HDF5 outputs to get them ready for plotting.
Once the data is processed it is outputed as an intermediary HDF5 file format so that
it can be read into the Marmot_plot_main.py file

"""
#===============================================================================
# Import Python Libraries
#===============================================================================

import pandas as pd
#import numpy as np
import os
import h5py
import sys
import pathlib
import time
from meta_data import MetaData

sys.path.append('../h5plexos')
from h5plexos.query import PLEXOSSolution

try:
    print("Will process row:" +(sys.argv[1]))
    print(str(len(sys.argv)-1)+" arguments were passed from commmand line.")
except IndexError:
    #No arguments passed
    pass
#===============================================================================
# Create HDF5 file from PLEXOS zip solution
#===============================================================================
#This is only required if your output has not been processed already on Eagle

#from h5plexos.process import process_solution
#PLEXOS_Solution = '/Volumes/PLEXOS CEII/Projects/Extreme Events/Model c_Feb01-04_2011_2024 Solution/Model c_Feb01-04_2011_2024 Solution.zip' #PLEXOS solution .zip file.
#process_solution(PLEXOS_Solution,'/Volumes/PLEXOS CEII/Projects/Extreme Events/Model c_Feb01-04_2011_2024 Solution/Model c_Feb01-04_2011_2024 Solution.h5') # Saves out to PLEXOS_Solution.h5

#===============================================================================
# Load Input Properties
#===============================================================================

#A bug in pandas requires this to be included, otherwise df.to_string truncates long strings
#Fix available in Pandas 1.0 but leaving here in case user version not up to date
pd.set_option("display.max_colwidth", 1000)

#changes working directory to location of this python file
os.chdir(pathlib.Path(__file__).parent.absolute())

Marmot_user_defined_inputs = pd.read_csv('Marmot_user_defined_inputs.csv', usecols=['Input','User_defined_value'],
                                         index_col='Input', skipinitialspace=True)

# File which determiens which plexos properties to pull from the h5plexos results and process, this file is in the repo
Plexos_Properties = pd.read_csv('plexos_properties.csv')

# Name of the Scenario(s) being run, must have the same name(s) as the folder holding the runs HDF5 file
Scenario_List = pd.Series(Marmot_user_defined_inputs.loc['Scenario_process_list'].squeeze().split(",")).str.strip().tolist()
# The folder that contains all PLEXOS h5plexos outputs - the h5 files should be contained in another folder with the Scenario_name
PLEXOS_Solutions_folder = Marmot_user_defined_inputs.loc['PLEXOS_Solutions_folder'].to_string(index=False).strip()

# Folder to save your processed solutions
Marmot_Solutions_folder = Marmot_user_defined_inputs.loc['Marmot_Solutions_folder'].to_string(index=False).strip()

# This folder contains all the csv required for mapping and selecting outputs to process
# Examples of these mapping files are within the Marmot repo, you may need to alter these to fit your needs
Mapping_folder = 'mapping_folder'

Region_Mapping = pd.read_csv(os.path.join(Mapping_folder, Marmot_user_defined_inputs.loc['Region_Mapping.csv_name'].to_string(index=False).strip()))
gen_names = pd.read_csv(os.path.join(Mapping_folder, Marmot_user_defined_inputs.loc['gen_names.csv_name'].to_string(index=False).strip()))

# Value of Lost Load for calculatinhg cost of unserved energy
VoLL = pd.to_numeric(Marmot_user_defined_inputs.loc['VoLL'].to_string(index=False))

#===============================================================================
# Standard Naming of Generation Data
#===============================================================================

gen_names_dict=gen_names[['Original','New']].set_index("Original").to_dict()["New"]
vre_gen_cat = pd.read_csv(os.path.join(Mapping_folder, 'vre_gen_cat.csv'),squeeze=True).str.strip().tolist()

#===============================================================================
# Region mapping
#===============================================================================
Region_Mapping = Region_Mapping.astype(str)
try:
    Region_Mapping = Region_Mapping.drop(["category"],axis=1) # delete category columns if exists
except Exception:
    pass

#=============================================================================
# FUNCTIONS FOR PLEXOS DATA EXTRACTION
#=============================================================================

# This function handles the pulling of the data from the H5plexos hdf5 file and then passes the data to one of the formating functions
# metadata is now a parameter of get data
def get_data(loc, prop,t, db, metadata):

    try:
        if "_" in loc:
            df = db.query_relation_property(loc,prop,timescale=t)
        else:
            df = db.query_object_property(loc,prop,timescale=t)

    except KeyError:
        df = report_prop_error(prop,loc)
        return df

    # Instantiate instance of Process Class
    # metadata is used as a paramter to initialize process_cl
    process_cl = Process(df, metadata)
    # Instantiate Method of Process Class
    process_att = getattr(process_cl,'df_process_' + loc)
    # Process attribute and return to df
    df = process_att()

    if loc == 'region' and prop == "Unserved Energy" and int(df.sum(axis=0)) > 0:
        print("\n WARNING! Scenario contains Unserved Energy: " + str(int(df.sum(axis=0))) + " MW\n")

    return df

# This fucntion prints a warning message when the get_data function cannot find the specified property in the H5plexos hdf5 file
def report_prop_error(prop,loc):
    print('CAN NOT FIND {} FOR {}. {} DOES NOT EXIST'.format(prop,loc,prop))
    df = pd.DataFrame()
    return df

# Process Class contains methods for processing data
class Process:

    def __init__(self, df, metadata):

        # certain methods require information from metadata.  metadata is now
        # passed in as an instance of MetaData class for the appropriate model
        self.df = df
        self.metadata = metadata


# Function for formatting data which comes form the PLEXOS Generator Category
    def df_process_generator(self):
        df = self.df.droplevel(level=["band", "property"])
        df.index.rename(['tech','gen_name'], level=['category','name'], inplace=True)

        if self.metadata.region_generator_category().empty == False:
            region_gen_idx = pd.CategoricalIndex(self.metadata.region_generator_category().index.get_level_values(0))
            region_gen_idx = region_gen_idx.repeat(len(df.index.get_level_values('timestamp').unique()))

            idx_region = pd.MultiIndex(levels= df.index.levels + [region_gen_idx.categories]
                                ,codes= df.index.codes +  [region_gen_idx.codes],
                                names= df.index.names + region_gen_idx.names)
        else:
            idx_region = df.index

        if self.metadata.zone_generator_category().empty == False:
            zone_gen_idx = pd.CategoricalIndex(self.metadata.zone_generator_category().index.get_level_values(0))
            zone_gen_idx = zone_gen_idx.repeat(len(df.index.get_level_values('timestamp').unique()))

            idx_zone = pd.MultiIndex(levels= idx_region.levels + [zone_gen_idx.categories]
                                ,codes= idx_region.codes + [zone_gen_idx.codes] ,
                                names= idx_region.names + zone_gen_idx.names)
        else:
            idx_zone = idx_region

        if len(Region_Mapping.columns)>1 == True:
            region_gen_mapping_idx = pd.MultiIndex.from_frame(self.metadata.region_generator_category().merge(Region_Mapping,
                                how="left", on='region').sort_values(by=['tech','gen_name']).drop(['region','tech','gen_name'], axis=1))
            region_gen_mapping_idx = region_gen_mapping_idx.repeat(len(df.index.get_level_values('timestamp').unique()))

            idx_map = pd.MultiIndex(levels= idx_zone.levels + region_gen_mapping_idx.levels
                                ,codes= idx_zone.codes + region_gen_mapping_idx.codes,
                                names = idx_zone.names + region_gen_mapping_idx.names)
        else:
            idx_map = idx_zone

        idx_map = idx_map.droplevel(level=["tech"])
        df_tech = pd.CategoricalIndex(df.index.get_level_values('tech').map(lambda x: gen_names_dict.get(x,x)))

        idx =  pd.MultiIndex(levels= [df_tech.categories] + idx_map.levels
                                ,codes= [df_tech.codes] + idx_map.codes,
                                names = df_tech.names + idx_map.names)

        df = pd.DataFrame(data=df.values.reshape(-1), index=idx)
        df_col = list(df.index.names) # Gets names of all columns in df and places in list
        df_col.insert(0, df_col.pop(df_col.index("timestamp"))) #move timestamp to start of df
        df = df.reorder_levels(df_col, axis=0)
        df[0] = pd.to_numeric(df[0], downcast='float')

        # Checks if all generator tech categorieses have been identified and matched. If not, lists categories that need a match
        if set(df.index.unique(level="tech")).issubset(gen_names["New"].unique()) == False:
            print("\n !! The Following Generators do not have a correct category mapping:")
            missing_gen_cat = list((set(df.index.unique(level="tech"))) - (set(gen_names["New"].unique())))
            print(missing_gen_cat)
            print("")
        return df

    # Function for formating data which comes from the PLEXOS Region Category
    def df_process_region(self):
        df = self.df.droplevel(level=["band", "property", "category"])
        df.index.rename('region', level='name', inplace=True)
        if len(Region_Mapping.columns)>1 == True: #checks if Region_Mapping contains data to merge, skips if empty
            mapping_idx = pd.MultiIndex.from_frame(self.metadata.regions().merge(Region_Mapping,
                                how="left", on='region').drop(['region','category'], axis=1))
            mapping_idx = mapping_idx.repeat(len(df.index.get_level_values('timestamp').unique()))

            idx = pd.MultiIndex(levels= df.index.levels + mapping_idx.levels
                                ,codes= df.index.codes + mapping_idx.codes,
                                names = df.index.names + mapping_idx.names)
        else:
            idx = df.index
        df = pd.DataFrame(data=df.values.reshape(-1), index=idx)
        df_col = list(df.index.names) # Gets names of all columns in df and places in list
        df_col.insert(0, df_col.pop(df_col.index("timestamp"))) #move timestamp to start of df
        df = df.reorder_levels(df_col, axis=0)
        df[0] = pd.to_numeric(df[0], downcast='float')
        return df

    # Function for formating data which comes from the PLEXOS Zone Category
    def df_process_zone(self):
        df = self.df.droplevel(level=["band", "property", "category"])
        df.index.rename('zone', level='name', inplace=True)
        df = pd.DataFrame(data=df.values.reshape(-1), index=df.index)
        df_col = list(df.index.names) # Gets names of all columns in df and places in list
        df_col.insert(0, df_col.pop(df_col.index("timestamp"))) #move timestamp to start of df
        df = df.reorder_levels(df_col, axis=0)
        df[0] = pd.to_numeric(df[0], downcast='float')
        return df

    # Function for formatting data which comes form the PLEXOS Line Category
    def df_process_line(self):
        df = self.df.droplevel(level=["band", "property", "category"])
        df.index.rename('line_name', level='name', inplace=True)
        df = pd.DataFrame(data=df.values.reshape(-1), index=df.index)
        df_col = list(df.index.names) # Gets names of all columns in df and places in list
        df_col.insert(0, df_col.pop(df_col.index("timestamp"))) #move timestamp to start of df
        df = df.reorder_levels(df_col, axis=0)
        df[0] = pd.to_numeric(df[0], downcast='float')
        return df

    # Function for formatting data which comes form the PLEXOS Line Category
    def df_process_interface(self):
        df = self.df.droplevel(level=["band", "property"])
        df.index.rename(['interface_name', 'interface_category'], level=['name','category'], inplace=True)
        df = pd.DataFrame(data=df.values.reshape(-1), index=df.index)
        df_col = list(df.index.names) # Gets names of all columns in df and places in list
        df_col.insert(0, df_col.pop(df_col.index("timestamp"))) #move timestamp to start of df
        df = df.reorder_levels(df_col, axis=0)
        df[0] = pd.to_numeric(df[0], downcast='float')
        return df

    # Function for formatting data which comes form the PLEXOS Reserve Category (To Fix: still uses old merging method)
    def df_process_reserve(self):
        df = self.df.droplevel(level=["band", "property"])
        df.index.rename(['parent','Type'], level=['name','category'], inplace=True)
        df = df.reset_index() # unzip the levels in index
        if self.metadata.reserves_regions().empty == False:
            df = df.merge(self.metadata.reserves_regions(), how='left', on='parent') # Merges in regions where reserves are located
        if self.metadata.reserves_zones().empty == False:
            df = df.merge(self.metadata.reserves_zones(), how='left', on='parent') # Merges in zones where reserves are located
        df_col = list(df.columns) # Gets names of all columns in df and places in list
        df_col.remove(0)
        df_col.insert(0, df_col.pop(df_col.index("timestamp"))) #move timestamp to start of df
        df.set_index(df_col, inplace=True)
        df[0] = pd.to_numeric(df[0], downcast='float')
        return df

    # Function for formatting data which comes form the PLEXOS Reserve_generators Category
    def df_process_reserves_generators(self):
        df = self.df.droplevel(level=["band", "property"])
        df.index.rename(['gen_name'], level=['child'], inplace=True)
        df = df.reset_index() # unzip the levels in index
        df = df.merge(self.metadata.generator_category(), how='left', on='gen_name')
        if self.metadata.reserves_regions().empty == False:
            df = df.merge(self.metadata.reserves_regions(), how='left', on='parent') # Merges in regions where reserves are located
        if self.metadata.reserves_zones().empty == False:
            df = df.merge(self.metadata.reserves_zones(), how='left', on='parent') # Merges in zones where reserves are located
        df['tech'] = df['tech'].map(lambda x: gen_names_dict.get(x,x))
        df_col = list(df.columns) # Gets names of all columns in df and places in list
        df_col.remove(0)
        df_col.insert(0, df_col.pop(df_col.index("timestamp"))) #move timestamp to start of df
        df.set_index(df_col, inplace=True)
        df[0] = pd.to_numeric(df[0], downcast='float')
        return df

    # Function for formatting data which comes form the PLEXOS Fuel Category
    def df_process_fuel(self):
        df = self.df.droplevel(level=["band", "property", "category"])
        df.index.rename('fuel_type', level='name', inplace=True)
        df = pd.DataFrame(data=df.values.reshape(-1), index=df.index)
        df_col = list(df.index.names) # Gets names of all columns in df and places in list
        df_col.insert(0, df_col.pop(df_col.index("timestamp"))) #move timestamp to start of df
        df = df.reorder_levels(df_col, axis=0)
        df[0] = pd.to_numeric(df[0], downcast='float')
        return df

    # Function for formatting data which comes form the PLEXOS Constraint Category
    def df_process_constraint(self):
        df = self.df.droplevel(level=["band", "property"])
        df.index.rename(['constraint_category', 'constraint'], level=['category', 'name'], inplace=True)
        df = pd.DataFrame(data=df.values.reshape(-1), index=df.index)
        df_col = list(df.index.names) # Gets names of all columns in df and places in list
        df_col.insert(0, df_col.pop(df_col.index("timestamp"))) #move timestamp to start of df
        df = df.reorder_levels(df_col, axis=0)
        df[0] = pd.to_numeric(df[0], downcast='float')
        return df

    # Function for formatting data which comes form the PLEXOS emission Category
    def df_process_emission(self):
        df = self.df.droplevel(level=["band", "property"])
        df.index.rename('emission_type', level='name', inplace=True)
        df = pd.DataFrame(data=df.values.reshape(-1), index=df.index)
        df_col = list(df.index.names) # Gets names of all columns in df and places in list
        df_col.insert(0, df_col.pop(df_col.index("timestamp"))) #move timestamp to start of df
        df = df.reorder_levels(df_col, axis=0)
        df[0] = pd.to_numeric(df[0], downcast='float')
        return df

    # Function for formatting data which comes form the PLEXOS storage Category (To Fix: still uses old merging method)
    def df_process_storage(self):
        df = self.df.droplevel(level=["band", "property", "category"])
        df = df.reset_index() # unzip the levels in index
        df = df.merge(self.metadata.generator_storage(), how='left', on='name')
        if self.metadata.region_generators().empty == False:
            df = df.merge(self.metadata.region_generators(), how='left', on='gen_name') # Merges in regions where generators are located
        if self.metadata.zone_generators().empty == False:
            df = df.merge(self.metadata.zone_generators(), how='left', on='gen_name') # Merges in zones where generators are located
        if len(Region_Mapping.columns)>1 == True: #checks if Region_Maping contains data to merge, skips if empty (Default)
            df = df.merge(Region_Mapping, how='left', on='region') # Merges in all Region Mappings
        df.rename(columns={'name':'storage_resource'}, inplace=True)
        df_col = list(df.columns) # Gets names of all columns in df and places in list
        df_col.remove(0) # Removes 0, the data column from the list
        df_col.insert(0, df_col.pop(df_col.index("timestamp"))) #move timestamp to start of df
        df.set_index(df_col, inplace=True)
        df[0] = pd.to_numeric(df[0], downcast='float')
        return df

    # Function for formatting data which comes form the PLEXOS region_regions Category
    def df_process_region_regions(self):
        df = self.df.droplevel(level=["band", "property"])
        df = pd.DataFrame(data=df.values.reshape(-1), index=df.index)
        df_col = list(df.index.names) # Gets names of all columns in df and places in list
        df_col.insert(0, df_col.pop(df_col.index("timestamp"))) #move timestamp to start of df
        df = df.reorder_levels(df_col, axis=0)
        df[0] = pd.to_numeric(df[0], downcast='float')
        return df

    def df_process_node(self):
        df = self.df.droplevel(level=["band","property","category"])
        df.index.rename('node', level='name', inplace=True)
        df.sort_index(level=['node'], inplace=True)
        if self.metadata.node_region().empty == False:
            node_region_idx = pd.CategoricalIndex(self.metadata.node_region().index.get_level_values(0))
            node_region_idx = node_region_idx.repeat(len(df.index.get_level_values('timestamp').unique()))
            idx_region = pd.MultiIndex(levels= df.index.levels + [node_region_idx.categories]
                                ,codes= df.index.codes +  [node_region_idx.codes],
                                names= df.index.names + node_region_idx.names)
        else:
            idx_region = df.index
        if self.metadata.node_zone().empty == False:
            node_zone_idx = pd.CategoricalIndex(self.metadata.node_zone().index.get_level_values(0))
            node_zone_idx = node_zone_idx.repeat(len(df.index.get_level_values('timestamp').unique()))
            idx_zone = pd.MultiIndex(levels= idx_region.levels + [node_zone_idx.categories]
                                ,codes= idx_region.codes + [node_zone_idx.codes] ,
                                names= idx_region.names + node_zone_idx.names)
        else:
            idx_zone = idx_region
        if len(Region_Mapping.columns)>1 == True:
            region_mapping_idx = pd.MultiIndex.from_frame(self.metadata.node_region().merge(Region_Mapping,
                                how="left", on='region').drop(['region','node'], axis=1))
            region_mapping_idx = region_mapping_idx.repeat(len(df.index.get_level_values('timestamp').unique()))

            idx_map = pd.MultiIndex(levels= idx_zone.levels + region_mapping_idx.levels
                                ,codes= idx_zone.codes + region_mapping_idx.codes,
                                names = idx_zone.names + region_mapping_idx.names)
        else:
            idx_map = idx_zone

        df = pd.DataFrame(data=df.values.reshape(-1), index=idx_map)
        df_col = list(df.index.names) # Gets names of all columns in df and places in list
        df_col.insert(0, df_col.pop(df_col.index("timestamp"))) #move timestamp to start of df
        df = df.reorder_levels(df_col, axis=0)
        df[0] = pd.to_numeric(df[0], downcast='float')
        return df

#===================================================================================
# Main
#===================================================================================
for Scenario_name in Scenario_List:

    print("\n#### Processing " + Scenario_name + " PLEXOS Results ####")

    #===============================================================================
    # Input and Output Directories
    #===============================================================================

    HDF5_output = Scenario_name + "_formatted.h5"

    Marmot_Scenario = os.path.join(Marmot_Solutions_folder, Scenario_name)
    try:
        os.makedirs(Marmot_Scenario )
    except FileExistsError:
        # directory already exists
        pass
    hdf_out_folder = os.path.join(Marmot_Scenario , 'Processed_HDF5_folder')
    try:
        os.makedirs(hdf_out_folder)
    except FileExistsError:
        # directory already exists
        pass
    HDF5_folder_in = os.path.join(PLEXOS_Solutions_folder, Scenario_name)
    try:
        os.makedirs(HDF5_folder_in)
    except FileExistsError:
        # directory already exists
        pass
    figure_folder = os.path.join(Marmot_Scenario , 'Figures_Output')
    try:
        os.makedirs(figure_folder)
    except FileExistsError:
        # directory already exists
        pass

    startdir=os.getcwd()
    os.chdir(HDF5_folder_in)     #Due to a bug on eagle need to chdir before listdir
    files = sorted(os.listdir()) # List of all files in hdf5 folder in alpha numeric order
    os.chdir(startdir)

##############################################################################
    files_list = []
    for names in files:
        if names.endswith(".h5"):
            files_list.append(names) # Creates a list of only the hdf5 files


###############################################################################

    # Read in all HDF5 files into dictionary
    print("Loading all HDF5 files to prepare for processing")
    hdf5_collection = {}
    for file in files_list:
        hdf5_collection[file] = PLEXOSSolution(os.path.join(HDF5_folder_in, file))

######### Process the Outputs##################################################

    # Creates Initial HDF5 file for ouputing formated data
    Processed_Data_Out=pd.DataFrame()
    if os.path.isfile(os.path.join(hdf_out_folder,HDF5_output))==True:
        print("\nWarning: "+hdf_out_folder + "/" + HDF5_output+" already exists; new variables will be added.")
    else:
        Processed_Data_Out.to_hdf(os.path.join(hdf_out_folder, HDF5_output), key= "generator_Generation" , mode="w", complevel=9, complib  ='blosc:zlib')

    # Filters for chosen Plexos properties to process
    if (len(sys.argv)-1) == 1: # If passed one argument (not including file name which is automatic)
        print("Will process row " +(sys.argv[1])+" of plexos properties regardless of T/F.")
        Plexos_Properties = Plexos_Properties.iloc[int(sys.argv[1])-1].to_frame().T
    else:
        Plexos_Properties = Plexos_Properties.loc[Plexos_Properties["collect_data"] == True]

    start = time.time()

    # Main loop to process each ouput and pass data to functions
    for index, row in Plexos_Properties.iterrows():

        Processed_Data_Out = pd.DataFrame()
        data_chuncks = []
        print("\nProcessing " + row["group"] + " " + row["data_set"])
        for model in files_list:

            # Create an instance of metadata, and pass that as a variable to get data.
            meta = MetaData(HDF5_folder_in, Region_Mapping,model)

            print("     "+ model)
            db = hdf5_collection.get(model)

            processed_data = get_data(row["group"], row["data_set"],row["data_type"], db, meta)

            if processed_data.empty == True:
                break

            if (row["data_type"] == "year")&((row["data_set"]=="Installed Capacity")|(row["data_set"]=="Export Limit")|(row["data_set"]=="Import Limit")):
                data_chuncks.append(processed_data*row["unit_multiplier"])
                print(row["data_set"]+" Year property reported from only the first partition.")
                break
            else:
                data_chuncks.append(processed_data*row["unit_multiplier"])

        if data_chuncks:
            Processed_Data_Out = pd.concat(data_chuncks, copy=False)

        if Processed_Data_Out.empty == False:
            if (row["data_type"]== "year"):
                print("\nPlease Note: Year properties can not be checked for duplicates. \nOverlaping data can not be removed from 'Year' grouped data.")
                print("This will effect Year data that differs between partitions such as cost results.\nIt will not effect Year data that is equal in all partitions such as Installed Capacity or Line Limit results.\n")

            else:
                oldsize=Processed_Data_Out.size
                Processed_Data_Out = Processed_Data_Out.loc[~Processed_Data_Out.index.duplicated(keep='first')] #Remove duplicates; keep first entry^M
                if  (oldsize-Processed_Data_Out.size) >0:
                    print('Drop duplicates removed '+str(oldsize-Processed_Data_Out.size)+' rows.')

            row["data_set"] = row["data_set"].replace(' ', '_')
            try:
                Processed_Data_Out.to_hdf(os.path.join(hdf_out_folder, HDF5_output), key= row["group"] + "_" + row["data_set"], mode="a", complevel=9, complib = 'blosc:zlib')
            except:
                print("File is probably in use, waiting to attempt save for a second time.")
                time.sleep(120)
                try:
                      Processed_Data_Out.to_hdf(os.path.join(hdf_out_folder, HDF5_output), key= row["group"] + "_" + row["data_set"], mode="a", complevel=9, complib = 'blosc:zlib')
                      print("File save succeded on second attempt.")
                except:
                    print("File is probably in use, waiting to attempt save for a third time.")
                    time.sleep(240)
                    try:
                        Processed_Data_Out.to_hdf(os.path.join(hdf_out_folder, HDF5_output), key= row["group"] + "_" + row["data_set"], mode="a",  complevel=9, complib = 'blosc:zlib')
                        print("File save succeded on third attempt.")
                    except:
                        print("Save failed on third try; will not attempt again.")
            # del Processed_Data_Out
        else:
            continue

######### Calculate Extra Ouputs################################################
    if "generator_Curtailment" not in h5py.File(os.path.join(hdf_out_folder, HDF5_output),'r'):
        try:
            print("\nProcessing generator Curtailment")
            try:
                Avail_Gen_Out = pd.read_hdf(os.path.join(hdf_out_folder, HDF5_output), 'generator_Available_Capacity')
                Total_Gen_Out = pd.read_hdf(os.path.join(hdf_out_folder, HDF5_output), 'generator_Generation')
                if Total_Gen_Out.empty==True:
                    print("WARNING!! generator_Available_Capacity & generator_Generation are required for Curtailment calculation")
            except KeyError:
                print("WARNING!! generator_Available_Capacity & generator_Generation are required for Curtailment calculation")
            # Adjust list of values to drop from vre_gen_cat depending on if it exhists in processed techs
            vre_gen_cat = [name for name in vre_gen_cat if name in Avail_Gen_Out.index.unique(level="tech")]

            if not vre_gen_cat:
                print("\nvre_gen_cat is not set up correctly with your gen_names")
                print("To Process Curtailment add correct names to vre_gen_cat.csv")
                print("For more information see Marmot Readme under 'Mapping Files'")
            # Output Curtailment#
            Curtailment_Out =  ((Avail_Gen_Out.loc[(slice(None), vre_gen_cat),:]) -
                                (Total_Gen_Out.loc[(slice(None), vre_gen_cat),:]))

            Curtailment_Out.to_hdf(os.path.join(hdf_out_folder, HDF5_output), key="generator_Curtailment", mode="a", complevel=9, complib = 'blosc:zlib')

            #Clear Some Memory
            del Total_Gen_Out
            del Avail_Gen_Out
            del Curtailment_Out
        except Exception:
            print("NOTE!! Curtailment not calculated, processing skipped\n")
            pass

    if "region_Cost_Unserved_Energy" not in h5py.File(os.path.join(hdf_out_folder, HDF5_output),'r'):
        try:
            print("Calculating Cost Unserved Energy: Regions")
            Cost_Unserved_Energy = pd.read_hdf(os.path.join(hdf_out_folder, HDF5_output), 'region_Unserved_Energy')
            Cost_Unserved_Energy = Cost_Unserved_Energy * VoLL
            Cost_Unserved_Energy.to_hdf(os.path.join(hdf_out_folder, HDF5_output), key="region_Cost_Unserved_Energy", mode="a", complevel=9, complib = 'blosc:zlib')
        except KeyError:
            print("NOTE!! Regional Unserved Energy not available to process, processing skipped\n")
            pass

    if "zone_Cost_Unserved_Energy" not in h5py.File(os.path.join(hdf_out_folder, HDF5_output),'r'):
        try:
            print("Calculating Cost Unserved Energy: Zones")
            Cost_Unserved_Energy = pd.read_hdf(os.path.join(hdf_out_folder, HDF5_output), 'zone_Unserved_Energy')
            Cost_Unserved_Energy = Cost_Unserved_Energy * VoLL
            Cost_Unserved_Energy.to_hdf(os.path.join(hdf_out_folder, HDF5_output), key="zone_Cost_Unserved_Energy", mode="a", complevel=9, complib = 'blosc:zlib')
        except KeyError:
            print("NOTE!! Zonal Unserved Energy not available to process, processing skipped\n")
            pass

    end = time.time()
    elapsed = end - start
    print('Main loop took ' + str(elapsed/60) + ' minutes.')
###############################################################################

# Code that can be used to test PLEXOS_H5_results_formatter

    # test = pd.read_hdf(os.path.join(hdf_out_folder, HDF5_output), 'reserve_Shortage')
    # test = test.xs("region_name",level='zone')
    # test = test.xs("Nuclear",level='tech')
    # test = test.reset_index(['timestamp','gen_name'])
    # test = test.groupby(["timestamp", "node"], as_index=False).sum()
    # test = test.pivot(index='timestamp', columns='gen_name', values=0)

    # test = test[['600003_PR IS31G_20','600005_MNTCE31G_22']]
    # test = test.reset_index()

    # test.index.get_level_values('region') = test.index.get_level_values('region').astype("category")

    # test['timestamp'] = test['timestamp'].astype("category")

    # test.index = test.index.set_levels(test.index.levels[-1].astype('category'), level=-1)

    # test.memory_usage(deep=True)
    # test[0] = pd.to_numeric(test[0], downcast='float')

    # test.memory_usage(deep=False)

    # Stacked_Gen_read = Stacked_Gen_read.reset_index() # unzip the levels in index
    # Stacked_Gen_read.rename(columns={'name':'zone'}, inplace=True)
    #         Stacked_Gen_read = Stacked_Gen_read.drop(["band", "property", "category"],axis=1)
        # if int(Stacked_Gen_read.sum(axis=0)) >= 0:
        #     print("WARNING! Scenario contains Unserved Energy: " + str(int(Stacked_Gen_read.sum(axis=0))) + "MW")

        #storage = db.storage("Generation")
        #storage = df_process_storage(storage, overlap)

    # df_old = df
    # t =df.loc[~df.index.duplicated()]
    # df_old.equals(df)
