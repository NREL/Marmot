# -*- coding: utf-8 -*-
"""
Created on Wed May 22 14:29:48 2019

@author: Daniel Levie

This code was written to process PLEXOS HDF5 outputs to get them ready for plotting. 
Once the data is processed it is outputed as an intermediary HDF5 file format so that 
it can be read into the PLEXOS Outputs Grapher.

"""

#===============================================================================
# Import Python Libraries
#===============================================================================

import pandas as pd
import numpy as np
import os
import h5py
from h5plexos.query import PLEXOSSolution


#===============================================================================
# Create HDF5 file from PLEXOS zip solution
#===============================================================================
##This is only required if your output has not been processed already on Eagle
#
#from h5plexos.process import process_solution
#PLEXOS_Solution = r"\\nrelqnap02\PLEXOS\Projects\Drivers_of_Curtailment\HDF5 Files\LA_2020_less_solar\Model base_2020_yr_remove_some_PV Solution.zip"
#
#process_solution(PLEXOS_Solution, r"\\nrelqnap02\PLEXOS\Projects\Drivers_of_Curtailment\HDF5 Files\LA_2020_less_solar\Model base_2020_yr_remove_some_PV Solution.h5") # Saves out to PLEXOS_Solution.h5

#===============================================================================
""" User Defined Names, Directories and Settings """
#===============================================================================

# Name of the Scenario being run, must have the same name as the folder holding the runs HDF5 file
Scenario_name = "Test_Data"

# Base directory to create folders and save outputs
Run_folder = r"\\nrelqnap02\PLEXOS\Projects\Drivers_of_Curtailment"

# The folder that contains all h5plexos outputs - the h5 files should be contained in another folder with the Scenario_name
HDF5_folder_in = r"\\nrelqnap02\PLEXOS\Projects\Drivers_of_Curtailment\HDF5 Files"

# This folder contains all the csv required for mapping and selecting outputs to process
# Examples of these mapping files are within the Marmot repo, you may need to alter these to fit your needs
Mapping_folder = r"\\nrelqnap02\PLEXOS\Projects\Drivers_of_Curtailment\Region_Mapping\\"

Region_Mapping = pd.read_csv(Mapping_folder + "Region_mapping.csv")
reserve_region_type = pd.read_csv(Mapping_folder + "reserve_region_type.csv")
gen_names = pd.read_csv(Mapping_folder + "gen_names.csv")

# File which determiens which plexos properties to pull from the h5plexos results and process, this file is in the repo
Plexos_Properties = pd.read_csv(r"C:\Users\DLEVIE\Documents\Marmot\plexos_properties.csv")

overlap = 0 # number of hours overlapped between two adjacent models

VoLL = 10000 # Value of Lost Load for calculatinhg cost of unserved energy


HDF5_output = "PLEXOS_outputs_formatted.h5" #name of hdf5 file which holds processed outputs 


#===============================================================================
# Input and Output Directories 
#===============================================================================


PLEXOS_Scenarios = Run_folder + r"\PLEXOS_Scenarios" + "/" + Scenario_name
try:
    os.makedirs(PLEXOS_Scenarios)
except FileExistsError:
    # directory already exists
    pass
hdf_out_folder = PLEXOS_Scenarios + r"\Processed_HDF5_folder"
try:
    os.makedirs(hdf_out_folder)
except FileExistsError:
    # directory already exists
    pass
HDF5_folder_in = HDF5_folder_in + "/" + Scenario_name
try:
    os.makedirs(HDF5_folder_in)
except FileExistsError:
    # directory already exists
    pass
figure_folder = PLEXOS_Scenarios + r"\Figures_Output"
try:
    os.makedirs(figure_folder)
except FileExistsError:
    # directory already exists
    pass


#===============================================================================
# Standard Naming of Generation Data
#===============================================================================

gen_names_dict=gen_names[['Original','New']].set_index("Original").to_dict()["New"]

#===============================================================================
# Region mapping
#===============================================================================

try:
    Region_Mapping = Region_Mapping.drop(["category"],axis=1) # delete category columns if exists
except Exception:
    pass


#=============================================================================
# FUNCTIONS FOR PLEXOS DATA EXTRACTION
#=============================================================================


# Function for formating data which comes from the PLEXOS Region Category
def df_process_region(df, overlap_hour): 
    df = df.reset_index() # unzip the levels in index
    df.rename(columns={'name':'region'}, inplace=True)
    df = df.merge(Region_Mapping, how='left', on='region') # Merges in all Region Mappings
    df = df.drop(["band", "property", "category"],axis=1) 
    if not df["timestamp"].iloc[0].is_year_start: # for results not start at the first hour in the year, remove the overlapped beginning hours
        df = df.drop(range(0, overlap_hour))  
    df_col = list(df.columns) # Gets names of all columns in df and places in list
    df_col.remove(0) # Removes 0, the data column from the list 
    df_col.insert(0, df_col.pop(df_col.index("timestamp"))) #move timestamp to start of df
    df =  df.groupby(df_col).sum() #moves all columns to multiindex except 0 column
    return df    

# Function for formatting data which comes form the PLEXOS Generator Category
def df_process_gen(df, overlap_hour): 
    df = df.reset_index() # unzip the levels in index
    df = df.drop(["band", "property"],axis=1) 
    df.rename(columns={'category':'tech', 'name':'gen_name'}, inplace=True)
    df = df.merge(region_generators, how='left', on='gen_name') # Merges in regions where generators are located
    df = df.merge(Region_Mapping, how='left', on='region') # Merges in all Region Mappings
    df['tech'].replace(gen_names_dict, inplace=True)
    if not df["timestamp"].iloc[0].is_year_start: # for results not start at the first hour in the year, remove the overlapped beginning hours
        df = df.drop(range(0, overlap_hour))
    df_col = list(df.columns) # Gets names of all columns in df and places in list
    df_col.remove(0) # Removes 0, the data column from the list 
    df_col.insert(0, df_col.pop(df_col.index("timestamp"))) #move timestamp to start of df
    df =  df.groupby(df_col).sum() #moves all columns to multiindex except 0 column 
    return df  

# Function for formatting data which comes form the PLEXOS Line Category
def df_process_line(df, overlap_hour): 
    df = df.reset_index() # unzip the levels in index
    df = df.drop(["band", "property", "category"],axis=1) # delete property and band columns
    df.rename(columns={'name':'line_name'}, inplace=True)
    if not df["timestamp"].iloc[0].is_year_start: # for results not start at the first hour in the year, remove the overlapped beginning hours
        df = df.drop(range(0, overlap_hour))
    df = df.groupby(["timestamp", "line_name"]).sum()
    return df 

# Function for formatting data which comes form the PLEXOS Line Category
def df_process_interface(df, overlap_hour): 
    df = df.reset_index() # unzip the levels in index
    df = df.drop(["band", "property"],axis=1) # delete property and band columns
    df.rename(columns={'category':'region_region', 'name':'line_name'}, inplace=True)
    if not df["timestamp"].iloc[0].is_year_start: # for results not start at the first hour in the year, remove the overlapped beginning hours
        df = df.drop(range(0, overlap_hour))
    df = df.groupby(["timestamp",  "region_region", "line_name"]).sum()
    return df                   

# Function for formatting data which comes form the PLEXOS Reserve Category
def df_process_reserve(df, overlap_hour): 
    df = df.reset_index() # unzip the levels in index
    df = df.drop(["band", "property", "category"],axis=1) # delete property and band columns
    df.rename(columns={'name':'parent'}, inplace=True)
    df = df.merge(reserve_region_type, how='left', on='parent')
    if not df["timestamp"].iloc[0].is_year_start: # for results not start at the first hour in the year, remove the overlapped beginning hours
        df = df.drop(range(0, overlap_hour))
    df = df.groupby(["timestamp",  "Reserve_Region", "Type"]).sum()
    return df      

# Function for formatting data which comes form the PLEXOS Reserve_generators Category
def df_process_reserve_generators(df, overlap_hour): 
    df = df.reset_index() # unzip the levels in index
    df = df.drop(["band", "property"],axis=1) # delete property and band columns
    df.rename(columns={'child':'gen_name'}, inplace=True)
    df = df.merge(generator_category, how='left', on='gen_name')
    df = df.merge(reserve_region_type, how='left', on='parent')
    df['tech'].replace(gen_names_dict, inplace=True)
    if not df["timestamp"].iloc[0].is_year_start: # for results not start at the first hour in the year, remove the overlapped beginning hours
        df = df.drop(range(0, overlap_hour))
    df = df.groupby(["timestamp",  "tech", "Reserve_Region", "Type"]).sum()
    return df      

# Function for formatting data which comes form the PLEXOS Fuel Category
def df_process_fuel(df, overlap_hour): 
    df = df.reset_index() # unzip the levels in index
    df = df.drop(["band", "property", "category"],axis=1) # delete property and band columns
    df.rename(columns={'name':'fuel_type'}, inplace=True)
    if not df["timestamp"].iloc[0].is_year_start: # for results not start at the first hour in the year, remove the overlapped beginning hours
        df = df.drop(range(0, overlap_hour))
    df = df.groupby(["timestamp",  "fuel_type"]).sum()
    return df   

# Function for formatting data which comes form the PLEXOS Constraint Category
def df_process_constraint(df, overlap_hour): 
    df = df.reset_index() # unzip the levels in index
    df = df.drop(["band", "property"],axis=1) # delete property and band columns
    df.rename(columns={'category':'constraint_category', 'name':'constraint'}, inplace=True)
    if not df["timestamp"].iloc[0].is_year_start: # for results not start at the first hour in the year, remove the overlapped beginning hours
        df = df.drop(range(0, overlap_hour))
    df = df.groupby(["timestamp",  "constraint_category", "constraint"]).sum()
    return df

# Function for formatting data which comes form the PLEXOS emission Category
def df_process_emission(df, overlap_hour): 
    df = df.reset_index() # unzip the levels in index
    df = df.drop(["band", "property"],axis=1) # delete property and band columns
    df.rename(columns={'name':'emission_type'}, inplace=True)
    if not df["timestamp"].iloc[0].is_year_start: # for results not start at the first hour in the year, remove the overlapped beginning hours
        df = df.drop(range(0, overlap_hour))
    df = df.groupby(["timestamp", "emission_type"]).sum()
    return df 

# Function for formatting data which comes form the PLEXOS storage Category
def df_process_storage(df, overlap_hour): 
    df = df.reset_index() # unzip the levels in index
    df = df.drop(["band", "property", "category"],axis=1) # delete property and band columns
    df = df.merge(generator_storage, how='left', on='name')
    df = df.merge(region_generators, how='left', on='gen_name')
    df = df.merge(Region_Mapping, how='left', on='region')
    df.rename(columns={'name':'storage_resource'}, inplace=True)
    if not df["timestamp"].iloc[0].is_year_start: # for results not start at the first hour in the year, remove the overlapped beginning hours
        df = df.drop(range(0, overlap_hour))
    df_col = list(df.columns) # Gets names of all columns in df and places in list
    df_col.remove(0) # Removes 0, the data column from the list 
    df_col.insert(0, df_col.pop(df_col.index("timestamp"))) #move timestamp to start of df
    df =  df.groupby(df_col).sum() #moves all columns to multiindex except 0 column 
    return df

# Function for formatting data which comes form the PLEXOS region_regions Category
def df_process_region_regions(df, overlap_hour): 
    df = df.reset_index() # unzip the levels in index
    df = df.drop(["band", "property"],axis=1) # delete property and band columns
    if not df["timestamp"].iloc[0].is_year_start: # for results not start at the first hour in the year, remove the overlapped beginning hours
        df = df.drop(range(0, overlap_hour))
    df =  df.groupby(["timestamp",  "parent", "child"]).sum()
    return df                                          


# This fucntion prints a warning message when the get_data function cannot find the specified property in the H5plexos hdf5 file
def report_prop_error(prop,loc):
        print('CAN NOT FIND {} FOR {}. {} DOES NOT EXIST'.format(prop,loc,prop))
        df = pd.DataFrame()
        return df



# This function handles the pulling of the data from the H5plexos hdf5 file and then passes the data to one of the formating functions                
def get_data(loc, prop,t, db, overlap):
    if loc == 'constraint': 
        try: 
            df = db.constraint(prop, timescale=t)
            df = df_process_constraint(df, overlap)
            return df
        except: df = report_prop_error(prop,loc)
        
    elif loc == 'emission':
        try: 
            df = db.emission(prop)
            df = df_process_emission(df, overlap)
            return df
        except: df = report_prop_error(prop,loc)
        
#    elif loc == 'emission_generators':
#        try: 
#            df = db.emission_generators(prop, timescale=t)
#        except: df = report_prop_error(prop,loc)
        
    elif loc == 'fuel':
        try: 
            df = db.fuel(prop, timescale=t)
            df = df_process_fuel(df, overlap)
            return df
        except: df = report_prop_error(prop,loc)
        
    elif loc == 'generator':
        try:
            df = db.generator(prop, timescale=t)
            df = df_process_gen(df, overlap)
            # Checks if all generator tech categorieses have been identified and matched. If not, lists categories that need a match
            if set(df.index.unique(level="tech")).issubset(gen_names["New"].unique()) == False:
                print("\n WARNING!! The Following Generators do not have a correct category mapping \n")
                print((set(df.index.unique(level="tech"))) - (set(gen_names["New"].unique())))
            return df
        except: df = report_prop_error(prop,loc)
        
    elif loc == 'line':
        try: 
            df = db.line(prop, timescale=t)
            df = df_process_line(df, overlap)
            return df
        except: df = report_prop_error(prop,loc)
       
    elif loc == 'interface':
        try: 
            df = db.line(prop, timescale=t)
            df = df_process_interface(df, overlap)
            return df
        except: df = report_prop_error(prop,loc)
        
    elif loc == 'region':
        try: 
            df = db.region(prop, timescale=t)
            df = df_process_region(df, overlap)
            return df
        except: df = report_prop_error(prop,loc)
        
    elif loc == 'reserve':
        try: 
            df = db.reserve(prop, timescale=t)
            df = df_process_reserve(df, overlap) 
            return df
        except: df = report_prop_error(prop,loc)
        
    elif loc == 'reserve_generators':
        try: 
            df = db.reserve_generators(prop, timescale=t)
            df = df_process_reserve_generators(df, overlap) 
            return df
        except: df = report_prop_error(prop,loc)
        
    elif loc == 'storage':
        try: 
            df = db.storage(prop, timescale=t)
            df = df_process_storage(df, overlap) 
            return df
        except: df = report_prop_error(prop,loc)
        
    elif loc == 'region_regions':
        try: 
            df = db.storage(prop, timescale=t)
            df = df_process_region_regions(df, overlap) 
            return df
        except: df = report_prop_error(prop,loc)       
        
    elif loc == 'zone': df = db.zone(prop, timescale=t)
    else: 
        df = pd.DataFrame()
        print('{} NOT RETRIEVED.\nNO H5 CATEGORY: {}'.format(prop,loc))

#===============================================================================
# Main         
#===============================================================================      


files = os.listdir(HDF5_folder_in) # List of all files in hdf5 folder
files_list = []
for names in files:
    if names.endswith(".h5"):
        files_list.append(names) # Creates a list of only the hdf5 files

hdf5_read = HDF5_folder_in + "/" + files_list[0]

data = h5py.File(hdf5_read, 'r')
metadata = np.asarray(data['metadata'])

# Region generators mapping
region_generators = pd.DataFrame(np.asarray(data['metadata/relations/region_generators']))
region_generators.rename(columns={'child':'gen_name'}, inplace=True)
region_generators.rename(columns={'parent':'region'}, inplace=True)
region_generators["gen_name"]=region_generators["gen_name"].str.decode("utf-8")
region_generators["region"]=region_generators["region"].str.decode("utf-8")
region_generators.drop_duplicates(subset=["gen_name"],keep='first',inplace=True) #For generators which belong to more than 1 region, drop duplicates.
 
# Generator categories mapping 
generator_category = pd.DataFrame(np.asarray(data['metadata/objects/generator']))
generator_category.rename(columns={'name':'gen_name'}, inplace=True)
generator_category.rename(columns={'category':'tech'}, inplace=True)
generator_category["gen_name"]=generator_category["gen_name"].str.decode("utf-8")
generator_category["tech"]=generator_category["tech"].str.decode("utf-8")

# Generator head and tail torage mapping 
generator_headstorage = pd.DataFrame(np.asarray(data['metadata/relations/generator_headstorage']))
generator_tailtorage = pd.DataFrame(np.asarray(data['metadata/relations/generator_tailstorage']))
generator_storage = pd.concat([generator_headstorage, generator_tailtorage])
generator_storage.rename(columns={'child':'name'}, inplace=True)
generator_storage.rename(columns={'parent':'gen_name'}, inplace=True)
generator_storage["name"]=generator_storage["name"].str.decode("utf-8")
generator_storage["gen_name"]=generator_storage["gen_name"].str.decode("utf-8")


# Read in all HDF5 files into dictionary 
print("Loading all HDF5 files to prepare for processing")
hdf5_collection = {}
for file in files_list:
    hdf5_collection[file] = PLEXOSSolution(HDF5_folder_in + "/" + file)



######### Process the Outputs################################################          

# Creates Initial HDF5 file for ouputing formated data
Processed_Data_Out=pd.DataFrame()
Processed_Data_Out.to_hdf(hdf_out_folder + "/" + HDF5_output , key="generator_Generation" , mode="w", complevel=9, complib="blosc")

# Filters for chosen Plexos properties to prcoess
Plexos_Properties = Plexos_Properties.loc[Plexos_Properties["collect_data"] == True]


# Main loop to process each ouput and pass data to functions
for index, row in Plexos_Properties.iterrows():
    Processed_Data_Out = pd.DataFrame()
    print("Processing " + row["group"] + " " + row["data_set"])
    for model in files_list:
        print("     "+ model) 
        db = hdf5_collection.get(model)
        
        processed_data = get_data(row["group"], row["data_set"], 
                                         row["data_type"], db, overlap)
        if row["data_type"] == "year":
            Processed_Data_Out = processed_data
            break
        else:    
            Processed_Data_Out = pd.concat([Processed_Data_Out, processed_data])
    
    Processed_Data_Out.sort_index(inplace=True)
    row["data_set"] = row["data_set"].replace(' ', '_')
    Processed_Data_Out.to_hdf(hdf_out_folder + "/" + HDF5_output , key= row["group"] + "_" + row["data_set"], mode="a", complevel=9, complib="blosc")


######### Calculate Extra Ouputs################################################
try:
    print("Processing Curtailment")  
    Avail_Gen_Out = pd.read_hdf(hdf_out_folder + "/" + HDF5_output, 'generator_Available_Capacity')
    Total_Gen_Out = pd.read_hdf(hdf_out_folder + "/" + HDF5_output, 'generator_Generation')
    # Output Curtailment# 
    Curtailment_Out =  ((Avail_Gen_Out.loc[(slice(None), ['Wind','PV']),:]) - 
                        (Total_Gen_Out.loc[(slice(None), ['Wind','PV']),:]))
    
    Curtailment_Out.to_hdf(hdf_out_folder + "/" + HDF5_output , key="generator_Curtailment", mode="a", complevel=9, complib="blosc")
    

    #Clear Some Memory
    del Total_Gen_Out
    del Avail_Gen_Out
    del Curtailment_Out
except Exception:
    print("NOTE!! Curtailment not calculated, processing skipped")
    pass    
###################################################################            

 
#Stacked_Gen_read = pd.read_hdf(hdf_out_folder + "/" + HDF5_output, 'generation_Curtailment')

#storage = db.storage("Generation")
#storage = df_process_storage(storage, overlap)