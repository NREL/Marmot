# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 16:14:56 2020

@author: rhousema
"""

# This is the new MetaData class
# Comments regarding the fix for generator_storage (head_storage & tail_storage) line 95
# Comments for interregional/intraregional line relations (Marmot_Scenario, Region_Mapping) line 204


import h5py
import os
import pandas as pd
import numpy as np



class MetaData:

# Marmot_Scenario and Region mapping were referred to in the original setup of one pickle file
# There are comments regarding that code below (line )
    
    def __init__(self, HDF5_folder_in, Region_Mapping, model=None):
        self.HDF5_folder_in = HDF5_folder_in
        self.Region_Mapping = Region_Mapping
        if model == None: 
            startdir=os.getcwd()
            os.chdir(self.HDF5_folder_in)     
            files = sorted(os.listdir()) 
            os.chdir(startdir)
            files_list = []
            for names in files:
                if names.endswith(".h5"):
                    files_list.append(names) # Creates a list of only the hdf5 files
            model=files_list[0]
        self.data = h5py.File(os.path.join(self.HDF5_folder_in, model), 'r')
        

# These methods are called in the Process class within results_formatter        

# Generator categories mapping
    def generator_category(self):
        try:
            gen_category = pd.DataFrame(np.asarray(self.data['metadata/objects/generators']))
        except KeyError:
            gen_category = pd.DataFrame(np.asarray(self.data['metadata/objects/generator']))
        gen_category.rename(columns={'name':'gen_name'}, inplace=True)
        gen_category.rename(columns={'category':'tech'}, inplace=True)
        gen_category = gen_category.applymap(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
        return gen_category
    
    
    
# Region generators mapping
    def region_generators(self):
        try:
            region_gen = pd.DataFrame(np.asarray(self.data['metadata/relations/regions_generators']))
        except KeyError:
            region_gen = pd.DataFrame(np.asarray(self.data['metadata/relations/region_generators']))
        region_gen.rename(columns={'child':'gen_name'}, inplace=True)
        region_gen.rename(columns={'parent':'region'}, inplace=True)
        region_gen = region_gen.applymap(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
        region_gen.drop_duplicates(subset=["gen_name"],keep='first',inplace=True) #For generators which belong to more than 1 region, drop duplicates.
        return region_gen
        
    def region_generator_category(self):
        region_gen = self.region_generators()
        gen_category = self.generator_category()
        region_gen_cat = region_gen.merge(gen_category,
                        how="left", on='gen_name').sort_values(by=['tech','gen_name']).set_index('region')
        return region_gen_cat
        
     # Zone generators mapping
    def zone_generators(self):
        try:
            zone_gen = pd.DataFrame(np.asarray(self.data['metadata/relations/zones_generators']))
        except KeyError:
            zone_gen = pd.DataFrame(np.asarray(self.data['metadata/relations/zone_generators']))    
        zone_gen.rename(columns={'child':'gen_name'}, inplace=True)
        zone_gen.rename(columns={'parent':'zone'}, inplace=True)
        zone_gen = zone_gen.applymap(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
        zone_gen.drop_duplicates(subset=["gen_name"],keep='first',inplace=True) #For generators which belong to more than 1 region, drop duplicates.
        return zone_gen
        
    def zone_generator_category(self): 
        zone_gen = self.zone_generators()
        gen_category = self.generator_category()
        zone_gen_cat = zone_gen.merge(gen_category,
                        how="left", on='gen_name').sort_values(by=['tech','gen_name']).set_index('zone')
        return zone_gen_cat
        
# Generator storage has been updated so that only one of tail_storage & head_storage is required
# If they are both available both are used just as before
    
    def generator_storage(self):
        head_tail = [0,0]
        # I assume its preferable to use both head_storage & tail_storage if they're available
        # Check for all options (generator or generators for both and use whatever is available)
        generator_headstorage = pd.DataFrame()
        generator_tailstorage = pd.DataFrame()
        
        try:
            generator_headstorage = pd.DataFrame(np.asarray(self.data['metadata/relations/generators_headstorage']))
            head_tail[0] = 1
        except KeyError:
            pass
        
        try:
            generator_headstorage = pd.DataFrame(np.asarray(self.data['metadata/relations/generator_headstorage']))
            head_tail[0] = 1
        except KeyError:
            pass
        
        try:
            generator_tailstorage = pd.DataFrame(np.asarray(self.data['metadata/relations/generators_tailstorage']))
            head_tail[1] = 1
        except KeyError:
            pass
        
        try:
            generator_tailstorage = pd.DataFrame(np.asarray(self.data['metadata/relations/generator_tailstorage']))
            head_tail[1] = 1
        except KeyError:
            pass
        
        if head_tail[0] == 1:
            if head_tail[1] == 1:
                gen_storage = pd.concat([generator_headstorage, generator_tailstorage])
            else:
                gen_storage = generator_headstorage
        else:
            gen_storage = generator_tailstorage
            
    
        # Original Code does not account for scenario where either head_storage or tail_storage 
        # does not exist within metadata relations 
        
        # try:
        #     generator_headstorage = pd.DataFrame(np.asarray(self.data['metadata/relations/generators_headstorage']))
        #     generator_tailtorage = pd.DataFrame(np.asarray(self.data['metadata/relations/generators_tailstorage']))
        # except KeyError:
        #     generator_headstorage = pd.DataFrame(np.asarray(self.data['metadata/relations/generator_headstorage']))
        #     generator_tailtorage = pd.DataFrame(np.asarray(self.data['metadata/relations/generator_tailstorage']))
        # gen_storage = pd.concat([generator_headstorage, generator_tailtorage])
                
        gen_storage.rename(columns={'child':'name'}, inplace=True)
        gen_storage.rename(columns={'parent':'gen_name'}, inplace=True)
        gen_storage = gen_storage.applymap(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
        return gen_storage
    
    def node_region(self):
        try:
            node_region = pd.DataFrame(np.asarray(self.data['metadata/relations/nodes_region']))
        except KeyError:
            node_region = pd.DataFrame(np.asarray(self.data['metadata/relations/node_region']))
        node_region.rename(columns={'child':'region','parent':'node'}, inplace=True)
        node_region = node_region.applymap(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
        node_region = node_region.sort_values(by=['node']).set_index('region')
        return node_region
        
    def node_zone(self):
        try:
            node_zone = pd.DataFrame(np.asarray(self.data['metadata/relations/nodes_zone']))
        except KeyError:
            node_zone = pd.DataFrame(np.asarray(self.data['metadata/relations/node_zone']))
        node_zone.rename(columns={'child':'zone','parent':'node'}, inplace=True)
        node_zone = node_zone.applymap(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
        node_zone = node_zone.sort_values(by=['node']).set_index('zone')
        return node_zone
    
    
##############################################################################
    
# These methods were not originally part of the MetaData class and were used to 
# Setup the pickle files for metadata.  They are now methods that can be called and return
# return that data instead of having to read a pickle file.

    # returns metadata regions
    def regions(self):
        try:
            try:
                regions = pd.DataFrame(np.asarray(self.data['metadata/objects/regions']))
            except KeyError:
                regions = pd.DataFrame(np.asarray(self.data['metadata/objects/region']))
            regions["name"]=regions["name"].str.decode("utf-8")
            regions["category"]=regions["category"].str.decode("utf-8")
            regions.rename(columns={'name':'region'}, inplace=True)
            regions.sort_values(['category','region'],inplace=True)
            return regions   
        except KeyError:
            print("\Regional data not included in h5plexos results.\nSkipping Regional properties\n")
    
    
    # returns metadata zones
    def zones(self):
        try:
            try:
                zones = pd.DataFrame(np.asarray(self.data['metadata/objects/zones']))
            except KeyError:
                zones = pd.DataFrame(np.asarray(self.data['metadata/objects/zone']))
            zones["name"]=zones["name"].str.decode("utf-8")
            zones["category"]=zones["category"].str.decode("utf-8")
            #zones.to_pickle(self.Marmot_Scenario +"/zones.pkl")
            return zones
        except KeyError:
            print("\nZonal data not included in h5plexos results.\nSkipping Zonal properties\n")
            
    
    # return metadata line relations
    def line_relations(self):
        try:
            try:
                line_relations=pd.DataFrame(np.asarray(self.data['metadata/objects/lines']))
            except KeyError:
                line_relations=pd.DataFrame(np.asarray(self.data['metadata/objects/line']))
            line_relations["name"]=line_relations["name"].str.decode("utf-8")
            line_relations["category"]=line_relations["category"].str.decode("utf-8")
            return line_relations
        except KeyError:
            print("\nLine data not included in h5plexos results.\nSkipping Line property\n")


#################################################################################
    # The following method was the one referring to region mapping and marmot scenario
    # It also has some issues/confusion with variable names
################################################################################            
            
# Interregional and Intraregional line relation metadata.
# This code references variables Region_Mapping & Marmot Scenario
# They are now passed in as parameters for the MetaData class
            
            
# I think the some of the variable names in this section could be improved
# There was atleast one error as well

            
            
    # return metadata interregional line relations
#     def regional_line_relations(self):
#         try:
#             try:
#                 line_relations_interregional=pd.DataFrame(np.asarray(self.data['metadata/relations/region_interregionallines']))
#                 line_relations_intraregional=pd.DataFrame(np.asarray(self.data['metadata/relations/region_intraregionallines']))
    
#             except KeyError:
#                 line_relations_interregional=pd.DataFrame(np.asarray(self.data['metadata/relations/region_interregionalline']))
#                 line_relations_intraregional=pd.DataFrame(np.asarray(self.data['metadata/relations/region_intraregionalline']))        
            
#             line_relations_interregional["parent"]=line_relations_interregional["parent"].str.decode("utf-8")
#             line_relations_interregional["child"]= line_relations_interregional["child"].str.decode("utf-8")
#             line_relations_interregional.rename(columns={"parent":"region","child":"line_name"},inplace=True)
# # Region Mapping referenced
#             line_relations_interregional=pd.merge(line_relations_interregional,self.Region_Mapping,how='left',on="region")
# # Marmot Scenario referenced   
#             line_relations_interregional.to_pickle(self.Marmot_Scenario +"/line_relations_interregional.pkl")   
#         except KeyError:      
#             print("\nLine data not included in h5plexos results.\nSkipping Line property\n")
            
            
    
#     # Get line <-> region mapping and save to pickle. Combine inter and intra regional lines.
#         try:
#             region_exportinglines = pd.DataFrame(np.asarray(self.data['metadata/relations/region_exportinglines']))
#             region_exportinglines["region"] = region_exportinglines["parent"].str.decode("utf-8")
#             region_exportinglines["line"] = region_exportinglines["child"].str.decode("utf-8")
#             region_exportinglines = region_exportinglines.drop(columns = ['parent','child'])
    
# # Is this the same line_relations_intraregional variable as above, just a different name?
#             region_intraregionallines = pd.DataFrame(np.asarray(self.data['metadata/relations/region_intraregionallines']))
           
#             region_intraregionallines["region"] = region_intraregionallines["parent"].str.decode("utf-8")
#             region_intraregionallines["line"] = region_intraregionallines["child"].str.decode("utf-8")
#             region_intraregionallines = region_intraregionallines.drop(columns = ['parent','child'])
    
#             region_lines = region_exportinglines.append(region_intraregionallines)
#             region_lines.to_pickle(self.Marmot_Scenario +"/line2region.pkl")
#         except KeyError:
#             print("\nLine relation data not included in h5plexos results.\nSkipping Line property\n") 
            
            
    def regional_line_relations(self):
        try:
            try:
                line_relations_interregional=pd.DataFrame(np.asarray(self.data['metadata/relations/region_interregionallines']))
            except KeyError:
                line_relations_interregional=pd.DataFrame(np.asarray(self.data['metadata/relations/region_interregionalline']))
            
            line_relations_interregional["parent"]=line_relations_interregional["parent"].str.decode("utf-8")
            line_relations_interregional["child"]= line_relations_interregional["child"].str.decode("utf-8")
            line_relations_interregional.rename(columns={"parent":"region","child":"line_name"},inplace=True)
            line_relations_interregional=pd.merge(line_relations_interregional,self.Region_Mapping,how='left',on="region")
            return line_relations_interregional
        except KeyError:      
            print("\nLine data not included in h5plexos results.\nSkipping Line property\n")
            
            
    
    # Get line <-> region mapping and save to pickle. Combine inter and intra regional lines.
    def region_lines(self):
        try:
            try:
                region_exportinglines = pd.DataFrame(np.asarray(self.data['metadata/relations/region_exportinglines']))
                region_intraregionallines = pd.DataFrame(np.asarray(self.data['metadata/relations/region_intraregionallines']))
            except KeyError:
                region_exportinglines = pd.DataFrame(np.asarray(self.data['metadata/relations/region_exportingline']))
                region_intraregionallines = pd.DataFrame(np.asarray(self.data['metadata/relations/region_intraregionalline']))
    # Try decoding all in one line
            region_exportinglines["region"] = region_exportinglines["parent"].str.decode("utf-8")
            region_exportinglines["line_name"] = region_exportinglines["child"].str.decode("utf-8")
            region_exportinglines = region_exportinglines.drop(columns = ['parent','child'])
               
            region_intraregionallines["region"] = region_intraregionallines["parent"].str.decode("utf-8")
            region_intraregionallines["line_name"] = region_intraregionallines["child"].str.decode("utf-8")
            region_intraregionallines = region_intraregionallines.drop(columns = ['parent','child'])
            
            region_lines = region_exportinglines.append(region_intraregionallines)
            return region_lines
        except KeyError:
            print("\nLine relation data not included in h5plexos results.\nSkipping Line property\n") 
            
            
####################################################################################   
            
####################################################################################
            
## Get line <-> interface mapping and save to pickle.
    def interface_line_relations(self):
        try:
            interface_lines = pd.DataFrame(np.asarray(self.data['metadata/relations/interface_lines']))
            interface_lines["interface"] = interface_lines["parent"].str.decode("utf-8")
            interface_lines["line"] = interface_lines["child"].str.decode("utf-8")
            interface_lines = interface_lines.drop(columns = ['parent','child'])
            return interface_lines
        
        
        except KeyError:
            print("\nLine relation data not included in h5plexos results.\nSkipping Line property\n")
 


###############################################################################
            
###############################################################################


            





