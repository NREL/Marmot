# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 16:14:56 2020

@author: Ryan Houseman
"""

# This is the new MetaData class

import h5py
import os
import pandas as pd
import numpy as np
import logging
import sys

class MetaData:
    
    def __init__(self, HDF5_folder_in, Region_Mapping, model=None):
        """
        Parameters
        ----------
        HDF5_folder_in : folder
            Folder containing h5plexos h5 files .
        Region_Mapping : DataFrame
            DataFrame of extra regions to map.
        model : string, optional
            Name of model h5 file. The default is None.
        """
        self.logger = logging.getLogger('marmot_format.'+__name__)        
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
            if len(files_list) == 0:
                self.logger.info("\n In order to initialize your database's metadata, Marmot is looking for an h5plexos solution file.  \n It is looking in " + self.HDF5_folder_in + ", but it cannot find any *.h5 files there. \n Please check the 'PLEXOS_Solutions_folder' input in row 2 of your 'Marmot_user_defined_inputs.csv'. \n Ensure that it matches the filepath containing the *.h5 files created by h5plexos. \n \n Marmot will now quit.")
                sys.exit()
            else:
                model=files_list[0]
        self.data = h5py.File(os.path.join(self.HDF5_folder_in, model), 'r')
        
# These methods are called in the Process class within results_formatter
# If metadata does not contain that object or relation and empty dataframe is returned        

# Generator categories mapping
    def generator_category(self):
        try:    
            try:
                gen_category = pd.DataFrame(np.asarray(self.data['metadata/objects/generators']))
            except KeyError:
                gen_category = pd.DataFrame(np.asarray(self.data['metadata/objects/generator']))
            gen_category.rename(columns={'name':'gen_name','category':'tech'}, inplace=True)
            gen_category = gen_category.applymap(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
            return gen_category
        except:
            generator_category = pd.DataFrame()
            return generator_category
    
# Region generators mapping
    def region_generators(self):
        try:    
            try:
                region_gen = pd.DataFrame(np.asarray(self.data['metadata/relations/regions_generators']))
            except KeyError:
                region_gen = pd.DataFrame(np.asarray(self.data['metadata/relations/region_generators']))
            region_gen.rename(columns={'child':'gen_name','parent':'region'}, inplace=True)
            region_gen = region_gen.applymap(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
            region_gen.drop_duplicates(subset=["gen_name"],keep='first',inplace=True) #For generators which belong to more than 1 region, drop duplicates.
            return region_gen
        except:
            region_generators = pd.DataFrame()
            return region_generators
        
    def region_generator_category(self):
        try:
            region_gen = self.region_generators()
            gen_category = self.generator_category()
            region_gen_cat = region_gen.merge(gen_category,
                            how="left", on='gen_name').sort_values(by=['tech','gen_name']).set_index('region')
            return region_gen_cat
        except:
            region_generator_category = pd.DataFrame()
            return region_generator_category
        
# Zone generators mapping
    def zone_generators(self):
        try:
            try:
                zone_gen = pd.DataFrame(np.asarray(self.data['metadata/relations/zones_generators']))
            except KeyError:
                zone_gen = pd.DataFrame(np.asarray(self.data['metadata/relations/zone_generators']))    
            zone_gen.rename(columns={'child':'gen_name','parent':'zone'}, inplace=True)
            zone_gen = zone_gen.applymap(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
            zone_gen.drop_duplicates(subset=["gen_name"],keep='first',inplace=True) #For generators which belong to more than 1 region, drop duplicates.
            return zone_gen
        except:
            zone_generators = pd.DataFrame()
            return zone_generators
        
    def zone_generator_category(self): 
        try:
            zone_gen = self.zone_generators()
            gen_category = self.generator_category()
            zone_gen_cat = zone_gen.merge(gen_category,
                            how="left", on='gen_name').sort_values(by=['tech','gen_name']).set_index('zone')
            return zone_gen_cat
        except:
            zone_generator_category = pd.DataFrame()
            return zone_generator_category
        
# Generator storage has been updated so that only one of tail_storage & head_storage is required
# If both are available, both are used
    
    def generator_storage(self):
        head_tail = [0,0]
        try:    
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
            gen_storage.rename(columns={'child':'name','parent':'gen_name'}, inplace=True)
            gen_storage = gen_storage.applymap(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
            return gen_storage
        except:
            generator_storage = pd.DataFrame()
            return generator_storage
    
    def node_region(self):
        try:
            try:
                node_region = pd.DataFrame(np.asarray(self.data['metadata/relations/nodes_region']))
            except KeyError:
                node_region = pd.DataFrame(np.asarray(self.data['metadata/relations/node_region']))
            node_region.rename(columns={'child':'region','parent':'node'}, inplace=True)
            node_region = node_region.applymap(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
            node_region = node_region.sort_values(by=['node']).set_index('region')
            return node_region
        except:
            node_region = pd.DataFrame()
            return node_region
        
    def node_zone(self):
        try:
            try:
                node_zone = pd.DataFrame(np.asarray(self.data['metadata/relations/nodes_zone']))
            except KeyError:
                node_zone = pd.DataFrame(np.asarray(self.data['metadata/relations/node_zone']))
            node_zone.rename(columns={'child':'zone','parent':'node'}, inplace=True)
            node_zone = node_zone.applymap(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
            node_zone = node_zone.sort_values(by=['node']).set_index('zone')
            return node_zone
        except:
            node_zone = pd.DataFrame()
            return node_zone
    
##############################################################################
    
# These methods were not originally part of the MetaData class and were used to 
# Setup the pickle files for metadata.  They are now methods that can be called and
# return that data instead of having to read a pickle file.

    # returns metadata regions
    def regions(self):
        try:
            try:
                regions = pd.DataFrame(np.asarray(self.data['metadata/objects/regions']))
            except KeyError:
                regions = pd.DataFrame(np.asarray(self.data['metadata/objects/region']))
            regions = regions.applymap(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
            regions.rename(columns={'name':'region'}, inplace=True)
            regions.sort_values(['category','region'],inplace=True)
            return regions   
        except KeyError:
            self.logger.warning("Regional data not included in h5plexos results")
            return pd.DataFrame()
    
    def zones(self):
        try:
            try:
                zones = pd.DataFrame(np.asarray(self.data['metadata/objects/zones']))
            except KeyError:
                zones = pd.DataFrame(np.asarray(self.data['metadata/objects/zone']))
            zones = zones.applymap(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
            return zones
        except KeyError:
            self.logger.warning("Zonal data not included in h5plexos results")
            return pd.DataFrame()
             
    def lines(self):
        try:
            try:
                lines=pd.DataFrame(np.asarray(self.data['metadata/objects/lines']))
            except KeyError:
                lines=pd.DataFrame(np.asarray(self.data['metadata/objects/line']))
            lines = lines.applymap(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
            lines.rename(columns={"name":"line_name"},inplace=True)
            return lines
        except KeyError:
            self.logger.warning("Line data not included in h5plexos results")
    
    def region_regions(self):
        try:
            region_regions = pd.DataFrame(np.asarray(self.data['metadata/relations/region_regions']))
            region_regions = region_regions.applymap(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
            return region_regions
        except KeyError:
            self.logger.warning("region_regions data not included in h5plexos results")
    
    
    def region_interregionallines(self):
        try:
            try:
                region_interregionallines=pd.DataFrame(np.asarray(self.data['metadata/relations/region_interregionallines']))
            except KeyError:
                region_interregionallines=pd.DataFrame(np.asarray(self.data['metadata/relations/region_interregionalline']))
            
            region_interregionallines = region_interregionallines.applymap(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
            region_interregionallines.rename(columns={"parent":"region","child":"line_name"},inplace=True)
            region_interregionallines=pd.merge(region_interregionallines,self.Region_Mapping,how='left',on="region")
            return region_interregionallines
        except KeyError:
            region_interregionallines = pd.DataFrame()
            self.logger.warning("Region Interregionallines data not included in h5plexos results")
            return region_interregionallines
            
      
    def region_intraregionallines(self):
       try:
           try:
               region_intraregionallines=pd.DataFrame(np.asarray(self.data['metadata/relations/region_intraregionallines']))
           except KeyError:
               region_intraregionallines=pd.DataFrame(np.asarray(self.data['metadata/relations/region_intraregionalline']))
           region_intraregionallines = region_intraregionallines.applymap(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
           region_intraregionallines.rename(columns={"parent":"region","child":"line_name"},inplace=True)
           region_intraregionallines=pd.merge(region_intraregionallines,self.Region_Mapping,how='left',on="region")
           return region_intraregionallines
       except KeyError: 
           region_intraregionallines = pd.DataFrame()
           self.logger.warning("Region Intraregionallines Lines data not included in h5plexos results")  
           return region_intraregionallines
                
      
    def region_exporting_lines(self):
      try:
          try:
              region_exportinglines = pd.DataFrame(np.asarray(self.data['metadata/relations/region_exportinglines']))
          except KeyError:
              region_exportinglines = pd.DataFrame(np.asarray(self.data['metadata/relations/region_exportingline']))
          region_exportinglines = region_exportinglines.applymap(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
          region_exportinglines = region_exportinglines.rename(columns={'parent':'region','child':'line_name'})
          region_exportinglines=pd.merge(region_exportinglines,self.Region_Mapping,how='left',on="region")
          return region_exportinglines 
      except KeyError:
          self.logger.warning("Region Exporting Lines data not included in h5plexos results") 
      
    def region_importing_lines(self):
        try:
            try:
                region_importinglines = pd.DataFrame(np.asarray(self.data['metadata/relations/region_importinglines']))
            except KeyError:
                region_importinglines = pd.DataFrame(np.asarray(self.data['metadata/relations/region_importingline']))
            region_importinglines = region_importinglines.applymap(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
            region_importinglines = region_importinglines.rename(columns={'parent':'region','child':'line_name'})
            region_importinglines=pd.merge(region_importinglines,self.Region_Mapping,how='left',on="region")
            return region_importinglines 
        except KeyError:
            self.logger.warning("Region Importing Lines data not included in h5plexos results") 

    def zone_interzonallines(self):
            try:
                try:
                    zone_interzonallines=pd.DataFrame(np.asarray(self.data['metadata/relations/zone_interzonallines']))
                except KeyError:
                    zone_interzonallines=pd.DataFrame(np.asarray(self.data['metadata/relations/zone_interzonalline']))
                
                zone_interzonallines = zone_interzonallines.applymap(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
                zone_interzonallines.rename(columns={"parent":"region","child":"line_name"},inplace=True)
                return zone_interzonallines
            except KeyError:      
                zone_interzonallines = pd.DataFrame()
                self.logger.warning("Zone Interzonallines data not included in h5plexos results")
                return zone_interzonallines
                
    def zone_intrazonallines(self):
       try:
           try:
               zone_intrazonallines=pd.DataFrame(np.asarray(self.data['metadata/relations/zone_intrazonallines']))
           except KeyError:
               zone_intrazonallines=pd.DataFrame(np.asarray(self.data['metadata/relations/zone_intrazonalline']))
           zone_intrazonallines = zone_intrazonallines.applymap(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
           zone_intrazonallines.rename(columns={"parent":"region","child":"line_name"},inplace=True)
           return zone_intrazonallines
       except KeyError:      
           zone_intrazonallines = pd.DataFrame()
           self.logger.warning("Zone Intrazonallines Lines data not included in h5plexos results") 
           return zone_intrazonallines
                 
    def zone_exporting_lines(self):
        try:
            try:
                zone_exportinglines = pd.DataFrame(np.asarray(self.data['metadata/relations/zone_exportinglines']))
            except KeyError:
                zone_exportinglines = pd.DataFrame(np.asarray(self.data['metadata/relations/zone_exportingline']))
            zone_exportinglines = zone_exportinglines.applymap(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
            zone_exportinglines = zone_exportinglines.rename(columns={'parent':'region','child':'line_name'})
            return zone_exportinglines 
        except KeyError:
            self.logger.warning("zone exporting lines data not included in h5plexos results") 
    
    def zone_importing_lines(self):
        try:
            try:
                zone_importinglines = pd.DataFrame(np.asarray(self.data['metadata/relations/zone_importinglines']))
            except KeyError:
                zone_importinglines = pd.DataFrame(np.asarray(self.data['metadata/relations/zone_importingline']))
            zone_importinglines = zone_importinglines.applymap(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
            zone_importinglines = zone_importinglines.rename(columns={'parent':'region','child':'line_name'})
            return zone_importinglines 
        except KeyError:
            self.logger.warning("zone importing lines data not included in h5plexos results") 

    def interface_lines(self):
            try:
                try:
                    interface_lines = pd.DataFrame(np.asarray(self.data['metadata/relations/interface_lines']))
                except KeyError:
                    interface_lines = pd.DataFrame(np.asarray(self.data['metadata/relations/interfaces_lines']))
                interface_lines = interface_lines.applymap(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
                interface_lines = interface_lines.rename(columns={'parent':'interface','child':'line'})
                return interface_lines
            except KeyError:
                self.logger.warning("Interface Lines data not included in h5plexos results")
    
    # All Regional lines
    def region_lines(self):
        region_interregionallines = self.region_interregionallines()
        region_intraregionallines = self.region_intraregionallines()
        region_lines = pd.concat([region_interregionallines,region_intraregionallines])
        return region_lines
    
    # All Zonal lines
    def zone_lines(self):
        zone_interzonallines = self.zone_interzonallines()
        zone_intrazonallines = self.zone_intrazonallines()
        zone_lines = pd.concat([zone_interzonallines,zone_intrazonallines])
        zone_lines = zone_lines.rename(columns={'region':'zone'})
        return zone_lines

    def reserves(self):
        try:
            try:
                reserves = pd.DataFrame(np.asarray(self.data['metadata/objects/reserves']))
            except KeyError:
                reserves = pd.DataFrame(np.asarray(self.data['metadata/objects/reserve']))
            reserves = reserves.applymap(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
            return reserves 
        except KeyError:
            self.logger.warning("Reserves data not included in h5plexos results") 
            
    def reserves_generators(self):
        try:
            try:
                reserves_generators = pd.DataFrame(np.asarray(self.data['metadata/relations/reserves_generators']))
            except KeyError:
                reserves_generators = pd.DataFrame(np.asarray(self.data['metadata/relations/reserve_generators']))
            reserves_generators = reserves_generators.applymap(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
            reserves_generators = reserves_generators.rename(columns={'child':'gen_name'})
            return reserves_generators 
        except KeyError:
            self.logger.warning("Reserves data not included in h5plexos results") 
            return pd.DataFrame()
    
    def reserves_regions(self):
        reserves_generators = self.reserves_generators()
        region_generators = self.region_generators()
        try:
            reserves_regions = reserves_generators.merge(region_generators, how='left', on='gen_name')
        except KeyError:
            self.logger.warning("Reserves Region data not available in h5plexos results") 
            return pd.DataFrame()
        reserves_regions=pd.merge(reserves_regions,self.Region_Mapping,how='left',on="region")
        reserves_regions.drop('gen_name', axis=1, inplace=True)
        reserves_regions.drop_duplicates(inplace=True)
        reserves_regions.reset_index(drop=True,inplace=True)
        return reserves_regions
        
    def reserves_zones(self):
        reserves_generators = self.reserves_generators()
        zone_generators = self.zone_generators()
        try:
            reserves_zones = reserves_generators.merge(zone_generators, how='left', on='gen_name')
        except KeyError:
            self.logger.warning("Reserves Zone data not available in h5plexos results") 
            return pd.DataFrame()
        reserves_zones.drop('gen_name', axis=1, inplace=True)
        reserves_zones.drop_duplicates(inplace=True)
        reserves_zones.reset_index(drop=True,inplace=True)
        return reserves_zones