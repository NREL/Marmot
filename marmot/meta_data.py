# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 16:14:56 2020

Module to retrieve metadata from 
prdocution cost modelling results 
@author: Ryan Houseman
"""

import os
import sys
import h5py
import pandas as pd
import numpy as np
import logging

class MetaData():
    """
    Class to handle the retreival of metadata from the formatted or
    original plexos solution h5 files

    Attributes
    ----------
    filename : str
        The name of the h5 file to retreive data from  
    h5_data : h5 file
        loaded h5 file in memory
    """

    filename = None
    h5_data = None
    
    def __init__(self, HDF5_folder_in, read_from_formatted_h5=True, 
                    Region_Mapping=pd.DataFrame(), partition_number=0):
        """
        Parameters
        ----------
        HDF5_folder_in : folder
            Folder containing h5plexos h5 files .
        read_from_formatted_h5 : Boolean, optional
            Boolean for whether the metadata is being read from the 
            formatted hdf5 file or the original PLEXOS solution file.
            default True
        Region_Mapping : pd.DataFrame, optional
            DataFrame of extra regions to map.
        partition_number : int, optional
            Which temporal partition of h5 data to retrieve metadata from
            in the formatted h5 file, default is 0 (first entry)

        """
        self.logger = logging.getLogger('marmot_format.'+__name__)        
        self.HDF5_folder_in = HDF5_folder_in
        self.Region_Mapping = Region_Mapping
        self.read_from_formatted_h5 = read_from_formatted_h5
        self.partition_number = partition_number

        self.start_index = None

    @classmethod
    def _check_if_exhisting_filename(cls, filename) -> bool:
        """
        classmethod to check if the passed filename
        is the same or different from previous calls
        if different replaces the filename with new value 
        and closes old file
        """
        
        if cls.filename != filename:
            cls.filename = filename
            cls.close_h5()
            return False
        elif cls.filename == filename:
            return True

    @classmethod
    def close_h5(cls) -> None:
        """Closes h5 file open in memory"""
        if cls.h5_data:
            cls.h5_data.close()
    
    def _read_data(self, filename) -> None:
        """Reads h5 file into memory"""
        
        self.logger.debug(f"Reading New h5 file: {filename}")
        processed_file_format = "{}_formatted.h5"
        
        try:  
            if self.read_from_formatted_h5:

                filename = processed_file_format.format(filename)
                self.h5_data = h5py.File(os.path.join(self.HDF5_folder_in, filename), 'r')
                partitions = [key for key in self.h5_data['metadata'].keys()]
                if self.partition_number > len(partitions):
                    self.logger.warning(f"\nYou have chosen to use metadata partition_number {self.partition_number}, "
                                    f"But there are only {len(partitions)} partitions in your formatted h5 file.\n"
                                    "Defaulting to partition_number 0")
                    self.partition_number = 0

                self.start_index = f"metadata/{partitions[self.partition_number]}/"
            else:
                self.h5_data = h5py.File(os.path.join(self.HDF5_folder_in, filename), 'r')
                self.start_index = "metadata/"

        except OSError:
            if self.read_from_formatted_h5:
                self.logger.warning("Unable to find processed HDF5 file to retrieve metadata.\n"
                                    "Check scenario name.")
                return
            else:
                self.logger.info("\nIn order to initialize your database's metadata, "
                                    "Marmot is looking for a h5plexos solution file.\n"
                                    f"It is looking in {self.HDF5_folder_in}, but it cannot "
                                    "find any *.h5 files there.\n"
                                    "Please check the 'PLEXOS_Solutions_folder' input in row 2 of your "
                                    "'Marmot_user_defined_inputs.csv'.\n"
                                    "Ensure that it matches the filepath containing the *.h5 files "
                                    "created by h5plexos.\n\nMarmot will now quit.")
                sys.exit()     

    def generator_category(self, filename) -> pd.DataFrame:
        """Generator categories mapping"""
        
        if not self._check_if_exhisting_filename(filename):
            self._read_data(filename)

        try:    
            try:
                gen_category = pd.DataFrame(np.asarray(self.h5_data[self.start_index + 'objects/generator']))
            except KeyError:
                gen_category = pd.DataFrame(np.asarray(self.h5_data[self.start_index + 'objects/generators']))
            gen_category.rename(columns={'name':'gen_name','category':'tech'}, inplace=True)
            gen_category = gen_category.applymap(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)   
        except KeyError:
            gen_category = pd.DataFrame()

        return gen_category
            
    def region_generators(self, filename) -> pd.DataFrame:
        """Region generators mapping"""
        
        if not self._check_if_exhisting_filename(filename):
            self._read_data(filename)

        try:    
            try:
                region_gen = pd.DataFrame(np.asarray(self.h5_data[self.start_index + 'relations/regions_generators']))
            except KeyError:
                region_gen = pd.DataFrame(np.asarray(self.h5_data[self.start_index + 'relations/region_generators']))
            region_gen.rename(columns={'child':'gen_name','parent':'region'}, inplace=True)
            region_gen = region_gen.applymap(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
            region_gen.drop_duplicates(subset=["gen_name"],keep='first',inplace=True) #For generators which belong to more than 1 region, drop duplicates.
        except KeyError:
            region_gen = pd.DataFrame()
        
        return region_gen
        
    def region_generator_category(self, filename) -> pd.DataFrame:
        """"Region generators category mapping"""
                
        try:
            region_gen = self.region_generators(filename)
            gen_category = self.generator_category(filename)
            region_gen_cat = region_gen.merge(gen_category,
                            how="left", on='gen_name').sort_values(by=['tech','gen_name']).set_index('region')  
        except KeyError:
            region_gen_cat = pd.DataFrame()

        return region_gen_cat
        
    def zone_generators(self, filename) -> pd.DataFrame:
        """Zone generators mapping"""
        
        if not self._check_if_exhisting_filename(filename):
            self._read_data(filename)
        try:
            try:
                zone_gen = pd.DataFrame(np.asarray(self.h5_data[self.start_index + 'relations/zones_generators']))
            except KeyError:
                zone_gen = pd.DataFrame(np.asarray(self.h5_data[self.start_index + 'relations/zone_generators']))    
            zone_gen.rename(columns={'child':'gen_name','parent':'zone'}, inplace=True)
            zone_gen = zone_gen.applymap(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
            zone_gen.drop_duplicates(subset=["gen_name"],keep='first',inplace=True) #For generators which belong to more than 1 region, drop duplicates.
        except KeyError:
            zone_gen = pd.DataFrame()
        
        return zone_gen
        
    def zone_generator_category(self, filename) -> pd.DataFrame:
        """Zone generators category mapping"""
                
        try:
            zone_gen = self.zone_generators(filename)
            gen_category = self.generator_category(filename)
            zone_gen_cat = zone_gen.merge(gen_category,
                            how="left", on='gen_name').sort_values(by=['tech','gen_name']).set_index('zone')  
        except KeyError:
            zone_gen_cat = pd.DataFrame()

        return zone_gen_cat
        
    # Generator storage has been updated so that only one of tail_storage & head_storage is required
    # If both are available, both are used
    def generator_storage(self, filename) -> pd.DataFrame:
        """Generator Storage mapping"""
        
        if not self._check_if_exhisting_filename(filename):
            self._read_data(filename)
        head_tail = [0,0]
        try:    
            generator_headstorage = pd.DataFrame()
            generator_tailstorage = pd.DataFrame()
            try:
                generator_headstorage = pd.DataFrame(np.asarray(self.h5_data[self.start_index + 'relations/generators_headstorage']))
                head_tail[0] = 1
            except KeyError:
                pass
            try:
                generator_headstorage = pd.DataFrame(np.asarray(self.h5_data[self.start_index + 'relations/generator_headstorage']))
                head_tail[0] = 1
            except KeyError:
                pass
            try:
                generator_headstorage = pd.DataFrame(np.asarray(self.h5_data[self.start_index + 'relations/exportinggenerators_headstorage']))
                head_tail[0] = 1
            except KeyError:
                pass
            try:
                generator_tailstorage = pd.DataFrame(np.asarray(self.h5_data[self.start_index + 'relations/generators_tailstorage']))
                head_tail[1] = 1
            except KeyError:
                pass
            try:
                generator_tailstorage = pd.DataFrame(np.asarray(self.h5_data[self.start_index + 'relations/generator_tailstorage']))
                head_tail[1] = 1
            except KeyError:
                pass
            try:
                generator_tailstorage = pd.DataFrame(np.asarray(self.h5_data[self.start_index + 'relations/importinggenerators_tailstorage']))
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
        except:
            gen_storage = pd.DataFrame()
        
        return gen_storage
    
    def node_region(self, filename) -> pd.DataFrame:
        """Node Region mapping"""
        
        if not self._check_if_exhisting_filename(filename):
            self._read_data(filename)
        try:
            try:
                node_region = pd.DataFrame(np.asarray(self.h5_data[self.start_index + 'relations/nodes_region']))
            except KeyError:
                node_region = pd.DataFrame(np.asarray(self.h5_data[self.start_index + 'relations/node_region']))
            node_region.rename(columns={'child':'region','parent':'node'}, inplace=True)
            node_region = node_region.applymap(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
            node_region = node_region.sort_values(by=['node']).set_index('region') 
        except:
            node_region = pd.DataFrame()

        return node_region
        
    def node_zone(self, filename) -> pd.DataFrame:
        """Node zone mapping"""
        
        if not self._check_if_exhisting_filename(filename):
            self._read_data(filename)
        try:
            try:
                node_zone = pd.DataFrame(np.asarray(self.h5_data[self.start_index + 'relations/nodes_zone']))
            except KeyError:
                node_zone = pd.DataFrame(np.asarray(self.h5_data[self.start_index + 'relations/node_zone']))
            node_zone.rename(columns={'child':'zone','parent':'node'}, inplace=True)
            node_zone = node_zone.applymap(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
            node_zone = node_zone.sort_values(by=['node']).set_index('zone')   
        except:
            node_zone = pd.DataFrame()
        
        return node_zone
    
    def generator_node(self, filename) -> pd.DataFrame:
        """generator node mapping"""
        
        if not self._check_if_exhisting_filename(filename):
            self._read_data(filename)
        try:
            try:
                generator_node = pd.DataFrame(np.asarray(self.h5_data[self.start_index + 'relations/generators_nodes']))
            except KeyError:
                generator_node = pd.DataFrame(np.asarray(self.h5_data[self.start_index + 'relations/generator_nodes']))
            generator_node.rename(columns={'child':'node','parent':'gen_name'}, inplace=True)
            generator_node = generator_node.applymap(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
            # generators_nodes = generators_nodes.sort_values(by=['generator'])   
        except:
            generator_node = pd.DataFrame()

        return generator_node
        
    def regions(self, filename) -> pd.DataFrame:
        """Region objects"""
        
        if not self._check_if_exhisting_filename(filename):
            self._read_data(filename)

        try:
            try:
                regions = pd.DataFrame(np.asarray(self.h5_data[self.start_index + 'objects/regions']))
            except KeyError:
                regions = pd.DataFrame(np.asarray(self.h5_data[self.start_index + 'objects/region']))
            regions = regions.applymap(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
            regions.rename(columns={'name':'region'}, inplace=True)
            regions.sort_values(['category','region'],inplace=True)
        except KeyError:
            self.logger.warning("Regional data not included in h5plexos results")
            regions = pd.DataFrame()
        
        return regions   
    
    def zones(self, filename) -> pd.DataFrame:
        """Zone objects"""
        
        if not self._check_if_exhisting_filename(filename):
            self._read_data(filename)
        try:
            try:
                zones = pd.DataFrame(np.asarray(self.h5_data[self.start_index + 'objects/zones']))
            except KeyError:
                zones = pd.DataFrame(np.asarray(self.h5_data[self.start_index + 'objects/zone']))
            zones = zones.applymap(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
        except KeyError:
            self.logger.warning("Zonal data not included in h5plexos results")
            zones = pd.DataFrame()

        return zones
             
    def lines(self, filename) -> pd.DataFrame:
        """Line objects"""
        
        if not self._check_if_exhisting_filename(filename):
            self._read_data(filename)
        try:
            try:
                lines=pd.DataFrame(np.asarray(self.h5_data[self.start_index + 'objects/lines']))
            except KeyError:
                lines=pd.DataFrame(np.asarray(self.h5_data[self.start_index + 'objects/line']))
            lines = lines.applymap(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
            lines.rename(columns={"name":"line_name"},inplace=True)
        except KeyError:
            self.logger.warning("Line data not included in h5plexos results")

        return lines
    
    def region_regions(self, filename) -> pd.DataFrame:
        """Region-region mapping"""
        
        if not self._check_if_exhisting_filename(filename):
            self._read_data(filename)
        try:
            region_regions = pd.DataFrame(np.asarray(self.h5_data[self.start_index + 'relations/region_regions']))
            region_regions = region_regions.applymap(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
        except KeyError:
            self.logger.warning("region_regions data not included in h5plexos results")

        return region_regions
    
    def region_interregionallines(self, filename) -> pd.DataFrame:
        """Region inter-regional lines mapping"""
        
        if not self._check_if_exhisting_filename(filename):
            self._read_data(filename)
        try:
            try:
                region_interregionallines=pd.DataFrame(np.asarray(self.h5_data[self.start_index + 'relations/region_interregionallines']))
            except KeyError:
                region_interregionallines=pd.DataFrame(np.asarray(self.h5_data[self.start_index + 'relations/region_interregionalline']))
            
            region_interregionallines = region_interregionallines.applymap(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
            region_interregionallines.rename(columns={"parent":"region","child":"line_name"},inplace=True)
            if not self.Region_Mapping.empty:
                region_interregionallines=pd.merge(region_interregionallines,self.Region_Mapping,how='left',on="region") 
        except KeyError:
            region_interregionallines = pd.DataFrame()
            self.logger.warning("Region Interregionallines data not included in h5plexos results")

        return region_interregionallines
            
    def region_intraregionallines(self, filename) -> pd.DataFrame:
        """Region intra-regional lines mapping"""

        if not self._check_if_exhisting_filename(filename):
            self._read_data(filename)
        try:
            try:
                region_intraregionallines=pd.DataFrame(np.asarray(self.h5_data[self.start_index + 'relations/region_intraregionallines']))
            except KeyError:
                try:
                    region_intraregionallines=pd.DataFrame(np.asarray(self.h5_data[self.start_index + 'relations/region_intraregionalline']))
                except KeyError:
                    region_intraregionallines=pd.concat([pd.DataFrame(np.asarray(self.h5_data[self.start_index + 'relations/region_importinglines'])),
                                                            pd.DataFrame(np.asarray(self.h5_data[self.start_index + 'relations/region_exportinglines']))]).drop_duplicates()
            region_intraregionallines = region_intraregionallines.applymap(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
            region_intraregionallines.rename(columns={"parent":"region","child":"line_name"},inplace=True)
            if not self.Region_Mapping.empty:
                region_intraregionallines=pd.merge(region_intraregionallines,self.Region_Mapping,how='left',on="region")
        except KeyError: 
            region_intraregionallines = pd.DataFrame()
            self.logger.warning("Region Intraregionallines Lines data not included in h5plexos results")  
        
        return region_intraregionallines
      
    def region_exporting_lines(self, filename) -> pd.DataFrame:
        """Region exporting lines mapping"""

        if not self._check_if_exhisting_filename(filename):
            self._read_data(filename)
        try:
            try:
                region_exportinglines = pd.DataFrame(np.asarray(self.h5_data[self.start_index + 'relations/region_exportinglines']))
            except KeyError:
                region_exportinglines = pd.DataFrame(np.asarray(self.h5_data[self.start_index + 'relations/region_exportingline']))
            region_exportinglines = region_exportinglines.applymap(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
            region_exportinglines = region_exportinglines.rename(columns={'parent':'region','child':'line_name'})
            if not self.Region_Mapping.empty:
                region_exportinglines=pd.merge(region_exportinglines,self.Region_Mapping,how='left',on="region")
        except KeyError:
            self.logger.warning("Region Exporting Lines data not included in h5plexos results") 

        return region_exportinglines 
      
    def region_importing_lines(self, filename) -> pd.DataFrame:
        """Region importing lines mapping"""

        if not self._check_if_exhisting_filename(filename):
            self._read_data(filename)
        try:
            try:
                region_importinglines = pd.DataFrame(np.asarray(self.h5_data[self.start_index + 'relations/region_importinglines']))
            except KeyError:
                region_importinglines = pd.DataFrame(np.asarray(self.h5_data[self.start_index + 'relations/region_importingline']))
            region_importinglines = region_importinglines.applymap(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
            region_importinglines = region_importinglines.rename(columns={'parent':'region','child':'line_name'})
            if not self.Region_Mapping.empty:
                region_importinglines=pd.merge(region_importinglines,self.Region_Mapping,how='left',on="region")
        except KeyError:
            self.logger.warning("Region Importing Lines data not included in h5plexos results") 

        return region_importinglines

    def zone_interzonallines(self, filename) -> pd.DataFrame:
        """zone inter-zonal lines mapping"""
        
        if not self._check_if_exhisting_filename(filename):
            self._read_data(filename)
        try:
            try:
                zone_interzonallines=pd.DataFrame(np.asarray(self.h5_data[self.start_index + 'relations/zone_interzonallines']))
            except KeyError:
                zone_interzonallines=pd.DataFrame(np.asarray(self.h5_data[self.start_index + 'relations/zone_interzonalline']))
            
            zone_interzonallines = zone_interzonallines.applymap(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
            zone_interzonallines.rename(columns={"parent":"region","child":"line_name"},inplace=True)
        except KeyError:      
            zone_interzonallines = pd.DataFrame()
            self.logger.warning("Zone Interzonallines data not included in h5plexos results")
        
        return zone_interzonallines
                
    def zone_intrazonallines(self, filename) -> pd.DataFrame:
        """zone intra-zonal lines mapping"""
        
        if not self._check_if_exhisting_filename(filename):
            self._read_data(filename)
        try:
            try:
                zone_intrazonallines=pd.DataFrame(np.asarray(self.h5_data[self.start_index + 'relations/zone_intrazonallines']))
            except KeyError:
                zone_intrazonallines=pd.DataFrame(np.asarray(self.h5_data[self.start_index + 'relations/zone_intrazonalline']))
            zone_intrazonallines = zone_intrazonallines.applymap(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
            zone_intrazonallines.rename(columns={"parent":"region","child":"line_name"},inplace=True)
        except KeyError:      
            zone_intrazonallines = pd.DataFrame()
            self.logger.warning("Zone Intrazonallines Lines data not included in h5plexos results") 

        return zone_intrazonallines
                 
    def zone_exporting_lines(self, filename) -> pd.DataFrame:
        """zone exporting lines mapping"""
        
        if not self._check_if_exhisting_filename(filename):
            self._read_data(filename)
        try:
            try:
                zone_exportinglines = pd.DataFrame(np.asarray(self.h5_data[self.start_index + 'relations/zone_exportinglines']))
            except KeyError:
                zone_exportinglines = pd.DataFrame(np.asarray(self.h5_data[self.start_index + 'relations/zone_exportingline']))
            zone_exportinglines = zone_exportinglines.applymap(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
            zone_exportinglines = zone_exportinglines.rename(columns={'parent':'region','child':'line_name'})
        except KeyError:
            self.logger.warning("zone exporting lines data not included in h5plexos results") 
            zone_exportinglines = pd.DataFrame()

        return zone_exportinglines 
    
    def zone_importing_lines(self, filename) -> pd.DataFrame:
        """zone importing lines mapping"""
        
        if not self._check_if_exhisting_filename(filename):
            self._read_data(filename)
        try:
            try:
                zone_importinglines = pd.DataFrame(np.asarray(self.h5_data[self.start_index + 'relations/zone_importinglines']))
            except KeyError:
                zone_importinglines = pd.DataFrame(np.asarray(self.h5_data[self.start_index + 'relations/zone_importingline']))
            zone_importinglines = zone_importinglines.applymap(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
            zone_importinglines = zone_importinglines.rename(columns={'parent':'region','child':'line_name'})
        except KeyError:
            self.logger.warning("zone importing lines data not included in h5plexos results") 
            zone_importinglines = pd.DataFrame()
        
        return zone_importinglines 

    def interface_lines(self, filename) -> pd.DataFrame:
        """Interface -> lines mapping"""
        
        if not self._check_if_exhisting_filename(filename):
            self._read_data(filename)
        try:
            try:
                interface_lines = pd.DataFrame(np.asarray(self.h5_data[self.start_index + 'relations/interface_lines']))
            except KeyError:
                interface_lines = pd.DataFrame(np.asarray(self.h5_data[self.start_index + 'relations/interfaces_lines']))
            interface_lines = interface_lines.applymap(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
            interface_lines = interface_lines.rename(columns={'parent':'interface','child':'line'})
        except KeyError:
            self.logger.warning("Interface Lines data not included in h5plexos results")

        return interface_lines

    def region_lines(self, filename) -> pd.DataFrame:
        """Lines within each region""" 
        
        region_interregionallines = self.region_interregionallines(filename)
        region_intraregionallines = self.region_intraregionallines(filename)
        region_lines = pd.concat([region_interregionallines,region_intraregionallines])
        return region_lines
    
    def zone_lines(self, filename) -> pd.DataFrame:
        """Lines within each zone""" 
        
        zone_interzonallines = self.zone_interzonallines(filename)
        zone_intrazonallines = self.zone_intrazonallines(filename)
        zone_lines = pd.concat([zone_interzonallines,zone_intrazonallines])
        zone_lines = zone_lines.rename(columns={'region':'zone'})
        return zone_lines

    def reserves(self, filename) -> pd.DataFrame:
        """Reserves objects"""
        
        if not self._check_if_exhisting_filename(filename):
            self._read_data(filename)
        try:
            try:
                reserves = pd.DataFrame(np.asarray(self.h5_data[self.start_index + 'objects/reserves']))
            except KeyError:
                reserves = pd.DataFrame(np.asarray(self.h5_data[self.start_index + 'objects/reserve']))
            reserves = reserves.applymap(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
        except KeyError:
            self.logger.warning("Reserves data not included in h5plexos results") 
 
        return reserves 
            
    def reserves_generators(self, filename) -> pd.DataFrame:
        """Reserves generators mapping"""
        
        if not self._check_if_exhisting_filename(filename):
            self._read_data(filename)
        try:
            try:
                reserves_generators = pd.DataFrame(np.asarray(self.h5_data[self.start_index + 'relations/reserves_generators']))
            except KeyError:
                reserves_generators = pd.DataFrame(np.asarray(self.h5_data[self.start_index + 'relations/reserve_generators']))
            reserves_generators = reserves_generators.applymap(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
            reserves_generators = reserves_generators.rename(columns={'child':'gen_name'})
        except KeyError:
            self.logger.warning("Reserves data not included in h5plexos results") 
            reserves_generators = pd.DataFrame()
        
        return reserves_generators 

    def reserves_regions(self, filename) -> pd.DataFrame:
        """Reserves regions mapping"""
        
        reserves_generators = self.reserves_generators(filename)
        region_generators = self.region_generators(filename)
        try:
            reserves_regions = reserves_generators.merge(region_generators, how='left', on='gen_name')
        except KeyError:
            self.logger.warning("Reserves Region data not available in h5plexos results") 
            return pd.DataFrame()
        if not self.Region_Mapping.empty:
            reserves_regions=pd.merge(reserves_regions,self.Region_Mapping,how='left',on="region")
        reserves_regions.drop('gen_name', axis=1, inplace=True)
        reserves_regions.drop_duplicates(inplace=True)
        reserves_regions.reset_index(drop=True,inplace=True)
        return reserves_regions
        
    def reserves_zones(self, filename) -> pd.DataFrame:
        """Reserves zones mapping"""
        
        reserves_generators = self.reserves_generators(filename)
        zone_generators = self.zone_generators(filename)
        try:
            reserves_zones = reserves_generators.merge(zone_generators, how='left', on='gen_name')
        except KeyError:
            self.logger.warning("Reserves Zone data not available in h5plexos results") 
            return pd.DataFrame()
        reserves_zones.drop('gen_name', axis=1, inplace=True)
        reserves_zones.drop_duplicates(inplace=True)
        reserves_zones.reset_index(drop=True,inplace=True)
        return reserves_zones
