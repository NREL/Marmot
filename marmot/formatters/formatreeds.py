
import logging
import re
import gdxpds
import pandas as pd
from pathlib import Path
from typing import List
from dataclasses import dataclass, field
from marmot.formatters.formatbase import Process

logger = logging.getLogger('marmot_format.'+__name__)

    

class ProcessReEDS(Process):
    """Process ReEDS  specific data from a ReEDS result set.
    """
    def __init__(self, input_folder: Path, Region_Mapping: pd.DataFrame, 
                *args, subset_years: list=None, **kwargs):
        """
        Args:
            input_folder (Path): Folder containing csv files.
            Region_Mapping (pd.DataFrame): DataFrame to map custom 
                regions/zones to create custom aggregations.
            plexos_block (str, optional): PLEXOS results type. Defaults to 'ST'.
        """

        self.gdx_datafiles: dict = {}
        # Internal cached data is saved to the following variables.
        # To access the values use the public api e.g self.property_units
        self._property_units: dict = {}

        self.subset_years = subset_years
        # Instantiation of Process Base class
        super().__init__(input_folder, Region_Mapping, *args, **kwargs) 

    @property
    def property_units(self):
        """Gets the property units from data, e.g MW, MWh

        Returns:
            dict: _property_units
        """
        return self._property_units

    @property_units.setter
    def property_units(self, gdx_filename: str):
        """Sets the property units, adds values to a dict

        Args:
            gdx_filename (str): Full path to gdx_file
        """
        #Extracts values between markers
        symbol_marker = "--(.*?)--"

        symbol_list = gdxpds.list_symbols(gdx_filename)
        for symbol in symbol_list:
            unit = re.search(symbol_marker, symbol.description)
            if unit:
                unit = unit.group(1)
            if symbol.name not in self._property_units:
                self._property_units[symbol.name] = unit


    def get_input_files(self) -> list:
        """Gets a list of input files within the scenario folders
        """

        reeds_outputs_dir = self.input_folder.joinpath('outputs')
        files = []
        for names in reeds_outputs_dir.iterdir():
            if names.name == f"rep_{input_folder.name}.gdx":
                files.append(names.name)
                
                self.property_units = str(names)                
                
        # List of all files in input folder in alpha numeric order
        files_list = sorted(files, key=lambda x:int(re.sub('\D', '', x)))
        for file in files_list:
            self.gdx_datafiles[file] = str(reeds_outputs_dir.joinpath(file))
        return files_list

    def get_processed_data(self, data_class: str, prop: str, 
                  timescale: str, model_filename: str) -> pd.DataFrame:
        """Handles the pulling of data from the ReEDS gdx
        file and then passes the data to one of the formating functions

        Args:
            data_class (str): Data class e.g Region, Generator, Zone etc
            prop (str): Property e.g gen_out, cap_out etc.
            timescale (str): Data timescale, e.g interval, summary.
            model_filename (str): name of model to process.

        Returns:
            pd.DataFrame: Formatted results dataframe.
        """
        gdx_file = self.gdx_datafiles.get(model_filename)

        df = gdxpds.to_dataframe(gdx_file, prop)[prop]

        reeds_prop_cols = PropertyColumns()
        df.columns = getattr(reeds_prop_cols, prop)
        df.year = df.year.astype(int)
        if self.subset_years:
            df = df.loc[df.year.isin(self.subset_years)]

        if timescale == 'interval':
            df = self.merge_timeseries_block_data(df)
        else:
            df['timestamp'] = pd.to_datetime(df.year.astype(str))
            
        if 'year' in df.columns:
            df = df.drop(['year'], axis=1)

        # Get desired method, used for extra processing if needed
        process_att = getattr(self, f'df_process_{data_class}', None)
        if process_att:
            # Process attribute and return to df
            df = process_att(df)

        df_col = list(df.columns)
        df_col.remove('Value')
        df_col.insert(0, df_col.pop(df_col.index("timestamp")))
        df = df.set_index(df_col)
        df = df.sort_index(level=['timestamp'])

        df_units = self.property_units[prop]
        # find unit conversion values
        converted_units = self.UNITS_CONVERSION.get(df_units, (df_units, 1))

        # Convert units and add unit column to index 
        df = df*converted_units[1]
        units_index = pd.Index([converted_units[0]] * len(df), name='units')
        df.set_index(units_index, append=True, inplace=True) 
        return df

    def merge_timeseries_block_data(self, df: pd.DataFrame) -> pd.DataFrame:

        timeslice_mapping_file = pd.read_csv(self.input_folder.joinpath('inputs_case', 
                                                    'h_dt_szn.csv'))

        # All year timeslice mappings are the same, defaulting to 2007 
        timeslice_mapping_file = timeslice_mapping_file.loc[timeslice_mapping_file.year == 2007]
        timeslice_mapping_file = timeslice_mapping_file.drop('year', axis=1)
        
        year_list = df.year.unique()
        year_list.sort()

        year_chunks = []
        for year in year_list:
            year_chunks.extend(pd.date_range(f"{year}-01-01", 
                                periods=8760, freq='H'))
        
        datetime_df = pd.DataFrame(data=list(range(1,8761))*len(year_list),
                                    index=pd.to_datetime(year_chunks), 
                                    columns=['hour'])
        datetime_df['year'] = datetime_df.index.year.astype(int)
        datetime_df = datetime_df.reset_index()
        datetime_df.rename(columns={'index': 'timestamp'},
                            inplace=True)

        datetime_block = datetime_df.merge(timeslice_mapping_file, on='hour')
        datetime_block.sort_values(by=['timestamp','h'], inplace=True)
        df_merged =  df.merge(datetime_block, on=['year', 'h'])
        return df_merged.drop(['h','hour', 'year'], axis=1)  


    def df_process_generator(self, df: pd.DataFrame) -> pd.DataFrame:
        
        if 'tech' not in df.columns:
            df['tech'] = 'reeds_vre'
        return df

    def df_process_line(self, df: pd.DataFrame) -> pd.DataFrame:

        df['line_name'] = df['region_from'] + "_" + df['region_to']
        return df

    def def_process_reserve(self, df: pd.DataFrame) -> pd.DataFrame:

        df['Type'] = '-'
        return df

@dataclass
class PropertyColumns():

    gen_out: List = field(default_factory=lambda: ['tech', 'region', 'h', 
                                                    'year', 'Value'])  #Marmot generator_Generation
    cap_out: List = field(default_factory=lambda: ['tech', 'region', 'year', 
                                                    'Value']) #Marmot generator_Installed_Capacity
    curt_out: List = field(default_factory=lambda: ['region', 'h', 'year', 
                                                    'Value']) #Marmot generator_Curtailment
    load_rt: List = field(default_factory=lambda: ['region', 'year', 'Value']) #Marmot region_Load (year)
    losses_tran_h: List = field(default_factory=lambda: ['region_from', 'region_to', 
                                                        'h', 'year', 'category', 'Value']) #Marmot line_Losses 
    tran_flow_power: List = field(default_factory=lambda: ['region_from', 'region_to', 
                                                        'h', 'category', 'year', 'Value']) #Marmot line_Flow
    tran_out: List = field(default_factory=lambda: ['region_from', 'region_to', 
                                                    'category', 'year', 'Value']) #Marmot line_Import_Limit                                            
    stor_in: List = field(default_factory=lambda: ['tech', 'sub-tech', 'region', 'h', 
                                                    'year', 'Value']) #Marmot generator_Pumped_Load
    stor_energy_cap: List = field(default_factory=lambda: ['tech', 'sub-tech', 'region', 
                                                            'year', 'Value']) #Marmot storage_Max_Volume
    emit_nat_tech: List = field(default_factory=lambda: ['emission_type', 'tech', 'year', 'Value'])
    emit_r: List = field(default_factory=lambda: ['emission_type', 'region', 'year', 'Value']) # Marmot emission_Production (year)
    opRes_supply_h: List = field(default_factory=lambda: ['parent', 'tech', 'region', 'h', 'year', 'Value']) # Marmot reserves_generators_Provision
    systemcost_tech_ba: List = field(default_factory=lambda: ['cost_type', 'tech', 'region', 'year', 'Value']) # Marmot generator_Total Generation Cost


@dataclass
class MarmotPropertyMapping():

    gen_out: str = 'generator_Generation'
    cap_out: str = 'generator_Installed_Capacity'
    curt_out: str = 'generator_Curtailment'
    load_rt: str = 'region_Load'
    losses_tran_h: str = 'line_Losses'
    tran_flow_power: str = 'line_Flow'
    tran_out: str = 'line_Import_Limit'
    stor_in: str = 'generator_Pumped_Load'
    stor_energy_cap: str = 'storage_Max_Volume'
    emit_nat_tech: str = ''
    emit_r: str = 'emission_Production'
    opRes_supply_h: str = 'reserves_generators_Provision'


input_folder = Path(r"\\nrelnas01\ReEDS\Users\pbrown\ReEDSruns\20220223_bokeh\v20220223_bokehspurD0_ref_seq")
Region_Mapping = pd.DataFrame()
emit_names = pd.DataFrame()
subset_years = [2050]
model = ProcessReEDS(input_folder, Region_Mapping, emit_names, subset_years=subset_years)
print(model.get_input_files())
# prop = 'gen_out'
model_filename = 'rep_v20220223_bokehspurD0_ref_seq.gdx'
# gen_df = model.get_processed_data('generator', prop, 'interval', model_filename)
prop = 'losses_tran_h'
df = model.get_processed_data('generator', prop, 'interval', model_filename)



# gen_df.xs(slice('2045-01-01', '2045-12-31'), level='timestamp', drop_level=False)