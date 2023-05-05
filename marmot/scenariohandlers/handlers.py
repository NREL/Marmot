from abc import ABC, abstractmethod
from marmot.datahelpers import calc_curtailment
import pandas as pd
import json
import os
from marmot.scenariohandlers.config_maps import tech_map_simple, egret_map_simple, plexos_map_simple
from functools import lru_cache
from glob import iglob
from marmot.datahelpers.parsers import *


#Need a pre-determined set of curtailable technology. Global value that can be updated
curt_tech = ['Wind',"Offshore-Wind", 'PV','dPV']


def calc_curtailment(gen_tech, avail_tech):

    avail_curt_tech = [val for val in avail_tech.columns.get_level_values(level='Technology').unique() if val in curt_tech]
    gen_curt_tech = [val for val in gen_tech.columns.get_level_values(level='Technology').unique() if val in curt_tech]


    curt_gen = pd.concat([avail_tech[avail_curt_tech], -1*gen_tech[gen_curt_tech]]).groupby(level=0).sum()
    # TODO track down negative curtailment
    return curt_gen.applymap(lambda x: x if x>=0.0 else 0.0)

def load_map(map):
    if type(map) == dict:
        return map
    elif type(map) == str:
        if os.path.exists(map):
            return json.loads(open(map,'r').read())
        else:
            print("path does not exist")
            return None
    else:
        print("Unable to open map")
        return None

class BaseScenario(ABC):


    def __init__(self, scenario_path, tech_map, gen_entity_map, load_entity_map, line_rating_map) -> None:
        #super().__init__()

        # Can be single file or directory of multiple files.
        # Up to the users
        self._scenario_path = scenario_path
        self._tech_map = load_map(tech_map)
        self._tech_simple = load_map(tech_map_simple)
        self._gen_entity_map = load_map(gen_entity_map)
        self._load_entity_map = load_map(load_entity_map)
        self._line_rating_map = load_map(line_rating_map)


    #Implement abstract classes below to gain access to others.
    @abstractmethod
    @lru_cache(1)
    def get_raw_generators(self):
        ...

    @abstractmethod
    @lru_cache(1)
    def get_raw_availability(self):
        ...

    @abstractmethod
    @lru_cache(1)
    def get_regional_load(self):
        ...

    @abstractmethod
    def get_line_flow_data(self):
        ...

    def simplify_technology(self, tech):

        if tech in self._tech_simple:
            return self._tech_simple[tech]
        else:
            return tech

    def get_generators(self):
        """ Looks for a saved generation dataset, if not available, calls the get_raw_generators concrete class"""
        cache_path = os.path.normpath(f'{self._scenario_path}/marmot_cache/generation_actual.pq.gz')
        if os.path.exists(cache_path):
            return pd.read_parquet(cache_path)
        else:
            print("generation dataset not present, calculating from raw files")
            df = self.get_raw_generators()

            if os.path.exists(f'{self._scenario_path}/marmot_cache') == False:
                os.makedirs(f'{self._scenario_path}/marmot_cache')
            df.to_parquet(cache_path)
            return df


    def get_availability(self):
        """ Looks for a saved availability dataset, if not available, calls the get_raw_availability concrete class"""
        cache_path = f'{self._scenario_path}/marmot_cache/generation_availability.pq.gz'
        if os.path.exists(cache_path):
            return pd.read_parquet(cache_path)
        else:
            print("availability dataset not present, calculating from raw files")
            df = self.get_raw_availability()
            if os.path.exists(f'{self._scenario_path}/marmot_cache') == False:
                os.makedirs(f'{self._scenario_path}/marmot_cache')
            df.to_parquet(cache_path)
            return df


    def get_generators_tech(self):

        """Returns a Dataframe with a column for each generator along with technology category"""

        gen_df = self.get_generators()

        # TODO replace simple tech map with standard EIA tech map
        if self._tech_simple != None:
            gen_df.columns = pd.MultiIndex.from_tuples([(self.simplify_technology(self._tech_map[col]), col) for col in gen_df.columns], names=['Technology', 'Generator'])
        else:
            gen_df.columns = pd.MultiIndex.from_tuples([(self._tech_map[col], col) for col in gen_df.columns], names=['Technology', 'Generator'])

        gen_df.attrs["Units"] = "MW"
        return gen_df

    def get_availability_tech(self, simplify=True):
        avail_df = self.get_availability()

        # TODO replace simple tech map with standard EIA tech map

        if self._tech_simple != None:
            avail_df.columns = pd.MultiIndex.from_tuples([(self.simplify_technology(self._tech_map[col]), col) for col in avail_df.columns],names=['Technology', 'Generator'])
        else:
            avail_df.columns = pd.MultiIndex.from_tuples([(self._tech_map[col], col) for col in avail_df.columns],names=['Technology', 'Generator'])

        avail_df.attrs["Units"] = "MW"
        return avail_df

    def get_curtailment(self):
        gen_tech = self.get_generators_tech()
        avail_tech = self.get_availability_tech()

        curt_tech = calc_curtailment(gen_tech, avail_tech)
        return curt_tech

    def get_gen_and_curtailment(self):
        gen_tech = self.get_generators_tech()
        avail_tech = self.get_availability_tech()

        curt_tech = calc_curtailment(gen_tech, avail_tech)

        curt_tech.columns = pd.MultiIndex.from_tuples([("Curtailment", col[1]) for col in curt_tech.columns], names=['Technology','Generator'])

        return pd.merge(gen_tech, curt_tech, left_index=True, right_index=True)

    # aggregates across generators
    def get_entity_tech_aggregates(self):

        gen_curt_tech = self.get_gen_and_curtailment()

        gen_curt_tech.columns = pd.MultiIndex.from_tuples([(self._gen_entity_map[col[1]], col[0],col[1]) for col in gen_curt_tech.columns], names=['Entity','Technology','Generator'])

        return gen_curt_tech

    def get_entity_load(self):

        df = self.get_regional_load()

        df.columns = pd.MultiIndex.from_tuples([
        (self._load_entity_map[str(col)], str(col)) if col in self._load_entity_map.keys() else ("other", str(col)) for col in df.columns
        ])

        return df

    def get_entity_tech_load_aggregates(self):

        regional_load = self.get_entity_load()
        regional_load_agg = regional_load.groupby(axis=1, level=0).sum()
        regional_load_agg.columns = pd.MultiIndex.from_tuples([(col, "Demand") for col in regional_load_agg.columns], names=['Entity', 'Technology'])

        gen_tech_load = self.get_entity_tech_aggregates()

        gen_tech_load_agg = gen_tech_load.groupby(axis=1, level=['Entity','Technology']).sum()

        gen_tech_load_entity = pd.merge(regional_load_agg, gen_tech_load_agg, left_index=True, right_index=True).sort_index(axis=1)
        gen_tech_load_entity.attrs['units'] = 'MW'

        return gen_tech_load_entity

    def get_entity_curtailment_aggregates(self):
        curt_entity = self.get_curtailment()
        curt_entity.columns = pd.MultiIndex.from_tuples([(self._gen_entity_map[col[1]], col[0],col[1]) for col in curt_entity.columns], names=['Entity', 'Technology','Generator'])

        return curt_entity.groupby(level=['Entity', 'Technology'], axis=1).sum()


    # Flow APIs

    def get_line_loading(self):

        flow = self.get_line_flow_data()

        loading = pd.DataFrame({col: abs(flow[col]/self._line_rating_map[col]) for col in flow.columns.values }, index=flow.index)

        return loading


    def get_line_utilization(self, threshold=[99,95,90,75]):

        loading = self.get_line_loading()
        frames = []
        for thresh in threshold:
            print(f"Calculating U{thresh} utilization")
            utilization = loading.applymap(lambda x: True if x >= thresh else False )
            utilization.columns = pd.MultiIndex.from_tuples([(f"U{str(thresh)}", col) for col in utilization.columns])
            frames.append(utilization)

        return pd.concat(frames, axis=1)

    @lru_cache(1)
    def get_line_congestion_hours(self, threshold=100.0):

        loading = self.get_line_loading()

        congestion = loading.applymap(lambda x: False if x < threshold else True)
        # TODO
        """
        Is line and hour at 100%
        """

        return congestion






class SIIPScenario(BaseScenario):


    def __init__(self, scenario_path,  gen_entity_map=None, load_entity_map=None, line_rating_map=None) -> None:
        super().__init__(scenario_path, tech_map=None, gen_entity_map=gen_entity_map, load_entity_map=load_entity_map, line_rating_map=line_rating_map)

        self._metadata = load_map(f"{scenario_path}/metadata.json")
        self._tech_map = self._metadata['Generator_fuel_mapping']
        self._lines_meta = self._metadata['Lines']
        # TODO for some reasone "rate" is throwing a key error even though it is present...
        # self._line_rating_map = {key:val['rate'] for key, val in self._lines_meta.items()}
        self._line_rating_map = pd.DataFrame.from_dict(self._lines_meta).transpose()['rate'].to_dict()



    def get_raw_generators(self):
        return pd.read_parquet(f"{self._scenario_path}/generation_actual.pq.gz")

    def get_raw_availability(self):
        return pd.read_parquet(f"{self._scenario_path}/generation_availability.pq.gz")

    def get_regional_load(self):
        return pd.read_parquet(f"{self._scenario_path}/regional_load.pq.gz")

    def get_line_flow_data(self):
        return pd.read_parquet(f"{self._scenario_path}/power_flow_actual.pq.gz")


class PlexosScenario(BaseScenario):

    def __init__(self, scenario_path,  gen_entity_map=None, load_entity_map=None, line_rating_map=None) -> None:
        super().__init__(scenario_path, tech_map=None, gen_entity_map=gen_entity_map, load_entity_map=load_entity_map, line_rating_map=line_rating_map)

        self._template_file = get_plexos_paths(scenario_path)[0]
        self._tech_map = get_h5_gen_tech_map(self._template_file)
        self._tech_simple = load_map(plexos_map_simple)

        self._gen_entity_map = get_h5_gen_region_map(self._template_file)
        self._load_entity_map = get_h5_region_region_map(self._template_file)


    def get_raw_generators(self):
        return agg_plexos_generation(self._scenario_path)

    def get_raw_availability(self):
        return agg_plexos_availability(self._scenario_path)


    def get_regional_load(self):
        return agg_plexos_load(self._scenario_path)

    def get_line_flow_data(self):
        return NotImplemented

    def set_entity_map(self, entity_map: str ) -> None:
        if entity_map.lower() == 'zone':
            self._gen_entity_map = get_h5_gen_zone_map(self._template_file)
            self._load_entity_map = get_h5_region_zone_map(self._template_file)
        elif entity_map.lower() == 'region':
            self._gen_entity_map = get_h5_gen_region_map(self._template_file)
            self._load_entity_map = get_h5_region_region_map(self._template_file)



class EGRETScenario(BaseScenario):

    def __init__(self, scenario_path,  gen_entity_map=None, load_entity_map=None, line_rating_map=None) -> None:
        super().__init__(scenario_path, tech_map=None, gen_entity_map=gen_entity_map, load_entity_map=load_entity_map, line_rating_map=line_rating_map)

        self._scenario_path = scenario_path
        self._tech_map = self.get_gen_tech_map()
        self._tech_simple = load_map(egret_map_simple)

    def list_simulation_files(self):
        """
        Returns list of simulation files to parse.
        May need Regex if directory has multiple types of json files.
        """

        return [file for file in iglob(f'{self._scenario_path}/*.json')]

    @lru_cache(5)
    def load_egret_file(self, file_path):
        return json.loads(open(file_path, 'r').read())

    def parse_egret_files(self, file_path):

        return NotImplemented


    def get_raw_generators(self):
        """Loops through the json files and aggregates to a generator dataframe"""
        files = self.list_simulation_files()

        frames = []
        for filepath in files:

            egret_obj = json.loads(open(filepath, 'r').read())

            df = self.get_generator_dataframe(egret_obj)

            # TODO JSON files might overlap timewindows
            frames.append(df)


        return pd.concat(frames)

    def get_raw_availability(self):
        return NotImplemented

    def get_regional_load(self):
        return NotImplemented

    def get_line_flow_data(self):
        return NotImplemented


    def get_generator_dataframe(self, egret_obj):

        """ Extracts the generator data from a parsed json file."""
        generators = egret_obj['elements']['generator'].keys()

        data = {gen: egret_obj['elements']['generator'][gen]['pg']['values'] for gen in generators}
        timestamps = egret_obj['system']['time_keys']

        df = pd.DataFrame(data, index=pd.to_datetime(timestamps))

        return df


    def get_gen_entity_map(self):
        """Loads the generator bus map based on first json file"""
        file_path = self.list_simulation_files()[0]
        egret_obj = self.load_egret_file(file_path)

        # TODO might be beneficial to pass in the entity key if you
        # want something besides generator to bus mappings.
        return self.get_generator_map(egret_obj, 'bus')

    def get_gen_tech_map(self):

        """Loads the generator egret tech map based on first json file"""
        file_path = self.list_simulation_files()[0]
        egret_obj = self.load_egret_file(file_path)
        return self.get_generator_map(egret_obj, 'fuel')


    def get_generator_map(self, egret_obj, gen_key):

        """Creates a Generator - Entity/Technology map based on loaded json obj."""
        generators = egret_obj['elements']['generator']
        return {gen: value[gen_key] for gen, value in generators.items()}

