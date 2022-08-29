import pandas as pd
import h5py

# fn = '/Users/mschwarz/WaterRisk local/test hydro inputs outputs/Processed_HDF5_folder/weekly_hydro_test_optdecomp_formatted.h5'

# fn = '/Volumes/PLEXOS CEII/Projects/Railbelt/Analysis/250tx/Processed_HDF5_folder/Scenario 3_formatted.h5'
# gen = pd.read_hdf(fn,'generator_Generation')
# gen = gen.reset_index(['timestamp','tech'])
# gen = gen.groupby(["timestamp", "tech"], as_index=False, observed=True).sum()
# gen = gen.pivot(index='timestamp', columns='tech', values=0)
# gen.to_csv('/Users/mschwarz/Railbelt local/Thermals/Scenario3_gen.csv')

fn = '/Users/mschwarz/CCS_local/CSU_202202_offline/Runs/CSU_Ref200x2050/plexos_export/solutions/Processed_HDF5_folder/Base_20502012_formatted.h5'
#fn = '/Volumes/PLEXOS CEII/Projects/FLECCS/ReEDS_Runs/Processed_HDF5_folder/r3_CO2_100n35_NoCCS2035_formatted.h5'
keys = h5py.File(fn).keys()
em = pd.read_hdf(fn,'emission_Production')
em = em.reset_index()
em = em.pivot(index = 'timestamp',columns = 'emission_type',values = 0)
em.to_csv('/Users/mschwarz/CCS_local/CSU_202202_offline/Runs/CSU_Ref200x2050/plexos_export/solutions/CO2_8760_CSU_Ref200x2050.csv')

gen = pd.read_hdf(fn,'generator_Generation')
hydro = gen.xs('Hydro_1',level = 'gen_name')
hydro.index = pd.to_datetime(hydro.index.get_level_values(0))
hydro['week'] = hydro.index.isocalendar().week
hydro_weekly = hydro.groupby('week').sum()
hydro.to_csv('/Volumes/PLEXOS CEII/Projects/WaterRisk/weekly hydro tests/weekly_hydro_daily_optdecomp_paritions.csv')

cost = pd.read_hdf(fn,'generator_Generation_Cost')
em = pd.read_hdf(fn,'emissions_generators_Production')

gens = {'CAISO':'gas-cc_p10_26',
        'ERCOT':'gas-cc_p65_11',
        'MISO-W':'gas-cc_p46_3',
        'NYISO':'gas-cc_p127_18',
        'PJM-W':'gas-cc_p112_8'}

writer = pd.ExcelWriter("/Users/mschwarz/CCS_local/example_NGCC_data.xlsx")

for gen_name in gens:
    print(gen_name)
    print(gens[gen_name])
    single_gen = gen.xs(gens[gen_name],level = 'gen_name')
    single_gen.index = single_gen.index.droplevel(['tech','region','State','country','Country','units'])

    single_cost = cost.xs(gens[gen_name],level = 'gen_name')
    single_cost.index = single_cost.index.droplevel(['tech','region','State','country','Country','units'])
    single_cost *= 1.367434246 #2004->2020 $ year

    single_em = em.xs(gens[gen_name],level = 'gen_name')
    single_em = single_em.xs('CO2',level = 'pollutant')
    single_em.index = single_em.index.droplevel(['tech','region','State','country','Country','units'])
    single_em /= 1000

    comb = pd.concat([single_gen,single_cost,single_em],axis = 1)
    comb.columns = ['Generation (MW)','Generation cost ($2020)','Emissions (tCO2)']
    comb.to_excel(writer,sheet_name = f'{gen_name} ({gens[gen_name]})')
writer.save()

