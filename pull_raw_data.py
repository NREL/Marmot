import pandas as pd
import os

os.chdir('../Processed_HDF5_folder')

#for scen in ['Cap100','Cap100_MinCF']:
for scen in ['Cap100_MinCF']:
    gen = pd.read_hdf(f'{scen}_formatted.h5','generator_Generation')
    gen = gen.reset_index()
    gen_tot = gen.groupby(['gen_name','tech','region','State','Country']).sum()/1000

    cap = pd.read_hdf(f'{scen}_formatted.h5','generator_Installed_Capacity')
    cap = cap.reset_index()
    cap = cap.groupby(['gen_name','tech','region','State','Country']).sum()

    cf = gen_tot*1000/(cap*8760)

    gen_tot.to_csv(f'{scen}_generation_totals_byplant.csv')
    cf.to_csv(f'{scen}_annual_capacityfactor_byplant.csv')
    #gen.to_csv(f'{scen}_generation_byplant.csv')
    #cap.to_csv(f'{scen}_capacity_byplant.csv')

