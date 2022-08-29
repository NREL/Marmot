using H5PLEXOS
# cd("/Volumes/PLEXOS CEII/Projects/WaterRisk/weekly hydro tests")
# scenario = "weekly_hydro_daysteps_optdecomp"

process("/Volumes/ReEDS/FY22-ARPAE_FLECCS/CSU_202202/Runs_ColdStorageFix/v0311_CSU_Ref225x2050/plexos_export/Model model_2050_m1 Solution/Model model_2050_m1 Solution.zip",
    "/Volumes/ReEDS/FY22-ARPAE_FLECCS/CSU_202202/Runs_ColdStorageFix/v0311_CSU_Ref225x2050/plexos_export/Model model_2050_m1 Solution/Model model_2050_m1 Solution.h5")

#Full year run
#process("/Volumes/PLEXOS CEII/Projects/WaterRisk/Model ",scenario," Solution/Model ",scenario," Solution.zip","/Users/mschwarz/WaterRisk local/test hydro inputs outputs/weekly_hydro_tests_optdecomp_fullyear.h5")

#Partitions
# for part in 1:92
#     raw_sln_fn = string("Model ",scenario,"_092P_OLd000_",lpad(part,3,"0")," Solution/Model ",scenario,"_092P_OLd000_",lpad(part,3,"0")," Solution.zip")
#     new_fn = string("/Users/mschwarz/WaterRisk local/test hydro inputs outputs/weekly_hydro_tests_optdecomp_",lpad(part,3,"0"),".h5")
#     println(string("Converting ",part))
#     process(raw_sln_fn,new_fn)
# end



# cd("/Volumes/PLEXOS CEII/Projects/FLECCS/test cases/Model Thermal_stor_50load_150carbon Solution")
# process("Model Thermal_stor_50load_150carbon Solution.zip","CSU_50load_150carbon.h5")

# cd("/Volumes/PLEXOS CEII/Projects/FLECCS/test cases/Model Thermal_stor_50load_150carbon_NOCCS Solution")
# process("Model Thermal_stor_50load_150carbon_NOCCS Solution.zip","CSU_50load_150carbon.h5")


#process("Model Existing_2040 Solution/Model Existing_2040 Solution.zip","Analysis/250tx/Base case/Existing_2040.h5")

# process("Model MaxHydro_250tx Solution/Model MaxHydro_250tx Solution.zip","Analysis/250tx/Scenario 1/Scenario 1_250tx.h5")
# process("Model HighHydro_HighWind_250tx Solution/Model HighHydro_HighWind_250tx Solution.zip","Analysis/250tx/Scenario 2/Scenario 2_250tx.h5")
# process("Model ModHydro_HighWindSolar_250tx Solution/Model ModHydro_HighWindSolar_250tx Solution.zip","Analysis/250tx/Scenario 3/Scenario 3_250tx.h5")
# process("Model ModHydro_NewTech_250tx Solution/Model ModHydro_NewTech_250tx Solution.zip","Analysis/250tx/Scenario 4/Scenario 4_250tx.h5")
# process("Model LowHydro_NewTech_250tx Solution/Model LowHydro_NewTech_250tx Solution.zip","Analysis/250tx/Scenario 5/Scenario 5_250tx.h5")

# process("Model MaxHydro_189tx Solution/Model MaxHydro_189tx Solution.zip","Analysis/Scenario 1_189tx/Scenario 1_189tx.h5")
# process("Model HighHydro_HighWind_189tx Solution/Model HighHydro_HighWind_189tx Solution.zip","Analysis/Scenario 2_189tx/Scenario 2_189tx.h5")
# process("Model ModHydro_HighWindSolar_189tx Solution/Model ModHydro_HighWindSolar_189tx Solution.zip","Analysis/Scenario 3_189tx/Scenario 3_189tx.h5")
# process("Model ModHydro_NewTech_189tx Solution/Model ModHydro_NewTech_189tx Solution.zip","Analysis/Scenario 4_189tx/Scenario 4_189tx.h5")
# process("Model LowHydro_NewTech_189tx Solution/Model LowHydro_NewTech_189tx Solution.zip","Analysis/Scenario 5_189tx/Scenario 5_189tx.h5")

# process("Model HighHydro_HighWind_350tx Solution/Model HighHydro_HighWind_350tx Solution.zip","Analysis/Scenario 2_350tx/Scenario 2_350tx.h5")
# process("Model ModHydro_HighWindSolar_350tx Solution/Model ModHydro_HighWindSolar_350tx Solution.zip","Analysis/Scenario 3_350tx/Scenario 3_350tx.h5")
# process("Model ModHydro_NewTech_350tx Solution/Model ModHydro_NewTech_350tx Solution.zip","Analysis/Scenario 4_350tx/Scenario 4_350tx.h5")
# process("Model LowHydro_NewTech_350tx Solution/Model LowHydro_NewTech_350tx Solution.zip","Analysis/Scenario 5_350tx/Scenario 5_350tx.h5")

#process("Model Existing_2040_OUTAGE Solution/Model Existing_2040_OUTAGE Solution.zip","Analysis/OUTAGE/Base case/Base case.h5")
# process("Model MaxHydro_OUTAGE Solution/Model MaxHydro_OUTAGE Solution.zip","Analysis/OUTAGE/Scenario 1/Scenario 1.h5")
# process("Model HighHydro_HighWind_OUTAGE Solution/Model HighHydro_HighWind_OUTAGE Solution.zip","Analysis/OUTAGE/Scenario 2/Scenario 2.h5")
# process("Model ModHydro_HighWindSolar_OUTAGE Solution/Model ModHydro_HighWindSolar_OUTAGE Solution.zip","Analysis/OUTAGE/Scenario 3/Scenario 3.h5")
# process("Model ModHydro_NewTech_OUTAGE Solution/Model ModHydro_NewTech_OUTAGE Solution.zip","Analysis/OUTAGE/Scenario 4/Scenario 4.h5")
# process("Model LowHydro_NewTech_OUTAGE Solution/Model LowHydro_NewTech_OUTAGE Solution.zip","Analysis/OUTAGE/Scenario 5/Scenario 5.h5")

# process("Model HighHydro_HighWind_2040 Solution/Model HighHydro_HighWind_2040 Solution.zip","Analysis/HighHydro_HighWind/HighHydro_HighWind.h5")

# for carbonprice in [125]
#     println(carbonprice)
#     for load in [50,75]
#         print(load)
#         process(string("Model DAC_MIT_",load,"load_",carbonprice,"carbon Solution/Model DAC_MIT_",load,"load_",carbonprice,"carbon Solution.zip"),
#                         string("Solutions/DAC MIT/",load,"load_",carbonprice,"carbon/",load,"load_",carbonprice,"carbon.h5"))
#     end
# end