import pandas as pd
import h5py
import sys
import os

PV_itc_multiplier = 65000 #$/MW in fuel storage costs
PV_ptc_multiplier = 39000 #$/MW in fuel storage costs
wind_itc_multiplier = 100000 #$/MW in fuel storage costs
wind_ptc_multiplier = 60000 #$/MW in fuel storage costs

spur_line_cost_factor = 9.821 #$2021/kW/km
substation_upgrade_factor = 2300 #$2021/MW of solar and wind

from pathlib import Path

run_path = Path(__file__).parent.resolve().parent
sys.path.insert(0, str(run_path))

try:
    import marmot.utils.mconfig as mconfig
except ModuleNotFoundError:
    from marmot.utils.definitions import INCORRECT_ENTRY_POINT
    print(INCORRECT_ENTRY_POINT.format(Path(__file__).name))
    sys.exit()

from marmot.utils.definitions import INPUT_DIR
import marmot.utils.dataio as dataio
import marmot.plottingmodules.plotutils.plot_data_helper as pdh

def apply_ratios(cost_df, ratio_df, annualized = False, ratio1 = "OTC Ratio", ratio2 = "Build Ratio"):
    if annualized:
        # populate ratio for all years not just build year
        sub_dfs = []
        index_list = list(ratio_df.index.get_level_values(level=2))
        #Preserve order
        index = [i for n, i in enumerate(index_list) if i not in index_list[:n]]
        for id in index:
            sub_df = ratio_df[ratio_df.index.isin([id],level=2)].replace(to_replace=0, method="ffill")
            sub_dfs.append(sub_df)
        ratio_df = pd.concat(sub_dfs)
    otc = pd.DataFrame(cost_df["values"] * ratio_df[ratio1])
    otc.columns = ["values"]
    build = pd.DataFrame(cost_df["values"] * ratio_df[ratio2])
    build.columns = ["values"]

    return (otc, build)


def main():
    Marmot_user_defined_inputs = pd.read_csv(
        INPUT_DIR.joinpath(mconfig.parser("user_defined_inputs_file")),
        usecols=["Input", "User_defined_value"],
        index_col="Input",
        skipinitialspace=True,
    )

    Scenario_List = Marmot_user_defined_inputs.loc[
        "Scenario_process_list", "User_defined_value"
    ]
    Scenario_List = [x.strip() for x in Scenario_List.split(",")]
    
    model_solutions_folder = Marmot_user_defined_inputs.loc[
        "Model_Solutions_folder", "User_defined_value"
    ].strip()

    gen_cats = Marmot_user_defined_inputs.loc[
        "ordered_gen_categories_file", "User_defined_value"
    ]

    gen_cats = pdh.GenCategories.set_categories(pd.read_csv(INPUT_DIR.joinpath("mapping_folder\\" + gen_cats)))

    # Folder to save your processed solutions
    if pd.isna(
        Marmot_user_defined_inputs.loc["Marmot_Solutions_folder", "User_defined_value"]
    ):
        marmot_solutions_folder = model_solutions_folder
    else:
        marmot_solutions_folder = Marmot_user_defined_inputs.loc[
            "Marmot_Solutions_folder", "User_defined_value"
        ].strip()

    for Scenario_name in Scenario_List:
        scenario_formatted_path = os.path.join(marmot_solutions_folder,"Processed_HDF5_folder",Scenario_name+"_formatted.h5") 

        #open capital cost solution data
        try:
            cap_cost = pd.read_hdf(scenario_formatted_path,"generator_Build_Cost");
            fom_cost = pd.read_hdf(scenario_formatted_path, "generator_FOM_Cost");
            fom_cost_npv = pd.read_hdf(scenario_formatted_path, "generator_FOM_Cost_NPV");
        except:
            print("generator_Build_Cost not found in formatted solutions file: ", scenario_formatted_path)

        #open capacity build and installed capacity data
        try:
            cap_built = pd.read_hdf(scenario_formatted_path,"generator_Capacity_Built");
            cap_installed = pd.read_hdf(scenario_formatted_path, "generator_Installed_Capacity");
        except:
            print("generator_Capacity_Built not found in formatted solutions file: ",scenario_formatted_path)

        #Read spur line cost data and append ITC and PTC to generator names as per PLEXOS solutions
        print("MAKE SURE TO UPDATE WIND SPUR LINE COST FILE")
        print("\n\n\n")
        spur_cost = pd.read_excel("Z:\\Projects\\Railbelt_nonCEII\\Inputs\\Costs\\cheapest_3GW_wind.xlsx", 
                                  sheet_name="cheapest_3GW_wind")
        spur_cost = spur_cost[["gen_name_full","total_spurline_km"]];
        spur_cost["spur_line_cost_kW"] = spur_cost["total_spurline_km"] * spur_line_cost_factor #$/kW
        spur_cost.rename({"gen_name_full":"gen_name"},axis=1, inplace=True)
        spur_cost_ptc = spur_cost.copy()
        spur_cost_ptc["gen_name"] = spur_cost_ptc["gen_name"].astype(str) + "_PTC"
        spur_cost_itc = spur_cost.copy()
        spur_cost_itc["gen_name"] = spur_cost_itc["gen_name"].astype(str) + "_ITC"
        spur_cost = pd.concat([spur_cost_ptc,spur_cost_itc])
        spur_cost.set_index("gen_name", inplace = True);

        #Combine spur cost $/kW and built capacity to calculate total spur cost
        otc = cap_built.join(spur_cost, on="gen_name").fillna(0)
        otc["One_Time_Cost"] = otc["values"] * otc["spur_line_cost_kW"] * 1000
        otc.index = otc.index.set_levels(otc.index.levels[6].str.replace("MW","$"), level=6)
        otc = otc[["One_Time_Cost"]]
        #Multiply built capacity by wind and pv multiplier to calculate fuel storage costs
        fuel_storage = cap_built.copy()
        wind = fuel_storage.loc[fuel_storage.index.get_level_values(level=1) == "Land-based wind"]
        wind_itc = wind.loc[wind.index.get_level_values(level=2).str.contains("_ITC")] * wind_itc_multiplier
        wind_ptc = wind.loc[wind.index.get_level_values(level=2).str.contains("_PTC")] * wind_ptc_multiplier
        pv = fuel_storage.loc[fuel_storage.index.get_level_values(level=1) == "PV"]
        pv_itc = pv.loc[pv.index.get_level_values(level=2).str.contains("_ITC")] * PV_itc_multiplier
        pv_ptc = pv.loc[pv.index.get_level_values(level=2).str.contains("_PTC")] * PV_ptc_multiplier
        fuel_storage = pd.concat([wind_itc, wind_ptc, pv_itc, pv_ptc], axis=0)
        fuel_storage = fuel_storage.rename(columns = {"values":"Fuel Storage Cost"})
        try:
            fuel_storage.index = fuel_storage.index.set_levels(fuel_storage.index.levels[6].str.replace("MW","$"), level=6)
        except: #e.g., for NoNewRE scenario where fuel_storage.index is length 0 so will error out
            fuel_storage = pd.concat([wind, pv], axis=0)
            fuel_storage = fuel_storage.rename(columns = {"values":"Fuel Storage Cost"})
            fuel_storage.index = fuel_storage.index.set_levels(fuel_storage.index.levels[6].str.replace("MW","$"), level=6)

        #Calculate ratio of build cost to OTC and fuel storage to apply to annualized and NPV costs
        costs = pd.concat([cap_cost,otc, fuel_storage], axis=1)
        costs.rename({"values":"Total Cap Cost"},axis=1, inplace = True)
        costs.fillna(0,inplace=True) # do this to catch any values that don't overlap between dfs
        costs["Adjusted Build_Cost"] = costs["Total Cap Cost"] - costs["One_Time_Cost"] - costs["Fuel Storage Cost"]
        costs["Build Ratio"] = costs["Adjusted Build_Cost"] / costs["Total Cap Cost"]
        costs["OTC Ratio"] = costs["One_Time_Cost"] / costs["Total Cap Cost"]
        costs["FSC Ratio"] = costs["Fuel Storage Cost"] / costs["Total Cap Cost"]
        costs.fillna(0,inplace=True) # do this again to handle division by zero
        ratio_df = costs[["Build Ratio","OTC Ratio","FSC Ratio"]]

        # Update FOM Cost to extract dPV Fuel Storage Cost
        dpv_fsc = fom_cost.loc[fom_cost.index.get_level_values(level=1) == "dPV"].rename(columns={"values":"DPV FSC"})
        fom_cost = pd.concat([fom_cost, dpv_fsc], axis=1).fillna(0)
        fom_cost["Adjusted FOM"] = fom_cost["values"] - fom_cost["DPV FSC"]
        dpv_cost = fom_cost[["DPV FSC"]].rename(columns = {"DPV FSC":"values"})
        fom_cost = fom_cost[["Adjusted FOM"]].rename(columns = {"Adjusted FOM":"values"})

        dpv_fsc_npv = fom_cost_npv.loc[fom_cost_npv.index.get_level_values(level=1) == "dPV"].rename(columns={"values":"DPV FSC"})
        fom_cost_npv = pd.concat([fom_cost_npv, dpv_fsc_npv], axis=1).fillna(0)
        fom_cost_npv["Adjusted FOM"] = fom_cost_npv["values"] - fom_cost_npv["DPV FSC"]
        dpv_cost_npv = fom_cost_npv[["DPV FSC"]].rename(columns = {"DPV FSC":"values"})
        fom_cost_npv = fom_cost_npv[["Adjusted FOM"]].rename(columns = {"Adjusted FOM":"values"})
        
        # Save edited costs
        cap_cost_new = costs[["Adjusted Build_Cost"]].copy()
        cap_cost_new.rename({"Adjusted Build_Cost":"values"},axis=1, inplace = True)
        otc.rename({"One_Time_Cost":"values"},axis=1, inplace = True)
        fuel_storage.rename(columns = {"Fuel Storage Cost":"values"}, inplace=True)

        # ADD EXTRA SPUR (SUBSTATION) COSTS
        existing_gens = list(set(cap_installed.loc[(cap_installed.index.get_level_values(level=0).year == 2024) & (cap_installed["values"] > 0)].index.get_level_values(level=2)))
        cap_installed.loc[cap_installed.index.get_level_values(level=2).isin(existing_gens)] *= 0
        cap_installed.loc[cap_installed.index.get_level_values(level=1).isin(["Land-based wind","PV"])] *= substation_upgrade_factor
        cap_installed.loc[~cap_installed.index.get_level_values(level=1).isin(["Land-based wind","PV"])] *= 0
        cap_installed.index = cap_installed.index.set_levels(cap_installed.index.levels[6].str.replace("MW","$"), level=6)
        dataio.save_to_h5(cap_installed, scenario_formatted_path, key="generator_Substation_Upgrade_Cost",)
        #EXPORT BACK TO H5 FILE
        dataio.save_to_h5(cap_cost_new, scenario_formatted_path, key="generator_Build_Cost",)
        dataio.save_to_h5(otc, scenario_formatted_path, key="generator_One_Time_Cost",)
        dataio.save_to_h5(fuel_storage, scenario_formatted_path, key="generator_Fuel_Storage_Cost",)
        dataio.save_to_h5(dpv_cost, scenario_formatted_path, key="generator_dPV_Fuel_Storage_Cost",)
        dataio.save_to_h5(dpv_cost_npv, scenario_formatted_path, key="generator_dPV_Fuel_Storage_Cost_NPV",)
        dataio.save_to_h5(fom_cost, scenario_formatted_path, key="generator_FOM_Cost",)
        dataio.save_to_h5(fom_cost_npv, scenario_formatted_path, key="generator_FOM_Cost_NPV",)

        # open and edit annualized build costs solution data using cost ratios
        try:
            cap_ann_cost = pd.read_hdf(scenario_formatted_path,"generator_Annualized_Build_Cost");
            otc_ann, build_ann = apply_ratios(cap_ann_cost, ratio_df, annualized = True, ratio1 = "OTC Ratio")
            fsc_ann, build_ann_fsc = apply_ratios(cap_ann_cost, ratio_df, annualized=True, ratio1="FSC Ratio")

            # Export back to H5
            dataio.save_to_h5(otc_ann, scenario_formatted_path, key = "generator_Annualized_One_Time_Cost",)
            dataio.save_to_h5(fsc_ann, scenario_formatted_path, key = "generator_Annualized_Fuel_Storage_Cost",)
            dataio.save_to_h5(build_ann, scenario_formatted_path, key = "generator_Annualized_Build_Cost",)

        except:
            print("Annualized Build Costs not found in formatted solutions file: ",scenario_formatted_path)

        #open capital NPV costs solution data
        try:
            cap_npv_cost = pd.read_hdf(scenario_formatted_path,"generator_Build_Cost_NPV")
            otc_npv, build_npv = apply_ratios(cap_npv_cost, ratio_df, annualized = False, ratio1 = "OTC Ratio")
            fsc_npv, build_npv_fsc = apply_ratios(cap_npv_cost, ratio_df, annualized = False, ratio1 = "FSC Ratio")

            cap_npv_ann_cost = pd.read_hdf(scenario_formatted_path, "generator_Annualized_Build_Cost_NPV")
            otc_ann_npv, build_ann_npv = apply_ratios(cap_npv_ann_cost, ratio_df, annualized = True, ratio1 = "OTC Ratio")
            fsc_ann_npv, build_ann_npv_fsc = apply_ratios(cap_npv_ann_cost, ratio_df, annualized = True, ratio1 = "FSC Ratio")

            # Export back to h5
            dataio.save_to_h5(otc_npv, scenario_formatted_path, key = "generator_One_Time_Cost_NPV",)
            dataio.save_to_h5(fsc_npv, scenario_formatted_path, key = "generator_Fuel_Storage_Cost_NPV",)
            dataio.save_to_h5(otc_ann_npv, scenario_formatted_path, key = "generator_Annualized_One_Time_Cost_NPV",)
            dataio.save_to_h5(fsc_ann_npv, scenario_formatted_path, key = "generator_Annualized_Fuel_Storage_Cost_NPV",)
            dataio.save_to_h5(build_npv, scenario_formatted_path, key = "generator_Build_Cost_NPV",)
            dataio.save_to_h5(build_ann_npv, scenario_formatted_path, key = "generator_Annualized_Build_Cost_NPV",)

        except:
            print("NPV Build Costs not found in formatted solutions file: ",scenario_formatted_path)


        # Now make additional edits to separate fossil build costs from renewable build costs
        # break every cost into renewable and non-renewable,
        # all renewable costs get lumped into renewable puchase cost (all non-re get broken out as before)
        costs_to_split = ["generator_Fuel_Cost","generator_FOM_Cost","generator_VOM_Cost",
                 "generator_Start_and_Shutdown_Cost", "generator_Annualized_Build_Cost",
                 "generator_Total_Generation_Cost", ]
                # "generator_Emissions_Cost" not included in model
                # "generator_Reserves_VOM_Cost" not to be split
                # "generator_UoS_Cost", "generator_Annualized_One_Time_Cost","batterie_Annualized_Build_Cost" do not need to be split
        
        re_costs = []
        re_npv_costs = []
        
        for cost in costs_to_split:
            print(cost)
            c1 = pd.read_hdf(scenario_formatted_path, cost);
            c2 = pd.read_hdf(scenario_formatted_path, cost + "_NPV");
            re_c1 = c1[c1.index.isin(gen_cats.re, level=1)].rename(columns={"values":cost})
            re_c2 = c2[c2.index.isin(gen_cats.re, level=1)].rename(columns={"values":cost+"_NPV"})
            fos_c1 = c1[~c1.index.isin(gen_cats.re, level=1)]
            fos_c2 = c2[~c2.index.isin(gen_cats.re, level=1)]
            if cost != "generator_Total_Generation_Cost":
                re_costs.append(re_c1)
                re_npv_costs.append(re_c2)
            # Export fossil costs back to h5
            dataio.save_to_h5(fos_c1, scenario_formatted_path, key = cost + "_Fossil",)
            dataio.save_to_h5(fos_c2, scenario_formatted_path, key = cost + "_Fossil_NPV",)

        re_costs = pd.concat(re_costs, axis=1).fillna(0).sum(axis=1).to_frame(name = "values")
        re_npv_costs = pd.concat(re_npv_costs, axis=1).fillna(0).sum(axis=1).to_frame(name = "values")

        # Export re costs back to h5
        dataio.save_to_h5(re_costs, scenario_formatted_path, key = "generator_Renewable_Purchases",)
        dataio.save_to_h5(re_npv_costs, scenario_formatted_path, key = "generator_Renewable_Purchases_NPV",)

        # Finally, separate running costs property out from Total Generation Cost (Fossil only)
        for suffix in ["", "_NPV"]:
            total_cost = pd.read_hdf(scenario_formatted_path, "generator_Total_Generation_Cost_Fossil"+suffix).rename(columns={"values":"Total"})
            fuel_cost = pd.read_hdf(scenario_formatted_path, "generator_Fuel_Cost_Fossil"+suffix).rename(columns={"values":"Fuel"})
            vom_cost = pd.read_hdf(scenario_formatted_path, "generator_VOM_Cost_Fossil"+suffix).rename(columns={"values":"VOM"})
            ss_cost = pd.read_hdf(scenario_formatted_path, "generator_Start_and_Shutdown_Cost_Fossil"+suffix).rename(columns={"values":"SS"})
            costs = pd.concat([total_cost, fuel_cost, vom_cost, ss_cost], axis=1)
            costs["Running"] = costs["Total"] - costs["Fuel"] - costs["VOM"] - costs["SS"]
            costs = costs[["Running"]].rename(columns={"Running": "values"})
            dataio.save_to_h5(costs, scenario_formatted_path, key = "generator_Running_Cost_Fossil"+suffix,)
            


if __name__ == "__main__":
    main()