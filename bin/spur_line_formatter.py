import pandas as pd
import h5py
import sys
import os

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

def apply_ratios(cost_df, ratio_df, annualized = False):
    if annualized:
        sub_dfs = []
        index_list = list(ratio_df.index.get_level_values(level=2))
        #Preserve order
        index = [i for n, i in enumerate(index_list) if i not in index_list[:n]]
        for id in index:
            sub_df = ratio_df[ratio_df.index.isin([id],level=2)].replace(to_replace=0, method="ffill")
            sub_dfs.append(sub_df)
        ratio_df = pd.concat(sub_dfs)
    otc = pd.DataFrame(cost_df["values"] * ratio_df["OTC Ratio"])
    otc.columns = ["values"]
    build = pd.DataFrame(cost_df["values"] * ratio_df["Build Ratio"])
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
        except:
            print("generator_Build_Cost not found in formatted solutions file: ", scenario_formatted_path)

        #open capacity build data
        try:
            cap_built = pd.read_hdf(scenario_formatted_path,"generator_Capacity_Built");
        except:
            print("generator_Capacity_Built not found in formatted solutions file: ",scenario_formatted_path)

        #Read spur line cost data and append ITC and PTC to generator names as per PLEXOS solutions
        print("MAKE SURE TO UPDATE WIND SPUR LINE COST FILE")
        print("\n\n\n")
        spur_cost = pd.read_csv("C:\\Users\\lstreitm\\Documents\\Railbelt_local\\cheapest_3GW_wind.csv")
        spur_cost = spur_cost[["gen_name_full","spur_line_cost"]];
        spur_cost.rename({"gen_name_full":"gen_name"},axis=1, inplace=True)
        spur_cost_ptc = spur_cost.copy()
        spur_cost_ptc["gen_name"] = spur_cost_ptc["gen_name"].astype(str) + "_PTC"
        spur_cost_itc = spur_cost.copy()
        spur_cost_itc["gen_name"] = spur_cost_itc["gen_name"].astype(str) + "_ITC"
        spur_cost = pd.concat([spur_cost_ptc,spur_cost_itc])
        spur_cost.set_index("gen_name", inplace = True);

        #Combine spur cost $/kW and built capacity to calculate total spur cost
        otc = cap_built.join(spur_cost, on="gen_name").fillna(0)
        otc["One_Time_Cost"] = otc["values"] * otc["spur_line_cost"] * 1000
        otc.index = otc.index.set_levels(otc.index.levels[6].str.replace("MW","$"), level=6)
        otc = otc[["One_Time_Cost"]]

        

        #Calculate ratio of build cost to OTC to apply to annualized and NPV costs
        costs = pd.concat([cap_cost,otc], axis=1)
        costs.rename({"values":"Total Cap Cost"},axis=1, inplace = True)
        costs["Adjusted Build_Cost"] = costs["Total Cap Cost"] - costs["One_Time_Cost"]
        costs["Build Ratio"] = costs["Adjusted Build_Cost"] / costs["Total Cap Cost"]
        costs["OTC Ratio"] = costs["One_Time_Cost"] / costs["Total Cap Cost"]
        costs.fillna(0,inplace=True)
        ratio_df = costs[["Build Ratio","OTC Ratio"]]
        
        # Save edited costs
        cap_cost_new = costs[["Adjusted Build_Cost"]].copy()
        cap_cost_new.rename({"Adjusted Build_Cost":"values"},axis=1, inplace = True)
        otc.rename({"One_Time_Cost":"values"},axis=1, inplace = True)

        #EXPORT BACK TO H5 FILE
        dataio.save_to_h5(cap_cost_new, scenario_formatted_path, key="generator_Build_Cost",)
        dataio.save_to_h5(otc, scenario_formatted_path, key="generator_One_Time_Cost",)

        # open and edit annualized build costs solution data using cost ratios
        try:
            cap_ann_cost = pd.read_hdf(scenario_formatted_path,"generator_Annualized_Build_Cost");
            otc_ann, build_ann = apply_ratios(cap_ann_cost, ratio_df, annualized = True)

            # Export back to H5
            dataio.save_to_h5(otc_ann, scenario_formatted_path, key = "generator_Annualized_One_Time_Cost",)
            dataio.save_to_h5(build_ann, scenario_formatted_path, key = "generator_Annualized_Build_Cost",)

        except:
            print("Annualized Build Costs not found in formatted solutions file: ",scenario_formatted_path)

        #open capital NPV costs solution data
        try:
            cap_npv_cost = pd.read_hdf(scenario_formatted_path,"generator_Build_Cost_NPV")
            otc_npv, build_npv = apply_ratios(cap_npv_cost, ratio_df, annualized = False)

            cap_npv_ann_cost = pd.read_hdf(scenario_formatted_path, "generator_Annualized_Build_Cost_NPV")
            otc_ann_npv, build_ann_npv = apply_ratios(cap_npv_ann_cost, ratio_df, annualized = True)

            # Export back to h5
            dataio.save_to_h5(otc_npv, scenario_formatted_path, key = "generator_One_Time_Cost_NPV",)
            dataio.save_to_h5(otc_ann_npv, scenario_formatted_path, key = "generator_Annualized_One_Time_Cost_NPV",)
            dataio.save_to_h5(build_npv, scenario_formatted_path, key = "generator_Build_Cost_NPV",)
            dataio.save_to_h5(build_ann_npv, scenario_formatted_path, key = "generator_Annualized_Build_Cost_NPV",)

        except:
            print("NPV Build Costs not found in formatted solutions file: ",scenario_formatted_path)


        # Now make additional edits to separate fossil build costs from renewable build costs
        build = pd.read_hdf(scenario_formatted_path,"generator_Annualized_Build_Cost");
        build_NPV = pd.read_hdf(scenario_formatted_path,"generator_Annualized_Build_Cost_NPV");

        re_build = build[build.index.isin(gen_cats.re,level=1)]
        re_build_NPV = build_NPV[build_NPV.index.isin(gen_cats.re,level=1)]

        fossil_build = build[~build.index.isin(gen_cats.re,level=1)]
        fossil_build_NPV = build_NPV[~build_NPV.index.isin(gen_cats.re,level=1)]

        # Export back to h5
        dataio.save_to_h5(re_build, scenario_formatted_path, key = "generator_RE_Annualized_Build_Cost",)
        dataio.save_to_h5(re_build_NPV, scenario_formatted_path, key = "generator_RE_Annualized_Build_Cost_NPV",)
        dataio.save_to_h5(fossil_build, scenario_formatted_path, key = "generator_Fossil_Annualized_Build_Cost",)
        dataio.save_to_h5(fossil_build_NPV, scenario_formatted_path, key = "generator_Fossil_Annualized_Build_Cost_NPV",)



if __name__ == "__main__":
    main()