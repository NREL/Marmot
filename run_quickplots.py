from marmot.scenariohandlers import PlexosScenario, ReEDsScenario
import marmot.datahelpers as dh
import marmot.quickplots as qp
import pathlib

reeds_scenario = "v20230730_ntpH0_AC_DemMd_90by2035EP__core"
root_dir = pathlib.Path("/Volumes/ReEDS/FY21-EMRE-BeyondVRE/BVRE_runs_2023-09-06/PLEXOS_solutions")
reeds_path = pathlib.Path(
    "/Volumes/ReEDS/FY22-NTP/Candidates/Archive/ReEDSruns/20230730/v20230730_ntpH0_AC_DemMd_90by2035EP__core"
)

ps = PlexosScenario(root_dir)

plexos_df = ps.get_entity_tech_load_aggregates()


#ps.get_plexos_partition("interval", "generators")

qp.plot_dispatch_stack_bar(plexos_df)