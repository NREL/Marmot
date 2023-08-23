tech_map_simple = {
    "BA":"Battery",
    "CC_NATURAL_GAS":"Gas-CC",
    "CT_COAL":"Coal",
    "CT_DISTILLATE_FUEL_OIL":"Oil",
    "CT_NATURAL_GAS": "Gas-CT",
    "CT_OTHER": "Other",
    "CT_RESIDUAL_FUEL_OIL": "Oil",
    "CT_WASTE_OIL": "Oil",
    "HY":"Hydro",
    "IC_DISTILLATE_FUEL_OIL":"Oil",
    "IC_NATURAL_GAS":"Gas",
    "OT":"Other",
    "OT_COAL":"Coal",
    "OT_MUNICIPAL_WASTE":"Landfill-Gas",
    "OT_NATURAL_GAS":"Gas",
    "OT_NUCLEAR":"Nuclear",
    "OT_OTHER":"Other",
    "OT_WASTE_COAL":"Coal",
    "PVe":"PV",
    "ST_COAL":"Coal",
    "ST_DISTILLATE_FUEL_OIL":"Oil",
    "ST_NATURAL_GAS":"Gas",
    "ST_NUCLEAR":"Nuclear",
    "ST_OTHER":"Other",
    "ST_RESIDUAL_FUEL_OIL":"Oil",
    "ST_WASTE_COAL":"Coal",
    "ST_WASTE_OIL":"Oil",
    "WS":"Offshore-Wind",
    "WT":"Wind"
}



egret_map_simple = {
    'Oil':'Oil',
    'NG':'Gas',
    'Coal':'Coal',
    'Solar':'PV',
    'Wind':'Wind',
    'Sync_Cond':'Other',
    'Nuclear':'Nuclear',
    'Hydro':'Hydro'
}


plexos_map_simple = {

    'ReEDS_battery': 'Battery',
    'ReEDS_biopower': 'Biopower',
    'ReEDS_coal-igcc_coal-ccs_mod': 'Coal',
    'ReEDS_coal-new_coal-ccs_mod':'Coal',
    'ReEDS_coaloldscr_coal-ccs_mod':'Coal',
    'ReEDS_distpv':'PV',
    'ReEDS_dupv':'dPV',
    'ReEDS_gas-cc_gas-cc-ccs_mod':'Gas-CC',
    'ReEDS_hyded':'Hydro',
    'ReEDS_hydud':'Hydro',
    'ReEDS_hydend':'Hydro',
    'ReEDS_lfill-gas':'Landfill-Gas',
    'ReEDS_gas-cc_re-cc':'Gas-CC',
    'ReEDS_gas-ct_re-ct':'Gas-CT',
    'ReEDS_re-cc':'RE-CC',
    'ReEDS_upv':'PV',
    'ReEDS_wind-ons':'Wind',
    'nuclear':'Nuclear',
    'ReEDS_can-imports':'Imports',
    'ReEDS_hydund':'Hydro',
    'ReEDS_pumped-hydro':'Storage',
    'ReEDS_re-ct':'RE-CT',
    'ReEDS_wind-ofs':'Offshore-Wind',
    'ReEDS_beccs_mod':'BECCS',
    'ReEDS_geothermal':'Geothermal',
    'ReEDS_gas-cc-ccs_mod':'Gas',
    'caes':'Storage',
    'biopower':'Biopower',
    'coalolduns':'Coal',
    'ReEDS_nuclear':'Nuclear',
    'gas-cc':'Gas-CC',
    'gas-ct':'Gas-CT',
    'ReEDS_gas-cc':'Gas-CC',
    'ReEDS_gas-ct':'Gas-CT',
    'coaloldscr':'Coal',
    'o-g-s':'Oil-Gas-Steam',
    'ReEDS_o-g-s':'Oil-Gas-Steam',
    'ReEDS_coal-new':'Coal',
    'ReEDS_coaloldscr':'Coal',
    'coal-new':'Coal',
    'ReEDS_csp-ns':'Other',
    'coal-igcc':'Coal',
    'ReEDS_coalolduns_coal-ccs_mod':'Coal',
    'ReEDS_dupv':'dPV',
    'battery':'Battery',
    'biopower': 'Biopower',
    'coal-igcc_coal-ccs_mod': 'Coal',
    'coal-new_coal-ccs_mod':'Coal',
    'coaloldscr_coal-ccs_mod':'Coal',
    'distpv':'PV',
    'dupv':'dPV',
    'gas-cc_gas-cc-ccs_mod':'Gas-CC',
    'hyded':'Hydro',
    'hydud':'Hydro',
    'hydend':'Hydro',
    'lfill-gas':'Landfill-Gas',
    'gas-cc_re-cc':'Gas-CC',
    'gas-ct_re-ct':'Gas-CT',
    're-cc':'RE-CC',
    'upv':'PV',
    'wind-ons':'Wind',
    'can-imports':'Imports',
    'hydund':'Hydro',
    'pumped-hydro':'Storage',
    're-ct':'RE-CT',
    'wind-ofs':'Offshore-Wind',
    'beccs_mod':'BECCS',
    'beccs':'BECCS',
    'geothermal':'Geothermal',
    'gas-cc-ccs_mod':'Gas',
    'caes':'Storage',
    'biopower':'Biopower',
    'coalolduns':'Coal',
    'coal-new':'Coal',
    'coaloldscr':'Coal',
    'csp-ns':'Other',
    'coalolduns_coal-ccs_mod':'Coal',
    'dupv':'dPV',
    'h2-ct':'H2-CT',
    'hydnpnd': 'Hydro',
    'gas-cc-ccs':'Gas-CC',
    'geohydro':'Hydro'
}


volt_line_width={
    220.0:0.75,
    230.0:0.75,
    315.0:1,
    345.0:1,
    500.0:1.25,
    765.0:1.5,
    735.0:1.5
}


def columns_to_simple(input_columns, input_map):
    from difflib import get_close_matches

    rmap = [] #list of tuples
    for column in input_columns:
        results = get_close_matches(column.lower(), list(input_map.keys()))
        if results:
            rmap.append((plexos_map_simple[results[0]], column))
        # return same value if not mapped
        else:
            rmap.append((column, column))

    return rmap