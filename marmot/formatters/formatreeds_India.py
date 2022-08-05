"""Main formatting module for ReEDS India results,
Contains classes and methods specific to ReEDS India outputs.
Inherits the ProcessReEDS class.

@author: Daniel Levie
"""

import logging
from typing import List
from dataclasses import dataclass, field

import marmot.utils.mconfig as mconfig
from marmot.formatters.formatreeds import ProcessReEDS

logger = logging.getLogger("formatter." + __name__)
formatter_settings = mconfig.parser("formatter_settings")


class ProcessReEDSIndia(ProcessReEDS):
    """Process ReEDS specific data from a ReEDS result set."""

    # Maps ReEDS property names to Marmot names
    PROPERTY_MAPPING: dict = {
        "generator_GEN": "generator_Generation",
        "generator_resgen": "generator_Generation_Annual",
        "generator_CAP": "generator_Installed_Capacity",
        "generator_CURT": "generator_Curtailment",
        "region_load_rt": "region_Demand",
        "line_tran_FLOW": "line_Flow",
        "line_CAPTRAN": "line_Import_Limit",
        "generator_STORAGE_IN": "generator_Pump_Load",
        "reserves_generators_OPRES": "reserves_generators_Provision",
    }
    """Maps simulation model property names to Marmot property names"""

    gdx_results_prefix = "output_"
    """Prefix of gdx results file"""

@dataclass
class PropertyColumns:
    """ReEDS property column names"""

    GEN: List = field(
        default_factory=lambda: ["tech", "sub-tech", "region", "h", "year", "Value", "marginal", "lower", "upper", "scale"]
    )
    """ReEDS 'gen_out' property columns (Marmot generator_Generation property)"""
    resgen: List = field(
        default_factory=lambda: ["tech", "region", "year", "Value"]
    )
    """ReEDS 'resgen' property columns (Marmot generator_Generation_Annual property)"""
    CAP: List = field(default_factory=lambda: ["tech", "sub-tech", "region", "year", "Value", "marginal", "lower", "upper", "scale"])
    """ReEDS 'cap_out' property columns (Marmot generator_Installed_Capacity property)"""
    CURT: List = field(default_factory=lambda: ["region", "h", "year", "Value", "marginal", "lower", "upper", "scale"])
    """ReEDS 'curt_out' property columns (Marmot generator_Curtailment property)"""
    load_rt: List = field(default_factory=lambda: ["region", "year", "Value"])
    """ReEDS 'load_rt' property columns (Marmot region_Load_Annual property)"""
    FLOW: List = field(
        default_factory=lambda: [
            "region_from",
            "region_to",
            "h",
            "year",
            "category",
            "Value","marginal", "lower", "upper", "scale"
        ]
    )
    """ReEDS 'tran_flow_power' property columns (Marmot line_Flow property)"""
    CAPTRAN: List = field(
        default_factory=lambda: [
            "region_from",
            "region_to",
            "category",
            "year",
            "Value","marginal", "lower", "upper", "scale"
        ]
    )
    """ReEDS 'tran_out' property columns (Marmot line_Import_Limit property)"""
    STORAGE_IN: List = field(
        default_factory=lambda: ["tech", "sub-tech", "region", "h", "src", "year", "Value"]
    )
    """ReEDS 'STORAGE_IN' property columns (Marmot generator_Pumped_Load property)"""
    OPRES: List = field(
        default_factory=lambda: ["parent", "tech", "sub-tech", "region", "h", "year", "Value", "marginal", "lower", "upper", "scale"]
    )