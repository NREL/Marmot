"""Main formatting module for ReEDS India results,
Contains classes and methods specific to ReEDS India outputs.
Inherits the ProcessReEDS class.

@author: Daniel Levie
"""

import logging
from typing import List
from dataclasses import dataclass, field

from marmot.formatters.formatbase import ReEDSPropertyColumnsBase
from marmot.formatters.formatreeds import ProcessReEDS
from marmot.formatters.formatextra import ExtraReEDSIndiaProperties

logger = logging.getLogger("formatter." + __name__)


class ProcessReEDSIndia(ProcessReEDS):
    """Process ReEDS India specific data from a ReEDS India result set."""

    # Maps ReEDS property names to Marmot names
    PROPERTY_MAPPING: dict = {
        "generator_gen_out": "generator_Generation",
        "generator_resgen": "generator_Generation_Annual",
        "generator_rescap": "generator_Installed_Capacity",
        "generator_curt_out": "generator_Curtailment",
        "region_load_mw": "region_Demand",
        "region_load_rt": "region_Demand_Annual",
        "line_FLOW": "line_Flow",
        "line_CAPTRAN": "line_Import_Limit",
        "generator_stor_charge": "generator_Pump_Load",
        "reserves_generators_OPRES": "reserves_generators_Provision",
    }
    """Maps simulation model property names to Marmot property names"""

    GDX_RESULTS_PREFIX = "output_"
    """Prefix of gdx results file"""

    EXTRA_PROPERTIES_CLASS = ExtraReEDSIndiaProperties

    @property
    def reeds_prop_cols(self) -> "ReEDSIndiaPropertyColumns":
        """Get the ReEDSIndiaPropertyColumns dataclass

        Returns:
            ReEDSIndiaPropertyColumns
        """
        return ReEDSIndiaPropertyColumns()


@dataclass
class ReEDSIndiaPropertyColumns(ReEDSPropertyColumnsBase):
    """ReEDS India property column names"""

    gen_out: List = field(
        default_factory=lambda: ["tech", "region", "h", "year", "Value"]
    )
    """ReEDS India 'gen_out' property columns (Marmot generator_Generation property)"""
    resgen: List = field(default_factory=lambda: ["tech", "region", "year", "Value"])
    """ReEDS India 'resgen' property columns (Marmot generator_Generation_Annual property)"""
    rescap: List = field(default_factory=lambda: ["tech", "region", "year", "Value"])
    """ReEDS India 'rescap' property columns (Marmot generator_Installed_Capacity property)"""
    curt_out: List = field(
        default_factory=lambda: ["tech", "region", "h", "year", "Value"]
    )
    """ReEDS India 'curt_out' property columns (Marmot generator_Curtailment property)"""
    load_mw: List = field(default_factory=lambda: ["region", "h", "year", "Value"])
    """ReEDS India 'load_mw' property columns (Marmot region_Demand property)"""
    load_rt: List = field(default_factory=lambda: ["region", "year", "Value"])
    """ReEDS India 'load_rt' property columns (Marmot region_Demand_Annual property)"""
    FLOW: List = field(
        default_factory=lambda: [
            "region_from",
            "region_to",
            "h",
            "year",
            "category",
            "Value",
            "marginal",
            "lower",
            "upper",
            "scale",
        ]
    )
    """ReEDS India 'FLOW' property columns (Marmot line_Flow property)"""
    CAPTRAN: List = field(
        default_factory=lambda: [
            "region_from",
            "region_to",
            "category",
            "year",
            "Value",
            "marginal",
            "lower",
            "upper",
            "scale",
        ]
    )
    """ReEDS India 'CAPTRAN' property columns (Marmot line_Import_Limit property)"""
    stor_charge: List = field(
        default_factory=lambda: ["tech", "region", "h", "year", "Value"]
    )
    """ReEDS India 'stor_charge' property columns (Marmot generator_Pumped_Load property)"""
    OPRES: List = field(
        default_factory=lambda: [
            "parent",
            "tech",
            "sub-tech",
            "region",
            "h",
            "year",
            "Value",
            "marginal",
            "lower",
            "upper",
            "scale",
        ]
    )
    """ReEDS India 'OPRES' property columns (Marmot reserves_generators_Provision property)"""
