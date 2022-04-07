"""
Execute and run the Marmot plotter 
"""

import sys
from pathlib import Path

run_path = Path(__file__).parent.resolve().parent
sys.path.insert(0, str(run_path))

from marmot.marmot_plot_main import main

main()
