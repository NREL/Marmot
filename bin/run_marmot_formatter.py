"""
Execute and run the Marmot formatter 
"""

import sys
from pathlib import Path

run_path = Path(__file__).parent.resolve().parent
sys.path.append(str(run_path))

from marmot.marmot_h5_formatter import main

main()
