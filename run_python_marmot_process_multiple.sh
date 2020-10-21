#! /bin/bash

echo 'Run python script'


## Run python script
module purge
module load conda
python_command="python PLEXOS_H5_results_formatter.py "${item}""
echo $python_command
$python_command
