#! /bin/bash

echo 'Run python script'


## Run python script
module purge
module load conda
. activate /home/adyreson/.conda-envs/h5plexos_ana
python_command="python Marmot_plot_main.py "${item}""
echo $python_command
$python_command
