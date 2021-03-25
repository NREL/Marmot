#!/bin/bash

############################ USER MODIFIED SECTION ########################
export feature="0-02:00:00"
export alloc="yourallocation"    #allcoation to use
export partition="short" #only for debugging on eagle, otherwise not needed
export runscript="run_python_marmot_plot_multiple.sh"
export description="1hr_plots_" #job name and scenario name
export myemail="\ana.dyreson@nrel.gov"
##########################################################################
models=$1 #List of items requested in a call

rootdir=$(pwd)
cd $rootdir

while read line; do #Go through list of plots requested and submit a job for each
	echo $line
 pwd
 SLURM_SUBMIT_DIR=$(pwd)
 submitcommand="sbatch --account=${alloc} --job-name="${description}${line}" --time=${feature} --mail-user=${myemail} --mail-type=ALL --partition="${partition}" --export=item="${line}" ${runscript}"
 echo $submitcommand
 $submitcommand
 cd $rootdir
done<$models

