#!/bin/bash

############################ USER MODIFIED SECTION ########################
export feature="0-04:00:00"
export alloc="continental"    #allcoation to use
export partition="short" #only for debugging on eagle, otherwise not needed, but can choose standard or short
export runscript="run_python_marmot_process_multiple.sh"
export priority="high" #choose high or normal
export myemail="\Marty.Schwrz@nrel.gov"
export description="OR_OSW_process_test" #job name and scenario name
##########################################################################
models=$24 #List of items requested in call

rootdir=$(pwd)
cd $rootdir


while read line; do #Go through list of items requested and submit a job for each.
	echo $line

pwd
 SLURM_SUBMIT_DIR=$(pwd)
 submitcommand="sbatch --account=${alloc} --job-name="${description}${line}" --time=${feature} --mail-user="${myemail}", --mail-type=ALL --partition="${partition}" --export=item="${line}" ${runscript}"
 echo $submitcommand
 $submitcommand
 cd $rootdir
done<$models
