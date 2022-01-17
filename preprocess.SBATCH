#!/bin/bash -e

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem=100GB
#SBATCH --job-name=set_topology

module purge

singularity exec $nv \
            --overlay $SCRATCH/environments/overlay_pytorch.ext3:ro \
            /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif \
            /bin/bash -c "
source /ext3/env.sh

algo = []
algo[0]=kt
algo[1]=antikt
algo[2]=ptasc
algo[3]=ptdesc
algo[4]=truth
algo[5]=CA
python set_topology.py  algo[${SLURM_ARRAY_TASK_ID}]

cd /recnn
num_jets = 10000
file_name ='ginkgo_'+algo[${SLURM_ARRAY_TASK_ID}]+_+string[num_jets]+'jets.pkl'
python makebatches.py –-jetalgorithm=algo[${SLURM_ARRAY_TASK_ID}] –-infilename=filename
"

## to submit for all algorithms: sbatch --array 0-5 set_topology.sbatch

