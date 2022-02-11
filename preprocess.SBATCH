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
cd /code_ginkgo
algo = [kt, antikt,ptdesc,truth,ptasc,CA]

python set_topology.py  algo[${SLURM_ARRAY_TASK_ID}]

cd /recnn
num_jets = 10000
file_name ='ginkgo_'+str(algo[${SLURM_ARRAY_TASK_ID}])+'_'+str(num_jets)+'jets.pkl'
python make_batches.py --jet_algorithm=algo[${SLURM_ARRAY_TASK_ID}] --in_filename=filename
"

## to submit for all algorithms: sbatch --array 0-5 set_topology.sbatch

