#!/bin/bash -e

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem=100GB
#SBATCH --job-name=preprocesskT

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

cd /recnn
python search_hyperparams.py  --jet_algorithm=algo[${SLURM_ARRAY_TASK_ID}] --NrunStart=0 --NrunFinish=1 --gpu=0
"