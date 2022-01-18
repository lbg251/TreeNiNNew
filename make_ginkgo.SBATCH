#!/bin/bash -e

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem=100GB
#SBATCH --job-name=make_ginkgo

module purge


singularity exec $nv \
            --overlay $SCRATCH/environments/overlay_pytorch.ext3:ro \
            /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif \
            /bin/bash -c "
source /ext3/env.sh
python makeginkgo.py  –-Nsamples=100 -–jobnum=${SLURM_ARRAY_TASK_ID}
"
## to submit(for 3 jobs): sbatch --array 0-2 make_ginkgo.sbatch