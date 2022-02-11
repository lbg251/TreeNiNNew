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

cd code_ginkgo/
python set_topology.py antikt

cd recnn/

python make_batches.py --jet_algorithm=antikt --in_filename=ginkgo_antikt_10000jets.pkl
"


