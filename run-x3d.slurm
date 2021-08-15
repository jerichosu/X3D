#!/bin/bash
#SBATCH --job-name="x3d_tuning"
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --output=%x-%j.log

SCRATCHDIR=/local/$SLURM_JOB_NAME-$SLURM_JOBID
AOI="Jiayi"
STEPNUM=1

mkdir -p $SCRATCHDIR

export TMPDIR=$SCRATCHDIR
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


module purge #start with a clean environment
module load openmpi-geib-cuda10.2-gcc/4.0.5 

source env/bin/activate

mpirun python train_x3d.py
#mpirun python dataset_v3.py





rm -fR $SCTRATCHDIR
