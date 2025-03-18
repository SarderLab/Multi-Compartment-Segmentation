#!/bin/sh
#SBATCH --account=pinaki.sarder
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --time=72:00:00
#SBATCH --output=logs/slurm_log.out
#SBATCH --job-name="mcs_training"
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST="$SLURM_JOB_NODELIST
echo "SLURM_NNODES="$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR

echo "working directory = "$SLURM_SUBMIT_DIR
ulimit -s unlimited
module load singularity
ls
ml

# Add your userid here:
USER=anish.tatke
# Add the name of the folder containing WSIs here
PROJECT=mcs_training

CODESDIR=/blue/pinaki.sarder/anish.tatke/Multi-Compartment-Segmentation/multic/segmentationschool
ORANGEDIR=/orange/pinaki.sarder/anish.tatke/MCS

DATADIR=$ORANGEDIR/TRAINING_data
MODELDIR=$ORANGEDIR/pretrained_model

CONTAINER=/blue/pinaki.sarder/anish.tatke/sif_containers/mcs_training.sif
CUDA_LAUNCH_BLOCKING=1

singularity exec --nv -B $(pwd):/exec/,$DATADIR/:/data,$MODELDIR/:/model/ $CONTAINER python3 /exec/segmentation_school.py \
    --option train \
    --base_dir $CODESDIR \
    --init_modelfile $MODELDIR/model_final.pth \
    --training_data_dir $DATADIR \
    --train_steps 10000 \
    --eval_period 2500 \
    --num_workers 8
