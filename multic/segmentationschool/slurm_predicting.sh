#!/bin/sh
#SBATCH --account=pinaki.sarder
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=16gb
#SBATCH --partition=gpu
#SBATCH --gpus=a100:2
#SBATCH --time=72:00:00
#SBATCH --output=./slurm_log.out
#SBATCH --job-name="multic_training"
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
USER=f.afsari
# Add the name of the folder containing WSIs here
PROJECT=multic_segment

CODESDIR=/blue/pinaki.sarder/f.afsari/Codes/MultiC_test/multic/segmentationschool
SIFDIR=/orange/pinaki.sarder/shared_singularity_files

DATADIR=$CODESDIR/test_data
MODELDIR=$CODESDIR/pretrained_model

CONTAINER=$SIFDIR/multic_segment.sif
CUDA_LAUNCH_BLOCKING=1
singularity exec --nv -B $(pwd):/exec/,$DATADIR/:/data/,$MODELDIR/:/model/ $CONTAINER python3 /exec/segmentation_school.py --option predict --base_dir $CODESDIR --modelfile /model/model_final.pth --files /data --wsi_ext svs 
