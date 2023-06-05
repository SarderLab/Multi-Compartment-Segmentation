#!/bin/sh
#SBATCH --account=pinaki.sarder
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8000mb
#SBATCH --partition=gpu
#SBATCH --gpus=a100
#SBATCH --time=96:00:00
#SBATCH --output=hail.out
#SBATCH --job-name="PAN-DL2"
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST="$SLURM_JOB_NODELIST
echo "SLURM_NNODES="$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR

echo "working directory = "$SLURM_SUBMIT_DIR
ulimit -s unlimited
module list
which python

echo "Launch job"
CUDA_LAUNCH_BLOCKING=1
#python3 test.py
python3 segmentation_school.py --option predict --project LNR01/ --base_dir /blue/pinaki.sarder/nlucarelli/Detectron/ --modelfile /blue/pinaki.sarder/nlucarelli/Detectron/model_0214999.pth --boxSize 2048


echo "All Done!"
