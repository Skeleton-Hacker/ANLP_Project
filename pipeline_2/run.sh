#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH -A research
#SBATCH --cpus-per-gpu=9
#SBATCH --mem=96G
#SBATCH --time=4-00:00:00
#SBATCH --output=job_output.txt
#SBATCH --mail-user=george.rahul@research.iiit.ac.in
#SBATCH --mail-type=ALL
#SBATCH --nodelist=gnode006
echo "=========================================="
echo "SLURM_JOB_ID = $SLURM_JOB_ID"
echo "SLURM_NODELIST = $SLURM_NODELIST"
echo "SLURM_JOB_GPUS = $SLURM_JOB_GPUS"
echo "=========================================="

# ----------------------------------------------------------
# Correct micromamba setup for NON-INTERACTIVE SLURM BASH
# ----------------------------------------------------------
export MAMBA_ROOT_PREFIX=$HOME/mamba
export PATH="$MAMBA_ROOT_PREFIX/bin:$PATH"

# The IMPORTANT line: initialize bash micromamba hook
eval "$(micromamba shell hook --shell bash)"

# now activate will work
micromamba activate next-level-lm

echo "Micromamba environment activated!"
python --version

# ----------------------------------------------------------
# Create scratch directory (safe)
# ----------------------------------------------------------
mkdir -p -m 700 /ssd_scratch/gr
mkdir -p /ssd_scratch/gr/cache

# ----------------------------------------------------------
# Navigate to your project
# ----------------------------------------------------------
cd /home2/gr/ANLP_Project/pipeline_2
# ----------------------------------------------------------
# RUN YOUR CODE
# ----------------------------------------------------------
#accelerate launch --num_processes 4 semantic_chunking.py
#accelerate launch --num_processes 4 chunk_embeddings.py
#accelerate launch t5_full_finetune.py
#accelerate launch t5_full_finetune.py --compare-base
python compare_all_models.py --gemini-api-key AIzaSyBWxD8iPXrgsEEzgdX5O_dp0l5GUm5FEWQ
