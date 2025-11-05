#!/bin/bash
#SBATCH -J "ANLP_Project"
#SBATCH -c 40
#SBATCH --mem-per-cpu=1024
#SBATCH -G 4
#SBATCH -w gnode026
#SBATCH -o ./logs/train_%j.log
#SBATCH -e ./logs/errors.err
#SBATCH --time="4-00:00:00"
#SBATCH --mail-type=ALL

source ~/.bashrc

# creates a symlink of .cache and the /scratch of the gnode so that I get space
~/symlink.sh

eval "$(mamba shell hook --shell bash)"
mamba activate

mamba activate next-level-lm

./run.sh