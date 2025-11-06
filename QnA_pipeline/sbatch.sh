#!/bin/bash
#SBATCH -J "ANLP_Project"
#SBATCH -c 40
#SBATCH --mem-per-cpu=1024
#SBATCH -G 4
#SBATCH -w gnode033
#SBATCH -o ./logs/train_%j.log
#SBATCH -e ./logs/errors_%j.err
#SBATCH --time="4-00:00:00"
#SBATCH --mail-type=ALL

source ~/.bashrc
mkdir -p chunked_data
chmod 700 chunked_data

# creates a symlink of .cache and the /scratch of the gnode so that I get space
~/symlink.sh

eval "$(mamba shell hook --shell bash)"
mamba activate

mamba activate next-level-lm

chmod +x run.sh
./run.sh