#!/bin/bash
#SBATCH --account=def-arashmoh
#SBATCH --gpus-per-node=1
#SBATCH --mem=16000M               # memory per node
#SBATCH --time=0-12:00
#SBATCH --mail-user=<jamal73sm@gmail.com>
#SBATCH --mail-type=ALL

cd /home/shahab33/projects/def-arashmoh/shahab33
module purge
module load python
source ~/Mamba/bin/activate

python Vim/pytorch_visionmamba/pytorch_visionmamba.py 
