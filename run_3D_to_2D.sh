#!/bin/bash -l
#SBATCH -A uppmax2024-2-21
#SBATCH -M snowy
#SBATCH -p node --gres=gpu:1
#SBATCH -t 02:00:00

module load singularity 

singularity exec --nv open3d-nvidia.sif python 3D_to_2D_with_z_coordinate.py /proj/uppmax2024-2-21/data/etteplan/data_downsampled10_without_ears0_01 0.0005


