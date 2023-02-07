#!/bin/bash

#SBATCH --time=12:00:00   # walltime limit (HH:MM:SS)
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=16   # 16 processor core(s) per node
#SBATCH --gres=gpu:1
#SBATCH --mem=20G   # maximum memory per node
#SBATCH -C V100|T4

#date;hostname;pwd
module load gcc
module load slurm
module load pigz/gcc-4.8.5/2.4
module load cuda10.1/blas/10.1.105
module load cuda10.1/toolkit/10.1.105
module load cudnn/cuda92/7.3.1

python3 BERT.py cseqdata/g1d1review1.txt data/movie_review.txt cseqdata/g1d1r1_seed1_real1 1 1