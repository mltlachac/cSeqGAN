#!/bin/bash
#SBATCH -n 8
#SBATCH -t 24:00:00
#SBATCH --mem=64G
#SBATCH -N 1

# run experiments 5 times for the mentioned dataset
model=seqgan_pos
dataset=data/movie_review_pos.csv
for i in {1..5}
do
  save_dir=save_pos_review_${i}
  if [ -d $save_dir ]
  then
    rm -r $save_dir
  fi
  mkdir $save_dir
  python3 main.py -g $model -t real -d $dataset -p $save_dir
done
