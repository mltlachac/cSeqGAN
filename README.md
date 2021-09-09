# Conditional Sequence Generative Adversarial Network (cSeqGAN)

In each branch is a different member of the cSeqGAN family. 

* G1 = Sentence Weighting
* G2 = Dual Unit Weighting
* G3 = Single Unit Weighting
* D1 = Single Task Feedback
* D2 = Dual Task Feedback
* D3 = Dual Unit Feedback

To run an SeqGAN model, navigate to the desired branch. 
An example for how to run the cSeqGAN model with sentece weighting and single task feedback:

```
#run experiments 5 times for the mentioned dataset
model=cseqgan_g1d1
dataset=data/movie_review.csv
for i in {1..5}
do
  save_dir=save_g1d1_review_${i}
  if [ -d $save_dir ]
  then
    rm -r $save_dir
  fi
  mkdir $save_dir
  python3 main.py -g $model -t real -d $dataset -p $save_dir
done
```
