# Conditional Sequence Generative Adversarial Networks (cSeqGAN)

Includes code to run all nine conditioned seqGAN (CSeqGAN) models, unconditioned SeqGAN models for positive/negative or depressed/notDepressed, unconditioned SeqGAN variants, and BERT classifier for evaluation. 

We borrowed and adapted generative code from [Texygen benchmarking platform](https://github.com/geek-ai/Texygen).


## Models:

g1 = sentence weighting strategy; 
g2 = dual unit weighting strategy; 
g3 = single unit weighting strategy

d1 = single task feedback mechanism; 
d2 = dual task feedback mechanism; 
d3 = dual critic feedback mechanism; 

* We used nine cSeqGAN models and two SeqGAN in this research: 
	* seqgan_pos (model name for generating unconditioned sequences for positive or not-depressed data)
		* runSeqGAN_pos.sh bash script to submit
	* seqgan_neg (model name for generating unconditioned sequences for negative or depressed data)
		* runSeqGAN_neg.sh bash script to submit
	* cseqgan_g1d1 (model name for generating conditioned combined sequences)
		* combination of generator-1, discriminator-1
		* runCSeqGAN_g1d1.sh bash script to submit
	* cseqgan_g1d2 (model name for generating conditioned combined sequences)
		* combination of generator-1, discriminator-2
		* runCSeqGAN_g1d2.sh bash script to submit
	* cseqgan_g1d3 (model name for generating conditioned combined sequences)
		* combination of generator-1, discriminator-3
		* runCSeqGAN_g1d3.sh bash script to submit
	* cseqgan_g2d1 (model name for generating conditioned combined sequences)
		* combination of generator-2, discriminator-1
		* runCSeqGAN_g2d1.sh bash script to submit
	* cseqgan_g2d2 (model name for generating conditioned combined sequences)
		* combination of generator-2, discriminator-2
		* runCSeqGAN_g2d2.sh bash script to submit
	* cseqgan_g2d3 (model name for generating conditioned combined sequences)
		* combination of generator-2, discriminator-3
		* runCSeqGAN_g2d3.sh bash script to submit
	* cseqgan_g3d1 (model name for generating conditioned combined sequences)
		* combination of generator-3, discriminator-1
		* runCSeqGAN_g3d1.sh bash script to submit
	* cseqgan_g3d2 (model name for generating conditioned combined sequences)
		* combination of generator-3, discriminator-2
		* runCSeqGAN_g3d2.sh bash script to submit
	* cseqgan_g3d3 (model name for generating conditioned combined sequences)
		* combination of generator-3, discriminator-3
		* runCSeqGAN_g3d3.sh bash script to submit

Bash file scripts to run the models:
python3 main.py -g <model_name> -t real -d  <path of the data to be used for training> -p <save directory path>


## Data

Due to Privacy concerns, we are only able to share the IMDb Movie Reviews (https://paperswithcode.com/dataset/imdb-movie-reviews ). 

### Real Data
* SSTB Movie Review Dataset:
    * data/movie_review.csv
        * combined movie review dataset relating to both positive and negative sentiment
        * header-less csv with two columns: review, sentiment (0=negative, 1=positive)
    * data/movie_review_pos.csv
        * positive movie reviews
        * header-less csv with one column: review
    * data/movie_review_neg.csv
        * negative movie reviews
        * header-less csv with one column: review
	
* Private Text Messages (Sent within last 2 months):
    * data/textMsgs.csv
        * the original data is filtered for messages with: <br/>
            * PHQ-9 <= 5: Not Depressed
            * PHQ-9 >=15: Depressed
        * combined depressed and not-depressed text messages
        * header-less csv with two columns: message, depressed (0=not depressed, 1=depressed)
    * data/textMsgs_Depressed.csv
        * Depressed text messages
        * header-less csv with one column: message
    * data/textMsgs_NotDepressed.csv
        * Not Depressed text messages
        * header-less csv with one column: message

* Interview Data (DAIC_WOZ processed text data):
    * data/interview_combined.csv
        * combined depressed and not-depressed interview data
        * header-less csv with two columns: message, depressed (0=not depressed, 1=depressed)
    * data/interview_Depressed.csv
        * Depressed interview data
        * header-less csv with one column: message
    * data/interview_NotDepressed.csv
        * Not Depressed interview data
        * header-less csv with one column: message

### Generated Sequences
The generated sequences for the Movie Review dataset are available under the folder generated, sorted into folders by the model that generated them. As we ran each model five times, there are five different files of generated sequences in each folder.


## Evaluation:

### Epoch-Wise Evaluation Metrics
The evaluation folder contains a folder for each cSeQGAN architecture that contains CSV files with the epoch-wise evaluation NLL, BLEU-2, Self-BLEU-2, BLEU-3, Self-BLEU-3, BLEU-4, and Self-BLEU-4 for each run. These epoch-wise evaluation are also available for the positive and negative SeqGAN models. BLEU.py was used to calculate the BLEU scores for the combined positive and negative SeqGAN generated text sequences. 

### BERT Classifier
BERT.py is the code we used to evaluate the predictive quality of the data. We include an exmaple of our genTrainTest script that we used to identify the train and test sets for the BERT models. We also include an example bash file script to run the BERT classifier, which includes the requested resources. 

