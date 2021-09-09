import json
from time import time

from models.Gan import Gan
from models.cseqgan.CSeqganDataLoader import DataLoader, DisDataloader
from models.cseqgan.CSeqganDiscriminator import Discriminator
from models.cseqgan.CSeqganGenerator import Generator
from models.cseqgan.CSeqganReward import Reward
from utils.metrics.Cfg import Cfg
from utils.metrics.EmbSim import EmbSim
from utils.metrics.Nll import Nll
from utils.oracle.OracleCfg import OracleCfg
from utils.oracle.OracleLstm import OracleLstm
from utils.text_process import *
from utils.utils import *
from utils.metrics.Bleu import Bleu
from utils.metrics.SelfBleu import SelfBleu


class CSeqgan(Gan):
    def __init__(self, oracle=None):
        super().__init__()
        # you can change parameters, generator here
        self.vocab_size = 20
        self.emb_dim = 32
        self.hidden_dim = 32
        self.sequence_length = 20
        self.filter_size = [2, 3]
        self.num_filters = [100, 200]
        self.l2_reg_lambda = 0.2
        self.dropout_keep_prob = 0.75
        self.batch_size = 64
        self.generate_num = 128
        self.start_token = 0
        self.n_classes = 2
        self.labels = []

        self.oracle_file = 'save/oracle_g2d1.txt' #ground truth file
        self.generator_file = 'save/generator_g2d1.txt'
        self.test_file = 'save/test_file_g2d1.txt'

    def train_discriminator(self):
        generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
        self.dis_data_loader.load_train_data(self.oracle_file, self.generator_file)
        for _ in range(3):
            self.dis_data_loader.next_batch()
            x_batch, y_batch= self.dis_data_loader.next_batch()
            feed = {
                self.discriminator.input_x: x_batch,
                self.discriminator.input_y: y_batch,
            }
            loss,_ = self.sess.run([self.discriminator.d_loss, self.discriminator.train_op], feed)
            print(loss)

    def evaluate(self):
        generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
        if self.oracle_data_loader is not None:
            self.oracle_data_loader.create_batches(self.generator_file)
        if self.log is not None:
            if self.epoch == 0 or self.epoch == 1:
                self.log.write('epochs, ')
                for metric in self.metrics:
                    self.log.write(metric.get_name() + ',')
                self.log.write('\n')
            scores = super().evaluate()
            for score in scores:
                self.log.write(str(score) + ',')
            self.log.write('\n')
            return scores
        return super().evaluate()

    def init_real_trainng(self, data_loc=None):
        from utils.text_process import csv_text_precess, text_precess, text_to_code
        from utils.text_process import get_tokenlized, get_word_list, get_dict, get_csv_tokenlized
        if data_loc is None:
            data_loc = 'data/image_coco.txt'
        self.sequence_length, self.vocab_size, labels = csv_text_precess(data_loc) #KratikaA: getting labels as well for all training inputs

        self.n_classes = len(set(labels))   #KratikaA: updating number of classes dynamically based on the input data
        generator = Generator(num_vocabulary=self.vocab_size, batch_size=self.batch_size, emb_dim=self.emb_dim,
                              hidden_dim=self.hidden_dim, sequence_length=self.sequence_length,
                              start_token=self.start_token, num_classes=self.n_classes) #KratikaA: passing total number of classes as a parameter for conditioning
        self.set_generator(generator)
        seqLength_with_label = self.sequence_length + 1
        discriminator = Discriminator(sequence_length=seqLength_with_label, num_classes=2, vocab_size=self.vocab_size,
                                      emd_dim=self.emb_dim, filter_sizes=self.filter_size, n_labels = self.n_classes, num_filters=self.num_filters,
                                      l2_reg_lambda=self.l2_reg_lambda) #KratikaA: passing n_labels = total number of classes for conditioning
        self.set_discriminator(discriminator)

        gen_dataloader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length)
        oracle_dataloader = None
        dis_dataloader = DisDataloader(batch_size=self.batch_size, seq_length=seqLength_with_label)

        self.set_data_loader(gen_loader=gen_dataloader, dis_loader=dis_dataloader, oracle_loader=oracle_dataloader)
        tokens, self.labels = get_csv_tokenlized(data_loc)
        self.unique_labels = list(set(self.labels))
        word_set, self.label_words_dict = get_word_list_conditional(tokens, self.labels)
        [word_index_dict, index_word_dict] = get_dict(word_set)

        with open(self.oracle_file, 'w') as outfile:
            outfile.write(text_to_code_with_labels(tokens, word_index_dict, self.sequence_length, self.labels)) #, label_index_dict)) #KratikaA: write label along with entire token
        return word_index_dict, index_word_dict

    def init_real_metric(self, data_loc=None):
        # from utils.metrics.DocEmbSim_Conditional import DocEmbSim_Conditional
        # docsim = DocEmbSim_Conditional(oracle_file=self.oracle_file, generator_file=self.generator_file, num_vocabulary=self.vocab_size, label_vocab=self.label_words_dict) #KratikaA: added vocab_dict based on label
        # self.add_metric(docsim)

        inll = Nll(data_loader=self.gen_data_loader, rnn=self.generator, sess=self.sess)
        inll.set_name('nll-test')
        self.add_metric(inll)

        n_grams = [2,3,4]
        for grams in n_grams:
            bleu_score = Bleu(test_text=self.test_file, real_text=data_loc, gram=grams)
            self.add_metric(bleu_score)

            self_bleu = SelfBleu(test_text=self.test_file, gram=grams)
            self.add_metric(self_bleu)


    def train_real(self, data_loc=None):
        from utils.text_process import code_to_text
        from utils.text_process import get_tokenlized
        wi_dict, iw_dict = self.init_real_trainng(data_loc)
        self.init_real_metric(data_loc)

        def get_real_test_file(dict=iw_dict):
            with open(self.generator_file, 'r') as file:
                codes, lb = parse_generator_file(self.generator_file)
            with open(self.test_file, 'w') as outfile:
                outfile.write(code_to_text_with_labels(codes=codes, dictionary=dict, labels=lb))

        self.sess.run(tf.global_variables_initializer())

        self.pre_epoch_num = 80
        self.adversarial_epoch_num = 50
        self.log = open('experiment-log-cseqgan-real_g2d1.csv', 'w')
        generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
        self.gen_data_loader.create_batches(self.oracle_file, self.labels)

        print('start pre-train generator:')
        for epoch in range(self.pre_epoch_num):
            start = time()
            loss = pre_train_epoch(self.sess, self.generator, self.gen_data_loader)
            end = time()
            print('epoch:' + str(self.epoch) + '\t time:' + str(end - start))
            self.add_epoch()
            if epoch % 5 == 0:
                generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
                get_real_test_file()
                self.evaluate()

        print('start pre-train discriminator:')
        self.reset_epoch()
        for epoch in range(self.pre_epoch_num):
            print('epoch:' + str(epoch))
            self.train_discriminator()

        self.reset_epoch()
        print('adversarial training:')
        self.reward = Reward(self.generator, .8)
        for epoch in range(self.adversarial_epoch_num):
            # print('epoch:' + str(epoch))
            start = time()
            for index in range(1):
                samples = self.generator.generate(self.sess)
                samples_x = samples[:,:-1]  #KratikaA: dividing sequences into text and label
                samples_label = samples[:,-1]
                samples_label = np.reshape(samples_label, [samples_label.shape[0], 1])
                rewards = self.reward.get_reward(self.sess, samples, 16, self.discriminator)
                feed = {
                    self.generator.x: samples_x,
                    self.generator.y: samples_label,
                    self.generator.rewards: rewards
                }
                loss, _ = self.sess.run([self.generator.g_loss, self.generator.g_updates], feed_dict=feed)
                print(loss)
            end = time()
            self.add_epoch()
            print('epoch:' + str(self.epoch) + '\t time:' + str(end - start))
            if epoch % 5 == 0 or epoch == self.adversarial_epoch_num - 1:
                generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
                get_real_test_file()
                self.evaluate()

            self.reward.update_params()
            for _ in range(15):
                self.train_discriminator()
