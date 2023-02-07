import getopt
import sys
import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2
config = tf.ConfigProto()
off = rewriter_config_pb2.RewriterConfig.OFF
config.graph_options.rewrite_options.memory_optimization  = off

from colorama import Fore

from models.seqgan_pos.Seqgan import Seqgan as Seqgan_pos
from models.seqgan_neg.Seqgan import Seqgan as Seqgan_neg
from models.cseqgan_g1d1.CSeqgan import CSeqgan as CSeqgan_g1d1
from models.cseqgan_g1d2.CSeqgan import CSeqgan as CSeqgan_g1d2
from models.cseqgan_g1d3.CSeqgan import CSeqgan as CSeqgan_g1d3
from models.cseqgan_g2d1.CSeqgan import CSeqgan as CSeqgan_g2d1
from models.cseqgan_g2d2.CSeqgan import CSeqgan as CSeqgan_g2d2
from models.cseqgan_g2d3.CSeqgan import CSeqgan as CSeqgan_g2d3
from models.cseqgan_g3d1.CSeqgan import CSeqgan as CSeqgan_g3d1
from models.cseqgan_g3d2.CSeqgan import CSeqgan as CSeqgan_g3d2
from models.cseqgan_g3d3.CSeqgan import CSeqgan as CSeqgan_g3d3



def set_gan(gan_name):
    gans = dict()
    gans['seqgan_pos'] = Seqgan_pos
    gans['seqgan_neg'] = Seqgan_neg
    gans['cseqgan_g1d1'] = CSeqgan_g1d1
    gans['cseqgan_g1d2'] = CSeqgan_g1d2
    gans['cseqgan_g1d3'] = CSeqgan_g1d3
    gans['cseqgan_g2d1'] = CSeqgan_g2d1
    gans['cseqgan_g2d2'] = CSeqgan_g2d2
    gans['cseqgan_g2d3'] = CSeqgan_g2d3
    gans['cseqgan_g3d1'] = CSeqgan_g3d1
    gans['cseqgan_g3d2'] = CSeqgan_g3d2
    gans['cseqgan_g3d3'] = CSeqgan_g3d3

    try:
        Gan = gans[gan_name.lower()]
        gan = Gan()
        gan.vocab_size = 5000
        gan.generate_num = 4500
        return gan
    except KeyError:
        print(Fore.RED + 'Unsupported GAN type: ' + gan_name + Fore.RESET)
        sys.exit(-2)



def set_training(gan, training_method):
    try:
        if training_method == 'oracle':
            gan_func = gan.train_oracle
        elif training_method == 'cfg':
            gan_func = gan.train_cfg
        elif training_method == 'real':
            gan_func = gan.train_real
        else:
            print(Fore.RED + 'Unsupported training setting: ' + training_method + Fore.RESET)
            sys.exit(-3)
    except AttributeError:
        print(Fore.RED + 'Unsupported training setting: ' + training_method + Fore.RESET)
        sys.exit(-3)
    return gan_func


def parse_cmd(argv):
    try:
        opts, args = getopt.getopt(argv, "hg:t:d:p:")

        opt_arg = dict(opts)
        if '-h' in opt_arg.keys():
            print('usage: python main.py -g <gan_type>')
            print('       python main.py -g <gan_type> -t <train_type>')
            print('       python main.py -g <gan_type> -t realdata -d <your_data_location> -p <location_to_save_generated_files>')
            sys.exit(0)
        if not '-g' in opt_arg.keys():
            print('unspecified GAN type, use MLE training only...')
            gan = set_gan('mle')
        else:
            gan = set_gan(opt_arg['-g'])
        if not '-t' in opt_arg.keys():
            gan.train_oracle()
        else:
            gan_func = set_training(gan, opt_arg['-t'])
            if opt_arg['-t'] == 'real' and '-d' in opt_arg.keys() and '-p' in opt_arg.keys():
                gan_func(opt_arg['-d'], opt_arg['-p'])
            else:
                gan_func()
    except getopt.GetoptError:
        print('invalid arguments!')
        print('`python main.py -h`  for help')
        sys.exit(-1)
    pass


if __name__ == '__main__':
    gan = None
    parse_cmd(sys.argv[1:])
