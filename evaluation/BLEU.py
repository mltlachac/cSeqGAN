import getopt
import sys
from utils.metrics.Bleu import Bleu
from utils.metrics.SelfBleu import SelfBleu

def divide_files(test_file, test_file_pos, test_file_neg):
    pos_seq = ''
    neg_seq = ''
    with open(test_file, 'r') as gen_file:
        for text in gen_file:
            text = text.strip().rpartition(',')
            label = text[-1]
            text = text[0]
            if label == '1':
                pos_seq += text + '\n'
            elif label == '0':
                neg_seq += text + '\n'
    with open(test_file_pos, 'w') as pos_file:
        pos_file.write(pos_seq)
    with open(test_file_neg, 'w') as neg_file:
        neg_file.write(neg_seq)

    return

def combine_files(test_file, test_file_pos, test_file_neg):
    data = ''
    with open(test_file_pos, 'r') as file:
        for line in file:
            data += line + ',1' + '\n'
    with open(test_file_neg, 'r') as file:
        for line in file:
            data += line + ',0' + '\n'
    with open(test_file, 'w') as file:
        file.write(data)
        
    return


def get_bleu_score(test_file, data_loc, gram):
    bleu_score = Bleu(test_text=test_file, real_text=data_loc, gram=gram)
    selfBleu_score = SelfBleu(test_text=test_file, gram=gram)
    return bleu_score, selfBleu_score


if __name__ == '__main__':
    argv = sys.argv[1:]
    opts, args = getopt.getopt(argv, "c:p:n:")
    opt_args = dict(opts)

    real_data = real_pos = real_neg = ''
    if '-c' in opt_args:
        real_data = opt_args['-c']
    if '-p' in opt_args:
        real_pos = opt_args['-p']
    if '-n' in opt_args:
        real_neg = opt_args['-n']

    file_path = 'save/test_file'
    generator = [0,1,2,3]
    discriminator = [0,1,2,3]

    # real files names
    #real_pos = 'data/movie_review_pos.csv'
    #real_neg = 'data/movie_review_pos.csv'
    #real_data = 'data/movie_review.csv'

    # log file
    pos_log = open('experiment_positive.csv', 'w')
    neg_log = open('experiment_negative.csv', 'w')
    combined_log = open('experiment_combined.csv', 'w')

    pos_log.write('model, bleu_2, selfBleu_2, bleu_3, selfBleu_3, bleu_4, selfBleu_4, \n')
    neg_log.write('model, bleu_2, selfBleu_2, bleu_3, selfBleu_3, bleu_4, selfBleu_4, \n')
    combined_log.write('model, bleu_2, selfBleu_2, bleu_3, selfBleu_3, bleu_4, selfBleu_4, \n')

    for g in generator:
        for d in discriminator:
            if (d==0 and g!=0) or (g==0 and d!=0):
                continue
            # generated files names
            testName = file_path + '_g' + str(g) + 'd' + str(d)
            test_file = testName + '.txt'
            test_file_pos = testName + '_pos.txt'
            test_file_neg = testName + '_neg.txt'
            
            if g==0 and d==0:
                combine_files(test_file, test_file_pos, test_file_neg)
            else:
                divide_files(test_file, test_file_pos, test_file_neg)

            grams = [2,3,4]

            model_name = 'model' + '_g' + str(g) + 'd' + str(d)
            pos_log.write(model_name + ',')
            neg_log.write(model_name + ',')
            combined_log.write(model_name + ',')

            for gram in grams:
                bleu_score, selfBleu_score = get_bleu_score(test_file_pos, real_pos, gram)
                pos_log.write(str(bleu_score.get_score()) + ', ' + str(selfBleu_score.get_score()) + ', ')

                bleu_score, selfBleu_score = get_bleu_score(test_file_neg, real_neg, gram)
                neg_log.write(str(bleu_score.get_score()) + ', ' + str(selfBleu_score.get_score()) + ', ')

                bleu_score, selfBleu_score = get_bleu_score(test_file, real_data, gram)
                combined_log.write(str(bleu_score.get_score()) + ', ' + str(selfBleu_score.get_score()) + ', ')

            pos_log.write('\n')
            neg_log.write('\n')
            combined_log.write('\n')

    pos_log.close()
    neg_log.close()
    combined_log.close()
