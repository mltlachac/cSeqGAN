import numpy as np
import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2


def generate_samples(sess, trainable_model, batch_size, generated_num, output_file=None, get_code=True):
    # Generate Samples
    generated_samples = []
    for _ in range(int(generated_num / batch_size)):
        generated_samples.extend(trainable_model.generate(sess))
    codes = list()
    if output_file is not None:
        with open(output_file, 'w') as fout:
            for poem in generated_samples:
                buffer = ' '.join([str(x) for x in poem]) +  '\n'
                fout.write(buffer)
                if get_code:
                    codes.append(poem)
        return np.array(codes)
    codes = ""
    for poem in generated_samples:
        buffer = ' '.join([str(x) for x in poem]) +  '\n'   #KratikaA: modified to append label followed by comma at the end of the generated text
        codes += buffer
    return codes


def init_sess():
    config = tf.ConfigProto()
    off = rewriter_config_pb2.RewriterConfig.OFF
    config.graph_options.rewrite_options.arithmetic_optimization = off
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    return sess


def pre_train_epoch(sess, trainable_model, data_loader):
    # Pre-train the generator using MLE for one epoch
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in range(data_loader.num_batch):
        batch, labels = data_loader.next_batch()    #KratikaA: added labels as well
        _, g_loss = trainable_model.pretrain_step(sess, batch, labels)  #KratikaA: passed labels to the pretrain_step
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)
