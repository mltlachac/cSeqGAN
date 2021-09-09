import numpy as np

from utils.metrics.Metrics import Metrics


class Nll(Metrics):
    def __init__(self, data_loader, rnn, sess):
        super().__init__()
        self.name = 'nll-oracle'
        self.data_loader = data_loader
        self.sess = sess
        self.rnn = rnn

    def set_name(self, name):
        self.name = name

    def get_name(self):
        return self.name

    def get_score(self):
        return self.nll_loss()

    def nll_loss(self):
        nll = []
        self.data_loader.reset_pointer()
        for it in range(self.data_loader.num_batch):
            batch = self.data_loader.next_batch()
            x = batch[0]
            y = np.reshape(batch[1], [batch[1].shape[0], 1])
            # fixme bad taste
            try:
                g_loss = self.rnn.get_nll(self.sess, x)
            except Exception as e:
                g_loss = self.sess.run(self.rnn.pretrain_loss, {self.rnn.x: x, self.rnn.y: y})
            nll.append(g_loss)
        return np.mean(nll)
