import numpy as np
import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops, control_flow_ops


class Reward(object):
    def __init__(self, lstm, update_rate):
        self.lstm = lstm
        self.update_rate = update_rate

        self.num_vocabulary = self.lstm.num_vocabulary
        self.batch_size = self.lstm.batch_size
        self.emb_dim = self.lstm.emb_dim
        self.hidden_dim = self.lstm.hidden_dim
        self.sequence_length = self.lstm.sequence_length
        self.start_token = tf.identity(self.lstm.start_token)
        self.label_token = tf.identity(self.lstm.label_token)
        self.learning_rate = self.lstm.learning_rate
        self.num_classes = self.lstm.num_classes

        self.g_embeddings = tf.identity(self.lstm.g_embeddings)
        self.g_recurrent_unit = self.create_recurrent_unit()  # maps h_tm1 to h_t for generator
        self.g_output_unit = self.create_output_unit()  # maps h_t to o_t (output token logits)
        #####################################################################################################
        # placeholder definition
        self.x = tf.placeholder(tf.int32, shape=[self.batch_size,
                                                 self.sequence_length])  # sequence of tokens generated by generator #KratikaA: reduced 1 from sequence length as it already had length adjusted in generator
        self.y = tf.placeholder(tf.int32, [self.batch_size, 1], name="y") #KratikaA: added for class label
        self.z = tf.concat([self.x, self.y], 1) #KratikaA: concatenated input sequence with label along 1-axis(column)

        self.given_num = tf.placeholder(tf.int32)

        # processed for batch
        with tf.device("/cpu:0"):
            self.processed_x = tf.transpose(tf.nn.embedding_lookup(self.g_embeddings[0:self.num_vocabulary], self.x),
                                            perm=[1, 0, 2])  # seq_length x batch_size x emb_dim
            self.processed_y = tf.transpose(tf.nn.embedding_lookup(self.g_embeddings[self.num_vocabulary:], self.y),    #KratikaA: embedding class label similar to the input text
                                            perm=[1, 0, 2])  # class_label x batch_size x emb_dim

            self.processed_z = tf.concat([self.processed_x, self.processed_y], 0)     #KratikaA: concatenated input embedded and label embedded tensors along 0-axis (column)

        ta_emb_x = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self.sequence_length+1)  #KratikaA: sequence_length+1 for handling label
        ta_emb_x = ta_emb_x.unstack(self.processed_z) #KratikaA: changed param from self.processed_x to self.processed_z

        ta_x = tensor_array_ops.TensorArray(dtype=tf.int32, size=self.sequence_length+1) #KratikaA: sequence_length+1 for handling label
        ta_x = ta_x.unstack(tf.transpose(self.z, perm=[1, 0]))  #KratikaA: changed param from self.x to self.z
        #####################################################################################################

        self.h0 = tf.zeros([self.batch_size, self.hidden_dim])
        self.h0 = tf.stack([self.h0, self.h0])

        gen_x = tensor_array_ops.TensorArray(dtype=tf.int32, size=self.sequence_length+1,#KratikaA: sequence_length+1 for handling label
                                             dynamic_size=False, infer_shape=True)

        # When current index i < given_num, use the provided tokens as the input at each time step
        def _g_recurrence_1(i, x_t, h_tm1, given_num, gen_x, c1_t):
            # x_t_combined = tf.math.add(x_t, c1_t)
            # h_t = self.g_recurrent_unit(x_t_combined, h_tm1)  # hidden_memory_tuple
            h_t = self.g_recurrent_unit(x_t, c1_t, h_tm1)  # hidden_memory_tuple
            x_tp1 = ta_emb_x.read(i)
            gen_x = gen_x.write(i, ta_x.read(i))
            return i + 1, x_tp1, h_t, given_num, gen_x, c1_t

        # When current index i >= given_num, start roll-out, use the output as time step t as the input at time step t+1
        def _g_recurrence_2(i, x_t, h_tm1, given_num, gen_x, c1_t):
            # x_t_combined = tf.math.add(x_t, c1_t)
            # h_t = self.g_recurrent_unit(x_t_combined, h_tm1)  # hidden_memory_tuple
            h_t = self.g_recurrent_unit(x_t, c1_t, h_tm1)  # hidden_memory_tuple
            o_t = self.g_output_unit(h_t)  # batch x vocab , logits not prob
            log_prob = tf.log(tf.nn.softmax(o_t))
            def get_token():
                start_idx = 0
                end_idx = self.num_vocabulary
                next_token = tf.cast(tf.reshape(tf.multinomial(log_prob[:,start_idx:end_idx], 1), [self.batch_size]), tf.int32)
                return start_idx, end_idx, next_token
            def get_label():
                start_idx = self.num_vocabulary
                end_idx = self.num_vocabulary+self.num_classes
                next_token = tf.cast(tf.reshape(self.label_token, [self.batch_size]), tf.int32) # return the same label as the last next_token
                return start_idx, end_idx, next_token

            start_idx, end_idx, next_token = control_flow_ops.cond(i < self.sequence_length, get_token, get_label)

            x_tp1 = tf.nn.embedding_lookup(self.g_embeddings[start_idx:end_idx], next_token)  # batch x emb_dim
            gen_x = gen_x.write(i, next_token)  # indices, batch_size
            return i + 1, x_tp1, h_t, given_num, gen_x, c1_t

        i, x_t, h_tm1, given_num, self.gen_x, c1_t = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, given_num, _4, _5: i < given_num,
            body=_g_recurrence_1,
            loop_vars=(tf.constant(0, dtype=tf.int32),
                       tf.nn.embedding_lookup(self.g_embeddings[0:self.num_vocabulary], self.start_token), self.h0, self.given_num, gen_x,
                       tf.nn.embedding_lookup(self.g_embeddings[self.num_vocabulary:], self.label_token)))

        _, _, _, _, self.gen_x, _ = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3, _4, _5: i < self.sequence_length+1,
            body=_g_recurrence_2,
            loop_vars=(i, x_t, h_tm1, given_num, self.gen_x, c1_t))

        self.gen_x = self.gen_x.stack()  # seq_length x batch_size
        self.gen_x = tf.transpose(self.gen_x, perm=[1, 0])  # batch_size x seq_length

    def get_reward(self, sess, input, rollout_num, discriminator):
        rewards = []
        input_x = input[:,:-1]  #KratikaA: used input to yield input_x and input_y
        input_y = input[:,-1]
        input_y = np.reshape(input_y, [input_y.shape[0], 1])
        for i in range(rollout_num):
            for given_num in range(1, len(input[0])):
                feed = {self.x: input_x, self.y: input_y, self.given_num: given_num}    #KratikaA: added input_y
                samples = sess.run(self.gen_x, feed)
                feed = {discriminator.input_x: samples}
                ypred_for_auc = sess.run(discriminator.ypred_for_auc, feed)
                ypred = np.array([item[1] for item in ypred_for_auc])
                if i == 0:
                    rewards.append(ypred)
                else:
                    rewards[given_num - 1] += ypred

            # the last token reward
            feed = {discriminator.input_x: input} #KratikaA: changed from input_x to input
            ypred_for_auc = sess.run(discriminator.ypred_for_auc, feed)
            ypred = np.array([item[1] for item in ypred_for_auc])
            if i == 0:
                rewards.append(ypred)
            else:
                rewards[(len(input_x[0])-1)] += ypred

        reward_res = np.transpose(np.array(rewards)) / (1.0 * rollout_num)  # batch_size x seq_length
        return reward_res

    def create_recurrent_unit(self):
        # Weights and Bias for input and hidden tensor
        self.Wi = tf.identity(self.lstm.Wi)
        self.Vi = tf.identity(self.lstm.Vi) #KratikaA for label
        self.Ui = tf.identity(self.lstm.Ui)
        self.bi = tf.identity(self.lstm.bi)

        self.Wf = tf.identity(self.lstm.Wf)
        self.Vf = tf.identity(self.lstm.Vf) #KratikaA for label
        self.Uf = tf.identity(self.lstm.Uf)
        self.bf = tf.identity(self.lstm.bf)

        self.Wog = tf.identity(self.lstm.Wog)
        self.Vog = tf.identity(self.lstm.Vog) #KratikaA for label
        self.Uog = tf.identity(self.lstm.Uog)
        self.bog = tf.identity(self.lstm.bog)

        self.Wc = tf.identity(self.lstm.Wc)
        self.Vc = tf.identity(self.lstm.Vc) #KratikaA for label
        self.Uc = tf.identity(self.lstm.Uc)
        self.bc = tf.identity(self.lstm.bc)

        def unit(x, c1, hidden_memory_tm1):
            previous_hidden_state, c_prev = tf.unstack(hidden_memory_tm1)

            # Input Gate
            i = tf.sigmoid(
                tf.matmul(x, self.Wi) +
                tf.matmul(c1, self.Vi) +
                tf.matmul(previous_hidden_state, self.Ui) + self.bi
            )

            # Forget Gate
            f = tf.sigmoid(
                tf.matmul(x, self.Wf) +
                tf.matmul(c1, self.Vf) +
                tf.matmul(previous_hidden_state, self.Uf) + self.bf
            )

            # Output Gate
            o = tf.sigmoid(
                tf.matmul(x, self.Wog) +
                tf.matmul(c1, self.Vog) +
                tf.matmul(previous_hidden_state, self.Uog) + self.bog
            )

            # New Memory Cell
            c_ = tf.nn.tanh(
                tf.matmul(x, self.Wc) +
                tf.matmul(c1, self.Vc) +
                tf.matmul(previous_hidden_state, self.Uc) + self.bc
            )

            # Final Memory cell
            c = f * c_prev + i * c_

            # Current Hidden state
            current_hidden_state = o * tf.nn.tanh(c)

            return tf.stack([current_hidden_state, c])

        return unit

    def update_recurrent_unit(self):
        # Weights and Bias for input and hidden tensor
        self.Wi = self.update_rate * self.Wi + (1 - self.update_rate) * tf.identity(self.lstm.Wi)
        self.Vi = self.update_rate * self.Vi + (1 - self.update_rate) * tf.identity(self.lstm.Vi) #KratikaA for label
        self.Ui = self.update_rate * self.Ui + (1 - self.update_rate) * tf.identity(self.lstm.Ui)
        self.bi = self.update_rate * self.bi + (1 - self.update_rate) * tf.identity(self.lstm.bi)

        self.Wf = self.update_rate * self.Wf + (1 - self.update_rate) * tf.identity(self.lstm.Wf)
        self.Vf = self.update_rate * self.Vf + (1 - self.update_rate) * tf.identity(self.lstm.Vf) #KratikaA for label
        self.Uf = self.update_rate * self.Uf + (1 - self.update_rate) * tf.identity(self.lstm.Uf)
        self.bf = self.update_rate * self.bf + (1 - self.update_rate) * tf.identity(self.lstm.bf)

        self.Wog = self.update_rate * self.Wog + (1 - self.update_rate) * tf.identity(self.lstm.Wog)
        self.Vog = self.update_rate * self.Vog + (1 - self.update_rate) * tf.identity(self.lstm.Vog) #KratikaA for label
        self.Uog = self.update_rate * self.Uog + (1 - self.update_rate) * tf.identity(self.lstm.Uog)
        self.bog = self.update_rate * self.bog + (1 - self.update_rate) * tf.identity(self.lstm.bog)

        self.Wc = self.update_rate * self.Wc + (1 - self.update_rate) * tf.identity(self.lstm.Wc)
        self.Vc = self.update_rate * self.Vc + (1 - self.update_rate) * tf.identity(self.lstm.Vc) #KratikaA for label
        self.Uc = self.update_rate * self.Uc + (1 - self.update_rate) * tf.identity(self.lstm.Uc)
        self.bc = self.update_rate * self.bc + (1 - self.update_rate) * tf.identity(self.lstm.bc)

        def unit(x, hidden_memory_tm1):
            previous_hidden_state, c_prev = tf.unstack(hidden_memory_tm1)

            # Input Gate
            i = tf.sigmoid(
                tf.matmul(x, self.Wi) +
                tf.matmul(c1, self.Vi) +
                tf.matmul(previous_hidden_state, self.Ui) + self.bi
            )

            # Forget Gate
            f = tf.sigmoid(
                tf.matmul(x, self.Wf) +
                tf.matmul(c1, self.Vf) +
                tf.matmul(previous_hidden_state, self.Uf) + self.bf
            )

            # Output Gate
            o = tf.sigmoid(
                tf.matmul(x, self.Wog) +
                tf.matmul(c1, self.Vog) +
                tf.matmul(previous_hidden_state, self.Uog) + self.bog
            )

            # New Memory Cell
            c_ = tf.nn.tanh(
                tf.matmul(x, self.Wc) +
                tf.matmul(c1, self.Vc) +
                tf.matmul(previous_hidden_state, self.Uc) + self.bc
            )

            # Final Memory cell
            c = f * c_prev + i * c_

            # Current Hidden state
            current_hidden_state = o * tf.nn.tanh(c)

            return tf.stack([current_hidden_state, c])

        return unit

    def create_output_unit(self):
        self.Wo = tf.identity(self.lstm.Wo)
        self.bo = tf.identity(self.lstm.bo)

        def unit(hidden_memory_tuple):
            hidden_state, c_prev = tf.unstack(hidden_memory_tuple)
            # hidden_state : batch x hidden_dim
            logits = tf.matmul(hidden_state, self.Wo) + self.bo
            # output = tf.nn.softmax(logits)
            return logits

        return unit

    def update_output_unit(self):
        self.Wo = self.update_rate * self.Wo + (1 - self.update_rate) * tf.identity(self.lstm.Wo)
        self.bo = self.update_rate * self.bo + (1 - self.update_rate) * tf.identity(self.lstm.bo)

        def unit(hidden_memory_tuple):
            hidden_state, c_prev = tf.unstack(hidden_memory_tuple)
            # hidden_state : batch x hidden_dim
            logits = tf.matmul(hidden_state, self.Wo) + self.bo
            # output = tf.nn.softmax(logits)
            return logits
        return unit

    def update_params(self):
        self.g_embeddings = tf.identity(self.lstm.g_embeddings)
        self.g_recurrent_unit = self.update_recurrent_unit()
        self.g_output_unit = self.update_output_unit()
