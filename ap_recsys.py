import numpy as np
from recsys.mongo import Mongo
from recsys.samplers.sampler import Sampler
from tensorflow as tf


class ApRecsys(object):

    def __init__(self, host, username, password, db_name):
        self._mongo = Mongo(host=host,
                            username=username, password=password, db_name=db_name)

        self._dim_item_embed = 50
        self._max_seq_len = 5
        self._total_iter = int(1e3)
        self._batch_size = 100
        self._eval_iter = 100
        self._train_percentage = 0.9

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = value

    @property
    def dim_item_embed(self):
        return self._dim_item_embed

    @dim_item_embed.setter
    def dim_item_embed(self, value):
        self._dim_item_embed = value

    @property
    def max_seq_len(self):
        return self._max_seq_len

    @max_seq_len.setter
    def max_seq_len(self, value):
        self._max_seq_len = value

    @property
    def total_iter(self):
        return self._total_iter

    @total_iter.setter
    def total_iter(self, value):
        self._total_iter = value

    @property
    def eval_iter(self):
        return self._eval_iter

    @eval_iter.setter
    def batch_size(self, value):
        self._eval_iter = value

    def make_raw_data(self):
        self._mongo.make_raw_data()

    def _train_batch(self):

        low_pos = self._mongo.total_raw_data * self._train_percentage
        while True:
            input_npy = np.zeros(self._batch_size,
                                 dtype=[('seq_item_id', (np.int32, self.max_seq_len)),
                                        ('seq_len', np.int32),
                                        ('label', np.int32)])

            index_list = np.random.randint(low=low_pos, high=self._mongo.total_raw_data - 1, size=self._batch_size)
            watch_histories_sample = [self._mongo.get_watch_list(ind) for ind in index_list]

            for ind, watch_history in enumerate(watch_histories_sample):
                predict_pos = np.random.randint(1, len(watch_history) - 1)
                train_items = watch_history[max(0, predict_pos - self._max_seq_len): predict_pos]
                pad_train_items = np.zeros(self.max_seq_len, np.int32)
                pad_train_items[:len(train_items)] = train_items
                input_npy[ind] = (pad_train_items, len(train_items), watch_history[predict_pos])

            yield input_npy

    def _test_batch(self):

        low_pos = self._mongo.total_raw_data * self._train_percentage
        index_list = np.arange(start=0, stop=low_pos, step=1)
        watch_histories_sample = dict()
        predict_pos = -1 # last position

        for ind in index_list:
            watch_histories_sample[ind] = self._mongo.get_watch_list(ind)

        while True:
            for ind, watch_history in watch_histories_sample:
                input_npy = np.zeros(1, dtype=[('seq_item_id', (np.int32, self.max_seq_len)),
                                               ('seq_len', np.int32)])

                self._mongo.get_watch_list(ind)
                train_items = watch_history[-self.max_seq_len-1:predict_pos]
                pad_train_items = np.zeros(self.max_seq_len, np.int32)
                pad_train_items[:len(train_items)] = train_items
                input_npy[0] = (pad_train_items, len(train_items))
                yield train_items[predict_pos], input_npy
                # yield [] []
            yield None, None

    def get_train_sampler(self):
        return Sampler(generate_batch=self._train_batch)

    def get_test_sampler(self):
        return Sampler(generate_batch=self._test_batch, num_process=1)

    def build_train_model(self):
        pass

    def build_serv_model(self):
        pass

    def get_model(self):
        t_seq_item_id = tf.placeholder(tf.int32, shape=(self.batch_size, self.max_seq_len), name='seq_item_id')
        t_seq_len = tf.placeholder(tf.int32, shape=(self.batch_size,), name='seq_len')
        t_label = tf.placeholder(tf.int32, shape=(self.batch_size,), name='label')

        # s_seq_item_id = tf.placeholder(tf.int32, shape=(self.batch_size, self.max_seq_len), name='seq_item_id')
        # s_seq_len = tf.placeholder(tf.int32, shape=(self.batch_size,), name='seq_len')

        initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01, dtype=tf.float32)

        item_embedding = tf.get_variable(name='embedding',
                                         shape=(self.total_items, self.dim_item_embed),
                                         trainable=True,
                                         initializer=initializer)

        t_seq_vec = tf.nn.embedding_lookup(item_embedding, t_seq_item_id)

        t_seq_vec = tf.nn.embedding_lookup(item_embedding, t_seq_item_id)
        # average item vectors user interacted with
        t_seq_mask = tf.sequence_mask(t_seq_len, self.max_seq_len, dtype=tf.float32)
        t_item = tf.reduce_mean(t_seq_vec * tf.expand_dims(t_seq_mask, axis=2), axis=1)
        t_tensor = tf.concat([t_item], axis=1)
        t_in = tf.nn.relu(t_tensor)


        s_seq_vec = tf.nn.embedding_lookup(item_embedding, s_seq_item_id)
        # average item vectors user interacted with
        s_seq_mask = tf.sequence_mask(s_seq_len, self.max_seq_len, dtype=tf.float32)
        s_item = tf.reduce_mean(s_seq_vec * tf.expand_dims(s_seq_mask, axis=2), axis=1)
        s_tensor = tf.concat([s_item], axis=1)
        s_in = tf.nn.relu(s_tensor)

        mat_1 = tf.get_variable(name='FC_1', shape=(t_in.shape[1], 300), trainable=True, reuse=True,
                              initializer=tf.contrib.layers.xavier_initializer())
        mat_1_bias = tf.get_variable(name='bias_1', shape=(300,), trainable=True, reuse=True,
                                initializer=tf.constant_initializer(value=0.0, dtype=tf.float32))

        mat_2 = tf.get_variable(name='FC_2', shape=(300, self._mongo.total_movies), trainable=True, reuse=True,
                                initializer=tf.contrib.layers.xavier_initializer())
        mat_2_bias = tf.get_variable(name='bias_2', shape=(300,), trainable=True, reuse=True,
                                     initializer=tf.constant_initializer(value=0.0, dtype=tf.float32))

        t_out = tf.nn.relu(tf.matmul(t_in, mat_1) + mat_1_bias)
        s_out = tf.nn.relu(tf.matmul(s_in, mat_1) + mat_1_bias)

        # t_out = tf.contrib.layers.batch_norm(t_out, fused=True, decay=0.95,
        #                                     center=True, scale=True, is_training=True,
        #                                     scope="bn_1", updates_collections=None)

        t_out = tf.nn.relu(tf.matmul(t_out, mat_2) + mat_2_bias)
        s_out = tf.nn.relu(tf.matmul(s_out, mat_2) + mat_2_bias)

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=t_label,
                                                              logits=t_out)
        loss = tf.add_n(tf.reduce_mean(loss))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        training = optimizer.minimize(loss)

def main():
    print('start')

    ap_model = ApRecsys(host='13.209.6.203',
                        username='romi',
                        password="Amore12345!",
                        db_name='recsys')

    ap_model.make_raw_data()

    train_sampler = ap_model.get_train_sampler()
    # test_sampler = ap_model.get_test_sampler()

    for _ in range(1000):
        print(train_sampler.next_batch())

    # train_dataset = Dataset(train_data, total)


if __name__ == '__main__':
    main()
