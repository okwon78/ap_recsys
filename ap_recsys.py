import tensorflow as tf
import numpy as np

from recsys.model import Model
from recsys.mongo import Mongo
from recsys.samplers.sampler import Sampler


class ApRecsys(object):

    def __init__(self, host, username, password, db_name):
        self._mongo = Mongo(host=host,
                            username=username, password=password, db_name=db_name)

        self._embedding_size = 20
        self._dim_item_embed = 50
        self._max_seq_len = 5
        self._total_iter = int(1e3)
        self._batch_size = 100
        self._eval_iter = 100
        self._train_percentage = 0.9
        self._model = Model('ap_recsys_dir')
        self._train_tensors = None
        self._serve_tensors = None
        self._train_session = None
        self._serve_session = None
        self._train_saver = None
        self._serve_saver = None


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
    def eval_iter(self, value):
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
        predict_pos = -1  # last position

        for ind in index_list:
            watch_histories_sample[ind] = self._mongo.get_watch_list(ind)

        while True:
            for ind, watch_history in watch_histories_sample:
                input_npy = np.zeros(1, dtype=[('seq_item_id', (np.int32, self.max_seq_len)),
                                               ('seq_len', np.int32)])

                self._mongo.get_watch_list(ind)
                train_items = watch_history[-self.max_seq_len - 1:predict_pos]
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

        self._train_tensors = self._model.build_train_model(batch_size=self._batch_size,
                                                            embedding_size=self._embedding_size,
                                                            dim_item_embed=self.dim_item_embed,
                                                            total_items=self._mongo.total_movies,
                                                            max_seq_len=self.max_seq_len)

        train_graph = self._model.get_train_graph()

        with train_graph.as_default():
            self._train_session = tf.Session()
            self._train_session.run(tf.global_variables_initializer())
            self._train_saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)


    def build_serve_model(self):

        self._serve_tensors = self._model.build_serve_model(embedding_size=self._embedding_size,
                                                           dim_item_embed=self.dim_item_embed,
                                                           total_items=self._mongo.total_movies,
                                                           max_seq_len=self.max_seq_len)

        serve_graph = self._model.get_serve_graph()

        with serve_graph.as_default():
            self._serve_session = tf.Session()
            self._serve_session.run(tf.global_variables_initializer())
            self._serve_saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)

    def train(self, batch_data):
        """train"""
        train_graph = self._model.get_train_graph()
        serve_graph = self._model.get_serve_graph()
        losses = self._train_tensors['losses']

        with train_graph.tf_graph.as_default():

            feed_dict = {
                self._train_tensors['seq_item_id']: batch_data['seq_item_id'],
                self._train_tensors['seq_len']: batch_data['seq_len'],
                self._train_tensors['label']: batch_data['label']
            }

            return self._train_session.run([losses])


def main():

    ap_model = ApRecsys(host='13.209.6.203',
                        username='romi',
                        password="Amore12345!",
                        db_name='recsys')

    # ap_model.make_raw_data()

    train_sampler = ap_model.get_train_sampler()
    #test_sampler = ap_model.get_test_sampler()

    ap_model.build_train_model()
    ap_model.build_serve_model()

    acc_loss = 0
    for _ in range(ap_model.total_iter):
        batch_data = train_sampler.next_batch()
        print(batch_data)
        # loss = np.sum(ap_model.train(batch_data))
        # acc_loss += loss



if __name__ == '__main__':
    main()
