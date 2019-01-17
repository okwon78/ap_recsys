import os
import shutil

import tensorflow as tf
import numpy as np

from recsys.model import Model
from recsys.mongo import Mongo
from recsys.samplers.sampler import Sampler

print(tf.__version__)


class ApRecsys(object):

    def __init__(self, host, username, password, db_name):
        self._mongo = Mongo(host=host,
                            username=username, password=password, db_name=db_name)

        self._embedding_size = 20
        self._dim_item_embed = 50
        self._max_seq_len = 5
        self._batch_size = 100
        self._eval_iter = 10
        self._train_percentage = 0.9
        self._model = Model()
        self._save_model_dir = './model'
        self._train_summary_path = '/Users/amore/Dev/ap_recsys/train'
        self._serve_summary_path = '/Users/amore/Dev/ap_recsys/serve'
        self._train_tensors = None
        self._serve_tensors = None
        self._train_session = None
        self._serve_session = None
        self._train_saver = None
        self._serve_saver = None
        self._train_writer = None
        self._serve_writer = None

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

    @property
    def train_writer(self):
        if self._train_writer is None:
            raise Exception

        return self._train_writer

    def make_raw_data(self):
        self._mongo.make_raw_data()

    def make_movie_index(self):
        self._mongo.make_movie_index()

    def _train_batch(self):

        low_pos = self._mongo.total_raw_data * self._train_percentage
        while True:
            input_npy = np.zeros(self._batch_size,
                                 dtype=[('seq_item_id', (np.int32, self.max_seq_len)),
                                        ('seq_len', np.int32),
                                        ('label', np.int32)])

            watch_histories_sample = []
            while True:
                index = np.random.randint(low=low_pos, high=self._mongo.total_raw_data - 1)
                watch_history = self._mongo.get_watch_list(index)
                if watch_history is not None:
                    watch_histories_sample.append(watch_history)

                if len(watch_histories_sample) == self._batch_size:
                    break

            for ind, watch_history in enumerate(watch_histories_sample):
                predict_pos = np.random.randint(low=1, high=len(watch_history))
                train_items = watch_history[max(0, predict_pos - self._max_seq_len): predict_pos]
                train_items = [self._mongo.get_index_from_movieId(movieId) for movieId in train_items]

                pad_train_items = np.zeros(self.max_seq_len, np.int32)
                pad_train_items[:len(train_items)] = train_items
                predict_index = self._mongo.get_index_from_movieId(watch_history[predict_pos])
                input_npy[ind] = (pad_train_items, len(train_items), predict_index)

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
                yield [], []
            yield None, None

    def get_train_sampler(self):
        return Sampler(generate_batch=self._train_batch, num_process=1)

    def get_test_sampler(self):
        return Sampler(generate_batch=self._test_batch, num_process=1)

    def build_train_model(self):

        self._train_tensors = self._model.build_train_model(batch_size=self._batch_size,
                                                            embedding_size=self._embedding_size,
                                                            dim_item_embed=self.dim_item_embed,
                                                            total_items=self._mongo.total_movies,
                                                            max_seq_len=self.max_seq_len)

        with self._model.get_train_graph().as_default():
            self._train_session = tf.Session()
            self._train_session.run(tf.global_variables_initializer())
            self._train_saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
            self.restore()
            self._train_writer = tf.summary.FileWriter(self._train_summary_path, self._model.get_train_graph())

    def build_serve_model(self):

        self._serve_tensors = self._model.build_serve_model(embedding_size=self._embedding_size,
                                                            dim_item_embed=self.dim_item_embed,
                                                            total_items=self._mongo.total_movies,
                                                            max_seq_len=self.max_seq_len)

        with self._model.get_serve_graph().as_default():
            self._serve_session = tf.Session()
            self._serve_session.run(tf.global_variables_initializer())
            self._serve_saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
            self._serve_writer = tf.summary.FileWriter(self._serve_summary_path, self._model.get_serve_graph())

    def train(self, step, batch_data):
        """train"""
        with self._model.get_train_graph().as_default():
            loss = self._train_tensors['loss']
            backprop = self._train_tensors['backprop']
            summary = self._train_tensors['summary']

            tf.summary.scalar('loss', loss)

            feed_dict = {
                self._train_tensors['seq_item_id']: batch_data['seq_item_id'],
                self._train_tensors['seq_len']: batch_data['seq_len'],
                self._train_tensors['label']: batch_data['label']
            }

            backprop, loss_, summary_ = self._train_session.run([backprop, loss, summary], feed_dict=feed_dict)
            self._train_writer.add_summary(summary_, step)
            return loss_

    def save(self, step):
        with self._model.get_train_graph().as_default():
            self._train_saver.save(self._train_session, os.path.join(self._save_model_dir, 'model.ckpt'))

    def restore(self):
        if not os.path.exists(self._save_model_dir):
            return

        # saved_dir = os.path.join(self._save_model_dir, 'model.ckpt')
        self._train_saver.restore(self._train_session, os.path.join(self._save_model_dir, 'model.ckpt'))


def main():
    ap_model = ApRecsys(host='13.209.6.203',
                        username='romi',
                        password="Amore12345!",
                        db_name='recsys')

    # ap_model.make_raw_data()
    ap_model.make_movie_index()

    train_sampler = ap_model.get_train_sampler()
    # test_sampler = ap_model.get_test_sampler()

    ap_model.build_train_model()
    ap_model.build_serve_model()

    acc_loss = 0
    total_iter = 0
    while True:
        batch_data = train_sampler.next_batch_debug()
        loss = ap_model.train(total_iter, batch_data)

        acc_loss += loss
        total_iter += 1
        print(f'[{total_iter}] loss: {loss}')

        if total_iter % ap_model.eval_iter == 0 or total_iter == 1:
            summary = tf.Summary()

            ap_model.save(step=total_iter)
            print('model saved')
            avg_loss = acc_loss / ap_model.eval_iter
            print(f'[{total_iter}] avg_loss: {avg_loss}')
            # summary.value.add(tag='avg_loss', simple_value=avg_loss)
            # ap_model.train_writer.add_summary(summary, total_iter)
            # ap_model.train_writer.flush()
            acc_loss = 0


if __name__ == '__main__':
    main()
