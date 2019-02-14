import os
import shutil

import numpy as np
import tensorflow as tf
from collections import defaultdict

from recsys.rec_model_impl import RecModel
from recsys.train.mongo_client import MongoClient
from recsys.train.eval_manager import EvalManager
from recsys.samplers.sampler import Sampler


class ApRecsys(object):

    def __init__(self, model_dir, mongoConfig):

        self._mongo = MongoClient(host=mongoConfig.host,
                                  username=mongoConfig.username,
                                  password=mongoConfig.password,
                                  db_name=mongoConfig.dbname)

        self._embedding_size = 100
        self._dim_item_embed = 100
        self._max_seq_len = 10
        self._batch_size = 100
        self._eval_iter = 1000
        self._eval_percentage = 0.1

        self._model = RecModel()

        self._train_tensors = None
        self._serve_tensors = None
        self._train_session = None
        self._serve_session = None

        self._train_saver = None

        self._train_writer = None
        self._serve_writer = None

        self._eval_manager = EvalManager()
        self._eval_histories_sample = dict()
        self._min_eval_item_count = 100
        self._flag_updated = False

        self._save_model_dir = model_dir
        self._save_model_filename = 'model.ckpt'

        self._save_model_path = os.path.join(model_dir, self._save_model_filename)
        self._train_summary_path = os.path.join(model_dir, 'train')
        self._serve_summary_path = os.path.join(model_dir, 'serve')

        if os.path.exists(self._train_summary_path):
            shutil.rmtree(self._train_summary_path)

        if os.path.exists(self._serve_summary_path):
            shutil.rmtree(self._serve_summary_path)

        self.load_item_index()

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

    def load_item_index(self):
        self._mongo.load_item_index()

    def get_itemId(self, index):
        return self._mongo.get_itemId(index)

    def get_index(self, itemId):
        return self._mongo.get_index(itemId)

    def get_movie_info(self, movieId):
        return self._mongo.get_item_info(movieId)

    def _train_batch(self):

        low_pos = int(self._mongo.total_users * self._eval_percentage)

        while True:
            input_npy = np.zeros(self._batch_size,
                                 dtype=[('seq_item_id', (np.int32, self.max_seq_len)),
                                        ('seq_len', np.int32),
                                        ('label', np.int32)])

            histories_sample = list()
            while True:
                index = np.random.randint(low=low_pos, high=self._mongo.total_users - 1)
                history = self._mongo.get_item_list(index)

                if history is None:
                    continue

                if len(history) > 1:
                    histories_sample.append(history)

                if len(histories_sample) == self._batch_size:
                    break

            for ind, history in enumerate(histories_sample):
                predict_pos = np.random.randint(low=1, high=len(history))
                train_items = history[max(0, predict_pos - self._max_seq_len): predict_pos]
                train_items = [self._mongo.get_index(itemId) for itemId in train_items]

                pad_train_items = np.zeros(self.max_seq_len, np.int32)
                pad_train_items[:len(train_items)] = train_items
                predict_index = self._mongo.get_index(history[predict_pos])
                input_npy[ind] = (pad_train_items, len(train_items), predict_index)

            yield input_npy

    def _eval_batch(self):

        if len(self._eval_histories_sample) == 0:
            low_pos = max(self._min_eval_item_count, int(self._mongo.total_users * self._eval_percentage))

            index_list = np.arange(start=0, stop=low_pos, step=1)
            for ind in index_list:
                history = self._mongo.get_item_list(ind)
                if len(history) > 1:
                    history = [self._mongo.get_index(itemId) for itemId in history]
                    self._eval_histories_sample[ind] = history

        predict_pos = -1  # last position

        while True:
            for history in self._eval_histories_sample.values():
                input_npy = np.zeros(1, dtype=[('seq_item_id', (np.int32, self.max_seq_len)),
                                               ('seq_len', np.int32)])

                input_items = history[-self.max_seq_len - 1:predict_pos]
                pad_input_items = np.zeros(self.max_seq_len, np.int32)
                pad_input_items[:len(input_items)] = input_items
                input_npy[0] = (pad_input_items, len(input_items))
                yield history[predict_pos], input_npy
            yield None, None

    def get_train_sampler(self):
        return Sampler(generate_batch=self._train_batch, num_process=2)

    def get_eval_sampler(self):
        s = Sampler(generate_batch=self._eval_batch, num_process=1)
        return s

    def build_train_model(self):

        self._train_tensors = self._model.build_train_model(batch_size=self._batch_size,
                                                            embedding_size=self._embedding_size,
                                                            dim_item_embed=self.dim_item_embed,
                                                            total_items=self._mongo.total_items,
                                                            max_seq_len=self.max_seq_len)

        with self._model.get_train_graph().as_default():
            self._train_session = tf.Session(graph=self._model.get_train_graph())
            self._train_session.run(tf.global_variables_initializer())
            self._train_writer = tf.summary.FileWriter(self._train_summary_path, self._model.get_train_graph())

            self._train_saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
            self.restore(restore_train=True)

    def build_serve_model(self):

        self._serve_tensors = self._model.build_serve_model(embedding_size=self._embedding_size,
                                                            dim_item_embed=self.dim_item_embed,
                                                            total_items=self._mongo.total_items,
                                                            max_seq_len=self.max_seq_len)

        with self._model.get_serve_graph().as_default():
            self._serve_session = tf.Session(graph=self._model.get_serve_graph())
            self._serve_session.run(tf.global_variables_initializer())
            self._serve_writer = tf.summary.FileWriter(self._serve_summary_path, self._model.get_serve_graph())

            self.restore(restore_serve=True)

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

            self._flag_updated = True

            self._train_writer.add_summary(summary_, step)
            return loss_

    def serve(self, input):

        if self._flag_updated:
            self._save_and_load_for_serve()
            self._flag_updated = False

        with self._model.get_serve_graph().as_default():
            logits = self._serve_tensors['logits']
            user_embedding = self._serve_tensors['user_embedding']

            feed_dict = {
                self._serve_tensors['seq_item_id']: input['seq_item_id'],
                self._serve_tensors['seq_len']: input['seq_len']
            }

            debug_opt = self._serve_tensors['debug']

            debug_out_, user_embedding_, logits_ = self._serve_session.run([debug_opt, user_embedding, logits], feed_dict=feed_dict)
            # print('seq_item_id: ', input['seq_item_id'])
            # print('debug: ', debug_out_[:10])
            return logits_

    def evaluate(self, eval_sampler, step):
        metric_results = defaultdict(list)

        completed_user_count = 0
        pos_item, input = eval_sampler.next_batch()
        while input is not None:
            scores = np.squeeze(self.serve(input))
            result, rank_above = self._eval_manager.full_eval(pos_sample=pos_item, predictions=scores)
            completed_user_count += 1

            for key in result:
                metric_results[key].append(result[key])

            metric_results['rank_above'].append(rank_above)

            pos_item, input = eval_sampler.next_batch()

        return metric_results

    def save(self):
        with self._model.get_train_graph().as_default():
            self._train_saver.save(self._train_session, self._save_model_path)

    def restore(self, restore_train=False, restore_serve=False):
        if not os.path.exists(self._save_model_path + '.index'):
            return

        if restore_train:
            with self._model.get_train_graph().as_default():
                self._restore_only_variable(self._train_session, self._save_model_path)
        if restore_serve:
            with self._model.get_serve_graph().as_default():
                self._restore_only_variable(self._serve_session, self._save_model_path)

    def _restore_only_variable(self, session, save_model_path):

        reader = tf.train.NewCheckpointReader(save_model_path)
        saved_shapes = reader.get_variable_to_shape_map()

        restore_vars = []
        for var in tf.global_variables():
            var_name = var.name.split(':')[0]
            if var_name in saved_shapes and len(var.shape) > 0:
                if var.get_shape().as_list() == saved_shapes[var_name]:
                    restore_vars.append(var)

        saver = tf.train.Saver(restore_vars)
        saver.restore(session, save_model_path)

    def _save_and_load_for_serve(self):
        self.save()
        self.restore(restore_train=True, restore_serve=True)

    def add_evaluator(self, evaluator):
        self._eval_manager.add_evaluator(evaluator=evaluator)
