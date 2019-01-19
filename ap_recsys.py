import os
import shutil
from collections import defaultdict

import tensorflow as tf
import numpy as np

from recsys.eval_manager import EvalManager
from recsys.evaluators.auc import AUC
from recsys.evaluators.precision import Precision
from recsys.evaluators.recall import Recall
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
        self._train_summary_path = './train'
        self._serve_summary_path = './serve'
        self._train_tensors = None
        self._serve_tensors = None
        self._train_session = None
        self._serve_session = None
        self._train_saver = None
        self._serve_saver = None
        self._train_writer = None
        self._serve_writer = None
        self._eval_manager = EvalManager()
        self._eval_watch_histories_sample = dict()
        self._min_eval_item_count = 10

        self._flag_updated = False

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

        low_pos = int(self._mongo.total_raw_data * self._train_percentage)
        while True:
            input_npy = np.zeros(self._batch_size,
                                 dtype=[('seq_item_id', (np.int32, self.max_seq_len)),
                                        ('seq_len', np.int32),
                                        ('label', np.int32)])

            watch_histories_sample = list()
            while True:
                index = np.random.randint(low=low_pos, high=self._mongo.total_raw_data - 1)
                watch_history = self._mongo.get_watch_list(index)
                if len(watch_history) > 0:
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

    def _eval_batch(self):

        if len(self._eval_watch_histories_sample) == 0:
            low_pos = min(self._min_eval_item_count, int(self._mongo.total_raw_data * self._train_percentage))
            index_list = np.arange(start=0, stop=low_pos, step=1)
            for ind in index_list:
                watch_history = self._mongo.get_watch_list(ind)
                if len(watch_history) > 1:
                    index_watch_history = [self._mongo.get_index_from_movieId(movieId) for movieId in watch_history]
                    self._eval_watch_histories_sample[ind] = index_watch_history

        predict_pos = -1  # last position

        while True:
            for index_watch_history in self._eval_watch_histories_sample.values():
                input_npy = np.zeros(1, dtype=[('seq_item_id', (np.int32, self.max_seq_len)),
                                               ('seq_len', np.int32)])

                train_items = index_watch_history[-self.max_seq_len - 1:predict_pos]
                pad_train_items = np.zeros(self.max_seq_len, np.int32)
                pad_train_items[:len(train_items)] = train_items
                input_npy[0] = (pad_train_items, len(train_items))
                yield index_watch_history[predict_pos], input_npy
            yield None, None

    def get_train_sampler(self):
        return Sampler(generate_batch=self._train_batch, num_process=1)

    def get_eval_sampler(self):
        return Sampler(generate_batch=self._eval_batch, num_process=1)

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
            self.restore(restore_train=True)
            self._train_writer = tf.summary.FileWriter(self._train_summary_path, self._model.get_train_graph())

    def build_serve_model(self):

        self._serve_tensors = self._model.build_serve_model(embedding_size=self._embedding_size,
                                                            dim_item_embed=self.dim_item_embed,
                                                            total_items=self._mongo.total_movies,
                                                            max_seq_len=self.max_seq_len)

        with self._model.get_serve_graph().as_default():
            self._serve_session = tf.Session()
            # self._serve_session.run(tf.global_variables_initializer())
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

            self._flag_updated = True

            self._train_writer.add_summary(summary_, step)
            return loss_
    def serve(self, input, step=None):

        if self._flag_updated:
            self._save_and_load_for_serve(step)
            self._flag_updated = False

        with self._model.get_serve_graph().as_default():
            logits = self._serve_tensors['logits']

            feed_dict = {
                self._serve_tensors['seq_item_id']: input['seq_item_id'],
                self._serve_tensors['seq_len']: input['seq_len']
            }

            logits_ = self._serve_session.run([logits], feed_dict=feed_dict)
            return logits_

    def evaluate(self, eval_sampler, step):
        metric_results = defaultdict(list)

        completed_user_count = 0
        pos_item, input = eval_sampler.next_batch()
        while input is not None:
            scores = np.squeeze(self.serve(input, step))
            result =  self._eval_manager.full_eval(pos_sample=pos_item, predictions=scores)
            completed_user_count += 1

            for key in result:
                metric_results[key].append(result[key])

            pos_item, input = eval_sampler.next_batch()

        return metric_results

    def save(self, step):
        with self._model.get_train_graph().as_default():
            self._train_saver.save(self._train_session,
                                   os.path.join(self._save_model_dir, 'model.ckpt'),
                                   global_step=step)

    def restore(self, restore_train=False, restore_serve=False):
        if not os.path.exists(self._save_model_dir):
            return

        if restore_train:
            with self._model.get_train_graph().as_default():
                self._train_saver.restore(self._train_session, os.path.join(self._save_model_dir, 'model.ckpt'))
        if restore_serve:
            with self._model.get_serve_graph().as_default():
                self.restore_only_variable(self._serve_session, os.path.join(self._save_model_dir, 'model.ckpt'))
                # self._train_saver.restore(self._serve_session, os.path.join(self._save_model_dir, 'model.ckpt'))

    def restore_only_variable(self, session, save_file):

        reader = tf.train.NewCheckpointReader(save_file)
        saved_shapes = reader.get_variable_to_shape_map()

        restore_vars = []
        for var in tf.global_variables():
            var_name = var.name.split(':')[0]
            if var_name in saved_shapes and len(var.shape) > 0:
                if var.get_shape().as_list() == saved_shapes[var_name]:
                    restore_vars.append(var)

        saver = tf.train.Saver(restore_vars)
        saver.restore(session, save_file)


    def _save_and_load_for_serve(self, step):
        self.save(step)
        self.restore(restore_serve=True)


    def add_evaluator(self, evaluator):
        self._eval_manager.add_evaluator(evaluator=evaluator)

def main():
    ap_model = ApRecsys(host='13.209.6.203',
                        username='romi',
                        password="Amore12345!",
                        db_name='recsys')

    # ap_model.make_raw_data()
    ap_model.make_movie_index()

    train_sampler = ap_model.get_train_sampler()
    eval_sampler = ap_model.get_eval_sampler()

    ap_model.build_train_model()
    ap_model.build_serve_model()

    ap_model.add_evaluator(Precision(precision_at=[5]))
    ap_model.add_evaluator(Recall(recall_at=[5]))
    ap_model.add_evaluator(AUC())

    acc_loss = 0
    min_loss = None
    total_iter = 0
    while True:
        summary = tf.Summary()

        batch_data = train_sampler.next_batch()
        loss = ap_model.train(total_iter, batch_data)
        if min_loss is None:
            min_loss = loss
            acc_loss = loss

        if loss < min_loss:
            min_loss = loss

        acc_loss += loss
        total_iter += 1
        summary.value.add(tag='min_loss', simple_value=min_loss)

        print(f'[{total_iter}] loss: {loss}')

        # eval
        if total_iter % ap_model.eval_iter == 0 :
            avg_loss = acc_loss / ap_model.eval_iter
            print(f'[{total_iter}] avg_loss: {avg_loss}')
            summary.value.add(tag='avg_loss', simple_value=avg_loss)
            acc_loss = 0

            eval_results = ap_model.evaluate(eval_sampler=eval_sampler, step=total_iter)

            for result in eval_results:
                print(f'[{result}] {eval_results[result]}')

            # for name, value in eval_results:
            #     summary.value.add(tag=name, simple_value=value)

        ap_model.train_writer.add_summary(summary, total_iter)


if __name__ == '__main__':
    main()
