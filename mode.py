import os

import tensorflow as tf
import numpy as np
from termcolor import colored

from recsys.ap_model import ApModel, MongoConfig

from recsys.evaluators.auc import AUC
from recsys.evaluators.precision import Precision
from recsys.evaluators.recall import Recall

print('tensorflow version: ', tf.__version__)
print('numpy version: ', np.__version__)


def train():
    mongo_config = MongoConfig(host='13.209.6.203',
                               username='romi',
                               password="Amore12345!",
                               dbname='recsys')

    model_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_save')

    ap_model = ApModel(model_save_path, mongo_config)

    # ap_model.make_raw_data()
    ap_model.make_movie_index()

    train_sampler = ap_model.get_train_sampler()
    eval_sampler = ap_model.get_eval_sampler()

    ap_model.build_train_model()
    ap_model.build_serve_model()

    ap_model.add_evaluator(Precision(precision_at=[5]))
    ap_model.add_evaluator(Recall(recall_at=[100, 200, 300, 400, 500]))
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
        if total_iter % ap_model.eval_iter == 0:
            avg_loss = acc_loss / ap_model.eval_iter
            print(colored(f'[{total_iter}] avg_loss: {avg_loss}', 'blue'))
            summary.value.add(tag='avg_loss', simple_value=avg_loss)
            acc_loss = 0

            eval_results = ap_model.evaluate(eval_sampler=eval_sampler, step=total_iter)
            eval_results = dict(eval_results)

            for result in eval_results:
                average_result = np.mean(eval_results[result], axis=0)
                print(colored(f'[{result}] {average_result}', 'green'))

            summary.value.add(tag='AUC', simple_value=np.mean(eval_results['AUC']))
            summary.value.add(tag='rank_above', simple_value=np.mean(eval_results['rank_above']))

        ap_model.train_writer.add_summary(summary, total_iter)

def serve():
    pass
