import os

import tensorflow as tf
import numpy as np
from termcolor import colored

from recsys.ap_recsys import ApRecsys

from recsys.evaluators.auc import AUC
from recsys.evaluators.precision import Precision
from recsys.evaluators.recall import Recall
from recsys.train.mongo_client import MongoConfig

print('tensorflow version: ', tf.__version__)
print('numpy version: ', np.__version__)


def train():
    mongo_config = MongoConfig(host='13.209.6.203',
                               username='romi',
                               password="Amore12345!",
                               dbname='recsys_apmall')

    model_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_save')

    ap_recsys = ApRecsys(model_save_path, mongo_config)

    train_sampler = ap_recsys.get_train_sampler()
    eval_sampler = ap_recsys.get_eval_sampler()

    ap_recsys.build_train_model()
    ap_recsys.build_serve_model()

    ap_recsys.add_evaluator(Precision(precision_at=[100]))
    ap_recsys.add_evaluator(Recall(recall_at=[50, 100, 150, 200, 250]))
    ap_recsys.add_evaluator(AUC())

    acc_loss = 0
    min_loss = None
    total_iter = 0
    while True:
        summary = tf.Summary()

        batch_data = train_sampler.next_batch()
        loss = ap_recsys.train(total_iter, batch_data)
        if min_loss is None:
            min_loss = loss
            acc_loss = loss

        if loss < min_loss:
            min_loss = loss

        acc_loss += loss
        total_iter += 1
        summary.value.add(tag='min_loss', simple_value=min_loss)

        # eval
        if total_iter % ap_recsys.eval_iter == 0:
            avg_loss = acc_loss / ap_recsys.eval_iter
            print(colored(f'[{total_iter}] avg_loss: {avg_loss}', 'blue'))
            summary.value.add(tag='avg_loss', simple_value=avg_loss)
            acc_loss = 0

            eval_results = ap_recsys.evaluate(eval_sampler=eval_sampler, step=total_iter)
            eval_results = dict(eval_results)

            result_stdout = ''
            for result in eval_results:
                average_result = np.mean(eval_results[result], axis=0)
                result_stdout += f'[{result}] {average_result} '
            print(colored(result_stdout, 'green'))

            summary.value.add(tag='AUC', simple_value=np.mean(eval_results['AUC']))
            summary.value.add(tag='rank_above', simple_value=np.mean(eval_results['rank_above']))

            # save item embedding
            # item_embeddings = ap_recsys.get_item_embeddings()

            # item_embedding_dict = dict()
            # for idx, item_embedding in item_embeddings:
            #     itemId = ap_recsys.get_itemId(idx)
            #     item_embedding_dict[itemId] = item_embedding


        ap_recsys.train_writer.add_summary(summary, total_iter)


if __name__ == '__main__':
    train()
