import numpy as np

class EvalManager(object):

    def __init__(self):
        self._evaluators = []

    def add_evaluator(self, evaluator):
        self._evaluators.append(evaluator)

    def full_eval(self, pos_sample, predictions):
        results = dict()
        rank_above, negative_num = self._full_rank(pos_sample, predictions)
        for evaluator in self._evaluators:
            results[evaluator.name] = evaluator.compute(rank_above=rank_above, negative_num=negative_num)

        return results

    def _full_rank(self, pos_sample, predictions):
        rank_above = 0
        pos_prediction = predictions[pos_sample]

        for ind in range(len(predictions)):
                if pos_prediction < predictions[ind]:
                    rank_above += 1

        return rank_above, len(predictions)