import os
import multiprocessing
from multiprocessing import Queue, Process


class _Process(Process):

    def __init__(self, queue, generate_batch):
        self._queue = queue
        self._generate_batch = generate_batch
        super(_Process, self).__init__()

    def run(self):
        for input in self._generate_batch():
            print(f'pid: {os.getpid()} = ', f'recved: {len(input)}')
            self._queue.put(input, block=True)
            print('added data to queue')


class Sampler(object):

    def __init__(self, generate_batch, num_process=None):
        self._queue = None
        self._runner_list = []
        self._start = False

        if num_process is None:
            self._num_process = max(1, int(multiprocessing.cpu_count()/2))
        else:
            self._num_process = num_process

        self._generate_batch = generate_batch

    def next_batch(self):
        if not self._start:
            self.reset()
        print('read from queue')
        return self._queue.get(block=True)

    def next_batch_debug(self):
        input = next(self._generate_batch())
        return input


    def reset(self):

        while len(self._runner_list) > 0:
            runner = self._runner_list.pop()
            runner.terminate()
            del runner

        if self._queue is not None:
            del self._queue

        self._queue = Queue(maxsize=self._num_process)

        for ind in range(self._num_process):
            runner = _Process(self._queue, self._generate_batch)
            runner.deamon = True
            self._runner_list.append(runner)
            runner.start()

        self._start = True
