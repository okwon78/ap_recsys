import os
import random

import pymongo

from recsys.samplers.sampler import Sampler


class Mongo(object):

    def __init__(self, username, password, host, db_name, port=27017, authSource='admin', authMechanism='SCRAM-SHA-256'):

        self._host = host
        self._port = port
        self._pid = os.getpid()
        self._client = pymongo.MongoClient(host=self._host,
                                           port=self._port,
                                           username='romi',
                                           password="Amore12345!",
                                           authSource='admin',
                                           authMechanism='SCRAM-SHA-256')

        self._db_name = db_name
        self._db = self._client.__getattr__(db_name)
        self._total_movies = self._db.movies.count()
        self._total_users = self._db.ratings.count()
        self._total_raw_data = self._db.raw_data.count()

        if self._total_movies < 0 or self._total_users < 0:
            raise ValueError("Invalid database")

    @property
    def db(self):

        pid = os.getpid()

        if pid == self._pid:
            return self._db
        else:
            self._client = pymongo.MongoClient(host=self._host,
                                               port=self._port,
                                               username='romi',
                                               password="Amore12345!",
                                               authSource='admin',
                                               authMechanism='SCRAM-SHA-256')

            self._db = self._client.__getattr__(self._db_name)
            return self._db




    def make_raw_data(self, batch_size=1000):
        ratings = self.db.ratings

        start = 0
        end = ratings.count()
        current = start

        projection = {
            '_id': 0,
            'userId': 1,
            'movie_pos': 1
        }

        self.db.raw_data.delete_many({})

        index_list = list(range(end))
        random.shuffle(index_list)

        index = 0
        while current < end:
            raw_data = []
            for doc in ratings.find({}, projection).skip(current).limit(batch_size):

                if len(doc['movie_pos']) < 2:
                    continue

                doc['movie_pos'].sort(key=lambda item: int(item['timestamp']))
                ordered_watch_list = [item['movieId'] for item in doc['movie_pos']]

                input = {
                    'index': int(index_list[index]),
                    'ordered_watch_list': ordered_watch_list,
                }

                index += 1
                raw_data.append(input)
            current += batch_size
            self.db.raw_data.insert_many(raw_data)

        self._total_raw_data = self.db.raw_data.count()

    def get_watch_list(self, index):
        if self._total_raw_data == 0:
            return None

        try:
            doc = next(self.db.raw_data.find({'index': int(index)}))
            return doc['ordered_watch_list']
        except Exception as e:
            print(e)


    @property
    def total_movies(self):
        return self._total_movies

    @property
    def total_users(self):
        return self._total_users

    @property
    def total_raw_data(self):
        return self._total_raw_data


