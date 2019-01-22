import os
import random

import pymongo

from recsys.samplers.sampler import Sampler


class MongoConfig(object):

    def __init__(self, host, username, password, dbname):
        self._host = host
        self._username = username
        self._password = password
        self._dbname = dbname

    @property
    def host(self):
        return self._host

    @property
    def username(self):
        return self._username

    @property
    def password(self):
        return self._password

    @property
    def dbname(self):
        return self._dbname


class MongoClient(object):

    def __init__(self, username, password, host, db_name, port=27017, authSource='admin',
                 authMechanism='SCRAM-SHA-256'):

        self._host = host
        self._port = port
        self._pid = os.getpid()
        self._client = pymongo.MongoClient(host=self._host,
                                           port=self._port,
                                           username='romi',
                                           password="Amore12345!",
                                           authSource='admin',
                                           authMechanism='SCRAM-SHA-256')

        print('Successfully connected to mongodb')

        self._db_name = db_name
        self._db = self._client.__getattr__(db_name)
        self._total_movies = self._db.movies.count()
        self._movies_to_index = dict()
        self._index_to_movies = dict()
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
            self._pid = pid

            self._client = pymongo.MongoClient(host=self._host,
                                               port=self._port,
                                               username='romi',
                                               password="Amore12345!",
                                               authSource='admin',
                                               authMechanism='SCRAM-SHA-256')

            self._db = self._client.__getattr__(self._db_name)

            print('New process connected to mongodb')
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

    def make_movie_index(self):
        projection = {
            '_id': 0,
            'index': 1,
            'movieId': 1
        }

        for doc in self.db.movies.find({}, projection):
            self._movies_to_index[doc['movieId']] = doc['index']
            self._index_to_movies[doc['index']] = doc['movieId']

    def get_watch_list(self, index):
        if self._total_raw_data == 0:
            return None

        try:
            doc = next(self.db.raw_data.find({'index': int(index)}))
            return doc['ordered_watch_list']
        except StopIteration:
            return []
        except Exception as e:
            print(e)

    def get_index_from_movieId(self, movieId):
        return self._movies_to_index[movieId]

    def get_movieId_from_index(self, index):
        return self._index_to_movies[index]

    @property
    def total_movies(self):
        return self._total_movies

    @property
    def total_users(self):
        return self._total_users

    @property
    def total_raw_data(self):
        return self._total_raw_data
