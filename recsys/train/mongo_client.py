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
        self._total_items = self._db.items.count()
        self._total_users = self._db.users.count()

        if self._total_items < 0 or self._total_users < 0:
            raise ValueError("Invalid database")

        self._itemId_to_index = dict()
        self._index_to_itemId = dict()

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

    def load_item_index(self):
        projection = {
            '_id': 0,
            'item_index': 1,
            'itemName': 1,
            'itemId': 1
        }
        tsv_file_path = 'item_label.tsv'

        with open(tsv_file_path, 'w') as f:
            f.write("Index\tLabel\n")
            for doc in self.db.items.find({}, projection):
                self._itemId_to_index[doc['itemId']] = doc['item_index']
                self._index_to_itemId[doc['item_index']] = doc['itemId']
                f.write("{}\t{}\n".format(doc['item_index'], doc['itemName']))


    def get_item_info(self, itemIds):

        projection = {
            '_id': 0,
            'itemId': 1,
            'itemName': 1,
            'item_index': 1,
            'url': 1
        }

        item_infos = []

        for doc in self.db.items.find({'itemId': {'$in': itemIds}}, projection):
            item_infos.append(doc)

        return item_infos

    def get_item_list(self, user_index):

        try:
            doc = next(self.db.users.find({'user_index': int(user_index)}))
            return doc['sorted_items']
        except StopIteration:
            return []
        except Exception as e:
            print(e)

    def get_index(self, itemId):
        return self._itemId_to_index[itemId]

    def get_itemId(self, index):
        return self._index_to_itemId[index]

    @property
    def total_items(self):
        return self._total_items

    @property
    def total_users(self):
        return self._total_users