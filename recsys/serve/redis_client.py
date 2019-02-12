from types import *

import redis


class RedisConnectionConfig(object):

    def __init__(self, host='52.79.250.155', port=6379, db=0, password='Amore12345!'):
        self._host = host
        self._port = port
        self._db = db
        self._password = password


class RedisClient(object):

    def __init__(self, redis_connection_config,
                 max_seq_len=5,
                 expire_time_seconds=20):

        self._redis_db = redis.Redis(host=redis_connection_config._host,
                                     port=redis_connection_config._port,
                                     db=redis_connection_config._db,
                                     password=redis_connection_config._password)
        self._max_seq_len = max_seq_len
        self._expire_time_seconds = expire_time_seconds

    def __setitem__(self, key, value):
        self._lpush(key, value)

    def __getitem__(self, key):
        return self._lrange(key)

    def _lpush(self, key, value):
        if isinstance(value, list):
            len = self._redis_db.lpush(key, *value)
        else:
            len = self._redis_db.lpush(key, value)

        self._sustain_seq_len(len, self._max_seq_len)
        self._set_expire(key)

    def _sustain_seq_len(self, key, current_len):
        if self._max_seq_len is None:
            return

        difference = current_len - self._max_seq_len
        if difference > 0:
            self._redis_db.ltrim(key, 0, self._max_seq_len - 1)

    def _lrange(self, key):
        self._set_expire(key)
        return self._redis_db.lrange(key, 0, self._max_seq_len - 1)

    def _set_expire(self, key):
        if self._expire_time_seconds is not None:
            self._redis_db.expire(key, self._expire_time_seconds)

    def flushall(self):
        self._redis_db.flushall()


def test():
    config = RedisConnectionConfig()
    client = RedisClient(redis_connection_config=config, expire_time_seconds=None)
    print(client['userId:1'])

if __name__ == '__main__':
    test()

