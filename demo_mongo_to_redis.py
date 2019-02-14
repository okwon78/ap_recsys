from recsys.serve.redis_client import RedisConnectionConfig, RedisClient
from recsys.train.mongo_client import MongoClient


def main():
    redis_config = RedisConnectionConfig()


    mongo_client = MongoClient(username='romi',
                               password="Amore12345!",
                               host='13.209.6.203',
                               db_name='recsys_apmall')

    redis_client = RedisClient(redis_config, expire_time_seconds=None)
    redis_client.flushall()

    start = 0
    end = mongo_client.db.users.count()
    current = start
    batch_size = 1000

    projection = {
        '_id': 0,
        'userId': 1,
        'user_index': 1,
        'sorted_items': 1
    }

    while current < end:
        for doc in mongo_client.db.users.find({}, projection).skip(current).limit(batch_size):

            if len(doc['sorted_items']) < 2:
                continue

            redis_client['ap_mall_userId:' + str(doc['user_index'])] = doc['sorted_items']

        current += batch_size


if __name__ is '__main__':
    main()
