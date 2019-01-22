import pymongo


class DemoMongoClient(object):

    def __init__(self, host, db_name, port=27017):

        self._host = host
        self._port = port
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
        return self._db

    def get_watch_history(self, userId):
        self._db.row_data.find({})

    def get_youtube_url(self, movieId):


        doc = next(self._db.movies.find({ 'mvoieId' : movieId}))

