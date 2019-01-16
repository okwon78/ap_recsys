import sys
import pymongo
import csv

print(pymongo.__version__)


def get_movie_url():
    movie_url = dict()

    with open('./ml-20m/ml-youtube.csv') as f:
        # skip csv head
        _ = f.readline()

        while True:
            lines = f.readlines()

            if len(lines) <= 0:
                break

            for line in lines:
                cols = line.split(sep=',')
                movie_url[int(cols[1].strip())] = cols[0]

    return movie_url


def add_records(recsys_db):
    recsys_db.records.delete_many({})

    records = dict()

    with open('./ml-20m/tags.csv', 'r') as f:
        header = f.readline()

        for cols in csv.reader(f, delimiter=','):
            try:
                userId = int(cols[0].strip())
                movieId = int(cols[1].strip())
                timestamp = int(cols[3].strip())

                if userId in records:
                    record = records[userId]
                else:
                    record = {
                        'userId': userId,
                        'movies': []
                    }
                    records[userId] = record

                record['movies'].append(
                    {
                        'movieId': movieId,
                        'timestamp': timestamp
                    }
                )
            except:
                print("Unexpected error:", sys.exc_info()[0])

        records = list(records.values())

    recsys_db.records.insert_many(records)


def add_movies(recsys_db):
    movie_url = get_movie_url()
    recsys_db.movies.delete_many({})

    with open('./ml-20m/movies.csv', 'r') as f:

        header = f.readline()

        movies = list()
        for ind, cols in enumerate(csv.reader(f, delimiter=',')):
            movieId = int(cols[0].strip())
            title = cols[1]
            genres = cols[2].split(sep='|')
            genres = [elem.strip() for elem in genres]

            if movieId in movie_url:
                movie = {
                    'index': ind,
                    'movieId': movieId,
                    'title': title,
                    'genres': genres,
                    'url': movie_url[movieId]
                }
            else:
                movie = {
                    'index': ind,
                    'movieId': movieId,
                    'title': title,
                    'genres': genres
                }

            movies.append(movie)

    recsys_db.movies.insert_many(movies)


def add_ratings(recsys_db, threshold):
    recsys_db.ratings.delete_many({})

    with open('./ml-20m/ratings.csv', 'r') as f:
        header = f.readline()

        records = dict()

        for cols in csv.reader(f, delimiter=','):
            movieId = int(cols[1].strip())
            rating = float(cols[2].strip())
            timestamp = int(cols[3].strip())

            if userId in records:
                record = records[userId]
            else:
                print('userId: ', userId)

                record = {
                    'userId': userId,
                    'movie_pos': [],
                    'movie_neg': []
                }

                records[userId] = record

            elem = {
                'movieId': movieId,
                'timestamp': timestamp
            }

            if rating > threshold:
                record['movie_pos'].append(elem)
            else:
                record['movie_neg'].append(elem)

    records = list(records.values())
    recsys_db.ratings.insert_many(records)


def main():
    client = pymongo.MongoClient(host='13.209.6.203',
                                 port=27017,
                                 username='romi',
                                 password="Amore12345!",
                                 authSource='admin',
                                 authMechanism='SCRAM-SHA-256')

    db = client.recsys
    add_movies(db)
    # add_records(db)
    # add_ratings(db, 3)


if __name__ is '__main__':
    main()
    print("the end")
