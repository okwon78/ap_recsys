import os
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from recsys.ap_recsys import ApRecsys
from recsys.serve.redis_client import RedisConnectionConfig, RedisClient
from recsys.train.mongo_client import MongoConfig


def get_api_server(ap_model, redis_client, top_k):
    app = Flask(__name__, static_url_path='/static')

    version = 'v1.0'

    info = {
        'name': 'ap recsys api',
        'version': version
    }

    @app.route("/info")
    def info():
        return jsonify({'server info': info})

    @app.route('/')
    def root():
        return app.send_static_file('index.html')

    @app.route('/static/<path:path>')
    def send_static(path):
        return send_from_directory('static', path)

    @app.route('/recsys/api/', methods=['POST'])
    def get_personal_recommendation():
        content = request.json
        userId = content['userId']
        key = f'userId:{userId}'
        input_seq = redis_client[key]

        if len(input_seq) == 0:
            response = {
                'message': 'user history does not exist'
            }

            return jsonify(response)

        input_seq = [ap_model.get_index_from_movieId(int(movieId)) for movieId in input_seq]
        input = np.zeros(1, dtype=[('seq_item_id', (np.int32, ap_model.max_seq_len)), ('seq_len', np.int32)])
        input[0] = (input_seq, len(input_seq))
        logit = np.squeeze(ap_model.serve(input))
        movie_index = np.argsort(logit)[::-1][:top_k]

        movieIds = []
        for index in movie_index:
            movieIds.append(ap_model.get_movieId_from_index(index))
        # print(logit[:10])
        # print(logit[movie_index])
        # print(movieIds)
        input_movies_info = ap_model.get_movie_info(input_seq)
        recommendation_movies_info = ap_model.get_movie_info(movieIds)

        response = {
            'userId': userId,
            'input_movies_info': input_movies_info,
            'recommendation_movies_info': recommendation_movies_info
        }

        # print('i: ', response['input_movies_info'])
        # print('r: ', response['recommendation_movies_info'])

        return jsonify(response)

    return app


def serve():
    mongo_config = MongoConfig(host='13.209.6.203',
                               username='romi',
                               password="Amore12345!",
                               dbname='recsys')

    model_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_save')

    ap_model = ApRecsys(model_save_path, mongo_config)
    ap_model.build_serve_model()
    ap_model.restore(restore_serve=True)
    ap_model.make_movie_index()

    redis_config = RedisConnectionConfig()
    redis_client = RedisClient(redis_connection_config=redis_config,
                               max_seq_len=ap_model.max_seq_len,
                               expire_time_seconds=None)

    # WAS
    api_server = get_api_server(ap_model, redis_client, top_k=20)
    api_server.run(host='0.0.0.0', debug=True)


if __name__ == '__main__':
    serve()