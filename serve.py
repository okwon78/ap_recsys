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

    @app.route("/restore")
    def restore():
        ap_model.restore(restore_serve=True)
        return "restore"

    @app.route("/embedding")
    def embedding():
        embeddings = ap_model.get_item_embeddings()
        print('len: ', len(embeddings), 'embeddings size: ',len(embeddings[0]))
        response = {
            'embeddings': embeddings.tolist()
        }

        return jsonify(response)

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
        key = f'ap_mall_userId:{userId}'

        print(key)

        input_itemId_seq = redis_client[key]
        input_itemId_seq.reverse()
        input_itemId_seq = [itemId.decode("utf-8") for itemId in input_itemId_seq]

        if len(input_itemId_seq) == 0:
            response = {
                'message': 'user history does not exist'
            }

            return jsonify(response)

        input_index_seq = [ap_model.get_index(itemId) for itemId in input_itemId_seq]
        pad_input_items = np.zeros(ap_model.max_seq_len, np.int32)
        pad_input_items[:len(input_index_seq)] = input_index_seq
        input = np.zeros(1, dtype=[('seq_item_id', (np.int32, ap_model.max_seq_len)), ('seq_len', np.int32)])
        input[0] = (pad_input_items, len(input_index_seq))
        user_embedding, logits = ap_model.serve(input)

        user_embedding = np.squeeze(user_embedding)
        logit = np.squeeze(logits)

        # print('user_embedding: ', user_embedding)

        #top K
        item_index = np.argsort(logit)[::-1][:top_k]

        print('top K: ', item_index[:top_k])
        print('logit: ', logit[:top_k])

        recommendation_itemIds = []
        for index in item_index:
            recommendation_itemIds.append(ap_model.get_itemId(index))


        items_info = ap_model.get_item_info(input_itemId_seq)
        recommendation_items_info = ap_model.get_item_info(recommendation_itemIds)

        response = {
            'userId': userId,
            'history_items_info': items_info,
            'recommendation_items_info': recommendation_items_info
        }
        for items in recommendation_items_info:
            print(items["item_index"], items["itemName"])

        return jsonify(response)

    return app


def serve():
    mongo_config = MongoConfig(host='13.209.6.203',
                               username='romi',
                               password="Amore12345!",
                               dbname='recsys_apmall')

    model_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_save')

    ap_model = ApRecsys(model_save_path, mongo_config)
    ap_model.build_serve_model()
    ap_model.restore(restore_serve=True)

    redis_config = RedisConnectionConfig()
    redis_client = RedisClient(redis_connection_config=redis_config,
                               max_seq_len=ap_model.max_seq_len,
                               expire_time_seconds=None)

    # WAS
    api_server = get_api_server(ap_model, redis_client, top_k=20)
    api_server.run(host='0.0.0.0', debug=True)


if __name__ == '__main__':
    serve()