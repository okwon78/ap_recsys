from flask import Flask, jsonify, request

from demo_mongo_client import DemoMongoClient


def get_api_server():
    app = Flask(__name__)
    client = DemoMongoClient(host='13.209.6.203', db_name='recsys')

    version = 'v1.0'

    info = {
        'name': 'ap recsys api',
        'version': version
    }

    @app.route("/")
    @app.route("/home")
    def home():
        return jsonify({'server info': info})

    @app.route('/recsys/api/test/db', methods=['GET'])
    def get_personal_recommendation():
        content = request.json
        userId = content['userId']

        watch_history = client.get_watch_history(userId)

        response = {
            'userId': userId,
            'watch_history': info
        }

        return jsonify(response)

    return app


def serve():
    api_server = get_api_server()
    api_server.run(debug=True)
