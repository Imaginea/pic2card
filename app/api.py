"""Flask service to predict the adaptove card json from the card design"""

from flask import Flask, make_response, jsonify
from flask import request
from flask_restplus import Api
import logging
from resources import PredictJson
from flask_cors import CORS
from logging.handlers import RotatingFileHandler

logger = logging.getLogger("mysitque")
logger.setLevel(logging.DEBUG)

file_handler = RotatingFileHandler(
    'mystique_app.log', maxBytes=1024 * 1024 * 100, backupCount=20)
formatter = logging.Formatter(
    "%(asctime)s - [%(filename)s:%(lineno)s - %(funcName)20s() ] - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)


app = Flask(__name__)
api = Api(app, title="Mystique", version="1.0", default="Jobs", default_label="",
          description="Mysique App For Adaptive card Json Prediction from UI Design",)
CORS(app)


api.add_resource(PredictJson, '/predict_json',  methods=['POST'])


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5050, debug=True)

