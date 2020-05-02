"""Flask service to predict the adaptive card json from the card design"""

import logging
from logging.handlers import RotatingFileHandler

from flask import Flask
from flask_cors import CORS
from flask_restplus import Api
from resources import PredictJson
from mystique.utils.initial_setups import set_graph_and_tensors

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
CORS(app)
app.config['DETECTION_GRAPTH'], app.config['CATEGORY_INDEX'], app.config['TENSOR_DICT'] = set_graph_and_tensors()
api = Api(app, title="Mystique", version="1.0", default="Jobs", default_label="",
          description="Mysique App For Adaptive card Json Prediction from UI Design",)
CORS(app)

api.add_resource(PredictJson, '/predict_json',  methods=['POST'])


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5050, debug=False)

