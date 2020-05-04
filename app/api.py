"""Flask service to predict the adaptive card json from the card design"""
import logging
from logging.handlers import RotatingFileHandler
from flask import Flask
from flask_cors import CORS
from flask_restplus import Api
from . import resources as res

from mystique.utils.initial_setups import set_graph_and_tensors
from mystique.detect_objects import ObjectDetection

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

# Load the model into flask cache
app.od_model = ObjectDetection(*set_graph_and_tensors())

api = Api(app, title="Mystique", version="1.0", default="Jobs", default_label="",
          description="Mysique App For Adaptive card Json Prediction from UI Design")

api.add_resource(res.PredictJson, '/predict_json',  methods=['POST'])
api.add_resource(res.GetCardTemplates, '/get_card_templates',  methods=['GET'])

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5050, debug=False)

