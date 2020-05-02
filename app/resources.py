import base64
import io
import json
import logging
import os
import sys
from datetime import datetime as dt

from PIL import Image
from flask import request
from flask_restplus import Resource

from mystique.utils.predict_card import PredictCard


logger = logging.getLogger('mysitque')

cur_dir = os.path.dirname(__file__)
input_image_collection = os.path.join(cur_dir, 'input_image_collection')
model_path = os.path.join(os.path.dirname(__file__), "../model/frozen_inference_graph.pb")
label_path = os.path.join(os.path.dirname(__file__), "../mystique/training/object-detection.pbtxt")


class PredictJson(Resource):

    def post(self):

        imgdata = base64.b64decode(request.json.get('image', ''))
        image = Image.open(io.BytesIO(imgdata))

        return_json = PredictCard().main(image=image)
        return json.loads(return_json)
