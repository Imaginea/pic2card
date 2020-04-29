from flask_restplus import Resource
from flask import request
import os
from datetime import datetime as dt
import logging
import subprocess
import sys
import io
import json
from PIL import Image
import base64
sys.path.append(os.getcwd())
from mystique.utils.predict_card import PredictCard


logger = logging.getLogger('mysitque')

cur_dir = os.path.dirname(__file__)
input_image_collection = os.path.join(cur_dir, 'input_image_collection')


class PredictJson(Resource):

    def post(self):

        imgdata = base64.b64decode(request.json.get('image', ''))
        image = Image.open(io.BytesIO(imgdata))
        suffic_timestamp = dt.now().strftime('_%Y_%m_%d_%H-%M-%S')
        file_path = os.path.join(
            input_image_collection, "Input_Image"+suffic_timestamp+".png")
        image.save(file_path)
        logger.debug(f"saving file {file_path}")
        return_json = PredictCard().main(image_path=file_path, forzen_graph_path='model/frozen_inference_graph.pb',
                                         labels_path='mystique/training/object-detection.pbtxt')
        if os.path.exists(file_path):
            os.remove(file_path)
        return json.loads(return_json)
