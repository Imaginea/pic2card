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

sys.path.append(os.getcwd())
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
        suffic_timestamp = dt.now().strftime('_%Y_%m_%d_%H-%M-%S')
        file_path = os.path.join(
            input_image_collection, "Input_Image"+suffic_timestamp+".png")

        if not os.path.exists(input_image_collection):
            os.mkdir(input_image_collection)

        image.save(file_path)
        logger.debug(f"saving file {file_path}")
        return_json = PredictCard(image_path=file_path, frozen_graph_path=model_path,
                                  labels_path=label_path).main()
        if os.path.exists(file_path):
            os.remove(file_path)
        return json.loads(return_json)
