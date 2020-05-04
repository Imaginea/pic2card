""" resources for api """
import base64
import io
import json
import logging
import os
from PIL import Image
from flask import request
from flask_restplus import Resource
from flask import current_app

from mystique.utils.predict_card import PredictCard
from .utils import get_templates


logger = logging.getLogger('mysitque')

cur_dir = os.path.dirname(__file__)
input_image_collection = os.path.join(cur_dir, 'input_image_collection')
model_path = os.path.join(os.path.dirname(__file__), "../model/frozen_inference_graph.pb")
label_path = os.path.join(os.path.dirname(__file__), "../mystique/training/object-detection.pbtxt")


class PredictJson(Resource):
    """
    Handling Adaptive Card Predictions
    """

    def post(self):
        """
        predicts the adaptive card json for the posted image
        :return: adaptive card json
        """
        try:
            imgdata = base64.b64decode(request.json.get('image', ''))
            image = Image.open(io.BytesIO(imgdata))
            predict_card = PredictCard(current_app.od_model)
            return_json = predict_card.main(image=image)
            return return_json

        except Exception as ex:
            error_msg = f"Unhandled Error, failed to process the request: {ex}"
            logger.error(error_msg)

            response = {
                "error": {
                    "msg": error_msg,
                    "code": 1001
                },
                "card_json": None
            }
            return response



class GetCardTemplates(Resource):
    """
    Handling adaptive card template images
    """

    def get(self):
        """
        returns adaptive card templates images
        :return: adaptive card templates images in str format
        """
        templates = get_templates()
        return templates
