from flask_restplus import Resource
from flask import request
import os
from datetime import datetime as dt
import logging

logger = logging.getLogger('mysitque')

cur_dir = os.path.dirname(__file__)
input_image_collection = os.path.join(cur_dir, 'input_image_collection')

class PredictJson(Resource):

    def post(self):
        input_image = request.files['image']
        suffic_timestamp = dt.now().strftime('_%Y_%m_%d_%H-%M-%S') + '.png'
        file_path = os.path.join(input_image_collection, input_image.filename.replace(".png", suffic_timestamp))
        input_image.save(file_path)
        logger.debug(f"saving file {file_path}")
        # Call Prediction Script ---> and send file location
        # predict(file_path)