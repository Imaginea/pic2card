from flask_restplus import Resource
from flask import request
import os
from datetime import datetime as dt
import logging
import subprocess
import sys,io
import json
from PIL import Image
import base64

logger = logging.getLogger('mysitque')

cur_dir = os.path.dirname(__file__)
input_image_collection = os.path.join(cur_dir, 'input_image_collection')

class PredictJson(Resource):

    def post(self):
        
        imgdata = base64.b64decode(request.json.get('image',''))
        image=Image.open(io.BytesIO(imgdata))
        suffic_timestamp = dt.now().strftime('_%Y_%m_%d_%H-%M-%S')
        file_path = os.path.join(input_image_collection, "Input_Image"+suffic_timestamp+".png")
        image.save(file_path)
        logger.debug(f"saving file {file_path}")
        # Call Prediction Script ---> and send file location
        return_json=subprocess.check_output([sys.executable, "object_detection/utils/predict.py", "--image_path={}".format(file_path)])
        return json.loads(return_json.decode("utf-8"))
