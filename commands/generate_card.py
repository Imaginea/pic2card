"""
Command to predict the adaptive card json

Usage :
python generate_card.py --image_path=/path/to/input/image
                        --frozen_graph_path=model/frozen_inference_graph.pb
                        --labels_path=mystique/training/object-detection.pbtxt
"""
import argparse
import sys
import os

import numpy as np
from mystique.detect_objects import ObjectDetection
from mystique.predict_card import PredictCard
from PIL import Image
from mystique.initial_setups import set_graph_and_tensors
import sys
import json
from flask import current_app
import cv2

def main (image_path=None):

    """
    Command runs the predict card function

    @param image_path: input image path
    """
    image=Image.open(image_path)
    image=image.convert('RGB')
    image_np = np.asarray(image)
    image_np=cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    object_detection=ObjectDetection(*set_graph_and_tensors())
    # Extract the design objects from faster rcnn model
    output_dict, category_index = object_detection.get_objects(image=image_np)
    card_json = PredictCard().generate_card(output_dict, image, image_np)
    print (json.dumps(card_json.get('card_json'), indent=2))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Predict the Card Json')
    parser.add_argument('--image_path', required=True, help='Enter Image path')
    args = parser.parse_args()
    main(image_path=args.image_path)