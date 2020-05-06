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
import json
import numpy as np
from PIL import Image
from flask import current_app

from mystique.detect_objects import ObjectDetection
from mystique.predict_card import PredictCard
from mystique.initial_setups import set_graph_and_tensors

def main (image_path=None):

    """
    Command runs the predict card function

    @param image_path: input image path
    """
    image=Image.open(image_path)
    object_detection=ObjectDetection(*set_graph_and_tensors())
    card_json = PredictCard(object_detection).main(image=image)
    print (json.dumps(card_json.get('card_json'), indent=2))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Predict the Card Json')
    parser.add_argument('--image_path', required=True, help='Enter Image path')
    args = parser.parse_args()
    main(image_path=args.image_path)
