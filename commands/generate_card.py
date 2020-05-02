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

from mystique.utils.predict_card import PredictCard
import os
from PIL import Image
import sys
import json


def main (image_path=None, frozen_graph_path=None, labels_path=None):

    """
    Command runs the predict card function

    @param image_path: input image path
    @param frozen_graph_path: path to frozen graph
    @param labels_path: path to labels mapping
    """
    image=Image.open(image_path)
    card_json = PredictCard().main(image=image)
    print (json.dumps(json.loads(card_json).get('card_json'), indent=2))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Predict the Card Json')
    parser.add_argument('--image_path', required=True, help='Enter Image path')
    parser.add_argument('--frozen_graph_path', required=True,
                        help='Enter frozen graph path')
    parser.add_argument('--labels_path', required=True,
                        help='Enter graph labels path')
    args = parser.parse_args()
    main(image_path=args.image_path,
         frozen_graph_path=args.frozen_graph_path, labels_path=args.labels_path)
