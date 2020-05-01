"""Command to predict the adaptive card json"""
import argparse
import sys
import os

sys.path.append (os.getcwd ())
from mystique.utils.predict_card import PredictCard
import os
import sys
import json

sys.path.append (os.getcwd () + "/mystique/utils")


def main (image_path=None, frozen_graph_path=None, labels_path=None):

    """[Command runs the predict card function]

    Keyword Arguments:
        image_path {[string]} -- [input image] (default: {None})
        frozen_graph_path {[string]} -- [trained model path] (default: {None})
        labels_path {[string]} -- [labels path] (default: {None})
    """
    card_json = PredictCard ().main (image_path=image_path,
                                     frozen_graph_path=frozen_graph_path, labels_path=labels_path)
    print (json.dumps (json.loads (card_json).get ('card_json'), indent=2))


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
