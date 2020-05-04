"""
Do inference using the tf-serve service.

We have loaded the saved model into the tf-serve
"""
import os
import io
import click
import base64
import json
import requests
import numpy as np
import cv2
from PIL import Image

from utils import timeit
from mystique.utils.initial_setups import set_graph_and_tensors
from mystique.detect_objects import ObjectDetection


@click.command()
@click.option("-t", "--tf_server", required=True,
              help="TF serving base URL")
@click.option("-i", "--image", required=True, help="Path to the image")
@click.option("-n", "--model_name", required=True,
              help="Model name to be used, as we can host multiple models in"
              " tf-serving")
def inference_graph(tf_server, image, model_name):
    with open(image, "rb") as f:
        bs64_img = base64.b64encode(f.read()).decode("utf8")

    tf_uri = os.path.join(tf_server, f"v1/models/{model_name}:predict")


    #print(bs64_img)
    #print(tf_uri)

    payloads = {
        "signature_name": "serving_default",
        "instances": [
            {"b64": bs64_img}
        ]
    }
    with timeit("tf-serving") as f:
        response = requests.post(tf_uri, json=payloads)
        pred_res = json.loads(response.content)

    # Frozen graph implementation.
    img = Image.open(open(image, "rb"))
    image_np = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    object_detection = ObjectDetection(*set_graph_and_tensors())
    with timeit("frozen-graph") as t:
        object_detection.get_objects(image=image_np)

    print(response)




if __name__ == "__main__":
    inference_graph()
