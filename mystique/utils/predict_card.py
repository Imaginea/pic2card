"""Module to  get the predicted adaptive card json"""

import json
import os
import sys

import cv2
from PIL import Image

from mystique.arrange_card import CardArrange
from mystique.detect_objects import ObjectDetection
from mystique.extract_properties import ExtractProperties
from mystique.image_extraction import ImageExtraction
import numpy as np


class PredictCard:

    def __init__(self, od_model):
        """
        Find the card components using Object detection model
        """
        self.od_model = od_model

    def collect_objects(self, output_dict=None, pil_image=None):

        """
        Returns the design elements from the faster rcnn model with its
        properties mapped

        @param output_dict: output dict from the object detection

        @return: Collected json of the design objects
                 and list of detected object's coordinates
        """
        extract_properties=ExtractProperties()
        boxes = output_dict["detection_boxes"]
        scores = output_dict["detection_scores"]
        classes = output_dict["detection_classes"]
        r, c = boxes.shape
        detected_coords = []
        json_object = {}.fromkeys(["objects"], [])
        width, height = pil_image.size
        for i in range(r):
            if scores[i] * 100 >= 90.0:
                object_json = dict().fromkeys(
                    ["object", "xmin", "ymin", "xmax", "ymax"], "")
                if str(classes[i]) == "1":
                    object_json["object"] = "textbox"
                elif str(classes[i]) == "2":
                    object_json["object"] = "radiobutton"
                elif str(classes[i]) == "3":
                    object_json["object"] = "checkbox"

                ymin = boxes[i][0] * height
                xmin = boxes[i][1] * width
                ymax = boxes[i][2] * height
                xmax = boxes[i][3] * width

                object_json["xmin"] = float(xmin)
                object_json["ymin"] = float(ymin)
                object_json["xmax"] = float(xmax)
                object_json["ymax"] = float(ymax)
                object_json["coords"] = ",".join([str(xmin), str(ymin), str(xmax), str(ymax)])
                object_json["score"] = scores[i]
                if object_json["object"] == "textbox":
                    detected_coords.append((xmin - 5, ymin, xmax + 5, ymax))
                    object_json["size"], object_json["weight"] = \
                        extract_properties.get_size_and_weight(image=pil_image,
                                                                coords=(xmin, ymin, xmax, ymax))
                    object_json["horizontal_alignment"] = \
                        extract_properties.get_alignment(image=pil_image,
                                                          xmin=float(xmin), xmax=float(xmax))
                    object_json["color"] =\
                        extract_properties.get_colors(image=pil_image,
                                                       coords=(xmin, ymin, xmax, ymax))
                else:
                    detected_coords.append((xmin, ymin, xmax, ymax))
                object_json["text"] = extract_properties.get_text(
                    image=pil_image, coords=(xmin, ymin, xmax, ymax))
                json_object["objects"].append(object_json)
        return json_object, detected_coords

    def main(self, image=None):

        """
        Handles the different components calling and returns the
        predicted card json to the API

        @param labels_path: faster rcnn model's label path
        @param forzen_graph_path: faster rcnn model path
        @param image: input image path

        @return: predicted card json
        """
        image_np = np.asarray(image)
        image_np=cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        # Extract the design objects from faster rcnn model
        output_dict, category_index = self.od_model.get_objects(image=image_np)
        # Collect the objects along with its design properites
        json_objects, detected_coords = self.collect_objects(
            output_dict=output_dict, pil_image=image)
        # Detect image coordinates inside the card design
        image_extraction=ImageExtraction()
        image_points = image_extraction.detect_image(
            image=image_np, detected_coords=detected_coords, pil_image=image)
        image_urls,image_sizes = image_extraction.image_crop_get_url(
            coords=image_points, image=image)

        # Arrange the design elements
        card_arrange=CardArrange()
        card_arrange.remove_overlapping_objects(json_object=json_objects)
        card_arrange.append_image_objects(
            image_urls=image_urls,
            image_sizes=image_sizes,
            image_coords=image_points,
            pil_image=image,
            json_object=json_objects)
        return_dict = {}.fromkeys(["card_json"], "")
        card_json = {"type": "AdaptiveCard", "version": "1.0", "body": [
        ], "$schema": "http://adaptivecards.io/schemas/adaptive-card.json"}
        body, ymins = card_arrange.build_card_json(
            objects=json_objects.get("objects", []))
        # Sort the elements vertically
        body = [x for _, x in sorted(zip(ymins, body), key=lambda x: x[0])]

        # Prepare the response with error code
        error = None
        if not body:
            error = {
                "msg": "Failed to generate card components",
                "code": 1000
            }
        else:
            card_json["body"] = body

        return_dict["card_json"] = card_json
        return_dict["error"] = error

        return return_dict
