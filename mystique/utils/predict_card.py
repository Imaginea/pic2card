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

sys.path.append(os.getcwd())


class PredictCard:

    
    def collect_objects(self, output_dict=None, image_path=None):

        """
        Returns the design elements from the faster rcnn model with its 
        properties mapped

        @param output_dict: output dict from the object detection

        @return: Collected json of the design objects 
                 and list of detected object's coordinates
        """
        boxes = output_dict["detection_boxes"]
        scores = output_dict["detection_scores"]
        classes = output_dict["detection_classes"]
        r, c = boxes.shape
        detected_coords = []
        json_object = {}.fromkeys(["objects"], [])
        pil_image = Image.open(image_path)
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
                else:
                    detected_coords.append((xmin, ymin, xmax, ymax))
                object_json["text"] = ExtractProperties().get_text(
                    image=pil_image, coords=(xmin, ymin, xmax, ymax))
                if object_json["object"] == "textbox":
                    object_json["size"], object_json["weight"] = \
                        ExtractProperties().get_size_and_weight(image=pil_image, 
                                                                coords=(xmin, ymin, xmax, ymax))
                    object_json["horizontal_alignment"] = \
                        ExtractProperties().get_alignment(image=pil_image, 
                                                          xmin=float(xmin), xmax=float(xmax))
                    object_json["color"] =\
                        ExtractProperties().get_colors(image=pil_image, 
                                                       coords=(xmin, ymin, xmax, ymax))
                json_object["objects"].append(object_json)
        return json_object, detected_coords

    def main(self, labels_path=None, frozen_graph_path=None, image_path=None):
        
        image_np = cv2.imread(image_path)
        pil_image = Image.open(image_path)
        # Extract the design objects from faster rcnn model
        output_dict, category_index = ObjectDetection().get_objects(image_path=image_path,
            path_to_frozen_graph=frozen_graph_path, 
            path_to_label=labels_path)
        # Collect the objects along with its design properites
        json_objects, detected_coords = self.collect_objects(
            output_dict=output_dict, image_path=image_path)
        # Detect image coordinates inside the card design
        image_points = ImageExtraction().detect_image(
            image=image_np, detected_coords=detected_coords, pil_image=pil_image)
        image_urls = ImageExtraction().image_crop_get_url(
            coords=image_points, image=pil_image)

        # Arrange the design elements
        CardArrange().remove_overlapping_objects(json_object=json_objects)
        CardArrange().append_image_objects(
            image_urls=image_urls,
            image_coords=image_points,
            pil_image=pil_image,
            json_object=json_objects)
        return_dict = {}.fromkeys(["card_json"], "")
        card_json = {"type": "AdaptiveCard", "version": "1.0", "body": [
        ], "$schema": "http://adaptivecards.io/schemas/adaptive-card.json"}
        body, ymins = CardArrange().build_card_json(
            objects=json_objects.get("objects", []))
        # Sort the elements vertically
        body = [x for _, x in sorted(zip(ymins, body))]
        card_json["body"] = body
        return_dict["card_json"] = card_json
        return json.dumps(return_dict)
