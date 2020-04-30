"""Module to  get the predicted adaptive card json"""

import argparse
import sys
from PIL import Image
import os
import sys
sys.path.append(os.getcwd())
from mystique.detect_objects import ObjectDetection
from mystique.extract_properties import ExtractProperties
from mystique.image_extraction import ImageExtraction
from mystique.arrange_card import CardArrange
from object_detection.utils import visualization_utils as vis_util
import cv2
import base64
import json


class PredictCard:

    def collect_objects(self, output_dict=None):
        """[Returns the design elements from the faster rcnn model with its properties]

        Keyword Arguments:
            output_dict {[dict]} -- [Output dict from the object detection] (default: {None})

        Returns:
            [dict] -- [Collected json of the design objects]
            [list] -- [list of detected object's coordinates]
        """
        boxes = output_dict['detection_boxes']
        scores = output_dict['detection_scores']
        classes = output_dict['detection_classes']
        r, c = boxes.shape
        detected_coords = []
        json_object = {}.fromkeys(['objects'], [])
        pil_image = Image.open(self.image_path)
        width, height = pil_image.size
        for i in range(r):
            if scores[i] * 100 >= 90.0:
                object_json = dict().fromkeys(
                    ['object', 'xmin', 'ymin', 'xmax', 'ymax'], '')
                if str(classes[i]) == "1":
                    object_json['object'] = "textbox"
                elif str(classes[i]) == "2":
                    object_json['object'] = "radiobutton"
                elif str(classes[i]) == "3":
                    object_json['object'] = "checkbox"

                ymin = boxes[i][0] * height
                xmin = boxes[i][1] * width
                ymax = boxes[i][2] * height
                xmax = boxes[i][3] * width

                object_json['xmin'] = str(xmin)
                object_json['ymin'] = str(ymin)
                object_json['xmax'] = str(xmax)
                object_json['ymax'] = str(ymax)
                object_json['coords'] = ','.join(
                    [str(xmin), str(ymin), str(xmax), str(ymax)])
                object_json['score'] = scores[i]
                if object_json['object'] == 'textbox':
                    detected_coords.append((xmin - 5, ymin, xmax + 5, ymax))
                else:
                    detected_coords.append((xmin, ymin, xmax, ymax))
                object_json['text'] = ExtractProperties().get_text(
                    image=pil_image, coords=((xmin, ymin, xmax, ymax)))
                if object_json['object'] == "textbox":
                    object_json["size"], object_json['weight'] = ExtractProperties(
                    ).get_size_and_weight(image=pil_image, coords=((xmin, ymin, xmax, ymax)))
                    object_json["horizontal_alignment"] = ExtractProperties().get_alignment(
                        image=pil_image, xmin=float(xmin), xmax=float(xmax))
                    object_json['color'] = ExtractProperties().get_colors(
                        image=pil_image, coords=((xmin, ymin, xmax, ymax)))
                json_object['objects'].append(object_json)
        return json_object, detected_coords

    def main(self, image_path=None, forzen_graph_path=None, labels_path=None):

        self.image_path = image_path
        self.forzen_graph_path = forzen_graph_path
        self.labels_path = labels_path
        image_np = cv2.imread(self.image_path)
        pil_image = Image.open(self.image_path)
        # Extract the design objects from faster rcnn model
        obj = ObjectDetection(
            path_to_frozen_graph=self.forzen_graph_path,
            path_to_labels=self.labels_path)
        output_dict, category_index = obj.get_objects(
            image_path=self.image_path)
        # Collect the objects along with its design properites
        json_objects, detected_coords = self.collect_objects(
            output_dict=output_dict)
        # Detect image coordinates inside the card design
        image_points = ImageExtraction().detect_image(
            image=image_np, detected_coords=detected_coords, pil_image=pil_image)
        image_urls,image_sizes = ImageExtraction().image_crop_get_url(
            coords=image_points, image=pil_image)

        # Arrange the design elements
        CardArrange().remove_overlapping_objects(json_object=json_objects)
        CardArrange().append_image_objects(
            image_urls=image_urls,
            image_sizes=image_sizes,
            image_coords=image_points,
            pil_image=pil_image,
            json_object=json_objects)
        return_dict = {}.fromkeys(['image', 'card_json'], '')

        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=1,
            skip_scores=True,
            skip_labels=True,
            min_score_thresh=0.9
        )
        return_dict["image"] = base64.b64encode(
            cv2.imencode('predicted.jpg', image_np)[1]).decode("utf-8")
        card_json = {"type": "AdaptiveCard", "version": "1.0", "body": [
        ], "$schema": "http://adaptivecards.io/schemas/adaptive-card.json"}
        body, ymins = CardArrange().build_card_json(
            objects=json_objects.get('objects', []))
        # Sort the elements vertically
        for obj in range(len(ymins) - 1, 0, -1):
            for i in range(obj):
                if i + 1 < len(ymins):
                    if float(ymins[i]) > float(ymins[i + 1]):
                        temp1 = ymins[i]
                        temp = body[i]
                        body[i] = body[i + 1]
                        ymins[i] = ymins[i + 1]
                        body[i + 1] = temp
                        ymins[i + 1] = temp1
        card_json["body"] = body
        return_dict["card_json"] = card_json
        return json.dumps(return_dict)

