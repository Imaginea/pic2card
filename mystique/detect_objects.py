"""Module for object detection using faster rcnn"""

from distutils.version import StrictVersion

import numpy as np
import tensorflow as tf
from typing import Dict, Tuple
from PIL import Image

import torch
import torchvision.transforms as T
from detecto.core import Model
from detecto.utils import read_image, normalize_transform

from mystique.utils import id_to_label
from mystique import config


if StrictVersion(tf.__version__) < StrictVersion("1.9.0"):
    raise ImportError(
        "Please upgrade your TensorFlow installation to v1.9.* or later!")


class PtObjectDetection:
    """
    Pytorch implementation of object detection classes.
    """
    transformer = T.Compose([
            T.ToPILImage(),
            lambda image: image.convert("RGB"),
            T.ToTensor(),
            normalize_transform()
    ])

    classes = ["checkbox",
               "radiobutton",
               "textbox",
               "actionset",
               "image",
               "rating",
               "textboox"]
    model = None

    def __init__(self, model_path=None):
        self.model_path = model_path or config.PTH_MODEL_PATH
        if self.model_path:
            self.model = self._load_model()

    def _transform(self, image: np.array) -> torch.Tensor:
        """
        Transform the image and convert to Tensor.
        """
        return self.transformer(image)

    def _load_model(self):
        """ Load the saved model and pass the required classes """
        return Model.load(self.model_path, classes=self.classes)

    def get_bboxes(self, image_path: str) -> Dict:
        """
        Do inference and return the bounding boxes compatible to caller.
        """
        image_np = read_image(image_path)
        image = self._transform(image_np)
        labels, boxes, scores = self.model.predict([image])[0]

        return labels, boxes.tolist(), scores.tolist()


class ObjectDetection:
    """
    Class handles generating faster rcnn models from the model inference
    graph and returning the ouput dict which consists of classes, scores,
    and object bounding boxes.
    """

    def __init__(self, detection_graph, category_index, tensor_dict):
        """
        Initialize the object detection using model loaded from forzen
        graph
        """
        self.detection_graph = detection_graph
        self.category_index = category_index
        self.tensor_dict = tensor_dict

    def get_bboxes(self, image_path: str) -> Tuple:
        """
        Get the bounding boxes with scores and label.
        """
        image = Image.open(image_path)
        width, height = image.size
        image = image.convert("RGB")
        width, height = image.size
        image_np = np.asarray(image)
        result, _index = self.get_objects(image_np)

        classes = [id_to_label(i) for i in result['detection_classes']]
        scores = result['detection_scores'].tolist()
        boxes = result['detection_boxes'].tolist()

        # Denormalize the bounding box coordinates.
        bbox_dnorm = []
        for bbox in boxes:
            ymin = bbox[0] * height
            xmin = bbox[1] * width
            ymax = bbox[2] * height
            xmax = bbox[3] * width
            bbox_dnorm.append([xmin, ymin, xmax, ymax])

        return classes, bbox_dnorm, scores

    def get_objects(self, image: np.array = None):
        """
        Returns the objects and coordiates detected
        from the faster rcnn detected boxes]

        @param image: Image tensor, dimension should be HxWx3

        @return: ouput dict from the faster rcnn inference
        """
        output_dict = self.run_inference_for_single_image(image)
        return output_dict, self.category_index

    def run_inference_for_single_image(self, image: np.array):
        """
        Runs the inference graph for the given image
        @param image: numpy array of input design image
        @return: output dict of objects, classes and coordinates
        """

        # Run inference
        detection_graph = self.detection_graph
        with detection_graph.as_default():
            image_tensor = detection_graph.get_tensor_by_name(
                "image_tensor:0")
            with tf.compat.v1.Session() as sess:
                output_dict = sess.run(
                    self.tensor_dict, feed_dict={
                        image_tensor: np.expand_dims(
                            image, 0)})

        # all outputs are float32 numpy arrays, so convert types as
        # appropriate
        output_dict["detection_classes"] = output_dict[
            "detection_classes"][0].astype(np.uint8)
        output_dict["detection_boxes"] = output_dict[
            "detection_boxes"][0]
        output_dict["detection_scores"] = output_dict[
            "detection_scores"][0]

        return output_dict
