"""Module for object detection using faster rcnn"""

import os
from distutils.version import StrictVersion
from flask import current_app as app
import cv2
import numpy as np
import tensorflow as tf

if StrictVersion(tf.__version__) < StrictVersion("1.9.0"):
    raise ImportError(
        "Please upgrade your TensorFlow installation to v1.9.* or later!")


class ObjectDetection:


    def get_objects(self, image_path=None):

        """
        Returns the objects and coordiates detected 
        from the faster rcnn detected boxes]

        @param image_path: input image path

        @return: ouput dict from the faster rcnn inference
        """
        image_np = cv2.imread(image_path)
        output_dict = self.run_inference_for_single_image (image_np)
        return output_dict, app.config['CATEGORY_INDEX']

    def run_inference_for_single_image(self, image):

        """
        Runs the inference graph for the given image
        @param image: numpy array of input design image
        @return: output dict of objects, classes and coordinates
        """

        # Run inference
        detetection_graph = app.config['DETECTION_GRAPTH']
        with detetection_graph.as_default():
            image_tensor = detetection_graph.get_tensor_by_name("image_tensor:0")
            with tf.Session () as sess:
                output_dict = sess.run(
                    app.config['TENSOR_DICT'], feed_dict={
                        image_tensor: np.expand_dims(
                            image, 0)})

        # all outputs are float32 numpy arrays, so convert types as appropriate
        output_dict["detection_classes"] = output_dict["detection_classes"][0].astype(np.uint8)
        output_dict["detection_boxes"] = output_dict["detection_boxes"][0]
        output_dict["detection_scores"] = output_dict["detection_scores"][0]

        return output_dict
