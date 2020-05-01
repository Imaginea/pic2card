"""[Module for object detection using faster rcnn]"""

import os
from distutils.version import StrictVersion

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from object_detection.utils import label_map_util
from object_detection.utils import ops as utils_ops

if StrictVersion (tf.__version__) < StrictVersion ('1.9.0'):
    raise ImportError (
            'Please upgrade your TensorFlow installation to v1.9.* or later!')


class ObjectDetection:
    model_name = None
    path_to_frozen_graph = None
    path_to_labels = None

    def __init__ (self,path_to_frozen_graph=path_to_frozen_graph,path_to_labels=path_to_labels):
        self.path_to_frozen_graph = path_to_frozen_graph
        self.path_to_labels = path_to_labels

    def get_objects(self, image_path=None):
        """[Returns the objects and coordiates from the faster rcnn detected boxes]

        Keyword Arguments:
            image_path {[String]} -- [Input image path] (default: {None})

        Returns:
            [dict] -- [Ouput dict from the faster rcnn inference]
        """
        detection_graph = tf.Graph ()
        with detection_graph.as_default ():
            od_graph_def = tf.GraphDef ()
            with tf.gfile.GFile (self.path_to_frozen_graph, 'rb') as fid:
                serialized_graph = fid.read ()
                od_graph_def.ParseFromString (serialized_graph)
                tf.import_graph_def (od_graph_def, name='')
        category_index = label_map_util.create_category_index_from_labelmap (
                self.path_to_labels, use_display_name=True)

        with detection_graph.as_default ():
            with tf.Session () as sess:
                # Get handles to input and output tensors
                ops = tf.get_default_graph ().get_operations ()
                all_tensor_names = {
                        output.name for op in ops for output in op.outputs}

                tensor_dict = {}
                for key in [
                        'num_detections', 'detection_boxes', 'detection_scores',
                        'detection_classes', 'detection_masks'
                ]:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph (
                        ).get_tensor_by_name (tensor_name)
                test_image_path = image_path
                if ".png" in test_image_path or ".jpeg" in test_image_path or ".jpg" in test_image_path:
                    images = [test_image_path]
                else:
                    images = os.listdir (test_image_path)
                for image in images:
                    if image.find (".png") != -1 or image.find (".jpg") != - \
                            1 or image.find (".jpeg") != -1:
                        img_path = test_image_path
                        image_pillow = Image.open (img_path)
                        image_np = cv2.imread (img_path)
                        output_dict = self.run_inference_for_single_image (
                                image_np, detection_graph, tensor_dict, sess)
                        return output_dict, category_index

    def run_inference_for_single_image(self, image, graph, tensor_dict, sess):
        """[Runs the inference graph for the given image]

        Arguments:
            image {[open-cv image]} -- [Input image]
            graph {[tf Graph]} -- [Detection graph]
            tensor_dict {[dict]} -- [Tensorflow graph and name dict]
            sess {[tf Session]} -- [Tensorflow Session]

        Returns:
            [dict] -- [Output dict of objects, classes and coordinates]
        """

        if 'detection_masks' in tensor_dict:
            # The following processing is only for single image
            detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
            detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
            # Reframe is required to translate mask from box coordinates to
            # image coordinates and fit the image size.
            real_num_detection = tf.cast(
                tensor_dict['num_detections'][0], tf.int32)
            detection_boxes = tf.slice(detection_boxes, [0, 0], [
                                       real_num_detection, -1])
            detection_masks = tf.slice(detection_masks, [0, 0, 0], [
                                       real_num_detection, -1, -1])
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                detection_masks, detection_boxes, image.shape[0], image.shape[1])
            detection_masks_reframed = tf.cast(
                tf.greater(detection_masks_reframed, 0.5), tf.uint8)
            # Follow the convention by adding back the batch dimension
            tensor_dict['detection_masks'] = tf.expand_dims(
                detection_masks_reframed, 0)
        image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

        # Run inference
        output_dict = sess.run(
            tensor_dict, feed_dict={
                image_tensor: np.expand_dims(
                    image, 0)})

        # all outputs are float32 numpy arrays, so convert types as appropriate
        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict[
            'detection_classes'][0].astype(np.uint8)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]
        if 'detection_masks' in output_dict:
            output_dict['detection_masks'] = output_dict['detection_masks'][0]
        return output_dict
