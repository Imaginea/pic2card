""" utils for the app """
import os
import base64
import json

def get_templates(templates_path='assets/samples'):
    """
    reads images from templates_path folder and returns images in str
    :param templates_path: path of templates folder
    :return: dict of template_image_name : encoded template_image_str
    """

    templates = []
    templates_path = os.path.join(os.path.dirname(__file__), templates_path)
    for file in os.listdir(templates_path):
        file_path = os.path.join(templates_path, file)
        with open(file_path, "rb") as template:
            templates.append(base64.b64encode(template.read()).decode())
    return {"templates":templates}
