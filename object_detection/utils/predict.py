

#import jsonlines
import base64
import numpy as np
import os
import math
import sys
import subprocess
import tarfile
import tensorflow as tf
import zipfile
import re
import json
import pytesseract
from pytesseract import pytesseract
import os
from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import csv
import cv2
import argparse

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
sys.path.append("/home/vasanth/mystique/models/research")
from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util

MODEL_NAME = '/home/vasanth/mystique/object_detection/inference_graph'
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = '/home/vasanth/mystique/object_detection/training/object-detection.pbtxt'


detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
    
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

from PIL import Image, ImageDraw

def run_inference_for_single_image(image, graph,tensor_dict,sess):
    
    if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

    # Run inference
    output_dict = sess.run(tensor_dict,
                            feed_dict={image_tensor: np.expand_dims(image, 0)})

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict[
        'detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict

def get_card_json(objects):
        body=[]
        for object in objects:
            if object.get('object')=="textbox":
                size='Default'
                if float(object.get('size'))==-1:
                    size="Default"
                elif float(object.get('size'))>=0 and  float(object.get('size'))<=5:
                    size="Small"
                elif float(object.get('size'))>5 and  float(object.get('size'))<=10:
                    size="Medium"
                elif float(object.get('size'))>15 and  float(object.get('size'))<=20:
                    size="Large"
                elif float(object.get('size'))>20:
                    size="ExtraLarge"
                weight='Default'    
                if object.get('weight',0)>=0 and object.get('weight',0)<=50:
                    weight='Lighter'
                elif object.get('weight',0)>50 and object.get('weight',0)<=90:
                    weight='Default'
                elif object.get('weight',0)>90:
                    weight='Bolder'



                if len(object.get('text','').split())>=10:
                    body.append( {
                        "type": "RichTextBlock",
                        "inlines": [
                        {
                        "type": "TextRun",
                        "text": object.get('text',''),
                        "size":size,
                        "horizontalAlignment":object.get('horizontal_alignment',''),
                        "color":object.get('color','Default'),
                        #"weight":weight,
                        #"coords":object.get('coords','')
                        }
                        ]
                        })
                else:
                    body.append({
                    "type": "TextBlock",
                    "text": object.get('text',''),
                    "size":size,
                    "horizontalAlignment":object.get('horizontal_alignment',''),
                    "color":object.get('color','Default'),
                    #"weight":weight,
                    #"coords":object.get('coords','')
                    })
            if object.get('object')=="checkbox":
                body.append({
                    "type": "Input.Toggle",
                    "title": object.get('text',''),
                    #"coords":object.get('coords','')
                    })
            if object.get('object')=="radio_button":
                body.append( {
                    "type": "Input.ChoiceSet",
                    "choices": [
                        {
                            "title": object.get('text',''),
                            "value": "",
                            #"coords":object.get('coords','')
                            }
                        ],
                     "style": "expanded"
                    })
            if object.get('object')=="image":
                body.append( {
                    "type": "Image",
                    "altText": "",
                    #"coords":object.get('coords','')
                    })
        return body

def get_text(image, coords):
    cropped_image = image.crop(coords)
    data = pytesseract.image_to_string(cropped_image, lang='eng',config='--psm 6')
    return data

def get_size(image,coords):
    cropped_image = image.crop(coords)
    hocr=pytesseract.image_to_pdf_or_hocr(cropped_image, extension='hocr',lang='eng')
    hocr_text=hocr.decode("utf-8")
    pattern=re.compile(r"x_size\s*=*\s*\d+\.*\d*")
    if re.findall(pattern,hocr_text):
        return float(re.findall(pattern,hocr_text)[0].strip('x_size '))
    else:
        return -1

def get_alignment(image,xmin,xmax):
    avg=math.ceil((xmin+xmax)/2)
    w,h=image.size
    if (avg/w)*100 >=0 and (avg/w)*100 <45:
        return "Left"
    elif (avg/w)*100 >=45 and (avg/w)*100 <55:
        return "Center"
    else:
        return "Right"

def get_distance(v1, v2):
    return np.sqrt(np.sum((v1 - v2) ** 2))

def get_colors(image,coords):
    cropped_image = image.crop(coords)
    cropped_image.save("temp_image.png")
    q = cropped_image.quantize(colors=2,method=2)
    dominant_color= q.getpalette()[3:6]
    
    colors={
        "Attention":(255,0,0),
        "Accent":(0,0,255),
        "Good":(0,128,0),
        "Dark":(0,0,0),
        "Light":(255,255,255),
        "Warning":(255,255,0)
    }
    color='Default'
    found_colors=[]
    distances=[]
    for key,values in colors.items():
        distance=get_distance(np.asarray(values),np.asarray(dominant_color))
        if distance<=150:
            found_colors.append(key)
            distances.append(distance)
    if found_colors!=[]:
        index=distances.index(min(distances))
        color=found_colors[index]
        if found_colors[index]=="Light":
            background=q.getpalette()[:3]
            foreground=q.getpalette()[3:6]
            distance=get_distance(np.asarray(background),np.asarray(foreground))
            if distance<150:
                color='Default'
    return color
    



def get_text_weight(image,coords,text,size):
    cropped_image = image.crop(coords)
    draw = ImageDraw.Draw(cropped_image)
    size=int(math.ceil(size))
    #print(size)
    font = ImageFont.truetype('/home/vasanth/mystique/object_detection/arial.ttf', size)
    #font = ImageFont.load_default()
    ascent, descent = font.getmetrics()
    (width, baseline), (offset_x, offset_y) = font.font.getsize(text)
    return width


def main(input_file_path):
    return_dict={"image":'',"card_json":''}
    with detection_graph.as_default():
        with tf.Session() as sess:
                # Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in [
                  'num_detections', 'detection_boxes', 'detection_scores',
                  'detection_classes', 'detection_masks'
                ]:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                      tensor_name)
                test_image_path=input_file_path
                if ".png" in test_image_path or ".jpeg" in test_image_path or ".jpg" in test_image_path:
                     images=[test_image_path]
                else:
                     images=os.listdir(test_image_path)
                json_lines=[]
                for image in images:
                   if image.find(".png")!=-1 or image.find(".jpg")!=-1 or image.find(".jpeg")!=-1 :
                    
                    img_path =test_image_path
                    image_pillow = Image.open(img_path)
                    image_np = cv2.imread(img_path)
                    image_np_expanded = np.expand_dims(image_np, axis=0)
                    output_dict = run_inference_for_single_image(image_np, detection_graph,tensor_dict,sess)
                    width,height = image_pillow.size
                    boxes=output_dict['detection_boxes']
                    scores=output_dict['detection_scores']
                    classes=output_dict['detection_classes']
                    def draw_bounding_box_on_image(image,ymin,xmin,ymax,xmax,color='red',thickness=4,display_str_list=(),use_normalized_coordinates=True):
                        
                        im_width, im_height = image.size
                        if use_normalized_coordinates:
                            (left, right, top, bottom) = (xmin * im_width, xmax * im_width,ymin * im_height, ymax * im_height)
                        draw.rectangle([(xmin,ymin),(xmax,ymax)], outline="red",width=3)
                        xi=str(round(xmin, 2))
                        yi=str(round(ymin, 2))
                        xa=str(round(xmax, 2))
                        ya=str(round(ymax, 2))
                        text="("+str(xi)+","+str(yi)+"),("+str(xa)+","+str(ya)+")"
                        
                        draw.text((xmin-10.0, ymin-10.0), text,fill="green",width=3)
                        
                    r,c=boxes.shape
                    #draw = ImageDraw.Draw(image_pillow)
                    json_object={"file_name":image}
                    json_object['objects']=[]
                    for i in range(r):
                      
                      if scores[i]*100>=70.0:
                        object_json=dict().fromkeys(['object','xmin','ymin','xmax','ymax'],'')
                        if str(classes[i])=="1":
                            object_json['object']="textbox"         
                        elif  str(classes[i])=="2":
                            object_json['object']="radio_button"
                        elif  str(classes[i])=="3":
                            object_json['object']="checkbox"
                        elif  str(classes[i])=="4":
                            object_json['object']="image"
                        ymin = boxes[i][0]*height
                        xmin = boxes[i][1]*width
                        ymax = boxes[i][2]*height
                        xmax = boxes[i][3]*width
                        object_json['xmin']=str(xmin)
                        object_json['ymin']=str(ymin)
                        object_json['xmax']=str(xmax)
                        object_json['ymax']=str(ymax)
                        object_json['coords']=','.join([str(xmin),str(ymin),str(xmax),str(ymax)])
                        object_json['text']=get_text(image_pillow,((xmin, ymin, xmax,ymax )))
                        if object_json['object']=="textbox":
                            object_json["size"]=get_size(image_pillow,((xmin, ymin, xmax,ymax )))
                            object_json["horizontal_alignment"]=get_alignment(image_pillow,float(xmin),float(xmax))
                            object_json['color']=get_colors(image_pillow,((xmin, ymin, xmax,ymax )))
                            object_json['weight']=get_text_weight(image_pillow,((xmin, ymin, xmax,ymax )),object_json['text'],object_json['size'])
                        json_object['objects'].append(object_json)

                        
                        
                        #draw_bounding_box_on_image(image_pillow,ymin,xmin,ymax,xmax)                    
                    
                    
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        image_np,
                        output_dict['detection_boxes'],
                        output_dict['detection_classes'],
                        output_dict['detection_scores'],
                        category_index,
                        instance_masks=output_dict.get('detection_masks'),
                        use_normalized_coordinates=True,
                        line_thickness=5,
                        skip_scores=False,
                        min_score_thresh=0.7
                        )
                    return_dict["image"]=base64.b64encode(cv2.imencode('predicted.jpg', image_np)[1]).decode("utf-8")
                    for obj in range(len(json_object.get('objects'))-1,0,-1):
                        for i in range(obj):
                            if float(json_object.get('objects')[i].get('ymin'))>float(json_object.get('objects')[i+1].get('ymin')):
                                temp = json_object.get('objects')[i]
                                json_object.get('objects')[i] = json_object.get('objects')[i+1]
                                json_object.get('objects')[i+1] = temp
                                
                    card_json = {"type": "AdaptiveCard", "version": "1.0", "body": [], "$schema": "http://adaptivecards.io/schemas/adaptive-card.json"}
                    card_json["body"]=get_card_json(json_object.get('objects',[]))
                    #print(card_json,"\n\nJSON")

                    return_dict["card_json"]=card_json
                    print(json.dumps(return_dict))
               

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate Objectss')
    parser.add_argument('--image_path',required=True,help='Enter Test Image Path or Test Images Folder Path')

    args = parser.parse_args()
    main(args.image_path)


