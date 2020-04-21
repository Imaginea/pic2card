

#import jsonlines
import base64
import numpy as np
import os
import math
import sys
import subprocess
import tarfile
import tensorflow as tf
import re
import json
import pytesseract
from pytesseract import pytesseract
import os
from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from PIL import Image
from matplotlib import pyplot as plt
import cv2
import pandas as pd

import argparse
import requests

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
#sys.path.append("/home/vasanth/mystique/models/research")
sys.path.append("/home/keerthanamanoharan/Documents/office_work/Pic2Code/mystique/models/research")
from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util

MODEL_NAME = '/home/vasanth/mystique/object_detection/inference_graph9000'
#MODEL_NAME = '/home/keerthanamanoharan/Documents/office_work/Pic2Code/mystique/object_detection/inference_graph'
#MODEL_NAME = '/home/keerthanamanoharan/Documents/office_work/Pic2Code/mystique/object_detection/data_varaiance_graph/inference_graph9000'

PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = '/home/vasanth/mystique/object_detection/training_variance/object-detection.pbtxt'
#PATH_TO_LABELS = '/home/keerthanamanoharan/Documents/office_work/Pic2Code/gitlab/mystique/object_detection/training/object-detection.pbtxt'
#PATH_TO_LABELS = '/home/keerthanamanoharan/Documents/office_work/Pic2Code/mystique/object_detection/data_varaince_files/training_variance/object-detection.pbtxt'



detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
    
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


"""[checks for overlapping rectangels in the countours]

Returns:
    [Boolean] -- [if overlaps or not]
"""
def contains(point, coords):
  return (float(coords[0])<=point[0]+5<=float(coords[2])) and (float(coords[1])<=point[1]+5<=float(coords[3]))



"""[image hosting service]

Returns:
    [list] -- [image urls , coords]
"""
def image_crop_get_url(coords_list,img_pillow):
  images=[] 
  image_coords=[] 
  #ctr=0
  for coords in coords_list:
    cropped=img_pillow.crop((coords[0],coords[1],coords[2],coords[3]))
    cropped.save("image_detected.png")
    img=open("image_detected.png","rb").read()
    base64_string=base64.b64encode(img).decode()
    url = "https://post.imageshack.us/upload_api.php"
    payload = {'key': 'DXOLV1ZA47ad805e246e758984d1198e982c0b8f',
    'format': 'json',
    'tags': 'sample',
    'public': 'yes'}
    files = [
        ('fileupload', open('image_detected.png','rb'))
        ]
    response = requests.request("POST", url, data = payload, files = files)
    images.append(response.json().get("links",{}).get("image_link",''))
    # images.append("")
    image_coords.append(coords)
  return images


"""[Returns the dected images coords and hosted urls]

Returns:
    [list] -- [results from image_crop_list]
"""
def image_detection(img,detected_coords,img_pillow):
    image_points=[]
    #pre processing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dst = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(dst,(5,5),0)
    ret, im_th =cv2.threshold(blur,150,255,cv2.THRESH_BINARY)
    # Set the kernel and perform opening
    # k_size = 6
    kernel = np.ones((5,5),np.uint8)
    opened = cv2.morphologyEx(im_th, cv2.MORPH_OPEN, kernel)
    #edge detection
    edged = cv2.Canny(opened,0,255)
    #countours
    _, contours, hierarchy = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #get the coords of the contours
    for c in contours:
        (x,y,w,h) = cv2.boundingRect(c)
        image_points.append((x,y,x+w,y+h))

    for i in range(len(image_points)):
        for j in range(len(image_points)):
          if j < len(image_points) and i < len(image_points):
            box1=[float(c) for c  in image_points[i]]
            box2=[float(c) for c  in image_points[j]]
            intersection=FindPoints(box1[0],box1[1],box1[2],box1[3],box2[0],box2[1],box2[2],box2[3],image=True)
            conatin=contains(box1,box2)
            if intersection and contains:
                if box1!=box2:
                    if box1[2]-box1[0] > box2[2]-box2[0]:
                        del image_points[j]
                    else:
                        del image_points[i]
    
    #extarct the points that lies inside the detected objects coords [ rectangle ]
    included_points_positions=[0] * len(image_points)
    for point in image_points:
      for p in detected_coords:
        if contains((point[0],point[1]),p):
          included_points_positions[image_points.index(point)]=1
    # now get the image points / coords that lies outside the detected objects coords     
    image_points1=[]
    for point in image_points:
      if included_points_positions[image_points.index(point)]!=1:
        image_points1.append(point)
    image_points=sorted(set(image_points1),key=image_points1.index)
    #=image_points[:-1]
    width,height=img_pillow.size
    widths=[point[2]-point[0] for point in image_points]
    position=widths.index(max(widths))
    if max(widths)-width <=10:
        del image_points[position]

    return image_crop_get_url(image_points,img_pillow),image_points
    



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


"""[returns the grouped image objects as adaptive card json body]

Returns:
    [None] -- [apends ymins and body]
"""
def group_image_objects(image_objects,body,ymins,objects):

    #group the image objects based on ymin
    groups=[]
    # left_over_images=[]
    unique_ymin=list(set([x.get('ymin') for x in image_objects]))
    for un in unique_ymin:
        l=[]
        for xx in image_objects:
            if abs(float(xx.get('ymin'))-float(un))<=10.0:
                l.append(xx)
        if l not in groups:
            groups.append(l)
    
    
    #now put similar ymin grouped objects into a imageset - if a group has more than one image object
    for group in groups:
        
        for obj in range(len(group)-1,0,-1):
            for i in range(obj):
                if float(group[i].get('xmin'))>float(group[i+1].get('xmin')):
                    temp = group[i]
                    group[i]=group[i+1]
                    group[i+1] = temp

        if len(group)>1:
            image_set= {
            "type": "ImageSet",
            "imageSize": "medium",
            "images": []}
            for object in group:
                if object in objects:
                    del objects[objects.index(object)]
                obj= {"type": "Image",
                "altText": "Image",
                "horizontalAlignment": object.get('horizontal_alignment',''),
                "url":object.get('url'),
                #"coords":object.get('coords','')
                }
                image_set['images'].append(obj)

            body.append(image_set)
            ymins.append(object.get('ymin'))


"""[appends the individual objects to the card_json]

Returns:
    [None] -- [None]
"""
def append_objects(object,body,ymins=None,column=None):

    if object.get('object')=="image":
        body.append({
                "type": "Image",
                "altText": "Image",
                "horizontalAlignment": object.get('horizontal_alignment',''),
                "url":object.get('url'),
                #"coords":object.get('coords','')
                })
        if ymins!=None:
            ymins.append(object.get('ymin'))
    if object.get('object')=="textbox":
        if (len(object.get('text','').split())>=11 and not column) or (column and len(object.get('text',''))>=15):
            body.append( {
                        "type": "RichTextBlock",
                        "inlines": [
                        {
                        "type": "TextRun",
                        "text": object.get('text',''),
                        "size":object.get('size',''),
                        "horizontalAlignment":object.get('horizontal_alignment',''),
                        "color":object.get('color','Default'),
                        "weight":object.get('weight',''),
                        #"coords":object.get('coords','')
                        }
                        ]
                        })
            if ymins!=None:
                ymins.append(object.get('ymin'))
        else:
            body.append({
                    "type": "TextBlock",
                    "text": object.get('text',''),
                    "size":object.get('size',''),
                    "horizontalAlignment":object.get('horizontal_alignment',''),
                    "color":object.get('color','Default'),
                    "weight":object.get('weight',''),
                    #"coords":object.get('coords','')
                    })
            if ymins!=None:
                ymins.append(object.get('ymin'))
    if object.get('object')=="checkbox":
            body.append({
                    "type": "Input.Toggle",
                    "title": object.get('text',''),
                    #"coords":object.get('coords',''),
                    #"score":str(object.get('score',''))
                    })
            if ymins!=None:
                ymins.append(object.get('ymin'))


def return_position(groups,obj):
    for i in range(len(groups)):
        if obj in groups[i]:
            return i
    return -1

"""[Group choices into numebr of choicesets]

Returns:
    [NOne] -- [None]
"""
 
def group_choicesets(radiobutons,body,ymins=None):
    groups=[]
    positions_grouped=[]
    for i in range(len(radiobutons)):
        l=[]
        if i not in positions_grouped:
            l=[radiobutons[i]]
        for j in range(len(radiobutons)):
            a=float(radiobutons[i].get('ymin'))
            b=float(radiobutons[j].get('ymin'))
            difference_in_ymin=abs(a-b)
            
            if a>b:
                difference=float(radiobutons[j].get('ymax'))-a
            else:
                difference=float(radiobutons[i].get('ymax'))-b
            if abs(difference)<=10  and difference_in_ymin<=30 and j not in positions_grouped:
                if i in positions_grouped:
                    position=return_position(groups,radiobutons[i])
                    if position <len(groups) and position>=0:
                       groups[position].append(radiobutons[j]) 
                       positions_grouped.append(j)
                    elif radiobutons[i] in l:
                        l.append(radiobutons[j])
                        positions_grouped.append(j)
                else:
                    l.append(radiobutons[j])
                    positions_grouped.append(j)
                    positions_grouped.append(i)
                    
                
    
        
        if l!=[]:
            flag=False
            for gr in groups:
                for ll in l:
                    if ll in gr:
                        flag=True

            if flag==False:
                groups.append(l)

    for group in groups:
        for ob in range(len(group)-1,0,-1):
            for i in range(ob):
                if float(group[i].get('ymin'))>float(group[i+1].get('ymin')):
                    temp = group[i]
                    group[i]=group[i+1]
                    group[i+1] = temp

        choice_set={
                "type": "Input.ChoiceSet",
                "choices": [],
                "style": "expanded"
        }
       
        for obj in group:
            choice_set['choices'].append({
                "title": obj.get('text',''),
                        "value": "",
                        #"coords":obj.get('coords',''),
                        #"score":str(obj.get('score',''))
                        })
        
        body.append(choice_set)
        if ymins!=None:
            ymins.append(obj.get('ymin'))


        


"""[Returns the adapative card json body based on the detection]

Returns:
    [list] -- [body, ymins]
    
"""
def get_card_json(objects,images_number):
        body=[] 
        ymins=[]
        image_objects=[]         
        for object in objects:
            if object.get('object')=="image":
                image_objects.append(object)
        group_image_objects(image_objects,body,ymins,objects)
        groups=[]
        unique_ymin=list(set([x.get('ymin') for x in objects]))
        for un in unique_ymin:
            l=[]
            for xx in objects:
                if abs(float(xx.get('ymin'))-float(un))<=10.0:
                    flag=0
                    for gr in groups:
                        if xx in gr:
                            flag=1
                    if flag==0:
                        l.append(xx)

            
            if l not in groups:
                groups.append(l)
        
        
        radio_buttons_dict={"normal":[]}
        for group in groups:
            radio_buttons_dict['columnset']={}
            if len(group)==1:
                
                if group[0].get('object')=="radiobutton":
                    radio_buttons_dict['normal'].append(group[0])
                else:
                    append_objects(group[0],body,ymins=ymins)
            elif len(group)>1:
                colummn_set={
                    "type": "ColumnSet",
                    "columns": []}
                ctr=0

                for obj in range(len(group)-1,0,-1):
                    for i in range(obj):
                        if float(group[i].get('xmin'))>float(group[i+1].get('xmin')):
                            temp = group[i]
                            group[i]=group[i+1]
                            group[i+1] = temp

                for obj in group:
                    
                    
                
                    colummn_set['columns'].append({
                        "type": "Column",
                        "width": "stretch",
                        "items": []})
                    position=group.index(obj)
                    if position+1<len(group):
                        greater=position
                        lesser=position+1
                        if float(obj.get('ymin'))< float(group[position+1].get('ymin')):
                            greater=position+1
                            lesser=position
                        
                        if abs(float(group[greater].get('xmax'))-float(group[lesser].get('xmin')))<=10:
                            colummn_set['columns'][ctr]['width']="auto"

                    if obj.get('object')=="radiobutton":
                        radio_buttons_dict['columnset']=radio_buttons_dict['columnset'].fromkeys([ctr],[])
                        radio_buttons_dict['columnset'][ctr].append(obj)
                        
                    else:
                        append_objects(obj,colummn_set['columns'][ctr].get('items',[]),column=True)
                        
                        
                    ctr+=1                        

                    
                   
                if len(radio_buttons_dict['columnset'])>0: 
                    if ctr-1 !=-1  and ctr-1 <=len(colummn_set['columns']) and len(radio_buttons_dict['columnset'])>0:
                        if radio_buttons_dict['columnset'].get(ctr-1):
                            group_choicesets(radio_buttons_dict['columnset'].get(ctr-1),colummn_set['columns'][ctr-1].get('items',[]))
                
                
                if colummn_set not in body:
                    for column in colummn_set['columns']:
                        if column.get('items',[])==[]:
                            del colummn_set['columns'][colummn_set['columns'].index(column)]

                    body.append(colummn_set)
                    ymins.append(group[0].get('ymin',''))
        if len(radio_buttons_dict['normal'])>0:
                group_choicesets(radio_buttons_dict['normal'],body,ymins=ymins)

       
        return body,ymins

"""[OCR to get the text of the detected coords]

Returns:
    [string] -- [OCR text]
"""
def get_text(image, coords):
    coords=(coords[0],coords[1],coords[2],coords[3])
    cropped_image = image.crop(coords)

    data = pytesseract.image_to_string(cropped_image, lang='eng',config='--psm 6')
    return data

"""[Returns the size and weight properites of an object]

Returns:
    [string] -- [size,weight]
"""
def get_size_and_weight(image,coords):
    cropped_image = image.crop(coords)
    cropped_image.save("temp_image.png")
    img=cv2.imread("temp_image.png")
    #preprocess
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray, (5,5))

    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    
    #edge detection
    edged = cv2.Canny(img, 30, 200)
    #contours bulding
    _, contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    box_width=[]
    box_height=[]

    #calculate the average width and height of the contour coords of the object
    for c in contours:
        rect = cv2.minAreaRect(c)
        (x,y,w,h) = cv2.boundingRect(c)
        box_width.append(w)
        box_height.append(h)

    weights=sum(box_width)/len(box_width)
    heights=sum(box_height)/len(box_height)
    size='Default'
    weight='Default'

    if heights<=5.5:
        size='Small'
    elif heights>5.5 and heights<=7:
        size='Default'
    elif heights>7 and heights<=15:
        size="Medium"
    elif heights>15 and heights<=20:
        size="Large"
    else:
        size="ExtraLarge"
    
    if (size=="Small" or size=="Default") and weights>=5:
        weight="Bolder"
    elif size=="Medium" and weights>6.5:
        weight="Bolder"
    elif size=="Large" and weights>8:
        weights="Bolder"
    elif size=="ExtraLarge" and weights>9:
        weight="Bolder"

    
    return size,weight

"""[Return the horizontal alignment of the object]

Returns:
    [string] 
"""
def get_alignment(image,xmin,xmax):
    
    avg=math.ceil((xmin+xmax)/2)
    w,h=image.size
    if (avg/w)*100 >=0 and (avg/w)*100 <45:
        return "Left"
    elif (avg/w)*100 >=45 and (avg/w)*100 <55:
        return "Center"
    else:
        return "Right"

"""[Returns the distance between 2 points  [ used for difference in RGB for color detecion ]]

Returns:
    [ndarray] 
"""
def get_distance(v1, v2):
    return np.sqrt(np.sum((v1 - v2) ** 2))

"""[Returns the text color of the object [ mainly textboxes and rich textboxes ]]

Returns:
    [text] 
"""
def get_colors(image,coords):
    cropped_image = image.crop(coords)
    # get 2 dominant colors
    q = cropped_image.quantize(colors=2,method=2)
    dominant_color= q.getpalette()[3:6]
    
    colors={
        "Attention":[(255,0,0),(180, 8, 0),(220, 54, 45), (194, 25, 18),(143, 7, 0)],
        "Accent":[(0,0,255),(7, 47, 95),(18, 97, 160),(56, 149, 211)],
        "Good":[(0,128,0),(145,255,0),(30, 86, 49),(164, 222, 2),(118, 186, 27),(76, 154, 42),(104, 187, 89)],
        "Dark":[(0,0,0),(76,76,76),(51, 51, 51),(102, 102, 102),(153, 153, 153)],
        "Light":[(255,255,255)],
        "Warning":[(255,255,0),(255,170,0),(184, 134, 11),(218, 165, 32),(234, 186, 61),(234, 162, 33)]
    }
    color='Default'
    found_colors=[]
    distances=[]
    #find the dominant text colors based on the RGB difference
    for key,values in colors.items():
        for value in values:
            distance=get_distance(np.asarray(value),np.asarray(dominant_color))
            if distance<=150:
                found_colors.append(key)
                distances.append(distance)
    #If the color is predicted as LIGHT check for false cases where both dominan colors are White
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

"""[Finds the intersecting bounding boxes]

Returns:
    [Boolean] -- [True if intersects]
"""
def FindPoints(x1, y1, x2, y2,  
               x3, y3, x4, y4,image=None): 
  
    x5 = max(x1, x3) 
    y5 = max(y1, y3) 
    x6 = min(x2, x4) 
    y6 = min(y2, y4) 
  
    if (x5 > x6 or y5 > y6) : 
        return False

    if image:
        return True
    else:
        intersection_area=(x6-x5)*(y6-y5)
        point1_area=(x2-x1)*(y2-y1)
        point2_area=(x4-x3)*(y4-y3)

        if intersection_area/point1_area > 0.55 and intersection_area/point2_area > 0.55:
            return True
        else:
            return False  

"""[Rejects overlapping boundin boxes]
"""
def reject_overlapping(coords,objects):
    overlap_objects=[]

    for i in range(len(coords)):
        for j in range(i+1,len(coords)):
            box1=[float(c) for c  in coords[i].split(",")]
            box2=[float(c) for c  in coords[j].split(",")]
            intersection=FindPoints(box1[0],box1[1],box1[2],box1[3],box2[0],box2[1],box2[2],box2[3])
            if intersection:
                if j< len(objects):
                    del objects[j]
                

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
                    image_pillow1=Image.open(img_path)
                    width,height = image_pillow.size
                    image_np = cv2.imread(img_path)
                    image_np_expanded = np.expand_dims(image_np, axis=0)
                    output_dict = run_inference_for_single_image(image_np, detection_graph,tensor_dict,sess)
                    
                    boxes=output_dict['detection_boxes']
                    scores=output_dict['detection_scores']
                    classes=output_dict['detection_classes']                        
                    r,c=boxes.shape
                    json_object={"file_name":image}
                    detected_coords=[]
                    json_object['objects']=[]
                    all_coords=[]

                    #For detected objects
                    for i in range(r):
                      
                      if scores[i]*100>=85.0:
                        object_json=dict().fromkeys(['object','xmin','ymin','xmax','ymax'],'')
                        if str(classes[i])=="1":
                            object_json['object']="textbox"         
                        elif  str(classes[i])=="2":
                            object_json['object']="radiobutton"
                        elif  str(classes[i])=="3":
                            object_json['object']="checkbox"
                        
                        ymin = boxes[i][0]*height
                        xmin = boxes[i][1]*width
                        ymax = boxes[i][2]*height
                        xmax = boxes[i][3]*width
                        
                        object_json['xmin']=str(xmin)
                        object_json['ymin']=str(ymin)
                        object_json['xmax']=str(xmax)
                        object_json['ymax']=str(ymax)
                        object_json['coords']=','.join([str(xmin),str(ymin),str(xmax),str(ymax)])
                        object_json['score']=scores[i]
                        all_coords.append(object_json['coords'])
                        detected_coords.append((xmin,ymin,xmax,ymax))
                        object_json['text']=get_text(image_pillow,((xmin, ymin, xmax,ymax )))
                        if object_json['object']=="textbox":
                            
                            object_json["size"],object_json['weight']=get_size_and_weight(image_pillow,((xmin, ymin, xmax,ymax )))
                            object_json["horizontal_alignment"]=get_alignment(image_pillow,float(xmin),float(xmax))
                            object_json['color']=get_colors(image_pillow,((xmin, ymin, xmax,ymax )))
                        json_object['objects'].append(object_json)
                    
                    

                    
                    for i in range(len(json_object['objects'])):
                        for j in range(i+1,len(json_object['objects'])):
                          if i<len(json_object['objects']) and j<len(json_object['objects']):
                            coordsi=json_object['objects'][i].get('coords')
                            coordsj=json_object['objects'][j].get('coords')
                            box1=[float(c) for c  in coordsi.split(",")]
                            box2=[float(c) for c  in coordsj.split(",")]
                            intersection=FindPoints(box1[0],box1[1],box1[2],box1[3],box2[0],box2[1],box2[2],box2[3])
                            if intersection:
                                if json_object['objects'][i].get('score')>json_object['objects'][j].get('score'):
                                    del json_object['objects'][j]
                                else:
                                    del json_object['objects'][i]
                    
                    #For image objects
                    images,image_coords=image_detection(image_np,detected_coords,image_pillow1)
                    ctr=0
                    for im in images:
                        
                        coords=image_coords[ctr]
                        coords=(coords[0],coords[1],coords[2],coords[3])
                        object_json=dict().fromkeys(['object','xmin','ymin','xmax','ymax'],'')
                        object_json["object"]="image"
                        object_json["horizontal_alignment"]=get_alignment(image_pillow,float(coords[0]),float(coords[2]))
                        object_json["url"]=im
                        object_json['xmin']=coords[0]
                        object_json['ymin']=coords[1]
                        object_json['xmax']=coords[2]
                        object_json['ymax']=coords[3]
                        object_json['coords']=','.join([str(coords[0]),str(coords[1]),str(coords[2]),str(coords[3])])
                        json_object['objects'].append(object_json)
                        ctr+=1

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
                        min_score_thresh=0.85
                        )

                    return_dict["image"]=base64.b64encode(cv2.imencode('predicted.jpg', image_np)[1]).decode("utf-8")
                   
                    
                    card_json = {"type": "AdaptiveCard", "version": "1.0", "body": [], "$schema": "http://adaptivecards.io/schemas/adaptive-card.json"}
                    body,ymins=get_card_json(json_object.get('objects',[]),len(images))
                    
                    #Vertical Alignment of the  card objects [  based on ymin ]
                    for obj in range(len(ymins)-1,0,-1):
                        for i in range(obj):
                            if float(ymins[i])>float(ymins[i+1]):
                                temp1=ymins[i]
                                temp = body[i]
                                body[i] = body[i+1]
                                ymins[i]=ymins[i+1]
                                body[i+1] = temp
                                ymins[i+1] = temp1
                    card_json["body"]=body

                    if os.path.exists("temp_image.png"):
                        os.remove('temp_image.png')

                    return_dict["card_json"]=card_json
                    return_dict['coords']=all_coords
                    print(json.dumps(return_dict))
               

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate Objectss')
    parser.add_argument('--image_path',required=True,help='Enter Test Image Path or Test Images Folder Path')

    args = parser.parse_args()
    main(args.image_path)


