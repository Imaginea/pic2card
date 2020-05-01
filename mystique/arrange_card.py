"""Module for arranging the design elements for the Card json"""

from .image_extraction import ImageExtraction
from .extract_properties import ExtractProperties


class CardArrange:

    def remove_overlapping_objects(self, json_object=None):

        """
        Removes the overlapping faster rcnn detected objects by
        finding the intersection between 2 objects.

        @param json_object: list of design objects
        """
        for i in range(len(json_object["objects"])):
            for j in range(i + 1, len(json_object["objects"])):
                if i < len(
                        json_object["objects"]) and j < len(
                        json_object["objects"]):
                    coordsi = json_object["objects"][i].get("coords")
                    coordsj = json_object["objects"][j].get("coords")
                    box1 = [float(c) for c in coordsi.split(",")]
                    box2 = [float(c) for c in coordsj.split(",")]
                    intersection = ImageExtraction().find_points(box1, box2)
                    if intersection:
                        if json_object["objects"][i].get(
                                "score") > json_object["objects"][j].get("score"):
                            del json_object["objects"][j]
                        else:
                            del json_object["objects"][i]

    def append_image_objects(self,image_urls=None,image_coords=None,
                             pil_image=None,json_object=None):

        """
        Appends the extracted image objects to the list of design objects 
        along with its proprties extarcted.

        @param image_urls: list of image object urls
        @param image_coords: list of image object cooridnates
        @param pil_image: input PIL image
        @param json_object: list of design objects
        """

        ctr = 0
        for im in image_urls:
            coords = image_coords[ctr]
            coords = (coords[0], coords[1], coords[2], coords[3])
            object_json = dict().fromkeys(
                ["object", "xmin", "ymin", "xmax", "ymax"], "")
            object_json["object"] = "image"
            object_json["horizontal_alignment"] = ExtractProperties().get_alignment(
                image=pil_image, xmin=float(coords[0]), xmax=float(coords[2]))
            object_json["url"] = im
            object_json["xmin"] = coords[0]
            object_json["ymin"] = coords[1]
            object_json["xmax"] = coords[2]
            object_json["ymax"] = coords[3]
            object_json["coords"] = ",".join(
                [str(coords[0]), str(coords[1]), str(coords[2]), str(coords[3])])
            json_object["objects"].append(object_json)
            ctr += 1

    def group_image_objects(self, image_objects, body, ymins, objects):
        
        """
        Groups the image objects into imagesets which are in 
        closer ymin range.

        @param image_objects: list of image objects
        @param body: list card deisgn elements.
        @param ymins: list of ymins of card design
                      elements
        @param objects: list of all design objects
        """

        # group the image objects based on ymin
        groups = []
        # left_over_images=[]
        unique_ymin = list(set([x.get("ymin") for x in image_objects]))
        for un in unique_ymin:
            temp_list = []
            for xx in image_objects:
                if abs (float (xx.get ("ymin")) - float (un)) <= 10.0:
                    temp_list.append (xx)
            if temp_list not in groups:
                groups.append (temp_list)
        # now put similar ymin grouped objects into a imageset - if a group has
        # more than one image object
        for group in groups:
            group = sorted (group, key=lambda i: i["xmin"])
            if len (group) > 1:
                image_set = {
                    "type": "ImageSet",
                    "imageSize": "medium",
                    "images": []
                    }
                for design_object in group:
                    if design_object in objects:
                        del objects[objects.index (design_object)]
                    obj = {
                        "type": "Image",
                        "altText": "Image",
                        "horizontalAlignment": design_object.get ("horizontal_alignment",""),
                        "url": design_object.get ("url"),
                        }
                    image_set["images"].append (obj)
                body.append (image_set)
                ymins.append (design_object.get ("ymin"))

    def return_position(self, groups, obj):

        """
        Returns the position of an dictionary inside 
        a list of dictionaries

        @param groups: list of dictionaries
        @param obj: dictionary

        @return: position if found else -1
        """
        for i in range(len(groups)):
            if obj in groups[i]:
                return i
        return -1

    def group_choicesets(self, radiobutons, body, ymins=None):

        """
        Groups the choice elements into choicesets based on
        the closer ymin range

        @param radiobuttons: list of individual choice
                             elements
        @param body: list of card deisgn elements
        @param ymins: list of ymin of deisgn elements  
        """
        groups = []
        positions_grouped = []
        for i in range(len(radiobutons)):
            temp_list = []
            for j in range(len(radiobutons)):
                a = float(radiobutons[i].get("ymin"))
                b = float(radiobutons[j].get("ymin"))
                difference_in_ymin = abs(a - b)

                if a > b:
                    difference = float(radiobutons[j].get("ymax")) - a
                else:
                    difference = float(radiobutons[i].get("ymax")) - b
                if abs(difference) <= 10 and difference_in_ymin <= 30 and\
                    j not in positions_grouped:
                    if i in positions_grouped:
                        position = self.return_position(groups, radiobutons[i])
                        if len (groups) > position >= 0:
                            groups[position].append(radiobutons[j])
                            positions_grouped.append (j)
                        elif radiobutons[i] in temp_list:
                            temp_list.append(radiobutons[j])
                            positions_grouped.append (j)
                    else:
                        temp_list = [radiobutons[i], radiobutons[j]]
                        positions_grouped.append(j)
                        positions_grouped.append(i)

            if temp_list:
                flag = False
                for gr in groups:
                    for temp in temp_list:
                        if temp in gr:
                            flag = True
                if not flag:
                    groups.append(temp_list)

        for group in groups:
            group = sorted (group, key=lambda i: i["ymin"])
            choice_set = {
                "type": "Input.ChoiceSet",
                "choices": [],
                "style": "expanded"
                }

            for obj in group:
                choice_set["choices"].append({
                    "title": obj.get("text", ""),
                    "value": "",
                    })

            body.append(choice_set)
            if ymins is not None and len (group) > 0:
                ymins.append (obj.get ("ymin"))

    def append_objects(self, design_object, body, ymins=None, is_column=None):

        """
        Appends the individaul design elements to card body

        @param design_objects: design element to append
        @param body: list of design elements
        @param ymins: list of ymin of design elements
        @param is_column: boolean flag to determine
                          a column element
        """
        if design_object.get("object") == "image":
            body.append ({
                "type": "Image",
                "altText": "Image",
                "horizontalAlignment": design_object.get("horizontal_alignment", ""),
                "url": design_object.get("url"),
                })
            if ymins is not None:
                ymins.append (design_object.get ("ymin"))
        if design_object.get ("object") == "textbox":
            if (len (design_object.get("text", "").split ()) >= 11 and not is_column) or (
                    is_column and len(design_object.get ("text", "")) >= 15):
                body.append ({
                    "type": "RichTextBlock",
                    "inlines": [{
                        "type": "TextRun", "text": design_object.get ("text", ""),
                        "size": design_object.get ("size", ""),
                        "horizontalAlignment": design_object.get ("horizontal_alignment", ""),
                        "color": design_object.get ("color", "Default"),
                        "weight": design_object.get ("weight", "")
                        }]})
                if ymins is not None:
                    ymins.append(design_object.get ("ymin"))
            else:
                body.append({
                    "type": "TextBlock",
                    "text": design_object.get ("text", ""),
                    "size": design_object.get ("size", ""),
                    "horizontalAlignment": design_object.get ("horizontal_alignment", ""),
                    "color": design_object.get ("color", "Default"),
                    "weight": design_object.get ("weight", ""),
                    })
                if ymins is not None:
                    ymins.append(design_object.get ("ymin"))

            if design_object.get("object") == "checkbox":
                body.append({
                    "type": "Input.Toggle",
                    "title": design_object.get ("text", ""),
                    })
                if ymins is not None:
                    ymins.append(design_object.get ("ymin"))

    def build_card_json (self, objects=None):

        """
        Builds the Adaptive card json

        @param objects: list of all design objects

        @return: card body and ymins of deisgn elements
        """
        body = []
        ymins = []
        image_objects = []
        for design_object in objects:
            if design_object.get ("object") == "image":
                image_objects.append (design_object)

        self.group_image_objects(image_objects, body, ymins, objects)

        groups = []
        unique_ymin = list(set([x.get ("ymin") for x in objects]))
        for un in unique_ymin:
            temp_list = []
            for xx in objects:
                if abs(float(xx.get("ymin")) - float(un)) <= 11.0:
                    flag = 0
                    for gr in groups:
                        if xx in gr:
                            flag = 1
                    if flag == 0:
                        temp_list.append (xx)

            if temp_list not in groups:
                groups.append(temp_list)

        radio_buttons_dict = {"normal": []}
        for group in groups:
            radio_buttons_dict["columnset"] = {}
            if len(group) == 1:

                if group[0].get("object") == "radiobutton":
                    radio_buttons_dict["normal"].append(group[0])
                else:
                    self.append_objects(group[0], body, ymins=ymins)
            elif len(group) > 1:
                colummn_set = {
                    "type": "ColumnSet",
                    "columns": []
                    }
                ctr = 0
                group = sorted (group, key=lambda i: i["xmin"])
                for obj in group:

                    colummn_set["columns"].append({
                        "type": "Column",
                        "width": "stretch",
                        "items": []
                        })
                    position = group.index(obj)
                    if position + 1 < len(group):
                        greater = position
                        lesser = position + 1
                        if float(obj.get("ymin")) < float(
                                group[position + 1].get("ymin")):
                            greater = position + 1
                            lesser = position

                        if abs(float(group[greater].get("xmax")) 
                               -float(group[lesser].get("xmin"))) <= 10:
                            colummn_set["columns"][ctr]["width"] = "auto"

                    if obj.get("object") == "radiobutton":
                        radio_buttons_dict["columnset"] = \
                            radio_buttons_dict["columnset"].fromkeys([ctr], [])
                        radio_buttons_dict["columnset"][ctr].append(obj)

                    else:
                        self.append_objects(
                            obj, colummn_set["columns"][ctr].get(
                                "items", []), is_column=True)

                    ctr += 1

                if len(radio_buttons_dict["columnset"]) > 0:
                    if ctr - 1 != -1 and \
                        ctr - 1 <= len(colummn_set["columns"]) and \
                            len(radio_buttons_dict["columnset"]) > 0:
                        if radio_buttons_dict["columnset"].get(ctr - 1):
                            self.group_choicesets(radio_buttons_dict["columnset"].get(
                                ctr - 1), colummn_set["columns"][ctr - 1].get("items", []))

                if colummn_set not in body:
                    for column in colummn_set["columns"]:
                        if column.get("items", []) == []:
                            del colummn_set["columns"][colummn_set["columns"].index(
                                column)]

                    body.append(colummn_set)
                    ymins.append(group[0].get("ymin", ""))
                    
        if len(radio_buttons_dict["normal"]) > 0:
            self.group_choicesets(
                radio_buttons_dict["normal"], body, ymins=ymins)

        return body, ymins
