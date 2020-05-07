"""Module to generate template data payload for the card design payload"""

import json
from typing import Dict, List

class Template:

    
    def int_to_roman(self, number: int):

        """
        Convert Integer to Roman Numeral

        @param number: integer to convert

        @return: converted roman numeral
        """
        values = (1000, 900,  500, 400, 100,  90, 50,  40, 10,  9,   5,  4,   1)
        symbols = ("M",  "CM", "D", "CD","C", "XC","L","XL","X","IX","V","IV","I")
        roman_numeral = ""
        for i in range(len(values)):
            count = int(number / values[i])
            roman_numeral += symbols[i] * count
            number -= values[i] * count
        return roman_numeral

    def generate_object_data_mapping(self, objects: List[Dict]):

        """
        Generate a objects to data mapping with occurance numbers

        @param objects: list of design objects

        @return: dict of design objects data mapping
        """
        template_object_map={}.fromkeys(\
            ["textbox","actionset","checkbox","radiobutton","image"])
        for deisgn_object in objects:
            key=deisgn_object.get("object","")
            if not template_object_map.get(key):
                roman=self.int_to_roman(1)
                template_object_map[key]=[(deisgn_object.get("data"),roman)]
                deisgn_object["data"]="{"+key+"_"+roman+"}"
            else:
                roman=self.int_to_roman(len(template_object_map[key])+1)
                template_object_map[key].append((deisgn_object.get("data"),roman))
                deisgn_object["data"]="{"+key+"_"+roman+"}"
        return template_object_map

    def build_template_data_payload(self, objects: List[Dict]):

        """
        Build the template data payload from the design objects

        @param objects: list of deisgn objects

        @return: template data payload json
        """
        template_object_map=self.generate_object_data_mapping(objects)
        data_payload={}
        for key,value in template_object_map.items():
            if value:
                for data in value:
                    if len(data)>1:
                        data_payload.update(\
                            {key+"_"+data[1]:data[0]})
        return data_payload
    

