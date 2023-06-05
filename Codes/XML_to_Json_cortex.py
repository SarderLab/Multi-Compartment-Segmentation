# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 14:42:57 2020

@author: avina
"""

import json
import xml.etree.ElementTree as ET
import argparse
import glob


def convert_xml_json(root, groupByName,names):

    colorList = ["rgb(0, 255, 0)", "rgb(0, 255,255)", "rgb(255, 255, 0)",  "rgb(255, 0, 0)","rgb(0, 0, 255)",
                 "rgb(255, 128, 0)", "rgb(0, 102, 0)", "rgb(153, 0, 0)", "rgb(0, 153, 0)", "rgb(102, 0, 204)",
                 "rgb(76, 216, 23)", "rgb(102, 51, 0)", "rgb(128, 128, 128)", "rgb(0, 153, 153)", "rgb(0, 0, 0)"]


    l_size = list(root)
    if len(l_size) == 1:
        # print(len(l_size))
        data = dict()
        ann = root.find('Annotation')
        attr = ann.find('Attributes')
        name=names[0]
        element = []
        reg = ann.find('Regions')
        for i in reg.findall('Region'):
            eleDict = dict()
            eleDict["group"] = groupByName
            eleDict["closed"] = True
            eleDict["fillColor"] = "rgba(0, 0, 0, 0)"
            eleDict["lineColor"] = colorList[0]
            eleDict["lineWidth"] = 2
            points = []
            ver = i.find('Vertices')
            if len(ver.findall('Vertex'))<2:
                continue

            for j in ver.findall('Vertex'):
                eachPoint = []
                eachPoint.append(float(j.get('X')))
                eachPoint.append(float(j.get('Y')))
                eachPoint.append(float(j.get('Z')))
                points.append(eachPoint)
            eleDict["points"] = points
            eleDict["type"] = "polyline"
            element.append(eleDict)
        data["elements"] = element
        data["name"] = name

        return data

    elif len(l_size) > 1:
        # print(len(l_size))
        data = []
        for n, child in enumerate(root, start=0):
            dataDict = dict()
            attr = child.find('Attributes')

            #name = attr.find('Attribute').get('Name')
            name=names[n]
            element = []
            reg = child.find('Regions')
            regs = reg.findall('Region')
            
            eleDict = dict()
            eleDict["group"] = groupByName
            eleDict["closed"] = True
            eleDict["fillColor"] = "rgba(0, 0, 0, 0)"
            eleDict["lineColor"] = colorList[n]
            eleDict["lineWidth"] = 2
            points = []
            
            for i in reg.findall('Region'):
                
                ver = i.find('Vertices')
                if len(ver.findall('Vertex'))<2:
                    continue
                for j in ver.findall('Vertex'):
                    eachPoint = []
                    eachPoint.append(float(j.get('X')))
                    eachPoint.append(float(j.get('Y')))
                    eachPoint.append(float(j.get('Z')))
                    points.append(eachPoint)
                eleDict["points"] = points
                eleDict["type"] = "polyline"
                element.append(eleDict)
                
            if len(regs)==0:
                eleDict["points"]=points
                eleDict["type"] = "polyline"
                element.append(eleDict)
            
            dataDict["elements"] = element
            dataDict["name"] = name
            data.append(dataDict)

        return data

    else:
        raise ValueError('Check the format of json file')
#
#
# def main(args):
#     #
#     # read annotation file
#     #
#
#     xmls=glob.glob('*.xml')
#
#     for xml in xmls:
#
#         output_file=xml.split('.xml')[0]+'.json'
#
#         tree = ET.parse(xml)
#         root = tree.getroot()
#
#         #
#         #  convert json to xml
#         #
#         print('Converting xml to json ...', flush=True)
#         annotation = convert_xml_json(root, 'Annotations')
#
#         #
#         # write annotation xml file
#         #
#         print('\n>> Writing xml file ...\n')
#         with open(output_file, 'w') as annotation_file:
#             json.dump(annotation, annotation_file, indent=2, sort_keys=False)
#
# if __name__ == "__main__":
#
#     parser = argparse.ArgumentParser()
#
#     parser.add_argument('--input', dest='input', default=' ' ,type=str,
#         help='input xml path')
#
#     parser.add_argument('--output', dest='output', default=' ' ,type=str,
#         help='output json path')
#
#     parser.add_argument('--groupBy', dest='groupBy', default='Other' ,type=str,
#         help='mention folder name in which annotation belongs')
#
#     args = parser.parse_args()
#     main(args=args)
