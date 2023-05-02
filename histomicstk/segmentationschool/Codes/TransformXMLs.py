import os,cv2 # sys, time
import numpy as np
# import matplotlib.pyplot as plt
import lxml.etree as ET
# from matplotlib import path
import glob
from .xml_to_mask_minmax import get_annotated_ROIs,xml_to_mask,write_minmax_to_xml
import openslide
from .XML_to_Json_cortex import convert_xml_json
import json
# import copy
# from tqdm import tqdm

def transform_XMLs(args):
    xml_color = [65280, 65535, 255, 16711680, 33023]
    assert args.target is not None, 'A directory of xmls must be specified for xml transformation! use --target /path/to/xmls'
    assert args.classNum != 0, 'Please provide the number of classes for XML transformation'
    annotatedXMLs=glob.glob(os.path.join(args.target, "*.xml"))
    for xml in annotatedXMLs:
        print(xml)
        write_minmax_to_xml(xml)
        boxes=get_annotated_ROIs(xml,(0,0),'full',['9'])
        for box in boxes:
            box=np.array(box['regionVerts'])

            xMin=np.min(box[:,0])
            xMax=np.max(box[:,0])
            yMin=np.min(box[:,1])
            yMax=np.max(box[:,1])

            # print(xMin,xMax,yMin,yMax)
            xmlpart=xml_to_mask(xml,(0,0),'full',ignore_id=['9'])
            xmlpart=xmlpart[yMin:yMax,xMin:xMax]
            xmloutname=xml.replace('.xml','_'.join(['',str(xMin),str(yMin),str(xMax-xMin),str(yMax-yMin)]))+'.xml'
            print(xmloutname)
            # plt.imshow(xmlpart)
            # plt.show()
            xml_suey(xmlpart,xmloutname,args.classNum,downsample=args.downsampleRate,glob_offset=[0,0],xml_color=xml_color)

def splice_cortex_XMLs(args):
    xml_color = [65280,65535,16776960,255, 16711680, 33023]
    assert args.target is not None, 'You must provide the directory of XMLs to splice cortex into with --target /path/to/xmls'
    assert args.cortextarget is not None, 'You must provide the directory with cortex XMLs with --cortextarget /path/to/xmls'
    assert args.output is not None, 'You must provide the directory for output XMLS with --output /path/to/save/location'
    baseXMLs=glob.glob(os.path.join(args.target, "*.xml"))
    corteXMLs=glob.glob(os.path.join(args.cortextarget, "*.xml"))
    output_path=args.output
    if args.groupBy is None:
        groupAnnotationsBy='Annotations'
    else:
        groupAnnotationsBy=args.groupBy
    if not os.path.isdir(args.output):
        print('Creating output folder: ' + args.output)
        os.makedirs(args.output)

    for xml in baseXMLs:
        print(xml,end='\n',flush=True)
        cortexxml=os.path.join(args.cortextarget,xml.split('/')[-1])
        newxmlpath=os.path.join(args.output,xml.split('/')[-1])
        try:
            write_minmax_to_xml(cortexxml)
        except Exception as e:
            print(e)
            exit()
        write_minmax_to_xml(xml)
        basetree = ET.parse(xml)
        baseroot = basetree.getroot()
        cortextree = ET.parse(cortexxml)
        cortexroot = cortextree.getroot()
        # print('\n')
        # print(len(cortexroot))
        # print(cortexroot[0].tag)
        # exit()
        Annotations_new=xml_create()
        for i in range(1,7):
            Annotations_new = xml_add_annotation(Annotations=Annotations_new,xml_color=xml_color,annotationID=i)
        #
        #     if i in [1,2]:
        #         Annotation = cortexroot.find("Annotation[@Id='" + str(i) + "']")
        #         # Annotations_new=ET.SubElement(Annotations_new,ET.tostring(Annotation, pretty_print=True))
        #         Annotations_new.append(Annotation)
        #
        #     else:
        #         Annotation = baseroot.find("Annotation[@Id='" + str(i-1) + "']")
        #
        #         Annotation.attrib['Id']=str(i)
        #          # Annotation.attrib['Id']
        #         # Annotations_new.append(Annotation)
        #         Annotations_new.append(Annotation)
        #         # Annotations_new.append(ET.fromstring(ET.tostring(Annotation)))
        for Annotation in cortexroot.findall("./Annotation"):
            annotationID = Annotation.attrib['Id']
            # print(annotationID)
            # Annotations_new.append(Annotation)
            for Region in Annotation.findall("./*/Region"): # iterate on all region
                # Annotation = Annotations.find("Annotation[@Id='" + str(annotationID) + "']")
                # Regions = Annotation.find('Regions')
                # Annotations_new = ET.SubElement(Annotations=Annotations_new, pointList=np.array(verts), annotationID=int(annotationID))
                verts=[]
                for Vert in Region.findall("./Vertices/Vertex"): # iterate on all vertex in region
                    verts.append({'X':int(float(Vert.attrib['X'])),'Y':int(float(Vert.attrib['Y']))})
                Annotations_new = xml_add_region(Annotations=Annotations_new, pointList=np.array(verts), annotationID=int(annotationID))


        for Annotation in baseroot.findall("./Annotation"):
            annotationID = Annotation.attrib['Id']
            if annotationID in ['1']:
                continue

            Annotation.attrib['Id']=str(int(Annotation.attrib['Id'])+1)
            print(annotationID)
            # Annotations_new.append(Annotation)
            for regionidx,Region in enumerate(Annotation.findall("./*/Region")): # iterate on all region
                verts=[]
                for Vert in Region.findall("./Vertices/Vertex"): # iterate on all vertex in region
                    verts.append({'X':int(float(Vert.attrib['X'])),'Y':int(float(Vert.attrib['Y']))})
                Annotations_new = xml_add_region(Annotations=Annotations_new, pointList=np.array(verts), annotationID=int(annotationID)+1,regionID=regionidx+1)
        xml_data = ET.tostring(Annotations_new, pretty_print=True)
        #xml_data = Annotations.toprettyxml()
        f = open(newxmlpath, 'wb')
        f.write(xml_data)
        f.close()

        # xml_save(Annotations=Annotations_new, filename=newxmlpath)

        # Convert XML to histomicsUI json
        tree = ET.parse(newxmlpath)
        root = tree.getroot()

        # names=['Interstitium','glomerulus','sclerotic glomerulus','tubules','artery/arteriole']
        names=['Cortex','Medulla','glomerulus','sclerotic glomerulus','tubules','artery/arteriole']
        annotation = convert_xml_json(root, groupAnnotationsBy,names)

        print('Convert to HistomicsUI json...',end='\r',flush=True)
        with open(newxmlpath.replace('.xml','.json'), 'w') as annotation_file:
            json.dump(annotation, annotation_file, indent=2, sort_keys=False)
    print('\nDone.')


def register_aperio_scn_xmls(args):
    xml_color = [0,65280, 65535, 255, 16711680, 33023]
    assert args.target is not None, 'You must provide the directory of XMLs and WSIs to register with --target /path/to/xmls'
    assert args.output is not None, 'You must provide the directory for output XMLS with --output /path/to/save/location'

    if args.groupBy is None:
        groupAnnotationsBy='Annotations'
    else:
        groupAnnotationsBy=args.groupBy
    if not os.path.isdir(args.output):
        print('Creating output folder: ' + args.output)
        os.makedirs(args.output)
    annotatedXMLs=glob.glob(os.path.join(args.target, "*.xml"))
    for xml in annotatedXMLs:
        print(xml,end='\r',flush=True)
        newxmlpath=os.path.join(args.output,xml.split('/')[-1])
        try:
            slide=openslide.OpenSlide(xml.replace('.xml','.scn'))
        except Exception as e:
            print(e)
            exit()

        dim_x=int(slide.properties['openslide.bounds-width'])## add to columns
        dim_y=int(slide.properties['openslide.bounds-height'])## add to rows
        offsetx=int(slide.properties['openslide.bounds-x'])##start column
        offsety=int(slide.properties['openslide.bounds-y'])##start row

        rotationoffset=dim_y-dim_x

        write_minmax_to_xml(xml)
        tree = ET.parse(xml)
        root = tree.getroot()
        Annotations_new = xml_create()
        for i in range(1,3):
            Annotations_new = xml_add_annotation(Annotations=Annotations_new,xml_color=xml_color,annotationID=i)
        for Annotation in root.findall("./Annotation"): # for all Annotations_new
            annotationID = Annotation.attrib['Id']

            IDs=[]
            for Region in Annotation.findall("./*/Region"): # iterate on all region
                verts=[]
                # cnt=[]
                for Vert in Region.findall("./Vertices/Vertex"): # iterate on all vertex in region
                    verts.append({'X':int(float(Vert.attrib['Y'])+offsetx),'Y':dim_x-int(float(Vert.attrib['X']))+offsety+rotationoffset})

                Annotations_new = xml_add_region(Annotations=Annotations_new, pointList=np.array(verts), annotationID=int(annotationID))
        xml_save(Annotations=Annotations_new, filename=newxmlpath)

        # Convert XML to histomicsUI json
        tree = ET.parse(newxmlpath)
        root = tree.getroot()

        names=['Cortex','Medulla','other']
        annotation = convert_xml_json(root, groupAnnotationsBy,names)

        print('Convert to HistomicsUI json...',end='\r',flush=True)
        with open(newxmlpath.replace('.xml','.json'), 'w') as annotation_file:
            json.dump(annotation, annotation_file, indent=2, sort_keys=False)
    print('\nDone.')
def xml_suey(wsiMask,xmloutname, classNum, downsample,glob_offset,xml_color):
    # make xml
    Annotations = xml_create()
    # add annotation
    for i in range(classNum)[1:]: # exclude background class
        Annotations = xml_add_annotation(Annotations=Annotations,xml_color=xml_color,annotationID=i)


    for value in np.unique(wsiMask)[1:]:
        # print output
        print('\t working on: annotationID ' + str(value))
        # get only 1 class binary mask
        binary_mask = np.zeros(np.shape(wsiMask)).astype('uint8')
        binary_mask[wsiMask == value] = 1

        # add mask to xml
        pointsList = get_contour_points(binary_mask, downsample=downsample,value=value,offset={'X':glob_offset[0],'Y':glob_offset[1]})
        for i in range(np.shape(pointsList)[0]):
            pointList = pointsList[i]
            Annotations = xml_add_region(Annotations=Annotations, pointList=pointList, annotationID=value)

    # save xml
    print(xmloutname)
    xml_save(Annotations=Annotations, filename=xmloutname)

def get_contour_points(mask, downsample,value, offset={'X': 0,'Y': 0}):
    # returns a dict pointList with point 'X' and 'Y' values
    # input greyscale binary image
    maskPoints, contours = cv2.findContours(np.array(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    pointsList = []
    #maskPoints2=copy.deepcopy(maskPoints)

    for j in np.array(range(np.shape(maskPoints)[0])):
        if len(maskPoints[j])>2:
            pointList = []
            for i in np.array(range(0,np.shape(maskPoints[j])[0],4)):
                point = {'X': (maskPoints[j][i][0][0] * downsample) + offset['X'], 'Y': (maskPoints[j][i][0][1] * downsample) + offset['Y']}
                pointList.append(point)
            pointsList.append(pointList)
    return np.array(pointsList)

### functions for building an xml tree of annotations ###
def xml_create(): # create new xml tree
    # create new xml Tree - Annotations
    Annotations = ET.Element('Annotations')
    return Annotations

def xml_add_annotation(Annotations,xml_color, annotationID=None): # add new annotation
    # add new Annotation to Annotations
    # defualts to new annotationID
    if annotationID == None: # not specified
        annotationID = len(Annotations.findall('Annotation')) + 1
    Annotation = ET.SubElement(Annotations, 'Annotation', attrib={'Type': '4', 'Visible': '1', 'ReadOnly': '0', 'Incremental': '0', 'LineColorReadOnly': '0', 'LineColor': str(xml_color[annotationID-1]), 'Id': str(annotationID), 'NameReadOnly': '0'})
    Regions = ET.SubElement(Annotation, 'Regions')
    return Annotations

def xml_add_region(Annotations, pointList, annotationID=-1, regionID=None): # add new region to annotation
    # add new Region to Annotation
    # defualts to last annotationID and new regionID
    Annotation = Annotations.find("Annotation[@Id='" + str(annotationID) + "']")
    Regions = Annotation.find('Regions')
    if regionID == None: # not specified
        regionID = len(Regions.findall('Region')) + 1
    Region = ET.SubElement(Regions, 'Region', attrib={'NegativeROA': '0', 'ImageFocus': '-1', 'DisplayId': '1', 'InputRegionId': '0', 'Analyze': '0', 'Type': '0', 'Id': str(regionID)})
    Vertices = ET.SubElement(Region, 'Vertices')
    for point in pointList: # add new Vertex
        ET.SubElement(Vertices, 'Vertex', attrib={'X': str(point['X']), 'Y': str(point['Y']), 'Z': '0'})
    # add connecting point
    ET.SubElement(Vertices, 'Vertex', attrib={'X': str(pointList[0]['X']), 'Y': str(pointList[0]['Y']), 'Z': '0'})
    return Annotations

def xml_save(Annotations, filename):
    xml_data = ET.tostring(Annotations, pretty_print=True)
    #xml_data = Annotations.toprettyxml()
    f = open(filename, 'wb')
    f.write(xml_data)
    f.close()

def read_xml(filename):
    # import xml file
    tree = ET.parse(filename)
    root = tree.getroot()
