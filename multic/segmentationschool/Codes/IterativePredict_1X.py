import cv2
import numpy as np
import os
import json
import sys
import lxml.etree as ET
import glob
from .xml_to_json import convert_xml_json
from tqdm import tqdm
from shutil import rmtree
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

from scipy.ndimage.morphology import binary_fill_holes
from tiffslide import TiffSlide
from skimage.color import rgb2hsv
from skimage.filters import gaussian
import girder_client


NAMES = ['cortical_interstitium','medullary_interstitium','non_globally_sclerotic_glomeruli','globally_sclerotic_glomeruli','tubules','arteries/arterioles']
XML_COLOR = [65280, 16776960,65535, 255, 16711680, 33023]
TITLE = 'Multi Compartment Segmentation on FUSION_MCS_FFPE_v1.pth'

def get_user_id(gc):
    try:
        user = gc.get("/user/me")
        if not user:
            token_info = gc.get("/token/current")
            if token_info and "userId" in token_info:
                return token_info["userId"]
            else:
                print("Unable to retrieve user ID from token.")
                return None
        return user["_id"]
    except girder_client.HttpError as e:
        print(f"Authentication failed: {e}")
        return None

def get_user_info(gc, id):
    try:
        user = gc.get(f'/user/{id}')
        return user
    except girder_client.HttpError as e:
        print(f"Failed to retrieve user info: {e}")
        return None

def get_user_running_jobs(gc, user_id):
    try:
        jobs = gc.get("job", parameters={
            "userId": user_id,
            "handlers": '["celery_handler"]',
            "statuses": '[2]'
        })
        assert len(jobs) > 0, "No running jobs found for user."
        return jobs
    except girder_client.HttpError as e:
        print(f"Failed to retrieve running jobs: {e}")
        return []

def get_job(gc, title):
    user_id = get_user_id(gc)
    if not user_id:
        print("No user ID found. Cannot retrieve jobs.")
        return None, None
    user = get_user_info(gc, user_id)

    running_jobs = get_user_running_jobs(gc, user_id)
    for job in running_jobs:
        if job["title"] == title:
            return job, user['login']
    print(f"No running jobs found with title '{title}'.")
    return None, user['login']

"""
Pipeline code to segment regions from WSI

"""


def decode_panoptic(image,segments_info,organType,args):

    detections=np.unique(image)
    detections=detections[detections>-1]

    out=np.zeros_like(image)
    if organType=='liver':
        for ids in segments_info:
            if ids['isthing']:
                out[image==ids['id']]=ids['category_id']+1

            else:
                out[image==ids['id']]=0

    elif organType=='kidney':
        for ids in segments_info:
            if ids['isthing']:
                out[image==ids['id']]=ids['category_id']+3

            else:
                if args.show_interstitium:
                    if ids['category_id'] in [1,2]:
                        out[image==ids['id']]=ids['category_id']



    else:
        print('unsupported organType ')
        print(organType)
        exit()

    return out.astype('uint8')



def predict(args):

    # Get job ID early while job is still in running state
    gc = args.gc
    job, user_login = get_job(gc, TITLE)
    job_id = job['_id'] if job else None
    if job_id:
        print(f"Using job ID: {job_id} for user: {user_login}")

    downsample = int(args.downsampleRateHR**.5)
    region_size = int(args.boxSize*(downsample))
    step = int((region_size-(args.bordercrop*2))*(1-args.overlap_percentHR))

    print('Building network configuration ...\n')

    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"))
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32],[64],[128], [256], [512], [1024]]
    cfg.MODEL.RPN.IN_FEATURES = ['p2', 'p3', 'p4', 'p5','p6','p6']
    # cfg.MODEL.PIXEL_MEAN=[189.409,160.487,193.422]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[.1,.2,0.33, 0.5, 1.0, 2.0, 3.0,5,10]]
    cfg.MODEL.ANCHOR_GENERATOR.ANGLES=[-90,-60,-30,0,30,60,90]
    cfg.DATALOADER.NUM_WORKERS = 10
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS=False
    if not args.Mag20X:
        cfg.INPUT.MIN_SIZE_TEST=region_size
        cfg.INPUT.MAX_SIZE_TEST=region_size
    else:
        cfg.INPUT.MIN_SIZE_TEST=int(region_size/2)
        cfg.INPUT.MAX_SIZE_TEST=int(region_size/2)
    cfg.MODEL.WEIGHTS = args.modelfile

    tc=['G','SG','T','A']
    sc=['Ob','C','M','B']
    classNum=len(tc)+len(sc)-1
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(tc)
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES =len(sc)

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.roi_thresh
    predictor = DefaultPredictor(cfg)

    wsi = args.file
    extsplit = os.path.splitext(wsi)
    basename = extsplit[0]
    extname = extsplit[-1]
    print(basename)

    try:
        slide=TiffSlide(wsi)
    except:
        raise Exception(f"The slide cannot be read!!")
        

    if extname=='.scn':
        dim_y=int(slide.properties['openslide.bounds-height'])
        dim_x=int(slide.properties['openslide.bounds-width'])
        offsetx=int(slide.properties['openslide.bounds-x'])
        offsety=int(slide.properties['openslide.bounds-y'])
    else:
        dim_x, dim_y=slide.dimensions
        offsetx=0
        offsety=0

    print(dim_x,dim_y)
    fileID=basename.split('/')
    # dirs['fileID'] = fileID[-1]
    # dirs['extension'] = extname
    # dirs['file_name'] = wsi.split('/')[-1]


    wsiMask = np.zeros([dim_y, dim_x], dtype='uint8')

    index_y=np.array(range(offsety,dim_y+offsety,step))
    index_x=np.array(range(offsetx,dim_x+offsetx,step))
    print('Getting thumbnail mask to identify predictable tissue...')
    fullSize=slide.level_dimensions[0]
    resRatio= args.chop_thumbnail_resolution
    ds_1=fullSize[0]/resRatio
    ds_2=fullSize[1]/resRatio
    thumbIm=np.array(slide.get_thumbnail((ds_1,ds_2)))
    if extname =='.scn':
        xStt=int(offsetx/resRatio)
        xStp=int((offsetx+dim_x)/resRatio)
        yStt=int(offsety/resRatio)
        yStp=int((offsety+dim_y)/resRatio)
        thumbIm=thumbIm[yStt:yStp,xStt:xStp]

    hsv=rgb2hsv(thumbIm)
    g=gaussian(hsv[:,:,1],5)
    binary=(g>0.05).astype('bool')
    binary=binary_fill_holes(binary)

    print('Segmenting tissue ...\n')
    totalpatches=len(index_x)*len(index_y)
    with tqdm(total=totalpatches,unit='image',colour='green',desc='Total WSI progress') as pbar:
        for i,j in coordinate_pairs(index_y,index_x):
    
            yEnd = min(dim_y+offsety,i+region_size)
            xEnd = min(dim_x+offsetx,j+region_size)
            yStart_small = int(np.round((i-offsety)/resRatio))
            yStop_small = int(np.round(((yEnd-offsety))/resRatio))
            xStart_small = int(np.round((j-offsetx)/resRatio))
            xStop_small = int(np.round(((xEnd-offsetx))/resRatio))
            box_total=(xStop_small-xStart_small)*(yStop_small-yStart_small)
            pbar.update(1)
            if np.sum(binary[yStart_small:yStop_small,xStart_small:xStop_small])>(args.white_percent*box_total):

                xLen=xEnd-j
                yLen=yEnd-i

                dxS=j
                dyS=i
                dxE=j+xLen
                dyE=i+yLen
                im=np.array(slide.read_region((dxS,dyS),0,(xLen,yLen)))[:,:,:3]

                panoptic_seg, segments_info = predictor(im)["panoptic_seg"]
                maskpart=decode_panoptic(panoptic_seg.to("cpu").numpy(),segments_info,'kidney',args)
                if dxE != dim_x:
                    maskpart[:,-int(args.bordercrop/2):]=0
                if dyE != dim_y:
                    maskpart[-int(args.bordercrop/2):,:]=0

                if dxS != offsetx:
                    maskpart[:,:int(args.bordercrop/2)]=0
                if dyS != offsety:
                    maskpart[:int(args.bordercrop/2),:]=0
                dyE-=offsety
                dyS-=offsety
                dxS-=offsetx
                dxE-=offsetx

                wsiMask[dyS:dyE,dxS:dxE]=np.maximum(maskpart,
                    wsiMask[dyS:dyE,dxS:dxE])
                
                

        slide.close()
        print('\n\nStarting XML construction: ')

        if extname=='.scn':
            print('here writing 1')
            xml_suey(wsiMask=wsiMask, args=args, classNum=classNum, downsample=downsample,glob_offset=[offsetx,offsety], job_id=job_id, user_login=user_login)
        else:
            print('here writing 2')
            xml_suey(wsiMask=wsiMask, args=args, classNum=classNum, downsample=downsample,glob_offset=[0,0], job_id=job_id, user_login=user_login)

def predict_notebook(args):

    downsample = int(args.downsampleRateHR**.5)
    region_size = int(args.boxSize*(downsample))
    step = int((region_size-(args.bordercrop*2))*(1-args.overlap_percentHR))

    print('Building network configuration ...\n')

    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"))
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32],[64],[128], [256], [512], [1024]]
    cfg.MODEL.RPN.IN_FEATURES = ['p2', 'p3', 'p4', 'p5','p6','p6']
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[.1,.2,0.33, 0.5, 1.0, 2.0, 3.0,5,10]]
    cfg.MODEL.ANCHOR_GENERATOR.ANGLES=[-90,-60,-30,0,30,60,90]
    cfg.DATALOADER.NUM_WORKERS = 10
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS=False
    if not args.Mag20X:
        cfg.INPUT.MIN_SIZE_TEST=region_size
        cfg.INPUT.MAX_SIZE_TEST=region_size
    else:
        cfg.INPUT.MIN_SIZE_TEST=int(region_size/2)
        cfg.INPUT.MAX_SIZE_TEST=int(region_size/2)
    cfg.MODEL.WEIGHTS = args.modelfile

    tc=['G','SG','T','A']
    sc=['Ob','C','M','B']
    classNum=len(tc)+len(sc)-1
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(tc)
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES =len(sc)

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.roi_thresh
    import torch
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = DefaultPredictor(cfg)
    device = cfg.MODEL.DEVICE

    wsi = args.file
    extsplit = os.path.splitext(wsi)
    basename = extsplit[0]
    extname = extsplit[-1]
    print(basename)

    try:
        slide=TiffSlide(wsi)
    except:
        raise Exception(f"The slide cannot be read!!")

    if extname=='.scn':
        dim_y=int(slide.properties['openslide.bounds-height'])
        dim_x=int(slide.properties['openslide.bounds-width'])
        offsetx=int(slide.properties['openslide.bounds-x'])
        offsety=int(slide.properties['openslide.bounds-y'])
    else:
        dim_x, dim_y=slide.dimensions
        offsetx=0
        offsety=0

    print(dim_x,dim_y)

    wsiMask = np.zeros([dim_y, dim_x], dtype='uint8')

    index_y=np.array(range(offsety,dim_y+offsety,step))
    index_x=np.array(range(offsetx,dim_x+offsetx,step))
    print('Getting thumbnail mask to identify predictable tissue...')
    fullSize=slide.level_dimensions[0]
    resRatio= args.chop_thumbnail_resolution
    ds_1=fullSize[0]/resRatio
    ds_2=fullSize[1]/resRatio
    thumbIm=np.array(slide.get_thumbnail((ds_1,ds_2)))
    if extname =='.scn':
        xStt=int(offsetx/resRatio)
        xStp=int((offsetx+dim_x)/resRatio)
        yStt=int(offsety/resRatio)
        yStp=int((offsety+dim_y)/resRatio)
        thumbIm=thumbIm[yStt:yStp,xStt:xStp]

    hsv=rgb2hsv(thumbIm)
    g=gaussian(hsv[:,:,1],5)
    binary=(g>0.05).astype('bool')
    binary=binary_fill_holes(binary)

    print(f'Segmenting tissue on {device} ...\n')
    totalpatches=len(index_x)*len(index_y)
    with tqdm(total=totalpatches,unit='image',colour='green',desc='Total WSI progress') as pbar:
        for i,j in coordinate_pairs(index_y,index_x):

            yEnd = min(dim_y+offsety,i+region_size)
            xEnd = min(dim_x+offsetx,j+region_size)
            yStart_small = int(np.round((i-offsety)/resRatio))
            yStop_small = int(np.round(((yEnd-offsety))/resRatio))
            xStart_small = int(np.round((j-offsetx)/resRatio))
            xStop_small = int(np.round(((xEnd-offsetx))/resRatio))
            box_total=(xStop_small-xStart_small)*(yStop_small-yStart_small)
            pbar.update(1)
            if np.sum(binary[yStart_small:yStop_small,xStart_small:xStop_small])>(args.white_percent*box_total):

                xLen=xEnd-j
                yLen=yEnd-i

                dxS=j
                dyS=i
                dxE=j+xLen
                dyE=i+yLen
                im=np.array(slide.read_region((dxS,dyS),0,(xLen,yLen)))[:,:,:3]

                panoptic_seg, segments_info = predictor(im)["panoptic_seg"]
                maskpart=decode_panoptic(panoptic_seg.to(device).numpy(),segments_info,'kidney',args)
                if dxE != dim_x:
                    maskpart[:,-int(args.bordercrop/2):]=0
                if dyE != dim_y:
                    maskpart[-int(args.bordercrop/2):,:]=0

                if dxS != offsetx:
                    maskpart[:,:int(args.bordercrop/2)]=0
                if dyS != offsety:
                    maskpart[:int(args.bordercrop/2),:]=0
                dyE-=offsety
                dyS-=offsety
                dxS-=offsetx
                dxE-=offsetx

                wsiMask[dyS:dyE,dxS:dxE]=np.maximum(maskpart,
                    wsiMask[dyS:dyE,dxS:dxE])

        slide.close()
        print('\n\nStarting XML construction: ')

        if extname=='.scn':
            xml_suey_local(wsiMask=wsiMask, args=args, classNum=classNum, downsample=downsample, glob_offset=[offsetx,offsety])
        else:
            xml_suey_local(wsiMask=wsiMask, args=args, classNum=classNum, downsample=downsample, glob_offset=[0,0])


def xml_suey_local(wsiMask, args, classNum, downsample, glob_offset):
    Annotations = xml_create()
    for i in range(classNum)[1:]:
        Annotations = xml_add_annotation(Annotations=Annotations, annotationID=i)

    unique_mask = []
    for i in range(0, len(wsiMask), 7000):
        unique_mask.extend(np.unique(wsiMask[i:i + 7000]))

    for value in np.unique(unique_mask)[1:]:
        print('\t working on: annotationID ' + str(value))
        binary_mask = np.zeros(np.shape(wsiMask), dtype='uint8')
        binary_mask[wsiMask == value] = 1
        pointsList = get_contour_points(binary_mask, args=args, downsample=downsample, value=value, offset={'X': glob_offset[0], 'Y': glob_offset[1]})
        for i in range(len(pointsList)):
            pointList = pointsList[i]
            Annotations = xml_add_region(Annotations=Annotations, pointList=pointList, annotationID=value)

    annots = convert_xml_json(Annotations, NAMES)

    output_dir = os.path.dirname(args.output_path)
    base_prefix = os.path.splitext(os.path.basename(args.output_path))[0].replace('_annotations', '')

    for annot in annots:
        if not annot.get('elements'):
            print(f'Skipping empty annotation: {annot.get("name")}')
            continue
        class_name = annot['name'].replace('/', '_')
        out_file = os.path.join(output_dir, f'{base_prefix}_{class_name}.json')
        if os.path.exists(out_file):
            os.remove(out_file)
            print(f'Removed existing annotation file: {out_file}')
        with open(out_file, 'w') as f:
            json.dump({"annotation": annot}, f)
        print(f'Annotations saved to: {out_file}')


def coordinate_pairs(v1,v2):
    for i in v1:
        for j in v2:
            yield i,j

def restart_line(): # for printing chopped image labels in command line
    sys.stdout.write('\r')
    sys.stdout.flush()

def xml_suey(wsiMask, args, classNum, downsample,glob_offset, job_id=None, user_login=None):
    # make xml
    Annotations = xml_create()
    # add annotation
    for i in range(classNum)[1:]: # exclude background class
        Annotations = xml_add_annotation(Annotations=Annotations, annotationID=i)

    unique_mask = []
    for i in range(0, len(wsiMask), 7000):
        unique_mask.extend(np.unique(wsiMask[i:i + 7000]))

    for value in np.unique(unique_mask)[1:]:
        # print output
        print('\t working on: annotationID ' + str(value))
        # get only 1 class binary mask
        binary_mask = np.zeros(np.shape(wsiMask),dtype='uint8')
        binary_mask[wsiMask == value] = 1

        # add mask to xml
        pointsList = get_contour_points(binary_mask, args=args, downsample=downsample,value=value,offset={'X':glob_offset[0],'Y':glob_offset[1]})
        for i in range(len(pointsList)):
            pointList = pointsList[i]
            Annotations = xml_add_region(Annotations=Annotations, pointList=pointList, annotationID=value)
    gc = args.gc

    attributes = {
        "job_id": job_id,
        "plugin": TITLE,
        "user": user_login if user_login else "system"
    }

    annots = convert_xml_json(Annotations, NAMES)
    for annot in annots:
        if not annot.get('elements'):
            print(f'Skipping empty annotation: {annot.get("name")}')
            continue
        annot['attributes'] = attributes
        _ = gc.post(path='annotation',parameters={'itemId':args.item_id}, data = json.dumps(annot))
        print('uploading layers')
    print('annotation uploaded...\n')

def get_contour_points(mask, args, downsample,value, offset={'X': 0,'Y': 0}):
    # returns a dict pointList with point 'X' and 'Y' values
    # input greyscale binary image
    maskPoints, contours = cv2.findContours(np.array(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    pointsList = []

    for j in np.array(range(len(maskPoints))):
        if len(maskPoints[j])>2:
            if cv2.contourArea(maskPoints[j]) > args.min_size[value-1]:
                pointList = []
                for i in np.array(range(0,len(maskPoints[j]),4)):
                    point = {'X': (maskPoints[j][i][0][0] * downsample) + offset['X'], 'Y': (maskPoints[j][i][0][1] * downsample) + offset['Y']}
                    pointList.append(point)
                pointsList.append(pointList)
    return pointsList

### functions for building an xml tree of annotations ###
def xml_create(): # create new xml tree
    # create new xml Tree - Annotations
    Annotations = ET.Element('Annotations')
    return Annotations

def xml_add_annotation(Annotations, annotationID=None): # add new annotation
    # add new Annotation to Annotations
    # defualts to new annotationID
    if annotationID == None: # not specified
        annotationID = len(Annotations.findall('Annotation')) + 1
    if annotationID in [1,2]:
        Annotation = ET.SubElement(Annotations, 'Annotation', attrib={'Type': '4', 'Visible': '0', 'ReadOnly': '0', 'Incremental': '0', 'LineColorReadOnly': '0', 'LineColor': str(XML_COLOR[annotationID-1]), 'Id': str(annotationID), 'NameReadOnly': '0'})
    else:
        Annotation = ET.SubElement(Annotations, 'Annotation', attrib={'Type': '4', 'Visible': '1', 'ReadOnly': '0', 'Incremental': '0', 'LineColorReadOnly': '0', 'LineColor': str(XML_COLOR[annotationID-1]), 'Id': str(annotationID), 'NameReadOnly': '0'})
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
