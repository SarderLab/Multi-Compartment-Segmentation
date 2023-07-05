import cv2
import numpy as np
import os
import json
import sys
import girder_client
# import argparse
# import multiprocessing
import lxml.etree as ET
# import warnings
# import time
# import copy
# from PIL import Image
import glob
from .xml_to_json import convert_xml_json
# from subprocess import call
# from joblib import Parallel, delayed
# from skimage.io import imread,imsave
# from skimage.segmentation import clear_border
from tqdm import tqdm
# from skimage.transform import resize
from shutil import rmtree
# import matplotlib.pyplot as plt
# from matplotlib import path
# import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
# from detectron2.utils.visualizer import Visualizer
# from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2 import model_zoo
from .get_dataset_list import decode_panoptic
from scipy.ndimage.morphology import binary_fill_holes
# import tifffile as ti
import tiffslide as openslide
# from skimage.morphology import binary_erosion, disk

# import warnings




from skimage.color import rgb2hsv
from skimage.filters import gaussian

# from skimage.segmentation import clear_border

#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
NAMES = ['cortical_interstitium','medullary_interstitium','non_globally_sclerotic_glomeruli','globally_sclerotic_glomeruli','tubules','arteries/arterioles']
# from IterativeTraining import get_num_classes
# from .get_choppable_regions import get_choppable_regions
# from .get_network_performance import get_perf

"""
Pipeline code to segment regions from WSI

"""
# os.environ['CUDA_VISIBLE_DEVICES']='0,1'

# define xml class colormap
xml_color = [65280, 16776960,65535, 255, 16711680, 33023]
def decode_panoptic(image,segments_info,organType,args):
    # plt.imshow(image)
    # plt.show()
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
    # define folder structure dict
    dirs = {'outDir': args.base_dir}
    dirs['txt_save_dir'] = '/txt_files/'
    dirs['img_save_dir'] = '/img_files/'
    dirs['mask_dir'] = '/wsi_mask/'
    dirs['chopped_dir'] = '/originals/'
    dirs['save_outputs'] = args.save_outputs
    dirs['modeldir'] = '/MODELS/'
    dirs['training_data_dir'] = '/TRAINING_data/'
    # find current iteration
    # if args.iteration == 'none':
    #     iteration = get_iteration(args=args)
    # else:
    #     iteration = int(args.iteration)
    downsample = int(args.downsampleRateHR**.5)
    region_size = int(args.boxSize*(downsample))
    step = int((region_size-(args.bordercrop*2))*(1-args.overlap_percentHR))
    # gc = girder_client.GirderClient(apiUrl=args.girderApiUrl)
    # gc.setToken(args.girderToken)
    # project_folder = args.project
    # project_dir_id = project_folder.split('/')[-2]
    #model_file = args.modelfile
    #print(model_file,'here model')
    #model_file_id = model_file .split('/')[-2]
    
    print('Handcoded iteration')

    iteration=1
    print(iteration)
    dirs['xml_save_dir'] = args.base_dir
    #real_path = os.path.realpath(args.project)
    #print(real_path)
    if iteration == 'none':
        print('ERROR: no trained models found \n\tplease use [--option train]')

    else:
        # check main directory exists
        # make_folder(dirs['outDir'])
        # outdir = gc.createFolder(project_directory_id,args.outDir)
        # it = gc.createFolder(outdir['_id'],str(iteration))

        # get all WSIs
        #WSIs = []
        # usable_ext=args.wsi_ext.split(',')
        # for ext in usable_ext:
        #     WSIs.extend(glob.glob(args.project + '/*' + ext))
        #     print('another one')

        # for file in args.files:
        #     print(file)
        #     slidename = file['name']
        #     _ = os.system("printf '\n---\n\nFOUND: [{}]\n'".format(slidename))
        #     WSIs.append(slidename)

        
        # print(len(WSIs), 'number of WSI' )
        print('Building network configuration ...\n')
        #modeldir = args.project + dirs['modeldir'] + str(iteration) + '/HR'

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
        # cfg.MODEL.PANOPTIC_FPN.ENABLED=False
        # cfg.MODEL.PANOPTIC_FPN.INSTANCES_CONFIDENCE_THRESH = args.roi_thresh
        # cfg.MODEL.PANOPTIC_FPN.OVERLAP_THRESH = 1

        predictor = DefaultPredictor(cfg)
        broken_slides=[]
        for wsi in [args.files]:

            # try:

            # except Exception as e:
            #     print('!!! Prediction on ' + wsi + ' failed\n')
            #     print(e)
            # reshape regions calc

            extsplit = os.path.splitext(wsi)
            basename = extsplit[0]
            extname = extsplit[-1]
            print(basename)
            # print(extname)
            # try:
            try:
                slide=openslide.TiffSlide(wsi)
            except:
                broken_slides.append(wsi)
                continue
            # slide = ti.imread(wsi)

            # except:
                # broken_slides.append(wsi)
                # continue
            # continue
            # get image dimensions
            if extname=='.scn':
                dim_y=int(slide.properties['openslide.bounds-height'])
                dim_x=int(slide.properties['openslide.bounds-width'])
                offsetx=int(slide.properties['openslide.bounds-x'])
                offsety=int(slide.properties['openslide.bounds-y'])
                # print(dim_x,dim_y,offsetx,offsety)
            else:
                dim_x, dim_y=slide.dimensions
                offsetx=0
                offsety=0

            print(dim_x,dim_y)
            fileID=basename.split('/')
            dirs['fileID'] = fileID[-1]
            dirs['extension'] = extname
            dirs['file_name'] = wsi.split('/')[-1]


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
                    # yStart_small = int(np.round((i-offsety)/resRatio))
                    # yStop_small = int(np.round(((i-offsety)+args.boxSize)/resRatio))
                    # xStart_small = int(np.round((j-offsetx)/resRatio))
                    # xStop_small = int(np.round(((j-offsetx)+args.boxSize)/resRatio))
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

                        # xmlbuilder.deconstruct(maskpart,dxS-offsetx,dyS-offsety,args)
                        # plt.subplot(121)
                        # plt.imshow(im)
                        # plt.subplot(122)
                        # plt.imshow(maskpart)
                        # plt.show()

                        dyE-=offsety
                        dyS-=offsety
                        dxS-=offsetx
                        dxE-=offsetx

                        wsiMask[dyS:dyE,dxS:dxE]=np.maximum(maskpart,
                            wsiMask[dyS:dyE,dxS:dxE])
                        
                   
                        # wsiMask[dyS:dyE,dxS:dxE]=maskpart

            # print('showing mask')
            # plt.imshow(wsiMask)
            # plt.show()
            slide.close()
            print('\n\nStarting XML construction: ')

            # wsiMask=np.swapaxes(wsiMask,0,1)
            # print('swapped axes')
            # xmlbuilder.sew(args)
            # xmlbuilder.dump_to_xml(args,offsetx,offsety)
            if extname=='.scn':
                print('here writing 1')
                xml_suey(wsiMask=wsiMask, dirs=dirs, args=args, classNum=classNum, downsample=downsample,glob_offset=[offsetx,offsety])
            else:
                print('here writing 2')
                xml_suey(wsiMask=wsiMask, dirs=dirs, args=args, classNum=classNum, downsample=downsample,glob_offset=[0,0])




        print('\n\n\033[92;5mPlease correct the xml annotations found in: \n\t' + dirs['xml_save_dir'])
        print('\nthen place them in: \n\t'+ dirs['training_data_dir'] + str(iteration) + '/')
        print('\nand run [--option train]\033[0m\n')
        print('The following slides were not openable by openslide:')
        print(broken_slides)




def coordinate_pairs(v1,v2):
    for i in v1:
        for j in v2:
            yield i,j
def get_iteration(args):
    currentmodels=os.listdir(args.base_dir)
    if not currentmodels:
        return 'none'
    else:
        currentmodels=list(map(int,currentmodels))
        Iteration=np.max(currentmodels)
        return Iteration

def get_test_model(modeldir):
    pretrains=glob.glob(modeldir + '/*.pth')
    if os.path.isfile(modeldir+'/model_final2.pth'):
        return modeldir+'/model_final.pth'
    else:
        maxmodel=0
        for modelfiles in pretrains:
            modelID=modelfiles.split('.')[0].split('_')[-1]
            print(modelID)
            try:
                modelIDi = int(modelID)
                if modelIDi>maxmodel:
                    maxmodel=modelID
            except: pass
        return ''.join([modeldir,'/model_',maxmodel,'.pth'])

def make_folder(directory):
    print(directory,'predict dir')
    #if not os.path.exists(directory):
    try:
        os.makedirs(directory) # make directory if it does not exit already # make new directory
    except:
        print('folder exists!')

def restart_line(): # for printing chopped image labels in command line
    sys.stdout.write('\r')
    sys.stdout.flush()

def getWsi(path): #imports a WSI
    import openslide
    slide = openslide.TiffSlide(path)
    return slide

def file_len(fname): # get txt file length (number of lines)
    with open(fname) as f:
        for i, l in enumerate(f):
            pass

    if 'i' in locals():
        return i + 1

    else:
        return 0


def xml_suey(wsiMask, dirs, args, classNum, downsample,glob_offset):
    # make xml
    Annotations = xml_create()
    # add annotation
    for i in range(classNum)[1:]: # exclude background class
        Annotations = xml_add_annotation(Annotations=Annotations, annotationID=i)


    for value in np.unique(wsiMask)[1:]:
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

    # save xml
    folder = args.base_dir
    girder_folder_id = folder.split('/')[-2]
    _ = os.system("printf 'Using data from girder_client Folder: {}\n'".format(folder))
    file_name = dirs['file_name']
    print(file_name)
    gc = girder_client.GirderClient(apiUrl=args.girderApiUrl)
    gc.setToken(args.girderToken)
    files = list(gc.listItem(girder_folder_id))
    # dict to link filename to gc id
    item_dict = dict()
    for file in files:
        d = {file['name']:file['_id']}
        item_dict.update(d)
    print(item_dict)
    print(item_dict[file_name])
    annots = convert_xml_json(Annotations, NAMES)
    for annot in annots:
        _ = gc.post(path='annotation',parameters={'itemId':item_dict[file_name]}, data = json.dumps(annot))
        print('uploating layers')
    print('annotation uploaded...\n')



def get_contour_points(mask, args, downsample,value, offset={'X': 0,'Y': 0}):
    # returns a dict pointList with point 'X' and 'Y' values
    # input greyscale binary image
    maskPoints, contours = cv2.findContours(np.array(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    pointsList = []
    #maskPoints2=copy.deepcopy(maskPoints)

    for j in np.array(range(len(maskPoints))):
        if len(maskPoints[j])>2:
            #m=np.squeeze(np.asarray(maskPoints2[j]))
            #xMax=np.max(m[:,1])
            #xMin=np.min(m[:,1])
            #yMax=np.max(m[:,0])
            #yMin=np.min(m[:,0])
            #for point in maskPoints2[j]:
            #    point[0][0]-=yMin
            #    point[0][1]-=xMin

            #mask=np.zeros((xMax-xMin,yMax-yMin))

            #mask=cv2.fillConvexPoly(img=mask,points=maskPoints2[j],color=1)

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
        Annotation = ET.SubElement(Annotations, 'Annotation', attrib={'Type': '4', 'Visible': '0', 'ReadOnly': '0', 'Incremental': '0', 'LineColorReadOnly': '0', 'LineColor': str(xml_color[annotationID-1]), 'Id': str(annotationID), 'NameReadOnly': '0'})
    else:
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

# def xml_save(Annotations, filename):
#     xml_data = ET.tostring(Annotations, pretty_print=True)
#     #xml_data = Annotations.toprettyxml()
#     f = open(filename, 'w')
#     f.write(xml_data.decode())
#     f.close()

# def read_xml(filename):
#     # import xml file
#     tree = ET.parse(filename)
#     root = tree.getroot()
