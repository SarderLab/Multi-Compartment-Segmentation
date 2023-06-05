import cv2
import numpy as np
import os
import sys
import argparse
import multiprocessing
import lxml.etree as ET
import warnings
import time
import copy
from PIL import Image
import glob
from subprocess import call
from joblib import Parallel, delayed
from skimage.io import imread,imsave
from skimage.segmentation import clear_border
from tqdm import tqdm
from skimage.transform import resize
from shutil import rmtree
import matplotlib.pyplot as plt
from matplotlib import path
import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2 import model_zoo
from get_dataset_list import *
from scipy.ndimage.morphology import binary_fill_holes
import tifffile as ti
import tiffslide as openslide
from skimage.morphology import binary_erosion, disk
from scipy.ndimage import zoom
# import czifile as czi
from aicspylibczi import CziFile
import warnings


from skimage.color import rgb2hsv
from skimage.filters import gaussian

from skimage.segmentation import clear_border
sys.path.append(os.getcwd()+'/Codes')

# from IterativeTraining import get_num_classes
from get_choppable_regions import get_choppable_regions
from get_network_performance import get_perf

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
    dirs = {'outDir': args.base_dir + '/' + args.project + args.outDir}
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


    print('Handcoded iteration')
    iteration=1
    print(iteration)
    dirs['xml_save_dir'] = args.base_dir + '/' + args.project + dirs['training_data_dir'] + str(iteration) + '/Predicted_XMLs/'

    if iteration == 'none':
        print('ERROR: no trained models found \n\tplease use [--option train]')

    else:
        # check main directory exists
        make_folder(dirs['outDir'])
        make_folder(dirs['xml_save_dir'])

        # get all WSIs
        WSIs = []
        usable_ext=args.wsi_ext.split(',')
        for ext in usable_ext:
            WSIs.extend(glob.glob(args.base_dir + '/' + args.project + dirs['training_data_dir'] + str(iteration) + '/*' + ext))
        print('Building network configuration ...\n')
        modeldir = args.base_dir + '/' + args.project + dirs['modeldir'] + str(iteration) + '/HR'

        os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"))
        cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32],[64],[128], [256], [512], [1024]]
        cfg.MODEL.RPN.IN_FEATURES = ['p2', 'p3', 'p4', 'p5','p6','p6']
        # cfg.MODEL.PIXEL_MEAN=[189.409,160.487,193.422]
        cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[.1,.2,0.33, 0.5, 1.0, 2.0, 3.0,5,10]]
        cfg.MODEL.ANCHOR_GENERATOR.ANGLES=[-90,-60,-30,0,30,60,90]
        cfg.DATALOADER.NUM_WORKERS = 8#normally at 10
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
        for wsi in WSIs:
            # try:

            # except Exception as e:
            #     print('!!! Prediction on ' + wsi + ' failed\n')
            #     print(e)
            # reshape regions calc
            xml_name = wsi.split('.')[0] + '.' + wsi.split('.')[1] + '.xml'
            print(xml_name)
            if os.path.isfile(xml_name):
                print('uh')
                continue

            extsplit = os.path.splitext(wsi)
            basename = extsplit[0]
            extname = extsplit[-1]
            print(basename)
            # print(extname)
            # try:
            if extname=='.czi':
                slide=CziFile(wsi)
            else:
                slide=openslide.OpenSlide(wsi)
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
            elif extname=='.czi':
                bbox = slide.get_mosaic_bounding_box()
                dim_x = bbox.w
                dim_y = bbox.h
                offsetx=bbox.x
                offsety=bbox.y
            else:
                dim_x,dim_y=slide.dimensions
                offsetx=0
                offsety=0


            fileID=basename.split('/')
            dirs['fileID'] = fileID[-1]


            wsiMask = np.zeros([dim_y, dim_x]).astype(np.uint8)

            index_y=np.array(range(offsety,dim_y+offsety,step))
            index_x=np.array(range(offsetx,dim_x+offsetx,step))
            print('Getting thumbnail mask to identify predictable tissue...')

            if extname=='.czi':
                fullSize = (bbox.h,bbox.w)
                resRatio = 8
                ds_1=fullSize[0]/resRatio
                ds_2=fullSize[1]/resRatio
                thumbIm=np.squeeze(slide.read_mosaic(C=0,Z=0,scale_factor=1./resRatio))
                # del slide
            else:
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

            xmlbuilder=XMLBuilder(dirs['xml_save_dir']+'/'+dirs['fileID']+'.xml',xml_color)

            print('Segmenting tissue ...\n')
            totalpatches=len(index_x)*len(index_y)
            with tqdm(total=totalpatches,unit='image',colour='green',desc='Total WSI progress') as pbar:
                for i,j in coordinate_pairs(index_y,index_x):
            # for i in tqdm(index_y,unit='strip',colour='green',desc='outer y-index iterator'):
            #     for j in tqdm(index_x,leave=False,unit='image',colour='blue',desc='inner x-index iterator'):
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

                        if extname=='.czi':
                            # im = slide[dyS:dyS+yLen,dxS:dxS+xLen]
                            im = np.squeeze(slide.read_mosaic((dxS,dyS,xLen,yLen),C=0,Z=0))
                        else:
                            im=np.array(slide.read_region((dxS,dyS),0,(xLen,yLen)))[:,:,:3]

                        #UPSAMPLE
                        im = zoom(im,(4,4,1),order=1)

                        panoptic_seg, segments_info = predictor(im)["panoptic_seg"]
                        maskpart=decode_panoptic(panoptic_seg.to("cpu").numpy(),segments_info,'kidney',args)
                        outImageName=basename+'_'.join(['',str(dxS),str(dyS)])

                        #DOWNSAMPLE
                        maskpart=zoom(maskpart,(0.25,0.25),order=0)
                        # imsave(outImageName+'_p.png',maskpart)
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
            if extname!='.czi':
                slide.close()
            print('\n\nStarting XML construction: ')

            # wsiMask=np.swapaxes(wsiMask,0,1)
            # print('swapped axes')
            # xmlbuilder.sew(args)
            # xmlbuilder.dump_to_xml(args,offsetx,offsety)
            if extname=='.scn':
                xml_suey(wsiMask=wsiMask, dirs=dirs, args=args, classNum=classNum, downsample=downsample,glob_offset=[offsetx,offsety])
            else:
                xml_suey(wsiMask=wsiMask, dirs=dirs, args=args, classNum=classNum, downsample=downsample,glob_offset=[0,0])




        print('\n\n\033[92;5mPlease correct the xml annotations found in: \n\t' + dirs['xml_save_dir'])
        print('\nthen place them in: \n\t'+ args.base_dir + '/' + args.project + dirs['training_data_dir'] + str(iteration) + '/')
        print('\nand run [--option train]\033[0m\n')
        print('The following slides were not openable by openslide:')
        print(broken_slides)




def coordinate_pairs(v1,v2):
    for i in v1:
        for j in v2:
            yield i,j
def get_iteration(args):
    currentmodels=os.listdir(args.base_dir + '/' + args.project + '/MODELS/')
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
    if not os.path.exists(directory):
        os.makedirs(directory) # make directory if it does not exit already # make new directory

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
        binary_mask = np.zeros(np.shape(wsiMask)).astype('uint8')
        binary_mask[wsiMask == value] = 1

        # add mask to xml
        pointsList = get_contour_points(binary_mask, args=args, downsample=downsample,value=value,offset={'X':glob_offset[0],'Y':glob_offset[1]})
        for i in range(np.shape(pointsList)[0]):
            pointList = pointsList[i]
            Annotations = xml_add_region(Annotations=Annotations, pointList=pointList, annotationID=value)

    # save xml
    print(dirs['xml_save_dir']+'/'+dirs['fileID']+'.xml')
    xml_save(Annotations=Annotations, filename=dirs['xml_save_dir']+'/'+dirs['fileID']+'.xml')

def get_contour_points(mask, args, downsample,value, offset={'X': 0,'Y': 0}):
    # returns a dict pointList with point 'X' and 'Y' values
    # input greyscale binary image
    maskPoints, contours = cv2.findContours(np.array(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    pointsList = []
    #maskPoints2=copy.deepcopy(maskPoints)

    for j in np.array(range(np.shape(maskPoints)[0])):
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


class XMLBuilder():
    def __init__(self,out_file,class_colors):
        self.dump_contours={'1':[],'2':[],'3':[],'4':[],'5':[]}
        self.merge_contours={'1':[],'2':[],'3':[],'4':[],'5':[]}
        self.out_file=out_file
        self.class_colors=class_colors
    def unique_pairs(self,n):
        for i in range(n):
            for j in range(i+1, n):
                yield i, j
    def deconstruct(self,mask,offsetx,offsety,args):
        classes_in_mask=np.unique(mask)
        classes_in_mask=classes_in_mask[classes_in_mask>0]
        for value in classes_in_mask:
            submask=np.array(mask==value).astype('uint8')
            contours, hierarchy = cv2.findContours(submask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
            contours=np.array(contours)
            for contour in contours:
                merge_or_dump=False
                for point in contour:
                    if point[0][0]<15 or point[0][1]<15 or point[0][0]>args.boxSize-15 or point[0][1]>args.boxSize-15:
                        merge_or_dump=True
                        break
                points=np.asarray(contour)
                points[:,0,0]+=offsetx
                points[:,0,1]+=offsety

                if merge_or_dump:
                    self.merge_contours[str(value)].append({'contour':np.squeeze(points,axis=1),'annotationID':value})
                else:
                    self.dump_contours[str(value)].append({'contour':np.squeeze(points,axis=1),'annotationID':value})

    def sew(self,args):
        for cID in range(1,args.classNum):
            print('Merging class... '+ str(cID))

            did_merge=True
            while did_merge:

                did_merge=self.check_and_merge_once(cID)
            print('\n')
    def check_and_merge_once(self,cID):
        contours_at_value=self.merge_contours[str(cID)]
        total=len(contours_at_value)


        print('Total contours... '+ str(total),end='\r')
        for idx1,idx2 in self.unique_pairs(total):
            containPath=path.Path(contours_at_value[idx1]['contour'])
            # print(containPath.contains_points(contour2['contour']))
            ovlpts=containPath.contains_points(contours_at_value[idx2]['contour'])

            if any(ovlpts):
                mergePath=path.Path(contours_at_value[idx2]['contour'])
                merged_verts=np.concatenate((containPath.vertices,mergePath.vertices),axis=0)
                merged_path=path.Path(merged_verts)
                bMinX=np.min(merged_verts[:,1]).astype('int32')
                bMaxX=np.max(merged_verts[:,1]).astype('int32')
                bMinY=np.min(merged_verts[:,0]).astype('int32')
                bMaxY=np.max(merged_verts[:,0]).astype('int32')
                # testim=np.zeros((bMaxX,bMaxY)).astype('uint8')
                # cv2.fillPoly(testim,[np.array(mergePath.vertices).astype('int32')],255)
                # # plt.imshow(testim)
                # # plt.title('mergee')
                # # plt.show()
                # cv2.fillPoly(testim,[np.array(containPath.vertices).astype('int32')],128)
                # plt.imshow(testim)
                # plt.title('merge and contain')
                # plt.show()

                testim=np.zeros((bMaxX-bMinX,bMaxY-bMinY)).astype('uint8')
                testim=np.pad(testim,((0,1),(0,1)))

                #add offsets back
                cvl=[np.array(containPath.vertices).astype('int32')]
                mvl=[np.array(mergePath.vertices).astype('int32')]
                m_dvl=[np.array(merged_path.vertices).astype('int32')]
                cvl[0][:,1]-=bMinX
                cvl[0][:,0]-=bMinY
                mvl[0][:,1]-=bMinX
                mvl[0][:,0]-=bMinY
                m_dvl[0][:,1]-=bMinX
                m_dvl[0][:,0]-=bMinY

                cv2.fillPoly(testim,cvl,1)
                cv2.fillPoly(testim,mvl,1)
                # plt.imshow(testim)
                # plt.title('merged')
                # plt.show()
                contours, hierarchy = cv2.findContours(testim, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
                points=np.asarray(contours[0])

                points[:,0,0]+=bMinY
                points[:,0,1]+=bMinX

                self.merge_contours[str(cID)].pop(idx2)
                self.merge_contours[str(cID)].pop(idx1)
                self.merge_contours[str(cID)].append({'contour':np.squeeze(points,axis=1),'annotationID':cID})

                return True
        return False
                    # testim=np.zeros((bMaxX-bMinX,bMaxY-bMinY)).astype('uint8')
                    # cv2.fillPoly(testim,contours,255)
                    # plt.imshow(testim)
                    # plt.title('merged')
                    # plt.show()


                # input('1')

    def dump_to_xml(self,args,offsetx,offsety):
        # make xml
        self.Annotations = ET.Element('Annotations')

        # add annotation
        for i in range(args.classNum)[1:]: # exclude background class
            print('\t working on: annotationID ' + str(i))
            Annotations = self.xml_add_annotation(annotationID=i)
            # for dump_contour in self.merge_contours[str(i)]:
            #     pointList=dump_contour['contour']
            #     pointList[:,0]+=offsetx
            #     pointList[:,1]+=offsety
            #     self.xml_add_region(pointList=pointList, annotationID=i)
            for dump_contour in self.dump_contours[str(i)]:
                pointList=dump_contour['contour']
                pointList[:,0]+=offsetx
                pointList[:,1]+=offsety
                self.xml_add_region(pointList=pointList, annotationID=i)
        self.xml_save()


    def xml_add_annotation(self, annotationID=None): # add new annotation
        # add new Annotation to Annotations
        # defualts to new annotationID
        if annotationID == None: # not specified
            annotationID = len(self.Annotations.findall('Annotation')) + 1
        Annotation = ET.SubElement(self.Annotations, 'Annotation', attrib={'Type': '4',
            'Visible': '1', 'ReadOnly': '0', 'Incremental': '0', 'LineColorReadOnly': '0',
            'LineColor': str(self.class_colors[annotationID-1]), 'Id': str(annotationID), 'NameReadOnly': '0'})
        Regions = ET.SubElement(Annotation, 'Regions')
        # return Annotations

    def xml_add_region(self,pointList, annotationID=-1, regionID=None): # add new region to annotation
        # add new Region to Annotation
        # defualts to last annotationID and new regionID
        Annotation = self.Annotations.find("Annotation[@Id='" + str(annotationID) + "']")
        Regions = Annotation.find('Regions')
        if regionID == None: # not specified
            regionID = len(Regions.findall('Region')) + 1
        Region = ET.SubElement(Regions, 'Region', attrib={'NegativeROA': '0', 'ImageFocus': '-1', 'DisplayId': '1', 'InputRegionId': '0', 'Analyze': '0', 'Type': '0', 'Id': str(regionID)})
        Vertices = ET.SubElement(Region, 'Vertices')
        for point in pointList: # add new Vertex
            ET.SubElement(Vertices, 'Vertex', attrib={'X': str(point[0]), 'Y': str(point[1]), 'Z': '0'})
        # add connecting point
        ET.SubElement(Vertices, 'Vertex', attrib={'X': str(pointList[0][0]), 'Y': str(pointList[0][1]), 'Z': '0'})
        # return Annotations

    def xml_save(self):
        xml_data = ET.tostring(self.Annotations, pretty_print=True)
        #xml_data = Annotations.toprettyxml()
        print('Writing... ' + self.out_file)
        f = open(self.out_file, 'wb')
        f.write(xml_data)
        f.close()
