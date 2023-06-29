import os, sys, cv2, time, random, warnings, argparse, csv, multiprocessing,json,copy
from skimage.color import rgb2hsv,hsv2rgb,rgb2lab,lab2rgb
import detectron2_custom2.detectron2
from utils import IdGenerator, id2rgb
import numpy as np
import matplotlib.pyplot as plt
import lxml.etree as ET
from matplotlib import path
from skimage.transform import resize
from skimage.io import imread, imsave
import glob
from getWsi import getWsi
from xml_to_mask_minmax import xml_to_mask
from joblib import Parallel, delayed
from shutil import move
# from generateTrainSet import generateDatalists
from subprocess import call
from get_choppable_regions import get_choppable_regions
from PIL import Image
import logging
from detectron2.utils.logger import setup_logger
from skimage import exposure

setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor,DefaultTrainer
from detectron2.config import get_cfg
from detectron2_custom2.detectron2.utils.visualizer import Visualizer,ColorMode
# from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data import detection_utils as utils
import detectron2.data.transforms as T
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data import (DatasetCatalog,
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.config import configurable
from typing import List, Optional, Union
import torch

from wsi_loader_utils import *
from imgaug import augmenters as iaa

global seq
seq = iaa.Sequential([
    iaa.Sometimes(0.5,iaa.OneOf([
        iaa.AddElementwise((-15,15),per_channel=0.5),
        iaa.ImpulseNoise(0.05),iaa.CoarseDropout(0.02, size_percent=0.5)])),
    iaa.Sometimes(0.5,iaa.OneOf([iaa.GaussianBlur(sigma=(0, 3.0)),
        iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 2.0))]))
])

#Record start time
totalStart=time.time()

def IterateTraining(args):
    hsv_aug_rate=args.hsv_aug_prob
    ## calculate low resolution block params
    downsampleLR = int(args.downsampleRateLR**.5) #down sample for each dimension
    region_sizeLR = int(args.boxSizeLR*(downsampleLR)) #Region size before downsampling
    stepLR = int(region_sizeLR*(1-args.overlap_percentLR)) #Step size before downsampling
    ## calculate low resolution block params
    downsample = int(args.downsampleRate**.5) #down sample for each dimension
    region_size = int(args.boxSize*(downsample)) #Region size before downsampling
    step = int(region_size*(1-args.overlap_rate)) #Step size before downsampling

    # global classNum_HR
    dirs = {'imExt': '.jpeg'}
    dirs['basedir'] = args.base_dir
    dirs['maskExt'] = '.png'
    dirs['modeldir'] = '/MODELS/'
    dirs['tempdirLR'] = '/TempLR/'
    dirs['tempdirHR'] = '/TempHR/'
    dirs['pretraindir'] = '/Deeplab_network/'
    dirs['training_data_dir'] = '/TRAINING_data/'
    dirs['model_init'] = 'deeplab_resnet.ckpt'
    dirs['project']= '/' + args.project
    dirs['data_dir_HR'] = args.base_dir +'/' + args.project + '/Permanent/HR/'
    dirs['data_dir_LR'] = args.base_dir +'/' +args.project + '/Permanent/LR/'

    currentmodels=os.listdir(dirs['basedir'] + dirs['project'] + dirs['modeldir'])
    print('Handcoded iteration')
    # currentAnnotationIteration=check_model_generation(dirs)
    currentAnnotationIteration=0
    print('Current training session is: ' + str(currentAnnotationIteration))
    dirs['xml_dir']=dirs['basedir'] + dirs['project'] + dirs['training_data_dir'] + str(currentAnnotationIteration) + '/'
    ##Create objects for storing class distributions
    # annotatedXMLs=glob.glob(dirs['basedir'] + dirs['project'] + dirs['training_data_dir'] + str(currentAnnotationIteration) + '/*.xml')


    # train_dset = WSITrainingLoader(args,dirs['basedir'] + dirs['project'] + dirs['training_data_dir'] + str(currentAnnotationIteration))

    modeldir_HR = dirs['basedir']+dirs['project'] + dirs['modeldir'] + str(currentAnnotationIteration+1) + '/HR/'

    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

    organType='kidney'
    print('Organ meta being set to... '+ organType)

    if organType=='liver':
        classnames=['Background','BD','A']
        isthing=[0,1,1]
        xml_color = [[0,255,0], [0,255,255], [0,0,255]]
        tc=['BD','AT']
        sc=['Ob','B']
    elif organType =='kidney':
        classnames=['interstitium','medulla','glomerulus','sclerotic glomerulus','tubule','arterial tree']
        classes={}
        isthing=[0,0,1,1,1,1]
        xml_color = [[0,255,0], [0,255,255], [255,255,0],[0,0,255], [255,0,0], [0,128,255]]
        tc=['G','SG','T','A']
        sc=['Ob','I','M','B']
    else:
        print('Provided organType not in supported types: kidney, liver')
    rand_sample=True
    json_dir=dirs['basedir']+'/'+dirs['project'] + '/Permanent/HR/'
    json_file=json_dir+'detectron_train'

    classNum=len(tc)+len(sc)-1
    print('Number classes: '+ str(classNum))
    classes={}

    for idx,c in enumerate(classnames):
        classes[idx]={'isthing':isthing[idx],'color':xml_color[idx]}
    IdGen=IdGenerator(classes)

    num_images=args.batch_size*args.train_steps
    # slide_idxs=train_dset.get_random_slide_idx(num_images)
    usable_slides=get_slide_data(args, wsi_directory=dirs['basedir'] + dirs['project'] + dirs['training_data_dir'] + str(currentAnnotationIteration))
    usable_idx=range(0,len(usable_slides))
    slide_idxs=random.choices(usable_idx,k=num_images)
    image_coordinates=get_random_chops(slide_idxs,usable_slides,region_size)
    # usable_slides=[]
    # num_cores=multiprocessing.cpu_count()
    # print('Generating detectron2 dictionary format...')
    # data_list=Parallel(n_jobs=num_cores,backend='threading')(delayed(get_image_meta)(i=i,
    #     train_dset=train_dset,args=args) for i in tqdm(image_coordinates))
    DatasetCatalog.register("my_dataset", lambda:train_samples_from_WSI(args,image_coordinates))
    MetadataCatalog.get("my_dataset").set(thing_classes=tc)
    MetadataCatalog.get("my_dataset").set(stuff_classes=sc)

    if args.check_training_data:
        seg_metadata=MetadataCatalog.get("my_dataset")
        new_list = DatasetCatalog.get("my_dataset")
        total=len(new_list)
        print(total)
        print('Visualizing dataset... spacebar to continue, q to quit')
        for d in tqdm(random.sample(new_list, total)):

            c=d['coordinates']
            h=d['height']
            w=d['width']
            slide=openslide.OpenSlide(d['slide_loc'])
            print(d['slide_loc'])
            x=dirs['xml_dir']+'_'.join(d['image_id'].split('_')[:-2])+'.xml'
            img=np.array(slide.read_region((c[0],c[1]),0,(h,w)))[:,:,:3]
            slide.close()

            if random.random()>hsv_aug_rate:
                # plt.subplot(131)
                # plt.imshow(img)

                hShift=np.random.normal(0,0.05)
                lShift=np.random.normal(1,0.025)
                # imageblock[im]=randomHSVshift(imageblock[im],hShift,lShift)
                img=rgb2hsv(img)
                img[:,:,0]=(img[:,:,0]+hShift)
                img=hsv2rgb(img)
                img=rgb2lab(img)
                img[:,:,0]=exposure.adjust_gamma(img[:,:,0],lShift)
                img=(lab2rgb(img)*255).astype('uint8')
                # plt.subplot(132)
                # plt.imshow(img)

                images_aug = seq(images=[img])[0].squeeze()
            #     plt.subplot(133)
            #     plt.imshow(images_aug)
            #     plt.show()
            #     print(images_aug.dtype)
            # continue
            visualizer = Visualizer(img[:, :, ::-1],metadata=seg_metadata, scale=0.5,idgen=IdGen)
            out = visualizer.draw_dataset_dict(d,x)



            cv2.namedWindow("output", cv2.WINDOW_NORMAL)
            cv2.imshow("output",out.get_image()[:, :, ::-1])
            wait_time = 1000
            while cv2.getWindowProperty('output', cv2.WND_PROP_VISIBLE) >= 1:
                keyCode = cv2.waitKey(wait_time)
                if (keyCode & 0xFF) == ord(" "):
                    cv2.destroyAllWindows()
                    break
                if (keyCode & 0xFF) == ord("q"):
                    cv2.destroyAllWindows()
                    exit()

        exit()

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"))
    cfg.DATASETS.TRAIN = ("my_dataset")
    cfg.DATASETS.TEST = ()
    num_cores = multiprocessing.cpu_count()
    cfg.DATALOADER.NUM_WORKERS = 10

    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml")  # Let training initialize from model zoo

    cfg.MODEL.WEIGHTS = os.path.join('/hdd/bg/Detectron2/HAIL_Detectron2/output_medulla_long1', "model_0214999.pth")
    # cfg.MODEL.WEIGHTS = os.path.join('/hdd/bg/Detectron2/HAIL_Detectron2/outputAugRCC', "model_final.pth")
    # if args.custom_image_means:
    #     x=np.array(train_dset.get_image_means())
    #     total_means=[float(np.round(np.mean(x[:,0]),3)),
    #         float(np.round(np.mean(x[:,1]),3)),
    #         float(np.round(np.mean(x[:,2]),3))]
    #     print('Using custom pixel means: ')
    #     print(total_means)
    #     cfg.MODEL.PIXEL_MEAN=total_means


    cfg.SOLVER.IMS_PER_BATCH = args.batch_size


    # cfg.SOLVER.BASE_LR = 0.0025  # pick a good LR
    # cfg.SOLVER.POLY = 0.9  # pick a good LR
    # cfg.SOLVER.LR_SCHEDULER_NAME = "ExponentialParamScheduler"


    cfg.SOLVER.LR_policy='steps_with_lrs'
    cfg.SOLVER.MAX_ITER = args.train_steps
    # cfg.SOLVER.STEPS = [int(.5*args.train_steps),int(.75*args.train_steps),int(.9*args.train_steps)]
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.LRS = [0.000025,0.0000025]
    cfg.SOLVER.STEPS = [100000,150000]
    # cfg.SOLVER.STEPS = [int(.3333*args.train_steps),int(.6666*args.train_steps),int(.85*args.train_steps)]
    # cfg.SOLVER.LRS = [0.00025,0.000025,0.0000025]
    # cfg.SOLVER.STEPS = [int(args.train_steps/2)]
    # cfg.SOLVER.LRS = [0.00025]

    # cfg.MODEL.BACKBONE.FREEZE_AT = 0
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32],[64],[128], [256], [512], [1024]]
    cfg.MODEL.RPN.IN_FEATURES = ['p2', 'p3', 'p4', 'p5','p6','p6']

    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[.1,.2,0.33, 0.5, 1.0, 2.0, 3.0,5,10]]
    cfg.MODEL.ANCHOR_GENERATOR.ANGLES=[-90,-60,-30,0,30,60,90]

    # cfg.MODEL.RPN.POSITIVE_FRACTION = 0.5

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(tc)
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES =len(sc)


    # cfg.INPUT.CROP.ENABLED = True
    # cfg.INPUT.CROP.TYPE='absolute'
    # cfg.INPUT.CROP.SIZE=[64,64]

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64

    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS=False
    cfg.INPUT.MIN_SIZE_TRAIN=args.boxSize

    cfg.INPUT.MAX_SIZE_TRAIN=args.boxSize

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    with open(cfg.OUTPUT_DIR+"/config_record.yaml", "w") as f:
        f.write(cfg.dump())   # save config to file
    trainer = Trainer(cfg)

    trainer.resume_or_load(resume=False)

    trainer.train()


    finish_model_generation(dirs,currentAnnotationIteration)

    print('\n\n\033[92;5mPlease place new wsi file(s) in: \n\t' + dirs['basedir'] + dirs['project']+ dirs['training_data_dir'] + str(currentAnnotationIteration+1))
    print('\nthen run [--option predict]\033[0m\n')





def check_model_generation(dirs):
    modelsCurrent=os.listdir(dirs['basedir'] + dirs['project'] + dirs['modeldir'])
    gens=map(int,modelsCurrent)
    modelOrder=np.sort(gens)[::-1]

    for idx in modelOrder:
        #modelsChkptsLR=glob.glob(dirs['basedir'] + dirs['project'] + dirs['modeldir']+str(modelsCurrent[idx]) + '/LR/*.ckpt*')
        modelsChkptsHR=glob.glob(dirs['basedir'] + dirs['project'] + dirs['modeldir']+ str(idx) +'/HR/*.ckpt*')
        if modelsChkptsHR == []:
            continue
        else:
            return idx
            break

def finish_model_generation(dirs,currentAnnotationIteration):
    make_folder(dirs['basedir'] + dirs['project'] + dirs['training_data_dir'] + str(currentAnnotationIteration + 1))

def get_pretrain(currentAnnotationIteration,res,dirs):

    if currentAnnotationIteration==0:
        pretrain_file = glob.glob(dirs['basedir']+dirs['project'] + dirs['modeldir'] + str(currentAnnotationIteration) + res + '*')
        pretrain_file=pretrain_file[0].split('.')[0] + '.' + pretrain_file[0].split('.')[1]

    else:
        pretrains=glob.glob(dirs['basedir']+dirs['project'] + dirs['modeldir'] + str(currentAnnotationIteration) + res + 'model*')
        maxmodel=0
        for modelfiles in pretrains:
            modelID=modelfiles.split('.')[-2].split('-')[1]
            if int(modelID)>maxmodel:
                maxmodel=int(modelID)
        pretrain_file=dirs['basedir']+dirs['project'] + dirs['modeldir'] + str(currentAnnotationIteration) + res + 'model.ckpt-' + str(maxmodel)
    return pretrain_file

def restart_line(): # for printing chopped image labels in command line
    sys.stdout.write('\r')
    sys.stdout.flush()

def file_len(fname): # get txt file length (number of lines)
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def make_folder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory) # make directory if it does not exit already # make new directory # Check if folder exists, if not make it

def make_all_folders(dirs):


    make_folder(dirs['basedir'] +dirs['project']+ dirs['tempdirLR'] + '/regions')
    make_folder(dirs['basedir'] +dirs['project']+ dirs['tempdirLR'] + '/masks')

    make_folder(dirs['basedir'] +dirs['project']+ dirs['tempdirLR'] + '/Augment' +'/regions')
    make_folder(dirs['basedir'] +dirs['project']+ dirs['tempdirLR'] + '/Augment' +'/masks')

    make_folder(dirs['basedir']+dirs['project'] + dirs['tempdirHR'] + '/regions')
    make_folder(dirs['basedir'] +dirs['project']+ dirs['tempdirHR'] + '/masks')

    make_folder(dirs['basedir']+dirs['project'] + dirs['tempdirHR'] + '/Augment' +'/regions')
    make_folder(dirs['basedir']+dirs['project']+ dirs['tempdirHR'] + '/Augment' +'/masks')

    make_folder(dirs['basedir'] +dirs['project']+ dirs['modeldir'])
    make_folder(dirs['basedir'] +dirs['project']+ dirs['training_data_dir'])


    make_folder(dirs['basedir'] +dirs['project']+ '/Permanent' +'/LR/'+ 'regions/')
    make_folder(dirs['basedir'] +dirs['project']+ '/Permanent' +'/LR/'+ 'masks/')
    make_folder(dirs['basedir'] +dirs['project']+ '/Permanent' +'/HR/'+ 'regions/')
    make_folder(dirs['basedir'] +dirs['project']+ '/Permanent' +'/HR/'+ 'masks/')

    make_folder(dirs['basedir'] +dirs['project']+ dirs['training_data_dir'])

    make_folder(dirs['basedir'] + '/Codes/Deeplab_network/datasetLR')
    make_folder(dirs['basedir'] + '/Codes/Deeplab_network/datasetHR')

def return_region(args, wsi_mask, wsiID, fileID, yStart, xStart, idxy, idxx, downsampleRate, outdirT, region_size, dirs, chop_regions,classNum_HR): # perform cutting in parallel
    sys.stdout.write('   <'+str(xStart)+'/'+ str(yStart)+'/'+str(chop_regions[idxy,idxx] != 0)+ '>   ')
    sys.stdout.flush()
    restart_line()

    if chop_regions[idxy,idxx] != 0:

        uniqID=fileID + str(yStart) + str(xStart)
        if wsiID.split('.')[-1] != 'tif':
            slide=getWsi(wsiID)
            Im=np.array(slide.read_region((xStart,yStart),0,(region_size,region_size)))
            Im=Im[:,:,:3]
        else:
            yEnd = yStart + region_size
            xEnd = xStart + region_size
            Im = np.zeros([region_size,region_size,3], dtype=np.uint8)
            Im_ = imread(wsiID)[yStart:yEnd, xStart:xEnd, :3]
            Im[0:Im_.shape[0], 0:Im_.shape[1], :] = Im_

        mask_annotation=wsi_mask[yStart:yStart+region_size,xStart:xStart+region_size]

        o1,o2=mask_annotation.shape
        if o1 !=region_size:
            mask_annotation=np.pad(mask_annotation,((0,region_size-o1),(0,0)),mode='constant')
        if o2 !=region_size:
            mask_annotation=np.pad(mask_annotation,((0,0),(0,region_size-o2)),mode='constant')

        '''
        if 4 in np.unique(mask_annotation):
            plt.subplot(121)
            plt.imshow(mask_annotation*20)
            plt.subplot(122)
            plt.imshow(Im)
            pt=[xStart,yStart]
            plt.title(pt)
            plt.show()
        '''
        if downsampleRate !=1:
            c=(Im.shape)
            s1=int(c[0]/(downsampleRate**.5))
            s2=int(c[1]/(downsampleRate**.5))
            Im=resize(Im,(s1,s2),mode='reflect')
        mask_out_name=dirs['basedir']+dirs['project'] + '/Permanent/HR/masks/'+uniqID+dirs['maskExt']
        image_out_name=mask_out_name.replace('/masks/','/regions/').replace(dirs['maskExt'],dirs['imExt'])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            imsave(image_out_name,Im)
            imsave(mask_out_name,mask_annotation)


def regions_in_mask(root, bounds, verbose=1):
    # find regions to save
    IDs_reg = []
    IDs_points = []

    for Annotation in root.findall("./Annotation"): # for all annotations
        annotationID = Annotation.attrib['Id']
        annotationType = Annotation.attrib['Type']

        # print(Annotation.findall(./))
        if annotationType =='9':
            for element in Annotation.iter('InputAnnotationId'):
                pointAnnotationID=element.text

            for Region in Annotation.findall("./*/Region"): # iterate on all region

                for Vertex in Region.findall("./*/Vertex"): # iterate on all vertex in region
                    # get points
                    x_point = np.int32(np.float64(Vertex.attrib['X']))
                    y_point = np.int32(np.float64(Vertex.attrib['Y']))
                    # test if points are in bounds
                    if bounds['x_min'] <= x_point <= bounds['x_max'] and bounds['y_min'] <= y_point <= bounds['y_max']: # test points in region bounds
                        # save region Id
                        IDs_points.append({'regionID' : Region.attrib['Id'], 'annotationID' : annotationID,'pointAnnotationID':pointAnnotationID})
                        break
        elif annotationType=='4':

            for Region in Annotation.findall("./*/Region"): # iterate on all region

                for Vertex in Region.findall("./*/Vertex"): # iterate on all vertex in region
                    # get points
                    x_point = np.int32(np.float64(Vertex.attrib['X']))
                    y_point = np.int32(np.float64(Vertex.attrib['Y']))
                    # test if points are in bounds
                    if bounds['x_min'] <= x_point <= bounds['x_max'] and bounds['y_min'] <= y_point <= bounds['y_max']: # test points in region bounds
                        # save region Id
                        IDs_reg.append({'regionID' : Region.attrib['Id'], 'annotationID' : annotationID})
                        break
    return IDs_reg,IDs_points


def get_vertex_points(root, IDs_reg,IDs_points, maskModes,excludedIDs,negativeIDs=None):
    Regions = []
    Points = []

    for ID in IDs_reg:
        Vertices = []
        if ID['annotationID'] not in excludedIDs:
            for Vertex in root.findall("./Annotation[@Id='" + ID['annotationID'] + "']/Regions/Region[@Id='" + ID['regionID'] + "']/Vertices/Vertex"):
                Vertices.append([int(float(Vertex.attrib['X'])), int(float(Vertex.attrib['Y']))])
            Regions.append({'Vertices':np.array(Vertices),'annotationID':ID['annotationID']})

    for ID in IDs_points:
        Vertices = []
        for Vertex in root.findall("./Annotation[@Id='" + ID['annotationID'] + "']/Regions/Region[@Id='" + ID['regionID'] + "']/Vertices/Vertex"):
            Vertices.append([int(float(Vertex.attrib['X'])), int(float(Vertex.attrib['Y']))])
        Points.append({'Vertices':np.array(Vertices),'pointAnnotationID':ID['pointAnnotationID']})
    if 'falsepositive' or 'negative' in maskModes:
        assert negativeIDs is not None,'Negatively annotated classes must be provided for negative/falsepositive mask mode'
        assert 'falsepositive' and 'negative' not in maskModes, 'Negative and false positive mask modes cannot both be true'

    useableRegions=[]
    if 'positive' in maskModes:
        for Region in Regions:
            regionPath=path.Path(Region['Vertices'])
            for Point in Points:
                if Region['annotationID'] not in negativeIDs:
                    if regionPath.contains_point(Point['Vertices'][0]):
                        Region['pointAnnotationID']=Point['pointAnnotationID']
                        useableRegions.append(Region)

    if 'negative' in maskModes:

        for Region in Regions:
            regionPath=path.Path(Region['Vertices'])
            if Region['annotationID'] in negativeIDs:
                if not any([regionPath.contains_point(Point['Vertices'][0]) for Point in Points]):
                    Region['pointAnnotationID']=Region['annotationID']
                    useableRegions.append(Region)
    if 'falsepositive' in maskModes:

        for Region in Regions:
            regionPath=path.Path(Region['Vertices'])
            if Region['annotationID'] in negativeIDs:
                if not any([regionPath.contains_point(Point['Vertices'][0]) for Point in Points]):
                    Region['pointAnnotationID']=0
                    useableRegions.append(Region)

    return useableRegions
def chop_suey_bounds(lb,xmlID,box_supervision_layers,wsiID,dirs,args):

    lbVerts=np.array(lb['BoxVerts'])
    xMin=min(lbVerts[:,0])
    xMax=max(lbVerts[:,0])
    yMin=min(lbVerts[:,1])
    yMax=max(lbVerts[:,1])

    # test=np.array(slide.read_region((xMin,yMin),0,(xMax-xMin,yMax-yMin)))[:,:,:3]

    local_bound = {'x_min' : xMin, 'y_min' : yMin, 'x_max' : xMax, 'y_max' : yMax}
    if args.chop_with_replacement:
        tree = ET.parse(xmlID)
        root = tree.getroot()
        IDs_reg,IDs_points = regions_in_mask_dots(root=root, bounds=local_bound,box_layers=box_supervision_layers)

        # find regions in bounds
        negativeIDs=['4']
        excludedIDs=['1']
        falsepositiveIDs=['4']
        usableRegions= get_vertex_points_dots(root=root, IDs_reg=IDs_reg,IDs_points=IDs_points,excludedIDs=excludedIDs,maskModes=['falsepositive','positive'],negativeIDs=negativeIDs,
            falsepositiveIDs=falsepositiveIDs)

        # image_sizes=
        masks_from_points(usableRegions,wsiID,dirs,50,args,[xMin,xMax,yMin,yMax])
    if args.standard_chop:
        l2=yMax-yMin #y
        l1=xMax-xMin #x
        pas_img = getWsi(wsiID)
        dim_x,dim_y=pas_img.dimensions
        mask=xml_to_mask(xmlID, [0,0], [dim_x,dim_y],ignore_id=box_supervision_layers, downsample_factor=1, verbose=0)
        mask=mask[yMin:yMax,xMin:xMax]

        # print(xMin,yMin,l1,l2)
        region=np.array(pas_img.read_region((xMin,yMin),0,(l1,l2)))[:,:,:3]

        basename=wsiID.split('/')[-1].split('.svs')[0]
        max_mask_size=args.training_max_size
        substepHR = int(max_mask_size*(1-args.overlap_rate)) #Step size before downsampling


        # plt.subplot(121)
        # plt.imshow(region)
        # plt.subplot(122)
        # plt.imshow(mask)
        # plt.show()

        if l1<max_mask_size or l2<max_mask_size:
            print('small image size')
            print(dims)
            exit()
        else:

            subIndex_yHR=np.array(range(0,l2,substepHR))
            subIndex_xHR=np.array(range(0,l1,substepHR))
            subIndex_yHR[-1]=l2-max_mask_size
            subIndex_xHR[-1]=l1-max_mask_size
            for i in subIndex_xHR:
                for j in subIndex_yHR:
                    subRegion=region[j:j+max_mask_size,i:i+max_mask_size,:]
                    subMask=mask[j:j+max_mask_size,i:i+max_mask_size]
                    image_identifier=basename+'_'.join(['',str(xMin),str(yMin),str(l1),str(l2),str(i),str(j)])
                    mask_out_name=dirs['basedir']+dirs['project'] + '/Permanent/HR/masks/'+image_identifier+dirs['maskExt']
                    image_out_name=mask_out_name.replace('/masks/','/regions/').replace(dirs['maskExt'],dirs['imExt'])

                    # image_sizes.append([max_mask_size,max_mask_size])
                    # plt.subplot(121)
                    # plt.imshow(subRegion)
                    # plt.subplot(122)
                    # plt.imshow(subMask)
                    # plt.show()
                    # # continue
                    # basename + '_' + str(image_identifier) + args.imBoxExt
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        imsave(image_out_name,subRegion)
                        imsave(mask_out_name,subMask)


def mask2polygons(mask):
    annotation=[]
    presentclasses=np.unique(mask)
    offset=-3
    presentclasses=presentclasses[presentclasses>2]
    presentclasses=list(presentclasses[presentclasses<7])
    for p in presentclasses:
        contours, hierarchy = cv2.findContours(np.array(mask==p).astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            if contour.size>=6:
                instance_dict={}
                contour_flat=contour.flatten().astype('float').tolist()
                xMin=min(contour_flat[::2])
                yMin=min(contour_flat[1::2])
                xMax=max(contour_flat[::2])
                yMax=max(contour_flat[1::2])
                instance_dict['bbox']=[xMin,yMin,xMax,yMax]
                instance_dict['bbox_mode']=BoxMode.XYXY_ABS
                instance_dict['category_id']=p+offset
                instance_dict['segmentation']=[contour_flat]
                annotation.append(instance_dict)
    return annotation




class Trainer(DefaultTrainer):

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=CustomDatasetMapper(cfg, True))

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=CustomDatasetMapper(cfg, True))

class CustomDatasetMapper:

    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        use_instance_mask: bool = False,
        use_keypoint: bool = False,
        instance_mask_format: str = "polygon",
        keypoint_hflip_indices: Optional[np.ndarray] = None,
        precomputed_proposal_topk: Optional[int] = None,
        recompute_boxes: bool = False,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            use_instance_mask: whether to process instance segmentation annotations, if available
            use_keypoint: whether to process keypoint annotations if available
            instance_mask_format: one of "polygon" or "bitmask". Process instance segmentation
                masks into this format.
            keypoint_hflip_indices: see :func:`detection_utils.create_keypoint_hflip_indices`
            precomputed_proposal_topk: if given, will load pre-computed
                proposals from dataset_dict and keep the top k proposals for each image.
            recompute_boxes: whether to overwrite bounding box annotations
                by computing tight bounding boxes from instance mask annotations.
        """
        if recompute_boxes:
            assert use_instance_mask, "recompute_boxes requires instance masks"
        # fmt: off
        self.is_train               = is_train
        self.augmentations          = T.AugmentationList(augmentations)
        self.image_format           = image_format
        self.use_instance_mask      = use_instance_mask
        self.instance_mask_format   = instance_mask_format
        self.use_keypoint           = use_keypoint
        self.keypoint_hflip_indices = keypoint_hflip_indices
        self.proposal_topk          = precomputed_proposal_topk
        self.recompute_boxes        = recompute_boxes
        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        augs = utils.build_augmentation(cfg, is_train)
        if cfg.INPUT.CROP.ENABLED and is_train:
            augs.insert(0, T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
            recompute_boxes = cfg.MODEL.MASK_ON
        else:
            recompute_boxes = False

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "instance_mask_format": cfg.INPUT.MASK_FORMAT,
            "use_keypoint": cfg.MODEL.KEYPOINT_ON,
            "recompute_boxes": recompute_boxes,
        }

        if cfg.MODEL.KEYPOINT_ON:
            ret["keypoint_hflip_indices"] = utils.create_keypoint_hflip_indices(cfg.DATASETS.TRAIN)

        if cfg.MODEL.LOAD_PROPOSALS:
            ret["precomputed_proposal_topk"] = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        return ret

    def _transform_annotations(self, dataset_dict, transforms, image_shape):
        # USER: Modify this if you want to keep them for some reason.
        for anno in dataset_dict["annotations"]:
            if not self.use_instance_mask:
                anno.pop("segmentation", None)
            if not self.use_keypoint:
                anno.pop("keypoints", None)

        annos = [
            utils.transform_instance_annotations(
                obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
            )
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]
        instances = utils.annotations_to_instances(
            annos, image_shape, mask_format=self.instance_mask_format
        )

        if self.recompute_boxes:
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
        dataset_dict["instances"] = utils.filter_empty_instances(instances)

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        c=dataset_dict['coordinates']
        h=dataset_dict['height']
        w=dataset_dict['width']

        slide=openslide.OpenSlide(dataset_dict['slide_loc'])
        image=np.array(slide.read_region((c[0],c[1]),0,(h,w)))[:,:,:3]
        slide.close()
        maskData=xml_to_mask(dataset_dict['xml_loc'], c, [h,w])

        if random.random()>0.5:
            hShift=np.random.normal(0,0.05)
            lShift=np.random.normal(1,0.025)
            # imageblock[im]=randomHSVshift(imageblock[im],hShift,lShift)
            image=rgb2hsv(image)
            image[:,:,0]=(image[:,:,0]+hShift)
            image=hsv2rgb(image)
            image=rgb2lab(image)
            image[:,:,0]=exposure.adjust_gamma(image[:,:,0],lShift)
            image=(lab2rgb(image)*255).astype('uint8')
            image = seq(images=[image])[0].squeeze()

        dataset_dict['annotations']=mask2polygons(maskData)
        utils.check_image_size(dataset_dict, image)

        sem_seg_gt = maskData
        sem_seg_gt[sem_seg_gt>2]=0
        sem_seg_gt[maskData==0] = 3
        sem_seg_gt=np.array(sem_seg_gt).astype('uint8')
        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        if "annotations" in dataset_dict:
            self._transform_annotations(dataset_dict, transforms, image_shape)

        return dataset_dict
