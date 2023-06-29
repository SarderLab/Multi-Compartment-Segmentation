import os, sys, cv2, time, random, warnings, argparse, csv, multiprocessing,json
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
from xml_to_mask2o import *
from joblib import Parallel, delayed
from shutil import move
# from generateTrainSet import generateDatalists
from subprocess import call
from get_choppable_regions import get_choppable_regions
from PIL import Image

from detectron2_custom2.detectron2.utils.logger import setup_logger
setup_logger()
from detectron2_custom2.detectron2 import model_zoo
from detectron2_custom2.detectron2.engine import DefaultPredictor,DefaultTrainer
from detectron2_custom2.detectron2.config import get_cfg
from detectron2_custom2.detectron2.utils.visualizer import Visualizer,ColorMode
from detectron2_custom2.detectron2.data import MetadataCatalog, DatasetCatalog

# from detectron2_custom.detectron2.checkpoint import DetectionCheckpointer

from wsi_loader_utils import *
import openslide
from xml_to_mask_minmax import xml_to_mask
import cv2
from detectron2.structures import BoxMode

def mask2polygons(mask):
    annotation=[]
    presentclasses=np.unique(mask)

    offset=-2
    presentclasses=presentclasses[presentclasses>1]
    presentclasses=list(presentclasses[presentclasses<6])

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

def custom_mapper(dataset_dict):
    # Implement a mapper, similar to the default DatasetMapper, but with your own customizations

    # transform_list = [T.Resize((200,300)), T.RandomFlip(())]
    transform_list = [T.Resize(1200,1200),
                      T.RandomFlip(prob=0.5, horizontal=True, vertical=True),
                      T.RandomContrast(0.8, 3),
                      T.RandomBrightness(0.8, 1.6),
                      ]
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    c=dataset_dict['coordinates']
    h=dataset_dict['height']
    w=dataset_dict['width']
    image=np.array(openslide.OpenSlide(dataset_dict['slide_loc']).read_region((c[0],c[1]),0,(h,w)))
    utils.check_image_size(dataset_dict, image)
    image, transforms = T.apply_transform_gens(transform_list, image)
    maskData=xml_to_mask(dataset_dict['xml_loc'], c, [h,w])
    dataset_dict['annotations']=mask2polygons(maskData)

    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop("annotations")
    ]
    instances = utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict["instances"] = utils.filter_empty_instances(instances)







      sem_seg_gt=np.array(maskData==1).astype('uint8')

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

      # USER: Remove if you don't use pre-computed proposals.
      # Most users would not need this feature.
      if self.proposal_topk is not None:
          utils.transform_proposals(
              dataset_dict, image_shape, transforms, proposal_topk=self.proposal_topk
          )

      if not self.is_train:
          # USER: Modify this if you want to keep them for some reason.
          dataset_dict.pop("annotations", None)
          dataset_dict.pop("sem_seg_file_name", None)
          return dataset_dict

      if "annotations" in dataset_dict:
          self._transform_annotations(dataset_dict, transforms, image_shape)

      return dataset_dict

# use this dataloader instead of the default
class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_test_loader(cls, cfg: CfgNode, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=custom_mapper)

    @classmethod
    def build_train_loader(cls, cfg: CfgNode):
        return build_detection_train_loader(cfg, mapper=custom_mapper)
trainer = CustomTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
"""

Code for - cutting / augmenting / training CNN

This uses WSI and XML files to train 2 neural networks for semantic segmentation
    of histopath tissue via human in the loop training

"""


#Record start time
totalStart=time.time()

def IterateTraining(args):
    ## calculate low resolution block params
    downsampleLR = int(args.downsampleRateLR**.5) #down sample for each dimension
    region_sizeLR = int(args.boxSizeLR*(downsampleLR)) #Region size before downsampling
    stepLR = int(region_sizeLR*(1-args.overlap_percentLR)) #Step size before downsampling
    ## calculate low resolution block params
    downsample = int(args.downsampleRate**.5) #down sample for each dimension
    region_size = int(args.boxSize*(downsample)) #Region size before downsampling
    step = int(region_size*(1-args.overlap_percent)) #Step size before downsampling


    global classNum_HR,classEnumLR,classEnumHR
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


    ##All folders created, initiate WSI loading by human
    #raw_input('Please place WSIs in ')

    ##Check iteration session

    currentmodels=os.listdir(dirs['basedir'] + dirs['project'] + dirs['modeldir'])
    print('Handcoded iteration')
    # currentAnnotationIteration=check_model_generation(dirs)
    currentAnnotationIteration=0
    print('Current training session is: ' + str(currentAnnotationIteration))
    dirs['xml_dir']=dirs['basedir'] + dirs['project'] + dirs['training_data_dir'] + str(currentAnnotationIteration) + '/'
    ##Create objects for storing class distributions
    annotatedXMLs=glob.glob(dirs['basedir'] + dirs['project'] + dirs['training_data_dir'] + str(currentAnnotationIteration) + '/*.xml')
    classes=[]


    if args.classNum == 0:
        for xml in annotatedXMLs:
            classes.append(get_num_classes(xml))

        classNum_HR = max(classes)
    else:
        classNum_LR = args.classNum
        if args.classNum_HR != 0:
            classNum_HR = args.classNum_HR
        else:
            classNum_HR = classNum_LR

    classNum_HR=args.classNum

    train_dset = WSITrainingLoader(args,dirs['basedir'] + dirs['project'] + dirs['training_data_dir'] + str(currentAnnotationIteration))



    modeldir_HR = dirs['basedir']+dirs['project'] + dirs['modeldir'] + str(currentAnnotationIteration+1) + '/HR/'


    ##### HIGH REZ ARGS #####
    dirs['outDirAIHR']=dirs['basedir']+'/'+dirs['project'] + '/Permanent/HR/regions/'
    dirs['outDirAMHR']=dirs['basedir']+'/'+dirs['project'] + '/Permanent/HR/masks/'


    numImagesHR=len(glob.glob(dirs['outDirAIHR'] + '*' + dirs['imExt']))

    numStepsHR=(args.epoch_HR*numImagesHR)/ args.CNNbatch_sizeHR


    #-----------------------------------------------------------------------------------------
    # os.environ["CUDA_VISIBLE_DEVICES"]='0'
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    # img_dir='/hdd/bg/Detectron2/chop_detectron/Permanent/HR'

    img_dir=dirs['outDirAIHR']
    organType='kidney'
    print('Chopping with rules... '+ organType)
    if organType=='liver':
        classnames=['Background','BD','A']
        isthing=[0,1,1]
        xml_color = [[0,255,0], [0,255,255], [0,0,255]]
        tc=['BD','AT']
        sc=['Ob','B']
    elif organType =='kidney':
        classnames=['interstitium','glomerulus','sclerotic glomerulus','tubule','arterial tree']
        classes={}
        isthing=[0,1,1,1,1]
        xml_color = [[0,255,0], [0,255,255], [0,0,255], [255,0,0], [0,128,255]]
        tc=['G','SG','T','A']
        sc=['Ob','I','B']
    else:
        print('Provided organType not in supported types: kidney, liver')
    rand_sample=True
    json_dir=dirs['basedir']+'/'+dirs['project'] + '/Permanent/HR/'
    json_file=json_dir+'detectron_train'
    classes={}

    for idx,c in enumerate(classnames):
        classes[idx]={'isthing':isthing[idx],'color':xml_color[idx]}
    IdGen=IdGenerator(classes)


    # if args.prepare_detectron_json:
    #     HAIL2Detectron(img_dir,rand_sample,json_file,classnames,isthing,xml_color,organType,dirs)

    #### From json
    # DatasetCatalog.register("my_dataset", lambda:samples_from_json(json_file,rand_sample))
    DatasetCatalog.register("my_dataset", lambda:train_samples_from_WSI(train_dset,1000,args,json_file,classnames,isthing,xml_color,organType,dirs))
    MetadataCatalog.get("my_dataset").set(thing_classes=tc)
    MetadataCatalog.get("my_dataset").set(stuff_classes=sc)
    # exit()
    # seg_metadata=MetadataCatalog.get("my_dataset")
    #
    #
    # new_list = DatasetCatalog.get("my_dataset")
    # print(len(new_list))
    # for d in random.sample(new_list, 1000):
    #     # ident=d["file_name"].split('/')[-1]
    #     # print(ident)
    #     c=d['coordinates']
    #     h=d['height']
    #     w=d['width']
    #     slide=openslide.OpenSlide(d['slide_loc'])
    #     x=dirs['xml_dir']+'_'.join(d['image_id'].split('_')[:-2])+'.xml'
    #     img=np.array(slide.read_region((c[0],c[1]),0,(h,w)))
    #     slide.close()
    #     # mask=xml_to_mask(x, c, [h,w])
    #     # plt.subplot(121)
    #     # plt.imshow(im)
    #     # plt.subplot(122)
    #     # plt.imshow(mask)
    #     # plt.show()
    #     # img = cv2.imread(d["file_name"])
    #     visualizer = Visualizer(img[:, :, ::-1],metadata=seg_metadata, scale=0.5,idgen=IdGen)
    #     out = visualizer.draw_dataset_dict(d,train_dset)
    #     cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    #     cv2.imshow("output",out.get_image()[:, :, ::-1])
    #     cv2.waitKey(0) # waits until a key is pressed
    #     cv2.destroyAllWindows()
    # exit()
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"))
    cfg.DATASETS.TRAIN = ("my_dataset")
    cfg.DATASETS.TEST = ()
    num_cores = multiprocessing.cpu_count()
    cfg.DATALOADER.NUM_WORKERS = 5
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml")  # Let training initialize from model zoo

    # cfg.MODEL.WEIGHTS = os.path.join('/hdd/bg/Detectron2/HAIL_Detectron2/liver/MODELS/0/HR', "model_final.pth")
    cfg.MODEL.WEIGHTS = os.path.join('/hdd/bg/Detectron2/HAIL_Detectron2/output_PASHE_finetune_1', "model_0064999.pth")

    cfg.SOLVER.IMS_PER_BATCH = 4


    cfg.SOLVER.BASE_LR = 0.00002  # pick a good LR
    cfg.SOLVER.LR_policy='steps_with_lrs'
    cfg.SOLVER.MAX_ITER = 80000
    cfg.SOLVER.STEPS = []
    # cfg.SOLVER.STEPS = []
    # cfg.SOLVER.LRS = [0.00002]

    # cfg.SOLVER.BASE_LR = 0.002  # pick a good LR
    # cfg.SOLVER.LR_policy='steps_with_lrs'
    # cfg.SOLVER.MAX_ITER = 200000
    # cfg.SOLVER.STEPS = [150000,180000]
    # # cfg.SOLVER.STEPS = []
    # cfg.SOLVER.LRS = [0.0002,0.00002]

    # cfg.INPUT.CROP.ENABLED = True
    # cfg.INPUT.CROP.TYPE='absolute'
    # cfg.INPUT.CROP.SIZE=[100,100]
    cfg.MODEL.BACKBONE.FREEZE_AT = 0
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32],[64],[128], [256], [512], [1024]]
    cfg.MODEL.RPN.IN_FEATURES = ['p2', 'p3', 'p4', 'p5','p6','p6']

    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.33, 0.5, 1.0, 2.0, 3.0]]
    cfg.MODEL.ANCHOR_GENERATOR.ANGLES=[-90,-60,-30,0,30,60,90]

    cfg.MODEL.RPN.POSITIVE_FRACTION = 0.75

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(tc)
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES =len(sc)


    # cfg.INPUT.CROP.ENABLED = True
    # cfg.INPUT.CROP.TYPE='absolute'
    # cfg.INPUT.CROP.SIZE=[64,64]

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256  # faster, and good enough for this toy dataset (default: 512)
    # cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)

    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS=False
    cfg.INPUT.MIN_SIZE_TRAIN=0
    # cfg.XML_DIR=dirs['xml_dir']
    # cfg.INPUT.MAX_SIZE_TRAIN=500
    # mapper=DatasetMapper(cfg, True,train_dset)
    # exit()
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    with open(cfg.OUTPUT_DIR+"/config_record.yaml", "w") as f:
        f.write(cfg.dump())   # save config to file
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.01   # set a custom testing threshold
    cfg.TEST.DETECTIONS_PER_IMAGE = 500
    #
    # cfg.INPUT.MIN_SIZE_TRAIN=64
    # cfg.INPUT.MAX_SIZE_TRAIN=4000
    cfg.INPUT.MIN_SIZE_TEST=1200
    cfg.INPUT.MAX_SIZE_TEST=1200


    predict_samples=100
    predictor = DefaultPredictor(cfg)

    dataset_dicts = samples_from_json_mini(json_file,predict_samples)
    iter=0
    if not os.path.exists(os.getcwd()+'/network_predictions/'):
        os.mkdir(os.getcwd()+'/network_predictions/')
    for d in random.sample(dataset_dicts, predict_samples):
        # print(d["file_name"])
        # imclass=d["file_name"].split('/')[-1].split('_')[-5].split(' ')[-1]
        # if imclass in ["TRI","HE"]:
        im = cv2.imread(d["file_name"])
        panoptic_seg, segments_info = predictor(im)["panoptic_seg"]  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        # print(segments_info)
        # plt.imshow(panoptic_seg.to("cpu"))
        # plt.show()
        v = Visualizer(im[:, :, ::-1], seg_metadata, scale=1.2)
        v = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)
        # panoptic segmentation result
        # plt.ion()
        plt.subplot(121)
        plt.imshow(im[:, :, ::-1])
        plt.subplot(122)
        plt.imshow(v.get_image())
        plt.savefig(f"./network_predictions/input_{iter}.jpg",dpi=300)
        plt.show()
        # plt.ioff()


        # v = Visualizer(im[:, :, ::-1],
        #                metadata=seg_metadata,
        #                scale=0.5,
        # )
        # out = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"),segments_info)

        # imsave('./network_predictions/pred'+str(iter)+'.png',np.hstack((im,v.get_image())))
        iter=iter+1
        # cv2.imshow('',out.get_image()[:, :, ::-1])
        # cv2.waitKey(0) # waits until a key is pressed
        # cv2.destroyAllWindows()
    #-----------------------------------------------------------------------------------------

    finish_model_generation(dirs,currentAnnotationIteration)

    print('\n\n\033[92;5mPlease place new wsi file(s) in: \n\t' + dirs['basedir'] + dirs['project']+ dirs['training_data_dir'] + str(currentAnnotationIteration+1))
    print('\nthen run [--option predict]\033[0m\n')




def moveimages(startfolder,endfolder):
    filelist=glob.glob(startfolder + '*')
    for file in filelist:
        fileID=file.split('/')[-1]
        move(file,endfolder + fileID)


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
        substepHR = int(max_mask_size*(1-args.overlap_percentHR)) #Step size before downsampling


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
'''
def masks_from_points(root,usableRegions,wsiID,dirs):
    pas_img = getWsi(wsiID)
    image_sizes=[]
    basename=wsiID.split('/')[-1].split('.svs')[0]

    for usableRegion in tqdm(usableRegions):
        vertices=usableRegion['Vertices']
        x1=min(vertices[:,0])
        x2=max(vertices[:,0])
        y1=min(vertices[:,1])
        y2=max(vertices[:,1])
        points = np.stack([np.asarray(vertices[:,0]), np.asarray(vertices[:,1])], axis=1)
        if (x2-x1)>0 and (y2-y1)>0:
            l1=x2-x1
            l2=y2-y1
            xMultiplier=np.ceil((l1)/64)
            yMultiplier=np.ceil((l2)/64)
            pad1=int(xMultiplier*64-l1)
            pad2=int(yMultiplier*64-l2)

            points[:,1] = np.int32(np.round(points[:,1] - y1 ))
            points[:,0] = np.int32(np.round(points[:,0] - x1 ))
            mask = 2*np.ones([y2-y1,x2-x1], dtype=np.uint8)
            if int(usableRegion['pointAnnotationID'])==0:
                pass
            else:
                cv2.fillPoly(mask, [points], int(usableRegion['pointAnnotationID'])-4)
            PAS = pas_img.read_region((x1,y1), 0, (x2-x1,y2-y1))
            # print(usableRegion['pointAnnotationID'])
            PAS = np.array(PAS)[:,:,0:3]
            mask=np.pad( mask,((0,pad2),(0,pad1)),'constant',constant_values=(2,2) )
            PAS=np.pad( PAS,((0,pad2),(0,pad1),(0,0)),'constant',constant_values=(0,0) )

            image_identifier=basename+'_'.join(['',str(x1),str(y1),str(l1),str(l2)])
            mask_out_name=dirs['basedir']+dirs['project'] + '/Permanent/HR/masks/'+image_identifier+'.png'
            image_out_name=mask_out_name.replace('/masks/','/regions/')
            # basename + '_' + str(image_identifier) + args.imBoxExt
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                imsave(image_out_name,PAS)
                imsave(mask_out_name,mask)
            # exit()
            # extract image region
            # plt.subplot(121)
            # plt.imshow(PAS)
            # plt.subplot(122)
            # plt.imshow(mask)
            # plt.show()
            # image_sizes.append([x2-x1,y2-y1])
        else:
            print('Broken region')
    return image_sizes
'''
