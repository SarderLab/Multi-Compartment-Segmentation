import os, sys, cv2, time, random, warnings, multiprocessing#json,# detectron2
import numpy as np
import matplotlib.pyplot as plt
import lxml.etree as ET
from matplotlib import path
from skimage.transform import resize
from skimage.io import imread, imsave
import glob
from .getWsi import getWsi

from .xml_to_mask2 import get_supervision_boxes, regions_in_mask_dots, get_vertex_points_dots, masks_from_points, restart_line 
from joblib import Parallel, delayed
from shutil import move
# from generateTrainSet import generateDatalists
#from subprocess import call
#from .get_choppable_regions import get_choppable_regions
from PIL import Image

from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor,DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer#,ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
#from detectron2.structures import BoxMode
from .get_dataset_list import HAIL2Detectron, samples_from_json, samples_from_json_mini
#from detectron2.checkpoint import DetectionCheckpointer
#from detectron2.modeling import build_model

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
    downsampleHR = int(args.downsampleRateHR**.5) #down sample for each dimension
    region_sizeHR = int(args.boxSizeHR*(downsampleHR)) #Region size before downsampling
    stepHR = int(region_sizeHR*(1-args.overlap_percentHR)) #Step size before downsampling


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
    currentAnnotationIteration=2
    print('Current training session is: ' + str(currentAnnotationIteration))

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

    ##for all WSIs in the initiating directory:
    if args.chop_data == 'True':
        print('Chopping')

        start=time.time()
        size_data=[]

        for xmlID in annotatedXMLs:

            #Get unique name of WSI
            fileID=xmlID.split('/')[-1].split('.xml')[0]
            print('-----------------'+fileID+'----------------')
            #create memory addresses for wsi files
            for ext in [args.wsi_ext]:
                wsiID=dirs['basedir'] + dirs['project']+  dirs['training_data_dir'] + str(currentAnnotationIteration) +'/'+ fileID + ext

                #Ensure annotations exist
                if os.path.isfile(wsiID)==True:
                    break


            #Load openslide information about WSI
            if ext != '.tif':
                slide=getWsi(wsiID)
                #WSI level 0 dimensions (largest size)
                dim_x,dim_y=slide.dimensions
            else:
                im = Image.open(wsiID)
                dim_x, dim_y=im.size
            location=[0,0]
            size=[dim_x,dim_y]
            tree = ET.parse(xmlID)
            root = tree.getroot()
            box_supervision_layers=['8']
            # calculate region bounds
            global_bounds = {'x_min' : location[0], 'y_min' : location[1], 'x_max' : location[0] + size[0], 'y_max' : location[1] + size[1]}
            local_bounds = get_supervision_boxes(root,box_supervision_layers)
            num_cores = multiprocessing.cpu_count()
            Parallel(n_jobs=num_cores)(delayed(chop_suey_bounds)(args=args,wsiID=wsiID,
                dirs=dirs,lb=lb,xmlID=xmlID,box_supervision_layers=box_supervision_layers) for lb in tqdm(local_bounds))
            # for lb in tqdm(local_bounds):

                # size_data.extend(image_sizes)

            '''
            wsi_mask=xml_to_mask(xmlID, [0,0], [dim_x,dim_y])



            #Enumerate cpu core count
            num_cores = multiprocessing.cpu_count()

            #Generate iterators for parallel chopping of WSIs in high resolution
            #index_yHR=range(30240,dim_y-stepHR,stepHR)
            #index_xHR=range(840,dim_x-stepHR,stepHR)
            index_yHR=range(0,dim_y,stepHR)
            index_xHR=range(0,dim_x,stepHR)
            index_yHR[-1]=dim_y-stepHR
            index_xHR[-1]=dim_x-stepHR
            #Create memory address for chopped images high resolution
            outdirHR=dirs['basedir'] + dirs['project'] + dirs['tempdirHR']

            #Perform high resolution chopping in parallel and return the number of
            #images in each of the labeled classes
            chop_regions=get_choppable_regions(wsi=wsiID,
                index_x=index_xHR,index_y=index_yHR,boxSize=region_sizeHR,white_percent=args.white_percent)

            Parallel(n_jobs=num_cores)(delayed(return_region)(args=args,
                wsi_mask=wsi_mask, wsiID=wsiID,
                fileID=fileID, yStart=j, xStart=i, idxy=idxy,
                idxx=idxx, downsampleRate=args.downsampleRateHR,
                outdirT=outdirHR, region_size=region_sizeHR,
                dirs=dirs, chop_regions=chop_regions,classNum_HR=classNum_HR) for idxx,i in enumerate(index_xHR) for idxy,j in enumerate(index_yHR))

            #for idxx,i in enumerate(index_xHR):
            #    for idxy,j in enumerate(index_yHR):
            #        if chop_regions[idxy,idxx] != 0:
            #            return_region(args=args,xmlID=xmlID, wsiID=wsiID, fileID=fileID, yStart=j, xStart=i,idxy=idxy, idxx=idxx,
            #            downsampleRate=args.downsampleRateHR,outdirT=outdirHR, region_size=region_sizeHR, dirs=dirs,
            #            chop_regions=chop_regions,classNum_HR=classNum_HR)
            #        else:
            #            print('pass')


        # exit()
        print('Time for WSI chopping: ' + str(time.time()-start))

        classEnumHR=np.ones([classNum_HR,1])*classNum_HR

        ##High resolution augmentation
        #Enumerate high resolution class distribution
        classDistHR=np.zeros(len(classEnumHR))
        for idx,value in enumerate(classEnumHR):
            classDistHR[idx]=value/sum(classEnumHR)
        print(classDistHR)
        #Define number of augmentations per class

        moveimages(dirs['basedir']+dirs['project'] + dirs['tempdirHR'] + '/regions/', dirs['basedir']+dirs['project'] + '/Permanent/HR/regions/')
        moveimages(dirs['basedir']+dirs['project'] + dirs['tempdirHR'] + '/masks/',dirs['basedir']+dirs['project'] + '/Permanent/HR/masks/')


        #Total time
        print('Time for high resolution augmenting: ' + str((time.time()-totalStart)/60) + ' minutes.')
        '''

    # with open('sizes.csv','w',newline='') as myfile:
    #     wr=csv.writer(myfile,quoting=csv.QUOTE_ALL)
    #     wr.writerow(size_data)
    # pretrain_HR=get_pretrain(currentAnnotationIteration,'/HR/',dirs)

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
    classnames=['Background','BD','A']
    isthing=[0,1,1]
    xml_color = [[0,255,0], [0,255,255], [0,0,255]]

    rand_sample=True

    json_file=img_dir+'/detectron_train.json'
    HAIL2Detectron(img_dir,rand_sample,json_file,classnames,isthing,xml_color)
    tc=['BD','AT']
    sc=['I','B']
    #### From json
    DatasetCatalog.register("my_dataset", lambda:samples_from_json(json_file,rand_sample))
    MetadataCatalog.get("my_dataset").set(thing_classes=tc)
    MetadataCatalog.get("my_dataset").set(stuff_classes=sc)

    seg_metadata=MetadataCatalog.get("my_dataset")


    # new_list = DatasetCatalog.get("my_dataset")
    # print(len(new_list))
    # for d in random.sample(new_list, 100):
    #
    #     img = cv2.imread(d["file_name"])
    #     visualizer = Visualizer(img[:, :, ::-1],metadata=seg_metadata, scale=0.5)
    #     out = visualizer.draw_dataset_dict(d)
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
    cfg.DATALOADER.NUM_WORKERS = num_cores-3
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml")  # Let training initialize from model zoo

    cfg.MODEL.WEIGHTS = os.path.join('/hdd/bg/Detectron2/HAIL_Detectron2/liver/MODELS/0/HR', "model_final.pth")

    cfg.SOLVER.IMS_PER_BATCH = 10


    # cfg.SOLVER.BASE_LR = 0.02  # pick a good LR
    # cfg.SOLVER.LR_policy='steps_with_lrs'
    # cfg.SOLVER.MAX_ITER = 50000
    # cfg.SOLVER.STEPS = [30000,40000]
    # # cfg.SOLVER.STEPS = []
    # cfg.SOLVER.LRS = [0.002,0.0002]

    cfg.SOLVER.BASE_LR = 0.002  # pick a good LR
    cfg.SOLVER.LR_policy='steps_with_lrs'
    cfg.SOLVER.MAX_ITER = 200000
    cfg.SOLVER.STEPS = [150000,180000]
    # cfg.SOLVER.STEPS = []
    cfg.SOLVER.LRS = [0.0002,0.00002]

    # cfg.INPUT.CROP.ENABLED = True
    # cfg.INPUT.CROP.TYPE='absolute'
    # cfg.INPUT.CROP.SIZE=[100,100]
    cfg.MODEL.BACKBONE.FREEZE_AT = 0
    # cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[4],[8],[16], [32], [64], [64], [64]]
    # cfg.MODEL.RPN.IN_FEATURES = ['p2', 'p2', 'p2', 'p3','p4','p5','p6']
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
    # cfg.INPUT.MAX_SIZE_TRAIN=500

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
    cfg.INPUT.MIN_SIZE_TEST=64
    cfg.INPUT.MAX_SIZE_TEST=500


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

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imsave(outdirT + '/regions/' + uniqID + dirs['imExt'],Im)
            imsave(outdirT + '/masks/' + uniqID +dirs['maskExt'],mask_annotation)


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
    tree = ET.parse(xmlID)
    root = tree.getroot()
    lbVerts=np.array(lb['BoxVerts'])
    xMin=min(lbVerts[:,0])
    xMax=max(lbVerts[:,0])
    yMin=min(lbVerts[:,1])
    yMax=max(lbVerts[:,1])

    # test=np.array(slide.read_region((xMin,yMin),0,(xMax-xMin,yMax-yMin)))[:,:,:3]

    local_bound = {'x_min' : xMin, 'y_min' : yMin, 'x_max' : xMax, 'y_max' : yMax}
    IDs_reg,IDs_points = regions_in_mask_dots(root=root, bounds=local_bound,box_layers=box_supervision_layers)

    # find regions in bounds
    negativeIDs=['4']
    excludedIDs=['1']
    falsepositiveIDs=['4']
    usableRegions= get_vertex_points_dots(root=root, IDs_reg=IDs_reg,IDs_points=IDs_points,excludedIDs=excludedIDs,maskModes=['falsepositive','positive'],negativeIDs=negativeIDs,
        falsepositiveIDs=falsepositiveIDs)

    # image_sizes=
    masks_from_points(usableRegions,wsiID,dirs,50,args,[xMin,xMax,yMin,yMax])
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
