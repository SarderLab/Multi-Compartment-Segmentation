import os,cv2, time, random, multiprocessing,copy
from skimage.color import rgb2hsv,hsv2rgb,rgb2lab,lab2rgb
import numpy as np
from tiffslide import TiffSlide
from .xml_to_mask_minmax import xml_to_mask
# from generateTrainSet import generateDatalists
import logging
from detectron2.utils.logger import setup_logger
from skimage import exposure

setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
# from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data import detection_utils as utils
import detectron2.data.transforms as T

from detectron2.data import (DatasetCatalog,
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.config import configurable
from typing import List, Optional, Union
import torch

# sys.append("..")
from .wsi_loader_utils import train_samples_from_WSI, get_slide_data, get_random_chops
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

    
    region_size = int(args.boxSize) #Region size before downsampling
    
    dirs = {'imExt': '.jpeg'}
    dirs['basedir'] = args.base_dir
    dirs['maskExt'] = '.png'
    dirs['training_data_dir'] = args.training_data_dir





    print('Handcoded iteration')


    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    os.environ["CUDA_LAUNCH_BLOCKING"] ='1'

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



    classNum=len(tc)+len(sc)-1
    print('Number classes: '+ str(classNum))
    classes={}

    for idx,c in enumerate(classnames):
        classes[idx]={'isthing':isthing[idx],'color':xml_color[idx]}


    num_images=args.batch_size*args.train_steps
    # slide_idxs=train_dset.get_random_slide_idx(num_images)
    usable_slides=get_slide_data(args, wsi_directory = dirs['training_data_dir'])
    print('Number of slides:', len(usable_slides))
    usable_idx=range(0,len(usable_slides))
    slide_idxs=random.choices(usable_idx,k=num_images)
    image_coordinates=get_random_chops(slide_idxs,usable_slides,region_size)
  
    
    
    usable_slides_val=get_slide_data(args, wsi_directory=dirs['val_data_dir'])
    
    usable_idx_val=range(0,len(usable_slides_val))
    slide_idxs_val=random.choices(usable_idx_val,k=int(args.batch_size*args.train_steps/100))
    image_coordinates_val=get_random_chops(slide_idxs_val,usable_slides_val,region_size)

    
    
    DatasetCatalog.register("my_dataset", lambda:train_samples_from_WSI(args,image_coordinates))
    MetadataCatalog.get("my_dataset").set(thing_classes=tc)
    MetadataCatalog.get("my_dataset").set(stuff_classes=sc)
    

    _ = os.system("printf '\tTraining starts...\n'")
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"))
    cfg.DATASETS.TRAIN = ("my_dataset")

    cfg.TEST.EVAL_PERIOD=args.eval_period
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.50
    num_cores = multiprocessing.cpu_count()
    cfg.DATALOADER.NUM_WORKERS = 10

    if args.init_modelfile:
        cfg.MODEL.WEIGHTS = args.init_modelfile
    else:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml")  # Let training initialize from model zoo


    cfg.SOLVER.IMS_PER_BATCH = args.batch_size


    cfg.SOLVER.LR_policy='steps_with_lrs'
    cfg.SOLVER.MAX_ITER = args.train_steps
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.LRS = [0.000025,0.0000025]
    cfg.SOLVER.STEPS = [70000,90000]
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32],[64],[128], [256], [512], [1024]]
    cfg.MODEL.RPN.IN_FEATURES = ['p2', 'p3', 'p4', 'p5','p6','p6']
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[.1,.2,0.33, 0.5, 1.0, 2.0, 3.0,5,10]]
    cfg.MODEL.ANCHOR_GENERATOR.ANGLES=[-90,-60,-30,0,30,60,90]
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(tc)
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES =len(sc)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS=False
    cfg.INPUT.MIN_SIZE_TRAIN=args.boxSize
    cfg.INPUT.MAX_SIZE_TRAIN=args.boxSize
    cfg.DATASETS.TEST = ()
    cfg.OUTPUT_DIR = args.training_data_dir
    #os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    with open(cfg.OUTPUT_DIR+"/config_record.yaml", "w") as f:
        f.write(cfg.dump())   # save config to file
    trainer = Trainer(cfg)

    trainer.resume_or_load(resume=False)
    trainer.train()

    print('\nTraining completed, You can now run [--option predict]\033[0m\n')

def mask2polygons(mask):
    annotation=[]
    presentclasses=np.unique(mask)
    offset=-3
    presentclasses=presentclasses[presentclasses>2]
    presentclasses=list(presentclasses[presentclasses<7])
    for p in presentclasses:
        contours, hierarchy = cv2.findContours(np.array(mask==p, dtype='uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            if contour.size>=6:
                instance_dict={}
                contour_flat=contour.flatten().astype('float').tolist()
                xMin=min(contour_flat[::2])
                yMin=min(contour_flat[1::2])
                xMax=max(contour_flat[::2])
                yMax=max(contour_flat[1::2])
                instance_dict['bbox']=[xMin,yMin,xMax,yMax]
                instance_dict['bbox_mode']=BoxMode.XYXY_ABS.value
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
            for obj in dataset_dict.pop('annotations')
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

        slide= TiffSlide(dataset_dict['slide_loc'])
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
