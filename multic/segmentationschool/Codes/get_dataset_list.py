from .utils import IdGenerator, id2rgb
from skimage.measure import label
from skimage.io import imsave#,imread
import random
import numpy as np
import glob
import warnings
import matplotlib.pyplot as plt
#from matplotlib import patches
import cv2
from detectron2.structures import BoxMode
from joblib import Parallel, delayed
import multiprocessing
import os, json
from tqdm import tqdm
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def mask2polygons(mask):
    annotation=[]
    presentclasses=np.unique(mask)
    presentclasses=presentclasses[presentclasses<2]


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
                instance_dict['category_id']=p
                instance_dict['segmentation']=[contour_flat]
                annotation.append(instance_dict)
    return annotation

def get_seg_info(mask,IdGen,imID):
    seg_info=[]
    mask_encoded=np.zeros(np.shape(mask))

    presentclasses=np.unique(mask)

    presentclasses=presentclasses[presentclasses<2]

    for p in presentclasses:
        masklabel=label(mask==p)
        for j in range(1,np.max(masklabel)+1):
            segment_id=IdGen.get_id(p)
            mask_encoded[masklabel==j]=segment_id
            seg_info.append({'id':segment_id,'category_id':p,'iscrowd':0,'area':np.sum(masklabel==j),'isthing':1})
    return seg_info,mask_encoded

def get_list_parallel(im,total,mask_dir,IdGen,out_dir,rand_sample,i):
    imID=im.split('/')[-1].split('.png')[0]
    maskname=im.replace('/regions/','/masks/')
    sname=maskname.replace('.png','_s.png')
    pname=maskname.replace('.png','_p.png')
    maskData=cv2.imread(maskname,0)

    stuff_mask=np.array(maskData==2).astype('uint8')

    # plt.subplot(131)
    # plt.imshow(maskData)
    # plt.subplot(132)
    # plt.imshow(stuff_mask)
    # plt.subplot(133)
    # plt.imshow(cv2.imread(im)[:,:,::-1])
    # plt.show()

    imsize=np.shape(maskData)

    image_annotation_info={}
    image_annotation_info['file_name']=im
    image_annotation_info['height']=imsize[0]
    image_annotation_info['width']=imsize[1]
    image_annotation_info['image_id']=imID
    image_annotation_info['sem_seg_file_name']=sname
    # image_annotation_info['sem_seg_file_name']=maskname

    out=get_seg_info(maskData,IdGen,imID)

    annotations={}
    image_annotation_info['segments_info']=out[0]
    image_annotation_info['annotations']=mask2polygons(maskData)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # plt.subplot(121)
        # plt.imshow(out[1])

        rgbim=id2rgb(out[1])
        # plt.subplot(122)
        # plt.imshow(rgbim)
        # plt.show()
        imsave(pname,rgbim)
        imsave(sname,stuff_mask)
    # plt.subplot(121)
    # plt.imshow()
    # plt.subplot(122)
    image_annotation_info['pan_seg_file_name']=pname
    return image_annotation_info

def HAIL2Detectron(img_dir,rand_sample,out_json,classnames,isthing,xml_color,num_images=None):

    classes={}

    for idx,c in enumerate(classnames):
        classes[idx]={'isthing':isthing[idx],'color':xml_color[idx]}
    IdGen=IdGenerator(classes)


    image_dir=img_dir
    mask_dir=img_dir.replace('/regions/','/masks/')
    out_dir=mask_dir
    images=[]

    images.extend(glob.glob(img_dir+'*.png'))

    num_cores = multiprocessing.cpu_count()
    # num_cores=5
    data_list=[]
    if num_images is not None:
        total=num_images
        if num_images != len(images) and not rand_sample:
            print('Warning: Deterministically sampling a length less than the total images may lead to biased results')
    else:
        total=len(images)


    if rand_sample:
        random.shuffle(images)
        data_list=Parallel(n_jobs=num_cores)(delayed(get_list_parallel)(i=i,total=total,
            im=im,mask_dir=mask_dir,IdGen=IdGen,out_dir=out_dir,
            rand_sample=rand_sample) for i,im in enumerate(tqdm(images)))
    else:
        data_list=Parallel(n_jobs=num_cores)(delayed(get_list_parallel)(i=i,total=total,
            im=im,mask_dir=mask_dir,IdGen=IdGen,out_dir=out_dir,
            rand_sample=rand_sample) for i,im in enumerate(tqdm(images)))
    with open(out_json,'w') as fout:
        json.dump(data_list,fout,cls=NpEncoder)

    return data_list

def samples_from_json(json_file,rand_sample,num_images=None):
    with open(json_file) as f:
        full_list=json.load(f)

    # json_length=len(full_list)
    # data_list=[]
    # if num_images is not None:
    #     total=num_images
    #     if num_images != json_length and not rand_sample:
    #         print('Warning: Deterministically sampling a length less than the total images may lead to biased results')
    # else:
    #     total=len(images)
    #
    #
    #
    # if rand_sample:
    #     data_list=random.choices(full_list, k=total)
    # else:
    #     data_list=full_list[:total]
    return full_list

def samples_from_json_mini(json_file,num_images):


    full_list=[]
    with open(json_file) as f:
        j=json.load(f)
        full_list.extend(random.sample(j,num_images))

    return full_list

def HAIL2Detectron_predict(img_dir,img_size):


    images=glob.glob(img_dir+'/*.jpeg')
    num_cores = multiprocessing.cpu_count()

    data_list=[]
    total=len(images)


    data_list=Parallel(n_jobs=num_cores)(delayed(get_list_parallel_predict)(i=i,total=total,im=im,img_size=img_size) for i,im in enumerate(images))

    return data_list


def get_list_parallel_predict(i,total,im,img_size):

    print("Preparing images {0}/{1}".format(i,total), end ="\r")

    imID=im.split('/')[-1].split('.jpeg')[0]

    image_annotation_info={}
    image_annotation_info['file_name']=im

    image_annotation_info['height']=img_size
    image_annotation_info['width']=img_size
    image_annotation_info['image_id']=imID

    return image_annotation_info

def decode_panoptic(image,segments_info,out_dir,file_name):
    # plt.imshow(image)
    # plt.show()
    detections=np.unique(image)
    detections=detections[detections>-1]
    out=np.zeros_like(image)
    for ids in segments_info:
        if ids['isthing']:
            out[image==ids['id']]=ids['category_id']+2

        else:
            if ids['category_id']==1:
                out[image==ids['id']]=ids['category_id']=1
    # plt.imshow(out)
    # plt.show()
    # exit()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        imsave(out_dir+'/'+file_name.split('/')[-1].replace('.jpeg','.png'),out.astype('uint8'))

def WSIGridIterator(wsi_name,choppable_regions,index_x,index_y,region_size,dim_x,dim_y,glob_offset):
    wsi_name=os.path.splitext(wsi_name.split('/')[-1])[0]

    data_list=[]
    for idxy, i in tqdm(enumerate(index_y)):
        for idxx, j in enumerate(index_x):
            if choppable_regions[idxy, idxx] != 0:
                yEnd = min(dim_y+glob_offset[1],i+region_size)
                #print(yEnd)
                xEnd = min(dim_x+glob_offset[0],j+region_size)

                #print(xEnd)
                xLen=xEnd-j
                yLen=yEnd-i

                image_annotation_info={}
                image_annotation_info['file_name']='_'.join([wsi_name,str(j),str(i),str(xEnd),str(yEnd)])

                image_annotation_info['height']=yLen
                image_annotation_info['width']=xLen
                image_annotation_info['image_id']=image_annotation_info['file_name']
                image_annotation_info['xStart']=j
                image_annotation_info['yStart']=i
                data_list.append(image_annotation_info)

    return data_list
