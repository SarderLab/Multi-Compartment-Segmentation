import openslide,glob,os
import numpy as np
from scipy.ndimage.morphology import binary_fill_holes
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv
from skimage.filters import gaussian
from skimage.morphology import binary_dilation, diamond
from skimage.segmentation import clear_border
import cv2
from tqdm import tqdm
from skimage.io import imread,imsave

from utils import IdGenerator, id2rgb
from skimage.measure import label
from skimage.io import imread,imsave
import random
import numpy as np
import glob
import warnings
import matplotlib.pyplot as plt
from matplotlib import patches
import cv2
from detectron2.structures import BoxMode
from joblib import Parallel, delayed
import multiprocessing
import os,json
from tqdm import tqdm
from xml_to_mask_minmax import xml_to_mask,write_minmax_to_xml
import lxml.etree as ET
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
#
# def mask2polygons(mask,organType):
#     annotation=[]
#     presentclasses=np.unique(mask)
#     if organType=='kidney':
#         offset=-2
#         presentclasses=presentclasses[presentclasses>1]
#         presentclasses=list(presentclasses[presentclasses<6])
#         if 6 in presentclasses:
#             print('Beware, this mask directory has level 6 (cortical) annotations in it!')
#             print('Triggered by '+ imID)
#             exit()
#             presentclasses=presentclasses[:-1]
#
#     elif organType=='liver':
#         offset=0
#         presentclasses=presentclasses[presentclasses<2]
#
#     for p in presentclasses:
#         contours, hierarchy = cv2.findContours(np.array(mask==p).astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#         for contour in contours:
#             if contour.size>=6:
#                 instance_dict={}
#                 contour_flat=contour.flatten().astype('float').tolist()
#                 xMin=min(contour_flat[::2])
#                 yMin=min(contour_flat[1::2])
#                 xMax=max(contour_flat[::2])
#                 yMax=max(contour_flat[1::2])
#                 instance_dict['bbox']=[xMin,yMin,xMax,yMax]
#                 instance_dict['bbox_mode']=BoxMode.XYXY_ABS
#                 instance_dict['category_id']=p+offset
#                 instance_dict['segmentation']=[contour_flat]
#                 annotation.append(instance_dict)
#     return annotation

# def get_seg_info(mask,IdGen,imID,organType):
#     seg_info=[]
#     mask_encoded=np.zeros(np.shape(mask))
#     presentclasses=np.unique(mask)
#     if organType=='kidney':
#         offset=-2
#         presentclasses=presentclasses[presentclasses>1]
#         presentclasses=list(presentclasses[presentclasses<6])
#         if 6 in presentclasses:
#             print('Beware, this mask directory has level 6 (cortical) annotations in it!')
#             print('Triggered by '+ imID)
#             exit()
#             presentclasses=presentclasses[:-1]
#
#     elif organType=='liver':
#         offset=0
#         presentclasses=presentclasses[presentclasses<2]
#
#
#     for p in presentclasses:
#         masklabel=label(mask==p)
#         for j in range(1,np.max(masklabel)+1):
#             segment_id=IdGen.get_id(p+offset)
#             mask_encoded[masklabel==j]=segment_id
#             seg_info.append({'id':segment_id,'category_id':p+offset,'iscrowd':0,'area':np.sum(masklabel==j),'isthing':1})
#
#     return seg_info,mask_encoded

def get_image_meta(i,args):
    image_annotation_info={}
    # image_annotation_info['slide_loc']=train_dset.get_single_slide_data(i[0])
    image_annotation_info['slide_loc']=i[0]
    slide=openslide.OpenSlide(image_annotation_info['slide_loc'])
    magx=np.round(float(slide.properties['openslide.mpp-x']),2)
    magy=np.round(float(slide.properties['openslide.mpp-y']),2)

    assert magx == magy
    if magx ==0.25:
        dx=args.boxSize
        dy=args.boxSize
    elif magx == 0.5:
        dx=int(args.boxSize/2)
        dy=int(args.boxSize/2)
    else:
        print('nonstandard image magnification')
        print(slide)
        print(magx,magy)
        exit()

    image_annotation_info['coordinates']=[i[2][1],i[2][0]]
    image_annotation_info['height']=dx
    image_annotation_info['width']=dy
    image_annotation_info['image_id']=i[1].split('/')[-1].replace('.xml','_'.join(['',str(i[2][1]),str(i[2][0])]))
    image_annotation_info['xml_loc']=i[1]
    slide.close()
    return image_annotation_info

def train_samples_from_WSI(args,image_coordinates):


    num_cores=multiprocessing.cpu_count()
    print('Generating detectron2 dictionary format...')
    data_list=Parallel(n_jobs=num_cores,backend='threading')(delayed(get_image_meta)(i=i,
        args=args) for i in tqdm(image_coordinates))
    # print('flushing')
    # train_dset.flush_xmldata()
    # print('flushed')
    # print(vars(train_dset))
    # exit()
    ##In serial for debugging
    # for i in tqdm(image_coordinates):
        # print(image_annotation_info['image_id'])
        # image_annotation_info['sem_seg_file_name']=sname
        # image_annotation_info['sem_seg_file_name']=maskname
        # out=get_seg_info(maskData,IdGen,imID,organType)
        # image_annotation_info['segments_info']=out[0]
        # image_annotation_info['annotations']=mask2polygons(maskData,organType)
        # rgbim=id2rgb(out[1])
        # image_annotation_info['pan_seg_file_name']=pname
        # data_list.append(image_annotation_info)
    return data_list

def WSIGridIterator(wsi_name,choppable_regions,index_x,index_y,region_size,dim_x,dim_y):
    wsi_name=os.path.splitext(wsi_name.split('/')[-1])[0]
    data_list=[]
    for idxy, i in tqdm(enumerate(index_y)):
        for idxx, j in enumerate(index_x):
            if choppable_regions[idxy, idxx] != 0:
                yEnd = min(dim_y,i+region_size)
                xEnd = min(dim_x,j+region_size)
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


# class WSITrainingLoader():
#     def __init__(self,args, wsi_directory=None, transform=None):
def get_slide_data(args, wsi_directory=None):
        assert wsi_directory is not None, 'location of training svs and xml must be provided'

        mask_out_loc=os.path.join(wsi_directory.replace('/TRAINING_data/0','Permanent/Tissue_masks/'),)
        if not os.path.exists(mask_out_loc):
            os.makedirs(mask_out_loc)
        all_slides=[]
        for ext in args.wsi_ext.split(','):
            all_slides.extend(glob.glob(wsi_directory+'/*'+ext))
        print('Getting slide metadata and usable regions...')

        usable_slides=[]
        for slide_loc in all_slides:
            slideID,slideExt=os.path.splitext(slide_loc.split('/')[-1])
            xmlpath=slide_loc.replace(slideExt,'.xml')
            if os.path.isfile(xmlpath):
                write_minmax_to_xml(xmlpath)

                print("Gathering slide data ... "+ slideID,end='\r')
                slide=openslide.OpenSlide(slide_loc)
                chop_array=get_choppable_regions(slide,args,slideID,slideExt,mask_out_loc)

                mag_x=np.round(float(slide.properties['openslide.mpp-x']),2)
                mag_y=np.round(float(slide.properties['openslide.mpp-y']),2)
                slide.close()
                tree = ET.parse(xmlpath)
                root = tree.getroot()
                balance_classes=args.balanceClasses.split(',')
                classNums={}
                for b in balance_classes:
                    classNums[b]=0
                # balance_annotations={}
                for Annotation in root.findall("./Annotation"):

                    annotationID = Annotation.attrib['Id']
                    if annotationID=='7':
                        print(xmlpath)
                        exit()
                    if annotationID in classNums.keys():

                        classNums[annotationID]=len(Annotation.findall("./*/Region"))
                    else:
                        pass

                usable_slides.append({'slide_loc':slide_loc,'slideID':slideID,
                    'chop_array':chop_array,'num_regions':len(chop_array),'mag':[mag_x,mag_y],
                    'xml_loc':xmlpath,'annotations':classNums,'root':root,
                    'thumb_loc':os.path.join(mask_out_loc,'_'.join([slideID,slideExt[1:]+'.jpeg']))})
            else:
                print('\n')
                print('no annotation XML file found for:')
                print(slideID)
                exit()

        print('\n')
        return usable_slides

# def get_random_slide_idx(self,train_images):
#     return random.choices(self.usable_idx,k=train_images)
# def get_single_slide_data(self,idx):
#     return self.usable_slides[idx]['slide_loc']
def get_random_chops(slide_idx,usable_slides,region_size):
    # chops=[]
    choplen=len(slide_idx)
    chops=Parallel(n_jobs=multiprocessing.cpu_count(),backend='threading')(delayed(get_chop_data)(idx=idx,
        usable_slides=usable_slides,region_size=region_size) for idx in tqdm(slide_idx))
    return chops
# def get_xml_folder(self):
#     return self.xml_folder
# def flush_xmldata(self):
#     import gc
#     del self.usable_slides
#     gc.collect()
#     input('collcted')
#     # for idx,slide in enumerate(self.usable_slides):
#     #     # slide['root']=None
#     #     # self.usable_slides[idx]=slide
#     #     del self.usable_slides[idx]['root']
#     #     del self.usable_slides[idx]['chop_array']
# def get_image_means(self):
#     image_means=[]
#     print('Getting custom image means...')
#     for usable_slide in tqdm(self.usable_slides):
#
#         thumbIm=imread(usable_slide['thumb_loc'])
#         binary=(imread(usable_slide['thumb_loc'].replace('.jpeg','.png'))/255)
#         maskedIm=(thumbIm*np.repeat(np.expand_dims(binary,axis=-1),axis=2,repeats=3))
#         rmean=np.sum(maskedIm[:,:,0])/np.count_nonzero(binary)
#         gmean=np.sum(maskedIm[:,:,1])/np.count_nonzero(binary)
#         bmean=np.sum(maskedIm[:,:,2])/np.count_nonzero(binary)
#         image_means.append([rmean,gmean,bmean])
#
#     return image_means

def get_chop_data(idx,usable_slides,region_size):
    if random.random()>0.5:
        randSelect=random.randrange(0,usable_slides[idx]['num_regions'])
        chopData=[usable_slides[idx]['slide_loc'],usable_slides[idx]['xml_loc'],
            usable_slides[idx]['chop_array'][randSelect]]
    else:
        # print(list(usable_slides[idx]['annotations'].values()))
        if sum(usable_slides[idx]['annotations'].values())==0:
            randSelect=random.randrange(0,usable_slides[idx]['num_regions'])
            chopData=[usable_slides[idx]['slide_loc'],usable_slides[idx]['xml_loc'],
                usable_slides[idx]['chop_array'][randSelect]]
        else:
            selectedClass=0
            classIDs=usable_slides[idx]['annotations'].keys()
            classSamples=random.sample(classIDs,len(classIDs))
            for c in classSamples:
                if usable_slides[idx]['annotations'][c]==0:
                    pass
                else:
                    sampledRegionID=random.randrange(1,usable_slides[idx]['annotations'][c]+1)
                    break
            # tree = ET.parse(usable_slides[idx]['xml_loc'])
            # root = tree.getroot()
            # root=
            Regions=usable_slides[idx]['root'].findall("./Annotation[@Id='{}']/Regions/Region".format(c))
            sampledRegionID=Regions[sampledRegionID-1].attrib['Id']
            Verts = usable_slides[idx]['root'].findall("./Annotation[@Id='{}']/Regions/Region[@Id='{}']/Vertices/Vertex".format(c,sampledRegionID))
            randVertIdx=random.randrange(0,len(Verts))
            randVertX=int(float(Verts[randVertIdx].attrib['X']))
            randVertY=int(float(Verts[randVertIdx].attrib['Y']))
            randVertX=max(0,randVertX-region_size)
            randVertY=max(0,randVertY-region_size)
            chopData=[usable_slides[idx]['slide_loc'],usable_slides[idx]['xml_loc'],
                [randVertY,randVertX]]

    return chopData

def get_choppable_regions(slide,args,slideID,slideExt,mask_out_loc):
    slide_regions=[]
    choppable_regions_list=[]
    downsample = int(args.downsampleRate**.5) #down sample for each dimension
    region_size = int(args.boxSize*(downsample)) #Region size before downsampling
    step = int(region_size*(1-args.overlap_rate)) #Step size before downsampling
    if slideExt =='.scn':
        dim_x=int(slide.properties['openslide.bounds-width'])## add to columns
        dim_y=int(slide.properties['openslide.bounds-height'])## add to rows
        offsetx=int(slide.properties['openslide.bounds-x'])##start column
        offsety=int(slide.properties['openslide.bounds-y'])##start row
        index_y=np.array(range(offsety,offsety+dim_y,step))
        index_x=np.array(range(offsetx,offsetx+dim_x,step))
        index_y[-1]=(offsety+dim_y)-step
        index_x[-1]=(offsetx+dim_x)-step
    elif slideExt in ['.ndpi','.svs']:
        dim_x, dim_y=slide.dimensions
        offsetx=0
        offsety=0
        index_y=np.array(range(offsety,offsety+dim_y,step))
        index_x=np.array(range(offsetx,offsetx+dim_x,step))
        index_y[-1]=(offsety+dim_y)-step
        index_x[-1]=(offsetx+dim_x)-step

    fullSize=slide.level_dimensions[0]
    resRatio= args.chop_thumbnail_resolution
    ds_1=fullSize[0]/resRatio
    ds_2=fullSize[1]/resRatio
    out_mask_name=os.path.join(mask_out_loc,'_'.join([slideID,slideExt[1:]+'.png']))
    if not os.path.isfile(out_mask_name) or args.get_new_tissue_masks:
        print(out_mask_name)
        thumbIm=np.array(slide.get_thumbnail((ds_1,ds_2)))
        if slideExt =='.scn':
            xStt=int(offsetx/resRatio)
            xStp=int((offsetx+dim_x)/resRatio)
            yStt=int(offsety/resRatio)
            yStp=int((offsety+dim_y)/resRatio)
            thumbIm=thumbIm[yStt:yStp,xStt:xStp]
        choppable_regions=np.zeros((len(index_y),len(index_x)))
        hsv=rgb2hsv(thumbIm)
        g=gaussian(hsv[:,:,1],5)
        binary=(g>0.05).astype('bool')
        binary=binary_fill_holes(binary)
        imsave(out_mask_name.replace('.png','.jpeg'),thumbIm)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imsave(out_mask_name,binary.astype('uint8')*255)

    binary=(imread(out_mask_name)/255).astype('bool')
    choppable_regions=np.zeros((len(index_y),len(index_x)))
    chop_list=[]
    for idxy,yi in enumerate(index_y):
        for idxx,xj in enumerate(index_x):
            yStart = int(np.round((yi-offsety)/resRatio))
            yStop = int(np.round(((yi-offsety)+args.boxSize)/resRatio))
            xStart = int(np.round((xj-offsetx)/resRatio))
            xStop = int(np.round(((xj-offsetx)+args.boxSize)/resRatio))
            box_total=(xStop-xStart)*(yStop-yStart)

            if np.sum(binary[yStart:yStop,xStart:xStop])>(args.white_percent*box_total):
                choppable_regions[idxy,idxx]=1
                chop_list.append([index_y[idxy],index_x[idxx]])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        imsave(out_mask_name.replace('.png','_chopregions.png'),choppable_regions.astype('uint8')*255)

    return chop_list
