import openslide,glob,os
import numpy as np
from scipy.ndimage.morphology import binary_fill_holes
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv
from skimage.filters import gaussian
# from skimage.morphology import binary_dilation, diamond
# import cv2
from tqdm import tqdm
from skimage.io import imread,imsave
import multiprocessing
from joblib import Parallel, delayed

def save_thumb(args,slide_loc):
    print(slide_loc)
    slideID,slideExt=os.path.splitext(slide_loc.split('/')[-1])
    slide=openslide.OpenSlide(slide_loc)
    if slideExt =='.scn':
        dim_x=int(slide.properties['openslide.bounds-width'])## add to columns
        dim_y=int(slide.properties['openslide.bounds-height'])## add to rows
        offsetx=int(slide.properties['openslide.bounds-x'])##start column
        offsety=int(slide.properties['openslide.bounds-y'])##start row
    elif slideExt in ['.ndpi','.svs']:
        dim_x, dim_y=slide.dimensions
        offsetx=0
        offsety=0

    # fullSize=slide.level_dimensions[0]
    # resRatio= args.chop_thumbnail_resolution
    # ds_1=fullSize[0]/resRatio
    # ds_2=fullSize[1]/resRatio
    # thumbIm=np.array(slide.get_thumbnail((ds_1,ds_2)))
    # if slideExt =='.scn':
    #     xStt=int(offsetx/resRatio)
    #     xStp=int((offsetx+dim_x)/resRatio)
    #     yStt=int(offsety/resRatio)
    #     yStp=int((offsety+dim_y)/resRatio)
    #     thumbIm=thumbIm[yStt:yStp,xStt:xStp]
    # imsave(slide_loc.replace(slideExt,'_thumb.jpeg'),thumbIm)
    slide.associated_images['label'].save(slide_loc.replace(slideExt,'_label.png'))
    # imsave(slide_loc.replace(slideExt,'_label.png'),slide.associated_images['label'])


def get_image_thumbnails(args):
    assert args.target is not None, 'Location of images must be provided'
    all_slides=[]
    for ext in args.wsi_ext.split(','):
        all_slides.extend(glob.glob(args.target+'/*'+ext))
    Parallel(n_jobs=multiprocessing.cpu_count())(delayed(save_thumb)(args,slide_loc) for slide_loc in tqdm(all_slides))
    # for slide_loc in tqdm(all_slides):

class WSIPredictLoader():
    def __init__(self,args, wsi_directory=None, transform=None):
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
            print("working slide... "+ slideID,end='\r')

            slide=openslide.OpenSlide(slide_loc)
            chop_array=get_choppable_regions(slide,args,slideID,slideExt,mask_out_loc)
            mag_x=np.round(float(slide.properties['openslide.mpp-x']),2)
            mag_y=np.round(float(slide.properties['openslide.mpp-y']),2)
            print(mag_x,mag_y)
            usable_slides.append({'slide_loc':slide_loc,'slideID':slideID,'slideExt':slideExt,'slide':slide,
                'chop_array':chop_array,'mag':[mag_x,mag_y]})
        self.usable_slides= usable_slides
        self.boxSize40X = args.boxSize
        self.boxSize20X = int(args.boxSize)/2

class WSITrainingLoader():
    def __init__(self,args, wsi_directory=None, transform=None):
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
            print("working slide... "+ slideID,end='\r')

            slide=openslide.OpenSlide(slide_loc)
            chop_array=get_choppable_regions(slide,args,slideID,slideExt,mask_out_loc)
            mag_x=np.round(float(slide.properties['openslide.mpp-x']),2)
            mag_y=np.round(float(slide.properties['openslide.mpp-y']),2)
            print(mag_x,mag_y)
            usable_slides.append({'slide_loc':slide_loc,'slideID':slideID,'slideExt':slideExt,'slide':slide,
                'chop_array':chop_array,'mag':[mag_x,mag_y]})
        self.usable_slides= usable_slides
        self.boxSize40X = args.boxSize
        self.boxSize20X = int(args.boxSize)/2

        print('\n')

def get_choppable_regions(slide,args,slideID,slideExt,mask_out_loc):
    slide_regions=[]
    choppable_regions_list=[]

    downsample = int(args.downsampleRate**.5) #down sample for each dimension
    region_size = int(args.boxSize*(downsample)) #Region size before downsampling
    step = int(region_size*(1-args.overlap_percent)) #Step size before downsampling

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
    if args.get_new_tissue_masks:
        thumbIm=np.array(slide.get_thumbnail((ds_1,ds_2)))
        if slideExt =='.scn':
            xStt=int(offsetx/resRatio)
            xStp=int((offsetx+dim_x)/resRatio)
            yStt=int(offsety/resRatio)
            yStp=int((offsety+dim_y)/resRatio)
            thumbIm=thumbIm[yStt:yStp,xStt:xStp]
            # plt.imshow(thumbIm)
            # plt.show()
            # input()
    # plt.imshow(thumbIm)
    # plt.show()

    out_mask_name=os.path.join(mask_out_loc,'_'.join([slideID,slideExt[1:]+'.png']))


    if not args.get_new_tissue_masks:
        try:
            binary=(imread(out_mask_name)/255).astype('bool')
        except:
            print('failed to load mask for '+ out_mask_name)
            print('please set get_new_tissue masks to True')
            exit()
        # if slideExt =='.scn':
        #     choppable_regions=np.zeros((len(index_x),len(index_y)))
        # elif slideExt in ['.ndpi','.svs']:
        choppable_regions=np.zeros((len(index_y),len(index_x)))
    else:
        print(out_mask_name)
        # if slideExt =='.scn':
        #     choppable_regions=np.zeros((len(index_x),len(index_y)))
        # elif slideExt in ['.ndpi','.svs']:
        choppable_regions=np.zeros((len(index_y),len(index_x)))

        hsv=rgb2hsv(thumbIm)
        g=gaussian(hsv[:,:,1],5)
        binary=(g>0.05).astype('bool')
        binary=binary_fill_holes(binary)
        imsave(out_mask_name.replace('.png','.jpeg'),thumbIm)
        imsave(out_mask_name,binary.astype('uint8')*255)

    chop_list=[]
    for idxy,yi in enumerate(index_y):
        for idxx,xj in enumerate(index_x):
            yStart = int(np.round((yi-offsety)/resRatio))
            yStop = int(np.round(((yi-offsety)+args.boxSize)/resRatio))
            xStart = int(np.round((xj-offsetx)/resRatio))
            xStop = int(np.round(((xj-offsetx)+args.boxSize)/resRatio))
            box_total=(xStop-xStart)*(yStop-yStart)
            if slideExt =='.scn':
                # print(xStart,xStop,yStart,yStop)
                # print(np.sum(binary[xStart:xStop,yStart:yStop]),args.white_percent,box_total)
                # plt.imshow(binary[xStart:xStop,yStart:yStop])
                # plt.show()
                if np.sum(binary[yStart:yStop,xStart:xStop])>(args.white_percent*box_total):

                    choppable_regions[idxy,idxx]=1
                    chop_list.append([index_y[idxy],index_x[idxx]])

            elif slideExt in ['.ndpi','.svs']:
                if np.sum(binary[yStart:yStop,xStart:xStop])>(args.white_percent*box_total):
                    choppable_regions[idxy,idxx]=1
                    chop_list.append([index_y[idxy],index_x[idxx]])

    imsave(out_mask_name.replace('.png','_chopregions.png'),choppable_regions.astype('uint8')*255)

        # plt.imshow(choppable_regions)
        # plt.show()
    # choppable_regions_list.extend(chop_list)
    # plt.subplot(131)
    # plt.imshow(thumbIm)
    # plt.subplot(132)
    # plt.imshow(binary)
    # plt.subplot(133)
    # plt.imshow(choppable_regions)
    # plt.show()
    return chop_list
