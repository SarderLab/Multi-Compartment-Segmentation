import openslide,glob,os
import numpy as np
from scipy.ndimage.morphology import binary_fill_holes
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv
from skimage.filters import gaussian
from skimage.morphology import binary_dilation, diamond
import cv2
from tqdm import tqdm
from skimage.io import imread,imsave



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
            # print("working slide... "+ slideID,end='\r')

            slide=openslide.OpenSlide(slide_loc)
            chop_array,num_slide_regions=get_choppable_regions(slide,args,slideID,slideExt,mask_out_loc)


            usable_slides.append({'slide_loc':slide_loc,'slideID':slideID,'slideExt':slideExt,'slide':slide,
                'chop_array':chop_array})
        for slide_meta in usable_slides:
            slide=openslide.OpenSlide(slide_meta['slide_loc'])
            print(slide_meta['slideID'])
            for corner in tqdm(slide_meta['chop_array']):
                if slide_meta['slideExt'] =='.scn':
                    print(corner)
                    cv2.imshow('test',np.array(slide.read_region((corner[0],corner[1]),0,(args.boxSize,args.boxSize)))[:,:,:3])
                    cv2.waitKey(500) # waits until a key is pressed
                    # cv2.destroyAllWindows()
                elif slide_meta['slideExt'] in ['.ndpi','.svs']:
                    continue
                    cv2.imshow('test',np.array(slide.read_region((corner[1],corner[0]),0,(args.boxSize,args.boxSize)))[:,:,:3])
                    cv2.waitKey(500) # waits until a key is pressed
                    # cv2.destroyAllWindows()
            # print(slide_meta['chop_array'])
            # input()
        print('\n')
            # 'choppable_regions_x':})
        # for i,name in enumerate(lib['slides']):
        #     sys.stdout.write('Opening SVS headers: [{}/{}]\r'.format(i+1, len(lib['slides'])))
        #     sys.stdout.flush()
        #     slides.append(openslide.OpenSlide(name))
        # print('')
        #Flatten grid
    #     grid = []
    #     slideIDX = []
    #     for i,g in enumerate(lib['grid']):
    #         grid.extend(g)
    #         slideIDX.extend([i]*len(g))
    #
    #     print('Number of tiles: {}'.format(len(grid)))
    #     self.slidenames = lib['slides']
    #     self.slides = slides
    #     self.targets = lib['targets']
    #     self.grid = grid
    #     self.slideIDX = slideIDX
    #     self.transform = transform
    #     self.mode = None
    #     self.mult = lib['mult']
    #     self.size = int(np.round(224*lib['mult']))
    #     self.level = lib['level']
    # def setmode(self,mode):
    #     self.mode = mode
    # def maketraindata(self, idxs):
    #     self.t_data = [(self.slideIDX[x],self.grid[x],self.targets[self.slideIDX[x]]) for x in idxs]
    # def shuffletraindata(self):
    #     self.t_data = random.sample(self.t_data, len(self.t_data))
    # def __getitem__(self,index):
    #     if self.mode == 1:
    #         slideIDX = self.slideIDX[index]
    #         coord = self.grid[index]
    #         img = self.slides[slideIDX].read_region(coord,self.level,(self.size,self.size)).convert('RGB')
    #         if self.mult != 1:
    #             img = img.resize((224,224),Image.BILINEAR)
    #         if self.transform is not None:
    #             img = self.transform(img)
    #         return img
    #     elif self.mode == 2:
    #         slideIDX, coord, target = self.t_data[index]
    #         img = self.slides[slideIDX].read_region(coord,self.level,(self.size,self.size)).convert('RGB')
    #         if self.mult != 1:
    #             img = img.resize((224,224),Image.BILINEAR)
    #         if self.transform is not None:
    #             img = self.transform(img)
    #         return img, target
    # def __len__(self):
    #     if self.mode == 1:
    #         return len(self.grid)
    #     elif self.mode == 2:
    #         return len(self.t_data)
    #


# img = self.slides[slideIDX].read_region(coord,self.level,(self.size,self.size)).convert('RGB')

def get_choppable_regions(slide,args,slideID,slideExt,mask_out_loc):
    slide_regions=[]
    choppable_regions_list=[]

    downsample = int(args.downsampleRate**.5) #down sample for each dimension
    region_size = int(args.boxSize*(downsample)) #Region size before downsampling
    step = int(region_size*(1-args.overlap_percent)) #Step size before downsampling

    if slideExt =='.scn':

        for i in range(0,1000):#arbitrary number of max regions
            try:
                dim_x=int(slide.properties[''.join(['openslide.region[',str(i),'].height'])])## add to columns
                dim_y=int(slide.properties[''.join(['openslide.region[',str(i),'].width'])])## add to rows
                offsetx=int(slide.properties[''.join(['openslide.region[',str(i),'].x'])])##start column
                offsety=int(slide.properties[''.join(['openslide.region[',str(i),'].y'])])##start row
                slide_regions.append([offsetx,offsety,dim_x,dim_y])
            except KeyError:
                break

        for p in slide.properties.keys():
            print(p,slide.properties[p])
        input()
        # print(dim_x,dim_y,offsetx,offsety)
    elif slideExt in ['.ndpi','.svs']:
        # return
        dim_x, dim_y=slide.dimensions

        offsetx=0
        offsety=0
        slide_regions.append([offsetx,offsety,dim_x,dim_y])



    fullSize=slide.level_dimensions[0]
    resRatio= args.chop_thumbnail_resolution
    ds_1=fullSize[0]/resRatio
    ds_2=fullSize[1]/resRatio
    if args.get_new_tissue_masks:
        thumbIm=np.array(slide.get_thumbnail((ds_1,ds_2)))
    # plt.imshow(thumbIm)
    # plt.show()
    for idx,sr in enumerate(slide_regions):
        print(sr)
        out_mask_name=os.path.join(mask_out_loc,'_'.join([slideID,slideExt[1:],str(idx)+'.png']))
        index_y=np.array(range(sr[1],sr[1]+sr[3],step))
        index_x=np.array(range(sr[0],sr[0]+sr[2],step))
        index_y[-1]=(sr[1]+sr[3])-step
        index_x[-1]=(sr[0]+sr[2])-step

        if not args.get_new_tissue_masks:
            try:
                binary=(imread(out_mask_name)/255).astype('bool')
            except:
                print('failed to load mask for '+ out_mask_name)
                print('please set get_new_tissue masks to True')
                exit()
            if slideExt =='.scn':
                choppable_regions=np.zeros((len(index_x),len(index_y)))
            elif slideExt in ['.ndpi','.svs']:
                choppable_regions=np.zeros((len(index_y),len(index_x)))
        else:
            # print('Getting new ')
            print(out_mask_name)

            # im=thumbIm[xStart:xStop,yStart:yStop,:]
            # choppable_regions=np.zeros((len(index_x),len(index_y)))
            if slideExt =='.scn':
                THxStart=int(sr[1]/resRatio)
                THxStop=int((sr[1]+sr[2])/resRatio)
                THyStart=int(sr[0]/resRatio)
                THyStop=int((sr[0]+sr[3])/resRatio)
                im=thumbIm[THxStart:THxStop,THyStart:THyStop,:]
                choppable_regions=np.zeros((len(index_x),len(index_y)))
            elif slideExt in ['.ndpi','.svs']:
                THxStart=int(sr[0]/resRatio)
                THxStop=int((sr[0]+sr[2])/resRatio)
                THyStart=int(sr[1]/resRatio)
                THyStop=int((sr[1]+sr[3])/resRatio)
                im=thumbIm[THyStart:THyStop,THxStart:THxStop,:]
                choppable_regions=np.zeros((len(index_y),len(index_x)))
            hsv=rgb2hsv(im)
            g=gaussian(hsv[:,:,1],5)
            binary=(g>0.05).astype('bool')
            binary=binary_fill_holes(binary)
            imsave(out_mask_name.replace('.png','.jpeg'),im)
            imsave(out_mask_name,binary.astype('uint8')*255)


        chop_list=[]
        for idxy,yi in enumerate(index_y):
            for idxx,xj in enumerate(index_x):
                yStart = int(np.round((yi-sr[1])/resRatio))
                yStop = int(np.round(((yi-sr[1])+args.boxSize)/resRatio))
                xStart = int(np.round((xj-sr[0])/resRatio))
                xStop = int(np.round(((xj-sr[0])+args.boxSize)/resRatio))
                box_total=(xStop-xStart)*(yStop-yStart)
                if slideExt =='.scn':
                    # print(xStart,xStop,yStart,yStop)
                    # print(np.sum(binary[xStart:xStop,yStart:yStop]),args.white_percent,box_total)
                    # plt.imshow(binary[xStart:xStop,yStart:yStop])
                    # plt.show()
                    if np.sum(binary[xStart:xStop,yStart:yStop])>(args.white_percent*box_total):

                        choppable_regions[idxx,idxy]=1
                        chop_list.append([index_x[idxx],index_y[idxy]])

                elif slideExt in ['.ndpi','.svs']:
                    if np.sum(binary[yStart:yStop,xStart:xStop])>(args.white_percent*box_total):
                        choppable_regions[idxy,idxx]=1
                        chop_list.append([index_y[idxy],index_x[idxx]])

        imsave(out_mask_name.replace('.png','_chopregions.png'),choppable_regions.astype('uint8')*255)

        # plt.imshow(choppable_regions)
        # plt.show()
    choppable_regions_list.extend(chop_list)
        # plt.subplot(131)
        # plt.imshow(im)
        # plt.subplot(132)
        # plt.imshow(binary)
        # plt.subplot(133)
        # plt.imshow(choppable_regions)
        # plt.show()
    return choppable_regions_list,idx+1
