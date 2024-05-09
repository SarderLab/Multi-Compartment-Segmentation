import os, cv2
import numpy as np

import lxml.etree as ET

from matplotlib import path
from skimage.color import rgb2lab,rgb2hsv

from .xml_to_mask_minmax import write_minmax_to_xml
import xlsxwriter
import multiprocessing
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage import binary_fill_holes

from tqdm import tqdm
import time
from tiffslide import TiffSlide
from joblib import Parallel, delayed
from skimage.color import rgb2hsv

from skimage.filters import *

def getKidneyReferenceFeatures(args):

    folder = args.base_dir

    # assert args.target is not None, 'Directory of xmls must be specified, use --target /path/to/files.xml'
    # assert args.wsis is not None, 'Directory of WSIs must be specified, use --wsis /path/to/wsis'
    if args.platform == 'DSA':
        import girder_client
        gc = girder_client.GirderClient(apiUrl=args.girderApiUrl)
        gc.setToken(args.girderToken)

        file_name = args.file.split('/')[-1]
        slide_item_id = args.item_id
        output_dir = args.base_dir
        slide_name,slideExt=file_name.split('.')
        items=[(args.file, args.xml_path)]

    elif args.platform == 'HPG':
        image_files = [image_name for _, _, files in os.walk(folder) for image_name in files if image_name.endswith('.xml')]
        image_names = [os.path.join(folder, f.split('.')[0]) for f in image_files]
        slideExt = args.ext
        output_dir = args.output_dir
        # each item in items is a tuple of (images, annotations)
        items = [(f + slideExt, f + '.xml') for f in image_names]

    else:
        raise Exception("Please Enter a valid Platform, DSA or HPG")

    for i in range(len(items)):

        svsfile, xmlfile = items[i]

        print(xmlfile,'here')
        write_minmax_to_xml(xmlfile)


        all_contours = {'1':[],'2':[],'3':[],'4':[],'5':[],'6':[]}
        # cortex medulla glomeruli scl_glomeruli tubules arteries(ioles)
        tree = ET.parse(xmlfile)
        root = tree.getroot()
        basename=os.path.splitext(xmlfile)[0]
        for Annotation in root.findall("./Annotation"): # for all annotations
            annotationID = Annotation.attrib['Id']
            if annotationID not in ['1','2','3','4','5','6']:
                pass
            else:
                for Region in Annotation.findall("./*/Region"): # iterate on all region
                    verts=[]
                    for Vert in Region.findall("./Vertices/Vertex"): # iterate on all vertex in region
                        verts.append([int(float(Vert.attrib['X'])),int(float(Vert.attrib['Y']))])
                    all_contours[annotationID].append(np.array(verts))

        cortexcontour=[]
        cortexcodes=[]
        medullacontour=[]
        medullacodes=[]
        cortexarea=0
        medullaarea=0

        slide=TiffSlide(svsfile)
        if slideExt =='.scn':
            dim_x=int(slide.properties['tiffslide.bounds-width'])## add to columns
            dim_y=int(slide.properties['tiffslide.bounds-height'])## add to rows
            offsetx=int(slide.properties['tiffslide.bounds-x'])##start column
            offsety=int(slide.properties['tiffslide.bounds-y'])##start row

        elif slideExt in ['.ndpi','.svs','.tiff']:
            dim_x, dim_y=slide.dimensions
            offsetx=0
            offsety=0


        fullSize=slide.level_dimensions[0]
        resRatio= args.chop_thumbnail_resolution
        ds_1=fullSize[0]/resRatio
        ds_2=fullSize[1]/resRatio

        thumbIm=np.array(slide.get_thumbnail((ds_1,ds_2)))
        slide.close()
        if slideExt =='.scn':
            xStt=int(offsetx/resRatio)
            xStp=int((offsetx+dim_x)/resRatio)
            yStt=int(offsety/resRatio)
            yStp=int((offsety+dim_y)/resRatio)
            thumbIm=thumbIm[yStt:yStp,xStt:xStp]

        hsv=rgb2hsv(thumbIm)

        binary=(hsv[:,:,1]>0.1).astype('bool')

        black=hsv[:,:,2]<.5

        binary[black]=0
        binary=binary_fill_holes(binary)

        total_tissue_area=np.sum(binary)*resRatio*resRatio
  
        for contour in all_contours['1']:
            cortexcontour.extend(contour)
            cortexarea+=cv2.contourArea(contour)
            cortexcodes.extend([path.Path.MOVETO])
            for i in range(1,np.shape(contour)[0]-1):
                cortexcodes.extend([path.Path.LINETO])
            cortexcodes.extend([path.Path.CLOSEPOLY])
        if len(cortexcontour)>0:
            cortex_path=path.Path(cortexcontour,codes=cortexcodes)
        else:
            cortex_path=None

        for contour in all_contours['2']:
            medullacontour.extend(contour)
            medullacodes.extend([path.Path.MOVETO])
            medullaarea+=cv2.contourArea(contour)
            for i in range(1,np.shape(contour)[0]-1):
                medullacodes.extend([path.Path.LINETO])
            medullacodes.extend([path.Path.CLOSEPOLY])
        if len(medullacontour)>0:
            medulla_path=path.Path(medullacontour,codes=medullacodes)
        else:
            medulla_path=None
        pseudocortexarea=total_tissue_area-medullaarea
        #
        xlsx_path = os.path.join(output_dir, os.path.basename(svsfile).split('.')[0] +'_extended_clinical'+'.xlsx')
        workbook=xlsxwriter.Workbook(xlsx_path)
        worksheet1 = workbook.add_worksheet('Summary')
        worksheet2 = workbook.add_worksheet('Interstitium')
        worksheet3 = workbook.add_worksheet('Glomeruli')
        worksheet4 = workbook.add_worksheet('Sclerosed glomeruli')
        worksheet5 = workbook.add_worksheet('Tubules')
        worksheet6 = workbook.add_worksheet('Arteries - Arterioles')
        cores=multiprocessing.cpu_count()
        MODx=np.zeros((3,))
        MODy=np.zeros((3,))
        MODz=np.zeros((3,))
        MODx[0]= 0.644211
        MODy[0]= 0.716556
        MODz[0]= 0.266844

        MODx[1]= 0.175411
        MODy[1]= 0.972178
        MODz[1]= 0.154589

        MODx[2]= 0.0
        MODy[2]= 0.0
        MODz[2]= 0.0
        MOD=[MODx,MODy,MODz]
        glom_features=Parallel(n_jobs=cores)(delayed(points_to_features_glom)(points,
            args,args.min_size[2],cortex_path,medulla_path) for points in tqdm(all_contours['3'],colour='yellow',unit='Glomerulus',leave=False))
        sglom_features=Parallel(n_jobs=cores)(delayed(points_to_features_glom)(points,
            args,args.min_size[3],cortex_path,medulla_path) for points in tqdm(all_contours['4'],colour='red',unit='Scl. glomerulus',leave=False))
        tub_features=Parallel(n_jobs=cores)(delayed(points_to_features_tub)(points,
            args,args.min_size[4],cortex_path,medulla_path) for points in tqdm(all_contours['5'],colour='blue',unit='Tubule',leave=False))

        art_features=Parallel(n_jobs=cores)(delayed(points_to_features_art)(points,
            args,args.min_size[5],cortex_path,medulla_path,svsfile,MOD) for points in tqdm(all_contours['6'],colour='magenta',unit='Artery(-iole)',leave=False))
        print('Generating output file..')
        glom_features=np.array([i for i in glom_features if i is not None])
        sglom_features=np.array([i for i in sglom_features if i is not None])
        tub_features=np.array([i for i in tub_features if i is not None])
        art_features=np.array([i for i in art_features if i is not None])

        # gloms_features=[i for i in glom_features if i[0]>args.min_sizes[2]]
        # sglom_features=[i for i in sglom_features if i[0]>args.min_sizes[3]]

        # cortexgloms=[i for i in glom_features if not i[3]]
        cortextubs=np.array([i for i in tub_features if not i[3]])
        cortexarts=np.array([i for i in art_features if not i[3]])

        medullatubs=np.array([i for i in tub_features if i[3]])
        medullaarts=np.array([i for i in art_features if i[3]])


        if pseudocortexarea>0:
            cortex_glom_area=np.sum(np.array(glom_features)[:,0])
            cortex_glom_density=float(cortex_glom_area)/float(pseudocortexarea)
            cortex_tub_area=np.sum(cortextubs[:,0])
            cortex_tub_density=float(cortex_tub_area)/float(pseudocortexarea)
            cortex_art_area=np.sum(cortexarts[:,0])
            cortex_art_density=float(cortex_art_area)/float(pseudocortexarea)
            # downsample_cortex=get_downsample_cortex(args,all_contours['1'])
            # exit()
            # capillary_densities=Parallel(n_jobs=cores)(delayed(get_capillary_densities)(points,
            #     args,args.min_size[4],cortex_path,medulla_path,wsi) for points in tqdm(all_contours['6'],colour='magenta',unit='Artery(-iole)'))
        else:
            cortex_glom_density=None
            cortex_tub_density=None
            cortex_art_density=None
        if medullaarea>0 and len(medullatubs)>0:

            medulla_tub_area=np.sum(medullatubs[:,0])
            if len(medullaarts)>0:
                medulla_art_area=np.sum(medullaarts[:,0])
                medulla_art_density=float(medulla_art_area)/float(medullaarea)
            else:
                medulla_art_density=None
            medulla_tub_density=float(medulla_tub_area)/float(medullaarea)
        else:
            medulla_tub_density=None
            medulla_art_density=None

        worksheet1.write(0,0,'Glomerular density - count:')
        worksheet1.write(0,1,len(glom_features)/pseudocortexarea)
        worksheet1.write(1,0,'Average glomerular area:')
        worksheet1.write(1,1,np.mean(glom_features[:,0]))
        worksheet1.write(2,0,'Std glomerular area:')
        worksheet1.write(2,1,np.std(glom_features[:,0]))
        worksheet1.write(3,0,'Average glomerular radius:')
        worksheet1.write(3,1,np.mean(glom_features[:,1]))
        worksheet1.write(4,0,'Std glomerular radius:')
        worksheet1.write(4,1,np.std(glom_features[:,1]))
        worksheet1.write(5,0,'Glomerulosclerosis density - count')
        worksheet1.write(6,0,'Average scl.glomerular area:')
        worksheet1.write(7,0,'Std scl.glomerular area:')
        worksheet1.write(8,0,'Average scl.glomerular radius:')
        worksheet1.write(9,0,'Std scl.glomerular radius:')
        if len(sglom_features)>0:
            worksheet1.write(5,1,len(sglom_features)/pseudocortexarea)
            worksheet1.write(6,1,np.mean(sglom_features[:,0]))
            worksheet1.write(7,1,np.std(sglom_features[:,0]))
            worksheet1.write(8,1,np.mean(sglom_features[:,1]))
            worksheet1.write(9,1,np.std(sglom_features[:,1]))

        worksheet1.write(10,0,'Cortical tubular density')

        worksheet1.write(11,0,'Average cortical tubular area:')
        worksheet1.write(12,0,'Std cortical tubular area:')
        worksheet1.write(13,0,'Average cortical tubular radius:')
        worksheet1.write(14,0,'Std cortical tubular radius:')

        worksheet1.write(15,0,'Average medullary tubular area:')
        worksheet1.write(16,0,'Std medullary tubular area:')
        worksheet1.write(17,0,'Average medullary tubular radius:')
        worksheet1.write(18,0,'Std medullary tubular radius:')
        if pseudocortexarea>0:
            cortextubs=np.array(cortextubs)
            worksheet1.write(10,1,len(cortextubs)/pseudocortexarea)

            worksheet1.write(11,1,np.mean(cortextubs[:,0]))
            worksheet1.write(12,1,np.std(cortextubs[:,0]))
            worksheet1.write(13,1,np.mean(cortextubs[:,1]))
            worksheet1.write(14,1,np.std(cortextubs[:,1]))
        if medullaarea>0:
            if len(medullatubs)>0:
                worksheet1.write(15,1,np.mean(medullatubs[:,0]))
                worksheet1.write(16,1,np.std(medullatubs[:,0]))
                worksheet1.write(17,1,np.mean(medullatubs[:,1]))
                worksheet1.write(18,1,np.std(medullatubs[:,1]))
            else:
                worksheet1.write(15,1,0)
                worksheet1.write(16,1,0)
                worksheet1.write(17,1,0)
                worksheet1.write(18,1,0)

        worksheet1.write(19,0,'Cortical arterial(olar) density')
        worksheet1.write(19,1,len(cortexarts)/pseudocortexarea)
        worksheet1.write(20,0,'Average lumen to wall ratio:')
        ltwr=np.array([i for i in art_features[:,4] if i is not None])
        worksheet1.write(20,1,np.mean(ltwr))

        worksheet1.write(21,0,'Glomerulosclerosis ratio:')
        worksheet1.write(21,1,float(len(sglom_features))/float(len(sglom_features)+len(glom_features)))
        worksheet1.write(22,0,'Cortical glomerular density - area:')
        worksheet1.write(22,1,cortex_glom_density)
        worksheet1.write(23,0,'Cortical tubular density - area')
        worksheet1.write(23,1,cortex_tub_density)
        worksheet1.write(24,0,'Cortical artery(iole) density - area')
        worksheet1.write(24,1,cortex_art_density)
        worksheet1.write(25,0,'Medullary tubular density - area')
        worksheet1.write(25,1,medulla_tub_density)

        worksheet1.write(26,0,'Medullary arteriole density - area')
        worksheet1.write(26,1,medulla_art_density)
        worksheet1.write(27,0,'Gloms/cortex tubules ratio:')
        worksheet1.write(27,1,np.sum(glom_features[:,0])/np.sum(cortextubs[:,0]))
        cInterstitial_area=cortexarea-np.sum(cortextubs[:,0])-np.sum(cortexarts[:,0])-np.sum(glom_features[:,0])
        if len(medullatubs)>0:
            mInterstitial_area=medullaarea-np.sum(medullatubs[:,0])
        else:
            mInterstitial_area=0
        if len(sglom_features)>0:
            cInterstitial_area-=np.sum(sglom_features[:,0])
        worksheet1.write(28,0,'Cortical interstitial density')
        worksheet1.write(28,1,cInterstitial_area/pseudocortexarea)
        worksheet1.write(29,0,'Overall tubule density - count')
        worksheet1.write(29,1,len(tub_features)/total_tissue_area)
        worksheet1.write(29,0,'Total nephron density - count')
        worksheet1.write(29,1,(len(tub_features)+len(sglom_features)+len(glom_features))/total_tissue_area)
        worksheet1.write(30,0,'Medullary interstitial density')
        if medullaarea>0:
            worksheet1.write(30,1,mInterstitial_area/medullaarea)
        worksheet1.write(31,0,'Cortical tubule count')
        worksheet1.write(31,1,len(cortextubs))
        worksheet1.write(32,0,'Cortical artery/arteriole count')
        worksheet1.write(32,1,len(cortexarts))
        worksheet1.write(33,0,'Glomerulus count')
        worksheet1.write(33,1,len(glom_features))
        worksheet1.write(34,0,'sGlomerulus count')
        worksheet1.write(34,1,len(sglom_features))

        worksheet3.write(0,0,'Area (pixel^2)')
        worksheet3.write(0,1,'Radius (pixel)')
        worksheet3.write(0,2,'x1')
        worksheet3.write(0,3,'x2')
        worksheet3.write(0,4,'y1')
        worksheet3.write(0,5,'y2')

        for idx,glom in enumerate(glom_features):
            worksheet3.write(idx+1,0,glom[0])
            worksheet3.write(idx+1,1,glom[1])

            worksheet3.write(idx+1,2,glom[4])
            worksheet3.write(idx+1,3,glom[5])
            worksheet3.write(idx+1,4,glom[6])
            worksheet3.write(idx+1,5,glom[7])


        worksheet4.write(0,0,'Area (pixel^2)')
        worksheet4.write(0,1,'Radius (pixel)')

        worksheet4.write(0,2,'x1')
        worksheet4.write(0,3,'x2')
        worksheet4.write(0,4,'y1')
        worksheet4.write(0,5,'y2')

        for idx,sglom in enumerate(sglom_features):
            worksheet4.write(idx+1,0,sglom[0])
            worksheet4.write(idx+1,1,sglom[1])

            worksheet4.write(idx+1,2,sglom[4])
            worksheet4.write(idx+1,3,sglom[5])
            worksheet4.write(idx+1,4,sglom[6])
            worksheet4.write(idx+1,5,sglom[7])

        worksheet5.write(0,0,'Area (pixel^2)')
        worksheet5.write(0,1,'Radius (pixel)')
        worksheet5.write(0,2,'In Medulla')

        worksheet5.write(0,3,'x1')
        worksheet5.write(0,4,'x2')
        worksheet5.write(0,5,'y1')
        worksheet5.write(0,6,'y2')
        for idx,tub in enumerate(tub_features):
            worksheet5.write(idx+1,0,tub[0])
            worksheet5.write(idx+1,1,tub[1])
            worksheet5.write(idx+1,2,tub[3])

            worksheet5.write(idx+1,3,tub[4])
            worksheet5.write(idx+1,4,tub[5])
            worksheet5.write(idx+1,5,tub[6])
            worksheet5.write(idx+1,6,tub[7])

        worksheet6.write(0,0,'Area (pixel^2)')
        worksheet6.write(0,1,'Radius (pixel)')
        worksheet6.write(0,2,'Luminal ratio')

        worksheet6.write(0,3,'x1')
        worksheet6.write(0,4,'x2')
        worksheet6.write(0,5,'y1')
        worksheet6.write(0,6,'y2')
        for idx,art in enumerate(art_features):
            worksheet6.write(idx+1,0,art[0])
            worksheet6.write(idx+1,1,art[1])
            worksheet6.write(idx+1,2,art[4])

            worksheet6.write(idx+1,3,art[5])
            worksheet6.write(idx+1,4,art[6])
            worksheet6.write(idx+1,5,art[7])
            worksheet6.write(idx+1,6,art[8])

        workbook.close()
        if args.platform == 'DSA':
            gc.uploadFileToItem(slide_item_id, xlsx_path, reference=None, mimeType=None, filename=None, progressCallback=None)
            print('Girder file uploaded!')

        print('Done.')

def points_to_features_glom(points,args,min_size,cortex,medulla):
    a=cv2.contourArea(points)
    if a>min_size:
        # if cortex is not None:
        #     containedcortex=any(cortex.contains_points(points))
        # else:
        #     containedcortex=False
        if medulla is not None:
            containedmedulla=any(medulla.contains_points(points))
        else:
            containedmedulla=False
        xMin,xMax,yMin,yMax=[np.min(points[:,0]),np.max(points[:,0]),np.min(points[:,1]),np.max(points[:,1])]
        binary_mask=np.zeros((yMax-yMin,xMax-xMin))
        points[:,0]-=xMin
        points[:,1]-=yMin
        binary_mask=cv2.fillPoly(binary_mask,[points],1)
        dist=distance_transform_edt(binary_mask)


        return [a,np.max(dist),None,containedmedulla,yMin,yMax,xMin,xMax]

def points_to_features_tub(points,args,min_size,cortex,medulla):
    a=cv2.contourArea(points)
    if a>min_size:
        # if cortex is not None:
        #     containedcortex=any(cortex.contains_points(points))
        # else:
        #     containedcortex=False
        if medulla is not None:
            containedmedulla=any(medulla.contains_points(points))
        else:
            containedmedulla=False
        xMin,xMax,yMin,yMax=[np.min(points[:,0]),np.max(points[:,0]),np.min(points[:,1]),np.max(points[:,1])]
        binary_mask=np.zeros((yMax-yMin,xMax-xMin))
        points[:,0]-=xMin
        points[:,1]-=yMin
        binary_mask=cv2.fillPoly(binary_mask,[points],1)
        dist=distance_transform_edt(binary_mask)

        return [a,np.max(dist),None,containedmedulla,yMin,yMax,xMin,xMax]


def points_to_features_art(points,args,min_size,cortex,medulla,wsi,MOD):
    a=cv2.contourArea(points)
    resizeFlag=0
    if a>min_size:
        # if cortex is not None:
        #     containedcortex=any(cortex.contains_points(points))
        # else:
        #     containedcortex=False
        if medulla is not None:
            containedmedulla=any(medulla.contains_points(points))
        else:
            containedmedulla=False
        xMin,xMax,yMin,yMax=[np.min(points[:,0]),np.max(points[:,0]),np.min(points[:,1]),np.max(points[:,1])]
        image=np.array(TiffSlide(wsi).read_region((xMin,yMin),0,(xMax-xMin,yMax-yMin)))[:,:,:3]

        if xMax-xMin>5000 or yMax-yMin>5000:
            return [a,None,None,containedmedulla,None, yMin,yMax,xMin,xMax]
        binary_mask=np.zeros((int(yMax-yMin),int(xMax-xMin)))
        points[:,0]-=xMin
        points[:,1]-=yMin
        binary_mask=cv2.fillPoly(binary_mask,[points],1)

        LAB=rgb2lab(image)
        # HSV=rgb2hsv(image)

        LAB[:,:,0]/=100

        # stain1,stain2,stain3=deconvolution(image,MOD)
        #prevent divide by zero errors by adding very small nonzero
        # stain2=np.invert(stain2)/255
        # stain3=np.invert(stain3)
        # WSsatscaling=np.divide(LAB[:,:,0],HSV[:,:,1]+0.000000001)
        # WSsatscaling2=np.divide(LAB[:,:,0],stain2+0.000000001)
        # WSseg=WSsatscaling2>3
        # WSseg=LAB[:,:,0]>.65
        # fig,ax=try_all_threshold(LAB[:,:,0],figsize=(10,8))
        # plt.show()
        # WSseg=LAB[:,:,0]>threshold_lcoal(LAB[:,:,0],tolerance=2)

        dist=distance_transform_edt(binary_mask)
        # localThresh=rank.otsu(LAB[:,:,0],disk(5))
        distMax=np.max(dist)
        # print(distMax)
        # tim=threshold_local(grayImage,block_size=3,method='mean')
        WSseg=LAB[:,:,0]>.7

        # WSseg=binary_fill_holes(WSseg)

        # WSseg=binary_dilation(WSseg,disk(2))
        WSseg[WSseg!=binary_mask.astype('bool')]=0
        WSseg=binary_fill_holes(WSseg)
        # WSseg=binary_opening(WSseg,disk(1))
        # WSseg=binary_erosion(WSseg,disk(2))

        WSdist=distance_transform_edt(WSseg)

        # plt.subplot(221)
        # plt.imshow(image)
        # plt.subplot(222)
        # plt.imshow(WSseg)
        # plt.subplot(223)
        # plt.imshow(grayImage)
        # plt.subplot(224)
        # plt.imshow(image)
        # plt.title(np.max(WSdist)/np.max(dist))
        # plt.show()
        # if resizeFlag:
        #
        #     return [a,distMax*2,containedcortex,containedmedulla,(np.max(WSdist)*2)/(distMax*2)]
        # else:
        return [a,distMax,None,containedmedulla,np.max(WSdist)/distMax,yMin,yMax,xMin,xMax]



# #     full_list.append(tubule_features)
# # full_list= pd.concat(full_list)
# # print(full_list)
# # exit()
