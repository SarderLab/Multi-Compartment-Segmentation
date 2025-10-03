import numpy as np
import os,glob, openslide#sys, argparse, warnings, ,
import lxml.etree as ET
from skimage.io import imsave#,imread
from tqdm import tqdm
# import matplotlib.pyplot as plt
from .xml_to_mask_minmax import get_annotated_ROIs_coords_withdots
from skimage.color import rgb2hsv#rgb2lab
from scipy.ndimage import binary_fill_holes
from skimage.filters import gaussian

def coordinate_pairs(v1,v2):
    for i in v1:
        for j in v2:
            yield i,j
            
def randomCropGenerator(args):
    # define folder structure dict
    assert args.target is not None, 'Please provide path to input data with --target'
    xml_color = [65280, 16776960,65535, 255, 16711680, 33023]
    dirs = {'outDir': args.base_dir + '/' + args.project + args.outDir}
    dirs['training_data_dir'] = '/TRAINING_data/'

    downsample = int(args.downsampleRateHR**.5)
    region_size = int(args.boxSize*(downsample))
    step = int((region_size-(args.bordercrop*2))*(1-args.overlap_percentHR))


    # get all WSIs
    WSIs = []
    usable_ext=args.wsi_ext.split(',')
    for ext in usable_ext:
        WSIs.extend(glob.glob(args.target+ '/*' + ext))

    broken_slides=[]
    for wsi in tqdm(WSIs):
        extsplit = os.path.splitext(wsi)
        basename = extsplit[0]
        extname = extsplit[-1]
        print(basename)
        try:
            slide=openslide.OpenSlide(wsi)
        except:
            broken_slides.append(wsi)
            continue
        xml_path=wsi.replace(extname,'.xml')
        if extname=='.scn':
            dim_y=int(slide.properties['openslide.bounds-height'])
            dim_x=int(slide.properties['openslide.bounds-width'])
            offsetx=int(slide.properties['openslide.bounds-x'])
            offsety=int(slide.properties['openslide.bounds-y'])
            # print(dim_x,dim_y,offsetx,offsety)
        else:
            dim_x, dim_y=slide.dimensions
            offsetx=0
            offsety=0


        index_y=np.array(range(offsety,dim_y+offsety,step))
        index_x=np.array(range(offsetx,dim_x+offsetx,step))

        choppable_regions=np.zeros((len(index_y),len(index_x)))


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


        # wsiMask = xml_to_mask(wsi.replace(extname,'.xml'),[0,0],[dim_x,dim_y])
        tree = ET.parse(xml_path)
        print('Segmenting tissue ...\n')
        totalpatches=len(index_x)*len(index_y)
        for patch_coords in tqdm(chop_list,colour='green',leave=False):

        # for i,j in coordinate_pairs(index_y,index_x):

            yEnd = min(dim_y+offsety,patch_coords[0]+region_size)
            xEnd = min(dim_x+offsetx,patch_coords[1]+region_size)
            # yStart_small = int(np.round((i-offsety)/resRatio))
            # yStop_small = int(np.round(((i-offsety)+args.boxSize)/resRatio))
            # xStart_small = int(np.round((j-offsetx)/resRatio))
            # xStop_small = int(np.round(((j-offsetx)+args.boxSize)/resRatio))
            yStart_small = int(np.round((patch_coords[0]-offsety)/resRatio))
            yStop_small = int(np.round(((yEnd-offsety))/resRatio))
            xStart_small = int(np.round((patch_coords[1]-offsetx)/resRatio))
            xStop_small = int(np.round(((xEnd-offsetx))/resRatio))
            box_total=(xStop_small-xStart_small)*(yStop_small-yStart_small)

            if np.sum(binary[yStart_small:yStop_small,xStart_small:xStop_small])>(args.white_percent*box_total):

                xLen=xEnd-patch_coords[1]
                yLen=yEnd-patch_coords[0]

                dxS=patch_coords[1]
                dyS=patch_coords[0]
                dxE=patch_coords[1]+xLen
                dyE=patch_coords[0]+yLen

                im=np.array(slide.read_region((dxS,dyS),0,(xLen,yLen)))[:,:,:3]
                annotationData,annotationTypes,linkIDs=get_annotated_ROIs_coords_withdots(xml_path,[dxS,dyS],[xLen,yLen],['50'],downsample=1,tree=tree)
                outpath=xml_path.replace('.xml','_'.join(['',str(dxS),str(dyS),str(xLen),str(yLen)])+'.xml')
                xml_suey(annotationData,annotationTypes,linkIDs, args,outpath,[dxS,dyS],[xLen,yLen], [offsetx,offsety],xml_color)
                imsave(outpath.replace('.xml','.tiff'),im)


        slide.close()
def xml_suey(annotationData,annotationTypes,linkIDs, args,outpath,local_offset,size, glob_offset,xml_color):

    annotationClasses=np.array(list(annotationData.keys())).astype('int32')
    Annotations = xml_create()

    for i in range(np.max(annotationClasses))[1:]:
        # print('\t working on: annotationID ' + str(i))
        Annotations = xml_add_annotation(Annotations=Annotations, annotationID=i,annotationType=annotationTypes[str(i)],classLink=linkIDs[str(i)],xml_color=xml_color)
        contours=annotationData[str(i)]

        # pointsList = get_contour_points(binary_mask, args=args, downsample=downsample,value=value,offset={'X':glob_offset[0],'Y':glob_offset[1]})
        for j in range(len(contours)):

            pointList = contours[j]
            if len(pointList)>0:
                pointList[:,0]-=local_offset[0]
                pointList[:,1]-=local_offset[1]
                pointList[:,0]=np.clip(pointList[:,0],0,size[0])
                pointList[:,1]=np.clip(pointList[:,1],0,size[1])
                Annotations = xml_add_region(Annotations=Annotations, pointList=pointList, annotationID=i,annotationType=annotationTypes[str(i)],regionID=j+1)

    # save xml
    print(outpath)
    xml_save(Annotations=Annotations, filename=outpath)

def get_contour_points(mask, args, downsample,value, offset={'X': 0,'Y': 0}):
    # returns a dict pointList with point 'X' and 'Y' values
    # input greyscale binary image
    maskPoints, contours = cv2.findContours(np.array(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    pointsList = []
    #maskPoints2=copy.deepcopy(maskPoints)

    for j in np.array(range(np.shape(maskPoints)[0])):
        if len(maskPoints[j])>2:

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

def xml_add_annotation(Annotations, annotationID,annotationType,classLink,xml_color): # add new annotation

    if annotationType=='9':
        Annotation = ET.SubElement(Annotations, 'Annotation', attrib={'Type': str(annotationType), 'Visible': '1',
            'ReadOnly': '0', 'Incremental': '0', 'LineColorReadOnly': '1', 'LineColor': str(xml_color[int(classLink)-1]), 'Id': str(annotationID), 'NameReadOnly': '0'})
        inputID=ET.SubElement(Annotation,'InputAnnotationId')
        inputID.text=classLink
    else:
        Annotation = ET.SubElement(Annotations, 'Annotation', attrib={'Type': str(annotationType), 'Visible': '1',
            'ReadOnly': '0', 'Incremental': '0', 'LineColorReadOnly': '0', 'LineColor': str(xml_color[int(classLink)-1]), 'Id': str(annotationID), 'NameReadOnly': '0'})
    Regions = ET.SubElement(Annotation, 'Regions')
    return Annotations

def xml_add_region(Annotations, pointList, annotationID,annotationType, regionID): # add new region to annotation
    # add new Region to Annotation
    # defualts to last annotationID and new regionID
    if annotationType=='4':
        regionType='0'
    elif annotationType=='9':
        regionType='5'
    else:
        print('unsupported annotation type')
        exit()

    Annotation = Annotations.find("Annotation[@Id='" + str(annotationID) + "']")
    Regions = Annotation.find('Regions')

    Region = ET.SubElement(Regions, 'Region', attrib={'NegativeROA': '0', 'ImageFocus': '-1', 'DisplayId': '1', 'InputRegionId': '0', 'Analyze': '0', 'Type': regionType, 'Id': str(regionID)})
    Vertices = ET.SubElement(Region, 'Vertices')
    for point in pointList: # add new Vertex
        # ET.SubElement(Vertices, 'Vertex', attrib={'X': str(point['X']), 'Y': str(point['Y']), 'Z': '0'})

        ET.SubElement(Vertices, 'Vertex', attrib={'X': str(point[0]), 'Y': str(point[1]), 'Z': '0'})
    # add connecting point
    # ET.SubElement(Vertices, 'Vertex', attrib={'X': str(pointList[0]['X']), 'Y': str(pointList[0]['Y']), 'Z': '0'})
    ET.SubElement(Vertices, 'Vertex', attrib={'X': str(pointList[0][0]), 'Y': str(pointList[0][1]), 'Z': '0'})
    return Annotations

def xml_save(Annotations, filename):
    xml_data = ET.tostring(Annotations, pretty_print=True)
    #xml_data = Annotations.toprettyxml()
    f = open(filename, 'wb')
    f.write(xml_data)
    f.close()
