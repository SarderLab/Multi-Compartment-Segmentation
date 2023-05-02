import numpy as np
import sys, warnings
import lxml.etree as ET
import cv2
# import matplotlib.pyplot as plt
from skimage.morphology import binary_erosion #binary_dilation,
from skimage.morphology import disk
from skimage.io import imsave
# import time
from matplotlib import path
from .getWsi import getWsi
# from tqdm import tqdm

def get_num_classes(xml_path):
    # parse xml and get root
    tree = ET.parse(xml_path)
    root = tree.getroot()
    annotation_num = 0
    for Annotation in root.findall("./Annotation"): # for all annotations
        annotation_num = annotation_num + 1

    return annotation_num + 1

"""
location (tuple) - (x, y) tuple giving the top left pixel in the level 0 reference frame
size (tuple) - (width, height) tuple giving the region size
"""

def get_supervision_boxes(root,boxlayerIDs):

    boxes=[]
    for Annotation in root.findall("./Annotation"): # for all annotations
        annotationID = Annotation.attrib['Id']
        if annotationID in boxlayerIDs:
            for Region in Annotation.findall("./*/Region"): # iterate on all region
                box_bounds=[]
                for Vertex in Region.findall("./*/Vertex"): # iterate on all vertex in region
                    # get points
                    x_point = np.int32(np.float64(Vertex.attrib['X']))
                    y_point = np.int32(np.float64(Vertex.attrib['Y']))
                    box_bounds.append([x_point,y_point])
                boxes.append({'BoxVerts':box_bounds,'annotationID':annotationID})
    return boxes


def regions_in_mask_dots(root, bounds,box_layers):
    # find regions to save
    IDs_reg = []
    IDs_points = []

    for Annotation in root.findall("./Annotation"): # for all annotations
        annotationID = Annotation.attrib['Id']
        if annotationID in box_layers:
            continue
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

def get_vertex_points_dots(root, IDs_reg,IDs_points, maskModes,excludedIDs,negativeIDs=None,falsepositiveIDs=None):
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
    if 'falsepositive' in maskModes:
        assert falsepositiveIDs is not None,'False positive annotated classes must be provided for falsepositive mask mode'

    if 'negative' in maskModes:
        assert negativeIDs is not None,'Negatively annotated classes must be provided for negative mask mode'
    assert 'falsepositive' and 'negative' not in maskModes, 'Negative AND false positive mask modes is not yet supported'

    useableRegions=[]
    if 'positive' in maskModes:

        for Region in Regions:

            regionPath=path.Path(Region['Vertices'])

            for Point in Points:
                if 'negative' in maskModes:
                    if Region['annotationID'] not in negativeIDs:
                        if regionPath.contains_point(Point['Vertices'][0]):
                            Region['pointAnnotationID']=Point['pointAnnotationID']
                            useableRegions.append(Region)
                else:
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
            if Region['annotationID'] in falsepositiveIDs:
                if not any([regionPath.contains_point(Point['Vertices'][0]) for Point in Points]):
                    Region['pointAnnotationID']=0
                    useableRegions.append(Region)

    return useableRegions

def masks_from_points(usableRegions,wsiID,dirs,dot_pad,args,dims):
    pas_img = getWsi(wsiID)
    dim_x, dim_y=pas_img.dimensions
    image_sizes=[]
    basename=wsiID.split('/')[-1].split('.svs')[0]
    max_mask_size=args.training_max_size
    stepHR = int(max_mask_size*(1-args.overlap_percentHR)) #Step size before downsampling

    region=np.array(pas_img.read_region((dims[0],dims[2]),0,(dims[1]-dims[0],dims[3]-dims[2])))[:,:,:3]
    mask = 2*np.ones([dims[3]-dims[2],dims[1]-dims[0]], dtype=np.uint8)
    for usableRegion in usableRegions:
        vertices=usableRegion['Vertices']
        # x1=min(vertices[:,0])
        # x2=max(vertices[:,0])
        # y1=min(vertices[:,1])
        # y2=max(vertices[:,1])

        points = np.stack([np.asarray(vertices[:,0]), np.asarray(vertices[:,1])], axis=1)

        points[:,1] = np.int32(np.round(points[:,1] - dims[2] ))
        points[:,0] = np.int32(np.round(points[:,0] - dims[0] ))

        if int(usableRegion['pointAnnotationID'])==0:
            pass
        else:
            cv2.fillPoly(mask, [points], int(usableRegion['pointAnnotationID'])-4)
    # plt.subplot(121)
    # plt.imshow(region)
    # plt.subplot(122)
    # plt.imshow(mask)
    # plt.show()
    l2=dims[3]-dims[2]
    l1=dims[1]-dims[0]
    if l1<max_mask_size or l2<max_mask_size:
        print('small image size')
        print(dims)
        exit()
    else:

        subIndex_yHR=np.array(range(0,l2,stepHR))
        subIndex_xHR=np.array(range(0,l1,stepHR))
        subIndex_yHR[-1]=l2-max_mask_size
        subIndex_xHR[-1]=l1-max_mask_size
        for i in subIndex_xHR:
            for j in subIndex_yHR:
                subRegion=region[j:j+max_mask_size,i:i+max_mask_size,:]
                subMask=mask[j:j+max_mask_size,i:i+max_mask_size]
                image_identifier=basename+'_'.join(['',str(dims[0]),str(dims[2]),str(l1),str(l2),str(i),str(j)])
                mask_out_name=dirs['basedir']+dirs['project'] + '/Permanent/HR/masks/'+image_identifier+'.png'
                image_out_name=mask_out_name.replace('/masks/','/regions/')
                # image_sizes.append([max_mask_size,max_mask_size])
                # plt.subplot(121)
                # plt.imshow(subRegion)
                # plt.subplot(122)
                # plt.imshow(subMask)
                # plt.show()
                # continue
                # basename + '_' + str(image_identifier) + args.imBoxExt
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    imsave(image_out_name,subRegion)
                    imsave(mask_out_name,subMask)
    '''
        l1=x2-x1
        l2=y2-y1
        bounds = {'x_min' : x1, 'y_min' : y1, 'x_max' : x2, 'y_max' : y2}
        subIDs=[]
        for subRegionCheck in usableRegions:
            subRegionVertCheck=subRegionCheck['Vertices']
            for vert in subRegionVertCheck:
                if bounds['x_min'] <= vert[0] <= bounds['x_max'] and bounds['y_min'] <= vert[1] <= bounds['y_max']: # test points in region bounds
                    # save region Id
                    subIDs.append(subRegionCheck)
                    break

            # if bounds['x_min'] <= x_point <= bounds['x_max'] and bounds['y_min'] <= y_point <= bounds['y_max']: # test points in region bounds
        ##
        ##
        ## we are here, we need to repair this to call xml_to_mask
        if (x2-x1)>0 and (y2-y1)>0:
            mask = 2*np.ones([y2-y1,x2-x1], dtype=np.uint8)
            for subRegion in subIDs:
                subvertices=subRegion['Vertices']
                points = np.stack([np.asarray(subvertices[:,0]), np.asarray(subvertices[:,1])], axis=1)



                # xMultiplier=np.ceil((l1)/64)
                # yMultiplier=np.ceil((l2)/64)
                # pad1=int(xMultiplier*64-l1)
                # pad2=int(yMultiplier*64-l2)

                points[:,1] = np.int32(np.round(points[:,1] - y1 ))
                points[:,0] = np.int32(np.round(points[:,0] - x1 ))

                if int(subRegion['pointAnnotationID'])==0:
                    pass
                else:
                    cv2.fillPoly(mask, [points], int(subRegion['pointAnnotationID'])-4)

            PAS = pas_img.read_region((x1,y1), 0, (x2-x1,y2-y1))
            PAS = np.array(PAS)[:,:,0:3]
            # plt.subplot(121)
            # plt.imshow(PAS)
            # plt.subplot(122)
            # plt.imshow(mask)
            # plt.show()
            # continue

            if l1>max_mask_size and l2>max_mask_size:

                subIndex_yHR=np.array(range(0,l2,max_mask_size))
                subIndex_xHR=np.array(range(0,l1,max_mask_size))
                subIndex_yHR[-1]=l2-max_mask_size
                subIndex_xHR[-1]=l1-max_mask_size
                for i in subIndex_xHR:
                    for j in subIndex_yHR:
                        subRegion=PAS[j:j+max_mask_size,i:i+max_mask_size,:]
                        subMask=mask[j:j+max_mask_size,i:i+max_mask_size]
                        image_identifier=basename+'_'.join(['',str(x1),str(y1),str(l1),str(l2),str(i),str(j)])
                        mask_out_name=dirs['basedir']+dirs['project'] + '/Permanent/HR/masks/'+image_identifier+'.png'
                        image_out_name=mask_out_name.replace('/masks/','/regions/')
                        image_sizes.append([max_mask_size,max_mask_size])
                        plt.subplot(121)
                        plt.imshow(subRegion)
                        plt.subplot(122)
                        plt.imshow(subMask)
                        plt.show()
                        continue
                        # basename + '_' + str(image_identifier) + args.imBoxExt
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            imsave(image_out_name,subRegion)
                            imsave(mask_out_name,subMask)

            elif l1>max_mask_size:
                plt.subplot(121)
                plt.imshow(subRegion)
                plt.subplot(122)
                plt.imshow(subMask)
                plt.show()
                print('small image')
                break
                subIndex_xHR=np.array(range(0,l1,max_mask_size))
                subIndex_xHR[-1]=l1-max_mask_size
                for i in subIndex_xHR:
                    subRegion=PAS[:,i:i+max_mask_size,:]
                    subMask=mask[:,i:i+max_mask_size]
                    image_identifier=basename+'_'.join(['',str(x1),str(y1),str(l1),str(l2),str(l2),str(i)])
                    image_sizes.append([max_mask_size,l2])
                    mask_out_name=dirs['basedir']+dirs['project'] + '/Permanent/HR/masks/'+image_identifier+'.png'
                    image_out_name=mask_out_name.replace('/masks/','/regions/')
                    # basename + '_' + str(image_identifier) + args.imBoxExt
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        imsave(image_out_name,subRegion)
                        imsave(mask_out_name,subMask)

            elif l2>max_mask_size:
                print('small image')
                break
                subIndex_yHR=np.array(range(0,l2,max_mask_size))
                subIndex_yHR[-1]=l2-max_mask_size
                for j in subIndex_yHR:
                    subRegion=PAS[j:j+max_mask_size,:,:]
                    subMask=mask[j:j+max_mask_size,:]
                    image_identifier=basename+'_'.join(['',str(x1),str(y1),str(l1),str(l2),str(j),str(l1)])
                    image_sizes.append([max_mask_size,l1])
                    mask_out_name=dirs['basedir']+dirs['project'] + '/Permanent/HR/masks/'+image_identifier+'.png'
                    image_out_name=mask_out_name.replace('/masks/','/regions/')
                    # basename + '_' + str(image_identifier) + args.imBoxExt
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        imsave(image_out_name,subRegion)
                        imsave(mask_out_name,subMask)

            else:
                print('small image')
                break
                # pass
                image_identifier=basename+'_'.join(['',str(x1),str(y1),str(l1),str(l2)])
                mask_out_name=dirs['basedir']+dirs['project'] + '/Permanent/HR/masks/'+image_identifier+'.png'
                image_out_name=mask_out_name.replace('/masks/','/regions/')
                image_sizes.append([l1,l2])
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
            image_sizes.append([l1,l2])

        else:
            print('Broken region')
        '''
    # return image_sizes


#------------------------------------------------------------------------------------------------------------------------------------------------------


def xml_to_mask(xml_path, location, size, downsample_factor=1, verbose=0):
    # parse xml and get root
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # calculate region bounds
    bounds = {'x_min' : location[0], 'y_min' : location[1], 'x_max' : location[0] + size[0], 'y_max' : location[1] + size[1]}

    IDs = regions_in_mask(root=root, bounds=bounds, verbose=verbose)

    if verbose != 0:
        print('\nFOUND: ' + str(len(IDs)) + ' regions')

    # find regions in bounds
    Regions = get_vertex_points(root=root, IDs=IDs, verbose=verbose)

    # fill regions and create mask
    mask = Regions_to_mask(Regions=Regions, bounds=bounds, IDs=IDs, downsample_factor=downsample_factor, verbose=verbose)
    if verbose != 0:
        print('done...\n')

    return mask

def restart_line(): # for printing labels in command line
    sys.stdout.write('\r')
    sys.stdout.flush()

def regions_in_mask(root, bounds, verbose=1):
    # find regions to save
    IDs = []

    for Annotation in root.findall("./Annotation"): # for all annotations
        annotationID = Annotation.attrib['Id']

        for Region in Annotation.findall("./*/Region"): # iterate on all region

            if verbose != 0:
                sys.stdout.write('TESTING: ' + 'Annotation: ' + annotationID + '\tRegion: ' + Region.attrib['Id'])
                sys.stdout.flush()
                restart_line()

            for Vertex in Region.findall("./*/Vertex"): # iterate on all vertex in region
                # get points
                x_point = np.int32(np.float64(Vertex.attrib['X']))
                y_point = np.int32(np.float64(Vertex.attrib['Y']))
                # test if points are in bounds
                if bounds['x_min'] <= x_point <= bounds['x_max'] and bounds['y_min'] <= y_point <= bounds['y_max']: # test points in region bounds
                    # save region Id
                    IDs.append({'regionID' : Region.attrib['Id'], 'annotationID' : annotationID})
                    break
    return IDs

def get_vertex_points(root, IDs, verbose=1):
    Regions = []

    for ID in IDs: # for all IDs
        if verbose != 0:
            sys.stdout.write('PARSING: ' + 'Annotation: ' + ID['annotationID'] + '\tRegion: ' + ID['regionID'])
            sys.stdout.flush()
            restart_line()

        # get all vertex attributes (points)
        Vertices = []

        for Vertex in root.findall("./Annotation[@Id='" + ID['annotationID'] + "']/Regions/Region[@Id='" + ID['regionID'] + "']/Vertices/Vertex"):
            # make array of points
            Vertices.append([int(float(Vertex.attrib['X'])), int(float(Vertex.attrib['Y']))])


        Regions.append(np.array(Vertices))

    return Regions

def Regions_to_mask(Regions, bounds, IDs, downsample_factor, verbose=1):
    downsample = int(np.round(downsample_factor**(.5)))
    strel=disk(3)
    if verbose !=0:
        print('\nMAKING MASK:')

    if len(Regions) != 0: # regions present
        # get min/max sizes
        min_sizes = np.empty(shape=[2,0], dtype=np.int32)
        max_sizes = np.empty(shape=[2,0], dtype=np.int32)
        for Region in Regions: # fill all regions
            min_bounds = np.reshape((np.amin(Region, axis=0)), (2,1))
            max_bounds = np.reshape((np.amax(Region, axis=0)), (2,1))
            min_sizes = np.append(min_sizes, min_bounds, axis=1)
            max_sizes = np.append(max_sizes, max_bounds, axis=1)
        min_size = np.amin(min_sizes, axis=1)
        max_size = np.amax(max_sizes, axis=1)



        # add to old bounds
        bounds['x_min_pad'] = min(min_size[1], bounds['x_min'])
        bounds['y_min_pad'] = min(min_size[0], bounds['y_min'])
        bounds['x_max_pad'] = max(max_size[1], bounds['x_max'])
        bounds['y_max_pad'] = max(max_size[0], bounds['y_max'])

        # make blank mask
        mask = np.zeros([ int(np.round((bounds['y_max_pad'] - bounds['y_min_pad']) / downsample)), int(np.round((bounds['x_max_pad'] - bounds['x_min_pad']) / downsample)) ], dtype=np.int8)
        mask_temp = np.zeros([ int(np.round((bounds['y_max_pad'] - bounds['y_min_pad']) / downsample)), int(np.round((bounds['x_max_pad'] - bounds['x_min_pad']) / downsample)) ], dtype=np.int8)

        # fill mask polygons
        index = 0
        for idx,Region in enumerate(Regions):

            # reformat Regions
            Region2=Region
            Region[:,1] = np.int32(np.round((Region[:,1] - bounds['y_min_pad']) / downsample))
            Region[:,0] = np.int32(np.round((Region[:,0] - bounds['x_min_pad']) / downsample))

            x_start = np.int32((np.round((bounds['x_min'] - bounds['x_min_pad'])) / downsample))
            y_start = np.int32((np.round((bounds['y_min'] - bounds['y_min_pad'])) / downsample))
            x_stop = np.int32((np.round((bounds['x_max'] - bounds['x_min_pad'])) / downsample))
            y_stop = np.int32((np.round((bounds['y_max'] - bounds['y_min_pad'])) / downsample))

            # get annotation ID for mask color
            ID = IDs[index]
            '''
            if int(ID['annotationID'])==4:
                xl=x_stop-x_start
                yl=y_stop-y_start
                Region2[:,0]=Region2[:,0]-x_start
                Region2[:,1]=Region2[:,1]-y_start
                for vert in Region2:
                    if vert[0]<0:
                        vert[0]=0
                    if vert[1]<0:
                        vert[1]=0
                    if vert[0]>xl:
                        vert[0]=xl
                    if vert[1]>yl:
                        vert[1]=yl



                mask_temp = np.zeros([int((xl) / downsample),int((yl) / downsample)], dtype=np.int8)

                cv2.fillPoly(mask_temp, [Region2], int(ID['annotationID']))


                s=disk(2)
                e=binary_erosion(mask_temp,s).astype('uint8')
                d=binary_dilation(mask_temp,s).astype('uint8')
                tub_divider=np.where((d-e)==1)

                mask_temp=mask_temp.astype('uint8')
                mask_temp[tub_divider]=5

                temp_pull=mask[ y_start:y_stop, x_start:x_stop ]
                temp_pull[np.where(mask_temp==4)]=4
                temp_pull[np.where(mask_temp==5)]=1
                mask[ y_start:y_stop, x_start:x_stop ]=temp_pull
            else:
            '''


            if int(ID['annotationID'])==4:
                #print(np.float(idx)/np.float(len(Regions)))

                #t=time.time()
                cv2.fillPoly(mask_temp, [Region], int(ID['annotationID']))
                #print(time.time()-t)
                x1=np.min(Region[:,1])
                x2=np.max(Region[:,1])
                y1=np.min(Region[:,0])
                y2=np.max(Region[:,0])
                #t=time.time()
                sub_mask=mask_temp[x1:x2,y1:y2]
                #print(time.time()-t)


                #t=time.time()
                e=binary_erosion(sub_mask,strel).astype('uint8')
                #print(time.time()-t)

                #t=time.time()
                #d=binary_dilation(sub_mask,strel).astype('uint8')
                #print(time.time()-t)

                #t=time.time()
                #tub_divider=np.where((d-e)==1)
                #print(time.time()-t)

                #t=time.time()
                #sub_mask[tub_divider]=1

                #print(time.time()-t)

                #t=time.time()
                tub_prev=mask[x1:x2,y1:y2]
                tub_prev[e==1]=int(ID['annotationID'])
                #sub_mask[tub_divider]=1

                #overlap=tub_prev&sub_mask
                #sub_mask[overlap]=1
                mask[x1:x2,y1:y2]=tub_prev

                #print(time.time()-t)
            else:
                cv2.fillPoly(mask, [Region], int(ID['annotationID']))
            index = index + 1

        # reshape mask

        # pull center mask region
        mask = mask[ y_start:y_stop, x_start:x_stop ]
        #plt.imshow(mask*50)
        #plt.show()
        '''
        msub=np.zeros((y_stop-y_start,x_stop-x_start))
        msub2=msub
        msub[np.where(mask==4)]=1
        s=disk(3)
        msub=binary_dilation(msub,selem=s)-msub2
        mask[np.where(msub==1)]=5
        '''
    else: # no Regions
        mask = np.zeros([ int(np.round((bounds['y_max'] - bounds['y_min']) / downsample)), int(np.round((bounds['x_max'] - bounds['x_min']) / downsample)) ])

    return mask
