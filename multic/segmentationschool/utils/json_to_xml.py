import os
from .mask_to_xml import xml_create, xml_add_annotation, xml_add_region, xml_save

def get_xml_path(annot, NAMES, tmp, slidename):

    xmlAnnot = xml_create()
    skipSlide = 0
    xml_color=[65280]*(len(NAMES)+1)
    # all compartments
    for class_,compart in enumerate(NAMES):

        compart = compart.replace(' ','')
        class_ +=1
        # add layer to xml
        xmlAnnot = xml_add_annotation(Annotations=xmlAnnot, xml_color=xml_color, annotationID=class_)

        # test all annotation layers in order created
        for iter,a in enumerate(annot):
            try:
                # check for annotation layer by name
                a_name = a['annotation']['name'].replace(' ','')
            except:
                a_name = None

            if a_name == compart:
                # track all layers present
                skipSlide +=1

                pointsList = []

                # load json data
                _ = os.system("printf '\tloading annotation layer: [{}]\n'".format(compart))

                a_data = a['annotation']['elements']

                for data in a_data:
                    pointList = []
                    points = data['points']
                    for point in points:
                        pt_dict = {'X': round(point[0]), 'Y': round(point[1])}
                        pointList.append(pt_dict)
                    pointsList.append(pointList)

                # write annotations to xml
                for i in range(len(pointsList)):
                    pointList = pointsList[i]
                    xmlAnnot = xml_add_region(Annotations=xmlAnnot, pointList=pointList, annotationID=class_)

                # print(a['_version'], a['updated'], a['created'])
                break
            
    if skipSlide != len(NAMES):
        _ = os.system("printf '\tThis slide is missing annotation layers\n'")
        _ = os.system("printf '\tSKIPPING SLIDE...\n'")
        
    _ = os.system("printf '\tFETCHING SLIDE...\n'")
    #os.rename('{}/{}'.format(folder, slidename), '{}/{}'.format(tmp, slidename))

    xml_path = '{}/{}.xml'.format(tmp, os.path.splitext(slidename)[0])
    _ = os.system("printf '\tsaving a created xml annotation file: [{}]\n'".format(xml_path))
    xml_save(Annotations=xmlAnnot, filename=xml_path)

    del xmlAnnot


    return xml_path