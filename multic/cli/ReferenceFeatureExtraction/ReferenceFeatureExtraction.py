import os
import sys
from glob import glob
import girder_client
from ctk_cli import CLIArgumentParser

sys.path.append("..")
from segmentationschool.utils.mask_to_xml import xml_create, xml_add_annotation, xml_add_region, xml_save
from segmentationschool.utils.xml_to_mask import write_minmax_to_xml

NAMES = ['cortical_interstitium','medullary_interstitium','non_globally_sclerotic_glomeruli','globally_sclerotic_glomeruli','tubules','arteries/arterioles']


def main(args):

    folder = args.base_dir
    wsi = args.input_file
    base_dir_id = folder.split('/')[-2]
    _ = os.system("printf '\nUsing data from girder_client Folder: {}\n'".format(folder))

    gc = girder_client.GirderClient(apiUrl=args.girderApiUrl)
    gc.setToken(args.girderToken)
    # get files in folder
    files = gc.listItem(base_dir_id)
    xml_color=[65280]*(len(NAMES)+1)
    cwd = os.getcwd()
    print(cwd)
    os.chdir(cwd)

    tmp = folder
    
    slides_used = []

    for file in files:
        slidename = file['name']
        if wsi.split('/')[-1] != slidename:
            continue
        _ = os.system("printf '\n---\n\nFOUND: [{}]\n'".format(slidename))
        skipSlide = 0

        # get annotation
        item = gc.getItem(file['_id'])
        annot = gc.get('/annotation/item/{}'.format(item['_id']), parameters={'sort': 'updated'})
        annot.reverse()
        annot = list(annot)
        _ = os.system("printf '\tfound [{}] annotation layers...\n'".format(len(annot)))

        # create root for xml file
        xmlAnnot = xml_create()

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
            del xmlAnnot
            continue # correct layers not present
        _ = os.system("printf '\tFETCHING SLIDE...\n'")
        os.rename('{}/{}'.format(folder, slidename), '{}/{}'.format(tmp, slidename))
        slides_used.append(slidename)

        xml_path = '{}/{}.xml'.format(tmp, os.path.splitext(slidename)[0])
        _ = os.system("printf '\tsaving a created xml annotation file: [{}]\n'".format(xml_path))
        xml_save(Annotations=xmlAnnot, filename=xml_path)
        write_minmax_to_xml(xml_path) # to avoid trying to write to the xml from multiple workers
        del xmlAnnot
    os.system("ls -lh '{}'".format(tmp))
    
    _ = os.system("printf '\ndone retriving data...\n\n'")



    cmd = "python3 ../segmentationschool/segmentation_school.py --option {} --base_dir {} --files {} --xml_path {} --girderApiUrl {} --girderToken {}".format('get_features', args.base_dir, args.input_file, xml_path, args.girderApiUrl, args.girderToken)
    print(cmd)
    sys.stdout.flush()
    os.system(cmd)

if __name__ == "__main__":
    main(CLIArgumentParser().parse_args())
