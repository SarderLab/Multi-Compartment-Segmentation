import os
import sys
from glob import glob
import girder_client
from ctk_cli import CLIArgumentParser

sys.path.append("..")
from segmentationschool.utils.json_to_xml import get_xml_path

NAMES = ['cortical_interstitium','medullary_interstitium','non_globally_sclerotic_glomeruli','globally_sclerotic_glomeruli','tubules','arteries/arterioles']


def main(args):

    folder = args.base_dir
    wsi = args.input_file
    file_name = wsi.split('/')[-1]
    base_dir_id = folder.split('/')[-2]
    _ = os.system("printf '\nUsing data from girder_client Folder: {}\n'".format(folder))

    gc = girder_client.GirderClient(apiUrl=args.girderApiUrl)
    gc.setToken(args.girderToken)
    # get files in folder
    files = list(gc.listItem(base_dir_id))
    # dict to link filename to gc id
    item_dict = dict()
    for file in files:
        d = {file['name']:file['_id']}
        item_dict.update(d)
    
    file_id = item_dict[file_name]

    cwd = os.getcwd()
    print(cwd)
    os.chdir(cwd)
    tmp = folder
    _ = os.system("printf '\n---\n\nFOUND: [{}]\n'".format(file_name))
    # get annotation
    annotations= gc.get('/annotation/item/{}'.format(file_id), parameters={'sort': 'updated'})
    annotations.reverse()
    annotations = list(annotations)
    
    annotations_filtered = [annot for annot in annotations if annot['annotation']['name'].strip() in NAMES]
    _ = os.system("printf '\tfound [{}] annotation layers...\n'".format(len(annotations_filtered)))
    del annotations
    # create root for xml file
        
    xml_path = get_xml_path(annotations_filtered, NAMES, tmp, file_name)    
    _ = os.system("printf '\ndone retriving data...\n\n'")

    cmd = "python3 ../segmentationschool/segmentation_school.py --option {} --base_dir {} --file {} --xml_path {} --platform {} --item_id {} --girderApiUrl {} --girderToken {}".format('get_features', args.base_dir, args.input_file, xml_path, 'DSA',file_id, args.girderApiUrl, args.girderToken)
    print(cmd)
    sys.stdout.flush()
    os.system(cmd)

if __name__ == "__main__":
    main(CLIArgumentParser().parse_args())
