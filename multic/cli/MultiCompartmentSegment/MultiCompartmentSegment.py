import os
import sys
from ctk_cli import CLIArgumentParser
import girder_client
sys.path.append('..')
from segmentationschool.segmentation_school import run_it


DEFAULT_VALS = {
        'girderApiUrl':' ',
        'girderToken':' ',
        'input_file':' ',
        'option':'predict',
        'modelfile':None,
        'white_percent':0.01,
        'chop_thumbnail_resolution':16,
        'overlap_percentHR':0,
        'boxSize':2048,
        'downsampleRateHR':1,
        'Mag20X':False,
        'roi_thresh':0.01,
        'min_size':[30,30,30,30,30,30],
        'bordercrop':300,
        'show_interstitium':True
    }


def main(args):

    gc = girder_client.GirderClient(apiUrl=args.girderApiUrl)
    gc.setToken(args.girderToken)

    # Finding the id for the current WSI (input_image)
    file_id = args.input_file
    file_info = gc.get(f'/file/{file_id}')
    item_id = file_info['itemId']

    item_info = gc.get(f'/item/{item_id}')

    file_name = file_info['name']
    print(f'Running on: {file_name}')

    mounted_path = '{}/{}'.format('/mnt/girder_worker', os.listdir('/mnt/girder_worker')[0])
    file_path = '{}/{}'.format(mounted_path,file_name)
    gc.downloadFile(file_id, file_path)

    print(f'This is slide path: {file_path}')

    print('new version')
    _ = os.system("echo '\n---\n\nFOUND: [{}]\n'".format(args.input_file))

    cwd = os.getcwd()
    print(cwd)
    os.chdir(cwd)

    for d in DEFAULT_VALS:
        if d not in list(vars(args).keys()):
            setattr(args,d,DEFAULT_VALS[d])

    setattr(args,'item_id', item_id)
    setattr(args,'file', file_path)
    setattr(args,'gc', gc)

    print(vars(args))
    for d in vars(args):
        print(f'argument: {d}, value: {getattr(args,d)}')

    run_it(args)

if __name__ == "__main__":
    main(CLIArgumentParser().parse_args())
