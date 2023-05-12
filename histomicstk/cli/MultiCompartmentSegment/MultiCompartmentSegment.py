import os
import sys
import girder_client
import torch
import tiffslide as openslide
from histomicstk.cli.utils import CLIArgumentParser


def main(args):

    

    folder = args.base_dir
    base_dir_id = folder.split('/')[-2]
    _ = os.system("printf '\nUsing data from girder_client Folder: {}\n'".format(folder))


    # newfile = gc.createFolder(girder_folder_id,'tes_file_with_girder')

    

    

    _ = os.system("printf '\n---\n\nFOUND: [{}]\n'".format(args.input_file))
    print(os.path.getsize(args.input_file))
    slide=openslide.TiffSlide(args.input_file)

    print('yess')


    WSIs = [slide]


    cwd = os.getcwd()
    print(cwd)
    os.chdir(cwd)


    
    modelfile = torch.load(args.modelfile)
    torch.save(modelfile)
    os.system('pwd')
    os.system('ls -lh')
    # tmp = args.outputAnnotationFile
    # tmp = os.path.dirname(tmp)
    # #print(tmp)

    
    
    #os.makedirs('test_dir')
    
    # os.makedirs('test_dir_1')
    # try:
    #     os.makedirs(cwd+'/test_dir_1')
    # except:
    #     print('yay')
    #model = glob('{}/*.pth'.format(tmp))[0]
    
    #print(model)
    #print('\noutput filename: {}\n'.format(args.outputAnnotationFile))
    cmd = "python3 ../segmentationschool/segmentation_school.py --option {} --base_dir {} --modelfile {} --girderApiUrl {} --girderToken {} --files {}".format(args.option, args.base_dir, modelfile, args.girderApiUrl, args.girderToken,WSIs)
    print(cmd)
    sys.stdout.flush()
    os.system(cmd)

if __name__ == "__main__":
    main(CLIArgumentParser().parse_args())