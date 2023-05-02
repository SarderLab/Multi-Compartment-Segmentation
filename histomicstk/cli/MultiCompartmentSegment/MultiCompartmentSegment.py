import os
import sys
# from glob import glob
from histomicstk.cli.utils import CLIArgumentParser


def main(args):

    cwd = os.getcwd()

    print(cwd)

    # tmp = args.outputAnnotationFile
    # tmp = os.path.dirname(tmp)
    # #print(tmp)

    #os.makedirs('test_dir')
    os.chdir(cwd)
    # os.makedirs('test_dir_1')
    # try:
    #     os.makedirs(cwd+'/test_dir_1')
    # except:
    #     print('yay')
    #model = glob('{}/*.pth'.format(tmp))[0]

    #print(model)
    #print('\noutput filename: {}\n'.format(args.outputAnnotationFile))
    cmd = "python3 ../segmentationschool/segmentation_school.py --option {} --project {} --base_dir {} --modelfile {}".format(args.option, args.project, args.base_dir, args.modelfile)
    print(cmd)
    sys.stdout.flush()
    os.system(cmd)

if __name__ == "__main__":
    main(CLIArgumentParser().parse_args())