import os
import sys
import argparse

def main(args):

    folder = args.base_dir
    base_dir_id = folder.split('/')[-2]
    _ = os.system("printf '\nUsing data from girder_client Folder: {}\n'".format(folder))
    print('new version')
    _ = os.system("printf '\n---\n\nFOUND: [{}]\n'".format(args.input_file))

    cwd = os.getcwd()
    print(cwd)
    os.chdir(cwd)

    cmd = "python3 ../segmentationschool/segmentation_school.py --option {} --base_dir {} --modelfile {} --girderApiUrl {} --girderToken {} --files {}".format('predict', args.base_dir, args.modelfile, args.girderApiUrl, args.girderToken, args.input_file)
    print(cmd)
    sys.stdout.flush()
    os.system(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir')
    parser.add_argument('--input_file')
    parser.add_argument('--modelfile')
    parser.add_argument('--girderApiUrl')
    parser.add_argument('--girderToken')
    main(parser.parse_args())
