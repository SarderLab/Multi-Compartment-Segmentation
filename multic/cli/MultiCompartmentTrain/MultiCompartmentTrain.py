import os
import sys
from ctk_cli import CLIArgumentParser

def main(args):

    folder = args.base_dir
    base_dir_id = folder.split('/')[-2]
    _ = os.system("printf '\nUsing data from girder_client Folder: {}\n'".format(folder))
    print('new version')
    _ = os.system("printf '\n---\n\nFOUND: [{}]\n'".format(args.input_file))

    cwd = os.getcwd()
    print(cwd)
    os.chdir(cwd)

    cmd = "python3 ../segmentationschool/segmentation_school.py --option {} --training_data_dir {} --val_data_dir {} --init_modelfile {} --girderApiUrl {} --girderToken {} --gpu {} --train_steps {} --eval_period {} --outDir {}".format('train', args.training_data_dir, args.val_data_dir, args.init_modelfile, args.girderApiUrl, args.girderToken, args.gpu, args.training_steps, args.eval_period, args.output_model)
    print(cmd)
    sys.stdout.flush()
    os.system(cmd)

if __name__ == "__main__":
    main(CLIArgumentParser().parse_args())