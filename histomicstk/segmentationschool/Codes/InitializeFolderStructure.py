import os
import numpy as np
from glob import glob
from shutil import rmtree,copy#move,copyfile

def purge_training_set(args):
    rmtree(args.project + '/Permanent/')
    rmtree(args.project + '/TempHR')
    rmtree(args.project + '/TempLR')
    initFolder(args=args)

def prune_training_set(args):
    # prune HR dataset
    regions_path =  args.project + '/Permanent/HR/regions/'
    masks_path = args.project + '/Permanent/HR/masks/'
    prune_percent = args.prune_HR
    prune_data(regions_path, masks_path, prune_percent, args)

    # prune LR dataset
    regions_path = args.project + '/Permanent/LR/regions/'
    masks_path = args.project + '/Permanent/LR/masks/'
    prune_percent = args.prune_LR
    prune_data(regions_path, masks_path, prune_percent, args)

def prune_data(regions_path, masks_path, prune_percent, args):
    imgs = glob(regions_path + '*' + args.imBoxExt)
    if imgs == None:
        return

    keep = (np.random.rand(len(imgs))) >= prune_percent
    for idx,img in enumerate(imgs):
        if keep[idx] == False:
            filename = os.path.basename(img)
            mask = masks_path + '/' + os.path.splitext(filename)[0] + '.png'
            os.remove(img) # remove region
            os.remove(mask) # remove mask
            print(img)


def initFolder(args):
    dirs = {'imExt': '.jpeg'}
    dirs['basedir'] = args.base_dir
    dirs['maskExt'] = '.png'
    dirs['modeldir'] = '/MODELS/'
    dirs['tempdirLR'] = '/TempLR'
    dirs['tempdirHR'] = '/TempHR'
    dirs['pretraindir'] = '/Deeplab_network/'
    dirs['training_data_dir'] = '/TRAINING_data/'
    dirs['validation_data_dir'] = '/HOLDOUT_data/'
    dirs['model_init'] = 'deeplab_resnet.ckpt'
    dirs['project']= args.project
    dirs['modelfile'] = args.modelfile
    dirs['data_dir_HR'] = args.project + '/Permanent/HR/'
    dirs['data_dir_LR'] = args.project + '/Permanent/LR/'
    initializeFolderStructure(dirs,args)
    print('Please add xmls/svs files to the newest TRAINING_data folder.')


def initializeFolderStructure(dirs,args):
    #os.makedirs(dirs['modelfile']+'/yummuy')
    tmp = os.path.dirname(dirs['project'])
    print(tmp)
    os.chdir(tmp)
    real_path = os.path.realpath(dirs['project'])
    print(real_path)
    os.mkdir('MODELS')
    print('b')
    os.makedirs(os.path.join(dirs['project'],'test2'))
    print('c')
    print(os.listdir())
    #os.makedirs(dirs['modeldir'])
    print('d')
    #make_folder(dirs['project'] + dirs['modeldir'] + str(0) + '/LR/')
    #make_folder(dirs['project']+ dirs['modeldir'] + str(0) + '/HR/')

    if args.transfer==' ':
        pass
    else:

        modelsCurrent=os.listdir(dirs['basedir'] + '/' + args.transfer + dirs['modeldir'])
        gens=map(int,modelsCurrent)
        modelOrder=np.argsort(gens)
        modelLast=np.max(gens)
        pretrainsLR=glob(dirs['basedir']+ '/' + args.transfer + dirs['modeldir'] + str(modelLast) + '/LR/' + 'model*')
        pretrainsHR=glob(dirs['basedir']+ '/' + args.transfer + dirs['modeldir'] + str(modelLast) + '/HR/' + 'model*')

        maxmodel=0
        for modelfiles in pretrainsLR:
            modelID=modelfiles.split('.')[-2].split('-')[1]
            if int(modelID)>maxmodel:
                maxmodelLR=int(modelID)
        for modelfiles in pretrainsHR:
            modelID=modelfiles.split('.')[-2].split('-')[1]
            if int(modelID)>maxmodel:
                maxmodelHR=int(modelID)

        pretrain_filesLR=glob(dirs['basedir']+ '/' + args.transfer + dirs['modeldir'] + str(modelLast) + '/LR/' + 'model.ckpt-' + str(maxmodelLR) + '*')
        pretrain_filesHR=glob(dirs['basedir']+ '/' + args.transfer + dirs['modeldir'] + str(modelLast) + '/HR/' + 'model.ckpt-' + str(maxmodelHR) + '*')
        for file in pretrain_filesLR:
            copy(file,dirs['project']+ dirs['modeldir'] + str(0) + '/LR/')

        for file in pretrain_filesHR:
            copy(file,dirs['project']+ dirs['modeldir'] + str(0) + '/HR/')


    make_folder(dirs['training_data_dir'] + str(0))

    make_folder(dirs['tempdirLR'] + '/regions')
    make_folder(dirs['project']+ dirs['tempdirLR'] + '/masks')

    make_folder(dirs['tempdirLR'] + '/Augment' +'/regions')
    make_folder(dirs['tempdirLR'] + '/Augment' +'/masks')

    make_folder(dirs['tempdirHR'] + '/regions')
    make_folder( dirs['tempdirHR'] + '/masks')

    make_folder(dirs['tempdirHR'] + '/Augment' +'/regions')
    make_folder(dirs['tempdirHR'] + '/Augment' +'/masks')

    make_folder(dirs['modeldir'])
    make_folder(dirs['training_data_dir'])
    make_folder(dirs['validation_data_dir'])


    make_folder('/Permanent' +'/LR/'+ 'regions/')
    make_folder('/Permanent' +'/LR/'+ 'masks/')
    make_folder('/Permanent' +'/HR/'+ 'regions/')
    make_folder('/Permanent' +'/HR/'+ 'masks/')

    make_folder(dirs['training_data_dir'])

    make_folder(dirs['basedir'] + '/Codes/Deeplab_network/datasetLR')
    make_folder(dirs['basedir'] + '/Codes/Deeplab_network/datasetHR')


def make_folder(directory):
    #if not os.path.exists(directory):
    print(directory,'directory')
    try:
        print('helllllloooooo')
        os.makedirs(directory) # make directory if it does not exit already # make new directory # Check if folder exists, if not make it
    except:
        print('Folder already exists!')
