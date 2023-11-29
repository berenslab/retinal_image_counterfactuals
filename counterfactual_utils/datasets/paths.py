import socket
print()
import os

def get_base_dir_ARAM():
    machine_name = socket.gethostname()

    if 'slurm' in machine_name:
        base = '/mnt/qb/hein/valentyn/RATIO/'
    else:
        base = '/gpfs01/berens/user/iilanchezian/Projects/DVCEs_private/'
    return base

def get_EyePacs_path():
    return '/gpfs01/berens/data/data/eyepacs/data_processed/images/'

def get_base_data_dir():
    path = '/scratch/datasets/'
    return path

def get_svhn_path():
    return os.path.join(get_base_data_dir(),  'SVHN')

def get_fundusKaggle_path(data_folder, background_subtraction, clahe=False, preprocess='none', version=None,
                          corruption_type=None):

    if background_subtraction:
        #version = '_' + version + '_gauss_correct' if version is not None else ''
        version = '_gauss' + '_' + version if version is not None else ''
        print('using background subtraction, version: '+str(version))
        #return f'{data_folder}/kaggle_data_bg_sub{version}/'
        return f'{data_folder}/kaggle_data_bg_sub{version}/'
    else:
        #version = '_' + version if version is not None else ''
        version = '_' + version + '_resized_new_qual_eval' if version is not None else ''
        #version = '_' + version + '_resized' if version is not None else ''
        #version = '_' + version if version is not None else ''
        if preprocess == 'hist_eq' and clahe:
            version += '_clahe'
        elif preprocess == 'add_artifacts':
            version += '_artifacts_green_circles_blue_squares'
        #version = version + '_correct' if version is not None else ''

        if corruption_type is None:
            print('using raw images, version' + str(version))
            return f'{data_folder}/kaggle_data{version}/'
        else:
            print('using corrupted images')
            return f'{data_folder}/eye_q_degradation/de_image/'

def get_CIFAR10_path():
    return os.path.join(get_base_data_dir(),  'CIFAR10')

def get_CIFAR100_path():
    return os.path.join(get_base_data_dir(),  'CIFAR100')

def get_CIFAR10_C_path():
    return os.path.join(get_base_data_dir(),  'CIFAR-10-C')

def get_CIFAR100_C_path():
    return os.path.join(get_base_data_dir(),  'CIFAR-100-C')

def get_CINIC10_path():
    return os.path.join(get_base_data_dir(),  'cinic_10')


def get_celebA_path():
    return get_base_data_dir()

def get_stanford_cars_path():
    return os.path.join(get_base_data_dir(),  'stanford_cars')

def get_flowers_path():
    return os.path.join(get_base_data_dir(),  'flowers')

def get_pets_path():
    return os.path.join(get_base_data_dir(),  'pets')

def get_food_101N_path():
    return os.path.join(get_base_data_dir(),   'Food-101N', 'Food-101N_release')

def get_food_101_path():
    return os.path.join(get_base_data_dir(),   'Food-101')

def get_fgvc_aircraft_path():
    return os.path.join(get_base_data_dir(),  'FGVC/fgvc-aircraft-2013b')

def get_cub_path():
    return os.path.join(get_base_data_dir(),  'CUB')

def get_LSUN_scenes_path():
    return os.path.join(get_base_data_dir(),  'LSUN_scenes')


def get_tiny_images_files(shuffled=True):
    if shuffled == True:
        raise NotImplementedError()
    else:
        return get_base_data_dir() + '80M Tiny Images/tiny_images.bin'

def get_tiny_images_lmdb():
    raise NotImplementedError()

def get_imagenet_path(data_folder=None):
    path = os.path.join(get_base_data_dir(), 'imagenet')
    return path

def get_imagenet_o_path():
    return os.path.join(get_base_data_dir(), 'imagenet-o')

def get_openimages_path():
    path = '/home/scratch/datasets/openimages/'
    return path

def get_oct_path(data_folder):
    return f'{data_folder}/OCT_preprocessed/' #_resized_1_channel'#OCT_preprocessed_resized_3_channels' #OCT_preprocessed_resized_3_channels/'


def get_tiny_imagenet_path():
    return get_base_data_dir() + 'TinyImageNet/tiny-imagenet-200/'
