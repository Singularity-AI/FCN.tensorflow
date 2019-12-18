import os
import cv2
import pdb
import json
import glob
import h5py
import time
import random
import logging
import argparse
import functools
import threading
import numpy as np
import pandas as pd
import tensorflow as tf

from PIL import Image
from PIL import ImageFile
from math import ceil
ImageFile.LOAD_TRUNCATED_IMAGES = True # to solve error: "OSError: image file is truncated"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


logger = None
def _get_logger(logger_level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logger_level)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


class Dataloader(object):
    def __init__(self, hdf5_files, label, labels_map, crop_patch_nb, percent, is_training=False, is_shuffle=True, aug_para={}):
        self.label = label
        self.labels_map = labels_map
        self.is_training = is_training
        self.is_shuffle = is_shuffle
        self.crop_patch_nb = crop_patch_nb
        self.percent = percent
        self.hdf5_files = hdf5_files
        self.nb_samples_per_epoch = 0
        # self.hdf5_files = self.get_hdf5_files(hdf5_path)
        # self.data_aug_obj = da.DataAugmentation(**aug_para)

    def get_hdf5_files(self, hdf5_path):
        return glob.glob(os.path.join(hdf5_path, '*hdf5'))
    
    def data_augmentation(self, image):
        for (func, kwargs) in self.data_aug_obj.pipeline():
            image = func(image.astype(np.float32), rng=None, **kwargs)
        return image, labels, image_name, rate

    def get_nb_samples_per_epoch(self):
        for h5file in self.hdf5_files: 
            with h5py.File(h5file, 'r') as hdf5_handle:
                label = h5file.split('/')[-3]
                self.nb_samples_per_epoch += hdf5_handle[label]['patches'].shape[0]
        return self.nb_samples_per_epoch


    def image_generator(self):
        while True:
            for h5file in self.hdf5_files:
                logger.debug('dealing with {}' .format(h5file))
                with h5py.File(h5file, 'r') as hdf5_handle:
                    label = h5file.split('/')[-3]
                    image_data_num = hdf5_handle[label]['patches'].shape[0]
                    batches_list = list(range(image_data_num // self.crop_patch_nb))
                    logger.debug('{} batches_list-1: {}' .format(h5file.split('/')[-1], len(batches_list)))
                    valid_batch_num = int(self.percent * len(batches_list))
                    batches_list = batches_list[valid_batch_num:valid_batch_num*2]
                    logger.debug('{} batches_list: {}-{}' .format(h5file.split('/')[-1], len(batches_list), valid_batch_num))
                    # random.shuffle(batches_list)
                    for index, elem in enumerate(batches_list):
                        elem_start = elem * self.crop_patch_nb  # index of the first image in this batch
                        elem_end = min([(elem + 1) * self.crop_patch_nb, image_data_num])  # index of the last image in this batch
                        images = hdf5_handle[label]['patches'][elem_start:elem_end, ...] # read batch images
                        yield images, np.array([self.labels_map[label]] * self.crop_patch_nb), np.array(range(elem_start, elem_end))


def custom_shuffle(list_to_be_shuffle, is_shuffle=True):
    """
       Args:
           list_to_be_shuffle: A list will be to shuffle.
           is_shuffle: bool, if True, list will be shuffle, if False, list will remain the same.

       Returns:
           list_to_be_shuffle:
    """
    if is_shuffle:
        shuffled_index = list(range(len(list_to_be_shuffle)))
        # random.seed(12345)
        random.shuffle(shuffled_index)
        list_to_be_shuffle = [list_to_be_shuffle[i] for i in shuffled_index]
    return list_to_be_shuffle


def batch_generator(generators, batch_size, is_shuffle=True):
    
    while True:
        start = time.time()
        batch_images, batch_labels, batch_indexes = next(generators[0])
        for idx, g in enumerate(generators[1:]):
            logger.debug(' the generator of {} is cropping now, id: {}!'.format(g, idx))
            images, labels, indexes = next(g)
            batch_images = np.concatenate((batch_images, images), axis=0)
            batch_labels = np.concatenate((batch_labels, labels), axis=0)
            batch_indexes = np.concatenate((batch_indexes, indexes), axis=0)
        
        #shuffle
        if is_shuffle:
            shuffle_index = random.sample(list(range(batch_images.shape[0])), batch_size)
            # shuffle_index = np.random.permutation(batch_images.shape[0])
            batch_images = batch_images[shuffle_index]
            batch_labels = batch_labels[shuffle_index]
            batch_indexes = batch_indexes[shuffle_index]

        # logger.info('====================================')
        # logger.info('batch time: {}' .format(time.time() - start))
        # logger.info('====================================')

        yield batch_images, batch_labels, batch_indexes


def get_instance_datagen(base_path,
                         labels,
                         labels_map,
                         nb_per_class,
                         percent,
                         is_training=False,
                         is_shuffle=True,
                         aug_para={},
                         logger_level=logging.INFO):
    """Generate tf.data.Dataset object for each class.
        Arguments:
            base_path: Base path of json file, for detail info, see README.md.
            labels: List of all classes need to be generator.
            labels_map: transform string labels to int32.
            nb_per_class: List of numbers of samples for each class.
            is_training: Default 'False', if set to 'True', generator data for training, or for eval.
            is_shuffle: Default 'True', If set to 'True', it will be shuffle.
            aug_para: Parameters for data augmentation.
            logger_level: Log level includes DEBUG,INFO,WARNING,ERROR,FATAL.
        Returns:
            obj: Dataloader object for each class in labels list.
            generators: List of generator for each class in labels list.
            nb_samples_per_epoch: Sample numbers for each class in labels list.
    """
    global logger
    if logger is None:
        logger = _get_logger(logger_level)

    # get all jsons and hdf5 files for all label in labels.
    jsons_and_hdf5s = get_hdf5_and_json_for_each_label(base_path, labels, labels_map)

    nb_samples_per_epoch = []
    generators = []
    objects = []
    for label, num in zip(labels.keys(), nb_per_class):
        logger.info('dealing with {} data, please wait patiently!'.format(label))
        obj = Dataloader(hdf5_files=jsons_and_hdf5s[label][0],
                         label=label,
                         labels_map=labels_map,
                         crop_patch_nb=num,
                         percent=percent,
                         is_training=is_training,
                         is_shuffle=is_shuffle,
                         aug_para=aug_para)
        objects.append(obj)
        generators.append(obj.image_generator())
        nums = obj.get_nb_samples_per_epoch()
        nb_samples_per_epoch.append(nums)
        logger.info('the lenght of {} patches list: {}'.format(label, nums))
    return objects, generators, nb_samples_per_epoch


def get_all_hdf5s_and_jsons(base_path, labels):
    jsons = []
    hdf5s = []
    for label in labels:
        josn_files = glob.glob(os.path.join(base_path, label, 'json', '*.json'))
        h5_files = glob.glob(os.path.join(base_path, label, 'hdf5', '*.hdf5'))
        jsons.extend(josn_files)
        hdf5s.extend(h5_files)
    return jsons, hdf5s


def get_hdf5_and_json_for_each_label(base_path, labels, labels_map):  
    jsons, hdf5s = get_all_hdf5s_and_jsons(base_path, labels_map.keys())
    df_jsons = pd.DataFrame({'jsons': jsons})
    df_hdf5s = pd.DataFrame({'hdf5s': hdf5s})

    jsons_and_hdf5s = {}
    for label in labels.keys():
        # jsons_and_hdf5s[label] = [df_jsons[df_jsons.jsons.str.contains(labels[label])].values.T.tolist()[0], \
        #     df_hdf5s[df_hdf5s.hdf5s.str.contains(labels[label])].values.T.tolist()[0]]

        jsons_and_hdf5s[label] = [df_hdf5s[df_hdf5s.hdf5s.str.contains(labels[label])].values.T.tolist()[0]]
    return jsons_and_hdf5s
    

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    logger = _get_logger(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--logger_level", type=str, default=logging.INFO, help="logging level")
    parser.add_argument("--base_path", type=str,
                        # default='/mnt/share/dulicui/project/colorectal/dataset/train_and_valid_dataset/train_dataset_hdf5_uncompressed/',
                        # default='/mnt/share/dulicui/project/colorectal/dataset/train_and_valid_dataset/train_dataset_hdf5_lzf/',
                        default="/mnt/disk_share/data/colon/colon_160_224/",
                        help="json base folder path")
    parser.add_argument("--anno_mask_non_zero_rate", type=float, default=0.7, help="non zero rate of mask in image")
    parser.add_argument("--max_retry_cnt", type=int, default=5, help="max cnts to retry")
    parser.add_argument("--num_threads", type=int, default=[8, 8], help="numbers of threads for data queue")
    parser.add_argument("--max_steps", type=int, default=10, help="the max step of training")
    parser.add_argument("--percent", type=float, default=0.2, help="the max step of training")
    FLAGS, unparsed = parser.parse_known_args()

    labels_map = {
        'high_ad': np.int32(1),
        'mid_ad': np.int32(1),
        'low_ad': np.int32(1),
        'mucinous_ad': np.int32(1),
        'ring_ad': np.int32(1),
        'mixed_ad': np.int32(1),
        'inflammation': np.int32(0),
        'lymphocyte': np.int32(0),
        'fat': np.int32(0),
        'smooth_muscle': np.int32(0),
        'normal_mucosa': np.int32(0),
        'neutrophil': np.int32(0),
        'plasmacyte': np.int32(0),
        'histocyte': np.int32(0),
        'eosnophils': np.int32(0),
        # 'ignore': np.int32(0)
    }

    labels = {
        'high_ad': 'high_ad',
        'mid_ad': 'mid_ad',
        'low_ad': 'low_ad',
        'mucinous_ad': 'mucinous_ad',
        'ring_ad': 'ring_ad',
        'mixed_ad': 'mixed_ad',
        'inflammation': 'inflammation',
        'lymphocyte': 'lymphocyte',
        'fat': 'fat',
        'smooth_muscle': 'smooth_muscle',
        'normal_mucosa': 'normal_mucosa',
        'neutrophil': 'neutrophil',
        'plasmacyte': 'plasmacyte|histocyte|eosnophils'
    }

    nb_per_class = [1, 14, 7, 4, 1, 5, 10, 3, 5, 8, 4, 1, 1]
    batch_size = sum(nb_per_class)

    # jsons_and_hdf5s = get_hdf5_and_json_for_each_label(FLAGS.base_path, labels, labels_map)
    objs, generators, nb_samples_per_epoch = \
        get_instance_datagen(base_path=FLAGS.base_path,
                             labels=labels,
                             labels_map=labels_map,
                             nb_per_class=nb_per_class,
                             percent=FLAGS.percent,
                             is_training=True,
                             is_shuffle=True,
                             #aug_para=aug_para,
                             )  

    batch_g = batch_generator(generators, batch_size, is_shuffle=True)
    for images, labels, indexes in batch_g:
        logger.info('image shape: {}, labels shape: {}' .format(images.shape, labels.shape)) 

    print('INFO over!')

