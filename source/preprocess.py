# -*- coding: utf-8 -*-
# @Time    : 28/03/22
# @Author  : Jackie LIU
# @File    : preprocess.py
# @Software: PyCharm

# convert the dataformat from data/classes foler to data/train/classes and data/val/claseses

import shutil
import sys
from sys import exit
from shutil import copyfile
import os

def self_mkdir(folder):
    isExists = os.path.exists(folder)
    if not isExists:
        os.makedirs(folder)
        print('path of %s is build' % (folder))
    else:
        shutil.rmtree(folder)
        os.makedirs(folder)
        print('path of %s already exist and rebuild' % (folder))


def train_test_split(img_folder, savedpath, cate):
    """train test split"""
    sets = ['train', 'valid']
    imgs_ls = os.listdir(img_folder)

    image_len = len(imgs_ls)
    num_train = image_len - int(image_len * 0.2)
    num_test = int(image_len * 0.2)
    print("<<<< NUM TRAIN: ", num_train)
    print("<<<< NUM TEST: ", num_test)

    train_path = os.path.join(savedpath, 'Train', cate)
    val_path = os.path.join(savedpath, 'Validation', cate)
    self_mkdir(train_path)
    self_mkdir(val_path)

    i = 0
    for image_id in imgs_ls:
        if i < num_train:
            source = os.path.join(img_folder, imgs_ls[i])
            target = os.path.join(train_path, imgs_ls[i])
            # print ("source", source)
            # print ("target", target)

        else:
            source = os.path.join(img_folder, imgs_ls[i])
            target = os.path.join(val_path, imgs_ls[i])
        # adding exception handling
        try:
            copyfile(source, target)
        except IOError as e:
            print("Unable to copy file. %s" % e)
            exit(1)
        except:
            print("Unexpected error:", sys.exc_info())
            exit(1)

        i = i + 1
    return

def main(input_foler,output_folder):
    #ls categories
    cate_ls = os.listdir(input_foler)
    for i in cate_ls:
        if i!= '.DS_Store':
            img_folder = os.path.join(input_foler,i)
            train_test_split(img_folder, output_folder, i)

    print ("process finished")


if __name__ == '__main__':
    main('../data','./data')

