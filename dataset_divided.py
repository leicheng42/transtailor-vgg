#!/usr/bin/python
# -*- coding: utf-8 -*-
# Description: 
# Created: lei.cheng 2022/4/1
# Modified: lei.cheng 2022/4/1
"""
    将原始数据集进行划分成训练集、验证集
"""

import os
import glob
import random
import shutil

dataset_dir = os.path.join("datasets/Images")
train_dir = os.path.join("datasets", "stanford_train")
test_dir = os.path.join("datasets", "stanford_test")


dataset_dir = os.path.join("datasets/stanford_train")
train_dir = os.path.join("datasets", "stanford_train_1")
test_dir = os.path.join("datasets", "stanford_test_1")

train_per = 0.1
test_per = 0.05

def makedir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)


if __name__ == '__main__':

    for root, dirs, files in os.walk(dataset_dir):
        for sDir in dirs:
            imgs_list = glob.glob(os.path.join(root, sDir, '*.jpg'))
            random.seed(666)
            random.shuffle(imgs_list)
            imgs_num = len(imgs_list)

            train_point = int(imgs_num * train_per)
            test_point = int(imgs_num * (train_per + test_per))

            for i in range(imgs_num):
                if i < train_point:
                    out_dir = os.path.join(train_dir, sDir)
                elif i < test_point:
                    out_dir = os.path.join(test_dir, sDir)
                else:
                    continue

                makedir(out_dir)
                out_path = os.path.join(out_dir, os.path.split(imgs_list[i])[-1])
                shutil.copy(imgs_list[i], out_path)

            print('Class:{}, train:{}, test:{}'.format(sDir, train_point, test_point - train_point))
