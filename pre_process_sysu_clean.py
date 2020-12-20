import numpy as np
from PIL import Image
import pdb
import os

import sys

data_path = '../Datasets/SYSU/'
file_path_train = os.path.join(data_path, 'exp/train_id.txt')
file_path_valid = os.path.join(data_path, 'exp/val_id.txt')

###GET VALID AND TRAIN initial IDS
with open(file_path_train, 'r') as file:
    ids = file.read().splitlines()
    ids = [int(y) for y in ids[0].split(',')]
    all_ids = ["%04d" % x for x in ids]

with open(file_path_valid, 'r') as file:
    ids = file.read().splitlines()
    ids = [int(y) for y in ids[0].split(',')]
    all_ids.extend(["%04d" % x for x in ids])


#PREPARE 5 FOLDS ids
training_lists=[[],[],[],[],[]]
val_lists = [[],[],[],[],[]]
for i in range(5):
    for j in range(1,len(all_ids)+1):
        if j >= 79*i + 1 and j <= (79*(i+1)) :
            val_lists[i].append(all_ids[j-1])
        else :
            training_lists[i].append(all_ids[j-1])


rgb_cameras = ['cam1', 'cam2', 'cam4', 'cam5']
ir_cameras = ['cam3', 'cam6']

### GET imgs associated to ids
files_rgb_train = [[], [], [], [], []]
files_ir_train = [[], [], [], [], []]
files_rgb_val = [[],[],[],[],[]]
files_ir_val = [[],[],[],[],[]]
for k in range(5) :
    for id in sorted(training_lists[k]):
        for cam in rgb_cameras:
            img_dir = os.path.join(data_path, cam, id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
                files_rgb_train[k].extend(new_files)

        for cam in ir_cameras:
            img_dir = os.path.join(data_path, cam, id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
                files_ir_train[k].extend(new_files)

    for id in sorted(val_lists[k]):
        for cam in rgb_cameras:
            img_dir = os.path.join(data_path, cam, id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
                files_rgb_val[k].extend(new_files)

        for cam in ir_cameras:
            img_dir = os.path.join(data_path, cam, id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
                files_ir_val[k].extend(new_files)

pid2label_train = [[], [], [], [], []]
pid2label_valid = [[], [], [], [], []]

for k in range(5):
    # relabel
    pid_container = set()
    for img_path in files_ir_train[k]:
        pid = int(img_path[-13:-9])
        pid_container.add(pid)
    pid2label_train[k] = {pid: label for label, pid in enumerate(pid_container)}
    #print(pid2label_train[k])

for k in range(5) :
    pid_container = set()
    for img_path in files_ir_val[k] :
        pid = int(img_path[-13:-9])
        pid_container.add(pid)
    pid2label_valid[k] = {pid: label for label, pid in enumerate(pid_container)}
    #print(pid2label_valid[k])

fix_image_width = 144
fix_image_height = 288


def read_imgs(train_image,k, mode):
    train_img = []
    train_label = []
    for img_path in train_image:
        # img
        img = Image.open(img_path)
        img = img.resize((fix_image_width, fix_image_height), Image.ANTIALIAS)
        pix_array = np.array(img)

        train_img.append(pix_array)

        # label
        pid = int(img_path[-13:-9])
        if mode == "train" :
            pid = pid2label_train[k][pid]
        if mode == "val" :
            pid = pid2label_valid[k][pid]
        train_label.append(pid)

    return np.array(train_img), np.array(train_label)

#Output .npy files

for k in range(5):

    # rgb imges train
    train_img, train_label = read_imgs(files_rgb_train[k], k, "train")
    np.save(data_path + f'train_rgb_img_{k}.npy', train_img)
    np.save(data_path +  f'train_rgb_label_{k}.npy', train_label)

    # ir imges train
    train_img, train_label = read_imgs(files_ir_train[k], k, "train")
    np.save(data_path +  f'train_ir_img_{k}.npy', train_img)
    np.save(data_path + f'train_ir_label_{k}.npy', train_label)

    train_img, train_label = read_imgs(files_rgb_val[k], k, "val")

    np.save(data_path + f'valid_rgb_img_{k}.npy', train_img)
    np.save(data_path + f'valid_rgb_label_{k}.npy', train_label)

    # ir imges valid
    train_img, train_label = read_imgs(files_ir_val[k], k, "val")
    np.save(data_path + f'valid_ir_img_{k}.npy', train_img)
    np.save(data_path + f'valid_ir_label_{k}.npy', train_label)
