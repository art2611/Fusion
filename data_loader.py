import numpy as np
from PIL import Image
import torch.utils.data as data
from torchvision import transforms
import matplotlib.pyplot as plt
import os
from random import randrange
import random

class RegDBData(data.Dataset):
    def __init__(self, data_dir, trial, transform=None, colorIndex=None, thermalIndex=None):
        # Load training images (path) and labels
        data_dir = '../Datasets/RegDB/'
        train_color_list = data_dir + 'idx/train_visible_{}'.format(trial) + '.txt'
        train_thermal_list = data_dir + 'idx/train_thermal_{}'.format(trial) + '.txt'
        #Load color and thermal images + labels
        color_img_file, train_color_label = load_data(train_color_list)
        thermal_img_file, train_thermal_label = load_data(train_thermal_list)

        #Get real and thermal images with good shape in a list
        train_color_image = []
        for i in range(len(color_img_file)):
            img = Image.open(data_dir + color_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_color_image.append(pix_array)

        train_thermal_image = []
        for i in range(len(thermal_img_file)):
            img = Image.open(data_dir + thermal_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_thermal_image.append(pix_array)

        train_color_image = np.array(train_color_image)
        train_thermal_image = np.array(train_thermal_image)

        # Init color images / labels
        self.train_color_image = train_color_image
        self.train_color_label = train_color_label

        # Init themal images / labels
        self.train_thermal_image = train_thermal_image
        self.train_thermal_label = train_thermal_label

        self.transform = transform

        # Prepare index
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):
        #Dataset[i] return images from both modal and the corresponding label
        img1, target1 = self.train_color_image[self.cIndex[index]], self.train_color_label[self.cIndex[index]]
        img2, target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]

        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2, target1, target2

    def __len__(self):
        return len(self.train_color_label)


class SYSUData(data.Dataset):
    def __init__(self, data_dir, transform=None, colorIndex=None, thermalIndex=None):
        data_dir = '../Datasets/SYSU/'
        # Load training images (path) and labels
        train_color_image = np.load(data_dir + 'train_rgb_resized_img.npy')
        self.train_color_label = np.load(data_dir + 'train_rgb_resized_label.npy')

        train_thermal_image = np.load(data_dir + 'train_ir_resized_img.npy')
        self.train_thermal_label = np.load(data_dir + 'train_ir_resized_label.npy')

        # BGR to RGB
        self.train_color_image = train_color_image
        self.train_thermal_image = train_thermal_image
        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):
        img1, target1 = self.train_color_image[self.cIndex[index]], self.train_color_label[self.cIndex[index]]
        img2, target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]

        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2, target1, target2

    def __len__(self):
        return len(self.train_color_label)

def load_data(input_data_path):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]

    return file_image, file_label

# generate the idx of each person identity for instance, identity 10 have the index 100 to 109

def GenIdx(train_color_label, train_thermal_label):
    color_pos = []
    unique_label_color = np.unique(train_color_label)
    for i in range(len(unique_label_color)):
        tmp_pos = [k for k, v in enumerate(train_color_label) if v == unique_label_color[i]]
        color_pos.append(tmp_pos)

    thermal_pos = []
    unique_label_thermal = np.unique(train_thermal_label)
    for i in range(len(unique_label_thermal)):
        tmp_pos = [k for k, v in enumerate(train_thermal_label) if v == unique_label_thermal[i]]
        thermal_pos.append(tmp_pos)

    return color_pos, thermal_pos


def process_test_regdb(img_dir, modal='visible', trial = 1, split="paper_based"):

    input_visible_data_path = img_dir + f'idx/test_visible_{trial}.txt'
    input_thermal_data_path = img_dir + f'idx/test_thermal_{trial}.txt'

    with open(input_visible_data_path) as f:
        data_file_list = open(input_visible_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image_visible = [img_dir + '/' + s.split(' ')[0] for s in data_file_list]
        file_label_visible = [int(s.split(' ')[1]) for s in data_file_list]

    with open(input_thermal_data_path) as f:
        data_file_list = open(input_thermal_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image_thermal = [img_dir + '/' + s.split(' ')[0] for s in data_file_list]
        file_label_thermal = [int(s.split(' ')[1]) for s in data_file_list]

    #If required, return half of the dataset in two slice
    if modal == "VtoV" :
        file_image = file_image_visible
        file_label = file_label_visible
    if modal == "TtoT" :
        file_image = file_image_thermal
        file_label = file_label_thermal
    if modal == "TtoT" or modal == "VtoV" :
        first_image_slice_query = []
        first_label_slice_query = []
        sec_image_slice_gallery = []
        sec_label_slice_gallery = []
        #On regarde pour chaque id
        for k in range(len(np.unique(file_label))):
            appeared=[]
            # On choisit cinq personnes en query aléatoirement, le reste est placé dans la gallery (5 images)
            for i in range(5):
                rand = random.choice(file_image[k*10:k*10+9])
                while rand in appeared:
                    rand = random.choice(file_image[k*10:k*10+9])
                appeared.append(rand)
                first_image_slice_query.append(rand)
                first_label_slice_query.append(file_label[k*10])
            #On regarde la liste d'images de l'id k, on récupère les images n'étant pas dans query (5 images)
            for i in file_image[k*10:k*10+10] :
                if i not in appeared :
                    sec_image_slice_gallery.append(i)
                    sec_label_slice_gallery.append(file_label[k*10])
        return(first_image_slice_query, np.array(first_label_slice_query), sec_image_slice_gallery, np.array(sec_label_slice_gallery))

    if split == "paper_based":
        if modal == "VtoT" :
            return (file_image_visible, np.array(file_label_visible), file_image_thermal,
                np.array(file_label_thermal))
        elif modal == "TtoV":
            return(file_image_thermal, np.array(file_label_thermal), file_image_visible, np.array(file_label_visible))

    elif split=="experience_based" :
        first_image_slice_query = []
        first_label_slice_query = []
        sec_image_slice_gallery = []
        sec_label_slice_gallery = []
        first_image_slice_query_2 = []
        sec_image_slice_gallery_2 = []
        # On regarde pour chaque id
        for k in range(len(np.unique(file_label_visible))):
            appeared = []
            # On choisit cinq personnes en query aléatoirement, le reste est placé dans la gallery (5 images)
            for i in range(5):
                rand = randrange(10)
                while rand in appeared:
                    rand = randrange(10)
                appeared.append(rand)
                if modal == "VtoT" :
                    first_image_slice_query.append(file_image_visible[k * 10 + rand])
                    first_label_slice_query.append(file_label_visible[k * 10])
                elif modal == "TtoV" :
                    first_image_slice_query.append(file_image_thermal[k * 10 + rand])
                    first_label_slice_query.append(file_label_thermal[k * 10])
                elif modal == "BtoB" :
                    first_image_slice_query.append(file_image_visible[k * 10 + rand])
                    first_image_slice_query_2.append(file_image_thermal[k * 10 + rand])
                    first_label_slice_query.append(file_label_visible[k * 10])

            # On regarde la liste d'images de l'id k, on récupère les images n'étant pas dans query (5 images)
            for i in [w for w in range(10)]:
                if i not in appeared:
                    if modal == "VtoT":
                        sec_image_slice_gallery.append(file_image_visible[k * 10 + i])
                        sec_label_slice_gallery.append(file_label_visible[k * 10])
                    elif modal == "TtoV":
                        sec_image_slice_gallery.append(file_image_thermal[k * 10 + i])
                        sec_label_slice_gallery.append(file_label_thermal[k * 10])
                    elif modal =="BtoB" :
                        sec_image_slice_gallery.append(file_image_visible[k * 10 + i])
                        sec_image_slice_gallery_2.append(file_image_thermal[k * 10 + i])
                        sec_label_slice_gallery.append(file_label_visible[k * 10])
            print(f"sec_image_slice_gallery len : {len(first_image_slice_query)}")
            print(f"sec_image_slice_gallery_2 len : {len(first_image_slice_query_2)}")
            print(f"sec_label_slice_gallery len : {len(first_label_slice_query)}")
        if modal == "BtoB" :
            return (first_image_slice_query, first_image_slice_query_2, np.array(first_label_slice_query), \
                    sec_image_slice_gallery, sec_image_slice_gallery_2, np.array(sec_label_slice_gallery))
        return (first_image_slice_query, np.array(first_label_slice_query), sec_image_slice_gallery,
                np.array(sec_label_slice_gallery))


def process_query_sysu(data_path, method, trial=0, mode='all', relabel=False, reid="VtoT"):
    random.seed(trial)
    print("query")
    if mode == 'all':
        rgb_cameras = ['cam1', 'cam2', 'cam4', 'cam5']
        ir_cameras = ['cam3', 'cam6']
    elif mode == 'indoor':
        rgb_cameras = ['cam1', 'cam2']
        ir_cameras = ['cam3', 'cam6']

    if method == "test":
        print("Test set called")
        file_path = os.path.join(data_path, 'exp/test_id.txt')
    elif method == "valid":
        print("Validation set called")
        file_path = os.path.join(data_path, 'exp/val_id.txt')

    files_rgb = []
    files_ir = []

    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%04d" % x for x in ids]

    for id in sorted(ids):
        for cam in rgb_cameras:
            img_dir = os.path.join(data_path, cam, id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
                files_rgb.extend(new_files)
        for cam in ir_cameras:
            img_dir = os.path.join(data_path, cam, id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
                #
                files_ir.extend(new_files)

    query_img = []
    query_id = []
    query_cam = []
    if reid=="VtoT" :
        files = files_rgb
    elif reid=="TtoV" :
        files = files_ir
    for img_path in files:
        camid, pid = int(img_path[-15]), int(img_path[-13:-9])
        query_img.append(img_path)
        query_id.append(pid)
        query_cam.append(camid)
    #print(query_img)
    return query_img, np.array(query_id), np.array(query_cam)


def process_gallery_sysu(data_path, method, mode='all', trial=0, relabel=False, reid="VtoT"):
    random.seed(trial)

    if mode == 'all':
        rgb_cameras = ['cam1', 'cam2', 'cam4', 'cam5']
        ir_cameras = ['cam3', 'cam6']
    elif mode == 'indoor':
        rgb_cameras = ['cam1', 'cam2']
        ir_cameras = ['cam3', 'cam6']

    if method == "test":
        print("Test set called")
        file_path = os.path.join(data_path, 'exp/test_id.txt')
    elif method == "valid":
        print("Validation set called")
        file_path = os.path.join(data_path, 'exp/val_id.txt')

    files_rgb = []
    files_ir = []
    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%04d" % x for x in ids]

    for id in sorted(ids):
        for cam in rgb_cameras:
            img_dir = os.path.join(data_path, cam, id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
                files_rgb.append(random.choice(new_files))
            # else :
            #     print(f'this dir does not exist : {img_dir}')
        for cam in ir_cameras:
            img_dir = os.path.join(data_path, cam, id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
                files_ir.append(random.choice(new_files))
    gall_img = []
    gall_id = []
    gall_cam = []
    if reid=="VtoT" :
        files = files_ir
    elif reid=="TtoV" :
        files = files_rgb
    for img_path in files:
        camid, pid = int(img_path[-15]), int(img_path[-13:-9])
        gall_img.append(img_path)
        gall_id.append(pid)
        gall_cam.append(camid)

    return gall_img, np.array(gall_id), np.array(gall_cam)

def process_test_single_sysu(data_path, method, trial=0, mode='all', relabel=False, reid="VtoT"):
    random.seed(trial)
    print("query")
    if mode == 'all':
        rgb_cameras = ['cam1', 'cam2', 'cam4', 'cam5']
        ir_cameras = ['cam3', 'cam6']
    elif mode == 'indoor':
        rgb_cameras = ['cam1', 'cam2']
        ir_cameras = ['cam3', 'cam6']

    if method == "test":
        print("Test set called")
        file_path = os.path.join(data_path, 'exp/test_id.txt')
    elif method == "valid":
        print("Validation set called")
        file_path = os.path.join(data_path, 'exp/val_id.txt')

    files_rgb = []
    files_ir = []

    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%04d" % x for x in ids]


    files_query_visible = []
    files_gallery_visible = []
    files_query_thermal = []
    files_gallery_thermal = []
    for id in sorted(ids):
        #Selection of 1 img for gallery per cam and per id, the rest as query
        for cam in rgb_cameras:
            img_dir = os.path.join(data_path, cam, id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
                rand = random.choice(new_files)
                files_gallery_visible.append(rand)
                for w in new_files:
                    if w != rand:
                        files_query_visible.append(w)

        for cam in ir_cameras:
            img_dir = os.path.join(data_path, cam, id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
                rand = random.choice(new_files)
                files_gallery_thermal.append(rand)
                for w in new_files:
                    if w != rand:
                        files_query_thermal.append(w)
    query_img = []
    query_id = []
    query_cam = []
    gall_img = []
    gall_id = []
    gall_cam = []

    if reid == "VtoV":
        files_query = files_query_visible
        files_gallery = files_gallery_visible
    elif reid == "TtoT":
        files_query = files_query_thermal
        files_gallery = files_gallery_thermal

    for img_path in files_query:
        camid, pid = int(img_path[-15]), int(img_path[-13:-9])
        query_img.append(img_path)
        query_id.append(pid)
        query_cam.append(camid)

    for img_path in files_gallery :
        camid, pid = int(img_path[-15]), int(img_path[-13:-9])
        gall_img.append(img_path)
        gall_id.append(pid)
        gall_cam.append(camid)

    return query_img, np.array(query_id), np.array(query_cam), gall_img, np.array(gall_id), np.array(gall_cam)

class TestData(data.Dataset):
    def __init__(self, test_img_file, test_label, transform=None, img_size = (144,288)):

        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            pix_array = np.array(img)
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        self.transform = transform

    def __getitem__(self, index):
        img1,  target1 = self.test_image[index],  self.test_label[index]
        img1 = self.transform(img1)
        return img1, target1

    def __len__(self):
        return len(self.test_image)

class TestData_both(data.Dataset):
    def __init__(self, test_img_file1,test_img_file2, test_label, transform=None, img_size = (144,288)):

        test_image1 = []
        test_image2 = []
        for i in range(len(test_img_file1)):
            img = Image.open(test_img_file1[i])
            img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            pix_array = np.array(img)
            test_image1.append(pix_array)
        for i in range(len(test_img_file2)):
            img = Image.open(test_img_file2[i])
            img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            pix_array = np.array(img)
            test_image2.append(pix_array)
        test_image1 = np.array(test_image1)
        test_image2 = np.array(test_image2)

        self.test_image1 = test_image1
        self.test_image2 = test_image2
        self.test_label = test_label
        self.transform = transform

    def __getitem__(self, index):
        img1, img2, target1 = self.test_image1[index], self.test_image2[index], self.test_label[index]
        img1 = self.transform(img1)
        img2 = self.transform(img2)
        return img1, img2, target1

    def __len__(self):
        #Should be the same len for both image 1 and image 2
        return len(self.test_image1)

# Print some of the images :
# print(trainset.train_color_image.shape)
# w=0
# for i in range(0, 250, 10):
#     w += 1
#     print(i)
#     plt.subplot(5,5,w)
#     plt.imshow(trainset.train_color_image[i])
# plt.show()

# testing set
# query_img, query_label = process_test_regdb(data_path, trial=args.trial, modal='visible')
# gall_img, gall_label = process_test_regdb(data_path, trial=args.trial, modal='thermal')