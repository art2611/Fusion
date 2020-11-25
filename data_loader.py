import numpy as np
from PIL import Image
import torch.utils.data as data
from torchvision import transforms
import matplotlib.pyplot as plt


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


def process_test_regdb(img_dir, trial=1, modal='visible'):
    if modal == 'visible':
        input_data_path = img_dir + 'idx/test_visible_{}'.format(trial) + '.txt'
    elif modal == 'thermal':
        input_data_path = img_dir + 'idx/test_thermal_{}'.format(trial) + '.txt'

    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [img_dir + '/' + s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]

    return file_image, np.array(file_label)

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