import numpy as np
from PIL import Image
import os

data_path = '../Datasets/RegDB/'

file_path_train_visible =  os.path.join(data_path, 'idx/train_visible_1.txt')
file_path_train_thermal =  os.path.join(data_path, 'idx/train_thermal_1.txt')


# train_thermal_list = data_dir + 'idx/train_thermal_1.txt'

###GET all initial training ids (50% of the dataset)
with open(file_path_train_visible) as f:
    data_file_list = open(file_path_train_visible, 'rt').read().splitlines()
    # Get full list of image and labels
    file_image_visible = [data_path + '/' + s.split(' ')[0] for s in data_file_list]

with open(file_path_train_thermal) as f:
    data_file_list = open(file_path_train_thermal, 'rt').read().splitlines()
    # Get full list of image and labels
    file_image_thermal = [data_path + '/' + s.split(' ')[0] for s in data_file_list]
    file_label = [int(s.split(' ')[1]) for s in data_file_list]

for k in range(10) :
    file_image_visible.pop()
    file_image_thermal.pop()
    file_label.pop()

all_ids = file_label

#PREPARE 5 FOLDS ids
training_lists=[[],[],[],[],[]]
val_lists = [[],[],[],[],[]]
for i in range(5):
    for j in range(1,len(all_ids)+1):
        if j >= 41*10*i + 1 and j <= (41*10*(i+1)) :
            val_lists[i].append(all_ids[j-1])
        else :
            training_lists[i].append(all_ids[j-1])

# for k in range(5) :
#     print(len(val_lists[k]))


### GET imgs associated to ids
files_rgb_train = [[], [], [], [], []]
files_ir_train = [[], [], [], [], []]
files_rgb_val = [[],[],[],[],[]]
files_ir_val = [[],[],[],[],[]]

for k in range(5) :
    for j in np.unique(training_lists[k]) :
        for i in range(10):
            files_rgb_train[k].append(file_image_visible[j*10 + i])
            files_ir_train[k].append(file_image_thermal[j*10 + i])
    for j in np.unique(val_lists[k]) :
        for i in range(10):
            files_rgb_val[k].append(file_image_visible[j*10 + i])
            files_ir_val[k].append(file_image_thermal[j*10 + i])

# for k in range(5) :
#     print(files_rgb_train[k])
#     print(training_lists[k])
#     print(len(files_ir_train[k]))
#     print(len(files_rgb_val[k]))
#     print(len(files_ir_val[k]))

fix_image_width = 144
fix_image_height = 288
def read_imgs(train_image, k):
    train_img = []
    for img_path in train_image:
        # img
        img = Image.open(img_path)
        img = img.resize((fix_image_width, fix_image_height), Image.ANTIALIAS)
        pix_array = np.array(img)

        train_img.append(pix_array)

    return np.array(train_img)


#Output .npy files

for k in range(5):
    # rgb imges train
    train_img = read_imgs(files_rgb_train[k], k)
    # print(train_img[0])
    np.save(data_path + f'train_rgb_img_{k}.npy', train_img)
    np.save(data_path +  f'train_label_{k}.npy', np.array(training_lists[k]))

    # ir imges train
    train_img  = read_imgs(files_ir_train[k], k)
    np.save(data_path +  f'train_ir_img_{k}.npy', train_img)


    train_img = read_imgs(files_rgb_val[k], k)
    np.save(data_path + f'valid_rgb_img_{k}.npy', train_img)
    np.save(data_path + f'valid_label_{k}.npy', np.array(val_lists[k]))

    # ir imges valid
    train_img = read_imgs(files_ir_val[k], k)
    np.save(data_path + f'valid_ir_img_{k}.npy', train_img)
