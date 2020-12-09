import os
import sys
import torch
import torch.utils.data
from torch.autograd import Variable
import time
from data_loader import *
import numpy as np
from model_layer5 import Network_layer5
from model_layer3 import Network_layer3
from model_layer1 import Network_layer1
from evaluation import eval_regdb, eval_sysu
from torchvision import transforms
import torch.utils.data
from multiprocessing import freeze_support
from tensorboardX import SummaryWriter
import argparse
from datetime import date


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# net = Network(class_num=nclass).to(device)

pool_dim = 2048

# Init variables :
img_w = 144
img_h = 288
test_batch_size = 64
batch_num_identities = 8  # 8 different identities in a batch
num_of_same_id_in_batch = 4  # Number of same identity in a batch
workers = 4
lr = 0.001
checkpoint_path = '../save_model/'
#
parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--fusion', default='layer1', help='Layer to fuse data')
parser.add_argument('--dataset', default='regdb', help='dataset name: regdb or sysu]')
parser.add_argument('--reid', default='VtoT', help='Visible to thermal reid')
parser.add_argument('--trained', default='VtoT', help='Model trained based on VtoT validation')
parser.add_argument('--split', default='paper_based', help='How to split data')
args = parser.parse_args()

split_list = ["paper_based", "experience_based"]
if args.split not in split_list:
    sys.exit(f"--split should be in {split_list}")

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((img_h, img_w)),
    transforms.ToTensor(),
    normalize,
])
today = date.today()
# dd/mm/YY
d1 = today.strftime("%d/%m/%Y")

writer = SummaryWriter(f"runs/{args.trained}_{args.fusion}_FusionModel_{args.reid}-test_{args.dataset}_day{d1}_{time.time()}")


if args.dataset == 'sysu':
    nclass = 296
    data_path = '../Datasets/SYSU/'
    suffix = f'SYSU_{args.trained}_person_fusion({num_of_same_id_in_batch})_same_id({batch_num_identities})_lr_{lr}'
elif args.dataset == 'regdb':
    nclass = 206
    data_path = '../Datasets/RegDB/'
    suffix = f'RegDB_{args.trained}_person_fusion({num_of_same_id_in_batch})_same_id({batch_num_identities})_lr_{lr}'
    print(f'model_to_load : {suffix}')

def extract_gall_feat(gall_loader, ngall, net):
    net.eval()
    print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat_pool = np.zeros((ngall, pool_dim))
    gall_feat_fc = np.zeros((ngall, pool_dim))
    if args.reid == "VtoT" or args.reid == "TtoT" :
        test_mode = 2
    if args.reid == "TtoV" or args.reid == "VtoV":
        test_mode = 1
    if args.reid == "BtoB" :
        test_mode = 0
        with torch.no_grad():
            for batch_idx, (input1, input2, label) in enumerate(gall_loader):
                batch_num = input1.size(0) + input2.size(0)

                input1 = Variable(input1.cuda())
                input2 = Variable(input2.cuda())
                feat_pool, feat_fc = net(input1, input2, modal=test_mode)
                gall_feat_pool[ptr:ptr + batch_num, :] = feat_pool.detach().cpu().numpy()
                gall_feat_fc[ptr:ptr + batch_num, :] = feat_fc.detach().cpu().numpy()
                ptr = ptr + batch_num
        print('Extracting Time:\t {:.3f}'.format(time.time() - start))
    else :
        print(f"Gallery test on mode {test_mode} supposed to be 1 if visible or 2 if thermal")
        with torch.no_grad():
            for batch_idx, (input, label) in enumerate(gall_loader):
                batch_num = input.size(0)
                input = Variable(input.cuda())
                feat_pool, feat_fc = net(input, input, modal=test_mode)
                gall_feat_pool[ptr:ptr + batch_num, :] = feat_pool.detach().cpu().numpy()
                gall_feat_fc[ptr:ptr + batch_num, :] = feat_fc.detach().cpu().numpy()
                ptr = ptr + batch_num
        print('Extracting Time:\t {:.3f}'.format(time.time() - start))
    return gall_feat_pool, gall_feat_fc


def extract_query_feat(query_loader, nquery, net):
    net.eval()
    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat_pool = np.zeros((nquery*2, pool_dim))
    query_feat_fc = np.zeros((nquery, pool_dim))
    print(query_feat_pool.shape)
    if args.reid == "VtoT" or args.reid == "VtoV":
        test_mode = 1
    if args.reid == "TtoV" or args.reid== "TtoT" :
        test_mode = 2
    if args.reid == "BtoB" :
        test_mode = 0
        print(f"Query test on mode {test_mode} supposed to be 1 if visible or 2 if thermal" )
        with torch.no_grad():
            for batch_idx, (input1, input2, label) in enumerate(query_loader):
                batch_num = input1.size(0) + input2.size(0)
                print(batch_num)
                print(input1.size(0))
                print(input2.size(0))
                print(label)
                print(batch_idx)
                input1 = Variable(input1.cuda())
                input2 = Variable(input2.cuda())
                feat_pool, feat_fc = net(input1, input2, modal=test_mode)

                query_feat_pool[ptr:ptr + batch_num, :] = feat_pool.detach().cpu().numpy()
                query_feat_fc[ptr:ptr + batch_num, :] = feat_fc.detach().cpu().numpy()
                ptr = ptr + batch_num
        print('Extracting Time:\t {:.3f}'.format(time.time() - start))
    else :
        print(f"Query test on mode {test_mode} supposed to be 1 if visible or 2 if thermal" )
        with torch.no_grad():
            for batch_idx, (input, label) in enumerate(query_loader):
                batch_num = input.size(0)
                input = Variable(input.cuda())
                feat_pool, feat_fc = net(input, input, modal=test_mode)
                query_feat_pool[ptr:ptr + batch_num, :] = feat_pool.detach().cpu().numpy()
                query_feat_fc[ptr:ptr + batch_num, :] = feat_fc.detach().cpu().numpy()
                ptr = ptr + batch_num
        print('Extracting Time:\t {:.3f}'.format(time.time() - start))
    return query_feat_pool, query_feat_fc

def multi_process() :

    end = time.time()
    if args.dataset == "regdb":
        for trial in range(10):
            test_trial = trial +1
            #model_path = checkpoint_path +  args.resume
            model_path = checkpoint_path + suffix + '_best.t'
            # model_path = checkpoint_path + 'regdb_awg_p4_n8_lr_0.1_seed_0_trial_{}_best.t'.format(test_trial)
            if os.path.isfile(model_path):
                print(f'==> loading {args.fusion} fuse checkpoint')
                checkpoint = torch.load(model_path)
                if args.fusion == "layer1":
                    net = Network_layer1(class_num=nclass).to(device)
                elif args.fusion == "layer3":
                    net = Network_layer3(class_num=nclass).to(device)
                elif args.fusion == "layer5":
                    net = Network_layer5(class_num=nclass).to(device)
                net.load_state_dict(checkpoint['net'])
            else :
                sys.exit("Saved model not loaded, care")

            #Prepare query and gallery
            if args.reid == "BtoB" :
                query_img, query_img_t, query_label, gall_img, gall_img_t, gall_label = process_test_regdb(data_path, trial=test_trial, modal=args.reid, split=args.split)
                gallset = TestData_both(gall_img, gall_img_t, gall_label, transform=transform_test, img_size=(img_w, img_h))
                gall_loader = torch.utils.data.DataLoader(gallset, batch_size=int(test_batch_size), shuffle=False,
                                                          num_workers=workers, drop_last=True)
                nquery = len(query_label)
                ngall = len(gall_label)

                queryset = TestData_both(query_img, query_img_t, query_label, transform=transform_test, img_size=(img_w, img_h))
                query_loader = torch.utils.data.DataLoader(queryset, batch_size=int(test_batch_size), shuffle=False,
                                                           num_workers=4, drop_last=True)

                query_feat_pool, query_feat_fc = extract_query_feat(query_loader, nquery=nquery, net=net)

                sys.exit()
            else :
                query_img, query_label, gall_img, gall_label = process_test_regdb(data_path, trial=test_trial, modal=args.reid, split=args.split)
                gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(img_w, img_h))
                gall_loader = torch.utils.data.DataLoader(gallset, batch_size=test_batch_size, shuffle=False,
                                                          num_workers=workers)

                nquery = len(query_label)
                ngall = len(gall_label)

                queryset = TestData(query_img, query_label, transform=transform_test, img_size=(img_w, img_h))
                query_loader = torch.utils.data.DataLoader(queryset, batch_size=test_batch_size, shuffle=False,
                                                           num_workers=4)

            print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

            query_feat_pool, query_feat_fc = extract_query_feat(query_loader, nquery = nquery, net = net)
            gall_feat_pool,  gall_feat_fc = extract_gall_feat(gall_loader, ngall = ngall, net = net)


            # pool5 feature
            distmat_pool = np.matmul(query_feat_pool, np.transpose(gall_feat_pool))
            cmc_pool, mAP_pool, mINP_pool = eval_regdb(-distmat_pool,query_label , gall_label)

            # fc feature
            distmat = np.matmul( query_feat_fc, np.transpose(gall_feat_fc))
            cmc, mAP, mINP = eval_regdb(-distmat,query_label ,gall_label)


            if trial == 0:
                all_cmc = cmc
                all_mAP = mAP
                all_mINP = mINP
                all_cmc_pool = cmc_pool
                all_mAP_pool = mAP_pool
                all_mINP_pool = mINP_pool
            else:
                all_cmc = all_cmc + cmc
                all_mAP = all_mAP + mAP
                all_mINP = all_mINP + mINP
                all_cmc_pool = all_cmc_pool + cmc_pool
                all_mAP_pool = all_mAP_pool + mAP_pool
                all_mINP_pool = all_mINP_pool + mINP_pool

            print('Test Trial: {}'.format(trial))
            print(
                'FC:     Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                    cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
            print(
                'POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                    cmc_pool[0], cmc_pool[4], cmc_pool[9], cmc_pool[19], mAP_pool, mINP_pool))

    if args.dataset == 'sysu':

        print('==> Resuming from checkpoint..')
        model_path = checkpoint_path + suffix + '_best.t'
        # model_path = checkpoint_path + 'regdb_awg_p4_n8_lr_0.1_seed_0_trial_{}_best.t'.format(test_trial)
        if os.path.isfile(model_path):
            print('==> loading checkpoint')
            checkpoint = torch.load(model_path)
            if args.fusion == "layer1":
                net = Network_layer1(class_num=nclass).to(device)
            elif args.fusion == "layer3":
                net = Network_layer3(class_num=nclass).to(device)
            elif args.fusion == "layer5":
                net = Network_layer5(class_num=nclass).to(device)
            net.load_state_dict(checkpoint['net'])
        else :
            sys.exit("Saved model not loaded, care")

        # testing set
        if args.reid == "VtoT" or args.reid== "TtoV":
            query_img, query_label, query_cam = process_query_sysu(data_path, "test", mode="all", trial=0, reid=args.reid)
            gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, "test", mode="all", trial=0, reid=args.reid)

            queryset = TestData(query_img, query_label, transform=transform_test, img_size=(img_w, img_h))
            query_loader = data.DataLoader(queryset, batch_size=test_batch_size, shuffle=False, num_workers=4)
            nquery = len(query_label)
            query_feat_pool, query_feat_fc = extract_query_feat(query_loader, nquery=nquery, net=net)
        elif args.reid =="VtoV" or args.reid == "TtoT" :
            query_img, query_label, query_cam, gall_img, gall_label, gall_cam =\
                process_test_single_sysu(data_path, "test", trial=0, mode='all', relabel=False, reid=args.reid)
            nquery = len(query_label)


        ngall = len(gall_label)
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # images")
        print("  ------------------------------")
        print("  query    | {:5d} | {:8d}".format(len(np.unique(query_label)), nquery))
        print("  gallery  | {:5d} | {:8d}".format(len(np.unique(gall_label)), ngall))
        print("  ------------------------------")
        print('Data Loading Time:\t {:.3f}'.format(time.time() - end))



        for trial in range(10):

            if args.reid == "VtoT" or args.reid == "TtoV":
                gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, "test", mode="all",  trial=trial, reid=args.reid)

                trial_gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(img_w, img_h))
                trial_gall_loader = data.DataLoader(trial_gallset, batch_size=test_batch_size, shuffle=False, num_workers=4)
                gall_feat_pool, gall_feat_fc = extract_gall_feat(trial_gall_loader,ngall = ngall, net = net)
            elif args.reid == "VtoV" or args.reid =="TtoT":
                query_img, query_label, query_cam, gall_img, gall_label, gall_cam = \
                    process_test_single_sysu(data_path, "test", trial=trial, mode='all', relabel=False, reid=args.reid)

                queryset = TestData(query_img, query_label, transform=transform_test, img_size=(img_w, img_h))
                query_loader = data.DataLoader(queryset, batch_size=test_batch_size, shuffle=False, num_workers=4)
                query_feat_pool, query_feat_fc = extract_query_feat(query_loader, nquery=nquery, net=net)

                trial_gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(img_w, img_h))
                trial_gall_loader = data.DataLoader(trial_gallset, batch_size=test_batch_size, shuffle=False, num_workers=4)
                gall_feat_pool, gall_feat_fc = extract_gall_feat(trial_gall_loader,ngall = ngall, net = net)

            # pool5 feature
            distmat_pool = np.matmul(query_feat_pool, np.transpose(gall_feat_pool))
            cmc_pool, mAP_pool, mINP_pool = eval_sysu(-distmat_pool, query_label, gall_label, query_cam, gall_cam)

            # fc feature
            distmat = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))
            cmc, mAP, mINP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)

            if trial == 0:
                all_cmc = cmc
                all_mAP = mAP
                all_mINP = mINP
                all_cmc_pool = cmc_pool
                all_mAP_pool = mAP_pool
                all_mINP_pool = mINP_pool
            else:
                all_cmc = all_cmc + cmc
                all_mAP = all_mAP + mAP
                all_mINP = all_mINP + mINP
                all_cmc_pool = all_cmc_pool + cmc_pool
                all_mAP_pool = all_mAP_pool + mAP_pool
                all_mINP_pool = all_mINP_pool + mINP_pool

            print('Test Trial: {}'.format(trial))
            print(
                'FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                    cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
            print(
                'POOL: Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                    cmc_pool[0], cmc_pool[4], cmc_pool[9], cmc_pool[19], mAP_pool, mINP_pool))

    cmc = all_cmc / 10
    mAP = all_mAP / 10
    mINP = all_mINP / 10

    cmc_pool = all_cmc_pool / 10
    mAP_pool = all_mAP_pool / 10
    mINP_pool = all_mINP_pool / 10
    print('All Average:')
    print('FC:     Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
    print('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
    cmc_pool[0], cmc_pool[4], cmc_pool[9], cmc_pool[19], mAP_pool, mINP_pool))

    for k in range(len(cmc)):
        writer.add_scalar('cmc curve', cmc[k]*100, k + 1)

if __name__ == '__main__':
    freeze_support()
    multi_process()