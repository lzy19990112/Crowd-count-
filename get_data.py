import glob
import os
import random

def get_file_id(filepath):
    return os.path.splitext(os.path.basename(filepath))[0]
    
def get_data_list(data_root, mode='train'):

    if mode == 'train':
        imagepath = os.path.join(data_root, 'train_data', 'images')
        gtpath = os.path.join(data_root, 'train_data', 'ground_truth')

    elif mode == 'valid':
        imagepath = os.path.join(data_root, 'valid_data', 'images')
        gtpath = os.path.join(data_root, 'valid_data', 'ground_truth')

    else:
        imagepath = os.path.join(data_root, 'test_data', 'images')
        gtpath = os.path.join(data_root, 'test_data', 'ground_truth')
    
    
    image_list = [file for file in glob.glob(os.path.join(imagepath,'*.jpg'))]
    gt_list = []

    for filepath in image_list:
        file_id = get_file_id(filepath)
        gt_file_path = os.path.join(gtpath, 'GT_'+ file_id + '.mat')
        gt_list.append(gt_file_path)
    xy = list(zip(image_list, gt_list))
    random.shuffle(xy)
    s_gt_list=[]
    s_image_list=[]
    for t in xy:
        s_image_list.append(t[0])
        s_gt_list.append(t[1])
    #s_image_list, s_gt_list = zip(*zip(xy))
   
    return s_image_list, s_gt_list