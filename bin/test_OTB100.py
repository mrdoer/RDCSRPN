import argparse
import os
import glob
import numpy as np
import re
import json
import matplotlib
import setproctitle
import functools
import multiprocessing as mp
import matplotlib.pyplot as plt

from run_SiamRPN import run_SiamRPN
from siamfc import config
from IPython import embed
from multiprocessing import Pool


def embeded_numbers(s):
    re_digits = re.compile(r'(\d+)')
    pieces = re_digits.split(s)
    return int(pieces[1])


def embeded_numbers_results(s):
    re_digits = re.compile(r'(\d+)')
    pieces = re_digits.split(s)
    return int(pieces[-2])


def cal_iou(box1, box2):
    r"""

    :param box1: x1,y1,w,h
    :param box2: x1,y1,w,h
    :return: iou
    """
    x11 = box1[0]
    y11 = box1[1]
    x21 = box1[0] + box1[2] - 1
    y21 = box1[1] + box1[3] - 1
    area_1 = (x21 - x11 + 1) * (y21 - y11 + 1)

    x12 = box2[0]
    y12 = box2[1]
    x22 = box2[0] + box2[2] - 1
    y22 = box2[1] + box2[3] - 1
    area_2 = (x22 - x12 + 1) * (y22 - y12 + 1)

    x_left = max(x11, x12)
    x_right = min(x21, x22)
    y_top = max(y11, y12)
    y_down = min(y21, y22)

    inter_area = max(x_right - x_left + 1, 0) * max(y_down - y_top + 1, 0)
    iou = inter_area / (area_1 + area_2 - inter_area)
    return iou


def cal_success(iou):
    success_all = []
    overlap_thresholds = np.arange(0, 1.05, 0.05)
    for overlap_threshold in overlap_thresholds:
        success = sum(np.array(iou) > overlap_threshold) / len(iou)
        success_all.append(success)
    return np.array(success_all)

if __name__ == '__main__':
    # save_name = 'D:/Huizhi/model/dsrpn_ex/DSRPNex_chhgcp1e-4_batch8_20_win40_pk20'
    save_name = '/home/ubuntu/Code/results/RDCSRPN_25_win{}_pk{}'.format\
                (config.window_influence*100,config.penalty_k*100)
    print(save_name)
    # save_name = 'D:/Huizhi/model/dsrpn_ex/DSRPNex_chh1e-4_batch8_170_62_win40_pk22'
    # seq = {'David':(300,770),'Football1':(1, 74),'Freeman3':(1, 460),'Freeman4':(1, 283),\
            # 'Diving':(1, 215),'blurcar1':(247,988),'blurcar3':(3,359),'blurcar4':(18,397)}
    if not os.path.isdir(save_name):
        os.mkdir(save_name)
    # ------------ prepare data  -----------
    data_path = '/home/ubuntu/Data/OTBData/'
    direct_file = '/home/ubuntu/Data/OTBData/tb13.txt'
    with open(direct_file, 'r') as f:
        direct_lines = f.readlines()
    video_names = np.sort([x.split('\n')[0] for x in direct_lines])
    video_paths = [data_path + x for x in video_names]

    # ------------ prepare models  -----------
    model_path = '/home/ubuntu/Code/RDCSRPN/models/siamrpn_25.pth'
    # ------------ starting validation  -----------
    results = {}
    for video_path in video_paths:

        video_name = video_path.split('/')[-1]
        # video_name = 'Human8'
        video_path = data_path+video_name
        print(video_name)
        groundtruth_path = video_path + '/groundtruth_rect.txt'
        if '-1' in video_name:
            gt_filename = '/groundtruth_rect{}.txt'.format('.1')
            video_path = video_path[:-2]
            groundtruth_path = video_path + gt_filename
        elif '-2' in video_name:
            gt_filename = '/groundtruth_rect{}.txt'.format('.2')
            video_path = video_path[:-2]
            groundtruth_path = video_path + gt_filename

        assert os.path.isfile(groundtruth_path), groundtruth_path + ' doesn\'t exist'
        with open(groundtruth_path, 'r') as f:
            boxes = f.readlines()
        if ',' in boxes[0]:
            boxes = [list(map(int, box.split(','))) for box in boxes]
        else:
            boxes = [list(map(int, box.split())) for box in boxes]
        boxes = [np.array(box) - [1, 1, 0, 0] for box in boxes]
        result = run_SiamRPN(video_path, model_path, boxes[0])
        result = result['res']
        res_txt = save_name+'/'+video_name+'.txt'
        np.savetxt(res_txt,result)
        # break