#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-07:31 16:57:15
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-02-22 19:28:24
# @Email:  cshzxie@gmail.com

import argparse
import cv2
import logging
import matplotlib
import os
import sys
import torch    # lgtm [py/unused-import]
import numpy as np
import random

# Fix no $DISPLAY environment variable
matplotlib.use('Agg')
# Fix deadlock in DataLoader
cv2.setNumThreads(0)

# from pprint import pprint

from configs.grnet.config import cfg
from grnet_point_cloud_completion.core.train import train_net
from grnet_point_cloud_completion.core.test import test_net
from grnet_point_cloud_completion.core.inference import inference_net


def get_args_from_command_line():
    parser = argparse.ArgumentParser(description='The argument parser of R2Net runner')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device to use', default=cfg.CONST.DEVICE, type=str)
    parser.add_argument('--test', dest='test', help='Test neural networks', action='store_true')
    parser.add_argument('--inference', dest='inference', help='Inference for benchmark', action='store_true')
    parser.add_argument('--weights', dest='weights', help='Initialize network from the weights file', default=None)
    parser.add_argument('--savepred', dest='save_pred', help='Save predicted point clouds', action='store_true')
    args = parser.parse_args()
    return args


def main():
    # Get args from command line
    args = get_args_from_command_line()

    if args.gpu_id is not None:
        cfg.CONST.DEVICE = args.gpu_id
    if args.weights is not None:
        cfg.CONST.WEIGHTS = args.weights
    if args.save_pred is not None:
        cfg.TEST.SAVE_PRED = args.save_pred

    # Print config
    print('Use config:')
    # pprint(cfg)
    print(cfg)

    # Set GPU to use
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CONST.DEVICE

    # Start train/test process
    if not args.test and not args.inference:
        train_net(cfg)
    else:
        if 'WEIGHTS' not in cfg.CONST or not os.path.exists(cfg.CONST.WEIGHTS):
            logging.error('Please specify the file path of checkpoint.')
            sys.exit(2)

        if args.test:
            test_net(cfg)
        else:
            inference_net(cfg)


if __name__ == '__main__':
    # Check python version
    if sys.version_info < (3, 0):
        raise Exception("Please follow the installation instruction on https://github.com/hzxie/GRNet")

    logging.basicConfig(format='[%(levelname)s] %(asctime)s %(message)s', level=logging.DEBUG)
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    main()
