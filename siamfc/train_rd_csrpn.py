import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
import numpy as np
import pandas as pd
import os
import cv2
import pickle
import lmdb
import torch.nn as nn
import time

from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
from collections import OrderedDict

from .config import config
from .rd_csrpn import DenseSiamese
from .dataset_rd_csrpn import ImagnetVIDDataset_ as ImagnetVIDDataset 
from .custom_transforms import Normalize, ToTensor, RandomStretch, \
    RandomCrop, CenterCrop, RandomBlur, ColorAug
from .loss import rpn_smoothL1
from .loss import rpn_cross_entropy_balance
from .visual import visual
from .utils import get_topk_box, add_box_img, compute_iou, box_transform_inv, adjust_learning_rate

from IPython import embed

torch.manual_seed(config.seed)


def train(data_dir, model_path=None, vis_port=None, init=None):
    # loading meta data
    meta_data_path = os.path.join(data_dir, "meta_data.pkl")
    meta_data = pickle.load(open(meta_data_path, 'rb'))
    all_videos = [x[0] for x in meta_data]

    # split train/valid dataset
    train_videos, valid_videos = train_test_split(all_videos,
                                                  test_size=1 - config.train_ratio, random_state=config.seed)

    # define transforms
    train_z_transforms = transforms.Compose([
        ToTensor()
    ])
    train_x_transforms = transforms.Compose([
        ToTensor()
    ])
    valid_z_transforms = transforms.Compose([
        ToTensor()
    ])
    valid_x_transforms = transforms.Compose([
        ToTensor()
    ])

    # open lmdb
    db = lmdb.open(data_dir + '.lmdb', readonly=True, map_size=int(50e9))

    # create dataset
    train_dataset = ImagnetVIDDataset(db, train_videos, data_dir, train_z_transforms, train_x_transforms)
    valid_dataset = ImagnetVIDDataset(db, valid_videos, data_dir, valid_z_transforms, valid_x_transforms,
                                      training=False)
    # create dataloader
    trainloader = DataLoader(train_dataset, batch_size=config.train_batch_size * torch.cuda.device_count(),
                             shuffle=True, pin_memory=True,
                             num_workers=config.train_num_workers * torch.cuda.device_count(), drop_last=True)
    validloader = DataLoader(valid_dataset, batch_size=config.valid_batch_size * torch.cuda.device_count(),
                             shuffle=False, pin_memory=True,
                             num_workers=config.valid_num_workers * torch.cuda.device_count(), drop_last=True)

    # create summary writer
    if not os.path.exists(config.log_dir):
        os.mkdir(config.log_dir)
    summary_writer = SummaryWriter(config.log_dir)
    if vis_port:
        vis = visual(port=vis_port)

    # start training
    model = DenseSiamese()
    model.init_weights()
    model = model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr,
                                momentum=config.momentum, weight_decay=config.weight_decay)
    start_epoch = 1
    if model_path and init:
        print("init checkpoint %s" % model_path)
        checkpoint = torch.load(model_path)
        if 'model' in checkpoint.keys():
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        del checkpoint
        torch.cuda.empty_cache()
        print("inited checkpoint")
    if model_path and not init:
        print("loading checkpoint %s" % model_path)
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if 'pooling_mode' in checkpoint.keys():
            C.pooling_mode = checkpoint['pooling_mode']
        del checkpoint
        torch.cuda.empty_cache()
        print("loaded checkpoint")
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    if config.warm_epoch and not model_path:
        adjust_learning_rate(optimizer, config.warm_scale)
        for epoch in range(config.warm_epoch):
            train_loss = []
            model.train()
            loss_temp_cls = 0
            loss_temp_reg = 0
            for i, data in enumerate(tqdm(trainloader)):
                exemplar_imgs, instance_imgs, regression_target, conf_target, target_gt = data
                # conf_target (8,1125) (8,225x5)
                regression_target, conf_target = regression_target.cuda(), conf_target.cuda()

                # stage1
                pred_score_stage1, pred_regression_stage1, pred_score_stage2, pred_regression_stage2 = model(exemplar_imgs.cuda(), instance_imgs.cuda())

                pred_conf_stage1 = pred_score_stage1.reshape(-1, 2,
                                               config.anchor_num * config.score_size * config.score_size).permute(0,
                                                                                                                  2,
                                                                                                                  1)
                pred_offset_stage1 = pred_regression_stage1.reshape(-1, 4,
                                                      config.anchor_num * config.score_size * config.score_size).permute(
                    0,
                    2,
                    1)
                cls_loss_stage1 = rpn_cross_entropy_balance(pred_conf_stage1, conf_target, config.num_pos, config.num_neg)
                reg_loss_stage1 = rpn_smoothL1(pred_offset_stage1, regression_target, conf_target)
                loss_stage1 = cls_loss_stage1 + config.lamb * reg_loss_stage1
                # stage2
                # TODO anchor filter
                new_anchors = box_transform_inv(train_dataset.anchors,pred_offset_stage1)
                # regression_target_stage2, conf_target_stage2 = regression_target.cuda(), conf_target.cuda()
                regression_target_stage2, conf_target_stage2 = train_dataset.compute_target(new_anchors,target_gt)
                anchor_num_stage2 = config.anchor_num
                pred_conf_stage2 = pred_score_stage2.reshape(-1, 2,
                                               anchor_num_stage2 * config.score_size * config.score_size).permute(0,
                                                                                                                  2,
                                                                                                                  1)
                pred_offset_stage2 = pred_regression_stage2.reshape(-1, 4,
                                                      anchor_num_stage2 * config.score_size * config.score_size).permute(
                    0,
                    2,
                    1)
                cls_loss_stage2 = rpn_cross_entropy_balance(pred_conf_stage2, conf_target_stage2, config.num_pos, config.num_neg)
                reg_loss_stage2 = rpn_smoothL1(pred_offset_stage2, regression_target_stage2, conf_target_stage2)
                loss_stage2 = cls_loss_stage2 + config.lamb * reg_loss_stage2

                loss = loss_stage1 + loss_stage2
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                step = (epoch - 1) * len(trainloader) + i
                summary_writer.add_scalar('train/loss', loss.data, step)
                train_loss.append(loss.detach().cpu())

                loss_temp_cls_stage1 += cls_loss_stage1.detach().cpu().numpy()
                loss_temp_reg_stage1 += reg_loss_stage1.detach().cpu().numpy()
                loss_temp_cls_stage2 += cls_loss_stage2.detach().cpu().numpy()
                loss_temp_reg_stage2 += reg_loss_stage2.detach().cpu().numpy()

                # if vis_port:
                #     vis.plot_error({'rpn_cls_loss': cls_loss.detach().cpu().numpy().ravel()[0],
                #                     'rpn_regress_loss': reg_loss.detach().cpu().numpy().ravel()[0]}, win=0)
                if (i + 1) % config.show_interval == 0:
                    tqdm.write("[warm epoch %2d][iter %4d] cls_loss_stage1: %.4f, reg_loss_stage1: %.4f, cls_loss_stage2: %.4f, reg_loss_stage2: %.4f lr: %.2e"
                               % (epoch, i, loss_temp_cls_stage1 / config.show_interval, loss_temp_reg_stage1 / config.show_interval,
                                  loss_temp_cls_stage2 / config.show_interval, loss_temp_reg_stage2 / config.show_interval, optimizer.param_groups[0]['lr']))
                    loss_temp_cls_stage1 = 0
                    loss_temp_reg_stage1 = 0
                    loss_temp_cls_stage2 = 0
                    loss_temp_reg_stage2 = 0
                    if vis_port:
                        anchors_show = train_dataset.anchors
                        exem_img = exemplar_imgs[0].cpu().numpy().transpose(1, 2, 0)
                        inst_img = instance_imgs[0].cpu().numpy().transpose(1, 2, 0)

                        # show detected box with max score
                        topk = 3
                        cls_pred = F.softmax(pred_conf, dim=2)[0, :, 1]
                        vis.plot_img(exem_img.transpose(2, 0, 1), win=1, name='exemple')
                        topk_box = get_topk_box(cls_pred, pred_offset[0], anchors_show, topk=topk)
                        img_box = add_box_img(inst_img, topk_box)
                        cls_pred = conf_target[0]
                        gt_box = get_topk_box(cls_pred, regression_target[0], anchors_show)
                        img_box = add_box_img(img_box, gt_box, color=(255, 0, 0))
                        vis.plot_img(img_box.transpose(2, 0, 1), win=2, name='box_max_score')

                        # show gt_box
                        cls_pred = conf_target[0]
                        gt_box = get_topk_box(cls_pred, regression_target[0], anchors_show)
                        img_box = add_box_img(inst_img, gt_box, color=(255, 0, 0))
                        vis.plot_img(img_box.transpose(2, 0, 1), win=3, name='box_gt')

                        # show anchor with max score
                        cls_pred = F.softmax(pred_conf, dim=2)[0, :, 1]
                        scores, index = torch.topk(cls_pred, k=topk)
                        img_box = add_box_img(inst_img, anchors_show[index.cpu()])
                        cls_pred = conf_target[0]
                        gt_box = get_topk_box(cls_pred, regression_target[0], anchors_show)
                        img_box = add_box_img(img_box, gt_box, color=(255, 0, 0))
                        vis.plot_img(img_box.transpose(2, 0, 1), win=4, name='anchor_max_score')

                        # show anchor and detected box with max iou
                        cls_pred = conf_target[0]
                        gt_box = get_topk_box(cls_pred, regression_target[0], anchors_show)[0]
                        iou = compute_iou(anchors_show, gt_box).flatten()
                        index = np.argsort(iou)[-topk:]
                        img_box = add_box_img(inst_img, anchors_show[index])
                        cls_pred = conf_target[0]
                        gt_box = get_topk_box(cls_pred, regression_target[0], anchors_show)
                        img_box = add_box_img(img_box, gt_box, color=(255, 0, 0))
                        vis.plot_img(img_box.transpose(2, 0, 1), win=5, name='anchor_max_iou')
                        # detected box
                        regress_offset = pred_offset[0].cpu().detach().numpy()
                        topk_offset = regress_offset[index, :]
                        anchors = anchors_show[index, :]
                        pred_box = box_transform_inv(anchors, topk_offset)
                        img_box = add_box_img(inst_img, pred_box)
                        cls_pred = conf_target[0]
                        gt_box = get_topk_box(cls_pred, regression_target[0], anchors_show)
                        img_box = add_box_img(img_box, gt_box, color=(255, 0, 0))
                        vis.plot_img(img_box.transpose(2, 0, 1), win=6, name='box_max_iou')
        adjust_learning_rate(optimizer, 1 / config.warm_scale)

        save_name = "./models/siamrpn_warm.pth"
        new_state_dict = model.state_dict()
        if torch.cuda.device_count() > 1:
            new_state_dict = OrderedDict()
            for k, v in model.state_dict().items():
                namekey = k[7:]  # remove `module.`
                new_state_dict[namekey] = v
        torch.save({
            'epoch': 0,
            'model': new_state_dict,
            'optimizer': optimizer.state_dict(),
        }, save_name)
        print('save model: {}'.format(save_name))

    for epoch in range(start_epoch, config.epoch + 1):
        train_loss = []
        model.train()
        loss_temp_cls = 0
        loss_temp_reg = 0
        for i, data in enumerate(tqdm(trainloader)):
            exemplar_imgs, instance_imgs, regression_target, conf_target, target_gt = data
            # conf_target (8,1125) (8,225x5)
            print('regression target: {}'.format(regression_target.shape))
            print('conf target: {}'.format(conf_target.shape))
            regression_target, conf_target = regression_target.cuda(), conf_target.cuda()

            # stage1
            pred_score_stage1, pred_regression_stage1, pred_score_stage2, pred_regression_stage2 = model(exemplar_imgs.cuda(), instance_imgs.cuda())

            pred_conf_stage1 = pred_score_stage1.reshape(-1, 2,
                                            config.anchor_num * config.score_size * config.score_size).permute(0,
                                                                                                                2,
                                                                                                                1)
            pred_offset_stage1 = pred_regression_stage1.reshape(-1, 4,
                                                    config.anchor_num * config.score_size * config.score_size).permute(
                                                                                                                    0,
                                                                                                                    2,
                                                                                                                    1)
            cls_loss_stage1 = rpn_cross_entropy_balance(pred_conf_stage1, conf_target, config.num_pos, config.num_neg)
            reg_loss_stage1 = rpn_smoothL1(pred_offset_stage1, regression_target, conf_target)
            loss_stage1 = cls_loss_stage1 + config.lamb * reg_loss_stage1
            # stage2
            # TODO anchor filter
            new_anchors = box_transform_inv(train_dataset.anchors,pred_offset_stage1[0].cpu().detach().numpy())
            print('new anchors shape:{}'.format(new_anchors.shape))
            print('target_gt shape:{}, {}'.format(target_gt.shape,target_gt))
            # regression_target_stage2, conf_target_stage2 = regression_target.cuda(), conf_target.cuda()
            # regression_target_stage2, conf_target_stage2 = train_dataset.compute_target(new_anchors,target_gt)
            regression_target_stage2, conf_target_stage2 = np.array([]),np.array([])
            for box_index in range(config.train_batch_size):
                print('{}th box {}'.format(box_index,target_gt[box_index]))
                rt_tmp,ct_tmp = train_dataset.compute_target(new_anchors,target_gt[box_index].cpu().detach().numpy())
                regression_target_stage2 = np.append(regression_target_stage2, rt_tmp)
                conf_target_stage2 = np.append(conf_target_stage2, ct_tmp)
            print('stage2 regression target: {}'.format(regression_target_stage2.shape))
            print('stage2 conf target: {}'.format(conf_target_stage2.shape))
            regression_target_stage2, conf_target_stage2 = torch.tensor(regression_target_stage2).cuda(), torch.tensor(conf_target_stage2).cuda()
            anchor_num_stage2 = config.anchor_num
            pred_conf_stage2 = pred_score_stage2.reshape(-1, 2,
                                            anchor_num_stage2 * config.score_size * config.score_size).permute(0,
                                                                                                                2,
                                                                                                                1)
            pred_offset_stage2 = pred_regression_stage2.reshape(-1, 4,
                                                    anchor_num_stage2 * config.score_size * config.score_size).permute(
                                                                                                                    0,
                                                                                                                    2,
                                                                                                                    1)
            cls_loss_stage2 = rpn_cross_entropy_balance(pred_conf_stage2, conf_target_stage2, config.num_pos, config.num_neg)
            reg_loss_stage2 = rpn_smoothL1(pred_offset_stage2, regression_target_stage2, conf_target_stage2)
            loss_stage2 = cls_loss_stage2 + config.lamb * reg_loss_stage2

            loss = loss_stage1 + loss_stage2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step = (epoch - 1) * len(trainloader) + i
            summary_writer.add_scalar('train/loss', loss.data, step)
            train_loss.append(loss.detach().cpu())

            loss_temp_cls_stage1 += cls_loss_stage1.detach().cpu().numpy()
            loss_temp_reg_stage1 += reg_loss_stage1.detach().cpu().numpy()
            loss_temp_cls_stage2 += cls_loss_stage2.detach().cpu().numpy()
            loss_temp_reg_stage2 += reg_loss_stage2.detach().cpu().numpy()

            # if vis_port:
            #     vis.plot_error({'rpn_cls_loss': cls_loss.detach().cpu().numpy().ravel()[0],
            #                     'rpn_regress_loss': reg_loss.detach().cpu().numpy().ravel()[0]}, win=0)
            if (i + 1) % config.show_interval == 0:
                tqdm.write("[epoch %2d][iter %4d] cls_loss: %.4f, reg_loss: %.4f lr: %.2e"
                           % (epoch, i, loss_temp_cls / config.show_interval, loss_temp_reg / config.show_interval,
                              optimizer.param_groups[0]['lr']))
                loss_temp_cls = 0
                loss_temp_reg = 0
                # if vis_port:
                #     anchors_show = train_dataset.anchors
                #     exem_img = exemplar_imgs[0].cpu().numpy().transpose(1, 2, 0)
                #     inst_img = instance_imgs[0].cpu().numpy().transpose(1, 2, 0)

                #     # show detected box with max score
                #     topk = 3
                #     cls_pred = F.softmax(pred_conf, dim=2)[0, :, 1]
                #     vis.plot_img(exem_img.transpose(2, 0, 1), win=1, name='exemple')
                #     topk_box = get_topk_box(cls_pred, pred_offset[0], anchors_show, topk=topk)
                #     img_box = add_box_img(inst_img, topk_box)
                #     cls_pred = conf_target[0]
                #     gt_box = get_topk_box(cls_pred, regression_target[0], anchors_show)
                #     img_box = add_box_img(img_box, gt_box, color=(255, 0, 0))
                #     vis.plot_img(img_box.transpose(2, 0, 1), win=2, name='box_max_score')

                #     # show gt_box
                #     cls_pred = conf_target[0]
                #     gt_box = get_topk_box(cls_pred, regression_target[0], anchors_show)
                #     img_box = add_box_img(inst_img, gt_box, color=(255, 0, 0))
                #     vis.plot_img(img_box.transpose(2, 0, 1), win=3, name='box_gt')

                #     # show anchor with max score
                #     cls_pred = F.softmax(pred_conf, dim=2)[0, :, 1]
                #     scores, index = torch.topk(cls_pred, k=topk)
                #     img_box = add_box_img(inst_img, anchors_show[index.cpu()])
                #     cls_pred = conf_target[0]
                #     gt_box = get_topk_box(cls_pred, regression_target[0], anchors_show)
                #     img_box = add_box_img(img_box, gt_box, color=(255, 0, 0))
                #     vis.plot_img(img_box.transpose(2, 0, 1), win=4, name='anchor_max_score')

                #     # show anchor and detected box with max iou
                #     cls_pred = conf_target[0]
                #     gt_box = get_topk_box(cls_pred, regression_target[0], anchors_show)[0]
                #     iou = compute_iou(anchors_show, gt_box).flatten()
                #     index = np.argsort(iou)[-topk:]
                #     img_box = add_box_img(inst_img, anchors_show[index])
                #     cls_pred = conf_target[0]
                #     gt_box = get_topk_box(cls_pred, regression_target[0], anchors_show)
                #     img_box = add_box_img(img_box, gt_box, color=(255, 0, 0))
                #     vis.plot_img(img_box.transpose(2, 0, 1), win=5, name='anchor_max_iou')
                #     # detected box
                #     regress_offset = pred_offset[0].cpu().detach().numpy()
                #     topk_offset = regress_offset[index, :]
                #     anchors = anchors_show[index, :]
                #     pred_box = box_transform_inv(anchors, topk_offset)
                #     img_box = add_box_img(inst_img, pred_box)
                #     cls_pred = conf_target[0]
                #     gt_box = get_topk_box(cls_pred, regression_target[0], anchors_show)
                #     img_box = add_box_img(img_box, gt_box, color=(255, 0, 0))
                #     vis.plot_img(img_box.transpose(2, 0, 1), win=6, name='box_max_iou')

        train_loss = np.mean(train_loss)

        valid_loss = []
        model.eval()
        for i, data in enumerate(tqdm(validloader)):
            exemplar_imgs, instance_imgs, regression_target, conf_target, target_gt = data
            # conf_target (8,1125) (8,225x5)
            regression_target, conf_target = regression_target.cuda(), conf_target.cuda()

            # stage1
            pred_score_stage1, pred_regression_stage1, pred_score_stage2, pred_regression_stage2 = model(exemplar_imgs.cuda(), instance_imgs.cuda())

            pred_conf_stage1 = pred_score_stage1.reshape(-1, 2,
                                            config.anchor_num * config.score_size * config.score_size).permute(0,
                                                                                                                2,
                                                                                                                1)
            pred_offset_stage1 = pred_regression_stage1.reshape(-1, 4,
                                                    config.anchor_num * config.score_size * config.score_size).permute(
                                                                                                                    0,
                                                                                                                    2,
                                                                                                                    1)
            cls_loss_stage1 = rpn_cross_entropy_balance(pred_conf_stage1, conf_target, config.num_pos, config.num_neg)
            reg_loss_stage1 = rpn_smoothL1(pred_offset_stage1, regression_target, conf_target)
            loss_stage1 = cls_loss_stage1 + config.lamb * reg_loss_stage1
            # stage2
            # TODO anchor filter
            new_anchors = box_transform_inv(train_dataset.anchors,pred_offset_stage1)
            # regression_target_stage2, conf_target_stage2 = regression_target.cuda(), conf_target.cuda()
            regression_target_stage2, conf_target_stage2 = train_dataset.compute_target(new_anchors,target_gt)
            anchor_num_stage2 = config.anchor_num
            pred_conf_stage2 = pred_score_stage2.reshape(-1, 2,
                                            anchor_num_stage2 * config.score_size * config.score_size).permute(0,
                                                                                                                2,
                                                                                                                1)
            pred_offset_stage2 = pred_regression_stage2.reshape(-1, 4,
                                                    anchor_num_stage2 * config.score_size * config.score_size).permute(
                                                                                                                    0,
                                                                                                                    2,
                                                                                                                    1)
            cls_loss_stage2 = rpn_cross_entropy_balance(pred_conf_stage2, conf_target_stage2, config.num_pos, config.num_neg)
            reg_loss_stage2 = rpn_smoothL1(pred_offset_stage2, regression_target_stage2, conf_target_stage2)
            loss_stage2 = cls_loss_stage2 + config.lamb * reg_loss_stage2

            loss = loss_stage1 + loss_stage2

            valid_loss.append(loss.detach().cpu())
        valid_loss = np.mean(valid_loss)
        print("EPOCH %d valid_loss: %.4f, train_loss: %.4f" % (epoch, valid_loss, train_loss))
        summary_writer.add_scalar('valid/loss',
                                  valid_loss, (epoch + 1) * len(trainloader))
        adjust_learning_rate(optimizer,
                             config.gamma)  # adjust before save, and it will be epoch+1's lr when next load
        if epoch % config.save_interval == 0:
            save_name = "./models/siamrpn_{}.pth".format(epoch)
            new_state_dict = model.state_dict()
            if torch.cuda.device_count() > 1:
                new_state_dict = OrderedDict()
                for k, v in model.state_dict().items():
                    namekey = k[7:]  # remove `module.`
                    new_state_dict[namekey] = v
            torch.save({
                'epoch': epoch,
                'model': new_state_dict,
                'optimizer': optimizer.state_dict(),
            }, save_name)
            print('save model: {}'.format(save_name))
