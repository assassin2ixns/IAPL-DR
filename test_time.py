import argparse

import time
from contextlib import nullcontext
import torch.distributed as dist
from copy import deepcopy
import pdb
from PIL import Image
import numpy as np
from torch.nn import functional as F
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from models import build_model
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler
from utils.dataset import Dataset_Creator, Dataset_Creator_GenImage, Dataset_Creator_Chameleon, Dataset_Creator_Chameleon_SD
import utils.misc as utils
from augmix import AugMixAugmenter
from sklearn.metrics import average_precision_score, accuracy_score
import random
import matplotlib.pyplot as plt
import os
import torch.nn as nn
from scipy.ndimage import filters


def unwrap_model(model):
    return model.module if hasattr(model, 'module') else model


def extract_logits(outputs):
    if isinstance(outputs, dict):
        return outputs['logits']
    if isinstance(outputs, (list, tuple)):
        return outputs[0]
    return outputs


def _strip_module_prefix(state_dict):
    if any(key.startswith('module.') for key in state_dict.keys()):
        return {
            key.replace('module.', '', 1) if key.startswith('module.') else key: value
            for key, value in state_dict.items()
        }
    return state_dict


@torch.no_grad()
def gather_together(data):
    world_size = utils.get_world_size()
    if world_size < 2:
        return data
    dist.barrier()
    gather_data = [None for _ in range(world_size)]
    dist.all_gather_object(gather_data, data)
    return gather_data

def testtime_main(args):
    torch.backends.cudnn.benchmark = True

    # infer dataset construction
    assert args.dataset in ['UniversalFakeDetect', 'GenImage', 'Chameleon', 'Chameleon_SD']
    if args.dataset == 'UniversalFakeDetect':
        dataset_creator = Dataset_Creator(dataset_path=args.dataset_path, batch_size=args.evalbatchsize, num_workers=args.num_workers, img_resolution=args.img_resolution, crop_resolution=args.crop_resolution)
    elif args.dataset == 'GenImage':
        dataset_creator = Dataset_Creator_GenImage(dataset_path=args.dataset_path, batch_size=args.evalbatchsize, num_workers=args.num_workers, img_resolution=args.img_resolution, crop_resolution=args.crop_resolution)               
    elif args.dataset == 'Chameleon':
        dataset_creator = Dataset_Creator_Chameleon(dataset_path=args.dataset_path, batch_size=args.evalbatchsize, num_workers=args.num_workers, img_resolution=args.img_resolution, crop_resolution=args.crop_resolution)
    elif args.dataset == 'Chameleon_SD':
        dataset_creator = Dataset_Creator_Chameleon_SD(dataset_path=args.dataset_path, batch_size=args.evalbatchsize, num_workers=args.num_workers, img_resolution=args.img_resolution, crop_resolution=args.crop_resolution)

    dataset_vals, selected_subsets = dataset_creator.build_dataset("tta", selected_subsets=args.test_selected_subsets)
    data_loader_vals = {}
    for dataset_val, selected_subset in zip(dataset_vals, selected_subsets):
        if args.distributed:
            sampler_val = DistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_val = SequentialSampler(dataset_val)
        data_loader_vals[selected_subset] = DataLoader(dataset_val, 1, sampler=sampler_val, drop_last=False, num_workers=args.num_workers)

    # build model
    device = torch.device(args.device)
    model = build_model(args)
    model = model.to(device)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
    checkpoint = torch.load(args.pretrained_model, map_location='cpu')
    state_dict = _strip_module_prefix(checkpoint['model'])
    m = unwrap_model(model)
    strict = getattr(args, 'method', 'iapl') != 'dream_cs'
    m.load_state_dict(state_dict, strict=strict)

    pretrained_ctx = state_dict.get('prompt_learner.ctx', None)

    m.freeze_tta()
    
    print([{"params": [n for n, p in m.named_parameters() if p.requires_grad]}])

    optimizer = torch.optim.AdamW([{"params": [p for n, p in m.named_parameters() if p.requires_grad]}], args.lr)
    optim_state = deepcopy(optimizer.state_dict())
    scaler = None

    # test for every sub-dataset
    test_dataset = []
    test_AP = []
    test_ACC = []
    test_real_ACC = []
    test_fake_ACC = []

    for data_name, data_loader in data_loader_vals.items():

        metric_logger = utils.MetricLogger(delimiter="  ")
        header = 'Test:'
        print_freq = args.print_freq

        y_true, y_pred = [], []

        for samples in metric_logger.log_every(data_loader, print_freq, header):

            sample_img, labels = samples[0], samples[1].to(device)
            images = [sample.to(device) for sample in sample_img]

            image = images[0] # original image
            images = torch.cat(images, dim=0) # multi-view images
            
            # for every new image reset it params and optimizer.
            with torch.no_grad():
                if pretrained_ctx is not None and m.prompt_learner.ctx is not None:
                    m.prompt_learner.ctx.copy_(pretrained_ctx.to(m.prompt_learner.ctx.device))
                
            optimizer.load_state_dict(optim_state)

            # tta 
            autocast_enabled = device.type == 'cuda'
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
                model.train()
                select_index = test_time_tuning(model, images, optimizer, scaler, args)

            # infer
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
                with torch.no_grad():
                    model.eval()
                    if args.ois:
                        selected_outputs = model(images[select_index])
                        selected_logits = extract_logits(selected_outputs).flatten()
                        if getattr(args, 'method', 'iapl') == 'dream_cs' and getattr(args, 'dream_tta_safe', True):
                            probs = selected_logits.sigmoid()
                            if args.dream_tta_agg == 'confidence':
                                conf_idx = torch.max(torch.abs(probs - 0.5), dim=0)[1]
                                pred = probs[conf_idx].view(1)
                            elif args.dream_tta_agg == 'trimmed_mean':
                                if probs.numel() >= 4:
                                    probs_sorted = torch.sort(probs)[0]
                                    pred = probs_sorted[1:-1].mean().view(1)
                                else:
                                    pred = probs.mean().view(1)
                            else:
                                pred = probs.mean().view(1)
                        else:
                            preds = selected_logits.sigmoid()
                            conf_idx = torch.max(torch.abs(preds - 0.5), dim=0)[1]
                            pred = preds[conf_idx].view(1)
                    else:
                        pred = extract_logits(model(image)).sigmoid()

            y_pred.extend(pred.flatten().tolist())
            y_true.extend(labels.flatten().tolist())

        world_size = utils.get_world_size()
        if world_size < 2:
            merge_y_true = y_true
        else:
            merge_y_true = []
            for data in gather_together(y_true):
                merge_y_true.extend(data)
        
        if world_size < 2:
            merge_y_pred = y_pred
        else:
            merge_y_pred = []
            for data in gather_together(y_pred):
                merge_y_pred.extend(data)

        y_true, y_pred = np.array(merge_y_true), np.array(merge_y_pred)
        r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > 0.5)
        f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > 0.5)
        acc = accuracy_score(y_true, y_pred > 0.5)
        ap = average_precision_score(y_true, y_pred)
        
        test_dataset.append(data_name)
        test_AP.append(ap)
        test_ACC.append(acc)
        test_real_ACC.append(r_acc)
        test_fake_ACC.append(f_acc)

        print("({}) acc: {:.2f}; ap: {:.2f}; racc: {:.2f}; facc: {:.2f};".format(data_name, acc*100, ap*100, r_acc*100, f_acc*100))

    output_strs = []
    for idx, [name, ap, acc, racc, facc] in enumerate(zip(test_dataset + ["mean"], test_AP + [np.mean(test_AP)], test_ACC + [np.mean(test_ACC)], test_real_ACC + [np.mean(test_real_ACC)], test_fake_ACC + [np.mean(test_fake_ACC)])):
        output_str = "({} {:10}) acc: {:.2f}; ap: {:.2f}; racc: {:.2f}; facc: {:.2f};".format(idx, name, acc*100, ap*100, racc*100, facc*100)
        output_strs.append(output_str)
        print(output_str)
    
    return "; ".join(output_strs), np.mean(test_AP), np.mean(test_ACC)



def test_time_tuning(model, inputs, optimizer, scaler, args):
    
    for j in range(args.tta_steps):

        sync_context = model.no_sync() if hasattr(model, 'no_sync') else nullcontext()
        with sync_context:  # 多卡时梯度不进行聚合，各卡独立进行参数更新
            
            outputs = model(inputs)
            logits = extract_logits(outputs).flatten()
            loss, index = binary_entropy(logits, args.selection_p, args.ois)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return index

def binary_entropy(logits, selection_p, ois):

    select_num = min(len(logits), max(1, int(len(logits) * selection_p)))

    if ois:

        with torch.no_grad():
            confidence = F.softmax(torch.abs(torch.sigmoid(logits) - 0.5) * 2, dim=0)
            select, index = torch.topk(confidence, select_num, dim=0)

        probs = torch.sigmoid(logits)
        avg_probs = probs[index].mean()
        loss = - (avg_probs * torch.log(avg_probs + 1e-8) + (1 - avg_probs) * torch.log(1 - avg_probs + 1e-8))

        return loss, index
    
    else:
        with torch.no_grad():
            confidence = F.softmax(torch.abs(torch.sigmoid(logits) - 0.5) * 2, dim=0)
            extra_select_num = max(0, select_num - 1)
            if extra_select_num > 0:
                select, index = torch.topk(confidence, extra_select_num, dim=0)
            else:
                index = torch.empty(0, dtype=torch.long, device=logits.device)
            index = torch.cat([torch.tensor([0], device=index.device).long(), index], dim=0)

        probs = torch.sigmoid(logits)
        avg_probs = probs[index].mean()
        loss = - (avg_probs * torch.log(avg_probs + 1e-8) + (1 - avg_probs) * torch.log(1 - avg_probs + 1e-8))
        return loss, index
