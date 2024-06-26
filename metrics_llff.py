import os
import cv2
import json
import math
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from utils.general_utils import AttrDict
from PIL import Image
from argparse import ArgumentParser

def get_iou(pred_mask, gt_mask):
    intersection = pred_mask & gt_mask
    union = pred_mask | gt_mask
    iou = intersection.sum() / union.sum()
    return iou

def get_accuracy(pred_mask, gt_mask):
    h, w = pred_mask.shape
    return (pred_mask==gt_mask).sum() / (h * w)

# def get_accuracy(pred_mask, gt_mask):
#     h, w = pred_mask.shape
#     # print(((pred_mask==255) & (gt_mask==255)).shape)
#     # print((pred_mask==255).sum())
#     pos_acc = ((pred_mask==255) & (gt_mask==255)).sum() / (gt_mask==255).sum()
#     # print(pos_acc)
#     neg_acc = ((pred_mask==0) & (gt_mask==0)).sum() / (gt_mask==0).sum()
#     # tp_fn = (pred_mask == gt_mask).sum()
#     return (pos_acc + neg_acc) / 2

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--cfg_path", type=str, default='scripts/16_llff_test_config.json')
    parser.add_argument("--gt_path", type=str, default='llff_reasoning_masks')
    # parser.add_argument("--save_tag", type=str, required=True)
    
    args = parser.parse_args()
    with open(args.cfg_path, 'r') as f:
        cfg = json.load(f)
    masks = []
    gt_masks = []
    scenes = []
    ious = []
    accs = []
    metrics = 'Scene\tIoU\tAcc\tPR\n'
    gt_masks_root_dirs = os.listdir(args.gt_path) 
    for scene, scene_cfg in cfg.items():
        scenes.append(scene)
        scene_cfg = AttrDict(scene_cfg)
        mask_path = os.path.join(scene_cfg.save_path, scene_cfg.reasoning_prompt, 'masks')
        masks_name = os.listdir(mask_path)
        masks_name.sort()
        masks_path = [os.path.join(mask_path, mask_name) for mask_name in masks_name]
        masks = [np.array(Image.open(mask_path)) for mask_path in masks_path]
        for root_dir in gt_masks_root_dirs:
            if scene in root_dir:   
                gt_mask_root_path = os.path.join(args.gt_path, root_dir, scene_cfg.reasoning_prompt)
        gt_masks_name = os.listdir(gt_mask_root_path)
        gt_masks_name = sorted(gt_masks_name.copy())
        gt_masks_name = [c for idx, c in enumerate(gt_masks_name) if idx % 8 == 0]
        gt_masks_path = [os.path.join(gt_mask_root_path, gt_mask_name) for gt_mask_name in gt_masks_name]
        gt_masks = [np.array(Image.open(gt_mask_path)) for gt_mask_path in gt_masks_path]
        iou = []
        acc = []
        # print(scene)
        # print(len(masks))
        # print(len(gt_masks))
        for i in tqdm(range(len(masks))):
            gt_mask = gt_masks[i]
            # gt_mask = gt_mask * 255
            # print(gt_mask.shape)
            # gt_mask[gt_mask>0] = 255
            # gt_mask = gt_masks[i] * 255
            mask = masks[i]
            iou.append(get_iou(mask, gt_mask))
            acc.append(get_accuracy(mask, gt_mask))
        scene_iou = sum(iou) / len(iou)
        scene_acc = sum(acc) / len(acc)
        ious.append(scene_iou)
        accs.append(scene_acc)
        metrics += f'{scene}\t{scene_iou}\t{scene_acc}\n'
    mean_iou = sum(ious) / len(ious)
    mean_acc = sum(accs) / len(accs)
    metrics += f'mean\t{mean_iou}\t{mean_acc}'
    with open(f'reasoning_metrics.txt', 'w') as f:
        f.write(metrics)
