import os
import json
import math
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from PIL import Image
from argparse import ArgumentParser
from utils.general_utils import AttrDict

def get_iou(pred_mask, gt_mask):
    intersection = pred_mask & gt_mask
    union = pred_mask | gt_mask
    iou = intersection.sum() / union.sum()
    return iou

# def get_iou(pred_mask, gt_mask):
#     pred_mask = pred_mask == 255
#     gt_mask = gt_mask == 255
#     intersection = pred_mask & gt_mask
#     union = pred_mask | gt_mask
#     iou = intersection.sum() / union.sum()
#     return iou

# def get_iou(pred_mask, gt_mask):
#     ious = []
#     intersection_pos = (pred_mask==255) & (gt_mask==255)
#     union_pos = (pred_mask==255) | (gt_mask==255)
#     ious.append(intersection_pos.sum() / union_pos.sum())
#     intersection_neg = (pred_mask==0) & (gt_mask==0)
#     union_neg = (pred_mask==0) | (gt_mask==0)
#     ious.append(intersection_neg.sum() / union_neg.sum())
#     return sum(ious) / len(ious)

def get_accuracy(pred_mask, gt_mask):
    h, w = pred_mask.shape
    # print(((pred_mask==255) & (gt_mask==255)).shape)
    # print((pred_mask==255).sum())
    pos_acc = ((pred_mask==255) & (gt_mask==255)).sum() / (gt_mask==255).sum()
    # print(pos_acc)
    neg_acc = ((pred_mask==0) & (gt_mask==0)).sum() / (gt_mask==0).sum()
    # tp_fn = (pred_mask == gt_mask).sum()
    return (pos_acc + neg_acc) / 2

# def get_accuracy(pred_mask, gt_mask):
#     h, w = pred_mask.shape
#     # print(((pred_mask==255) & (gt_mask==255)).shape)
#     # print((pred_mask==255).sum())
#     pos_acc = ((pred_mask==255) & (gt_mask==255)).sum() / (gt_mask==255).sum()
#     # print(pos_acc)
#     # neg_acc = ((pred_mask==0) & (gt_mask==0)).sum() / (gt_mask==0).sum()
#     # tp_fn = (pred_mask == gt_mask).sum()
#     return pos_acc

# def get_accuracy(pred_mask, gt_mask):
#     h, w = pred_mask.shape
#     # print(((pred_mask==255) & (gt_mask==255)).shape)
#     # print((pred_mask==255).sum())
#     pos_acc = ((pred_mask==255) & (gt_mask==255)).sum() / (gt_mask==255).sum()
#     # print(pos_acc)
#     # neg_acc = ((pred_mask==0) & (gt_mask==0)).sum() / (gt_mask==0).sum()
#     # tp_fn = (pred_mask == gt_mask).sum()
#     return pos_acc

# def get_pr(pred_mask, gt_mask):
#     # tp = (pred_mask == gt_mask).sum()
#     tp = ((pred_mask & gt_mask) == 255).sum()
#     tp_fp = (pred_mask.reshape(-1) == 255).sum()
#     return tp / tp_fp

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--cfg_path", type=str, default='scripts/16_ovs3d_test_config.json')
    parser.add_argument("--pred_path", type=str, default='output/16_ovs3d_masks')
    parser.add_argument("--gt_path", type=str, default='data/ovs3d')
    
    args = parser.parse_args()
    with open(args.cfg_path, 'r') as f:
        cfg = json.load(f)
    scenes = list(cfg.keys())
    # masks = []
    # gt_masks = []
    ious = []
    accs = []
    # prs = []
    metrics = 'Scene\tIoU\tAcc\n'
    for scene in scenes:
        print(scene)
        # scene_cfg = cfg[scene]
        # print(scenes)
        # for prompt in prompts:
        iou = []
        acc = []
        mask_path = os.path.join(args.gt_path, scene, 'segmentations')
        views = os.listdir(mask_path)
        views = [view for view in views if 'txt' not in view]
        # scene_ious = {}
        # scene_accs = {}
        scene_ious = []
        scene_accs = []
        for view in tqdm(views):
            view_masks = os.listdir(os.path.join(mask_path, view))
            view_ious = []
            view_accs = []
            for view_mask in view_masks:
                gt_mask_path = os.path.join(mask_path, view, view_mask)
                pred_mask_path = os.path.join(args.pred_path, scene, view, 'masks', view_mask)
                mask = np.array(Image.open(pred_mask_path))
                gt_mask = np.array(Image.open(gt_mask_path).resize((mask.shape[1], mask.shape[0])))[..., 0]
                
                iou = get_iou(mask, gt_mask)
                acc = get_accuracy(mask, gt_mask)
                view_ious.append(iou)
                view_accs.append(acc)
            scene_ious.append(sum(view_ious) / len(view_ious))
            scene_accs.append(sum(view_accs) / len(view_accs))
            
                # if scene_ious.get(view_mask, 0) == 0:
                #     scene_ious[view_mask] = [iou]
                #     scene_accs[view_mask] = [acc]
                # else:
                #     scene_ious[view_mask].append(iou)
                #     scene_accs[view_mask].append(acc)
        # scene_iou = []
        # scene_acc = []
        # for key in scene_ious.keys():
        #     scene_iou.append(sum(scene_ious[key]) / len(scene_ious[key]))
        #     scene_acc.append(sum(scene_accs[key]) / len(scene_accs[key]))

        scene_iou = sum(scene_ious) / len(scene_ious)
        scene_acc = sum(scene_accs) / len(scene_accs)
        ious.append(scene_iou)
        accs.append(scene_acc)
        metrics += f'{scene}\t{scene_iou}\t{scene_acc}\n'
    mean_iou = sum(ious) / len(ious)
    mean_acc = sum(accs) / len(accs)
    metrics += f'mean\t{mean_iou}\t{mean_acc}'
    with open(f'ovs3d.txt', 'w') as f:
        f.write(metrics)


