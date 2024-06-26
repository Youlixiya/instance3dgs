import os
import cv2
import torch
import json
import time
import numpy as np
import torch.nn.functional as F
from PIL import Image
from scene.cameras import Simple_Camera, C2W_Camera, MiniCam
from gaussian_renderer import render
from scene.camera_scene import CamScene
from scene import Scene, GaussianModel, GaussianFeatureModel
from utils.colormaps import ColormapOptions, apply_colormap, get_pca_dict
from utils.color import generate_contrasting_colors
from utils.general_utils import AttrDict
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args

COLORS = torch.tensor(generate_contrasting_colors(500), dtype=torch.uint8, device='cuda')

def point_instance_segmentation(image, gaussian, points, render_instance_feature, mask_threshold, device='cuda'):
    h, w = render_instance_feature.shape[1:]
    points = torch.tensor(points, dtype=torch.int64, device=device)
    instance_embeddings = []
    t1 = time.time()
    for point in points:
        instance_embedding = F.normalize(render_instance_feature[:, point[0], point[1]][None], dim=-1)
        instance_embedding_index = torch.argmax((instance_embedding @ gaussian.instance_embeddings.T).softmax(-1))
        instance_embedding = gaussian.instance_embeddings[instance_embedding_index]
        instance_embeddings.append(instance_embedding)
    instance_embeddings = torch.stack(instance_embeddings)
    similarity_map = (F.normalize(render_instance_feature.reshape(-1, h * w).permute(1, 0), dim=1) @ instance_embeddings.T).reshape(h, w, -1)
    masks = (similarity_map > mask_threshold)
    t2 = time.time()
    print(f'time:{t2 - t1}')
    masks_all_instance = masks.any(-1)
    instance_mask_map = image.clone()
    instance_object_map = image.clone()
    for i, mask in enumerate(masks.permute(2, 0, 1)):
        instance_mask_map[mask, :] = instance_mask_map[mask, :] * 0.5 + COLORS[i] /255 * 0.5
    instance_mask_map[~masks_all_instance, :] /= 2
    instance_object_map[~masks_all_instance, :] = torch.tensor([1, 1, 1], dtype=torch.float32, device=device)
    
    return masks_all_instance, instance_mask_map, instance_object_map

# def point_instance_segmentation(image, gaussian, points, render_instance_feature, mask_threshold, device='cuda'):
#     h, w = render_instance_feature.shape[1:]
#     points = torch.tensor(points, dtype=torch.int64, device=device)
#     instance_embeddings_index = []
#     for point in points:
#         instance_embedding = F.normalize(render_instance_feature[:, point[0], point[1]][None], dim=-1)
#         # instance_embedding = render_instance_feature[:, point[0], point[1]][None]
#         instance_embedding_index = torch.argmax((instance_embedding @ gaussian.instance_embeddings.T).softmax(-1))
#         instance_embeddings_index.append(instance_embedding_index)
        
#     instance_index = torch.argmax((render_instance_feature.reshape(-1, h * w).permute(1, 0) @ gaussian.instance_embeddings.T).softmax(-1), dim=-1).reshape(h, w)
#     masks = []
#     for instance_embedding_index in instance_embeddings_index:
#         masks.append(instance_index == instance_embedding_index)
#     masks = torch.stack(masks)
#     # masks = (similarity_map > mask_threshold)
#     masks_all_instance = masks.any(0)
#     instance_mask_map = image.clone()
#     instance_object_map = image.clone()
#     for i, mask in enumerate(masks):
#         instance_mask_map[mask, :] = instance_mask_map[mask, :] * 0.5 + COLORS[i] /255 * 0.5
#     instance_mask_map[~masks_all_instance, :] /= 2
#     instance_object_map[~masks_all_instance, :] = torch.tensor([1, 1, 1], dtype=torch.float32, device=device)
    
#     return masks_all_instance, instance_mask_map, instance_object_map


def instance_segmentation_all(image, gaussian, render_instance_feature):
    h, w = render_instance_feature.shape[1:]
    instance_index = torch.argmax((render_instance_feature.reshape(-1, h * w).permute(1, 0) @ gaussian.instance_embeddings.T).softmax(-1), dim=-1).cpu()
    # print(instance_index)
    instance_masks = COLORS[instance_index].reshape(h, w, 3) /255 * 0.5 + image * 0.5
    return instance_masks

if __name__ == '__main__':
    parser = ArgumentParser()
    lp = ModelParams(parser)
    pipe = PipelineParams(parser)
    parser.add_argument("--cfg_path", type=str, required=True)
    parser.add_argument("--scene", type=str, required=True)
    args = parser.parse_args()
    with open(args.cfg_path, 'r') as f:
        cfg = AttrDict(json.load(f)[args.scene])
    args = AttrDict(args.__dict__)
    args.update(cfg)
    # if 'rgb' in args.feature_gs_source:
    #     rgb_decode = True
    # else:
    #     rgb_decode = False
    # if 'depth' in args.feature_gs_source:
    #     depth_decode = True
    # else:
    #     depth_decode = False
    gaussian = GaussianFeatureModel(sh_degree=3, gs_feature_dim=args.gs_feature_dim)
    gaussian.load_ply(args.gs_source)
    if args.feature_gs_source:
        gaussian.load_feature_params(args.feature_gs_source)
    
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    feature_bg = torch.tensor([0] *gaussian.gs_feature_dim, dtype=torch.float32, device="cuda")
    colmap_cameras = None
    render_cameras = None
    if args.colmap_dir is not None:
        img_name = os.listdir(os.path.join(args.colmap_dir, args.images))[0]
        h, w = cv2.imread(os.path.join(args.colmap_dir, args.images, img_name)).shape[:2]
        scene = CamScene(args.colmap_dir, h=h, w=w)
        cameras_extent = scene.cameras_extent
        colmap_cameras = scene.cameras
        img_suffix = os.listdir(os.path.join(args.colmap_dir, args.images))[0].split('.')[-1]
        imgs_name = [f'{camera.image_name}.{img_suffix}' for camera in colmap_cameras]
        imgs_path = [os.path.join(args.colmap_dir, args.images, img_name) for img_name in imgs_name]
    for i, img in enumerate(imgs_name):
        if args.image_name == img:
            break
    cam = colmap_cameras.pop(i)
    os.makedirs(args.save_path, exist_ok=True)
    rendered_feature_pca_dict = None
    instance_feature_pca_dict = None
    with torch.no_grad():
        render_pkg = render(cam, gaussian, pipe, background)
        image_tensor = render_pkg['render'].permute(1, 2, 0).clamp(0, 1)
        image = Image.fromarray((image_tensor.cpu().numpy() * 255).astype(np.uint8))
        render_feature = render(cam, gaussian, pipe, feature_bg, render_feature=True, override_feature=gaussian.gs_features)['render_feature']
        if rendered_feature_pca_dict is None:
            rendered_feature_pca_dict = get_pca_dict(render_feature)
        # total_rendered_feature = [render_feature]
        # if gaussian.rgb_decode:
        #     total_rendered_feature.append(render_pkg['render'])
        # if gaussian.depth_decode:
        #     total_rendered_feature.append(render_pkg['depth_3dgs'])
        # total_rendered_feature = torch.cat(total_rendered_feature, dim=0)
        # h, w = total_rendered_feature.shape[1:]
        # total_rendered_feature = total_rendered_feature.reshape(-1, h*w).permute(1, 0)
        # if gaussian.feature_aggregator:
        #     total_rendered_feature = F.normalize(gaussian.feature_aggregator(total_rendered_feature), dim=-1)
        # else:
        instance_feature = F.normalize(render_feature.reshape(-1, h*w), dim=0).reshape(-1, h, w)
        if instance_feature_pca_dict is None:
            instance_feature_pca_dict = get_pca_dict(instance_feature)
        # instance_feature = F.normalize(gaussian.instance_feature_decoder(render_feature[None])[0].reshape(-1, h*w), dim=0).reshape(-1, h, w)
        masks_all_instance, instance_mask_map, instance_object_map = point_instance_segmentation(image_tensor, gaussian, args.points, instance_feature, args.mask_threshold, device='cuda')
        instance_masks = instance_segmentation_all(image_tensor, gaussian, instance_feature)
        image.save(os.path.join(args.save_path, f'rendered_rgb_{args.image_name}'))
        Image.fromarray((apply_colormap(render_feature.permute(1, 2, 0), ColormapOptions(colormap="pca", pca_dict=rendered_feature_pca_dict)).cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(args.save_path, f'rendered_feature_pca_{args.image_name}'))
        Image.fromarray((apply_colormap(instance_feature.permute(1, 2, 0), ColormapOptions(colormap="pca", pca_dict=instance_feature_pca_dict)).cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(args.save_path, f'instance_feature_pca_{args.image_name}'))
        Image.fromarray(np.stack([(masks_all_instance.cpu().numpy() * 255).astype(np.uint8)] * 3, axis=-1)).save(os.path.join(args.save_path, args.mask_save_name))
        Image.fromarray((instance_mask_map.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(args.save_path, f'instance_mask_map_{args.image_name}'))
        Image.fromarray((instance_object_map.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(args.save_path, f'instance_object_map_{args.image_name}'))
        Image.fromarray((instance_masks.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(args.save_path, f'instance_masks_{args.image_name}'))
    # device = "cuda:0"
    # self.colors = np.random.random((500, 3))