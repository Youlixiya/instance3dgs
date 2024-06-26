import os
import cv2
import torch
import json
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from utils.general_utils import AttrDict
from PIL import Image
from sklearn.metrics import jaccard_score, accuracy_score
from scene.cameras import Simple_Camera, C2W_Camera, MiniCam
from gaussian_renderer import render
from scene.camera_scene import CamScene
from scene import Scene, GaussianModel, GaussianFeatureModel
from scene.clip_encoder import OpenCLIPNetwork, OpenCLIPNetworkConfig
from utils.colormaps import ColormapOptions, apply_colormap, get_pca_dict
from utils.color import generate_contrasting_colors
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args

COLORS = torch.tensor(generate_contrasting_colors(500), dtype=torch.uint8, device='cuda')

@torch.no_grad()
def select_semantic_embeddings(clip_model, gaussian, text_prompt, neg_features, text_query_threshold, device='cuda'):
    
    text_prompt = clip.tokenize(text_prompt.split(',')).to(device)
    # text_prompt = alpha_clip.tokenize([self.text_prompt.value]).to(self.device)
    pos_features = clip_model.encode_text(text_prompt)
    pos_features /= pos_features.norm(dim=-1, keepdim=True)
    
    total_features = torch.cat([neg_features, pos_features])
    total_mm = gaussian.clip_embeddings @ total_features.T
    pos_mm = total_mm[:, 1:]
    neg_mm = total_mm[:, [0]].repeat(1, pos_mm.shape[-1])
    # print(pos_mm.shape)
    # print(pos_mm.shape)
    total_similarity = torch.stack([pos_mm, neg_mm], dim=-1)
    softmax = (100 * total_similarity).softmax(-1)
    pos_softmax = softmax[..., 0]
    valid_mask = pos_softmax > text_query_threshold
    semantic_valid_num = valid_mask.sum(0)
    semantic_embeddings = []
    for i in range(valid_mask.shape[-1]):
        semantic_embeddings.append(gaussian.instance_embeddings[valid_mask[:, i], :])
    semantic_embeddings = torch.cat(semantic_embeddings)
    return semantic_valid_num, semantic_embeddings

def text_semantic_segmentation(image, semantic_embeddings, instance_feature, mask_threshold, semantic_valid_num=None, device='cuda'):
    similarity_map = (instance_feature.reshape(-1, h * w).permute(1, 0) @ semantic_embeddings.T).reshape(h, w, -1)
    masks = (similarity_map > mask_threshold)
    masks_all = masks.any(-1)
    semantic_mask_map = image.clone()
    # sematic_object_map = image.clone()
    sematic_object_map = torch.cat([image.clone(), masks_all[..., None]], dim=-1)
    start_index = 0
    # print(semantic_valid_num)
    # print(semantic_embeddings.shape)
    for i in range(len(semantic_valid_num)):
        mask = masks[..., start_index:start_index + semantic_valid_num[i]].any(-1)
        # print(mask.shape)
        semantic_mask_map[mask, :] = semantic_mask_map[mask, :] * 0.5 + COLORS[i] / 255 * 0.5
        # semantic_mask_map[~mask, :] = semantic_mask_map[~mask, :] * 0.5 + torch.tensor([0, 0, 0], device=self.device) * 0.5
        start_index += semantic_valid_num[i]
    semantic_mask_map[~masks_all, :] /= 2
    # sematic_object_map[~masks_all, :] = torch.tensor([1, 1, 1], dtype=torch.float32, device=device)
    # sematic_object_map[masks_all, -1] = torch.tensor([1], dtype=torch.float32, device=device)
    
    return masks_all, semantic_mask_map, sematic_object_map

def get_instance_embeddings(gaussian, points, render_instance_feature, device='cuda'):
    h, w = render_instance_feature.shape[1:]
    points = torch.tensor(points, dtype=torch.int64, device=device)
    instance_embeddings = []
    for point in points:
        instance_embedding = F.normalize(render_instance_feature[:, point[0], point[1]][None], dim=-1)
        instance_embedding_index = torch.argmax((instance_embedding @ gaussian.instance_embeddings.T).softmax(-1))
        instance_embedding = gaussian.instance_embeddings[instance_embedding_index]
        instance_embeddings.append(instance_embedding)
    instance_embeddings = torch.stack(instance_embeddings)
    return instance_embeddings
    
def point_instance_segmentation(image, gaussian, instance_embeddings, render_instance_feature, mask_threshold, device='cuda'):
    similarity_map = (F.normalize(render_instance_feature.reshape(-1, h * w).permute(1, 0), dim=1) @ instance_embeddings.T).reshape(h, w, -1)
    masks = (similarity_map > mask_threshold)
    masks_all_instance = masks.any(-1)
    instance_mask_map = image.clone()
    instance_object_map = image.clone()
    for i, mask in enumerate(masks.permute(2, 0, 1)):
        instance_mask_map[mask, :] = instance_mask_map[mask, :] * 0.5 + torch.tensor(COLORS[i], dtype=torch.float32, device=device) /255 * 0.5
    instance_mask_map[~masks_all_instance, :] /= 2
    instance_object_map[~masks_all_instance, :] = torch.tensor([1, 1, 1], dtype=torch.float32, device=device)
    
    return masks_all_instance, instance_mask_map, instance_object_map

def instance_segmentation_all(gaussian, render_instance_feature):
    h, w = render_instance_feature.shape[1:]
    instance_index = torch.argmax((F.normalize(render_instance_feature.reshape(-1, h * w).permute(1, 0), dim=1) @ gaussian.instance_embeddings.T).softmax(-1), dim=-1).cpu()
    # print(instance_index)
    instance_masks = gaussian.instance_colors[instance_index].reshape(h, w, 3)
    return instance_masks

def hex_to_rgb(x):
    return [int(x[i:i + 2], 16) / 255 for i in (1, 3, 5)]

class DistinctColors:

    def __init__(self):
        colors = [
            '#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f55031', '#911eb4', '#42d4f4', '#bfef45', '#fabed4', '#469990',
            '#dcb1ff', '#404E55', '#fffac8', '#809900', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9', '#f032e6',
            '#806020', '#ffffff',

            "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0030ED", "#3A2465", "#34362D", "#B4A8BD", "#0086AA",
            "#452C2C", "#636375", "#A3C8C9", "#FF913F", "#938A81", "#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700",

            "#04F757", "#C8A1A1", "#1E6E00",
            "#7900D7", "#A77500", "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700",
            "#549E79", "#FFF69F", "#201625", "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329",
            "#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804", "#324E72", "#6A3A4C",
        ]
        self.hex_colors = colors
        # 0 = crimson / red, 1 = green, 2 = yellow, 3 = blue
        # 4 = orange, 5 = purple, 6 = sky blue, 7 = lime green
        self.colors = [hex_to_rgb(c) for c in colors]
        self.color_assignments = {}
        self.color_ctr = 0
        self.fast_color_index = torch.from_numpy(np.array([hex_to_rgb(colors[i % len(colors)]) for i in range(8096)] + [hex_to_rgb('#000000')]))

    def get_color(self, index, override_color_0=False):
        colors = [x for x in self.hex_colors]
        if override_color_0:
            colors[0] = "#3f3f3f"
        colors = [hex_to_rgb(c) for c in colors]
        if index not in self.color_assignments:
            self.color_assignments[index] = colors[self.color_ctr % len(self.colors)]
            self.color_ctr += 1
        return self.color_assignments[index]

    def get_color_fast_torch(self, index):
        return self.fast_color_index[index]

    def get_color_fast_numpy(self, index, override_color_0=False):
        index = np.array(index).astype(np.int32)
        if override_color_0:
            colors = [x for x in self.hex_colors]
            colors[0] = "#3f3f3f"
            fast_color_index = torch.from_numpy(np.array([hex_to_rgb(colors[i % len(colors)]) for i in range(8096)] + [hex_to_rgb('#000000')]))
            return fast_color_index[index % fast_color_index.shape[0]].numpy()
        else:
            return self.fast_color_index[index % self.fast_color_index.shape[0]].numpy()

    def apply_colors(self, arr):
        out_arr = torch.zeros([arr.shape[0], 3])

        for i in range(arr.shape[0]):
            out_arr[i, :] = torch.tensor(self.get_color(arr[i].item()))
        return out_arr

    def apply_colors_fast_torch(self, arr):
        return self.fast_color_index[arr % self.fast_color_index.shape[0]]

    def apply_colors_fast_numpy(self, arr):
        return self.fast_color_index.numpy()[arr % self.fast_color_index.shape[0]]

def get_boundary_mask(arr, dialation_size=1):
    import cv2
    arr_t, arr_r, arr_b, arr_l = arr[1:, :], arr[:, 1:], arr[:-1, :], arr[:, :-1]
    arr_t_1, arr_r_1, arr_b_1, arr_l_1 = arr[2:, :], arr[:, 2:], arr[:-2, :], arr[:, :-2]
    kernel = np.ones((dialation_size, dialation_size), 'uint8')
    if isinstance(arr, torch.Tensor):
        arr_t = torch.cat([arr_t, arr[-1, :].unsqueeze(0)], dim=0)
        arr_r = torch.cat([arr_r, arr[:, -1].unsqueeze(1)], dim=1)
        arr_b = torch.cat([arr[0, :].unsqueeze(0), arr_b], dim=0)
        arr_l = torch.cat([arr[:, 0].unsqueeze(1), arr_l], dim=1)

        arr_t_1 = torch.cat([arr_t_1, arr[-2, :].unsqueeze(0), arr[-1, :].unsqueeze(0)], dim=0)
        arr_r_1 = torch.cat([arr_r_1, arr[:, -2].unsqueeze(1), arr[:, -1].unsqueeze(1)], dim=1)
        arr_b_1 = torch.cat([arr[0, :].unsqueeze(0), arr[1, :].unsqueeze(0), arr_b_1], dim=0)
        arr_l_1 = torch.cat([arr[:, 0].unsqueeze(1), arr[:, 1].unsqueeze(1), arr_l_1], dim=1)

        boundaries = torch.logical_or(torch.logical_or(torch.logical_or(torch.logical_and(arr_t != arr, arr_t_1 != arr), torch.logical_and(arr_r != arr, arr_r_1 != arr)), torch.logical_and(arr_b != arr, arr_b_1 != arr)), torch.logical_and(arr_l != arr, arr_l_1 != arr))

        boundaries = boundaries.cpu().numpy().astype(np.uint8)
        boundaries = cv2.dilate(boundaries, kernel, iterations=1)
        boundaries = torch.from_numpy(boundaries).to(arr.device)
    else:
        arr_t = np.concatenate([arr_t, arr[-1, :][np.newaxis, :]], axis=0)
        arr_r = np.concatenate([arr_r, arr[:, -1][:, np.newaxis]], axis=1)
        arr_b = np.concatenate([arr[0, :][np.newaxis, :], arr_b], axis=0)
        arr_l = np.concatenate([arr[:, 0][:, np.newaxis], arr_l], axis=1)

        arr_t_1 = np.concatenate([arr_t_1, arr[-2, :][np.newaxis, :], arr[-1, :][np.newaxis, :]], axis=0)
        arr_r_1 = np.concatenate([arr_r_1, arr[:, -2][:, np.newaxis], arr[:, -1][:, np.newaxis]], axis=1)
        arr_b_1 = np.concatenate([arr[0, :][np.newaxis, :], arr[1, :][np.newaxis, :], arr_b_1], axis=0)
        arr_l_1 = np.concatenate([arr[:, 0][:, np.newaxis], arr[:, 1][:, np.newaxis], arr_l_1], axis=1)

        boundaries = np.logical_or(np.logical_or(np.logical_or(np.logical_and(arr_t != arr, arr_t_1 != arr), np.logical_and(arr_r != arr, arr_r_1 != arr)), np.logical_and(arr_b != arr, arr_b_1 != arr)), np.logical_and(arr_l != arr, arr_l_1 != arr)).astype(np.uint8)
        boundaries = cv2.dilate(boundaries, kernel, iterations=1)

    return boundaries
def vis_seg(dc, class_index, H, W, rgb=None, alpha = 0.65):
    segmentation_map = dc.apply_colors_fast_torch(class_index)
    if rgb is not None:
        segmentation_map = segmentation_map * alpha + rgb * (1 - alpha)
    boundaries = get_boundary_mask(class_index.view(H, W))
    segmentation_map = segmentation_map.reshape(H, W, 3)
    segmentation_map[boundaries > 0, :] = 0
    segmentation_map = segmentation_map.detach().numpy().astype(np.float32)
    segmentation_map *= 255.
    segmentation_map = segmentation_map.astype(np.uint8)
    return segmentation_map

def read_segmentation_maps(root_dir, downsample=8):
        segmentation_path = os.path.join(root_dir, 'segmentations')
        classes_file_path = os.path.join(root_dir, 'segmentations', 'classes.txt')
        with open(classes_file_path, 'r') as f:
            classes = f.readlines()
        classes = [class_.strip() for class_ in classes]
        classes.sort()
        # print(classes)
        # get a list of all the folders in the directory
        folders = [f for f in sorted(os.listdir(segmentation_path)) if os.path.isdir(os.path.join(segmentation_path, f))]
        seg_maps = []
        idxes = [] # the idx of the test imgs
        for folder in folders:
            idxes.append(int(folder))  # to get the camera id
            seg_for_one_image = []
            for class_name in classes:
                # check if the seg map exists
                seg_path = os.path.join(root_dir, f'segmentations/{folder}/{class_name}.png')
                if not os.path.exists(seg_path):
                    raise Exception(f'Image {class_name}.png does not exist')
                img = Image.open(seg_path).convert('L')
                # resize the seg map
                if downsample != 1.0:

                    img_wh = (int(img.size[0] / downsample), int(img.size[1] / downsample))
                    img = img.resize(img_wh, Image.NEAREST) # [W, H]
                img = (np.array(img) / 255.0).astype(np.int8) # [H, W]
                img = img.flatten() # [H*W]
                seg_for_one_image.append(img)

            seg_for_one_image = np.stack(seg_for_one_image, axis=0)
            seg_for_one_image = seg_for_one_image.transpose(1, 0)
            seg_maps.append(seg_for_one_image)

        seg_maps = np.stack(seg_maps, axis=0) # [n_frame, H*W, n_class]
        return seg_maps

if __name__ == '__main__':
    parser = ArgumentParser()
    lp = ModelParams(parser)
    pipe = PipelineParams(parser)
    parser.add_argument("--cfg_path", type=str, required=True)
    # parser.add_argument("--scene", type=str, required=True)
    # parser.add_argument("--gt_path", type=str, required=True)
    # parser.add_argument("--output_path", type=str, required=True)
    # parser.add_argument("--scene", type=str, required=True)
    scene_names = ['bed', 'bench', 'lawn', 'room', 'sofa']
    args = parser.parse_args()
    metrics = 'Scene\tIoU\tAcc\n'
    for scene_name in tqdm(scene_names):
        with open(args.cfg_path, 'r') as f:
            cfg = AttrDict(json.load(f)[scene_name])
        args = AttrDict(args.__dict__)
        args.update(cfg)
        gaussian = GaussianFeatureModel(sh_degree=3, gs_feature_dim=args.gs_feature_dim)
        gaussian.load_ply(args.gs_source)
        if args.feature_gs_source:
            gaussian.load_feature_params(args.feature_gs_source)
        
        background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        feature_bg = torch.tensor([0] *gaussian.gs_feature_dim, dtype=torch.float32, device="cuda")
        colmap_cameras = None
        render_cameras = None
        target_image_names = args.image_names
        seg_maps = read_segmentation_maps(args.colmap_dir)
        if args.colmap_dir is not None:
            img_name = os.listdir(os.path.join(args.colmap_dir, args.images))[0]
            # if args.h == -1 and args.w == -1:
            h, w = cv2.imread(os.path.join(args.colmap_dir, args.images, img_name)).shape[:2]
            # else:
            #     h = args.h
            #     w = args.w
            scene = CamScene(args.colmap_dir, h=h, w=w)
            cameras_extent = scene.cameras_extent
            colmap_cameras = scene.cameras
            img_suffix = os.listdir(os.path.join(args.colmap_dir, args.images))[0].split('.')[-1]
            imgs_name = [f'{camera.image_name}.{img_suffix}' for camera in colmap_cameras]
            # print(imgs_name)
            imgs_path = [os.path.join(args.colmap_dir, args.images, img_name) for img_name in imgs_name]
        instance_embeddings = []
        rendered_feature_pca_dict = None
        instance_feature_pca_dict = None
        IoUs, accuracies = [], []
        dc = DistinctColors()
        #clip
        clip = OpenCLIPNetwork(OpenCLIPNetworkConfig())
        clip_text_prompt = args.clip_text_prompt
        clip_text_prompt.sort()
        # print(clip_text_prompt)
        clip.set_positives(clip_text_prompt)
        valid_index = 0
        for i in tqdm(range(len(colmap_cameras))):
            cam = colmap_cameras[i]
            if cam.image_name not in target_image_names or valid_index == len(seg_maps):
                continue
            gt_seg = seg_maps[valid_index]
            valid_index += 1
            with torch.no_grad():
                mask_map_save_path = os.path.join(args.save_path, cam.image_name, 'mask_maps')
                mask_save_path = os.path.join(args.save_path, cam.image_name, 'masks')
                mask_object_save_path = os.path.join(args.save_path, cam.image_name, 'objects')
                os.makedirs(mask_map_save_path, exist_ok=True)
                os.makedirs(mask_save_path, exist_ok=True)
                os.makedirs(mask_object_save_path, exist_ok=True)
                render_pkg = render(cam, gaussian, pipe, background)
                image_tensor = render_pkg['render'].permute(1, 2, 0).clamp(0, 1)
                H, W = image_tensor.shape[:2]
                rgb = image_tensor.reshape(-1, 3)
                image = Image.fromarray((image_tensor.cpu().numpy() * 255).astype(np.uint8))
                render_feature = render(cam, gaussian, pipe, feature_bg, render_feature=True, override_feature=gaussian.gs_features)['render_feature']
                instance_feature = F.normalize(render_feature.reshape(-1, h*w), dim=0).reshape(-1, h, w)
                if rendered_feature_pca_dict is None:
                    rendered_feature_pca_dict = get_pca_dict(render_feature)
                if instance_feature_pca_dict is None:
                    instance_feature_pca_dict = get_pca_dict(instance_feature)
                Image.fromarray((apply_colormap(render_feature.permute(1, 2, 0), ColormapOptions(colormap="pca", pca_dict=rendered_feature_pca_dict)).cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(args.save_path, cam.image_name, f'rendered_feature_pca.jpg'))
                Image.fromarray((apply_colormap(instance_feature.permute(1, 2, 0), ColormapOptions(colormap="pca", pca_dict=instance_feature_pca_dict)).cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(args.save_path, cam.image_name, f'instance_feature_pca.jpg'))
                if instance_embeddings == []:
                    for j in range(len(clip.positives)):
                        relevancy = clip.get_relevancy(gaussian.aggregation_clip_embeddings, j)[..., 0]
                        print(relevancy.max())
                        # mask_index = relevancy > args.text_query_threshold[j]
                        mask_index = relevancy.argmax()
                        instance_embeddings.append(gaussian.instance_embeddings[mask_index][None])
                    print(torch.cat(instance_embeddings).shape)
                    instance_embeddings = torch.cat(instance_embeddings)
                p_class = (instance_embeddings @ instance_feature.reshape(-1, h*w)).softmax(0)
                # print(p_class.shape)
                class_index = torch.argmax(p_class, dim=0).cpu() # [N1]
                segmentation_map = vis_seg(dc, class_index, H, W, rgb=rgb.cpu())

                one_hot = F.one_hot(class_index.long(), num_classes=gt_seg.shape[-1]) # [N1, n_classes]
                one_hot = one_hot.detach().cpu().numpy().astype(np.int8)
                IoUs.append(jaccard_score(gt_seg, one_hot, average=None))
                print('iou for classes:', IoUs[-1], 'mean iou:', np.mean(IoUs[-1]))
                accuracies.append(accuracy_score(gt_seg, one_hot))
                print('accuracy:', accuracies[-1])
                Image.fromarray(segmentation_map).save(os.path.join(mask_map_save_path, 'segmentation_map.png'))
        metrics += f'{scene_name}\t{np.mean(IoUs)}\t{np.mean(accuracies)}\n'
        with open(f'ovs3d.txt', 'w') as f:
            f.write(metrics)
                    # semantic_valid_num = [len(instance_embeddings)]
                    # masks_all_semantic, semantic_mask_map, semantic_object_map = text_semantic_segmentation(image_tensor, instance_embeddings, instance_feature, args.text_mask_threshold[j], semantic_valid_num)
                    # clip_mask = (masks_all_semantic.cpu().numpy() * 255).astype(np.uint8)
                    # clip_mask_map = (semantic_mask_map.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                    # clip_object = (semantic_object_map.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                    # text_prompt = args.clip_text_prompt[j]
                    # Image.fromarray(clip_mask).save(os.path.join(mask_save_path, f'{text_prompt}.png'))
                    # Image.fromarray(clip_mask_map).save(os.path.join(mask_map_save_path, f'{text_prompt}.png'))
                    # Image.fromarray(clip_object).save(os.path.join(mask_object_save_path, f'{text_prompt}.png'))
                    # clip_semantic_embeddings = instance_embeddings[mask_index]
                # if instance_embeddings == None:
                #     instance_embeddings = get_instance_embeddings(gaussian, args.points, render_feature)
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
                #     total_rendered_feature = F.normalize(total_rendered_feature, dim=-1)
                # total_rendered_feature = total_rendered_feature.permute(1, 0).reshape(-1, h, w)
                # masks_all_instance, instance_mask_map, instance_object_map = point_instance_segmentation(image_tensor, gaussian, instance_embeddings, render_feature, args.mask_threshold, device='cuda')
                # instance_masks = instance_segmentation_all(gaussian, render_feature)
                # image.save(os.path.join(args.save_path, f'rendered_rgb_{args.image_name}'))
                # Image.fromarray((apply_colormap(render_feature.permute(1, 2, 0), ColormapOptions(colormap="pca")).cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(args.save_path, f'rendered_feature_pca_{args.image_name}'))
                # Image.fromarray((apply_colormap(render_feature.permute(1, 2, 0), ColormapOptions(colormap="pca")).cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(args.save_path, f'total_rendered_feature_pca_{args.image_name}'))
                # Image.fromarray(np.stack([(masks_all_instance.cpu().numpy() * 255).astype(np.uint8)] * 3, axis=-1)).save(os.path.join(mask_save_path, f'mask_{i:03d}.png'))
                # Image.fromarray((instance_mask_map.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(mask_map_save_path, f'mask_map_{i:03d}.jpg'))
                # Image.fromarray((instance_object_map.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(args.save_path, f'instance_object_map_{args.image_name}'))
                # Image.fromarray((instance_masks.cpu().numpy()).astype(np.uint8)).save(os.path.join(mask_save_path, f'mask_{i}.png'))
        # device = "cuda:0"
        # self.colors = np.random.random((500, 3))