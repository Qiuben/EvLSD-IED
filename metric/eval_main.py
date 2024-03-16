
import os
import cv2
import math
import torch
import tqdm
import math
import numpy as np
from config.cfg import parse
from network.dataset import Dataset
from network.detector import WireframeDetector
import torch.utils.data as Data
import warnings
import matplotlib.pyplot as plt
from kornia.geometry import warp_perspective, transform_points

from cal_rep_utils import *
warnings.filterwarnings('ignore')

import wandb
wandb.init(project='event-HAWP',
           name='eval York (ortho)')

# wandb.init(project='event-HAWP',
#            name='eval wore sAP')

def to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, dict):
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(device)
        return data
    if isinstance(data, list):
        return [to_device(d, device) for d in data]
    
##vis##
def plot_event(est_event, line_pred, target_event_show, line_pred_target, H,W, plot_name):  #numpy  10,H,W
    fig = plt.figure(figsize=(6,4), clear= True)
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title('ref')
    H,W = 512, 512
    pos_image = (est_event[9, :, :] * 255).astype(np.uint8)
    neg_image = (est_event[4, :, :] * 255).astype(np.uint8)
    pos_image = cv2.equalizeHist(pos_image)
    neg_image = cv2.equalizeHist(neg_image)
    ref_image = np.concatenate((neg_image[..., None], np.zeros((H, W, 1)), pos_image[..., None]), axis=-1)
    plt.imshow(ref_image)
    
    if line_pred is not None:
        for pts in line_pred:
            pts = pts - 0.5
            ax1.plot(pts[:, 0], pts[:, 1], color="yellow", linewidth=1.5)
            ax1.scatter(pts[:, 0], pts[:, 1], color="#33FFFF", s=2.5, edgecolors="none", zorder=5)
    
    ax2 = fig.add_subplot(1,2,2)
    # plt.xlim(512)
    # plt.ylim(512)
    ax2.set_title('target')
    pos_image = (target_event_show[9, :, :] * 255).astype(np.uint8)
    neg_image = (target_event_show[4, :, :] * 255).astype(np.uint8)
    pos_image = cv2.equalizeHist(pos_image)
    neg_image = cv2.equalizeHist(neg_image)
    target_image = np.concatenate((neg_image[..., None], np.zeros((H, W, 1)), pos_image[..., None]), axis=-1)
    plt.imshow(target_image)
    
    if line_pred_target is not None:
        for pts in line_pred_target:
            pts = pts - 0.5
            ax2.plot(pts[:, 0], pts[:, 1], color="yellow", linewidth=1.5)
            ax2.scatter(pts[:, 0], pts[:, 1], color="#33FFFF", s=2.5, edgecolors="none", zorder=5)
    
    wandb.log({str(plot_name): plt})



##### functions for calculating the rep and loc metric #####
def compute_metrics_a_pair(
        line_segments_ref, 
        line_segments_target, 
        dist_tolerance_lst,
        distance_metric="sAP",
    ):
    """
    line_segments_ref: Nx2x2 array.
    line_segments_target: Nx2x2 array.
    valid_mask: 2D mask (same size as the image)
    H_mat: the 3x3 array containing the homography matrix.
    image_size: list containing [H, W].
    dist_tolerance_lst: list of all distance tolerances of interest.
    distance_metric: "sAP" or "orthogonal_distance".
    """
    
    # Check the distance_metric to use
    supported_metrics = ["sAP", "orthogonal_distance", "sAP_square"]
    if not distance_metric in supported_metrics:
        raise ValueError(f"[Error] The specified distnace metric is not in supported metrics {supported_metrics}.")

    # Compute repeatability
    num_segments_ref = line_segments_ref.shape[0]
    num_segments_target = line_segments_target.shape[0]

    # Compute closest segments in target segments for each ref segment. 
    ref_target_min_dist = compute_distances(  
        line_segments_ref, line_segments_target,
        dist_tolerance_lst, distance_metric,
        group_num=1000
    )

    ref_target_correctness_lst = []
    ref_target_loc_error_lst = []
    for dist_tolerance in dist_tolerance_lst:
        # Compute the correctness for repeatability
        ref_correct_mask = ref_target_min_dist <= dist_tolerance
        ref_target_correctness = np.sum((ref_correct_mask).astype(np.int))
        ref_target_correctness_lst.append(ref_target_correctness)

        # Compute the localization error
        ref_target_loc_error = ref_target_min_dist[ref_correct_mask]
        ref_target_loc_error_lst.append(ref_target_loc_error)

    # Compute closest segments in taget segments for each ref segment.
    target_ref_min_dist = compute_distances(
        line_segments_target, line_segments_ref,
        dist_tolerance_lst, distance_metric,
        group_num=1000
    )

    
    target_ref_correctness_lst = []
    target_ref_loc_error_lst = []
    for dist_tolerance in dist_tolerance_lst:
        # Compute the correctness for repeatability
        traget_correct_mask = target_ref_min_dist <= dist_tolerance
        target_ref_correctness = np.sum((traget_correct_mask).astype(np.int))
        target_ref_correctness_lst.append(target_ref_correctness)

        # Compute the localization error
        target_ref_loc_error = target_ref_min_dist[traget_correct_mask]
        target_ref_loc_error_lst.append(target_ref_loc_error)
    
    # Record the final correctness
    repeatability_results = {}
    loc_error_results = {}
    for i, dist in enumerate(dist_tolerance_lst):
        # Compute the final repeatability
        correctness = (ref_target_correctness_lst[i] + target_ref_correctness_lst[i]) / (num_segments_ref + num_segments_target)
        if np.isnan(correctness) or np.isinf(correctness):
            correctness = 0
        repeatability_results[dist] = correctness

        # Compute the final localization error
        # loc_error_lst = np.concatenate([ref_target_loc_error_lst[i], 
        #                                 target_ref_loc_error_lst[i]])
        # Only compute over target segments
        # import ipdb; ipdb.set_trace()
        loc_error_lst = target_ref_loc_error_lst[i]
        if 0 in loc_error_lst.shape:
            loc_error = 0
        else:
            loc_error = np.mean(loc_error_lst)
        loc_error_results[dist] = loc_error
    
    return repeatability_results, loc_error_results


if __name__ == '__main__':
    cfg = parse()
    homography_cfg = cfg.get("homography_adaptation", None)
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    # Load model
    model = WireframeDetector(cfg).to(device)
    model_filename = os.path.join(cfg.model_path, 'ewireframe_pretrained_56.5.pkl')
    checkpoint = torch.load(model_filename, map_location=device)
    if 'model' in checkpoint.keys():
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)

    # Load dataset
    dataset = Dataset(cfg, split='test', mode = 'val')
    # mini_dataset, _ = torch.utils.data.random_split(dataset = dataset, lengths =[2,460])
    export_loader = Data.DataLoader(dataset=dataset, batch_size=cfg.test_batch_size,
                             num_workers=cfg.num_workers, shuffle=False, collate_fn=dataset.collate)
    
    
    # dist_tolerance_lst = [5, 10, 15, 20, 25, 35, 45, 55]

    dist_tolerance_lst = [5]
     # Start the evaluation

    # Initialize the repeatability dict
    repeatability_dict = {}
    for dist in dist_tolerance_lst:
        repeatability_dict[dist] = np.zeros([len(dataset)], 
                    dtype=np.float32)
    
    # Initialize the localization error dict
    local_error_dict = {}
    for dist in dist_tolerance_lst:
        local_error_dict[dist] = np.zeros([len(dataset)], 
                    dtype=np.float32)

    # start prediction
    model.eval()
    for idx,(images, annotations) in enumerate(tqdm.tqdm(export_loader, desc='export test data: ')):
        images, annotations = images.to(device), to_device(annotations, device)
        W, H = annotations[0]['width'] , annotations[0]['height']
        
        homo_mat, _ = sample_homography([H,W], 
                                **homography_cfg["homographies"])
        
        # convert homo_mat to tensor
        H_tensor = torch.tensor(homo_mat, dtype=torch.float, device=device).unsqueeze(dim = 0)
        H_inv_tensor = torch.inverse(H_tensor.detach().cpu())
        ref_image = images
        target_image = warp_perspective(ref_image.to(device), H_tensor, (512, 512), mode="bilinear")

        warp_twice_image = warp_perspective(target_image.to(device), H_inv_tensor.to(device), (512, 512), mode="bilinear")

        # Compute the valid mask for the target image
        valid_mask = compute_valid_mask((H, W), homo_mat, 0)

        # Predictions on ref image and target image
        
        outputs_ref = model(ref_image, annotations)
        #target image should resize to 512
        
        target_image = target_image.squeeze(0)
        
        target_image = cv2.resize(np.array(target_image.permute(1,2,0).cpu()), (512, 512), cv2.INTER_LINEAR)
        
        target_image = torch.tensor(target_image).permute(2,0,1).unsqueeze(0)
        target_image = target_image.to(device)
        
        
        outputs_target = model(target_image, annotations)
        out_ref = outputs_ref[0]
        out_target = outputs_target[0]
        if len(out_ref['line_pred']):
            line_pred_ref = out_ref['line_pred'].detach().cpu().numpy()
            line_score_ref = out_ref['line_score'].detach().cpu().numpy()
            mask = line_score_ref > 0.97
            line_pred_ref = line_pred_ref[mask]
            line_score_ref = line_score_ref[mask]

        
        if len(out_target['line_pred']):
            line_pred_target = out_target['line_pred'].detach().cpu().numpy()
            line_score_target = out_target['line_score'].detach().cpu().numpy()
            mask = line_score_target > 0.97
            line_pred_target = line_pred_target[mask]
            line_score_target = line_score_target[mask]

        # TODO:visualize the prediction on a pair of images

        # resize ref image to original size
        ref_event_show = ref_image.squeeze(0).permute(1,2,0).to('cpu').numpy() 
        # ref_event_show = cv2.resize(ref_event_show, (W, H), cv2.INTER_LINEAR)
        ref_event_show = ref_event_show.transpose(2,0,1)

        # resize ref image to original size
        target_event_show = target_image.squeeze(0).permute(1,2,0).to('cpu').numpy() 
        # target_event_show = cv2.resize(target_event_show, (W, H), cv2.INTER_LINEAR)
        target_event_show = target_event_show.transpose(2,0,1)


        #VIS
        # plot_event(ref_event_show, line_pred_ref, target_event_show, line_score_target, H,W) #numpy  10,H,W
        
        # Exclude the target segments with endpoints in the clip border
        target_valid_mask = valid_mask
        target_clip_valid_mask = np.ones((H,W), dtype=np.float)
        # if erode_border_margin > 0:
        #     target_clip_valid_mask = binary_erosion(target_clip_valid_mask, iterations=erode_border_margin).astype(np.float)
        # target_valid_mask = target_valid_mask * target_clip_valid_mask
        
        line_segments_target = line_pred_target

        # target_valid_region_mask1 = target_valid_mask[line_segments_target[:, 0, 1].astype(np.int), line_segments_target[:, 0, 0].astype(np.int)] == 1.
        # target_valid_region_mask2 = target_valid_mask[line_segments_target[:, 1, 1].astype(np.int), line_segments_target[:, 1, 0].astype(np.int)] == 1.
        # target_valid_region_mask = target_valid_region_mask1 * target_valid_region_mask2
        # line_segments_target = line_segments_target[target_valid_region_mask, :]

        # VIS
        plot_name = 'ref_and_target'
        plot_event(ref_event_show, line_pred_ref, target_event_show, line_segments_target, H,W, plot_name)

        # TODO: 将 line_segments_target 反变换回原视图
        # Warp target line segments to ref
        line_unwarped = []
        
        line_pred_warped_tensor = torch.tensor(line_segments_target)  #N,2,2
        
        for i in range(len(line_pred_warped_tensor)):
            l_tensor = line_pred_warped_tensor[i].unsqueeze(dim=0)
            single_line_unwarped:torch.Tensor =transform_points(H_inv_tensor, l_tensor.cpu())
            line_unwarped.append(single_line_unwarped)

        if line_unwarped:   
            line_unwarped = np.concatenate(line_unwarped)
    
        line_segments_target_warped = line_unwarped
        # Filter out the out-of-border segments in ref view (True => keep)
        image_size = (512,512)
        boundary_mask = np.sum(np.sum((line_segments_target_warped < 0).astype(np.int), axis=-1), axis=-1)
        boundary_mask += np.sum((line_segments_target_warped[:, :, 0] >= image_size[0]-1).astype(np.int), axis=-1)
        boundary_mask += np.sum((line_segments_target_warped[:, :, 1] >= image_size[1]-1).astype(np.int), axis=-1)
        boundary_mask = (boundary_mask == 0)
        line_segments_target_warped = line_segments_target_warped[boundary_mask, :]
        # Filter out the out of valid_mask segments in taget view (True => keep)
        ref_valid_mask = np.ones(image_size, dtype=np.float)
        valid_region_mask1 = ref_valid_mask[line_segments_target_warped[:, 0, 0].astype(np.int), line_segments_target_warped[:, 0, 1].astype(np.int)] == 1.
        valid_region_mask2 = ref_valid_mask[line_segments_target_warped[:, 1, 0].astype(np.int), line_segments_target_warped[:, 1, 1].astype(np.int)] == 1.
        valid_region_mask = valid_region_mask1 * valid_region_mask2
        line_segments_target_warped = line_segments_target_warped[valid_region_mask, :]
        
        # VIS 
        plot_name = 'warp target to ref'
        plot_event(ref_event_show, line_pred_ref, ref_event_show, line_segments_target_warped, H,W, plot_name)

        # erode_border=False,
        # erode_border_margin= 2
        
        # 计算一对图片对中的 rep 和 loc_error, 这里的 line_pred_target 是 unwarped to 原视图的标签坐标
        distance_metric= "sAP"   # #"sAP", "", "sAP_square"
        rep_results, loc_results = compute_metrics_a_pair(
            line_pred_ref, line_segments_target_warped,
            dist_tolerance_lst, distance_metric
        )

        # print(pd.DataFrame([rep_results]), '\n', pd.DataFrame([loc_results]))

        for dist in dist_tolerance_lst:
            repeatability_dict[dist][idx] = rep_results[dist]
            local_error_dict[dist][idx] = loc_results[dist]

        # wandb.log({ 'rep table': wandb.Table(dataframe=pd.DataFrame([repeatability_dict])) })
        # wandb.log({ 'loc table': wandb.Table(dataframe=pd.DataFrame([local_error_dict])) })

    # for dist in dist_tolerance_lst:
    #     print("\t Rep-%02d: %f \t   Loc-%02d: %f" % (
    #             dist, np.sum(repeatability_dict[dist][ :]) / \
    #             np.sum((repeatability_dict[dist][ :] > 0.).astype(np.int)), 
    #             dist, np.sum(local_error_dict[dist][ :]) / \
    #             np.sum((local_error_dict[dist][:] > 0.).astype(np.int))))
    
    # avg_repeatability_dict = {}
    # avg_local_error_dict = {}
    # for dist in dist_tolerance_lst:
    #     avg_repeatability_dict[dist] = np.sum(repeatability_dict[dist][ :]) / \
    #                 np.sum((repeatability_dict[dist][ :] > 0.).astype(np.int))
    #     avg_local_error_dict[dist] = np.sum(local_error_dict[dist][ :]) / \
    #                 np.sum((local_error_dict[dist][:] > 0.).astype(np.int))

    

    # wandb.log({ 'rep table(orthogonal_distance)': wandb.Table(dataframe=pd.DataFrame([avg_repeatability_dict])) })
    # wandb.log({ 'loc table(orthogonal_distance)': wandb.Table(dataframe=pd.DataFrame([avg_local_error_dict])) })
    
    





    

