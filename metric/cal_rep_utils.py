


import cv2
import math
import math
import numpy as np
import warnings
warnings.filterwarnings('ignore')


##### functions for calculating the distances #####

# Given a list of line segments and a list of points (2D or 3D coordinates),
# compute the orthogonal projection of all points on all lines.
# This returns the 1D coordinates of the projection on the line,
# as well as the list of orthogonal distances.
def project_point_to_line(line_segs, points):
    # Compute the 1D coordinate of the points projected on the line
    dir_vec = (line_segs[:, 1] - line_segs[:, 0])[:, None]
    coords1d = (((points[None] - line_segs[:, None, 0]) * dir_vec).sum(axis=2)
                / np.linalg.norm(dir_vec, axis=2) ** 2)
    # coords1d is of shape (n_lines, n_points)
    
    # Compute the orthogonal distance of the points to each line
    projection = line_segs[:, None, 0] + coords1d[:, :, None] * dir_vec
    dist_to_line = np.linalg.norm(projection - points[None], axis=2)

    return coords1d, dist_to_line

# Given a list of segments parameterized by the 1D coordinate of the endpoints
# compute the overlap with the segment [0, 1]
def get_segment_overlap(seg_coord1d):
    seg_coord1d = np.sort(seg_coord1d, axis=-1)
    overlap = ((seg_coord1d[..., 1] > 0) * (seg_coord1d[..., 0] < 1)
               * (np.minimum(seg_coord1d[..., 1], 1)
                  - np.maximum(seg_coord1d[..., 0], 0)))
    return overlap

def get_overlap_orth_line_dist(line_seg1, line_seg2, min_overlap=0.5):
    n_lines1, n_lines2 = len(line_seg1), len(line_seg2)

    # Compute the average orthogonal line distance
    coords_2_on_1, line_dists2 = project_point_to_line(
        line_seg1, line_seg2.reshape(n_lines2 * 2, -1))
    line_dists2 = line_dists2.reshape(n_lines1, n_lines2, 2).sum(axis=2)
    coords_1_on_2, line_dists1 = project_point_to_line(
        line_seg2, line_seg1.reshape(n_lines1 * 2, -1))
    line_dists1 = line_dists1.reshape(n_lines2, n_lines1, 2).sum(axis=2)
    line_dists = (line_dists2 + line_dists1.T) / 2

    # Compute the average overlapping ratio
    coords_2_on_1 = coords_2_on_1.reshape(n_lines1, n_lines2, 2)
    overlaps1 = get_segment_overlap(coords_2_on_1)
    coords_1_on_2 = coords_1_on_2.reshape(n_lines2, n_lines1, 2)
    overlaps2 = get_segment_overlap(coords_1_on_2).T
    overlaps = (overlaps1 + overlaps2) / 2

    # Enforce a max line distance for line segments with small overlap
    low_overlaps = overlaps < min_overlap
    line_dists[low_overlaps] = np.amax(line_dists)
    return line_dists


def compute_distances(line_segments_anchor, line_segments_cand, dist_tolerance_lst, distance_metric="sAP", group_num=1000):
    if not distance_metric in ["sAP", "sAP_square", "orthogonal_distance"]:
        raise ValueError("[Error] The specified distance metric is not supported.")
    
    # Compute distance matrix
    if distance_metric == "sAP" or distance_metric == "sAP_square":
        num_anchor_seg = line_segments_anchor.shape[0]
        min_dist_lst = []
        if num_anchor_seg > group_num:
            num_iter = math.ceil(num_anchor_seg / group_num)
            for iter_idx in range(num_iter):
                if iter_idx == num_iter - 1:
                    if distance_metric == "sAP":
                        diff = (((line_segments_anchor[iter_idx*group_num:, None, :, None] - line_segments_cand[None, :, None]) ** 2).sum(-1)) ** 0.5
                    else:
                        diff = (((line_segments_anchor[iter_idx*group_num:, None, :, None] - line_segments_cand[None, :, None]) ** 2).sum(-1))
                    # np.minimum返回两者当中较小的
                    diff = np.minimum(
                        diff[:, :, 0, 0] + diff[:, :, 1, 1], diff[:, :, 0, 1] + diff[:, :, 1, 0]
                    )
                else:
                    if distance_metric == "sAP":
                        diff = (((line_segments_anchor[iter_idx*group_num:(iter_idx+1)*group_num, None, :, None] - line_segments_cand[:, None]) ** 2).sum(-1)) ** 0.5
                    else:
                        diff = (((line_segments_anchor[iter_idx*group_num:(iter_idx+1)*group_num, None, :, None] - line_segments_cand[:, None]) ** 2).sum(-1))
                    diff = np.minimum(
                        diff[:, :, 0, 0] + diff[:, :, 1, 1], diff[:, :, 0, 1] + diff[:, :, 1, 0]
                    )
                # Compute reference to target correctness
                try:
                    anchor_cand_min_dist_ = np.min(diff, 1)  #第二个维度上的最小值
                except:
                    # if diff is empty
                    anchor_cand_min_dist_ = np.ones([diff.shape[0], 1]) * (dist_tolerance_lst[-1] + 100.)
                min_dist_lst.append(anchor_cand_min_dist_)
            anchor_cand_min_dist = np.concatenate(min_dist_lst)
        else:
            if distance_metric == "sAP":

                diff = (((line_segments_anchor[:, None, :, None] - line_segments_cand[None, :, None]) ** 2).sum(-1)) ** 0.5
            else:
                diff = (((line_segments_anchor[:, None, :, None] - line_segments_cand[None, :, None]) ** 2).sum(-1))
            diff = np.minimum(
                diff[:, :, 0, 0] + diff[:, :, 1, 1], diff[:, :, 0, 1] + diff[:, :, 1, 0]
            )
            # Compute reference to target correctness
            try:
                anchor_cand_min_dist = np.min(diff, 1)
            except:
                # if diff is empty
                anchor_cand_min_dist = np.ones([diff.shape[0], 1]) * (dist_tolerance_lst[-1] + 100.)

    elif distance_metric == "orthogonal_distance":
        if 0 in line_segments_anchor.shape or 0 in line_segments_cand.shape:
            if 0 in line_segments_cand.shape:
                diff = np.ones([line_segments_anchor.shape[0], 1]) * (dist_tolerance_lst[-1] + 100.)
            else:
                diff = np.ones([1, 1]) * (dist_tolerance_lst[-1] + 100.)
        else:
            diff = get_overlap_orth_line_dist(
                line_segments_anchor,
                line_segments_cand,
                min_overlap=0.5
            )
        
        # Compute reference to target correctness
        try:
            anchor_cand_min_dist = np.min(diff, 1)
            
        except:
            # if diff is empty
            anchor_cand_min_dist = np.ones([diff.shape[0], 1]) * (dist_tolerance_lst[-1] + 100.)
        # import ipdb; ipdb.set_trace()
    
    return anchor_cand_min_dist



##### homography related functions #####
# Sample a random valid homography.
def sample_homography(
        shape, perspective=True, scaling=True, rotation=True, translation=True,
        n_scales=5, n_angles=25, scaling_amplitude=0.1, perspective_amplitude_x=0.1,
        perspective_amplitude_y=0.1, patch_ratio=0.5, max_angle=math.pi/2,
        allow_artifacts=False, translation_overflow=0.):
    """
    Computes the homography transformation between a random patch in the original image
    and a warped projection with the same image size.
    As in `tf.contrib.image.transform`, it maps the output point (warped patch) to a
    transformed input point (original patch).
    The original patch, which is initialized with a simple half-size centered crop, is
    iteratively projected, scaled, rotated and translated.

    Arguments:
        shape: A rank-2 `Tensor` specifying the height and width of the original image.
        perspective: A boolean that enables the perspective and affine transformations.
        scaling: A boolean that enables the random scaling of the patch.
        rotation: A boolean that enables the random rotation of the patch.
        translation: A boolean that enables the random translation of the patch.
        n_scales: The number of tentative scales that are sampled when scaling.
        n_angles: The number of tentatives angles that are sampled when rotating.
        scaling_amplitude: Controls the amount of scale.
        perspective_amplitude_x: Controls the perspective effect in x direction.
        perspective_amplitude_y: Controls the perspective effect in y direction.
        patch_ratio: Controls the size of the patches used to create the homography.
        max_angle: Maximum angle used in rotations.
        allow_artifacts: A boolean that enables artifacts when applying the homography.
        translation_overflow: Amount of border artifacts caused by translation.

    Returns:
        homo_mat: A numpy array of shape `[1, 3, 3]` corresponding to the homography transform.
        selected_scale: The selected scaling factor.
    """
    # Convert shape to ndarry
    if not isinstance(shape, np.ndarray):
        shape = np.array(shape)

    # Corners of the output image
    pts1 = np.array([[0., 0.], [0., 1.], [1., 1.], [1., 0.]])
    # Corners of the input patch
    margin = (1 - patch_ratio) / 2
    pts2 = margin + np.array([[0, 0], [0, patch_ratio],
                             [patch_ratio, patch_ratio], [patch_ratio, 0]])

    # Random perspective and affine perturbations
    if perspective:
        if not allow_artifacts:
            perspective_amplitude_x = min(perspective_amplitude_x, margin)
            perspective_amplitude_y = min(perspective_amplitude_y, margin)

        # normal distribution with mean=0, std=perspective_amplitude_y/2
        perspective_displacement = np.random.normal(0., perspective_amplitude_y/2, [1])
        h_displacement_left = np.random.normal(0., perspective_amplitude_x/2, [1])
        h_displacement_right = np.random.normal(0., perspective_amplitude_x/2, [1])
        pts2 += np.stack([np.concatenate([h_displacement_left, perspective_displacement], 0),
                          np.concatenate([h_displacement_left, -perspective_displacement], 0),
                          np.concatenate([h_displacement_right, perspective_displacement], 0),
                          np.concatenate([h_displacement_right, -perspective_displacement], 0)])

    # Random scaling
    # sample several scales, check collision with borders, randomly pick a valid one
    if scaling:
        scales = np.concatenate([[1.], np.random.normal(1, scaling_amplitude/2, [n_scales])], 0)
        center = np.mean(pts2, axis=0, keepdims=True)
        scaled = (pts2 - center)[None, ...] * scales[..., None, None] + center
        # all scales are valid except scale=1
        if allow_artifacts:
            valid = np.array(range(n_scales))
        # Chech the valid scale
        else:
            # ipdb.set_trace()
            valid = np.where(np.all((scaled >= 0.) & (scaled < 1.), (1, 2)))[0]
        
        # No valid scale found => recursively call
        if valid.shape[0] == 0:
            return sample_homography(
                shape, perspective, scaling, rotation, translation,
                n_scales, n_angles, scaling_amplitude, 
                perspective_amplitude_x, perspective_amplitude_y, patch_ratio, 
                max_angle, allow_artifacts, translation_overflow
            )

        idx = valid[np.random.uniform(0., valid.shape[0], ()).astype(np.int32)]
        pts2 = scaled[idx]

        # Additionally save and return the selected scale.
        selected_scale = scales[idx]

    # Random translation
    if translation:
        t_min, t_max = np.min(pts2, axis=0), np.min(1 - pts2, axis=0)
        if allow_artifacts:
            t_min += translation_overflow
            t_max += translation_overflow
        pts2 += (np.stack([np.random.uniform(-t_min[0], t_max[0], ()),
                           np.random.uniform(-t_min[1], t_max[1], ())]))[None, ...]

    # Random rotation
    # sample several rotations, check collision with borders, randomly pick a valid one
    if rotation:
        angles = np.linspace(-max_angle, max_angle, n_angles)
        # in case no rotation is valid
        angles = np.concatenate([[0.], angles], axis=0)
        center = np.mean(pts2, axis=0, keepdims=True)
        rot_mat = np.reshape(np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)], axis=1), [-1, 2, 2])
        rotated = np.matmul(
                np.tile((pts2 - center)[None, ...], [n_angles+1, 1, 1]),
                rot_mat) + center
        if allow_artifacts:
            valid = np.array(range(n_angles))  # all angles are valid, except angle=0
        else:
            valid = np.where(np.all((rotated >= 0.) & (rotated < 1.), axis=(1, 2)))[0]
        
        if valid.shape[0] == 0:
            return sample_homography(
                shape, perspective, scaling, rotation, translation,
                n_scales, n_angles, scaling_amplitude, 
                perspective_amplitude_x, perspective_amplitude_y, patch_ratio, 
                max_angle, allow_artifacts, translation_overflow
            )

        idx = valid[np.random.uniform(0., valid.shape[0], ()).astype(np.int32)]
        pts2 = rotated[idx]

    # ipdb.set_trace()
    # Rescale to actual size
    shape = shape[::-1].astype(np.float32)  # different convention [y, x]
    pts1 *= shape[None, ...]
    pts2 *= shape[None, ...]

    def ax(p, q): return [p[0], p[1], 1, 0, 0, 0, -p[0] * q[0], -p[1] * q[0]]

    def ay(p, q): return [0, 0, 0, p[0], p[1], 1, -p[0] * q[1], -p[1] * q[1]]

    a_mat = np.stack([f(pts1[i], pts2[i]) for i in range(4) for f in (ax, ay)], axis=0)
    p_mat = np.transpose(np.stack([[pts2[i][j] for i in range(4) for j in range(2)]], axis=0))
    # ipdb.set_trace()
    homo_vec, _, _, _ = np.linalg.lstsq(a_mat, p_mat, rcond=None)

    # Compose the homography vector back to matrix
    homo_mat = np.concatenate([homo_vec[0:3, 0][None, ...],
                                homo_vec[3:6, 0][None, ...],
                                np.concatenate((homo_vec[6], homo_vec[7], [1]),
                                axis=0)[None, ...]], axis=0)

    # ToDo: figure out if we don't apply scaling, how to return
    # return homo_mat, selected_scale
    return homo_mat, 1

# ToDo: what if there are junctions on the image border that are eroded by the border margin?
def compute_valid_mask(image_size, homography, border_margin, valid_mask=None):
    # warp the mask
    if valid_mask is None:
        initial_mask = np.ones(image_size)
    else:
        initial_mask = valid_mask
    # 经过了变换之后的 mask
    mask = cv2.warpPerspective(initial_mask, homography, (image_size[1], image_size[0]),
                               flags=cv2.INTER_NEAREST)

    # Optionally perform erosion
    if border_margin > 0:
        # cv2.MORPH_ELLIPSE 是椭圆形， 这里有三种形状可以选择， (border_margin*2, )*2是指定形状的尺寸
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (border_margin*2, )*2)
        mask = cv2.erode(mask, kernel)
    
    # Perform dilation if border_margin is negative
    if border_margin < 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,  (abs(int(border_margin))*2, )*2)
        mask = cv2.dilate(mask, kernel)

    return mask

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
