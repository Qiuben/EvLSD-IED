import sys
sys.path.append('..')
import os
import numpy as np
import json
import time
import matplotlib.pyplot as plt
from config.cfg import parse
from metric.eval_metric import calc_msAP, calc_sAP, plot_pr_curve


def eval_sAP(gt_file, pred_file, lim, bins, cfg=None):
    with open(gt_file, 'r') as f:
        gt_annotations = json.load(f)
    with open(pred_file, 'r') as f:
        pred_annotations = json.load(f)

    
    arr = np.linspace(lim[0], lim[1], bins + 1)
    msAPs = np.zeros(bins)
    for j in range(bins):
        line_gts, line_preds, line_scores, im_ids = [], [], [], []
        id = 0
        for i, (gt_ann, pred_ann) in enumerate(zip(gt_annotations, pred_annotations)):
            sx, sy = 128.0 / gt_ann['width'], 128.0 / gt_ann['height']

            flow = np.asarray(gt_ann['flow'])
            blur = np.linalg.norm(flow, axis=-1).mean()
            blur = np.clip(blur, a_min=lim[0], a_max=lim[1] - 1e-4)
            if blur < arr[j] or blur >= arr[j + 1]:
                continue

            line_gt = np.asarray(gt_ann['lines']).reshape(-1, 2, 2)
            line_gt[:, :, 0] = line_gt[:, :, 0] * sx
            line_gt[:, :, 1] = line_gt[:, :, 1] * sy
            line_gts.append(line_gt)

            line_pred = np.asarray(pred_ann['line_pred'])
            line_score = np.asarray(pred_ann['line_score'])
            if len(line_pred):
                line_pred[:, :, 0] = line_pred[:, :, 0] * sx
                line_pred[:, :, 1] = line_pred[:, :, 1] * sy
                line_preds.append(line_pred)
                line_scores.append(line_score)
                im_ids.append(np.array([id] * line_pred.shape[0], dtype=np.int32))
            id += 1

        if len(line_gts) == 0:
            continue

        line_preds = np.concatenate(line_preds)
        line_scores = np.concatenate(line_scores)
        im_ids = np.concatenate(im_ids)
        indices = np.argsort(-line_scores)
        line_preds = line_preds[indices]
        im_ids = im_ids[indices]

        msAP, P, R, sAP = calc_msAP(line_gts, line_preds, im_ids, [5.0, 10.0, 15.0])
        msAPs[j] = msAP

    
    return msAPs
    


if __name__ == "__main__":
    # Parameter
    os.chdir('..')
    cfg = parse()
    os.makedirs(cfg.figure_path, exist_ok=True)

    # Path
    gt_file = os.path.join(cfg.dataset_path, 'test.json')
    # FE-WHU
    if cfg.dataset_name=='FE-WHU':
        test_file_HAWP = 'output/FE-WHU-7.1.0.20/result.json'
        test_file_FE_HAWP = 'output/FE-WHU-7.2.2.1114/result.json'
    if cfg.dataset_name=='FE-Wireframe':
        test_file_HAWP = 'output/FE-Wireframe-7.1.0.20/result.json'
        test_file_FE_HAWP = 'output/FE-Wireframe-7.2.0.1114/result.json'

    bins = 15
    lims=[0, 60]
    start = time.time()
    msAPs_HAWP= eval_sAP(gt_file, test_file_HAWP, lims, bins, cfg)
    msAPs_FE_HAWP = eval_sAP(gt_file, test_file_FE_HAWP, lims, bins, cfg)
    end = time.time()

    arr = np.linspace(lims[0], lims[1], bins + 1)
    blurs = arr[:-1]
    width = lims[1] / bins
    plt.figure()
    plt.xlim(lims)
    plt.xticks(arr)
    plt.xlabel('Blur (pixel)', fontsize=14)
    plt.ylabel('msAP ($\%$)', fontsize=14)
    # bar = plt.bar(blurs + width / 2, msAPs, width=width, color='deepskyblue', edgecolor='k')
    # plt.bar_label(bar, [f'{val:.0f}' if val >= 0.5 else '' for val in msAPs])
    # if cfg.dataset_name=='FE-Wireframe':
    blurs = blurs[1:-1]
    msAPs_HAWP = msAPs_HAWP[1:-1]
    msAPs_FE_HAWP = msAPs_FE_HAWP[1:-1]

    l1=plt.plot(blurs+ width / 2,msAPs_HAWP,'r',label='HAWP')
    l2=plt.plot(blurs+ width / 2,msAPs_FE_HAWP,'g',label='FE-HAWP')
    plt.plot(blurs+ width / 2,msAPs_HAWP,'ro-',blurs+ width / 2,msAPs_FE_HAWP,'g+-')
    plt.legend()
    plt.savefig(os.path.join(cfg.figure_path, f'msAP-blur_HAWP-FE_HAWP.pdf'), format='pdf', bbox_inches='tight')
    plt.show()
    print('Time: %f s' % (end - start))
