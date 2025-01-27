import os
import json
import scipy.io as sio
import cv2
import time
import torch
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as Data
from network.detector import WireframeDetector
from network.dataset import Dataset
from config.cfg import parse
from metric.eval_mAPJ import eval_mAPJ
from metric.eval_sAP import eval_sAP
from util.helpers import *

import warnings
warnings.filterwarnings('ignore')


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


def save_lines(image, lines, filename, plot=False):
    height, width = image.shape[:2]

    fig = plt.figure()
    fig.set_size_inches(width / height, 1, forward=False)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.xlim([-0.5, width - 0.5])
    plt.ylim([height - 0.5, -0.5])
    plt.imshow(image[:, :, ::-1])
    for pts in lines:
        pts = pts - 0.5
        # plt.plot(pts[:, 0], pts[:, 1], color="blue", linewidth=0.5)
        # plt.scatter(pts[:, 0], pts[:, 1], color="#FF0000", s=1.5, edgecolors="none", zorder=5)
        plt.plot(pts[:, 0], pts[:, 1], color="orange", linewidth=0.5)
        plt.scatter(pts[:, 0], pts[:, 1], color="#33FFFF",
                    s=1.2, edgecolors="none", zorder=5)

    plt.savefig(filename, dpi=height, bbox_inches=0)
    if plot:
        plt.show()
    plt.close()


def show_event(est_event):
    plt.xlim(512)
    plt.ylim(512)
    pos_image = (est_event[9, :, :] * 255).astype(np.uint8)
    neg_image = (est_event[4, :, :] * 255).astype(np.uint8)
    pos_image = cv2.equalizeHist(pos_image)
    neg_image = cv2.equalizeHist(neg_image)
    image = np.concatenate((neg_image[..., None], np.zeros(
        (512, 512, 1)), pos_image[..., None]), axis=-1)
    plt.imshow(image)
    plt.show()


def test(model, loader, cfg, device):
    # Test
    model.eval()

    results = []
    start = time.time()

    for images, annotations in tqdm.tqdm(loader, desc='test: '):
        images, annotations = images.to(device), to_device(annotations, device)

        outputs = model(images, annotations)

        for output in outputs:
            # Save image
            # if cfg.save_image:
            if len(output['line_pred']):
                line_pred = output['line_pred'].detach().cpu().numpy()
                line_score = output['line_score'].detach().cpu().numpy()
                filename = output['filename']

            #     src_file = os.path.join(cfg.dataset_path, 'images', filename)
            #     dst_file = os.path.join(cfg.output_path, 'images', filename)

            #     image = cv2.imread(src_file)

                mask = line_score > cfg.score_thresh
                line_pred = line_pred[mask]
                line_score = line_score[mask]
            #     save_lines(image, line_pred, dst_file)

            for k in output.keys():
                if isinstance(output[k], torch.Tensor):
                    output[k] = output[k].tolist()
            results.append(output)

    end = time.time()
    # evaluate
    with open(os.path.join(cfg.output_path, 'result.json'), 'w') as f:
        json.dump(results, f)

    print(f'FPS: {len(loader) / (end - start):.1f}')

    gt_file = os.path.join(cfg.dataset_path, 'E-WHU/test.json')
    pred_file = os.path.join(cfg.output_path, 'result.json')
    mAPJ, P, R = eval_mAPJ(gt_file, pred_file)
    msAP, P, R, sAP = eval_sAP(gt_file, pred_file, cfg)
    prWhiteBlack(
        f'metric: {sAP[0]:.1f} | {sAP[1]:.1f} | {sAP[2]:.1f} | {msAP:.1f} | {mAPJ:.1f}')


if __name__ == '__main__':
    # Parameter
    cfg = parse()
    os.makedirs(cfg.output_path, exist_ok=True)
    os.makedirs(cfg.figure_path, exist_ok=True)

    # Use GPU or CPU
    use_gpu = cfg.gpu >= 0 and torch.cuda.is_available()
    device = torch.device(f'cuda:{cfg.gpu}' if use_gpu else 'cpu')

    modal = cfg.modal

    # Load model
    if modal == 'RGB':
        model = WireframeDetector(cfg, backbone_in_channels = 3).to(device)
    else:
        model = WireframeDetector(cfg, backbone_in_channels = 10).to(device)
  
    model_filename = os.path.join(cfg.model_path, cfg.model_name)

    checkpoint = torch.load(model_filename, map_location=device)
    if 'model' in checkpoint.keys():
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)

    # Load dataset
    dataset = Dataset(cfg, split='test', mode='val', dataset='E-WHU')
    loader = Data.DataLoader(dataset=dataset, batch_size=cfg.test_batch_size,
                             num_workers=cfg.num_workers, shuffle=False, collate_fn=dataset.collate)

    # Test network
    test(model, loader, cfg, device)
