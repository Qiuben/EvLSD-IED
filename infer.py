import os
import json
import scipy.io as sio
import cv2
import time
import torch
import tqdm
import matplotlib.pyplot as plt
import torch.utils.data as Data
from network.detector import WireframeDetector
from network.dataset import Dataset
from config.cfg import parse
import numpy as np

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
    

def plot_on_event(est_event, lines, filename):
    est_event = est_event.cpu().numpy().squeeze(0)
    pos_image = (est_event[9, :, :] * 255).astype(np.uint8)
    neg_image = (est_event[4, :, :] * 255).astype(np.uint8)
    pos_image = cv2.equalizeHist(pos_image)
    neg_image = cv2.equalizeHist(neg_image)
    event = np.concatenate((neg_image[..., None], np.zeros((512, 512, 1)), pos_image[..., None]), axis=-1)
    plt.imshow(event)
    for pts in lines:
        pts = pts - 0.5
        # plt.plot(pts[:, 0], pts[:, 1], color="blue", linewidth=0.5)
        # plt.scatter(pts[:, 0], pts[:, 1], color="#FF0000", s=1.5, edgecolors="none", zorder=5)
        plt.plot(pts[:, 0], pts[:, 1], color="yellow", linewidth=2)
        plt.scatter(pts[:, 0], pts[:, 1], color="#33FFFF", s=1.5, edgecolors="none", zorder=5)

    plt.savefig('/home/lihao/data2/5.23拍摄/line_pred_seq/dvSave-2023_05_23_10_41_34/'+filename, dpi=400, bbox_inches=0)
    
    plt.close()

def plot_on_image(image, line_pred, filename, img_save_folder):
    height, width = image.shape[:2]
    fig = plt.figure()
    fig.set_size_inches(width / height, 1, forward=False)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.xlim([-0.5, width - 0.5])
    plt.ylim([height - 0.5, -0.5])
    plt.imshow(image[:, :, ::-1])
    for pts in line_pred:
        pts = pts - 0.5
        # plt.plot(pts[:, 0], pts[:, 1], color="blue", linewidth=0.5)
        # plt.scatter(pts[:, 0], pts[:, 1], color="#FF0000", s=1.5, edgecolors="none", zorder=5)
        plt.plot(pts[:, 0], pts[:, 1], color="orange", linewidth=0.8)
        plt.scatter(pts[:, 0], pts[:, 1], color="#33FFFF", s=2.0, edgecolors="none", zorder=5)

    plt.savefig(os.path.join(img_save_folder, filename), dpi=height, bbox_inches=0)
    

def infer_single_event(event, model, filename, img_save_folder):
    model.eval()
    annotations = None
    event= event.to(device)
    event = event.unsqueeze(0)
    
    outputs = model(event, annotations)
    for output in outputs:
        # Save image
        if len(output['line_pred']):
            line_pred = output['line_pred'].detach().cpu().numpy()
            line_score = output['line_score'].detach().cpu().numpy()
            mask = line_score > cfg.score_thresh
            line_pred = line_pred[mask]
            line_score = line_score[mask]
            filename = filename + '.png'
            
            image_path = img_save_folder + '/' + filename
            image = cv2.imread(image_path)
            # plot_on_event(event, line_pred, filename)
            plot_on_image(image, line_pred, filename, img_save_folder)

def images_to_video(path, seq_name):
    img_array = []
    
    imgList = os.listdir(path)
    imgList.sort(key=lambda x: int(x.split('.')[0])) 
    for count in range(0, len(imgList)): 
        filename = imgList[count]
        img = cv2.imread(path +'/'+ filename)
        
        if img is None:
            print(filename + " is error!")
            continue
        img_array.append(img)

    height, width, layers = img.shape
    size = (width, height)
    fps = 15  # 设置每帧图像切换的速度
    video_save_folder = '/home/lihao/data2/5.23拍摄/line_pred_show_on_recon_img/'
    out = cv2.VideoWriter(video_save_folder + seq_name+'.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
 
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

if __name__ == '__main__':
    # Parameter
    cfg = parse()
   
    # Use GPU or CPU
    use_gpu = cfg.gpu >= 0 and torch.cuda.is_available()
    device = torch.device(f'cuda:{cfg.gpu}' if use_gpu else 'cpu')
    print('use_gpu: ', use_gpu)

    # Load model
    model = WireframeDetector(cfg, backbone_in_channels = 10).to(device)
    
    model_filename = os.path.join(cfg.model_path, 'student_58.4.pkl')
    checkpoint = torch.load(model_filename, map_location=device)
    if 'model' in checkpoint.keys():
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    
    # est_root = '/home/lihao/data2/真实数据集初步测试/3.11-6.11/EST/'  # 404
    # img_save_folder = '/home/lihao/data2/真实数据集初步测试/3.11-6.11/pred/'
    # # if not os.path.exists(img_save_folder):
    # #     os.mkdir(img_save_folder)
    # for event_file in tqdm.tqdm(os.listdir(est_root)):
    #     filename = event_file.split('.')[0]
        
    root = '/home/lihao/Desktop/论文图/extreme_environment/'
    EST_name = 'dvSave-2023_06_19_15_41_52.npz'
    filename =  EST_name.split('.')[0] 
    with np.load(os.path.join(root, EST_name)) as events:
        event = events['event'].astype(np.float32)


    event = cv2.resize(event, (512, 512), cv2.INTER_LINEAR)
    event = torch.from_numpy(event).permute(2, 0, 1)


    infer_single_event(event, model, filename, root)
