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
from PIL import Image
import torchvision
import torch.nn.functional as F


from termcolor import colored

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
    


if __name__ == '__main__':
    # Parameter
    cfg = parse()
   
    # Use GPU or CPU
    use_gpu = cfg.gpu >= 0 and torch.cuda.is_available()
    device = torch.device(f'cuda:{cfg.gpu}' if use_gpu else 'cpu')
    print('use_gpu: ', use_gpu)

    #### Load student model
    student_model = WireframeDetector(cfg, backbone_in_channels= 10).to(device)
    
    student_model_path = os.path.join(cfg.model_path, 'no_distill.pkl')
    student_checkpoint = torch.load(student_model_path, map_location=device)
    if 'model' in student_checkpoint.keys():
        state_dict_s = student_checkpoint['model']
    else:
        state_dict_s = student_checkpoint
    student_model.load_state_dict(state_dict_s)

    #### Load teacher model
    teacher_model = WireframeDetector(cfg, backbone_in_channels= 3).to(device)
    
    teacher_model_path  = os.path.join(cfg.model_path, 'teacher.pkl')
    teacher_checkpoint = torch.load(teacher_model_path, map_location=device)
    if 'model' in teacher_checkpoint.keys():
        state_dict_t = teacher_checkpoint['model']
    else:
        state_dict_t = teacher_checkpoint
    teacher_model.load_state_dict(state_dict_t)

    
    est_root = '/data/xinya/e-wireframe/EST-10'  # 404
    img_save_folder = 'affinity_save/'
    if not os.path.exists(img_save_folder):
        os.mkdir(img_save_folder)
    

    #推断单张
    # event_file = '00077583.npz'
    event_file = '00101020.npz'
    # event_file = '00053542.npz'
    filename = event_file.split('.')[0]
    with np.load(os.path.join(est_root, event_file)) as events:
        event = events['event'].astype(np.float32)

    image_path = '/data/xinya/e-wireframe/images/'+str(filename)+'.png'
    image = Image.open(image_path).convert('RGB')

    # 同时保存图像 和 EST
    image = image.resize((349,349))
    image.save(img_save_folder + str(filename)+'.png')

    class Compose(object):
        def __init__(self, transforms):
            self.transforms = transforms

        def __call__(self, image, event, ann=None):
            for transform in self.transforms:
                image, event, ann = transform(image, event, ann)
            return image, event, ann
        
    class ResizeImage(object):
        def __init__(self):
            self.image_height = 512
            self.image_width = 512

        def __call__(self, image, event, ann=None):
            image = torchvision.transforms.functional.resize(image, (self.image_height, self.image_width))
            if event is not None:
                event = cv2.resize(event, (self.image_width, self.image_height), cv2.INTER_LINEAR)
            if ann is None:
                return image, event
            else:
                return image, event, ann
            
    class ToTensor(object):
        def __call__(self, image, event, ann=None):
            image = torchvision.transforms.functional.to_tensor(image)
            if event is not None:
                event = torch.from_numpy(event).permute(2, 0, 1)
            if ann is None:
                return image, event
            else:
                for key, val in ann.items():
                    if isinstance(val, np.ndarray):
                        ann[key] = torch.from_numpy(val)
                return image, event, ann

    class Normalize(object):
        def __init__(self, mean, std):
            self.mean = mean
            self.std = std

        def __call__(self, image, event, ann=None):
            image = torchvision.transforms.functional.normalize(image, mean=self.mean, std=self.std)
            if ann is None:
                return image, event
            else:
                return image, event, ann
        
    mean = [0.43031373, 0.40718431, 0.38698431]
    std=  [0.08735294, 0.08676078, 0.09109412]
    transforms = Compose([
                ResizeImage(),
                ToTensor(),
                Normalize(mean, std)
            ])
    with open('/data/xinya/e-wireframe/train.json', 'r') as f:
        annotations = json.load(f)
    ann = annotations[0]
    image, event, ann = transforms(image, event, ann)
    

    # 切换teacher/ student
    role = 'student'

    if role =='teacher':
        model = teacher_model
    else:
        model = student_model
    model.eval()

    event, image = event.to(device), image.to(device)
    event, image= event.unsqueeze(0), image.unsqueeze(0)

    if role =='teacher':
        save_results, feature, _ = model(image, annotations)
    else:
        save_results, feature , _ = model(event, annotations)
    
    B, C, H, W = feature.shape
    
    resize_shape =[64, 64]
    feature = F.interpolate(feature, size=resize_shape, mode="bilinear")


    feature = feature.reshape(B, C, -1)

    print(colored(feature.shape, 'grey'))
    affinity = torch.bmm(feature.permute(0, 2, 1), feature)
    print(colored(affinity.shape, 'blue'))

    # 将亲和力图转换为NumPy数组
    affinity_map_np = affinity.detach().cpu().squeeze().numpy()  #4096, 4096
    # affinity_map_np = affinity_map_np[:512, :512]

    # 显示亲和力图
    plt.imshow(affinity_map_np, cmap='viridis')
    plt.show()

    # feature_vis = feature[0].detach().cpu().numpy()
    # feature_avg = np.mean(feature_vis, axis = 0)
    # plt.imshow(feature_avg, cmap='viridis')
    plt.axis ('off')
    plt.savefig(img_save_folder+str(role)+'_'+ str(filename)+'.png', bbox_inches='tight', pad_inches = 0)
 