import os
import numpy as np
import copy
import random
import json
import shutil
import tqdm
from torch import nn
from sklearn.metrics import confusion_matrix
import torch
import torch.utils.data as Data
import torch.nn.functional as F

from network.detector import WireframeDetector
from network.dataset import Dataset
from config.cfg import parse
from metric.eval_mAPJ import eval_mAPJ
from metric.eval_sAP import eval_sAP

import os
import cv2
import wandb
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)
import ipdb

from termcolor import colored

import warnings
warnings.filterwarnings('ignore')

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

    plt.show()
    plt.savefig('tmp/student/'+ str(filename), dpi=300, bbox_inches=0)
    # if plot:
    #     plt.show()
    # plt.close()

    # wandb.log({str(filename): plt})

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

def plot_image(input_vis, annotations, batchsize, plot_name):
    input_vis = input_vis.detach().cpu().numpy()

    plt.figure(figsize=(12,4), clear= True)
    for i in range(batchsize):
        plt.subplot(1, batchsize, i+1)
        input = input_vis[i]
        plt.imshow(input.transpose(1,2,0))
        ann_vis = annotations[i]
        junctions = ann_vis['junc']
        sx = 512 / 128
        sy = 512 / 128
        junctions[:, 0] *=  sx
        junctions[:, 1] *=  sy
        edges_positive = ann_vis['edges_positive']
        gt = torch.cat([junctions[edges_positive[:, 0]].unsqueeze(dim=1), junctions[edges_positive[:, 1]].unsqueeze(dim=1)], dim=1)
        gt = gt.detach().cpu()
        
        for pts in np.array(gt):
            pts = pts - 0.5
            plt.plot(pts[:, 0], pts[:, 1], color="red", linewidth=1.5)
            plt.scatter(pts[:, 0], pts[:, 1], color="#33FFFF", s=2.5, edgecolors="none", zorder=5)

    wandb.log({str(plot_name): plt})
   
def FeatureDistillLoss(teacher_feature, student_feature):
    device = 'cuda:0'
    # 计算逐像素特征
    # loss = torch.nn.MSELoss()

    # 如果对特征图归一化
    # teacher_feature = (teacher_feature - teacher_feature.min()) / (teacher_feature.max() - teacher_feature.min())
    # student_feature = (student_feature - student_feature.min()) / (student_feature.max() - student_feature.min())

    feature_loss_map = torch.nn.functional.l1_loss(student_feature, teacher_feature, reduction='none')
    # 1. Tanh
    edge_weight = torch.tanh(teacher_feature)
    # edge_weight = teacher_feature
    # 2. Sigmoid
    # edge_weight = (torch.sigmoid(teacher_feature) - 0.5) * 2
    feature_loss = torch.mean(edge_weight * feature_loss_map) * 0.3
    # 3. Gt map
    # feature_loss = torch.mean(gt_mask * feature_loss_map) * 3

    return feature_loss

def AffinityDistillLoss(teacher_feature, student_feature):
    B, C, H, W = student_feature.shape
    
    resize_shape = [64, 64]
    student_feature = F.interpolate(student_feature, size=resize_shape, mode="bilinear")
    teacher_feature = F.interpolate(teacher_feature, size=resize_shape, mode="bilinear")

    feature_teacher = teacher_feature.reshape(B, C, -1)
    # print(colored(feature_teacher.permute(0, 2, 1).shape, 'red'))
    # print(colored(feature_teacher.shape, 'blue'))
    teacher_affinity = torch.bmm(feature_teacher.permute(0, 2, 1), feature_teacher)

    feature_student = student_feature.reshape(B, C, -1)
    student_affinity = torch.bmm(feature_student.permute(0, 2, 1), feature_student)
    
    affinity_map_loss = F.l1_loss(student_affinity, teacher_affinity, reduction='mean') / B
    return affinity_map_loss
 
def distill(student_model, teacher_model, loader, cfg, device):
    # Option
    optimizer = torch.optim.Adam(student_model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, amsgrad=True)
    if cfg.last_epoch != -1:
        print('Load pretrained model...')
        checkpoint_file = os.path.join(cfg.model_path, 'student.pkl')
        checkpoint = torch.load(checkpoint_file, map_location=device)
        student_model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, last_epoch=cfg.last_epoch)

    # Train
    step = (cfg.last_epoch + 1) * len(loader['train'].dataset) // cfg.train_batch_size + 1
    
    best_sAP = [0 for _ in range(5)]
    best_state_dict = None
    
    for epoch in range(cfg.last_epoch + 2, cfg.num_epochs + 1):
        # Train
        student_model.train()
        teacher_model.eval()
        for images, events, annotations in tqdm.tqdm(loader['train'], desc='epoch:{}'.format(str(epoch))):
            images, events, annotations = images.to(device), events.to(device), to_device(annotations, device)
            
            torch.cuda.empty_cache()
            # train mode
            loss_dict, labels, scores, output, stu_feature = student_model(events, annotations)
            # infer mode
            _, tea_feature = teacher_model(images, annotations)

            task_spec_loss = sum([cfg.loss_weights[k] * loss_dict[k] for k in cfg.loss_weights.keys()])

            EA_loss = FeatureDistillLoss(tea_feature, stu_feature)

            AF_loss = AffinityDistillLoss(tea_feature, stu_feature)

            loss = task_spec_loss + 100 * EA_loss + AF_loss 

            wandb.log({'task specific loss': task_spec_loss.item(),
                       'edge-alignment loss': 100 * EA_loss,
                       'affinity-alignment': AF_loss
                    })

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # log loss
            lr = scheduler.get_last_lr()[0]
            # score = student_scores.detach().cpu().numpy() > 0.5
            # label = labels.detach().cpu().numpy() > 0.5
            # tn, fp, fn, tp = confusion_matrix(label, score).ravel()

            loss_md, loss_dis, loss_res, loss_joff, loss_jloc, loss_pos, loss_neg = \
                loss_dict['loss_md'], loss_dict['loss_dis'], loss_dict['loss_res'], loss_dict['loss_joff'], \
                loss_dict['loss_jloc'], loss_dict['loss_pos'], loss_dict['loss_neg']
            
            # select the variable you want to visualize
            wandb.log({'loss': loss.item(), 'lr':lr})

            step += 1
        scheduler.step()


        if epoch % cfg.save_freq == 0:
            # Save model
            save_path = os.path.join(cfg.model_path, f'{epoch:03d}')
            os.makedirs(save_path, exist_ok=True)

            print(save_path)
            checkpoint_file = os.path.join(cfg.model_path, 'student.pkl')
            checkpoint = {'model': student_model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(checkpoint, checkpoint_file)

            # Val
            student_model.eval()
            results = []
            for images, events, annotations in tqdm.tqdm(loader['val'], desc='val: '):
                images, events, annotations = images.to(device), events.to(device),to_device(annotations, device)
                student_results, _,= student_model(events, annotations)
                for output in student_results:
                    for k in output.keys():
                        if isinstance(output[k], torch.Tensor):
                            output[k] = output[k].tolist()
                    results.append(output)

            with open(os.path.join(save_path, 'result.json'), 'w') as f:
                json.dump(results, f)

            gt_file = os.path.join(cfg.dataset_path, cfg.dataset_name, 'test.json')
            pred_file = os.path.join(save_path, 'result.json')


            mAPJ, PJ, RJ = eval_mAPJ(gt_file, pred_file)
            wandb.log({'mAPJ': mAPJ})
            print(f'mAPJ: {mAPJ:.1f} | {PJ:.1f} | {RJ:.1f}')

            msAP, P, R, sAP = eval_sAP(gt_file, pred_file)
            wandb.log({'msAP': msAP})
            print(f'msAP: {msAP:.1f} | {P:.1f} | {R:.1f} | {sAP[0]:.1f} | {sAP[1]:.1f} | {sAP[2]:.1f}')

            shutil.rmtree(save_path)

            if msAP > best_sAP[3]:
                best_sAP = [mAPJ, PJ, RJ, msAP, P, R, *sAP]
                best_state_dict = copy.deepcopy(student_model.state_dict())

            msg = f'best msAP: {best_sAP[0]:.1f} | {best_sAP[1]:.1f} | {best_sAP[2]:.1f} | ' \
                  f'{best_sAP[3]:.1f} | {best_sAP[4]:.1f} | {best_sAP[5]:.1f} | ' \
                  f'{best_sAP[6]:.1f} | {best_sAP[7]:.1f} | {best_sAP[8]:.1f}'
            print(msg)

        # Save best model
        torch.save(best_state_dict, 'model/student.pkl')


if __name__ == '__main__':
    os.environ["WANDB_API_KEY"] = '43d81c680e2727082f8588d0d76d494f0f6d7e0f'

    # Parameter
    cfg = parse()
    os.makedirs(cfg.model_path, exist_ok=True)
    config = {
        'initial_lr': cfg.lr, #4e-4
        'step_size':cfg.step_size,
        'num_epochs':cfg.num_epochs,
        'train_batch_size': cfg.train_batch_size, # 16
        'test_batch_size': cfg.test_batch_size # 8
    }

    wandb.init(project='EvLSD-IED',
           name = 'train student', 
           config = config)
    
    # Use GPU or CPU
    use_gpu = cfg.gpu >= 0 and torch.cuda.is_available()
    device = torch.device('cuda')
    print('use_gpu: ', use_gpu)


    # Seed
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = False
    if use_gpu:
        torch.cuda.manual_seed_all(cfg.seed)

    # Load model
    teacher_model = WireframeDetector(cfg, backbone_in_channels = 3).to(device)
    print('Load Teacher pretrained model...')
    checkpoint_file = os.path.join(cfg.model_path, 'teacher.pkl')
    checkpoint = torch.load(checkpoint_file, map_location=device)
    
    teacher_model.load_state_dict(checkpoint)

    student_model = WireframeDetector(cfg, backbone_in_channels = 10).to(device)

    # Load dataset
    train_dataset = Dataset(cfg, split='train', mode='train', dataset='E-wireframe')
    val_dataset = Dataset(cfg, split='test', mode='val',  dataset='E-wireframe')
    
    train_loader = Data.DataLoader(dataset = train_dataset, batch_size=cfg.train_batch_size,
                                   num_workers=cfg.num_workers, shuffle=True, collate_fn=train_dataset.collate)
    val_loader = Data.DataLoader(dataset=val_dataset, batch_size=cfg.test_batch_size,
                                 num_workers=cfg.num_workers, shuffle=False, collate_fn=val_dataset.collate)
    loader = {'train': train_loader, 'val': val_loader}

    # Train network

    distill(student_model, teacher_model, loader, cfg, device)
