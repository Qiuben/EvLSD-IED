import os
import numpy as np
import copy
import random
import json
import shutil
import tqdm
from sklearn.metrics import confusion_matrix
import torch
import torch.utils.data as Data

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

    # plt.savefig(filename, dpi=height, bbox_inches=0)
    # if plot:
    #     plt.show()
    # plt.close()

    wandb.log({str(filename): plt})

def plot_event(est_events, annotations, batchsize, plot_name):  #numpy  10,H,W

    # ADD
    est_events = est_events.detach().cpu().numpy()

    plt.figure(figsize=(12,4), clear= True)
    for i in range(batchsize):
        plt.subplot(1, batchsize, i+1)
        est_event = est_events[i]
        
        pos_image = (est_event[9, :, :] * 255).astype(np.uint8)
        neg_image = (est_event[4, :, :] * 255).astype(np.uint8)
        pos_image = cv2.equalizeHist(pos_image)
        neg_image = cv2.equalizeHist(neg_image)
        event = np.concatenate((neg_image[..., None], np.zeros((512, 512, 1)), pos_image[..., None]), axis=-1)
        plt.imshow(event)
        ann = annotations[i]
        junctions = ann['junc']
        sx = 512 / 128
        sy = 512 / 128
        junctions[:, 0] *=  sx
        junctions[:, 1] *=  sy
        edges_positive = ann['edges_positive']
        gt = torch.cat([junctions[edges_positive[:, 0]].unsqueeze(dim=1), junctions[edges_positive[:, 1]].unsqueeze(dim=1)], dim=1)

        gt = gt.detach().cpu()
        for pts in np.array(gt):
            pts = pts - 0.5
            plt.plot(pts[:, 0], pts[:, 1], color="yellow", linewidth=1.5)
            plt.scatter(pts[:, 0], pts[:, 1], color="#33FFFF", s=2.5, edgecolors="none", zorder=5)

    wandb.log({str(plot_name): plt})

def train(model, loader, cfg, device):
    # Option
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, amsgrad=True)
    if cfg.last_epoch != -1:
        print('Load pretrained model...')
        checkpoint_file = os.path.join(cfg.model_path, cfg.model_name)
        checkpoint = torch.load(checkpoint_file, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, last_epoch=cfg.last_epoch)

    # Train
    step = (cfg.last_epoch + 1) * len(loader['train'].dataset) // cfg.train_batch_size + 1
    
    best_sAP = [0 for _ in range(5)]
    best_state_dict = None

    for epoch in range(cfg.last_epoch + 2, cfg.num_epochs + 1):
        # Train
        model.train()

        for images, annotations in tqdm.tqdm(loader['train'], desc='epoch {}: '.format(str(epoch))):
            #TODO  查看这里的事件和标签是否对应
            plot_name =  None
            vis_event = images.clone()
            # plot_event(vis_event, annotations, images.shape[0], plot_name)
            images, annotations = images.to(device), to_device(annotations, device)

            torch.cuda.empty_cache()
            loss_dict, labels, scores = model(images, annotations)
            loss = sum([cfg.loss_weights[k] * loss_dict[k] for k in cfg.loss_weights.keys()])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
        scheduler.step()

        # Visualize
            # if step % cfg.print_freq == 0:
        lr = scheduler.get_last_lr()[0]
        score = scores.detach().cpu().numpy() > 0.5
        label = labels.detach().cpu().numpy() > 0.5
        tn, fp, fn, tp = confusion_matrix(label, score).ravel()

        loss_md, loss_dis, loss_res, loss_joff, loss_jloc, loss_pos, loss_neg = \
            loss_dict['loss_md'], loss_dict['loss_dis'], loss_dict['loss_res'], loss_dict['loss_joff'], \
            loss_dict['loss_jloc'], loss_dict['loss_pos'], loss_dict['loss_neg']
        
        wandb.log({'loss': loss.item(), 'lr':lr, 'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn})
            

        if epoch % cfg.save_freq == 0:
            # Save model
            save_path = os.path.join(cfg.model_path, f'{epoch:03d}')
            os.makedirs(save_path, exist_ok=True)

            print(save_path)
            checkpoint_file = os.path.join(cfg.model_path, cfg.model_name)
            checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(checkpoint, checkpoint_file)

            # Val
            model.eval()

            results = []
            for images, annotations in tqdm.tqdm(loader['val'], desc='val: '):
                images, annotations = images.to(device), to_device(annotations, device)

                outputs = model(images, annotations)
                for output in outputs:

                    if len(output['line_pred']):
                        line_pred = output['line_pred'].detach().cpu().numpy()
                        line_score = output['line_score'].detach().cpu().numpy()
                        filename = output['filename']

                        src_file = os.path.join(cfg.dataset_path, cfg.dataset_name, 'images', filename)
                        dst_file = os.path.join(cfg.output_path, 'images', filename)

                        image = cv2.imread(src_file)

                        mask = line_score > cfg.score_thresh
                        line_pred = line_pred[mask]
                        line_score = line_score[mask]
                        # wandb 可视化测试结果
                        # save_lines(image, line_pred, dst_file)

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
            # print(f'mAPJ: {mAPJ:.1f} | {PJ:.1f} | {RJ:.1f}')

            msAP, P, R, sAP = eval_sAP(gt_file, pred_file)
            wandb.log({'msAP': msAP})
            # print(f'msAP: {msAP:.1f} | {P:.1f} | {R:.1f} | {sAP[0]:.1f} | {sAP[1]:.1f} | {sAP[2]:.1f}')

            shutil.rmtree(save_path)

            if msAP > best_sAP[3]:
                best_sAP = [mAPJ, PJ, RJ, msAP, P, R, *sAP]
                best_state_dict = copy.deepcopy(model.state_dict())

            msg = f'best msAP: {best_sAP[0]:.1f} | {best_sAP[1]:.1f} | {best_sAP[2]:.1f} | ' \
                  f'{best_sAP[3]:.1f} | {best_sAP[4]:.1f} | {best_sAP[5]:.1f} | ' \
                  f'{best_sAP[6]:.1f} | {best_sAP[7]:.1f} | {best_sAP[8]:.1f}'
            print(msg)

        # Save best model
        model_filename = 'model/'+ 'add_module.pkl'
        torch.save(best_state_dict, model_filename)
    wandb.save(model_filename)


if __name__ == '__main__':

    # Parameter
    cfg = parse()
    os.makedirs(cfg.model_path, exist_ok=True)
    config = {
        'initial_lr': cfg.lr,
        'step_size':cfg.step_size,
        'num_epochs':cfg.num_epochs,
        'train_batch_size': cfg.train_batch_size, # 8
        'test_batch_size': cfg.test_batch_size # 8
    }

    wandb.init(project='EvLSD-IED',
           name = 'exp1', 
           config = config)
    
    # Use GPU or CPU
    use_gpu = cfg.gpu >= 0 and torch.cuda.is_available()
    device = torch.device(f'cuda:{cfg.gpu}' if use_gpu else 'cpu')
    print('use_gpu: ', use_gpu)


    # Seed
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = False
    if use_gpu:
        torch.cuda.manual_seed_all(cfg.seed)
        
    # Load model
    if cfg.modal == 'event':
        model = WireframeDetector(cfg, backbone_in_channels = 10).to(device)
    else:
        model = WireframeDetector(cfg, backbone_in_channels = 3).to(device)
 
    # Load dataset
    train_dataset = Dataset(cfg, split='train', mode='train', dataset='e-wireframe')
    val_dataset = Dataset(cfg, split='test', mode='val',dataset='e-wireframe')
    print('the length of trainset:',len(train_dataset))
    mini_train_dataset, _ = torch.utils.data.random_split(dataset=train_dataset, lengths=[3000, 27000])
    
    # a_sample_loader = Data.DataLoader(dataset=train_dataset, sampler= [1, 501, 1001, 1501, 2001, 2501],
    #                                num_workers=cfg.num_workers, shuffle=False, collate_fn=train_dataset.collate)
    # for est, ann in a_sample_loader:
    #     est = est.cpu().numpy()
    #     plot_event(est, ann, cfg.train_batch_size, 'show all augmentation') 


    train_loader = Data.DataLoader(dataset= mini_train_dataset, batch_size=cfg.train_batch_size,
                                   num_workers=cfg.num_workers, shuffle=True, collate_fn=train_dataset.collate)
    val_loader = Data.DataLoader(dataset=val_dataset, batch_size=cfg.test_batch_size,
                                 num_workers=cfg.num_workers, shuffle=False, collate_fn=train_dataset.collate)
    loader = {'train': train_loader, 'val': val_loader}

    # Train network

    train(model, loader, cfg, device)
