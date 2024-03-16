import os
import argparse
import numpy as np
import cv2
from termcolor import colored

def plot_image_vstack(methods, paths, filenames, save_file):
    height = 512

    image = None
    for method, path in zip(methods, paths):
        print(colored('method:{}'.format(method), 'green')) 
        temp = None
        for filename in filenames:
            src_file = os.path.join(path, filename)

            print(colored(src_file, 'blue'))
            src = cv2.imread(src_file)
            try:
                w, h = src.shape[1], src.shape[0]
            except:
                print(src_file)
            dst = cv2.resize(src, (round(w * height / h), height))
            if temp is None:
                temp = dst.copy()
            else:
                padding = np.ones((dst.shape[0], 10, 3), np.uint8) * 255
                temp = np.hstack((temp, padding, dst))

        if image is None:
            image = temp.copy()
        else:
            temp = cv2.resize(temp, (image.shape[1], height))
            padding = np.ones((5, temp.shape[1], 3), np.uint8) * 255
            image = np.vstack((image, padding, temp))

    print(colored('save to {}'.format(save_file), 'yellow'))
    cv2.imwrite(save_file, image)


def plot_image_hstack(methods, paths, filenames, save_file):
    width = 640

    image = None
    for method, path in zip(methods, paths):
        print(method)
        temp = None
        for filename in filenames:
            src_file = os.path.join(path, filename)
            print(src_file)
            src = cv2.imread(src_file)
            if method == 'Event':
                for i in range(3):
                    src[:, :, i] = cv2.equalizeHist(src[:, :, i])
            w, h = src.shape[1], src.shape[0]
            dst = cv2.resize(src, (width, round(h * width / w)))
            if temp is None:
                temp = dst.copy()
            else:
                padding = np.ones((5, dst.shape[1], 3), np.uint8) * 255
                temp = np.vstack((temp, padding, dst))

        if image is None:
            image = temp.copy()
        else:
            temp = cv2.resize(temp, (width, image.shape[0]))
            padding = np.ones((temp.shape[0], 5, 3), np.uint8) * 255
            image = np.hstack((image, padding, temp))
    cv2.imwrite(save_file, image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_name', type=str, default='RE-LSD',
                        choices=['e-wireframe', 'RE-LSD'], help='dataset name')
    opts = parser.parse_args()
    dataset_name = opts.dataset_name
    os.makedirs('figure', exist_ok=True)

    methods = ['image_or_APS', 'EST', 'EC-LSD', 'E2VID-HAWP', 'EST-LCNN', 'EST-FClip', 'EvLSD', 'EvLSD-IED', 'GT']
    save_file = f'comparison/{dataset_name}.png'
    if dataset_name == 'e-wireframe':
        paths = [f'/data/xinya/e-wireframe/images',
                 f'ewireframe-EST',
                 f'../../LSD/result_cv/{dataset_name}-event',
                 f'../../HAWP/outputs/{dataset_name}',
                 f'../../EST-LCNN/outputs/images/{dataset_name}',
                 f'../../EST-FClip/outputs/images/{dataset_name}',
                 f'../output/images/{dataset_name}/ours',
                 f'../output/images/{dataset_name}/ours-IED', 
                 f'GT/{dataset_name}']
        filenames = ['00031608.png', '00031591.png', '00031841.png', '00031810.png', '00031723.png']
    else:  #  [f'../../LSD/result_cv/{dataset_name}-event',
        paths = [f'image_or_APS',
                 f'RE-LSD-EST',
                 f'../../LSD/result_cv/{dataset_name}-event',
                 f'../../HAWP/outputs/{dataset_name}',
                 f'../../EST-LCNN/outputs/F_RE-LSD/images',
                 f'../../EST-FClip/outputs/F_RE-LSD/images',
                 f'../output/images/{dataset_name}/ours',
                 f'../output/images/{dataset_name}/ours-IED', 
                 f'GT/{dataset_name}']
        # 第一行放图片， 第二行放事件
        # 过曝的 000084   000132   000250  000586
        # 暗光的 000353   2023_03_11_12_00_40   2023_03_11_12_01_57  2023_03_11_12_01_57  2023_06_23_15_26_08 2023_06_29_10_33_49
        filenames = ['000041.png', '000030.png', '000586.png', '000586.png', '2023_03_11_12_00_40.png', '2023_03_11_12_01_57.png']   #430 442

    plot_image_vstack(methods, paths, filenames, save_file)
