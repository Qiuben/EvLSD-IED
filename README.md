# EvLSD-IED: Event-Based Line Segment Detection with Image-to-Event Distillation
This repository contains the official PyTorch implementation of the paper: “EvLSD-IED: Event-Based Line Segment Detection with Image-to-Event Distillation”

## Table of Contents
- [Installation](#installatuion)
- [Testing Pre-trained Models](#testing-pre-trained-models)
- [Downloading the Dataset](#downloading-the-dataset)
- [Distillation](#distillation)


## Installation
We have trained and tested our models on CUDA 11.1, Python 3.8.0, torch 1.10.1.

For ease of reproducibility, you are suggested to install miniconda (or anaconda if you prefer) before following executing the following commands.

`git clone https://github.com/Qiuben/EvLSD-IED`

`cd EvLSD-IED`

`pip install -r requirements.txt`

## Testing Pre-trained Models
You can download the pretrained model on E-wirferame 
from [OneDrive](https://1drv.ms/f/c/93289205239bc375/EoSWLjyUd4JDgzARyahZtTcBjfqtTmDchmW_w_GWYltV8A?e=vkLnVt).

## Downloading the Dataset
You can download the synthetic dataset E-wireframe as well as the real-scene datset RE-LSD from [OneDrive](https://1drv.ms/f/c/93289205239bc375/EoSWLjyUd4JDgzARyahZtTcBjfqtTmDchmW_w_GWYltV8A?e=vkLnVt). 
You can directly download the EST representation or download the raw event data and convert it to frame representations using the event2frame.py
## Distillation 

#### Step 1 : training the teacher 
`python train.py -m image -d E-wireframe`

#### Step 2 : training the student 
`python train_IED.py -m event -d E-wireframe`
