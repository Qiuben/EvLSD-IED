# Learning to Detect Line Segment from Events
This repository contains the official PyTorch implementation of the paper: “Learning to Detect Line Segment from Events”

## Table of Contents
- [Installation](#installatuion)
- [Testing Pre-trained Models](#testing-pre-trained-models)
- [Downloading the Dataset](#downloading-the-dataset)
- [Distillation](#distillation)


## Installation
For the ease of reproducibility, you are suggested to install miniconda (or anaconda if you prefer) before following executing the following commands.

`git clone https://github.com/Qiuben/EvLSD-IED`

`cd EvLSD-IED`


## Testing Pre-trained Models
You can download the pretrained model on E-wirferame 
from xxx.

## Downloading the Dataset
You can download the synthetic dataset Ewireframe as well as the real-scene datset RE-LSD from . 

## Distillation 

#### Step 1 : training the teacher 
`python train.py -m image -d E-wireframe`

#### Step 2 : training the student 
`python train_IED.py -m event -d E-wireframe`
