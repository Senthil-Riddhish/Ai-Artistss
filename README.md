# intel-oneAPI

#### Team Name - Ai Artistss
#### Problem Statement - Object Detection For Autonomous Vehicles
#### Team Leader Email - riddhishwar.s2020@vitstudent.ac.in

## Introduction
"Enhanced Object Detection with Intel: Leveraging Intel Technologies for Accurate and Efficient YOLOv5 Model"

In our project, we have harnessed the power of Intel technologies to enhance the capabilities of the YOLOv5 algorithm for object detection. By leveraging Intel's optimized libraries and frameworks, such as Intel oneDAL, Intel optimized PyTorch, and the SYCL/DPC++ libraries, we have achieved superior performance, accuracy, and efficiency in our object detection model. This integration enables us to process data faster, optimize resource utilization, and streamline post-processing steps, leading to robust and real-time object detection for autonomous vehicle applications.

## **Table of Contents**
 - Purpose
 - A Brief of the Prototype
 - Architecture Diagram
 - Flow Diagram
 - Expected Input-Output
 - Dataset and Annotations
 - Folder Structure
 - Tech Stack
      - Optimized software components
      - Optimized Solution setup
 - Step-by-Step Code Execution Instructions
      - Installation
 - Overview
      - Training
 - Output Videos From Our Model
 - Output Graph
 - Object Detection for Autonomus vehicles
 - What I Learned
 
 <!-- Purpose -->
## Purpose
The purpose of our project is to leverage Intel technologies to enhance the YOLOv5 algorithm for object detection in the context of autonomous vehicles. By utilizing Intel oneDAL, Intel optimized PyTorch, and the SYCL/DPC++ libraries, we aim to achieve improved performance, accuracy, and efficiency in detecting and classifying objects in real-time. Our goal is to provide a reliable and effective solution for autonomous vehicles to detect and respond to various objects and obstacles on the road, ensuring enhanced safety and efficiency in autonomous driving systems.

<!-- A Brief of the Prototype -->
## 📜 A Brief of the Prototype:
Our prototype's real-time object detection and distance recognition features are meant to make self-driving cars safer and more efficient. By leveraging the power of Intel technologies and frameworks, we've created a robust system that combines advanced computer vision algorithms and deep learning models.Intel AI Analytics Toolkit, featuring optimised deep learning frameworks like PyTorch and TensorFlow, powers the prototype. Intel-optimized libraries like oneDNN and oneDAL were used to train and infer deep learning models. This helps us locate items around the car.
<!-- Architecture -->
## 📜 Architecture Diagram: 
![Screenshot 2023-06-09 181317](https://github.com/Senthil-Riddhish/Ai-Artistss/assets/82893678/93c796c8-9a13-4a41-ac06-c343c20f3ebf)
<!--Flow_Diagram -->
## 📜 Flow Diagram:
![Screenshot 2023-06-09 181606](https://github.com/Senthil-Riddhish/Ai-Artistss/assets/82893678/8c6dc9d9-6955-4908-b164-c00728b7738f)

### Expected Input-Output

**Input**                                 | **Output** |
| :---: | :---: |
| The expected input for our object detection system is a video stream or a series of images captured by the sensors installed on the autonomous vehicle. These inputs contain visual data representing the surroundings of the vehicle.          |  The expected output of our system is the detection and classification of objects present in the input data. This includes bounding box coordinates for each detected object and their corresponding class labels. The output can be visualized by drawing bounding boxes around the detected objects on the input images or video stream, along with the associated class labels and confidence scores. This information provides valuable insights for decision-making in autonomous vehicle operations, allowing the vehicle to detect and respond to various objects and obstacles on the road effectively.

## Dataset and Annotations
   ![Screenshot 2023-06-09 225151](https://github.com/Senthil-Riddhish/Ai-Artistss/assets/82893678/dcd478b6-cae1-4eb8-9c28-f658ef331900)
   *Dataset*: https://universe.roboflow.com/roboflow-gw7yv/self-driving-car/dataset/3
   
<!--Folder-->
## 🍞 Folder Structure
```
root_directory/
└── client
|   ├── public
|   │   ├── favicon.ico
|   │   ├── index.html
|   │   ├── logo192.png
|   │   ├── logo512.png
|   │   ├── manifest.json
|   │   ├── robots.txt
|   └── src
|        ├── assets
|        ├── components
|        |   ├── Header
|        |   ├── Footer
|        |   ├── Hero
|        |   ├── NavBar
|        |   ├── ...
|        ├── App.js
|        ├── index.js
|        └── ...
└── client
    └──export
    ├── images
    │   ├── VOCdevkit
    │   │   ├── VOC2007
    │   │   │   ├── Annotations
    │   │   │   ├── ImageSets
    │   │   │   │   ├── Layout
    │   │   │   │   ├── Main
    │   │   │   │   └── Segmentation
    │   │   │   ├── JPEGImages
    │   │   │   ├── SegmentationClass
    │   │   │   └── SegmentationObject
    │   │   └── VOC2012
    │   │       ├── Annotations
    │   │       ├── ImageSets
    │   │       │   ├── Action
    │   │       │   ├── Layout
    │   │       │   ├── Main
    │   │       │   └── Segmentation
    │   │       ├── JPEGImages
    │   │       ├── SegmentationClass
    │   │       └── SegmentationObject
    │   ├── test2007
    │   ├── train2007
    │   ├── train2012
    │   ├── val2007
    │   └── val2012
    └── labels
        ├── test2007
        ├── train2007
        ├── train2012
        ├── val2007
        └── val2012
yolov5/
└── data
    ├── classify
    ├── models
    ├── segement
    ├── data
    │   ├── custom_data.yaml
```
## 📜 Tech Stack: 
### **Optimized software components**
| **Package**                | **Intel® Distribution for Python**                
| :---                       | :---                            
| OpenCV     | opencv-python=4.5.5.64
| NumPy               | numpy=1.23.4
| PyTorch              | pytorch=1.13.0
| Intel® Extension for PyTorch         | intel-extension-for-pytorch=1.13.0                              
| Intel® Neural Compressor         | neural-compressor=2.0   
| Intel® Distribution of OpenVINO™ | openvino-dev=2022.3.0

### **Optimized Solution setup**

**YAML file**                                 | **Environment Name** |  **Configuration** |
| :---: | :---: | :---: |
`env/intel/intel-pt.yml`             | `intel-pt` | Python v3.9 with Intel® Extension for PyTorch v1.13.0 |

## Step-by-Step Code Execution Instructions:
## Installation
  Clone repo and install [requirements.txt](https://github.com/Senthil-Riddhish/Ai-Artistss/blob/main/yolov5/requirements.txt) in a
   [**Python>=3.7.0**](https://www.python.org/) environment, including
   [**PyTorch>=1.7**](https://pytorch.org/get-started/locally/).

   ```bash
   git clone https://github.com/ultralytics/yolov5  # clone
   cd yolov5
   pip install -r requirements.txt  # install
   ```
   <summary>Training</summary>

The commands below reproduce YOLOv5 [COCO](https://github.com/ultralytics/yolov5/blob/master/data/scripts/get_coco.sh)
results. [Models](https://github.com/ultralytics/yolov5/tree/master/models)
and [datasets](https://github.com/ultralytics/yolov5/tree/master/data) download automatically from the latest
YOLOv5 [release](https://github.com/ultralytics/yolov5/releases). Training times for YOLOv5n/s/m/l/x are
1/2/4/6/8 days on a V100 GPU ([Multi-GPU](https://docs.ultralytics.com/yolov5/tutorials/multi_gpu_training) times faster). Use the
largest `--batch-size` possible, or pass `--batch-size -1` for
YOLOv5 [AutoBatch](https://github.com/ultralytics/yolov5/pull/5092). Batch sizes shown for V100-16GB.
### Just get into the yolov5 folder and execute this commnd for training the model
```bash
python train.py --data custom_dataset.yaml --epochs 300 --weights '' --cfg yolov5n.yaml  --batch-size 128
                                                                 yolov5s                    64
                                                                 yolov5m                    40
                                                                 yolov5l                    24
                                                                 yolov5x                    16
```
## Overview
### Training
```
 usage: train.py [-h] [--weights WEIGHTS] [--cfg CFG] [--data DATA] [--hyp HYP] [--epochs EPOCHS]
                [--batch-size BATCH_SIZE] [--imgsz IMGSZ] [--rect] [--resume [RESUME]] [--nosave] [--noval]
                [--noautoanchor] [--noplots] [--evolve [EVOLVE]] [--bucket BUCKET] [--cache [CACHE]] [--image-weights]
                [--device DEVICE] [--multi-scale] [--single-cls] [--optimizer {SGD,Adam,AdamW}] [--sync-bn]
                [--workers WORKERS] [--project PROJECT] [--name NAME] [--exist-ok] [--quad] [--cos-lr]
                [--label-smoothing LABEL_SMOOTHING] [--patience PATIENCE] [--freeze FREEZE [FREEZE ...]]
                [--save-period SAVE_PERIOD] [--seed SEED] [--local_rank LOCAL_RANK] [--entity ENTITY]
                [--upload_dataset [UPLOAD_DATASET]] [--bbox_interval BBOX_INTERVAL] [--artifact_alias ARTIFACT_ALIAS]
                [--intel INTEL] [--bf16]

optional arguments:
  -h, --help            show this help message and exit
  --weights WEIGHTS     initial weights path
  --cfg CFG             model.yaml path
  --data DATA           dataset.yaml path
  --hyp HYP             hyperparameters path
  --epochs EPOCHS       total training epochs
  --batch-size BATCH_SIZE
                        total batch size for all GPUs, -1 for autobatch
  --imgsz IMGSZ, --img IMGSZ, --img-size IMGSZ
                        train, val image size (pixels)
  --rect                rectangular training
  --resume [RESUME]     resume most recent training
  --nosave              only save final checkpoint
  --noval               only validate final epoch
  --noautoanchor        disable AutoAnchor
  --noplots             save no plot files
  --evolve [EVOLVE]     evolve hyperparameters for x generations
  --bucket BUCKET       gsutil bucket
  --cache [CACHE]       --cache images in "ram" (default) or "disk"
  --image-weights       use weighted image selection for training
  --device DEVICE       cuda device, i.e. 0 or 0,1,2,3 or cpu
  --multi-scale         vary img-size +/- 50%
  --single-cls          train multi-class data as single-class
  --optimizer {SGD,Adam,AdamW}
                        optimizer
  --sync-bn             use SyncBatchNorm, only available in DDP mode
  --workers WORKERS     max dataloader workers (per RANK in DDP mode)
  --project PROJECT     save to project/name
  --name NAME           save to project/name
  --exist-ok            existing project/name ok, do not increment
  --quad                quad dataloader
  --cos-lr              cosine LR scheduler
  --label-smoothing LABEL_SMOOTHING
                        Label smoothing epsilon
  --patience PATIENCE   EarlyStopping patience (epochs without improvement)
  --freeze FREEZE [FREEZE ...]
                        Freeze layers: backbone=10, first3=0 1 2
  --save-period SAVE_PERIOD
                        Save checkpoint every x epochs (disabled if < 1)
  --seed SEED           Global training seed
  --local_rank LOCAL_RANK
                        Automatic DDP Multi-GPU argument, do not modify
  --entity ENTITY       Entity
  --upload_dataset [UPLOAD_DATASET]
                        Upload data, "val" option
  --bbox_interval BBOX_INTERVAL
                        Set bounding-box image logging interval
  --artifact_alias ARTIFACT_ALIAS
                        Version of dataset artifact to use
  --intel INTEL, -i INTEL
                        To Enable Intel Optimization set to 1, default 0
  --bf16                Enable only on Intel® Fourth Gen Xeon, BF16
 ```
**Command to run training**
```sh
cd src/yolov5 # should be inside the "yolov5" cloned repo folder ignore if already in "yolov5"
python train.py --weights yolov5s.pt --data ./data/custom_data.yaml --epochs 10 -i 0
```
### Inference
```
usage: run_inference.py [-h] [-c CONFIG] [-d DATA_YAML] [-b BATCHSIZE] [-w WEIGHTS] [-i INTEL] [-int8inc INT8INC] [-qw QUANT_WEIGHTS] [-si SAVE_IMAGE] [-sip SAVE_IMAGE_PATH]

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        Yaml file for quantizing model, default is "./deploy.yaml"
  -d DATA_YAML, --data_yaml DATA_YAML
                        Absolute path to the data yaml file containing configurations
  -b BATCHSIZE, --batchsize BATCHSIZE
                        batchsize for the dataloader....default is 1
  -w WEIGHTS, --weights WEIGHTS
                        Model Weights ".pt" format
  -i INTEL, --intel INTEL
                        Run Intel optimization (Ipex) when 1....default is 0
  -int8inc INT8INC      Run INC quantization when 1....default is 0
  -qw QUANT_WEIGHTS, --quant_weights QUANT_WEIGHTS
                        Quantization Model Weights folder containing ".pt" format model
  -si SAVE_IMAGE, --save_image SAVE_IMAGE
                        Save images in the save image path specified if 1, default 0
  -sip SAVE_IMAGE_PATH, --save_image_path SAVE_IMAGE_PATH
                        Path to save images after post processing/ detected results
```
## Output Videos From Our Model
https://drive.google.com/drive/folders/1WSYknCyP11lraXsjaTRNCGYYS4ejhJXf
## Output Graph:
![Screenshot 2023-06-10 000404](https://github.com/Senthil-Riddhish/Ai-Artistss/assets/82893678/ab9b5551-0320-4dcc-8a72-ae2f22b591c1)
![Screenshot 2023-06-10 000424](https://github.com/Senthil-Riddhish/Ai-Artistss/assets/82893678/0efc3c16-dbe4-44b1-9682-2593a2bb00b3)
![Screenshot 2023-06-10 000437](https://github.com/Senthil-Riddhish/Ai-Artistss/assets/82893678/008a3ee6-a0aa-462e-b652-687ca17c4676)

## Object Detection for Autonomus vehicles:
![Screenshot 2023-06-10 002233](https://github.com/Senthil-Riddhish/Ai-Artistss/assets/82893678/338041a1-85f8-4d7f-8e1b-53502972de2d)

## What I Learned:
- Familiarity with YOLO Architecture: YOLO (You Only Look Once) is a popular and efficient object detection algorithm. Implementing YOLOv8 helped us familiarise ourselves with its architecture, including the backbone network (e.g., Darknet), feature extraction layers, and detection layers.

- Preprocessing and Augmentation Techniques: Object detection often requires preprocessing and data augmentation to improve model performance. While implementing YOLOv8, We have learnt about various techniques such as resizing, normalization, data augmentation (e.g., random cropping, flipping), and handling object annotations.
Hyperparameter Tuning: YOLOv8 has several hyperparameters that affect the model's performance, such as learning rate, batch size, anchor sizes, and confidence thresholds.
- We have learnt and gained experience in tuning these hyperparameters to optimize the model's accuracy and efficiency.
We explored a range of Intel's software development tools and libraries, including the AI analytics toolkit and its libraries.
Using Intel® AI Analytics Toolkits we were able to enhance performance speed in training data.
- Brainstormed with novel algorithms for different kinds of object detection specific to autonomous vehicles.Implemented Object detection alongwith distance mapping of nearby objects in order to prevent collisions. We've been able to custom label/annotate the objects in detection.
- We explored a range of Intel's software development tools and libraries, including the AI analytics toolkit and its libraries.
Using Intel® AI Analytics Toolkits we were able to enhance performance speed in training data. Brainstormed with novel algorithms for different kinds of object detection specific to autonomous vehicles. Implemented Object detection alongwith distance mapping of nearby objects in order to prevent collisions.
We've been able to custom label/annotate the objects in detection.
