import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

from mrcnn_unet.config import Config
from mrcnn_unet import utils
import mrcnn_unet.model as modellib
from mrcnn_unet import visualize
from mrcnn_unet.model import log
import tensorflow as tf
from mrcnn_unet.dataset import MyDataset
from mrcnn_unet.dataset import buildVIA_Anno

class spkConfig(Config):
    NAME = "spkConfig"  # Override in sub-classes
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # steps/Tensorboard update
    STEPS_PER_EPOCH = 10
    # Number of validation steps to run at the end of every training epoch.
    VALIDATION_STEPS = 5
    BACKBONE = "resnet101"  #"resnet50"
    # Only useful if you supply a callable to BACKBONE. Should compute
    # the shape of each layer of the FPN Pyramid.
    # See model.compute_backbone_shapes
    COMPUTE_BACKBONE_SHAPE = None     #也可以直接設定 [[160,160],[80,80],[40,40]]
    # The strides of each layer of the FPN Pyramid. These values
    # are based on a Resnet101 backbone.
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]   #resnet50 101 架構下 不變
    # Size of the fully-connected layers in the classification graph
    FPN_CLASSIF_FC_LAYERS_SIZE = 1024
    # Size of the top-down layers used to build the feature pyramid         
    TOP_DOWN_PYRAMID_SIZE = 256                                                     #最底層的特徵大小  之後需要修改
    # Number of classification classes (including background)
    NUM_CLASSES = 1+2  # Override in sub-classes                                    #之後需要修改
    # Length of square anchor side in pixels                              
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)                                     #之後需要修改
    # Ratios of anchors at each cell (width/height)
    # A value of 1 represents a square anchor, and 0.5 is a wide anchor
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]
    # Anchor stride
    # If 1 then anchors are created for each cell in the backbone feature map.
    # If 2, then anchors are created for every other cell, and so on.
    RPN_ANCHOR_STRIDE = 1                                                           #也可以調整這個來減少錨框數量
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7
    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256                                               #之後需要調整
    # ROIs kept after tf.nn.top_k and before non-maximum suppression
    PRE_NMS_LIMIT = 6000                                                            #proposal layer 使用到的 resnet101(220k取6k)
    # ROIs kept after non-maximum suppression (training and inference)
    POST_NMS_ROIS_TRAINING = 2000                                                   #之後需要調整
    POST_NMS_ROIS_INFERENCE = 1000                                                  #之後需要調整
    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask                  #之後需要調整
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1024                                                            #之後需要調整
    # Minimum scaling ratio. Checked after MIN_IMAGE_DIM and can force further
    # up scaling. For example, if set to 2 then images are scaled up to double
    # the width and height, or more, even if MIN_IMAGE_DIM doesn't require it.
    # However, in 'square' mode, it can be overruled by IMAGE_MAX_DIM.
    IMAGE_MIN_SCALE = 0
    # Number of color channels per image. RGB = 3, grayscale = 1, RGB-D = 4
    IMAGE_CHANNEL_COUNT = 3
    # Image mean (RGB)
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])
    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 200
    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.3
    # Percent of positive ROIs used to train classifier/mask heads
    ROI_POSITIVE_RATIO = 0.33

    # Pooled ROIs
    POOL_SIZE = 7                                                                       #之後需要調整
    MASK_POOL_SIZE = 14                                                                 #之後需要調整
    # Shape of output mask
    # To change this you also need to change the neural network mask branch
    MASK_SHAPE = [28, 28]                                                               #之後需要調整

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 100
    # Bounding box refinement standard deviation for RPN and final detections.
    RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    # Max number of final detections
    DETECTION_MAX_INSTANCES = 100
    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.7

    # Learning rate and momentum
    # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
    # weights to explode. Likely due to differences in optimizer
    # implementation.
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9
    # Weight decay regularization
    WEIGHT_DECAY = 0.0001
    # Loss weights for more precise optimization.
    # Can be used for R-CNN training setup.
    LOSS_WEIGHTS = {                                           
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.,
        #-----------------------
        "u_net_mask_loss":1.                                   #之後需要修改
        #-----------------------
    }
    # Use RPN ROIs or externally generated ROIs for training
    USE_RPN_ROIS = True
    # Train or freeze batch normalization layers
    #     None: Train BN layers. This is the normal mode
    #     False: Freeze BN layers. Good when using a small batch size
    #     True: (don't use). Set layer in training mode even when predicting
    TRAIN_BN = False  # Defaulting to False since batch size is often small
    # Gradient norm clipping
    GRADIENT_CLIP_NORM = 5.0



if __name__ == "__main__":
    config = spkConfig()  
    config.display()
    
    #--------------custom------------------
    imageDir=r"C:\Users\tnt\Desktop\spk_env\imageDir"
    VIA_csvPath=r"C:\Users\tnt\Desktop\spk_env\mask.csv"
    classDict={0:'eye',1:'spk'}
    # classDict={0:'flour',1:'saltAndPerrer',2:'breadCrust',3:'breadCrumb',4:'glueAndFillement'}
    trainAnnoList,valAnnoList=buildVIA_Anno(imageDir,VIA_csvPath,classDict,shuffle=True,splitRate=0.9)
    
    dataset_train = MyDataset(imageDir,trainAnnoList,classDict)
    dataset_val = MyDataset(imageDir,valAnnoList,classDict)
    #--------------------------------------


    # Root directory of the project
    ROOT_DIR = os.path.abspath("./")
    # Import Mask RCNN
    sys.path.append(ROOT_DIR)  # To find local version of the library
    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")

    # Local path to trained weights file
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
    # Download COCO trained weights from Releases if needed
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    model = modellib.MaskRCNN(mode="training", config=config,
                            model_dir=MODEL_DIR)
    # model.load_weights(COCO_MODEL_PATH, by_name=True,
    #                     exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
    #                                 "mrcnn_bbox", "mrcnn_mask"])

    model.train(dataset_train, dataset_val, 
                learning_rate=config.LEARNING_RATE, 
                epochs=1, 
                layers='heads')