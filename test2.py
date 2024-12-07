import warnings
warnings.filterwarnings('ignore')

# Import necessary libraries
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import yaml
import wandb
from PIL import Image
from ultralytics import YOLO
from IPython.display import Video

# 初始化 W&B
wandb.init(project="YOVOV8-T1", name="Experiment1", mode="offline")

# Load a pretrained YOLOv8n model from Ultralytics
#model = YOLO('yolov8n.pt')

# 从头开始创建一个新的YOLO模型
#model = YOLO('yolov8.yaml')

# 加载预训练的YOLO模型（推荐用于训练）
model = YOLO('yolov8n.pt')
#reference params: YOLOv8n, YOLOv8s, YOLOv8m, YOLOv8l, YOLOv8x


#part1
# Define the dataset_path
dataset_path = 'Vehicle_Detection_Image_Dataset'

# Set the path to the YAML file
yaml_file_path = os.path.join(dataset_path, 'data.yaml')

# Load and print the contents of the YAML file
with open(yaml_file_path, 'r') as file:
    yaml_content = yaml.load(file, Loader=yaml.FullLoader)
    print(yaml.dump(yaml_content, default_flow_style=False))

#part2
# Set paths for training and validation image sets
train_images_path = os.path.join(dataset_path, 'train', 'images')
valid_images_path = os.path.join(dataset_path, 'valid', 'images')

# Initialize counters for the number of images
num_train_images = 0
num_valid_images = 0

# Initialize sets to hold the unique sizes of images
train_image_sizes = set()
valid_image_sizes = set()

# Check train images sizes and count
for filename in os.listdir(train_images_path):
    if filename.endswith('.jpg'):
        num_train_images += 1
        image_path = os.path.join(train_images_path, filename)
        with Image.open(image_path) as img:
            train_image_sizes.add(img.size)

# Check validation images sizes and count
for filename in os.listdir(valid_images_path):
    if filename.endswith('.jpg'):
        num_valid_images += 1
        image_path = os.path.join(valid_images_path, filename)
        with Image.open(image_path) as img:
            valid_image_sizes.add(img.size)

# Print the results
print(f"Number of training images: {num_train_images}")
print(f"Number of validation images: {num_valid_images}")

# Check if all images in training set have the same size
if len(train_image_sizes) == 1:
    print(f"All training images have the same size: {train_image_sizes.pop()}")
else:
    print("Training images have varying sizes.")

# Check if all images in validation set have the same size
if len(valid_image_sizes) == 1:
    print(f"All validation images have the same size: {valid_image_sizes.pop()}")
else:
    print("Validation images have varying sizes.")

#part3
# # List all jpg images in the directory
# image_files = [file for file in os.listdir(train_images_path) if file.endswith('.jpg')]
#
# # Select 8 images at equal intervals
# num_images = len(image_files)
# selected_images = [image_files[i] for i in range(0, num_images, num_images // 8)]
#
# # Create a 2x4 subplot
# fig, axes = plt.subplots(2, 4, figsize=(20, 11))
#
# # Display each of the selected images
# for ax, img_file in zip(axes.ravel(), selected_images):
#     img_path = os.path.join(train_images_path, img_file)
#     image = Image.open(img_path)
#     ax.imshow(image)
#     ax.axis('off')
#
# plt.suptitle('Sample Images from Training Dataset', fontsize=20)
# plt.tight_layout()
# plt.show()

#part4
# Train the model on our custom dataset
results = model.train(
    data=yaml_file_path,     # Path to the dataset configuration file
    epochs=100,              # Number of epochs to train for
    imgsz=640,               # Size of input images as integer
    device=0,                # Device to run on, i.e. cuda device=0
    patience=50,             # Epochs to wait for no observable improvement for early stopping of training
    batch=32,                # Number of images per batch
    optimizer='auto',        # Optimizer to use, choices=[SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto]
    lr0=0.0001,              # Initial learning rate
    lrf=0.1,                 # Final learning rate (lr0 * lrf)
    dropout=0.1,             # Use dropout regularization
    seed=0,                   # Random seed for reproducibility
    workers=0,  # 禁用多线程数据加载
    #project="YOLOV8-T1",
    #wandb=True    #wandb开启
)