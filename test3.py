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

# 从头开始创建一个新的YOLO模型
#model = YOLO('yolov8.yaml')

# 加载预训练的YOLO模型（推荐用于训练）
model = YOLO('yolov8n.pt')
#reference params: YOLOv8n, YOLOv8s, YOLOv8m, YOLOv8l, YOLOv8x


# Define the dataset_path
dataset_path = 'Vehicle_Detection_Image_Dataset'


# Define the path to the directory
post_training_files_path = 'runs\\detect\\train9\\'

# List the files in the directory
#!ls {post_training_files_path}


# # Define a function to plot learning curves for loss values
# def plot_learning_curve(df, train_loss_col, val_loss_col, title):
#     plt.figure(figsize=(12, 5))
#     sns.lineplot(data=df, x='epoch', y=train_loss_col, label='Train Loss', color='#141140', linestyle='-', linewidth=2)
#     sns.lineplot(data=df, x='epoch', y=val_loss_col, label='Validation Loss', color='orangered', linestyle='--', linewidth=2)
#     plt.title(title)
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.show()


# Create the full file path for 'results.csv' using the directory path and file name
# results_csv_path = os.path.join(post_training_files_path, 'results.csv')

# # Load the CSV file from the constructed path into a pandas DataFrame
# df = pd.read_csv(results_csv_path)
#
# # Remove any leading whitespace from the column names
# df.columns = df.columns.str.strip()
#
# # Plot the learning curves for each loss
# plot_learning_curve(df, 'train/box_loss', 'val/box_loss', 'Box Loss Learning Curve')
# plot_learning_curve(df, 'train/cls_loss', 'val/cls_loss', 'Classification Loss Learning Curve')
# plot_learning_curve(df, 'train/dfl_loss', 'val/dfl_loss', 'Distribution Focal Loss Learning Curve')


# # Construct the path to the normalized confusion matrix image
# confusion_matrix_path = os.path.join(post_training_files_path, 'confusion_matrix_normalized.png')
#
# # Read the image using cv2
# cm_img = cv2.imread(confusion_matrix_path)
#
# # Convert the image from BGR to RGB color space for accurate color representation with matplotlib
# cm_img = cv2.cvtColor(cm_img, cv2.COLOR_BGR2RGB)
#
# # Display the image
# plt.figure(figsize=(10, 10), dpi=120)
# plt.imshow(cm_img)
# plt.axis('off')
# plt.show()


# Construct the path to the best model weights file using os.path.join
best_model_path = os.path.join(post_training_files_path, 'weights/best.pt')

# Load the best model weights into the YOLO model
best_model = YOLO(best_model_path)

# Validate the best model using the validation set with default parameters
metrics = best_model.val(split='val', workers=0)


# Convert the dictionary to a pandas DataFrame and use the keys as the index
#metrics_df = pd.DataFrame.from_dict(metrics.results_dict, orient='index', columns=['Metric Value'])

# Display the DataFrame
#metrics_df.round(3)

# 假设 metrics.results_dict 是模型验证返回的结果字典
metrics_dict = metrics.results_dict  # 获取验证结果字典

# 检查字典是否为空
if metrics_dict:
    # 转换为 DataFrame
    metrics_df = pd.DataFrame.from_dict(metrics_dict, orient='index', columns=['Metric Value'])

    # 保留三位小数
    metrics_df = metrics_df.round(3)

    # 显示 DataFrame
    print(metrics_df)
else:
    print("Metrics dictionary is empty.")


# # Define the path to the validation images
# valid_images_path = os.path.join(dataset_path, 'valid', 'images')
#
# # List all jpg images in the directory
# image_files = [file for file in os.listdir(valid_images_path) if file.endswith('.jpg')]
#
# # Select 9 images at equal intervals
# num_images = len(image_files)
# selected_images = [image_files[i] for i in range(0, num_images, num_images // 9)]
#
# # Initialize the subplot
# fig, axes = plt.subplots(3, 3, figsize=(20, 21))
# fig.suptitle('Validation Set Inferences', fontsize=24)
#
# # Perform inference on each selected image and display it
# for i, ax in enumerate(axes.flatten()):
#     image_path = os.path.join(valid_images_path, selected_images[i])
#     results = best_model.predict(source=image_path, imgsz=640, conf=0.5)
#     annotated_image = results[0].plot(line_width=1)
#     annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
#     ax.imshow(annotated_image_rgb)
#     ax.axis('off')
#
# plt.tight_layout()
# plt.show()



# # Path to the image file
sample_image_path = 'Vehicle_Detection_Image_Dataset/sample_image.jpg'
#
# # Perform inference on the provided image using best model
results = best_model.predict(source=sample_image_path, imgsz=640, conf=0.7)
#
# # Annotate and convert image to numpy array
sample_image = results[0].plot(line_width=2)
#
# # Convert the color of the image from BGR to RGB for correct color representation in matplotlib
sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)
#
# Display annotated image
plt.figure(figsize=(20, 15))
plt.imshow(sample_image)
plt.title('Detected Objects in Sample Image by the Fine-tuned YOLOv8 Model', fontsize=20)
plt.axis('off')
plt.show()

# #video输出
# # Define the path to the sample video in the dataset
# dataset_video_path = 'Vehicle_Detection_Image_Dataset/sample_video.mp4'
#
# # Define the destination path in the working directory
# video_path = 'working/sample_video.mp4'
#
# # Copy the video file from its original location in the dataset to the current working directory in Kaggle for further processing
# shutil.copyfile(dataset_video_path, video_path)
#
# # Initiate vehicle detection on the sample video using the best performing model and save the output
# results = best_model.predict(source=video_path, save=True, verbose=True)
# print(results)


# Define the threshold for considering traffic as heavy
# heavy_traffic_threshold = 10
#
# # Define the vertices for the quadrilaterals
# vertices1 = np.array([(465, 350), (609, 350), (510, 630), (2, 630)], dtype=np.int32)
# vertices2 = np.array([(678, 350), (815, 350), (1203, 630), (743, 630)], dtype=np.int32)
#
# # Define the vertical range for the slice and lane threshold
# x1, x2 = 325, 635
# lane_threshold = 609
#
# # Define the positions for the text annotations on the image
# text_position_left_lane = (10, 50)
# text_position_right_lane = (820, 50)
# intensity_position_left_lane = (10, 100)
# intensity_position_right_lane = (820, 100)
#
# # Define font, scale, and colors for the annotations
# font = cv2.FONT_HERSHEY_SIMPLEX
# font_scale = 1
# font_color = (255, 255, 255)  # White color for text
# background_color = (0, 0, 255)  # Red background for text
#
# # Open the video
# cap = cv2.VideoCapture('Vehicle_Detection_Image_Dataset/sample_video.mp4')
#
# # Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('traffic_density_analysis.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
#
# # Read until video is completed
# while cap.isOpened():
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#     if ret:
#         # Create a copy of the original frame to modify
#         detection_frame = frame.copy()
#
#         # Black out the regions outside the specified vertical range
#         detection_frame[:x1, :] = 0  # Black out from top to x1
#         detection_frame[x2:, :] = 0  # Black out from x2 to the bottom of the frame
#
#         # Perform inference on the modified frame
#         results = best_model.predict(detection_frame, imgsz=640, conf=0.4)
#         processed_frame = results[0].plot(line_width=1)
#
#         # Restore the original top and bottom parts of the frame
#         processed_frame[:x1, :] = frame[:x1, :].copy()
#         processed_frame[x2:, :] = frame[x2:, :].copy()
#
#         # Draw the quadrilaterals on the processed frame
#         cv2.polylines(processed_frame, [vertices1], isClosed=True, color=(0, 255, 0), thickness=2)
#         cv2.polylines(processed_frame, [vertices2], isClosed=True, color=(255, 0, 0), thickness=2)
#
#         # Retrieve the bounding boxes from the results
#         bounding_boxes = results[0].boxes
#
#         # Initialize counters for vehicles in each lane
#         vehicles_in_left_lane = 0
#         vehicles_in_right_lane = 0
#
#         # Loop through each bounding box to count vehicles in each lane
#         for box in bounding_boxes.xyxy:
#             # Check if the vehicle is in the left lane based on the x-coordinate of the bounding box
#             if box[0] < lane_threshold:
#                 vehicles_in_left_lane += 1
#             else:
#                 vehicles_in_right_lane += 1
#
#         # Determine the traffic intensity for the left lane
#         traffic_intensity_left = "Heavy" if vehicles_in_left_lane > heavy_traffic_threshold else "Smooth"
#         # Determine the traffic intensity for the right lane
#         traffic_intensity_right = "Heavy" if vehicles_in_right_lane > heavy_traffic_threshold else "Smooth"
#
#         # Add a background rectangle for the left lane vehicle count
#         cv2.rectangle(processed_frame, (text_position_left_lane[0] - 10, text_position_left_lane[1] - 25),
#                       (text_position_left_lane[0] + 460, text_position_left_lane[1] + 10), background_color, -1)
#
#         # Add the vehicle count text on top of the rectangle for the left lane
#         cv2.putText(processed_frame, f'Vehicles in Left Lane: {vehicles_in_left_lane}', text_position_left_lane,
#                     font, font_scale, font_color, 2, cv2.LINE_AA)
#
#         # Add a background rectangle for the left lane traffic intensity
#         cv2.rectangle(processed_frame, (intensity_position_left_lane[0] - 10, intensity_position_left_lane[1] - 25),
#                       (intensity_position_left_lane[0] + 460, intensity_position_left_lane[1] + 10), background_color,
#                       -1)
#
#         # Add the traffic intensity text on top of the rectangle for the left lane
#         cv2.putText(processed_frame, f'Traffic Intensity: {traffic_intensity_left}', intensity_position_left_lane,
#                     font, font_scale, font_color, 2, cv2.LINE_AA)
#
#         # Add a background rectangle for the right lane vehicle count
#         cv2.rectangle(processed_frame, (text_position_right_lane[0] - 10, text_position_right_lane[1] - 25),
#                       (text_position_right_lane[0] + 460, text_position_right_lane[1] + 10), background_color, -1)
#
#         # Add the vehicle count text on top of the rectangle for the right lane
#         cv2.putText(processed_frame, f'Vehicles in Right Lane: {vehicles_in_right_lane}', text_position_right_lane,
#                     font, font_scale, font_color, 2, cv2.LINE_AA)
#
#         # Add a background rectangle for the right lane traffic intensity
#         cv2.rectangle(processed_frame, (intensity_position_right_lane[0] - 10, intensity_position_right_lane[1] - 25),
#                       (intensity_position_right_lane[0] + 460, intensity_position_right_lane[1] + 10), background_color,
#                       -1)
#
#         # Add the traffic intensity text on top of the rectangle for the right lane
#         cv2.putText(processed_frame, f'Traffic Intensity: {traffic_intensity_right}', intensity_position_right_lane,
#                     font, font_scale, font_color, 2, cv2.LINE_AA)
#
#         # Write the processed frame to the output video
#         out.write(processed_frame)
#
#         # Uncomment the following 3 lines if running this code on a local machine to view the real-time processing results
#         # cv2.imshow('Real-time Analysis', processed_frame)
#         # if cv2.waitKey(1) & 0xFF == ord('q'):  # Press Q on keyboard to exit the loop
#         #     break
#     else:
#         break
#
# Video("traffic_density_analysis.mp4", embed=True, width=960)
#
# # Release the video capture and video write objects
# cap.release()
# out.release()

# Close all the frames
# cv2.destroyAllWindows()