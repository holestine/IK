import pandas as pd
import os
from pathlib import Path
import shutil
import cv2

label_file = 'data/labeled/labels.csv'
src_base_dir   = os.path.dirname(label_file)
dataset_dir = 'datasets'
test_label_dir   = '{}/{}/labels'.format(dataset_dir, 'val')
test_image_dir   = '{}/{}/images'.format(dataset_dir, 'val')
train_label_dir  = '{}/{}/labels'.format(dataset_dir, 'train')
train_image_dir  = '{}/{}/images'.format(dataset_dir, 'train')

# Convert from bounding boxes and image size to YOLO representations
def get_yolo_annotations(x1, y1, x2, y2, w, h):
    center_x = (x2+x1)/2/w
    center_y = (y2+y1)/2/h
    width    = (x2-x1)/w
    height   = (y2-y1)/h
    return center_x, center_y, width, height

# Write annotations in YOLO format
def write_yolo_annotations(x1, y1, x2, y2, image_base_dir, image, image_dst, anno_dst):
    
    # Get YOLO annotations
    image_src = '{}/{}'.format(image_base_dir, image)
    (h, w) = cv2.imread(image_src, cv2.IMREAD_GRAYSCALE).shape
    center_x, center_y, width, height = get_yolo_annotations(x1, y1, x2, y2, w, h)

    # Make sure values are reasonable
    if center_x > 0 and center_y > 0 and width > 0 and height > 0 and \
       center_x < 1 and center_y < 1 and width < 1 and height < 1:
        # Copy image to dataset
        shutil.copy(image_src, image_dst)

        # Write the annotiations
        anno_name = Path(image_src).stem
        f = open('{}/{}.txt'.format(anno_dst, anno_name), 'w')
        f.write('0 {} {} {} {}'.format(center_x, center_y, width, height))
        f.close()
    else:
        print(image)

# Create dataset directories
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)
if not os.path.exists(test_label_dir):
    os.makedirs(test_label_dir)
if not os.path.exists(test_image_dir):
    os.makedirs(test_image_dir)
if not os.path.exists(train_label_dir):
    os.makedirs(train_label_dir)
if not os.path.exists(train_image_dir):
    os.makedirs(train_image_dir)

# Shuffle the labelled data
labels_df = pd.read_csv(label_file)
labels_df = labels_df.sample(frac = 1, random_state = 0)

# Split the labelled data
split_loc = int(len(labels_df)*.2)

# Create dataset
for index, (image,label,x1,y1,x2,y2) in labels_df.iterrows():
    if index < split_loc:
        # Process Test items
        write_yolo_annotations(x1, y1, x2, y2, src_base_dir, image, test_image_dir, test_label_dir)
    else:
        # Process Train items
        write_yolo_annotations(x1, y1, x2, y2, src_base_dir, image, train_image_dir, train_label_dir)
