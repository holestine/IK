import cv2
import numpy as np
import os

DATA_DIR = "./datasets/"
IMG_WIDTH = 1000
IMG_HEIGHT = 500

# Convert from bounding boxes and image size to YOLO representations
def get_yolo_annotations(x1, y1, x2, y2, w, h):
    center_x = (x2+x1)/2/w
    center_y = (y2+y1)/2/h
    width    = (x2-x1)/w
    height   = (y2-y1)/h
    return center_x, center_y, width, height

# Write annotations in YOLO format
def write_yolo_annotations(x1, y1, x2, y2, w, h, label_file, class_id=0):
    
    # Get YOLO annotations
    center_x, center_y, width, height = get_yolo_annotations(x1, y1, x2, y2, w, h)

    # Make sure values are reasonable
    if center_x > 0 and center_y > 0 and width > 0 and height > 0 and \
       center_x < 1 and center_y < 1 and width < 1 and height < 1:
        
        # Write the annotiations
        f = open(label_file, 'a')
        f.write('{} {} {} {} {}\n'.format(class_id, center_x, center_y, width, height))
        f.close()
    else:
        print("Something Wrong")

def draw_random_circle(img):
    radius = np.random.randint(10, 100)

    center_x = np.max([np.random.randint(0, IMG_WIDTH), radius])
    center_x = np.min([center_x, IMG_WIDTH-radius])

    center_y = np.max([np.random.randint(0, IMG_HEIGHT), radius])
    center_y = np.min([center_y, IMG_HEIGHT-radius])

    color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
    cv2.circle(img, (center_x, center_y), radius, color, -1)

    return center_x, center_y, radius

def draw_random_triangle(img):
    pt1 = (np.random.randint(100, IMG_WIDTH-100), np.random.randint(100, IMG_HEIGHT-100))
    pt2 = (pt1[0]+np.random.randint(50, 100), pt1[1]+np.random.randint(50, 100))
    pt3 = (pt1[0]-np.random.randint(50, 100), pt1[1]+np.random.randint(50, 100))
    triangle_cnt = np.array( [pt1, pt2, pt3] )

    color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
    cv2.drawContours(img, [triangle_cnt], 0, color, -1)

    x_vals = [x[0] for x in triangle_cnt]
    y_vals = [x[1] for x in triangle_cnt]
    return np.min(x_vals), np.max(x_vals), np.min(y_vals), np.max(y_vals), IMG_WIDTH, IMG_HEIGHT

def create_data(dir, num_images):
    img_dir = os.path.join(DATA_DIR, dir, "images")
    label_dir = os.path.join(DATA_DIR, dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    for i in range(num_images):
        img = np.random.randint(0, 255, (IMG_HEIGHT, IMG_WIDTH,3), "uint8")
        img_file = "{}/{}.jpg".format(img_dir, i)
        label_file = "{}/{}.txt".format(label_dir, i)

        center_x, center_y, radius = draw_random_circle(img)
        write_yolo_annotations(center_x-radius, center_y-radius, center_x+radius, center_y+radius, IMG_WIDTH, IMG_HEIGHT, label_file, 0)

        x_min, x_max, y_min, y_max, width, height = draw_random_triangle(img)
        write_yolo_annotations(x_min, y_min, x_max, y_max, IMG_WIDTH, IMG_HEIGHT, label_file, 1)

        cv2.imwrite(img_file, img)

def main():
    create_data("train", 100)
    create_data("test", 25) 
    create_data("val", 10)  

main()
