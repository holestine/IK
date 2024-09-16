from ultralytics import YOLO
from pathlib import Path
import cv2
import math
best_yolo_model = 'yolo/runs/detect/yolov8n/weights/best.pt'
image_folder = 'yolo/datasets/val/images/'
image_ext = '*.jpg'

def show_image(path, title='unknown', boxes=[], classes=[]):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    
    for box in boxes:
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])
        img = cv2.rectangle(img, (x1,y1), (x2,y2), (0, 0, 255), 3)
    
    img = cv2.resize(img,(0, 0),fx=0.5, fy=0.5, interpolation = cv2.INTER_AREA)
    
    # show image
    cv2.imshow(title, img)
    cv2.moveWindow(title, 10, 10)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

model = YOLO(best_yolo_model)
test_images = [str(x) for x in list(Path(image_folder).glob(image_ext))]
chunk_size = math.ceil(len(test_images) / 16)

for i in range(0, len(test_images), chunk_size):
    results = model.predict(test_images[i:i + chunk_size])
    for res in results:
        if len(res.boxes.conf) == 0:
            show_image(res.path, 'No Shapes Detected')
        else:
            show_image(res.path, 'Detections', res.boxes.xyxy.detach().cpu().numpy(), res.boxes.cls.detach().cpu().numpy())


