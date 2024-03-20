from ultralytics import YOLO

YOLO_MODEL = 'yolov8n'
OBJECT     = 'object'

model = YOLO('{}.pt'.format(YOLO_MODEL))

results = model.train(
    data = 'object_v8.yaml',
    imgsz = 640,
    epochs = 10,
    batch = 8,
    name = '{}-{}'.format(YOLO_MODEL, OBJECT)
)
