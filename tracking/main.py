import cv2, os, random
from ultralytics import YOLO
import matplotlib.pyplot as plt
from tracker import bb_tracker
from video import video_editor

def main(video_path=None):

    # Validate input
    if video_path == None:
        print('Please specify a video')
        return

    # Initialize YOLO model
    YOLO_MODEL = 'yolov8n'
    model = YOLO('{}.pt'.format(YOLO_MODEL))

    # Initialize the tracker
    tracker = bb_tracker()

    # Load the video and process it one frame at a time
    vidcap = cv2.VideoCapture(video_path)
    images_to_process, img = vidcap.read()
    video = video_editor('out', 'video.mp4', width=int(img.shape[1]/4), height=int(img.shape[0]/4))

    while images_to_process:

        # Detect objects with YOLO
        yolo_result = model.predict(img, verbose=False)
        
        # Process the detections
        tracker.process_yolo_result(yolo_result)

        # Get all the matched bounding boxes and their properties needed for rendering
        (boxes, confidences, classes, colors) = tracker.get_matches()

        # Draw matches on the frame and add it to the video
        for box, confidence, type, color in zip(boxes, confidences, classes, colors):
            type = yolo_result[0].names[type]
            img = video.drawPred(img, type, confidence, box, color)
        video.add_frame(img)

        images_to_process, img = vidcap.read()

        # For Debug
        #tracker.print_matches()
        #tracker.print_unmatched_trackers()
        #tracker.print_unmatched_detections()
            
    # Save the video
    video.save_video()

if __name__ == '__main__':
    # Get a random test image (dataset used is BDD100k)
    video_path = random.choice(os.listdir('videos/test/'))

    main('videos/test/{}'.format(video_path))
