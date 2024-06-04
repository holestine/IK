import cv2, os, random
from ultralytics import YOLO
import matplotlib.pyplot as plt
from tracker import bb_tracker, video_editor


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
    video = video_editor('out', 'video.mp4', width=img.shape[1], height=img.shape[0])

    while images_to_process:

        # Detect objects with YOLO
        yolo_result = model.predict(img, verbose=False)
        
        # Process the detections
        tracker.process_yolo_result(yolo_result)

        # Get all the matched bounding boxes and their properties needed for rendering
        (boxes, confidences, classes, colors) = tracker.get_matches()

        # Draw matches on the frame and add it to the video
        for box, confidence, type, color in zip(boxes, confidences, classes, colors):
            left   = int(box[0])
            top    = int(box[1])
            right  = int(box[2])
            bottom = int(box[3])
            img = video.drawPred(yolo_result[0].names, img, type, confidence, left, top, right, bottom, color)
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
