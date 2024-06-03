import pickle
import matplotlib.pyplot as plt
from ultralytics import YOLO
import numpy as np
import os, cv2
from scipy.optimize import linear_sum_assignment

class video:

    def __init__(self, out_dir, filename, fps=5, width=1200, height=800, is_rgb=True) -> None:
        '''
        Creates a video 
    
        Parameters
        ----------
        out_dir : str
            The directory to store the video
        filename : str
            The name of the video
        fps : int
            The frames per second
        width : int
            The video width
        height : int
            The video height
        is_rgb : bool
            Flag to specify whether or not the video will be in color
        '''
        
        # Make sure directory is there
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # Construct the path
        video_path = os.path.join(out_dir, filename)

        # Construct the video writer
        self.out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height), is_rgb)

    def add_frame(self, img):
        '''
        Adds a single frame to the video 
    
        Parameters
        ----------
        img : array(int)
            The image
        '''
        
        self.out.write(img)

    def save_video(self):
        '''
        Saves the video
        '''

        self.out.release()


class trajectory:
    current_id = 1
    def __init__(self, box, confidence, type) -> None:

        self.id = trajectory.current_id
        trajectory.current_id = self.id + 1
        self.boxes = [box]
        self.confidences = [confidence]
        self.type = type
        self.color = self.id_to_color(self.id)
        self.consecutive_detections = 0
        self.missed_detections = 0

    def id_to_color(self, id):
        """
        Random function to convert an id to a color
        Do what you want here but keep numbers below 255
        """
        blue  = id*107 % 256
        green = id*149 %256
        red   = id*227 %256
        return (red, green, blue)

class yolo_detection:

    def __init__(self, box, confidence, type) -> None:
        self.box = box
        self.confidence = confidence
        self.type = type

class tracker:
    
    def __init__(self) -> None:
        self.confThreshold = 0.5
        self.nmsThreshold = 0.8
        self.iou_threshold = 0.3
        self.trajectories = []

    def drawPred(self, classes, frame, classId, conf, left, top, right, bottom):
        '''
        Draw a bounding box around a detected object given the box coordinates
        Later, we could repurpose that to display an ID
        '''
        # Draw a bounding box.
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), thickness=5)
        label = '%.2f' % conf
        # Get the label for the class name and its confidence
        if classes:
            assert(classId < len(classes))
            label = '%s:%s' % (classes[classId], label)

        #Display the label at the top of the bounding box
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), thickness=3)
        return frame

    def process_yolo_result(self, yolo_result):

        # Get the bounding box locations and the associated classes and confidences
        boxes_xyxy  = yolo_result[0].boxes.xyxy.cpu().numpy()
        confidences = yolo_result[0].boxes.conf.cpu().numpy()
        classes     = yolo_result[0].boxes.cls.cpu().numpy()

        # Filter out low confidence boxes
        filtered_boxes_xyxy  = boxes_xyxy[confidences > self.confThreshold]
        filtered_confidences = confidences[confidences > self.confThreshold]
        filtered_classes     = classes[confidences > self.confThreshold]

        # Perform Non Maximum Suppression to remove redundant boxes with low confidence
        indices = cv2.dnn.NMSBoxes(filtered_boxes_xyxy, filtered_confidences, self.confThreshold, self.nmsThreshold)

        detections = []
        for i in indices:
            detection = yolo_detection(filtered_boxes_xyxy[i], filtered_confidences[i], filtered_classes[i])
            detections.append(detection)

        self.associate(detections)


        return filtered_boxes_xyxy[indices], filtered_confidences[indices], filtered_classes[indices]

    def box_iou(self, box1, box2):
        xA = max(box1[0], box2[0]) # The max left hand side
        yA = max(box1[1], box2[1]) # The max of the top
        xB = min(box1[2], box2[2]) # The min right hand side
        yB = min(box1[3], box2[3]) # The min of the bottom

        inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1) 

        # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
        box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1) #abs((box1[3] - box1[1])*(box1[2]- box1[0]))
        box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1) #abs((box2[3] - box2[1])*(box2[2]- box2[0]))
        union_area = (box1_area + box2_area) - inter_area
        
        # Compute the IoU
        iou = inter_area/float(union_area)
        return iou

    def associate(self, detections):
        """
        old_boxes will represent the former bounding boxes (at time 0)
        new_boxes will represent the new bounding boxes (at time 1)
        Function goal: Define a Hungarian Matrix with IOU as a metric and return, for each box, an id
        """

        # Nothing to associate, add missed detection to all trajectories
        if len(detections) == 0:
            for traj in self.trajectories:
                traj.missed_detections += 1
            return

        # Nothing to match so just create new trajectories
        if len(self.trajectories) == 0:
            for detection in detections:
                t = trajectory(detection.box, detection.confidence, detection.type)
                self.trajectories.append(t)
            return
        
        # Get the last know location for each trajectory
        old_boxes = [t.boxes[-1] for t in self.trajectories]

        # Get the location of all the new detections
        new_boxes = [detection.box for detection in detections]

        # Define an IOU Matrix with dimensions old_boxes x new_boxes
        iou_matrix = np.zeros((len(old_boxes), len(new_boxes)), dtype=np.float32)

        # Go through all the boxes and store each IOU value
        # TODO: do this by class type
        for i, old_box in enumerate(old_boxes):
            for j, new_box in enumerate(new_boxes):
                iou_matrix[i][j] = self.box_iou(old_box, new_box)
                #iou_matrix[i][j] = hungarian_cost(old_box, new_box)

        # Call the Hungarian Algorithm
        hungarian_row, hungarian_col = linear_sum_assignment(-iou_matrix)
        hungarian_matrix = np.array(list(zip(hungarian_row, hungarian_col)))

        # Create new unmatched lists for old and new boxes
        matches, unmatched_detections, unmatched_trackers = [], [], []

        # Go through the matches in the Hungarian Matrix 
        for h in hungarian_matrix:
            # If it's under the IOU threshold increment the missed detections in the tracker and add a new trajectory
            if(iou_matrix[h[0],h[1]] < self.iou_threshold):
                unmatched_trackers.append(old_boxes[h[0]])
                self.trajectories[h[0]].missed_detections += 1 #############

                unmatched_detections.append(new_boxes[h[1]])
                detection = detections[h[1]]
                t = trajectory(detection.box, detection.confidence, detection.type)
                self.trajectories.append(t)
            # Else, it's a match, add the box to the trajectory
            else:
                matches.append(h.reshape(1,2))
        
        if(len(matches)==0):
            matches = np.empty((0,2), dtype=int)
        else:
            matches = np.concatenate(matches,axis=0)

        # Add matched bounding boxes to the trajectories
        for match in matches:
            self.trajectories[match[0]].boxes.append(detections[match[1]].box)
            self.trajectories[match[0]].consecutive_detections += 1
        
        # Go through old boxes, if no matched detection, add it to the unmatched_old_boxes
        for t, box in enumerate(old_boxes):
            if(t not in hungarian_matrix[:,0]):
                unmatched_trackers.append(box)
                self.trajectories[t].missed_detections += 1
        
        # Go through new boxes, if no matched tracking, add it to the unmatched_new_boxes
        for d, det in enumerate(new_boxes):
            if(d not in hungarian_matrix[:,1]):
                unmatched_detections.append(det)

        new_boxes = [detection.box for detection in detections]
        for d, det in enumerate(detections):
            if(d not in hungarian_matrix[:,1]):
                t = trajectory(det.box, det.confidence, detection.type)
                self.trajectories.append(t)
        
        print("Matched Detections")
        for match in matches:
            print(self.trajectories[match[0]].boxes[-2])
            print(detections[match[1]].box)
            print('\n')
        
        print("\nUnmatched Detections ")
        print(np.array(unmatched_detections))
        
        print("\nUnmatched trackers ")
        print(np.array(unmatched_trackers))

        return

    def print_matches(self):
        print("Matched Detections:")
        for t in self.trajectories:
            if len(t.boxes) > 1:
                if t.missed_detections == 0:
                    print(t.boxes[-2])
                    print(t.boxes[-1])
                    print('\n')

    def print_unmatched_trackers(self):
        print("Unmatched Trackers:")
        for t in self.trajectories:
            if len(t.boxes) > 0:
                if t.missed_detections > 0:
                    print(t.boxes[-1])
                    print('\n')

    def print_unmatched_detections(self):
        print("Unmatched Detections:")
        for t in self.trajectories:
            if len(t.boxes) == 1:
                if t.missed_detections == 0:
                    print(t.boxes[-1])
                    print('\n')

    


YOLO_MODEL = 'yolov8n'

model = YOLO('{}.pt'.format(YOLO_MODEL))

result_boxes  = [] # Empty list for output boxes

# Load some images
dataset_images = pickle.load(open('Images/images_tracking.p', "rb"))

tracker = tracker()
video = video('out', 'video.mp4', width=dataset_images[0].shape[1], height=dataset_images[0].shape[0])

for img in dataset_images:
    # Detect objects with YOLO
    yolo_result = model.predict(img)
    
    boxes, confidences, classes = tracker.process_yolo_result(yolo_result)

    tracker.print_matches()
    tracker.print_unmatched_trackers()
    tracker.print_unmatched_detections()

    for box, confidence, type in zip(boxes, confidences, classes):
        left   = int(box[0])
        top    = int(box[1])
        right  = int(box[2])
        bottom = int(box[3])
        img = tracker.drawPred(yolo_result[0].names, img, type, confidence, left, top, right, bottom)
        
    video.add_frame(img)
        

video.save_video()
