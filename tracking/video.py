import os, cv2

class video_editor:

    def __init__(self, out_dir, filename, fps=30, width=1200, height=800, is_rgb=True) -> None:
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

        # Store video size
        self.width, self.height = width, height

        # Construct the video writer
        self.out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (self.width, self.height), is_rgb)

    def add_frame(self, img) -> None:
        '''
        Adds a single frame to the video 
    
        Parameters
        ----------
        img : array(int)
            The image
        '''
        img = cv2.resize(img, (self.width, self.height))
        self.out.write(img)

    def save_video(self) -> None:
        '''
        Saves the video
        '''

        self.out.release()

    def drawPred(self, frame, type, conf, box, color=(255, 0, 0)):
        '''
        Draws a bounding box around the detected object 

        Parameters
        ----------
        frame : array[width, height, color depth]
            The name of the video
        type : str
            The object type
        conf : float
            A value between 0 and 1 indicating the confidence of the detection
        box : array[4]
            The bounding box
        color : bool
            The color of the bounding box
        '''

        left   = int(box[0])
        top    = int(box[1])
        right  = int(box[2])
        bottom = int(box[3])

        # Draw the bounding box
        cv2.rectangle(frame, (left, top), (right, bottom), color, thickness=5)

        # Create label with type and confidence
        label = "{}: {:.2f}".format(type, conf)

        # Display the label at the top of the bounding box
        labelSize, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), thickness=3)

        return frame
