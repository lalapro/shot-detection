# Avi Shah - Basketball Shot Detector/Tracker - July 2023

from ultralytics import YOLO
import cv2
import os
import cvzone
import math
import numpy as np
from utils import score, detect_down, detect_up, in_hoop_region, clean_hoop_pos, clean_ball_pos

save_path = "output/"

class ShotDetector:
    def __init__(self, num_images_to_save=1):
        # Load the YOLO model created from main.py - change text to your relative path
        self.model = YOLO("./best_newdata.pt")
        self.class_names = ['Basketball']

        # Use video - replace text with your video path
        self.cap = cv2.VideoCapture("clip2.mp4")

        self.frames = []  # Store frames for reverse playback
        self.ball_pos = []  # array of tuples ((x_pos, y_pos), frame count, width, height, conf)
        self.hoop_pos = []  # array of tuples ((x_pos, y_pos), frame count, width, height, conf)

        self.frame_count = 0
        self.frame = None

        self.makes = 0
        self.attempts = 0

        # Used to detect shots (upper and lower region)
        self.up = False
        self.down = False
        self.up_frame = 0
        self.down_frame = 0

        # Used for green and red colors after make/miss
        self.fade_frames = 20
        self.fade_counter = 0
        self.overlay_color = (0, 0, 0)

        self.num_images_to_save = num_images_to_save
        self.load_frames()
        self.run()

    def load_frames(self):
        """ Load all frames from the video into memory. """
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            self.frames.append(frame)
        
        self.cap.release()

    def run(self):
        images_saved = 0  # Track the number of images saved

        for self.frame_count in range(len(self.frames) - 1, -1, -1):  # Process frames in reverse order
            self.frame = self.frames[self.frame_count]
            original_frame = self.frame.copy()  # Make a copy for drawing
            
            results = self.model(self.frame, stream=True)

            def is_fully_contained(box1, box2):
                x1_min, y1_min, x1_max, y1_max = box1
                x2_min, y2_min, x2_max, y2_max = box2

                # Check if all corners of box1 are within box2
                return (x1_min >= x2_min and x1_max <= x2_max and y1_min >= y2_min and y1_max <= y2_max)
            
            def save_bounding_box(frame, box, save_path, frame_count):
                x1, y1, x2, y2 = box
                cropped_img = frame[y1:y2, x1:x2]
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                filename = os.path.join(save_path, f"person_{frame_count}.jpg")
                cv2.imwrite(filename, cropped_img)

            ball_boxes = []
            person_boxes = []

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Bounding box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1

                    # Confidence
                    conf = math.ceil((box.conf[0] * 100)) / 100

                    # Class Name
                    cls = int(box.cls[0])
                    print(f"Class: {cls}, Confidence: {conf}")

                    center = (int(x1 + w / 2), int(y1 + h / 2))

                    # Only create ball points if high confidence or near hoop
                    if (conf > 0.7) and cls == 0:
                        ball_boxes.append((x1, y1, x2, y2, center, conf))
                        self.ball_pos.append((center, self.frame_count, w, h, conf))
                        # Draw bounding box only on the copied frame
                        cvzone.cornerRect(original_frame, (x1, y1, w, h))

                    elif cls == 2:  # Person
                        person_boxes.append((x1, y1, x2, y2))

            for ball in ball_boxes:
                for person in person_boxes:
                    if is_fully_contained(ball[:4], person):
                        print(f"Full containment detected: ball at {ball[4]} is within person")
                        save_bounding_box(self.frame, person, save_path, self.frame_count)
                        images_saved += 1  # Increment the image counter
                        if images_saved >= self.num_images_to_save:
                            break
                if images_saved >= self.num_images_to_save:
                    break
            if images_saved >= self.num_images_to_save:
                break

            self.clean_motion()
            # self.shot_detection()
            # self.display_score()

            cv2.imshow('Frame', original_frame)  # Show the frame with bounding boxes

            # Close if 'q' is clicked
            if cv2.waitKey(1) & 0xFF == ord('q'):  # higher waitKey slows video down, use 1 for webcam
                break

        cv2.destroyAllWindows()

    def clean_motion(self):
        # Clean and display ball motion
        self.ball_pos = clean_ball_pos(self.ball_pos, self.frame_count)
        for i in range(0, len(self.ball_pos)):
            cv2.circle(self.frame, self.ball_pos[i][0], 2, (0, 0, 255), 2)

if __name__ == "__main__":
    ShotDetector(num_images_to_save=5)  # Specify the number of images to save
