import os
import cv2
import time
import yaml
import torch
import numpy as np
from torch import nn
import mediapipe as mp
from src.deployment.controller import ModbusMaster
from src.utils.hand_landmarks_detector import HandLandmarksDetector
from src.utils.label_dict import label_dict_from_config_file
from src.model.neural_network import NeuralNetwork

class LightGesture:
    def __init__(self, model_path, device=False):
        self.device = device
        self.height = 720
        self.width = 1280

        self.detector = HandLandmarksDetector()
        self.status_text = None
        self.signs = label_dict_from_config_file("data/hand_gesture.yaml")
        self.classifier = NeuralNetwork()
        self.classifier.load_state_dict(torch.load(model_path))
        self.classifier.eval()

        if self.device:
            self.controller = ModbusMaster()
        self.light1 = False
        self.light2 = False
        self.light3 = False
    
    def light_simulation(self, img, lights):
        # Append a white rectangle at the bottom of the image
        height, width, _ = img.shape
        rect_height = int(0.15 * height)
        rect_width = width
        white_rect = np.ones((rect_height, rect_width, 3), dtype=np.uint8) * 255

        # Draw a red border around the rectangle
        cv2.rectangle(white_rect, (0, 0), (rect_width, rect_height), (0, 0, 255), 2)

        # Calculate circle positions
        circle_radius = int(0.45*rect_height)
        circle1_center = (int(rect_width * 0.25), int(rect_height / 2))
        circle2_center = (int(rect_width * 0.5), int(rect_height / 2))
        circle3_center = (int(rect_width * 0.75), int(rect_height / 2))

        # Draw the circles
        on_color = (0, 255, 255)
        off_color = (0, 0, 0)
        colors = [off_color, on_color]
        circle_centers = [circle1_center, circle2_center, circle3_center]
        for cc, light in zip(circle_centers, lights):
            color = colors[int(light)]
            cv2.circle(white_rect, cc, circle_radius, color, -1)

        # Append the white rectangle to the bottom of the image
        img = np.vstack((img, white_rect))
        return img

    def run(self):
        cam = cv2.VideoCapture(0)
        cam.set(3, 1280)
        cam.set(4, 720)

        while cam.isOpened():
            _, frame = cam.read()

            hand, img = self.detector.detectHand(frame)
            if len(hand) != 0:
                with torch.no_grad():
                    hand_landmark = torch.from_numpy(np.array(hand[0], dtype=np.float32).flatten()).unsqueeze(0)
                    class_number = self.classifier.predict(hand_landmark).item()
                    if class_number != -1:
                        self.status_text = self.signs[class_number]

                        if self.status_text == "light1":
                            if self.light1 == False:
                                print("lights on")
                                self.light1 = True
                                if self.device:
                                    self.controller.switch_actuator_1(True)
                        elif self.status_text == "light2":
                            if self.light2 == False:
                                self.light2 = True
                                if self.device:
                                    self.controller.switch_actuator_2(True)
                        elif self.status_text == "light3":
                            if self.light3 == False:
                                self.light3 = True
                                if self.device:
                                    self.controller.switch_actuator_3(True)          
                        elif self.status_text == "turn_on":
                            self.light1 = self.light2 = self.light3 = True    
                            if self.light1 and self.light2 and self.light3:
                                pass
                            else:
                                self.light1 = self.light2 = self.light3 = True
                                if self.device:
                                    self.controller.switch_actuator_1(self.light1)
                                    time.sleep(0.03)
                                    self.controller.switch_actuator_2(self.light2)
                                    time.sleep(0.03)
                                    self.controller.switch_actuator_3(self.light3)                                       
                        elif self.status_text == "turn_off":
                            if not self.light1 and not self.light2 and not self.light3:
                                pass
                            else:
                                self.light1 = self.light2 = self.light3 = False
                                if self.device:
                                    self.controller.switch_actuator_1(self.light1)
                                    time.sleep(0.03)
                                    self.controller.switch_actuator_2(self.light2)
                                    time.sleep(0.03)
                                    self.controller.switch_actuator_3(self.light3)
                                
                    else:
                        self.status_text = "undefined command"
                        
            else:
                self.status_text = None

            img = self.light_simulation(img, [self.light1, self.light2, self.light3])

            cv2.putText(img, self.status_text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.namedWindow('window', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('window', 1920, 1080)
            cv2.imshow("window", img)
            key = cv2.waitKey(1)
            if key == ord("q"):
                break
        cv2.destroyAllWindows()        

def main():
    model_path = './models/model_27-11 12_38_NeuralNetwork_best'
    light = LightGesture(model_path, device=False)
    light.run()
    

if __name__ == "__main__":
    main()
