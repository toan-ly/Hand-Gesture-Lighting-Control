import os
import cv2
import csv
import yaml
import numpy as np
import mediapipe as mp
from src.utils.hand_landmarks_detector import HandLandmarksDetector
from src.utils.label_dict import label_dict_from_config_file

class HandDatasetWriter():
    def __init__(self, filepath) -> None:
        self.csv_file = open(filepath, "a")
        self.file_writer = csv.writer(self.csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

    def add(self, hand, label):
        self.file_writer.writerow([label, *np.array(hand).flatten().tolist()])

    def close(self):
        self.csv_file.close()


def is_handsign_character(char:str):
    return ord('a') <= ord(char) < ord('q') or char == ' '

def run(LABEL_TAG, data_path, sign_img_path, split="val", resolution=(1280, 720)):
    hand_detector = HandLandmarksDetector()

    # Camera setup
    cam = cv2.VideoCapture(0)
    cam.set(3, resolution[0])
    cam.set(4, resolution[1])

    os.makedirs(data_path, exist_ok=True)
    os.makedirs(sign_img_path, exist_ok=True)
    print(sign_img_path)

    dataset_path = f"./{data_path}/landmark_{split}.csv"
    hand_dataset = HandDatasetWriter(dataset_path)

    current_letter = None # track current letter being recorded
    status_text = None 
    cannot_switch_char = False # prevent switching char without unbinding current one
    saved_frame = None
    while cam.isOpened():
        _, frame = cam.read()
        hands, annotated_image = hand_detector.detectHand(frame)
        
        if(current_letter is None):
            status_text = "Press a character to record"
        else:
            label = ord(current_letter) - ord("a")
            if label == -65: # spacebar  
                status_text = "Recording unknown, press spacebar again to stop"
                label = -1
            else:
                status_text = f"Recording {LABEL_TAG[label]}, press {current_letter} again to stop"

        key = cv2.waitKey(1)
        if(key == -1):
            if(current_letter is None ):
                # no current letter recording, just skip it
                pass
            else:
                if len(hands) != 0: # hands detected
                    hand = hands[0]
                    hand_dataset.add(hand=hand, label=label)
                    saved_frame = frame
        else: # some key is pressed
            # pressed some key, do not push this image, assign current letter to the key just pressed
            key = chr(key)
            if key == "q": # terminate if q is pressed
                break

            if (is_handsign_character(key)):
                if (current_letter is None):
                    current_letter = key
                elif (current_letter == key):
                    # pressed again?, reset the current state
                    if saved_frame is not None:
                        if label >=0:
                            cv2.imwrite(f"./{sign_img_path}/{LABEL_TAG[label]}.jpg", saved_frame)

                    cannot_switch_char = False
                    current_letter = None
                    saved_frame = None
                else:
                    cannot_switch_char = True
                    # warned user to unbind the current_letter first
        if(cannot_switch_char):
            cv2.putText(annotated_image, f"please press {current_letter} again to unbind", (0, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.putText(annotated_image, status_text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow(f"{split}", annotated_image)
    cv2.destroyAllWindows()

def main():
    config_path = "./data/hand_gesture.yaml"
    LABEL_TAG = label_dict_from_config_file(config_path)
    data_path = './data/raw'
    sign_img_path = os.path.join(data_path, 'sign_img')

    for split in ["train", "val", "test"]:
        run(LABEL_TAG, data_path, sign_img_path, split, (1280, 720))

if __name__ == "__main__":
    main()
   
