import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import csv
import os
import time

# Initialize CVZone Hand Detector
detector = HandDetector(maxHands=1, detectionCon=0.8)

DATA_FILE = "hand_data.csv"

# Check if file exists to write headers
if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        # 21 landmarks * 3 coordinates (x, y, z) + label
        header = []
        for i in range(21):
            header.extend([f"x{i}", f"y{i}", f"z{i}"])
        header.append("label")
        writer.writerow(header)

def save_landmarks(lmList, label):
    data = []
    if len(lmList) < 21: return

    # Normalize
    base_x, base_y, base_z = lmList[0][0], lmList[0][1], lmList[0][2]
    scale_ref = np.linalg.norm(np.array(lmList[5]) - np.array(lmList[0]))
    if scale_ref == 0: scale_ref = 1

    for lm in lmList:
        rel_x = (lm[0] - base_x) / scale_ref
        rel_y = (lm[1] - base_y) / scale_ref
        rel_z = (lm[2] - base_z) / scale_ref
        data.extend([rel_x, rel_y, rel_z])
    
    data.append(label)
    
    with open(DATA_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(data)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 60) # Try to request 60 FPS

current_label = None
recording = False
counter = 0

print("Smart Collector Started")
print("Press 'u', 'd', 'l', 'r' to START recording that gesture.")
print("Press 's' to STOP recording.")
print("Press 'q' to QUIT.")

while True:
    ret, frame = cap.read()
    if not ret: break
        
    frame = cv2.flip(frame, 1)
    hands, img = detector.findHands(frame, flipType=False)
    
    # Key Controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord('s'): 
        recording = False
        current_label = None
        print("Recording STOPPED")
    elif key == ord('u'): 
        recording = True
        current_label = "UP"
        print("Recording UP...")
    elif key == ord('d'): 
        recording = True
        current_label = "DOWN"
        print("Recording DOWN...")
    elif key == ord('l'): 
        recording = True
        current_label = "LEFT"
        print("Recording LEFT...")
    elif key == ord('r'): 
        recording = True
        current_label = "RIGHT"
        print("Recording RIGHT...")

    # Recording Logic
    if hands and recording and current_label:
        hand = hands[0]
        lmList = hand['lmList']
        save_landmarks(lmList, current_label)
        counter += 1
    
    # UI
    if recording:
        cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1) # Red recording dot
        cv2.putText(frame, f"REC: {current_label} ({counter})", (50, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame, f"Total Samples: {counter}", (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Smart Data Collector", frame)

cap.release()
cv2.destroyAllWindows()
