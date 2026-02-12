import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import csv
import os

# Initialize CVZone Hand Detector
detector = HandDetector(maxHands=1, detectionCon=0.8)

DATA_FILE = "hand_data.csv"

# Check if file exists to write headers
if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        # 21 landmarks * 3 coordinates + label
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

current_label = None
recording = False
counter = 0

print("Smart Collector Started")
print("New: 'v' (VICTORY), 'f' (FIST), 'o' (OPEN)")
print("Standard: 'u', 'd', 'l', 'r', 's' (SPECIAL)")
print("Press 'q' to QUIT, 'x' key to STOP")

while True:
    ret, frame = cap.read()
    if not ret: break
        
    frame = cv2.flip(frame, 1)
    hands, img = detector.findHands(frame, flipType=False)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord('x'): 
        recording = False; current_label = None; print("Recording STOPPED")
    elif key == ord('u'): 
        recording = True; current_label = "UP"; print("REC: UP")
    elif key == ord('d'): 
        recording = True; current_label = "DOWN"; print("REC: DOWN")
    elif key == ord('l'): 
        recording = True; current_label = "LEFT"; print("REC: LEFT")
    elif key == ord('r'): 
        recording = True; current_label = "RIGHT"; print("REC: RIGHT")
    elif key == ord('f'): 
        recording = True; current_label = "FIST"; print("REC: FIST")
    elif key == ord('o'): 
        recording = True; current_label = "OPEN"; print("REC: OPEN")
    elif key == ord('v'): 
        recording = True; current_label = "VICTORY"; print("REC: VICTORY")
    elif key == ord('s'): 
        recording = True; current_label = "SPECIAL"; print("REC: SPECIAL")

    if hands and recording and current_label:
        hand = hands[0]
        lmList = hand['lmList']
        save_landmarks(lmList, current_label)
        counter += 1
    
    if recording:
        cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)
        cv2.putText(frame, f"REC: {current_label} ({counter})", (50, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, "Press 'x' to STOP", (50, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.putText(frame, f"Total Samples: {counter}", (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Data Collector", frame)

cap.release()
cv2.destroyAllWindows()
