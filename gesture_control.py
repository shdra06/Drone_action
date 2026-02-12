import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import pickle
import os
import math
import time

MODEL_FILE = "gesture_model.pkl"

if not os.path.exists(MODEL_FILE):
    print("Error: Model not found. Run train_landmarks.py first.")
    exit()

print("Loading Neural Network...")
with open(MODEL_FILE, 'rb') as f:
    model = pickle.load(f)
print(f"Model loaded. Classes: {model.classes_}")

detector = HandDetector(maxHands=1, detectionCon=0.8)
cap = cv2.VideoCapture(0)

# State Variables
mode = "IDLE"
countdown_start = 0
trajectory = []
drone_pos = [100, 100]
path_index = 0

# Smoothing & Logic Variables
smooth_window = []
SMOOTH_FACTOR = 5 
victory_start_time = 0 

def simplify_path(points, epsilon=5.0):
    if len(points) < 3: return points
    curve = np.array(points)
    simplified = cv2.approxPolyDP(curve, epsilon, False)
    return simplified.reshape(-1, 2).tolist()

print("Control Started.")
print("1. HOLD 'VICTORY' (Peace Sign) to Start.")
print("2. Track path (Index Finger).")
print("3. Show 'OPEN' (Hand Open) to Stop & Execute.")

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    # 1. Detection
    hands, img = detector.findHands(frame, flipType=False)
    
    gesture_name = "..."
    confidence_score = 0.0
    index_finger_tip = None
    
    if hands:
        hand = hands[0]
        lmList = hand['lmList']
        raw_tip = (lmList[8][0], lmList[8][1])
        
        # Smooth Cursor
        smooth_window.append(raw_tip)
        if len(smooth_window) > SMOOTH_FACTOR:
            smooth_window.pop(0)
        
        avg_x = sum([p[0] for p in smooth_window]) / len(smooth_window)
        avg_y = sum([p[1] for p in smooth_window]) / len(smooth_window)
        index_finger_tip = (int(avg_x), int(avg_y))
        
        if len(lmList) >= 21:
            data = []
            base_x, base_y, base_z = lmList[0][0], lmList[0][1], lmList[0][2]
            scale_ref = np.linalg.norm(np.array(lmList[5]) - np.array(lmList[0]))
            if scale_ref == 0: scale_ref = 1
            
            for lm in lmList:
                rel_x = (lm[0] - base_x) / scale_ref
                rel_y = (lm[1] - base_y) / scale_ref
                rel_z = (lm[2] - base_z) / scale_ref
                data.extend([rel_x, rel_y, rel_z])
            
            try:
                probabilities = model.predict_proba([data])[0]
                class_index = np.argmax(probabilities)
                confidence_score = probabilities[class_index]
                prediction = model.classes_[class_index]
                
                if confidence_score > 0.8: 
                    gesture_name = prediction
                else:
                    gesture_name = "UNCERTAIN"
            except: pass

    # 2. State Machine Logic
    
    # --- IDLE ---
    if mode == "IDLE":
        # Require HOLDING Victory for 0.5s to start
        if gesture_name == "VICTORY":
            if victory_start_time == 0:
                victory_start_time = time.time()
            elif time.time() - victory_start_time > 0.5:
                mode = "COUNTDOWN"
                countdown_start = time.time()
                trajectory = []
                victory_start_time = 0
                if index_finger_tip: drone_pos = list(index_finger_tip)
                print(">>> Countdown Started")
        else:
            victory_start_time = 0 

    
    # --- COUNTDOWN ---
    elif mode == "COUNTDOWN":
        elapsed = time.time() - countdown_start
        remaining = 3 - int(elapsed)
        
        cv2.putText(frame, str(remaining), (w//2, h//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 165, 255), 5)
        
        if remaining <= 0:
            mode = "RECORDING"
            smooth_window = [] 
            print(">>> Recording Started")
        
        # Abort check (if user opens hand fully before start)
        if gesture_name == "OPEN":
            mode = "IDLE"; print(">>> Aborted")

    # --- RECORDING ---
    elif mode == "RECORDING":
        # Stop Condition: OPEN
        if gesture_name == "OPEN":
            mode = "EXECUTING"
            path_index = 0
            
            # 1. Trim 300ms (10 frames) to remove hand-opening motion
            if len(trajectory) > 10:
                trajectory = trajectory[:-10]
            
            # 2. Simplify Path
            original_len = len(trajectory)
            trajectory = simplify_path(trajectory, epsilon=5.0)
            new_len = len(trajectory)
            
            if trajectory:
                drone_pos = list(trajectory[0])
            
            print(f">>> Execution Started. Points: {original_len} -> {new_len}")
            
        elif index_finger_tip:
             trajectory.append(index_finger_tip)

    # --- EXECUTING ---
    elif mode == "EXECUTING":
        if gesture_name == "SPECIAL":
            mode = "IDLE"; trajectory = []; print(">>> Reset")
            
        elif trajectory and path_index < len(trajectory):
            target = trajectory[path_index]
            dx = target[0] - drone_pos[0]
            dy = target[1] - drone_pos[1]
            dist = math.hypot(dx, dy)
            
            speed = 10
            if dist < speed:
                drone_pos = list(target)
                path_index += 1
            else:
                drone_pos[0] += (dx / dist) * speed
                drone_pos[1] += (dy / dist) * speed
        else:
            mode = "IDLE"; print(">>> Execution Finished")

    # 3. Drawing
    # Draw Trajectory
    if len(trajectory) > 1:
        color = (0, 0, 255) 
        if mode == "RECORDING": color = (0, 255, 0)
        elif mode == "EXECUTING": color = (255, 0, 0)
        
        pts = np.array(trajectory, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], False, color, 3) 

        if mode == "EXECUTING":
            for p in trajectory:
                cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0,255,255), -1)

    # Cursor
    if index_finger_tip:
        cv2.circle(frame, index_finger_tip, 8, (0, 255, 0), -1)
        
    # Drone
    cv2.circle(frame, (int(drone_pos[0]), int(drone_pos[1])), 15, (0, 0, 255), -1)
    
    # UI Info
    # Draw Hold Progress for Victory
    if mode == "IDLE" and victory_start_time > 0:
        elapsed_hold = time.time() - victory_start_time
        progress = int((elapsed_hold / 0.5) * 100)
        cv2.rectangle(frame, (10, 110), (10 + progress, 120), (0, 255, 0), -1)
        cv2.putText(frame, "HOLD VICTORY", (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.putText(frame, f"Mode: {mode}", (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(frame, f"Gesture: {gesture_name}", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    cv2.imshow("Gesture Pattern Control", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
