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

def chaikin_smooth(points, iterations=2):
    if len(points) < 3: return points
    for _ in range(iterations):
        new_points = [points[0]]
        for i in range(len(points) - 1):
            p0 = points[i]; p1 = points[i+1]
            Q = (0.75 * p0[0] + 0.25 * p1[0], 0.75 * p0[1] + 0.25 * p1[1])
            R = (0.25 * p0[0] + 0.75 * p1[0], 0.25 * p0[1] + 0.75 * p1[1])
            new_points.append(Q); new_points.append(R)
        new_points.append(points[-1])
        points = new_points
    return points

def detect_and_snap_shape(points):
    """
    Analyzes the path to detect geometric shapes (Circle, Rect, Triangle).
    Returns a new set of points representing the perfect shape if detected.
    """
    if len(points) < 10: return points # Too short to be a shape
    
    # Convert to contour format for OpenCV
    contour = np.array(points).astype(np.int32)
    
    # 1. Closed Shape Check?
    # Distance between start and end
    dist = np.linalg.norm(contour[0] - contour[-1])
    perimeter = cv2.arcLength(contour, False) # Open arc length
    
    # If endpoints are close relative to perimeter (e.g., < 30%), assume closed loop
    # Relaxed to 0.3 to handle cases where the user stops a bit early or trim eats the end
    is_closed = dist < (perimeter * 0.3)
    
    if is_closed:
        # Auto-close the contour for detection
        contour_closed = np.vstack([contour, contour[0]])
        perimeter_closed = cv2.arcLength(contour_closed, True)
        approx = cv2.approxPolyDP(contour_closed, 0.04 * perimeter_closed, True)
        
        # Calculate Check Metrics
        area = cv2.contourArea(approx)
        if area < 1000: return points # Too small
        
        # Vertices count in approximation
        vtc = len(approx)
        
        # --- SHAPE LOGIC ---
        
        # TRIANGLE (3 vertices)
        if vtc == 3:
            print(">>> Snapped to TRIANGLE")
            return approx.reshape(-1, 2).tolist()
        
        # RECTANGLE / SQUARE (4 vertices)
        elif vtc == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            
            # Generate perfect box points
            box_points = [
                (x, y),
                (x + w, y),
                (x + w, y + h),
                (x, y + h),
                (x, y) # Close loop
            ]
            
            if 0.8 <= aspect_ratio <= 1.2:
                print(">>> Snapped to SQUARE")
            else:
                print(">>> Snapped to RECTANGLE")
            return box_points
            
        # CIRCLE (> 4 vertices, high circularity)
        else:
            # Circularity = 4*pi*Area / P^2
            k = (4 * np.pi * area) / (perimeter_closed * perimeter_closed)
            
            if k > 0.7:
                print(">>> Snapped to CIRCLE")
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)
                
                # Generate points for a circle
                circle_points = []
                for i in range(361): # 0 to 360 degrees
                    angle = math.radians(i)
                    px = center[0] + radius * math.cos(angle)
                    py = center[1] + radius * math.sin(angle)
                    circle_points.append((px, py))
                return circle_points

    # Default: Not a recognized closed geometric shape
    # Apply standard smoothing (RDP + Chaikin)
    
    # RDP first
    simplified = cv2.approxPolyDP(contour, 5.0, False).reshape(-1, 2).tolist()
    # Chaikin second
    smoothed = chaikin_smooth(simplified, iterations=3)
    return smoothed

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
        
        # Real-time smooth for cursor
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
                
                if confidence_score > 0.8: gesture_name = prediction
                else: gesture_name = "UNCERTAIN"
            except: pass

    # 2. State Machine Logic
    
    # --- IDLE ---
    if mode == "IDLE":
        if gesture_name == "VICTORY":
            if victory_start_time == 0: victory_start_time = time.time()
            elif time.time() - victory_start_time > 0.5:
                mode = "COUNTDOWN"
                countdown_start = time.time()
                trajectory = []
                victory_start_time = 0
                if index_finger_tip: drone_pos = list(index_finger_tip)
                print(">>> Countdown Started")
        else: victory_start_time = 0 
    
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
        
        if gesture_name == "OPEN": mode = "IDLE"; print(">>> Aborted")

    # --- RECORDING ---
    elif mode == "RECORDING":
        if gesture_name == "OPEN":
            mode = "EXECUTING"
            path_index = 0
            
            # Trim (Reduced to 5 to save the end of shapes)
            if len(trajectory) > 5: trajectory = trajectory[:-5]
            
            original_len = len(trajectory)
            
            # --- SHAPE DETECTION & SNAPPING ---
            trajectory = detect_and_snap_shape(trajectory)
            
            new_len = len(trajectory)
            if trajectory: drone_pos = list(trajectory[0])
            
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
    if len(trajectory) > 1:
        color = (0, 0, 255) 
        if mode == "RECORDING": color = (0, 255, 0)
        elif mode == "EXECUTING": color = (255, 0, 0)
        
        # Convert floats to int for polylines
        pts = np.array(trajectory).astype(np.int32)
        pts = pts.reshape((-1, 1, 2))
        
        # Handle closed shapes (Circles/Rects) vs Open paths
        is_closed = (len(trajectory) > 2 and 
                     np.linalg.norm(np.array(trajectory[0]) - np.array(trajectory[-1])) < 5)
        
        cv2.polylines(frame, [pts], is_closed, color, 3, lineType=cv2.LINE_AA) 

    # Cursor & Drone
    if index_finger_tip: cv2.circle(frame, index_finger_tip, 8, (0, 255, 0), -1)
    cv2.circle(frame, (int(drone_pos[0]), int(drone_pos[1])), 15, (0, 0, 255), -1)
    
    # UI
    if mode == "IDLE" and victory_start_time > 0:
        elapsed = time.time() - victory_start_time
        progress = int((elapsed / 0.5) * 100)
        cv2.rectangle(frame, (10, 110), (10 + progress, 120), (0, 255, 0), -1)
        cv2.putText(frame, "HOLD VICTORY", (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.putText(frame, f"Mode: {mode}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(frame, f"Gesture: {gesture_name}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    cv2.imshow("Gesture Pattern Control", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
