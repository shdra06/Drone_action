import cv2
import numpy as np
import mediapipe as mp
import math
import time
from collections import deque

# ---------------- INIT ----------------
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

mp_draw = mp.solutions.drawing_utils

# ---------------- VARS ----------------
history = deque(maxlen=5)
stable_direction = "..."

# Drone Simulation
drone_x, drone_y = 320, 240

# ---------------- LfD VARS ----------------
mode = "IDLE"  # IDLE, COUNTDOWN, RECORDING, EXECUTING
trajectory = []
waypoints = []
exec_index = 0
countdown_start_time = 0
# -------------------------------------


def detect_gesture(landmarks):
    """
    Simple gesture detection:
    FIST: All fingers curled.
    PALM: All fingers extended.
    """
    # Finger tip IDs: 8, 12, 16, 20 (Index, Middle, Ring, Pinky)
    # Finger dip IDs (one joint down): 6, 10, 14, 18
    tips = [8, 12, 16, 20]
    dips = [6, 10, 14, 18]
    
    fingers_open = 0
    for tip, dip in zip(tips, dips):
        # If tip is higher (smaller y) than dip, finger is OPEN
        if landmarks.landmark[tip].y < landmarks.landmark[dip].y:
            fingers_open += 1
            
    # Thumb check (tip 4 vs ip 3) - x comparison for simplicity
    if landmarks.landmark[4].x > landmarks.landmark[3].x: # Right hand specific, simplistic
         pass 

    if fingers_open == 0:
        return "FIST"
    elif fingers_open == 4:
        return "PALM"
    return "UNKNOWN"


def smooth_path(path, window=5):
    if not path: return []
    smooth = []
    for i in range(len(path)):
        xs, ys = [], []
        for j in range(max(0, i - window), min(len(path), i + window)):
            xs.append(path[j][0])
            ys.append(path[j][1])
        smooth.append((int(sum(xs) / len(xs)), int(sum(ys) / len(ys))))
    return smooth


def finger_direction_from_roi(roi):
    """
    Takes cropped hand ROI and returns direction
    using fingertip + palm center geometry
    """
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower_skin = np.array([0, 30, 60], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    contours, _ = cv2.findContours(mask,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, roi

    hand = max(contours, key=cv2.contourArea)
    if cv2.contourArea(hand) < 1000:
        return None, roi

    M = cv2.moments(hand)
    if M["m00"] == 0:
        return None, roi

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    hull = cv2.convexHull(hand)

    max_dist = 0
    fingertip = None
    for p in hull:
        x, y = p[0]
        dist = math.hypot(x - cx, y - cy)
        if dist > max_dist:
            max_dist = dist
            fingertip = (x, y)

    if fingertip is None:
        return None, roi

    dx = fingertip[0] - cx
    dy = cy - fingertip[1]

    if abs(dx) > abs(dy):
        direction = "RIGHT" if dx > 0 else "LEFT"
    else:
        direction = "UP" if dy > 0 else "DOWN"

    # Visualization on ROI
    cv2.circle(roi, (cx, cy), 6, (255, 0, 0), -1)
    cv2.circle(roi, fingertip, 8, (0, 0, 255), -1)
    cv2.line(roi, (cx, cy), fingertip, (255, 255, 0), 2)

    return direction, roi


print("Hand detector + finger direction started. Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    direction = "..."

    current_gesture = "UNKNOWN"
    fingertip_pt = None

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Gesture Detection
            current_gesture = detect_gesture(hand_landmarks)

            # Get Fingertip (Index Tip is ID 8)
            tip_x = int(hand_landmarks.landmark[8].x * w)
            tip_y = int(hand_landmarks.landmark[8].y * h)
            fingertip_pt = (tip_x, tip_y)

            # Get bounding box (existing logic)
            xs = [lm.x for lm in hand_landmarks.landmark]
            ys = [lm.y for lm in hand_landmarks.landmark]
            x1, y1 = int(min(xs) * w) - 20, int(min(ys) * h) - 20
            x2, y2 = int(max(xs) * w) + 20, int(max(ys) * h) + 20
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            # Draw Hand
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Existing Direction Logic (Only in IDLE)
            if mode == "IDLE":
                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    direction, roi_vis = finger_direction_from_roi(roi)
                    frame[y1:y2, x1:x2] = roi_vis
    
    # ---------------- STATE MACHINE ----------------
    if mode == "IDLE":
        if current_gesture == "FIST":
            mode = "COUNTDOWN"
            countdown_start_time = time.time()
            print(">>> COUNTDOWN STARTED")

    elif mode == "COUNTDOWN":
        elapsed = time.time() - countdown_start_time
        remaining = 3 - int(elapsed)
        
        if remaining <= 0:
            mode = "RECORDING"
            trajectory = []
            print(">>> RECORDING STARTED")
        else:
            # Display Countdown
            cv2.putText(frame, str(remaining), (w//2 - 50, h//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 5)

    
    elif mode == "RECORDING":
        if current_gesture == "PALM":
            mode = "EXECUTING"
            
            # Trim the last 15 points (approx 0.5s) to remove "opening hand" noise
            if len(trajectory) > 15:
                trajectory = trajectory[:-15]
            
            waypoints = smooth_path(trajectory)[::5] # Downsample
            exec_index = 0
            print(f">>> PLAYBACK STARTED: {len(waypoints)} waypoints")
        elif fingertip_pt:
             trajectory.append(fingertip_pt)
             
    elif mode == "EXECUTING":
        # Check if we have waypoints to follow
        if exec_index < len(waypoints):
            target = waypoints[exec_index]
            # Move drone towards target
            dx = target[0] - drone_x
            dy = target[1] - drone_y
            dist = math.hypot(dx, dy)
            
            if dist < 10: # Reached waypoint
                exec_index += 1
            else:
                # Move at constant speed
                speed = 5
                drone_x += (dx / dist) * speed
                drone_y += (dy / dist) * speed
        else:
            mode = "IDLE"
            print(">>> PLAYBACK FINISHED")

    # ---------------- STABILIZATION (IDLE ONLY) ----------------
    if mode == "IDLE":
        history.append(direction)
        if history.count(direction) >= 3:
            stable_direction = direction
        
        # Manual Control
        if stable_direction == "UP": drone_y -= 5
        elif stable_direction == "DOWN": drone_y += 5
        elif stable_direction == "LEFT": drone_x -= 5
        elif stable_direction == "RIGHT": drone_x += 5

    # ---------------- DRAWING ----------------
    # Draw Trajectory (Green = Recording)
    if len(trajectory) > 1:
        cv2.polylines(frame, [np.array(trajectory)], False, (0, 255, 0), 2)
        
    # Draw Waypoints (Blue = Planned Path)
    if len(waypoints) > 1:
        cv2.polylines(frame, [np.array(waypoints)], False, (255, 0, 0), 2)

    # Keep drone on screen
    drone_x = max(10, min(w - 10, drone_x))
    drone_y = max(10, min(h - 10, drone_y))

    # Draw Drone
    cv2.circle(frame, (int(drone_x), int(drone_y)), 10, (0, 0, 255), -1)

    # UI Information
    info_text = f"Mode: {mode} | Gesture: {current_gesture}"
    if mode == "IDLE": info_text += f" | Dir: {stable_direction}"
    
    cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Hand Detector → Finger Direction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
