import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import pickle
import os

MODEL_FILE = "gesture_model.pkl"

# Check files
if not os.path.exists(MODEL_FILE):
    print("Error: Model not found. Run train_landmarks.py first.")
    exit()

# Load Model
print("Loading Neural Network model...")
with open(MODEL_FILE, 'rb') as f:
    model = pickle.load(f)
print(f"Model loaded. Classes: {model.classes_}")

detector = HandDetector(maxHands=1, detectionCon=0.8)
cap = cv2.VideoCapture(0)

print("Gesture Control Started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    hands, img = detector.findHands(frame, flipType=False)
    
    gesture_name = "..."
    confidence_score = 0.0
    
    if hands:
        hand = hands[0]
        lmList = hand['lmList']
        
        if len(lmList) >= 21:
            # Preprocess (Same as training)
            data = []
            base_x, base_y, base_z = lmList[0][0], lmList[0][1], lmList[0][2]
            scale_ref = np.linalg.norm(np.array(lmList[5]) - np.array(lmList[0]))
            if scale_ref == 0: scale_ref = 1
            
            for lm in lmList:
                rel_x = (lm[0] - base_x) / scale_ref
                rel_y = (lm[1] - base_y) / scale_ref
                rel_z = (lm[2] - base_z) / scale_ref
                data.extend([rel_x, rel_y, rel_z])
            
            # Predict
            try:
                # Scikit-learn expects 2D array
                input_data = [data]
                
                # Get probabilities
                probabilities = model.predict_proba(input_data)[0]
                class_index = np.argmax(probabilities)
                confidence_score = probabilities[class_index]
                
                prediction = model.classes_[class_index]
                
                if confidence_score > 0.8:
                    gesture_name = prediction
                else:
                    gesture_name = "UNCERTAIN"
            except Exception as e:
                print(f"Prediction Error: {e}")
    
    # UI
    color = (0, 255, 0) if gesture_name != "UNCERTAIN" else (0, 0, 255)
    cv2.putText(frame, f"Gesture: {gesture_name}", (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(frame, f"Conf: {confidence_score:.2f}", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    cv2.imshow("Neural Network Control", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
