import cv2
import numpy as np
import tensorflow as tf
from collections import deque

# ================= CONFIG =================
IMG_SIZE = 128
FRAMES = 5

CLASSES = ["up", "down", "left", "right", "none"]

CONF_THRESHOLD = 0.6        # prediction confidence gate
MOTION_THRESHOLD = 10       # motion energy gate
VOTE_WINDOW = 7             # majority voting window
LOCK_HOLD_FRAMES = 8        # direction lock duration
# =========================================

# Load trained model
model = tf.keras.models.load_model("model.h5")

cap = cv2.VideoCapture(0)
buffer = []

history = deque(maxlen=VOTE_WINDOW)
locked_gesture = None
lock_frames = 0


def motion_energy(frames):
    """Measure pixel movement between last two frames"""
    diff = np.abs(frames[-1] - frames[-2])
    return np.mean(diff)


print("Live gesture recognition started. Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ---------- PREPROCESS ----------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    buffer.append(resized)

    if len(buffer) > FRAMES:
        buffer.pop(0)

    display_gesture = "..."

    # ---------- PREDICTION ----------
    if len(buffer) == FRAMES:
        stack = np.stack(buffer, axis=-1)
        energy = motion_energy(stack)

        # Predict only if motion exists
        if energy > MOTION_THRESHOLD:
            stack = stack / 255.0
            stack = np.expand_dims(stack, axis=0)

            pred = model.predict(stack, verbose=0)[0]
            confidence = np.max(pred)
            gesture = CLASSES[np.argmax(pred)]

            if confidence > CONF_THRESHOLD:
                history.append(gesture)

    # ---------- MAJORITY VOTING ----------
    if len(history) > 0:
        voted = max(set(history), key=history.count)
    else:
        voted = "none"

    # ---------- DIRECTION LOCK ----------
    if lock_frames == 0 and voted != "none":
        locked_gesture = voted
        lock_frames = LOCK_HOLD_FRAMES

    if lock_frames > 0:
        if locked_gesture != "none":
            display_gesture = locked_gesture
        lock_frames -= 1
    else:
        display_gesture = "..."

    # ---------- DISPLAY ----------
    cv2.putText(
        frame,
        f"Gesture: {display_gesture.upper()}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3
    )

    cv2.imshow("Live Gesture Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
