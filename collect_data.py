import cv2
import numpy as np
import os

IMG_SIZE = 128
FRAMES = 5
SAVE_DIR = "dataset"

os.makedirs(SAVE_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)
buffer = []

label = input("Enter gesture label (left/right/up/down/curve): ")
count = 0

print("Recording... Press Q to stop")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    buffer.append(resized)

    if len(buffer) > FRAMES:
        buffer.pop(0)

    if len(buffer) == FRAMES:
        stack = np.stack(buffer, axis=-1)
        filename = f"{SAVE_DIR}/{label}_{count}.npy"
        np.save(filename, stack)
        count += 1
        cv2.putText(frame, f"Saved: {count}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Recording", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Done. Samples saved:", count)
