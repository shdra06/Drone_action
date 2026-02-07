import cv2
import numpy as np

IMG_SIZE = 128
FRAMES = 5

cap = cv2.VideoCapture(0)
buffer = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    buffer.append(resized)

    if len(buffer) > FRAMES:
        buffer.pop(0)

    stacked = np.stack(buffer, axis=-1)

    cv2.imshow("Current Frame", resized)
    print("Buffer shape:", stacked.shape)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
