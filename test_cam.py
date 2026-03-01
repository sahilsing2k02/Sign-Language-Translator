import cv2
print("OpenCV version:", cv2.__version__)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera 0")
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Cannot open camera 1")
        exit()
    else:
        print("Camera 1 opened successfully")
else:
    print("Camera 0 opened successfully")

ret, frame = cap.read()
if ret:
    print(f"Captured frame with shape {frame.shape}")
else:
    print("Failed to capture frame")
cap.release()
