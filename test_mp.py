import cv2
import warnings
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Suppress specific warnings
warnings.filterwarnings("ignore", message="SymbolDatabase.GetPrototype() is deprecated")

print("DEBUG: generate_frames called")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("DEBUG: Camera 0 failed to open!")
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("DEBUG: Camera 1 also failed to open!")
        exit()
else:
    print("DEBUG: Camera 0 opened successfully!")

try:
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options,
                                           num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)
    print("Successfully instantiated hands module")
    
    ret, frame = cap.read()
    if ret:
        print(f"Captured frame shape: {frame.shape}")
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        results = detector.detect(mp_image)
        print("Successfully processed frame with mediapipe!")
    else:
        print("Failed to capture frame")
except Exception as e:
    print(f"Error during execution: {e}")
finally:
    cap.release()
