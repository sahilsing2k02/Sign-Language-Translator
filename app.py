# python -m venv venv
# source venv/bin/activate
# pip install -r requirements.txt

from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import pickle
import cv2
import mediapipe as mp
import numpy as np
import warnings

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

warnings.filterwarnings("ignore")

# -------------------- Flask Setup --------------------

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

# -------------------- Load Model --------------------

try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
    print("Model loaded successfully!")
except Exception as e:
    print("Error loading model:", e)
    model = None

# -------------------- Routes --------------------

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')

# -------------------- Video Generator --------------------

def generate_frames():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("No camera available!")
        return

    base_options = python.BaseOptions(
        model_asset_path='hand_landmarker.task'
    )

    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2
    )

    detector = vision.HandLandmarker.create_from_options(options)

    labels_dict = {
        0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E',
        5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
        10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
        15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
        20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y',
        25: 'Z', 26: 'Hello', 27: 'Done',
        28: 'Thank You', 29: 'I Love You',
        30: 'Sorry', 31: 'Please',
        32: 'You are welcome'
    }

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            H, W, _ = frame.shape

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=frame_rgb
            )

            results = detector.detect(mp_image)

            if results.hand_landmarks:
                for hand_landmarks in results.hand_landmarks:

                    data_aux = []
                    x_, y_ = [], []

                    # Draw landmarks manually (no protobuf conversion needed)
                    for lm in hand_landmarks:
                        cx = int(lm.x * W)
                        cy = int(lm.y * H)
                        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

                        x_.append(lm.x)
                        y_.append(lm.y)

                    # Normalize data
                    for lm in hand_landmarks:
                        data_aux.append(lm.x - min(x_))
                        data_aux.append(lm.y - min(y_))

                    x1 = int(min(x_) * W) - 10
                    y1 = int(min(y_) * H) - 10
                    x2 = int(max(x_) * W) - 10
                    y2 = int(max(y_) * H) - 10

                    if model:
                        try:
                            prediction = model.predict(
                                [np.asarray(data_aux)]
                            )

                            prediction_proba = model.predict_proba(
                                [np.asarray(data_aux)]
                            )

                            confidence = max(prediction_proba[0])
                            predicted_character = labels_dict[
                                int(prediction[0])
                            ]

                            socketio.emit('prediction', {
                                'text': predicted_character,
                                'confidence': confidence
                            })

                            cv2.rectangle(
                                frame,
                                (x1, y1),
                                (x2, y2),
                                (0, 0, 0),
                                3
                            )

                            cv2.putText(
                                frame,
                                f"{predicted_character} ({confidence*100:.2f}%)",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0, 0, 0),
                                2
                            )

                        except Exception as e:
                            print("Prediction error:", e)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' +
                frame +
                b'\r\n'
            )

    finally:
        cap.release()
        cv2.destroyAllWindows()

# -------------------- Video Route --------------------

@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

# -------------------- Run App --------------------

if __name__ == '__main__':
    socketio.run(app, debug=False)
