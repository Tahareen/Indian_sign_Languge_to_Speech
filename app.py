from flask import Flask, render_template, Response, Blueprint, jsonify, request

import cv2
import mediapipe as mp
import time
import numpy as np
import math
from collections import deque
import threading
import game
import base64
import os

app = Flask(__name__)

# Create Blueprint for the learning page
learn_bp = Blueprint('learn', __name__, template_folder='templates')

# Load MediaPipe classes
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.RunningMode
VisionRunningMode = mp.tasks.vision.RunningMode
gesture_name = "No Gesture"
gesture_score = 0.0

# Global variables for web app
finger_spelling_buffer = []
total_blinks = 0
output_frame = None
lock = threading.Lock()

def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    global gesture_name, gesture_score
    if result.gestures:
        for gesture in result.gestures:
            gesture_name = gesture[0].category_name
            gesture_score = float(gesture[0].score)
            print("Gesture:", gesture[0].category_name, "Score:", gesture[0].score)

# Model paths
one_hand_model_path = 'gesture_recognizer.task'
two_hand_model_path = 'two_hand_model_new.task'

print(f"One-hand model path: {one_hand_model_path}")
print(f"Two-hand model path: {two_hand_model_path}")

# Model options
options_one_hand = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=one_hand_model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result,
    num_hands=1,
    min_hand_detection_confidence=0.4,
    min_hand_presence_confidence=0.4,
    min_tracking_confidence=0.4,
)

options_two_hand = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=two_hand_model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result,
    num_hands=2,
    min_hand_detection_confidence=0.4,
    min_hand_presence_confidence=0.4,
    min_tracking_confidence=0.4,
)

# Recognizer instances
recognizer_one_hand = GestureRecognizer.create_from_options(options_one_hand)
recognizer_two_hand = GestureRecognizer.create_from_options(options_two_hand)

current_recognizer = recognizer_one_hand
current_num_hands = 1

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

hands = mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=2
)

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

landmark_color = (0, 0, 255)

last_call_time = 0
recognition_interval = 200  # in ms

def calculate_EAR(eye_landmarks, image_width, image_height):
    def euclidean(p1, p2):
        return math.dist(p1, p2)
    points = [(int(l.x * image_width), int(l.y * image_height)) for l in eye_landmarks]
    vertical1 = euclidean(points[1], points[5])
    vertical2 = euclidean(points[2], points[4])
    horizontal = euclidean(points[0], points[3])
    EAR = (vertical1 + vertical2) / (2.0 * horizontal)
    return EAR

EAR_THRESHOLD = 0.3
EAR_CONSEC_FRAMES = 3
ear_buffer = deque(maxlen=EAR_CONSEC_FRAMES)

gesture_prediction_buffer = deque(maxlen=3)

blink_in_progress = False
camera = None
is_streaming = True



def get_camera():
    """Initialize and return the camera object."""
    return cv2.VideoCapture(0)

def generate_frames():
    """Video streaming generator function."""
    global output_frame, camera, gesture_name, finger_spelling_buffer, total_blinks
    global current_recognizer, current_num_hands, last_call_time, blink_in_progress
    global ear_buffer, gesture_prediction_buffer
    
    if camera is None:
        camera = get_camera()
        # Allow camera sensor to warm up
        time.sleep(2.0)
    
    while is_streaming:
        ret, frame = camera.read()
        if not ret:
            print("Failed to grab frame")
            continue

        frame_flipped = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame_flipped, cv2.COLOR_BGR2RGB)

        current_time_ms = int(time.time() * 1000)
        results = hands.process(frame_rgb)
        face_results = face_mesh.process(frame_rgb)

        if results.multi_hand_landmarks:
            num_hands_detected = len(results.multi_hand_landmarks)
        else:
            num_hands_detected = 0

        if num_hands_detected == 1 and current_num_hands != 1:
            print("Switching to one hand model")
            current_recognizer = recognizer_one_hand
            current_num_hands = 1
        elif num_hands_detected == 2 and current_num_hands != 2:
            print("Switching to two hand model")
            current_recognizer = recognizer_two_hand
            current_num_hands = 2

        if current_time_ms - last_call_time >= recognition_interval:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            current_recognizer.recognize_async(mp_image, current_time_ms)
            last_call_time = current_time_ms
            gesture_prediction_buffer.append(gesture_name)

        if results.multi_hand_landmarks:
            hand_boxes = []

            for hand_landmarks in results.multi_hand_landmarks:
                x_coords = [landmark.x for landmark in hand_landmarks.landmark]
                y_coords = [landmark.y for landmark in hand_landmarks.landmark]

                x_min = int(min(x_coords) * frame_flipped.shape[1])
                x_max = int(max(x_coords) * frame_flipped.shape[1])
                y_min = int(min(y_coords) * frame_flipped.shape[0])
                y_max = int(max(y_coords) * frame_flipped.shape[0])

                hand_boxes.append((x_min, y_min, x_max, y_max))

                mp_drawing.draw_landmarks(
                    frame_flipped,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2),
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                    )
                for landmark in hand_landmarks.landmark:
                    cv2.circle(frame_flipped,
                                (int(landmark.x * frame_flipped.shape[1]), int(landmark.y * frame_flipped.shape[0])),
                                5, landmark_color, -1)

            if len(hand_boxes) == 2:
                x_min = min(hand_boxes[0][0], hand_boxes[1][0])
                y_min = min(hand_boxes[0][1], hand_boxes[1][1])
                x_max = max(hand_boxes[0][2], hand_boxes[1][2])
                y_max = max(hand_boxes[0][3], hand_boxes[1][3])

                cv2.rectangle(frame_flipped, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame_flipped, gesture_name, (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)

            elif len(hand_boxes) == 1:
                x_min, y_min, x_max, y_max = hand_boxes[0]
                cv2.rectangle(frame_flipped, (x_min, y_min), (x_max, y_max), (0, 0, 0), 2)
                cv2.putText(frame_flipped, gesture_name, (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                #Left eye outline
                mp_drawing.draw_landmarks(
                image=frame_flipped,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_LEFT_EYE,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=1, circle_radius=1)
                )

                #Right eye Outline
                mp_drawing.draw_landmarks(
                    image=frame_flipped,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_RIGHT_EYE,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=1, circle_radius=1)
                )

                # Eye landmarks
                left_eye_indices = [33, 160, 158, 133, 153, 144]  # Reordered
                right_eye_indices = [362, 385, 387, 263, 373, 380]

                for idx in left_eye_indices + right_eye_indices:
                    x = int(face_landmarks.landmark[idx].x * frame_flipped.shape[1])
                    y = int(face_landmarks.landmark[idx].y * frame_flipped.shape[0])
                    cv2.circle(frame_flipped, (x, y), 2, (0, 0, 0), -1)

                left_eye_landmarks = [face_landmarks.landmark[i] for i in [33, 160, 158, 133, 153, 144]]
                right_eye_landmarks = [face_landmarks.landmark[i] for i in [362, 387, 385, 263, 373, 380]]

                left_EAR = calculate_EAR(left_eye_landmarks, frame_flipped.shape[1], frame_flipped.shape[0])
                right_EAR = calculate_EAR(right_eye_landmarks, frame_flipped.shape[1], frame_flipped.shape[0])
                avg_EAR = (left_EAR + right_EAR) / 2.0

                cv2.putText(frame_flipped, f"EAR: {avg_EAR:.2f}", (10, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (179, 5, 127), 2)

                ear_buffer.append(avg_EAR)
                ear_avg = np.mean(ear_buffer)

                if ear_avg < EAR_THRESHOLD and not blink_in_progress:
                    blink_in_progress = True
                    total_blinks += 1
                    if gesture_prediction_buffer:
                        letter = max(set(gesture_prediction_buffer), key=gesture_prediction_buffer.count)
                        if letter not in ["none", "No Gesture"]:
                            finger_spelling_buffer.append(letter)
                            print("Buffer:", ''.join(finger_spelling_buffer))
                elif ear_avg >= EAR_THRESHOLD:
                    blink_in_progress = False

        # Calculate the width and height of the text
        text = ''.join(finger_spelling_buffer)
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_width, text_height = text_size

        # Get the center position for the text
        x = (frame_flipped.shape[1] - text_width) // 2  # Center horizontally
        y = frame_flipped.shape[0] - 20  # Near the bottom (20 pixels from the bottom)

        # Display blink count
        cv2.putText(frame_flipped, f"Blinks: {total_blinks}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (13, 5, 163), 2)

        # Display the finger-spelling buffer
        cv2.putText(frame_flipped, f"CONTENT: {''.join(finger_spelling_buffer)}", (0, y),
                        cv2.FONT_HERSHEY_TRIPLEX, 1, (255,255,255), 2)
        
        with lock:
            output_frame = frame_flipped.copy()
            
        # Encode the output frame as JPEG
        ret, buffer = cv2.imencode('.jpg', output_frame)
        if not ret:
            continue
            
        # Convert to bytes and yield
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Main application routes
@app.route("/")
def home():
    return render_template("home.html")

@app.route('/main')
def Speak():
    return render_template('Speak.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_text')
def get_text():
    global texts
    global finger_spelling_buffer
    
    if finger_spelling_buffer:
        cleaned_text = finger_spelling_buffer
        
        # If the text contains commas, clean it up
        if isinstance(cleaned_text, str) and ',' in cleaned_text:
            cleaned_text = cleaned_text.replace(',', '')
        
        # If it's a list or tuple, join it without commas
        if isinstance(cleaned_text, (list, tuple)):
            cleaned_text = ''.join(cleaned_text)
        
        # Important: Return the cleaned text, not the original!
        return jsonify({'text': cleaned_text})
    
    return jsonify({'text': ''})  # Return empty JSON object instead of empty string

@app.route('/add_space')
def add_space():
    global finger_spelling_buffer
    finger_spelling_buffer.append(' ')
    return '', 204

@app.route('/backspace')
def backspace():
    global finger_spelling_buffer
    if finger_spelling_buffer:
        finger_spelling_buffer.pop()
    return '', 204


@app.route('/clear')
def clear():
    global finger_spelling_buffer
    finger_spelling_buffer.clear()
    return '', 204
    
@app.route('/stop')
def stop_stream():
    global is_streaming, camera
    is_streaming = False
    if camera:
        camera.release()
        camera = None
    return "Stream stopped"

@app.route('/start')
def start_stream():
    global is_streaming, camera
    if not is_streaming:
        is_streaming = True
        if camera is None or not camera.isOpened():
            camera = get_camera()
        return "Stream started"
    return "Stream already running"


# Learning page routes
@learn_bp.route('/')
def learn_page():
    """Render the learning page"""
    return render_template('learn.html')

@learn_bp.route('/process_frame', methods=['POST'])
def process_frame():
    """Process webcam frames for sign recognition"""
    global recognizer_one_hand, recognizer_two_hand, gesture_name, gesture_score
    
    # Get data from request
    data = request.json
    image_data = data['image'].split(',')[1]  # Remove the data:image/jpeg;base64 part
    target_sign = data['target_sign']
    
    # Decode base64 image
    image_bytes = base64.b64decode(image_data)
    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    # Get current timestamp
    current_time_ms = int(time.time() * 1000)
    
    # Process with MediaPipe Hands to detect number of hands
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=2
    ) as hands:
        # Convert to RGB
        frame_flipped = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        # Determine number of hands
        num_hands_detected = 0
        hand_boxes = []

         # Choose the appropriate model based on number of hand
        
        if results.multi_hand_landmarks:
            num_hands_detected = len(results.multi_hand_landmarks)
            
            for hand_landmarks in results.multi_hand_landmarks:
                x_coords = [landmark.x for landmark in hand_landmarks.landmark]
                y_coords = [landmark.y for landmark in hand_landmarks.landmark]

                x_min = int(min(x_coords) * frame_flipped.shape[1])
                x_max = int(max(x_coords) * frame_flipped.shape[1])
                y_min = int(min(y_coords) * frame_flipped.shape[0])
                y_max = int(max(y_coords) * frame_flipped.shape[0])
                
                # Add some padding
                # padding = 20
                # x_min = max(0, x_min - padding)
                # y_min = max(0, y_min - padding)
                # x_max = min(w, x_max + padding)
                # y_max = min(h, y_max + padding)
                
                hand_boxes.append((x_min, y_min, x_max, y_max))

                mp_drawing.draw_landmarks(
                    frame_flipped,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2),
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                    )
                for landmark in hand_landmarks.landmark:
                    cv2.circle(frame_flipped,
                                (int(landmark.x * frame_flipped.shape[1]), int(landmark.y * frame_flipped.shape[0])),
                                5, landmark_color, -1)

        # Choose the appropriate model based on number of hands
        if num_hands_detected == 1:
            current_recognizer = recognizer_one_hand
        elif num_hands_detected == 2:
            current_recognizer = recognizer_two_hand
        else:
            # No hands detected
            return jsonify({
                'gesture': 'No hands detected',
                'score': 0.0,
                'hand_boxes': [],
                'match': False
            })
                
        
       
        
        if len(hand_boxes) == 2:
                x_min = min(hand_boxes[0][0], hand_boxes[1][0])
                y_min = min(hand_boxes[0][1], hand_boxes[1][1])
                x_max = max(hand_boxes[0][2], hand_boxes[1][2])
                y_max = max(hand_boxes[0][3], hand_boxes[1][3])

                cv2.rectangle(frame_flipped, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame_flipped, gesture_name, (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)

        elif len(hand_boxes) == 1:
                x_min, y_min, x_max, y_max = hand_boxes[0]
                cv2.rectangle(frame_flipped, (x_min, y_min), (x_max, y_max), (0, 0, 0), 2)
                cv2.putText(frame_flipped, gesture_name, (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        # Process the frame with MediaPipe Gesture Recognizer
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        current_recognizer.recognize_async(mp_image, current_time_ms)
        
        # Small delay to allow the callback to process
        time.sleep(0.05)
        
        # Check if the detected gesture matches the target sign
        is_match = False
        if gesture_name.lower() == target_sign.lower():
            is_match = True
        
        return jsonify({
            'gesture': gesture_name,
            'score': round(gesture_score * 100, 2),  # Convert to percentage
            'hand_boxes': hand_boxes,
            'match': is_match
        })

# Register blueprints
game.get_game_routes(app)
app.register_blueprint(learn_bp, url_prefix='/learn')

# Clean up resources when application exits
def cleanup():
    global recognizer_one_hand, recognizer_two_hand, camera
    try:
        if camera:
            camera.release()
        
        # Use a try-except block to safely close recognizers
        if recognizer_one_hand:
            try:
                recognizer_one_hand.close()
            except Exception as e:
                print(f"Error closing one_hand_recognizer: {e}")
                
        if recognizer_two_hand:
            try:
                recognizer_two_hand.close()
            except Exception as e:
                print(f"Error closing two_hand_recognizer: {e}")
    except Exception as e:
        print(f"Error during cleanup: {e}")

@app.teardown_appcontext
def teardown_app(exception):
    # Only call cleanup on final shutdown, not during request handling
    if not app.debug or os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
        cleanup()

if __name__ == '__main__':
    # Register atexit handler for proper cleanup
    import atexit
    atexit.register(cleanup)
    
    # Start camera thread
    camera = get_camera()
    
    # Run Flask app - use reloader=False to avoid duplicate resource initialization
    # app.run(host='0.0.0.0', port=5000, debug=True, threaded=True, use_reloader=False)