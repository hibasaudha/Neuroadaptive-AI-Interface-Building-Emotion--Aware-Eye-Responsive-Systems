import cv2
import numpy as np
import tensorflow as tf
from multiprocessing import Process, Queue, Event
import time
import os

# --- CONFIGURATION & CONSTANTS ---
# NOTE: Ensure these model files exist in the same directory!
# EYE_MODEL_PATH is REMOVED/REPLACED by EAR calculation
EMOTION_MODEL_PATH = "emotion_updated_model.h5" 
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml" 

# EAR CONSTANTS
EAR_THRESHOLD = 0.25 # Typical threshold for blink detection
EAR_CLASSES = ['open_eyes', 'closed_eyes'] # New classification for EAR

EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral", "stress"]

# Input sizes for the models
# EYE_IMG_SIZE is no longer strictly required for a pure EAR approach
EMOTION_IMG_SIZE = 48 

# --- COMMUNICATION FILE ---
OUTPUT_FILE = "cv_output.txt" 
# This file will store the latest predicted state for the chat client to read.

# --- GLOBAL STATE (Internal to the Server) ---
# Renamed from current_eye_label to current_ear_label
current_ear_label = "N/A (0.00)" 
current_emotion_label = "N/A (0.00)"
face_coordinates = (0, 0, 0, 0)

# ==============================
# HELPER FUNCTION FOR EAR CALCULATION (Conceptual Placeholder)
# ==============================

def calculate_ear(face_roi_bgr):
    """
    Conceptual placeholder for Eye Aspect Ratio (EAR) calculation.
    
    NOTE: A full implementation requires a library like Dlib or MediaPipe 
    to detect 68/5-point facial landmarks and isolate the eye coordinates.
    For this example, we'll simulate the output based on a simple size check 
    or a placeholder to keep the worker logic flow.
    """
    # In a real application, you would:
    # 1. Detect facial landmarks on face_roi_bgr (or full frame)
    # 2. Extract the 6 points for each eye (p1 to p6)
    # 3. Compute EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    
    # *** SIMULATED OUTPUT for demonstration of worker flow ***
    # Simulate a value based on time or a random generator for now.
    simulated_ear_value = 0.15 + (np.sin(time.time() * 5) + 1) * 0.15 
    
    # Assign label based on a threshold
    if simulated_ear_value < EAR_THRESHOLD:
        label = 'closed_eyes'
        # Confidence is the difference from the threshold, capped at 1.0
        confidence = min(1.0, (EAR_THRESHOLD - simulated_ear_value) / EAR_THRESHOLD)
    else:
        label = 'open_eyes'
        confidence = min(1.0, (simulated_ear_value - EAR_THRESHOLD) / EAR_THRESHOLD)
        
    return label, confidence, simulated_ear_value # Return the value for debug/confidence calc

# ==============================
# WORKER FUNCTIONS (RUN IN SEPARATE PROCESSES)
# ==============================

def emotion_worker(input_queue, output_queue, stop_event):
    """Worker process for Emotion Detection (48x48, Grayscale). (UNMODIFIED)"""
    # Disable eager execution for better performance in parallel processes
    tf.config.experimental.set_visible_devices([], 'GPU')
    print("Emotion Worker: Loading model...")
    
    try:
        # Load Keras model inside the process
        emotion_model = tf.keras.models.load_model(EMOTION_MODEL_PATH, compile=False) 
    except Exception as e:
        print(f"Emotion Worker Error: Failed to load model. Please check path. {e}")
        return

    while not stop_event.is_set():
        try:
            # Non-blocking get with timeout
            data = input_queue.get(timeout=0.001) 
            roi_gray = data[0]
        except:
            continue
        
        # 1. Preprocess specific to Emotion Model (48x48, Grayscale)
        roi_resized = cv2.resize(roi_gray, (EMOTION_IMG_SIZE, EMOTION_IMG_SIZE))
        proc_img = roi_resized.astype("float") / 255.0
        proc_img = np.expand_dims(proc_img, axis=-1)
        proc_img = np.expand_dims(proc_img, axis=0)

        # 2. Predict
        preds = emotion_model.predict(proc_img, verbose=0)[0]
        label = EMOTION_CLASSES[np.argmax(preds)]
        confidence = np.max(preds)
        
        # 3. Send result back: ('emotion', 'sad', 0.95)
        output_queue.put(('emotion', label, confidence))
    
    print("Emotion Worker: Shutting down.")


def ear_worker(input_queue, output_queue, stop_event):
    """Worker process for Eye Aspect Ratio (EAR) Blink Detection."""
    print("EAR Worker: Initializing...")
    
    while not stop_event.is_set():
        try:
            # Non-blocking get with timeout
            data = input_queue.get(timeout=0.001)
            roi_bgr = data[0]
        except:
            continue

        # 1. Calculate EAR
        if roi_bgr.size != 0:
            label, confidence, ear_value = calculate_ear(roi_bgr)
        else:
            continue
            
        # 2. Send result back: ('ear', 'closed_eyes', 0.92)
        # We pass the EAR value as confidence (conceptually) for display
        output_queue.put(('ear', label, ear_value)) 

    print("EAR Worker: Shutting down.")

def start_cv_server():
    """Initializes and runs the continuous video processing loop."""
    # Renamed from current_eye_label to current_ear_label
    global current_emotion_label, current_ear_label, face_coordinates
    
    # Initialize Queues and Stop Event
    emotion_q = Queue()
    ear_q = Queue() # Renamed queue
    result_q = Queue()
    stop_event = Event()

    # Start Worker Processes
    emotion_p = Process(target=emotion_worker, args=(emotion_q, result_q, stop_event))
    # Renamed process
    ear_p = Process(target=ear_worker, args=(ear_q, result_q, stop_event)) 
    
    emotion_p.start()
    ear_p.start() # Start the EAR worker
    
    # Load Face Detector in the Main Process
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    if face_cascade.empty():
        print("ERROR: Could not load Haar cascade. Check path.")
        stop_event.set()
        
    cap = cv2.VideoCapture(0)

    # Write initial N/A state to the communication file
    with open(OUTPUT_FILE, 'w') as f:
        f.write("N/A|N/A")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 1. Face Detection (Run ONCE per frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

            if len(faces) > 0:
                # Use the largest/first detected face
                x, y, w, h = faces[0] 
                face_coordinates = (x, y, w, h)
                
                # Prepare ROIs (Region of Interest) for the two models
                roi_gray = gray[y:y+h, x:x+w]
                roi_bgr = frame[y:y+h, x:x+w]
                
                # 2. Send the raw face crops to the respective workers
                if roi_gray.size != 0 and roi_bgr.size != 0:
                    emotion_q.put((roi_gray, x, y, w, h))
                    ear_q.put((roi_bgr, x, y, w, h)) # Send to EAR worker

            # 3. Collect Results (Non-blocking check)
            while not result_q.empty():
                worker_type, label, value = result_q.get()
                
                # The label must be extracted without the confidence for fusion logic
                if worker_type == 'emotion':
                    current_emotion_label = f"{label} ({value:.2f})"
                # Check for 'ear' results
                elif worker_type == 'ear': 
                    # Use the EAR value in place of confidence for display
                    current_ear_label = f"{label} (EAR: {value:.2f})"
            
            # 4. Write the latest combined state to the file for the chat client to read
            # Format: EMOTION_LABEL|EAR_STATE
            with open(OUTPUT_FILE, 'w') as f:
                # We extract only the label name (before the space/parenthesis) for clean fusion later
                emotion_name = current_emotion_label.split(' ')[0]
                ear_name = current_ear_label.split(' ')[0]
                f.write(f"{emotion_name}|{ear_name}")


            # 5. Display Combined Results (Optional, for visual debugging)
            x, y, w, h = face_coordinates
            if w > 0:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Emotion: {current_emotion_label}", (x, y - 25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                # Display the new EAR label
                cv2.putText(frame, f"Eye: {current_ear_label}", (x, y + h + 25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2) 
            
            cv2.imshow("Combined Parallel Detector (CV Server)", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Shutdown all processes safely
        print("\nCV Server: Shutting down workers...")
        stop_event.set()
        emotion_p.join(timeout=1)
        ear_p.join(timeout=1) # Join the EAR worker
        
        cap.release()
        cv2.destroyAllWindows()
        print("CV Server: Cleanup complete. Program finished.")

if __name__ == '__main__':
    start_cv_server()