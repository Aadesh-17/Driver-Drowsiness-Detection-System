import cv2
import dlib
import time
import threading
import numpy as np
import os
from scipy.spatial import distance as dist
from imutils import face_utils
import pygame
pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)

def eye_aspect_ratio(eye):
    # Compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    
    # Compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])
    
    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    # Compute the euclidean distances between the three sets
    # of vertical mouth landmarks (x, y)-coordinates
    A = dist.euclidean(mouth[2], mouth[10]) # 51, 59
    B = dist.euclidean(mouth[4], mouth[8])  # 53, 57
    
    # Compute the euclidean distance between the horizontal
    # mouth landmark (x, y)-coordinates
    C = dist.euclidean(mouth[0], mouth[6])  # 49, 55
    
    # Compute the mouth aspect ratio
    mar = (A + B) / (2.0 * C)
    return mar

def create_alarm_sound():
    """Create a simple beep sound using pygame"""
    def beep_sound():
        duration = 1000  # milliseconds
        sample_rate = 22050
        frames = int(duration * sample_rate / 1000)
        arr = np.zeros((frames, 2))
        for i in range(frames):
            wave = 4096 * np.sin(2 * np.pi * 1000 * i / sample_rate)
            arr[i][0] = wave
            arr[i][1] = wave
        sound = pygame.sndarray.make_sound(arr.astype(np.int16))
        sound.play()
        pygame.time.wait(duration)
    return beep_sound

def sound_alarm(path):
    try:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
            pygame.time.wait(2000)
        else:
            backup_alarm()
    except Exception as e:
        print(f"Alarm error: {e}")
        print('\a' * 3)

# Improved thresholds
EYE_AR_THRESH = 0.3     # Increased threshold for better accuracy
EYE_AR_CONSEC_FRAMES = 15  # Reduced frames for faster detection
MOUTH_AR_THRESH = 0.5   # Threshold for yawning detection
YAWN_CONSEC_FRAMES = 15

# Counters
EYE_COUNTER = 0
MOUTH_COUNTER = 0
ALARM_ON = False
LAST_ALARM_TIME = 0

# Initialize face detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# Get facial landmark indices
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
time.sleep(1.0)

print("=" * 50)
print("Enhanced Driver Drowsiness Detection System")
print("=" * 50)
print("Features:")
print("- Eye closure detection (EAR)")
print("- Yawning detection (MAR)")
print("- Multi-level audio alerts")
print("- Real-time feedback")
print("\nControls: Press ESC to quit")
print("=" * 50)

# Create backup alarm function
backup_alarm = create_alarm_sound()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break
        
    # Resize frame for better performance
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = detector(gray, 0)
    
    if len(faces) == 0:
        cv2.putText(frame, "No face detected!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    for face in faces:
        # Get facial landmarks
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)
        
        # Extract eye and mouth coordinates
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        mouth = shape[mStart:mEnd]
        
        # Calculate ratios
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        mar = mouth_aspect_ratio(mouth)
        
        # Draw eye and mouth contours
        leftHull = cv2.convexHull(leftEye)
        rightHull = cv2.convexHull(rightEye)
        mouthHull = cv2.convexHull(mouth)
        
        cv2.drawContours(frame, [leftHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 255), 1)
        
        # Current time for alarm throttling
        current_time = time.time()
        
        # Drowsiness detection (closed eyes)
        if ear < EYE_AR_THRESH:
            EYE_COUNTER += 1
            
            if EYE_COUNTER >= EYE_AR_CONSEC_FRAMES:
                # Throttle alarms to prevent spam (minimum 3 seconds between alarms)
                if current_time - LAST_ALARM_TIME > 3.0:
                    LAST_ALARM_TIME = current_time
                    threading.Thread(target=sound_alarm, args=('audio/alarm.wav',), daemon=True).start()
                    print(f"DROWSINESS ALERT! EAR: {ear:.3f}")
                
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "WAKE UP!", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            EYE_COUNTER = 0
            
        # Yawning detection
        if mar > MOUTH_AR_THRESH:
            MOUTH_COUNTER += 1
            
            if MOUTH_COUNTER >= YAWN_CONSEC_FRAMES:
                cv2.putText(frame, "YAWNING DETECTED!", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Light alarm for yawning (less intrusive)
                if current_time - LAST_ALARM_TIME > 5.0:
                    LAST_ALARM_TIME = current_time
                    print('\a')  # Simple beep
                    print(f"Yawning detected! MAR: {mar:.3f}")
        else:
            MOUTH_COUNTER = 0
            
        # Display metrics
        cv2.putText(frame, f"EAR: {ear:.3f}", (480, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"MAR: {mar:.3f}", (480, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Eye Counter: {EYE_COUNTER}", (480, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Mouth Counter: {MOUTH_COUNTER}", (480, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Status indicators
        if ear < EYE_AR_THRESH:
            cv2.circle(frame, (600, 150), 10, (0, 0, 255), -1)  # Red for drowsy
        else:
            cv2.circle(frame, (600, 150), 10, (0, 255, 0), -1)  # Green for alert
            
        if mar > MOUTH_AR_THRESH:
            cv2.circle(frame, (580, 150), 10, (255, 255, 0), -1)  # Yellow for yawn
        else:
            cv2.circle(frame, (580, 150), 10, (100, 100, 100), -1)  # Gray normal
    
    # Show frame
    cv2.imshow("Enhanced Driver Drowsiness Detection", frame)
    
    # Exit on ESC key
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC key
        break
    elif key == ord('t'):  # Test alarm with 't' key
        print("Testing alarm...")
        threading.Thread(target=sound_alarm, args=('audio/alarm.wav',), daemon=True).start()

# Cleanup
print("\nShutting down...")
cap.release()
cv2.destroyAllWindows()
