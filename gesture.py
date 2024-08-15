import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

def recognize_gesture(hand_landmarks):
    thumb_up = False
    thumb_down = False
    closed_wrist = False

    for landmarks in hand_landmarks:
        coords = [(lm.x, lm.y) for lm in landmarks.landmark]
        
        # Check for thumb up and thumb down gestures
        thumb_tip = coords[mp_hands.HandLandmark.THUMB_TIP.value]
        thumb_ip = coords[mp_hands.HandLandmark.THUMB_IP.value]
        thumb_angle = np.arctan2(thumb_tip[1] - thumb_ip[1], thumb_tip[0] - thumb_ip[0])
        
        if thumb_angle < -1.5:
            thumb_up = True
        if thumb_angle > 1.5:
            thumb_down = True
        
        # Check for closed wrist gesture
        wrist_landmark = coords[mp_hands.HandLandmark.WRIST.value]
        hand_center = np.mean([coords[i] for i in range(5)], axis=0)
        if np.linalg.norm(np.array(wrist_landmark) - np.array(hand_center)) < 0.05:
            closed_wrist = True
    
    index_finger_open = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP.value].y < landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP.value].y
    middle_finger_open = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP.value].y < landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP.value].y
    
    if thumb_down:
        return 'clear'
    if thumb_up:
        return 'submit'
    if closed_wrist:
        return 'undo'
    if index_finger_open and middle_finger_open:
        return 'draw'
    
    return 'none'
