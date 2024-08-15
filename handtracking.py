import cv2
import mediapipe as mp
import numpy as np

def initialize_hand_tracking():
    """
    Initializes MediaPipe Hands and Drawing utilities.
    """
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    mp_draw = mp.solutions.drawing_utils
    return hands, mp_hands, mp_draw

def process_frame(frame, hands):
    """
    Processes a frame to detect hands using MediaPipe.
    """
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    return results

def draw_hand_landmarks(frame, results, mp_draw):
    """
    Draws hand landmarks on the frame and displays which hand is detected.
    """
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
            
        if results.multi_handedness:
            for i, hand in enumerate(results.multi_handedness):
                hand_label = hand.classification[0].label
                # Correctly assign hand labels based on camera orientation
                corrected_label = 'Right Hand' if hand_label == 'Left' else 'Left Hand'
                # Adjust text position for each detected hand
                cv2.putText(frame, corrected_label, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

def main():
    cap = cv2.VideoCapture(0)
    hands, mp_hands, mp_draw = initialize_hand_tracking()

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Flip the frame horizontally if the image appears mirrored
        frame = cv2.flip(frame, 1)

        results = process_frame(frame, hands)
        draw_hand_landmarks(frame, results, mp_draw)

        cv2.imshow('Hand Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
