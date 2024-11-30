import mediapipe as mp

mp_hands = mp.solutions.hands

def count_raised_fingers(hand_landmarks_list):
    total_count = 0
    tips_ids = [mp_hands.HandLandmark.THUMB_TIP, 
                mp_hands.HandLandmark.INDEX_FINGER_TIP, 
                mp_hands.HandLandmark.MIDDLE_FINGER_TIP, 
                mp_hands.HandLandmark.RING_FINGER_TIP, 
                mp_hands.HandLandmark.PINKY_TIP]
    pip_ids = [mp_hands.HandLandmark.THUMB_IP, 
               mp_hands.HandLandmark.INDEX_FINGER_PIP, 
               mp_hands.HandLandmark.MIDDLE_FINGER_PIP, 
               mp_hands.HandLandmark.RING_FINGER_PIP, 
               mp_hands.HandLandmark.PINKY_PIP]

    for hand_landmarks in hand_landmarks_list:
        count = 0
        for tip, pip in zip(tips_ids, pip_ids):
            if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
                count += 1
        total_count += count

    return total_count

def get_head_gesture(face_landmarks):
    nose_tip = face_landmarks.landmark[1]  # Nose tip
    if nose_tip.x < 0.4:
        return "Looking left"
    elif nose_tip.x > 0.6:
        return "Looking right"
    elif nose_tip.y < 0.4:
        return "Looking up"
    elif nose_tip.y > 0.6:
        return "Looking down"
    else:
        return "Looking straight"
