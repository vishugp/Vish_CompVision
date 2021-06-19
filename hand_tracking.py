import cv2
import mediapipe as mp


def dist(a,b):
    ans = ( (b.x**2-a.x**2)**2 + (b.y**2-a.y**2)**2 + (b.z**2-a.z**2)**2 )**0.5
    print(ans)
mp_draw = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        success, image = cap.read()

        img = cv2.cvtColor(cv2.flip(image,1), cv2.COLOR_BGR2RGB)
        img.flags.writeable = False

        result = hands.process(img)

        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS
                ,mp_draw.DrawingSpec(color=(0,215,255),thickness=2, circle_radius=4)
                ,mp_draw.DrawingSpec(color=(0,0,255),thickness = 2, circle_radius=2))
                index_tip = hand_landmarks.landmark[8]
                thumb_tip = hand_landmarks.landmark[4]
                # dist(index_tip,thumb_tip)
        
        print(mp_hands.HAND_CONNECTIONS)

        # img = cv2.resize(img , (1000,700))
        cv2.imshow("Hand Detector", img)

        if cv2.waitKey(5) & 0xFF == 27:
            break


