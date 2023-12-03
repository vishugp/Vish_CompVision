# IMPORTING LIBRARIES
import cv2
import mediapipe as mp

### MEDIAPIPE LIBRARY FUNCTIONS FOR LANDMARKS
mp_draw = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

### CV2 Capturing the Video in cap
cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.65, min_tracking_confidence=0.65) as hands:
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
                ,mp_draw.DrawingSpec(color=(0,215,255),thickness=4, circle_radius=4)
                ,mp_draw.DrawingSpec(color=(0,0,255),thickness = 8, circle_radius=2))
            count = 0    
            for index in [8,12,16,20]:
                d1 = hand_landmarks.landmark[index].y - hand_landmarks.landmark[0].y
                d2 = hand_landmarks.landmark[index-3].y - hand_landmarks.landmark[0].y
                if d1<d2 :
                    count+=1
            d1 = hand_landmarks.landmark[4].x - hand_landmarks.landmark[0].x
            d2 = hand_landmarks.landmark[3].x - hand_landmarks.landmark[0].x
            # print(result.multi_handedness)
           
            if d2<0:
                if(d1<d2):
                    count+=1
            else:
                if(d1>d2):
                    count+=1
            
            print(count)
        else:
            count = 0
            print(0)

        cv2.putText(img, str(count), (10,30), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, color = (255,0,0), thickness=2)

        # img = cv2.resize(img , (1000,700))
        cv2.imshow("Hand Number Detector", img)

        if cv2.waitKey(5) & 0xFF == 27:
            break


