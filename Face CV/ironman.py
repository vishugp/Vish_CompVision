import cv2
import mediapipe as mp

mp_draw = mp.solutions.drawing_utils
mp_face = mp.solutions.face_mesh
mp_hand = mp.solutions.hands

cap = cv2.VideoCapture(0)

with mp_face.FaceMesh(min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as faces:
    with mp_hand.Hands(min_detection_confidence=0.5,min_tracking_confidence=0.5) as hands:

        while cap.isOpened():
            success, img = cap.read()
            if not success:
                print("Camera Feed is empty!")
                continue
            
            img = cv2.cvtColor(cv2.flip(img,1) , cv2.COLOR_BGR2RGB)
            img.flags.writeable = False
            resulth = hands.process(img)
            resultf = faces.process(img)

            img.flags.writeable = True
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if resulth.multi_hand_landmarks:
               for hand_landmarks in resulth.multi_hand_landmarks:
                   mp_draw.draw_landmarks(img, hand_landmarks, mp_hand.HAND_CONNECTIONS
                   ,mp_draw.DrawingSpec(color=(0,215,255),thickness=4, circle_radius=4)
                   ,mp_draw.DrawingSpec(color=(0,0,255),thickness = 6, circle_radius=2))
            
            if resultf.multi_face_landmarks:
                for face_landmarks in resultf.multi_face_landmarks:

                    mp_draw.draw_landmarks(
                        image=img,
                        landmark_list=face_landmarks,
                        connections=mp_face.FACE_CONNECTIONS,
                        landmark_drawing_spec = mp_draw.DrawingSpec(color=(0,215,255),thickness = 2, circle_radius=1),
                        connection_drawing_spec = mp_draw.DrawingSpec(color=(0,0,255),thickness = 6, circle_radius=2))

        
            cv2.imshow("IRON MAN", img)

            if cv2.waitKey(5) & 0xFF == 27:
                print(face_landmarks)
                print("#####################################################")
                print(hand_landmarks)
                break
    
cap.release()
