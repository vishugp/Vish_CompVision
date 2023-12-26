import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands = 2, min_detection_confidence = 0.7)
mp_draw = mp.solutions.drawing_utils

model = load_model('mp_hand_gesture')