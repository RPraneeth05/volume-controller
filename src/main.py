# Reference material
# https://pypi.org/project/pycaw/
# https://www.section.io/engineering-education/creating-a-hand-tracking-module/

import cv2
import mediapipe as mp
import numpy as np

from math import hypot
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

capture = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

minVol, maxVol = volume.GetVolumeRange()[:2]

while True:
    success, img = capture.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    lmList = []
    if results.multi_hand_landmarks:
        for handLandmark in results.multi_hand_landmarks:
            for id, lm in enumerate(handLandmark.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id, cx, cy])
            mpDraw.draw_landmarks(img, handLandmark, mpHands.HAND_CONNECTIONS,
                                  mpDraw.DrawingSpec(
                                      color=(0, 0, 255), thickness=2, circle_radius=4),
                                  mpDraw.DrawingSpec(
                                      color=(0, 0, 0), thickness=2, circle_radius=3),
                                  )

    if lmList != []:
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]

        cv2.circle(img, (x1, y1), 4, (0, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 4, (0, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)

        length = hypot(x2-x1, y2-y1)

        vol = np.interp(length, [15, 220], [minVol, maxVol])
        # print(vol, length)
        volume.SetMasterVolumeLevel(vol, None)

    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
