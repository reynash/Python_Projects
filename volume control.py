import cv2
import numpy as np
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from mediapipe.python import solutions as mp_solutions
wcam, hcam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wcam)
cap.set(4, hcam)
detector = mp_solutions.hands.Hands()
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
audioEndpoint = cast(interface, POINTER(IAudioEndpointVolume))
minVol = audioEndpoint.GetVolumeRange()[0]
maxVol = audioEndpoint.GetVolumeRange()[1]
while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detector.process(img_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            index_finger = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.INDEX_FINGER_TIP]
            thumb = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.THUMB_TIP]
            length = math.hypot(index_finger.x - thumb.x, index_finger.y - thumb.y)
            vol = np.interp(length, [0, 0.2], [minVol, maxVol])
            volBar = np.interp(length, [0, 0.2], [400, 150])
            volPer = np.interp(length, [0, 0.2], [0, 100])
            audioEndpoint.SetMasterVolumeLevel(vol, None)
            cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
            cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, f'{int(volPer)}%', (40, 450), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            if length < 0.1:
                cv2.circle(img, (int(thumb.x * wcam), int(thumb.y * hcam)), 15, (0, 255, 0), cv2.FILLED)
    cv2.imshow("Volume Control", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
