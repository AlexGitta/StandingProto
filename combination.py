# TESTER PROGRAM TO COMBINE POSE TRACKING WITH A MUJOCO SIMULATION

import cv2
import mediapipe as mp
import time
import mujoco
import mujoco_viewer
import numpy as np

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize Mediapipe Hand Detection
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

model = mujoco.MjModel.from_xml_path("shadow_hand/scene_right.xml")
data = mujoco.MjData(model)

viewer = mujoco_viewer.MujocoViewer(model, data)


pTime = 0
cTime = 0

def is_peace(landmarks):
    # Function for recognizing peace sign
    return (landmarks[8].y < landmarks[6].y and
            landmarks[12].y < landmarks[10].y and
            landmarks[16].y > landmarks[14].y and
            landmarks[20].y > landmarks[18].y)

while True:

    if viewer.is_alive:
        mujoco.mj_step(model, data)
        viewer.render()
    else:
        break

    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                if id in [4, 8, 12, 16, 20]:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

            landmarks = handLms.landmark
            if is_peace(landmarks):
                cv2.putText(img, "Peace", (10, 100), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
                mujoco.mj_resetDataKeyframe(model, data, 0)
            else:
                mujoco.mj_resetDataKeyframe(model, data, 3)
            

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        viewer.close()
        break

cap.release()
cv2.destroyAllWindows()
