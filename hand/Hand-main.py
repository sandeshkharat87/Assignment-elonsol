import cv2
import mediapipe as mp
import numpy as np

#mediapipe for Hand pose
mphands = mp.solutions.hands
hands = mphands.Hands(max_num_hands=1, min_tracking_confidence=0.70)
mp_drawing = mp.solutions.drawing_utils

# VideoCapture
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    h, w, c = frame.shape
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)
    hand_landmarks = result.multi_hand_landmarks
    # Draw line
    # This is the RED line . If Worker Hands crosses this line then Alert is given
    cv2.line(frame, (0, 139), (632, 139), (0, 0, 255), 3)

    #
    if hand_landmarks:
        for handLMs in hand_landmarks:
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h
            for lm in handLMs.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            pt1 = int((x_min+x_max)//2)
            pt2 = int((y_min+y_min)//2)
            cv2.circle(frame, (pt1,pt2), 10, (234, 23, 123), -1)
            cv2.putText(
                img=frame,
                text=f"x: {pt1},y: {pt2}",
                org=(pt1, pt2- 5),
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=0.7,
                color=(1, 12, 233),
                thickness=2)

            # Check : I hand crossing RED line, if crosses then Alert Stop the machine
            if (pt2 < 139):
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,0,255), 2)
                cv2.putText(
                img=frame,
                text=f"Stop Machine",
                org=(18, 19),
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=0.7,
                color=(0, 0,255),
                thickness=2)


    cv2.imshow("Frame", frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
