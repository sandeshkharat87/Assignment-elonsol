from ultralytics import YOLO
import cv2
import numpy as np
from centroidtracker import CentroidTracker



TRACKER = CentroidTracker(maxDisappeared=30, maxDistance=90)
model = YOLO("/home/wpnx/Downloads/bangle-best.pt")
video_path = "/home/wpnx/Downloads/bagleTest.mp4"



# VideoCapture
cap = cv2.VideoCapture(video_path)

while True:
    _, frame = cap.read()
    h, w, c = frame.shape

    # frame = cv2.resize(frame,(640,640))
    BBX = []
    rr = model.predict(frame)
    # rr = rr[0].boxes.xyxy.tolist()
    rr = rr[0].boxes.xyxy.tolist()
    if rr:
        for cords in rr:
            x1,y1,x2,y2 = np.array(cords).astype(np.int32)
            BBX.append([x1, y1, x2, y2])


    OBJECTS_IDS = TRACKER.update(BBX)

    for bid, bbx in OBJECTS_IDS.items():
        x1, y1, x2, y2 = [int(i) for i in bbx]
        # print("Frist loop (--2--)", x1, y1, x2, y2)
        bid = bid
        # cv2.rectangle(frame, (x1, y1), (x1 + x2, y1 + y2), (0, 123, 255), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 123, 255), 2)
        cx = int((x1 + x2) // 2)
        cy = int((y1 + y2) // 2)

        cv2.putText(
            img=frame,
            text=f"{bid} ",
            org=(x1, y1 - 5),
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=0.7,
            color=(1, 1, 255),
            thickness=2
        )



    cv2.imshow("Frame", frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



