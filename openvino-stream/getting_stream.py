import cv2
import numpy as np
import imutils

cap = cv2.VideoCapture('http://localhost:8000/video_feed')

while(True):
    ret, frame = cap.read()
    frame = imutils.resize(frame,width=400,height=400)    
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()